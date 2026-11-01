"""What the drawer tasks ask vLLM, and how its answers are checked.

The model reads the command (put / take / find, which object, which drawer),
says from the cabinet view which drawers are open and which to search next,
lists what is in an opened drawer (colour, shape, bbox) and which of those is
the one wanted, and answers yes/no checks. Code knows the cabinet: it marks
in the prompt where each drawer's handle or floor falls in the image (0-1000
grid, from the calib and the measured angles), so the model judges a local
patch and does not have to work out which drawer is which.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mycobot_brain import kinematics
from mycobot_brain.fixture import Drawer, Fixture
from mycobot_brain.grounding import _number, to_pixels
from mycobot_brain.planner import DEFAULT_MODEL, PlannerError
from mycobot_brain.vision import image_part

TASKS = ("put", "take", "find")
MAX_OBJECTS = 4
SHAPE_HINTS = {
    "cube": "square faces and straight edges",
    "cylinder": "flat round top and straight sides",
    "cone": "round base narrowing to a point",
    "sphere": "a ball, round from every side",
}


def _json(content: str) -> Dict[str, Any]:
    try:
        reply = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"not JSON: {content[:200]!r}") from exc
    if not isinstance(reply, dict):
        raise PlannerError(f"not an object: {content[:200]!r}")
    return reply


def _request(system: str, content: List[Dict[str, Any]], name: str, schema: Dict[str, Any], model: str,
             max_tokens: int) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def describe(colour: str, shape: str) -> str:
    """``("yellow", "cone")`` → "yellow cone"; "any" drops out ("any object" if both do)."""
    words = [w for w in (colour, shape) if w and w != "any"]
    if not words:
        return "any object"
    if shape in ("", "any"):
        words.append("object")
    return " ".join(words)


# -- the command ----------------------------------------------------------------


def parse_schema(fx: Fixture) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": list(TASKS)},
            "colour": {"type": "string", "enum": fx.colours + ["any"]},
            "shape": {"type": "string", "enum": sorted(fx.shapes) + ["any"]},
            "drawer": {"type": "string", "enum": sorted(fx.drawers) + ["none"]},
        },
        "required": ["task", "colour", "shape", "drawer"],
        "additionalProperties": False,
    }


def build_parse_request(instruction: str, fx: Fixture, *, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    system = "\n".join([
        "You turn a command to a small robot arm into fields. The arm stands in front of a cabinet of 3x3 drawers.",
        "task: 'put' = an object lying on the table goes into the named drawer; 'take' = an object comes out of "
        "the drawer the command names and is handed over; 'find' = search the drawers for the object and hand "
        "it over. A command that names no drawer is 'find', even when it also says to take the object out "
        "(찾아서 꺼내 줘 = find).",
        "colour and shape: of the object the command is about, 'any' for what it does not say.",
        "drawer: the drawer the command names, 'none' if it names none. The drawers, as the robot sees them:",
        fx.drawer_table(),
    ])
    return _request(system, [{"type": "text", "text": f"Command: {instruction}"}], "parse", parse_schema(fx), model,
                    100)


def parse_command(content: str, fx: Fixture) -> Dict[str, Any]:
    r = _json(content)
    task, colour, shape, drawer = (r.get(k) for k in ("task", "colour", "shape", "drawer"))
    if task not in TASKS or colour not in fx.colours + ["any"] or shape not in list(fx.shapes) + ["any"] \
            or drawer not in list(fx.drawers) + ["none"]:
        raise PlannerError(f"command fields out of range: {content[:200]!r}")
    if task == "take" and drawer == "none" and not (colour == "any" and shape == "any"):
        # Taking a named object out of no named drawer means searching for it first: in D3 the model
        # read "노란색 원뿔 찾아서 꺼내 줘" as take, drawer none, on every seed.
        task = "find"
    if task in ("put", "take") and drawer == "none":
        raise PlannerError(f"{task} needs a drawer: {content[:200]!r}")
    if task in ("put", "find") and colour == "any" and shape == "any":
        raise PlannerError(f"{task} needs an object: {content[:200]!r}")
    return {"task": task, "colour": colour, "shape": shape, "drawer": None if drawer == "none" else drawer}


# -- where things fall in the image ------------------------------------------


def grid(chain: kinematics.Chain, k: kinematics.Intrinsics, q: Sequence[float],
         p: Sequence[float]) -> Optional[Tuple[int, int]]:
    """A base-frame point on the 0-1000 grid of the image the camera at ``q`` takes; None if outside."""
    uv = kinematics.project(chain, k, q, p)
    if uv is None or not (0 <= uv[0] <= k.width and 0 <= uv[1] <= k.height):
        return None
    return round(uv[0] * 1000 / k.width), round(uv[1] * 1000 / k.height)


def floor_box(fx: Fixture, d: Drawer, opening: float, chain: kinematics.Chain, k: kinematics.Intrinsics,
              q: Sequence[float]) -> Optional[List[int]]:
    """The opened drawer floor's corners as a 0-1000 box in the image, clipped; None if none of it shows."""
    c = fx.interior_center(d, opening)
    hx, hy = d.interior_size[0] / 2, d.interior_size[1] / 2
    pts = [kinematics.project(chain, k, q, (c[0] + sx * hx, c[1] + sy * hy, c[2]))
           for sx in (-1, 1) for sy in (-1, 1)]
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    box = [max(0.0, min(us)), max(0.0, min(vs)), min(float(k.width), max(us)), min(float(k.height), max(vs))]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return [round(box[0] * 1000 / k.width), round(box[1] * 1000 / k.height),
            round(box[2] * 1000 / k.width), round(box[3] * 1000 / k.height)]


# -- the cabinet view ------------------------------------------------------------


Track = Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]


def handle_tracks(fx: Fixture, chain: kinematics.Chain, k: kinematics.Intrinsics,
                  q: Sequence[float]) -> Dict[str, Track]:
    """Each drawer's handle on the 0-1000 grid shut and fully out, from the camera at ``q``."""
    return {d.id: (grid(chain, k, q, fx.knob(d, 0.0)), grid(chain, k, q, fx.knob(d, fx.open_m)))
            for d in fx.drawers.values()}


def tray_boxes(fx: Fixture, chain: kinematics.Chain, k: kinematics.Intrinsics, q: Sequence[float],
               opening: float) -> Dict[str, Optional[List[float]]]:
    """Each drawer's tray (floor to front top, wall to wall) ``opening`` m out as a 0-1000 box; None if
    a corner falls behind the camera."""
    out: Dict[str, Optional[List[float]]] = {}
    for d in fx.drawers.values():
        c = fx.interior_center(d, opening)
        hx, hy = d.interior_size[0] / 2, d.interior_size[1] / 2
        top = d.rim_z if d.rim_z is not None else d.floor_z + d.interior_size[2]
        pts = [kinematics.project(chain, k, q, (c[0] + sx * hx, c[1] + sy * hy, z))
               for sx in (-1, 1) for sy in (-1, 1) for z in (d.floor_z, top)]
        if any(p is None for p in pts):
            out[d.id] = None
            continue
        us = [p[0] * 1000 / k.width for p in pts]  # type: ignore[index]
        vs = [p[1] * 1000 / k.height for p in pts]  # type: ignore[index]
        out[d.id] = [min(us), min(vs), max(us), max(vs)]
    return out


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def assign_trays(boxes: Sequence[Sequence[float]], open_boxes: Dict[str, Optional[List[float]]], *,
                 min_iou: float = 0.25) -> Dict[str, Any]:
    """Each pulled-out tray the model boxed → the drawer whose fully-open tray box it overlaps most."""
    opened, dropped = set(), []
    for b in boxes:
        scored = [(_iou(b, ob), did) for did, ob in open_boxes.items() if ob is not None]
        iou, did = max(scored, default=(0.0, None))
        if did is not None and iou >= min_iou:
            opened.add(did)
        else:
            dropped.append({"box": list(b), "near": did and {"drawer": did, "iou": round(iou, 2)}})
    return {"open_drawers": sorted(opened), "dropped": dropped}


def cabinet_schema(fx: Fixture, candidates: Optional[Sequence[str]], point: bool = False,
                   tray: bool = False) -> Dict[str, Any]:
    props: Dict[str, Any] = {"seen": {"type": "string"}}
    if tray:
        props["open_trays"] = {"type": "array", "maxItems": len(fx.drawers), "items": {
            "type": "object", "required": ["bbox_2d"], "additionalProperties": False,
            "properties": {"bbox_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}}}}
    elif point:
        props["open_handles"] = {"type": "array", "maxItems": len(fx.drawers), "items": {
            "type": "object", "required": ["point_2d"], "additionalProperties": False,
            "properties": {"point_2d": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}}}
    else:
        # No uniqueItems: vLLM's grammar backend answers 400 to it. parse_cabinet drops repeats.
        props["open_drawers"] = {"type": "array", "items": {"type": "string", "enum": sorted(fx.drawers)},
                                 "maxItems": len(fx.drawers)}
    if candidates is not None:
        props["next"] = {"type": "string", "enum": list(candidates) + ["none"]}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def build_cabinet_request(
    jpeg: bytes,
    fx: Fixture,
    handles: Optional[Dict[str, Optional[Tuple[int, int]]]],
    *,
    goal: str,
    searched: Optional[Dict[str, str]] = None,
    candidates: Optional[Sequence[str]] = None,
    handles_open: Optional[Dict[str, Optional[Tuple[int, int]]]] = None,
    point: bool = False,
    tray: bool = False,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Which drawers are open; with ``candidates``, also which of them to search next.

    ``handles`` maps each drawer to where its handle (shut) is on the 0-1000 grid; None leaves the hint out.
    ``handles_open``, the same with the drawer fully out: an opened drawer's handle is nowhere near its
    shut mark, and from above the opened top row lands low in the picture.
    ``point``: the model points at the handle of each pulled-out drawer instead of naming drawers, and
    ``parse_cabinet`` (given the handle tracks) says which drawers those are. In D2 the model named the
    opened top row "the bottom row" in sim-scene's rolled cabinet view, marks or not.
    ``tray``: the model boxes each pulled-out tray, matched to drawers by ``parse_cabinet`` (given the
    fully-open tray boxes).
    ``searched`` is what earlier looks found in the drawers already opened.
    """
    lines = [
        "You are the eyes of a small robot arm. The image is the arm's camera looking at the front of a cabinet "
        "of 3x3 small drawers. Rows r0 (bottom) to r2 (top); columns c0 (the robot's left) to c2 (right).",
    ]
    if handles:
        def at(g: Optional[Tuple[int, int]]) -> str:
            return f"({g[0]}, {g[1]})" if g else "outside the image"

        if handles_open:
            lines.append("Each drawer has one dark handle on its front. Where that handle is in the image "
                         "(x, y on a 0-1000 grid, 0,0 top left) with the drawer shut and with it pulled fully out:")
        else:
            lines.append("Where each drawer's handle is in the image when the drawer is shut "
                         "(x, y on a 0-1000 grid, 0,0 top left):")
        for did in sorted(fx.drawers, key=lambda i: (-fx.drawers[i].row, fx.drawers[i].col)):
            where = at(handles.get(did))
            if handles_open:
                where = f"shut {where}, open {at(handles_open.get(did))}"
            lines.append(f"  {did} ({fx.drawers[did].name_ko} 서랍): {where}")
        if handles_open:
            lines.append("Name a drawer open when its handle is at or towards its open point, with its tray "
                         "sticking out of the cabinet; a handle at its shut point is a shut drawer. The camera "
                         "may be rolled, so go by these points, not by up and down in the picture.")
    else:
        lines.append("Drawers: " + "; ".join(f"{d.id} = {d.name_ko} 서랍" for d in sorted(
            fx.drawers.values(), key=lambda d: (-d.row, d.col))) + ".")
    if tray:
        lines.append("open_trays: a box around the tray of every drawer that is pulled out of the cabinet "
                     "(the open box sticking out in front of the drawer fronts, with its dark handle), "
                     "bbox_2d = [x1, y1, x2, y2] on a 0-1000 grid with 0,0 the top left of the image; [] if "
                     "every drawer is shut.")
    elif point:
        lines.append("open_handles: one point on the dark handle of every drawer that is pulled out of the "
                     "cabinet (its tray sticks out in front of the other drawer fronts), x, y on a 0-1000 grid "
                     "with 0,0 the top left of the image; [] if every drawer is shut. Leave out the handles of "
                     "shut drawers, flush with the cabinet front.")
    else:
        lines.append("open_drawers: every drawer that is pulled out, even a little; [] if all are shut.")
    if candidates is not None:
        lines.append(f"Goal: {goal}.")
        if searched:
            lines.append("Already opened and looked into: " + "; ".join(f"{k}: {v}" for k, v in searched.items()) + ".")
        lines.append("next: the drawer to open and look into next, from: " + ", ".join(candidates)
                     + "; 'none' if no drawer is worth opening.")
    lines.append("seen: one short sentence on the cabinet.")
    content = [image_part(jpeg), {"type": "text", "text": f"Goal: {goal}"}]
    return _request("\n".join(lines), content, "cabinet", cabinet_schema(fx, candidates, point, tray), model, 300)


def assign_handles(points: Sequence[Sequence[float]], tracks: Dict[str, Track], *, min_out: float = 0.25,
                   max_off: float = 70.0, shut_r: float = 30.0, open_bias: float = 60.0) -> Dict[str, Any]:
    """Each pointed handle → the drawer whose shut-to-open handle track passes nearest it (0-1000 grid).

    A point within ``shut_r`` of any shut handle is a shut handle pointed at by mistake: from above, a
    column's tracks run over the shut handles below them. Otherwise the cost of a track is how far the
    point is off it plus ``open_bias`` times how far short of fully open it is (drawers are left shut or
    fully out, and the tracks of one column overlap in the picture); the cheapest within ``max_off`` and
    at least ``min_out`` along names that drawer open.
    """
    opened, dropped = set(), []
    shut = [a for a, _ in tracks.values() if a is not None]
    for p in points:
        if any(math.dist(p, a) <= shut_r for a in shut):
            dropped.append({"point": list(p), "near": "a shut handle"})
            continue
        best = None
        for did, (a, b) in tracks.items():
            if a is None or b is None:
                continue
            ab = (b[0] - a[0], b[1] - a[1])
            span = ab[0] ** 2 + ab[1] ** 2
            t = 0.0 if span == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / span))
            off = ((p[0] - a[0] - t * ab[0]) ** 2 + (p[1] - a[1] - t * ab[1]) ** 2) ** 0.5
            cost = off + open_bias * (1.0 - t)
            if best is None or cost < best[3]:
                best = (did, off, t, cost)
        if best is not None and best[1] <= max_off and best[2] >= min_out:
            opened.add(best[0])
        else:
            dropped.append({"point": list(p), "near": best and {"drawer": best[0], "off": round(best[1]),
                                                                 "along": round(best[2], 2)}})
    return {"open_drawers": sorted(opened), "dropped": dropped}


def parse_cabinet(content: str, fx: Fixture, candidates: Optional[Sequence[str]],
                  tracks: Optional[Dict[str, Track]] = None,
                  open_boxes: Optional[Dict[str, Optional[List[float]]]] = None) -> Dict[str, Any]:
    """With ``tracks`` the reply is pointed handles (``point=True``), with ``open_boxes`` boxed trays
    (``tray=True``), turned into drawer ids here."""
    r = _json(content)
    if open_boxes is not None:
        raw = r.get("open_trays")
        if not isinstance(raw, list):
            raise PlannerError(f"open_trays must be a list, got {str(raw)[:200]}")
        boxes = []
        for t in raw:
            b = t.get("bbox_2d") if isinstance(t, dict) else None
            if not (isinstance(b, list) and len(b) == 4 and all(_number(v) for v in b)):
                raise PlannerError(f"open_trays bbox_2d must be 4 numbers, got {str(t)[:200]}")
            x1, y1, x2, y2 = (float(v) for v in b)
            boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        out: Dict[str, Any] = {"seen": str(r.get("seen", "")), "boxes": boxes, **assign_trays(boxes, open_boxes)}
    elif tracks is not None:
        pts = r.get("open_handles")
        if not isinstance(pts, list):
            raise PlannerError(f"open_handles must be a list, got {str(pts)[:200]}")
        points = []
        for h in pts:
            xy = h.get("point_2d") if isinstance(h, dict) else None
            if not (isinstance(xy, list) and len(xy) == 2 and all(_number(v) for v in xy)):
                raise PlannerError(f"open_handles point_2d must be [x, y], got {str(h)[:200]}")
            x, y = float(xy[0]), float(xy[1])
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                raise PlannerError(f"open_handles point {xy} is outside the 0-1000 grid")
            points.append((x, y))
        out = {"seen": str(r.get("seen", "")), "points": [list(p) for p in points],
                               **assign_handles(points, tracks)}
    else:
        opened = r.get("open_drawers")
        if not isinstance(opened, list) or any(d not in fx.drawers for d in opened):
            raise PlannerError(f"open_drawers must list drawer ids, got {str(opened)[:200]}")
        out = {"seen": str(r.get("seen", "")), "open_drawers": sorted(set(opened))}
    if candidates is not None:
        nxt = r.get("next")
        if nxt not in list(candidates) + ["none"]:
            raise PlannerError(f"next must be one of {list(candidates)} or 'none', got {nxt!r}")
        out["next"] = None if nxt == "none" else nxt
    return out


# -- looking into a drawer ------------------------------------------------------


def drawer_schema(fx: Fixture, with_target: bool) -> Dict[str, Any]:
    obj = {
        "type": "object",
        "properties": {
            "colour": {"type": "string", "enum": fx.colours + ["other"]},
            "shape": {"type": "string", "enum": sorted(fx.shapes) + ["other"]},
            "bbox_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
        },
        "required": ["colour", "shape", "bbox_2d"],
        "additionalProperties": False,
    }
    props: Dict[str, Any] = {
        "seen": {"type": "string"},
        "open": {"type": "boolean"},
        "objects": {"type": "array", "items": obj, "maxItems": MAX_OBJECTS},
    }
    if with_target:
        props["target"] = {"type": "integer", "minimum": -1, "maximum": MAX_OBJECTS - 1}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def build_drawer_request(
    jpeg: bytes,
    fx: Fixture,
    d: Drawer,
    floor: Optional[Sequence[int]],
    *,
    target: Optional[str] = None,
    view: str = "tilted",
    shape_hints: bool = True,
    colour_hints: bool = False,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """What is in drawer ``d``, opened; with ``target`` (e.g. "yellow cone"), which one that is.

    ``floor`` is where the drawer floor should be in the image (0-1000 box), None to leave it out.
    ``shape_hints`` describes each shape in a few words (D2 measures whether that helps).
    ``colour_hints`` tells yellow from orange in words: in D2, 3 of the 4 tilted-view slips were
    orange called yellow, one of them an orange cone taken for the yellow one.
    """
    where = (f"If it is pulled out, its floor fills about x {floor[0]}-{floor[2]}, y {floor[1]}-{floor[3]} "
             "of the 0-1000 grid. " if floor else "")
    look = ("from above and from the robot's side, so you see the objects' sides" if view == "tilted"
            else "straight down from above")
    lines = [
        f"You are the eyes of a small robot arm. The camera looks {look} into drawer {d.id} "
        f"({d.name_ko} 서랍, {d.name_en}) of a small cabinet. {where}",
        "open: true if the drawer is pulled out so you see its floor.",
        "objects: every object inside this drawer, at most "
        f"{MAX_OBJECTS}; nothing outside it. For each: colour ({', '.join(fx.colours)}), "
        "shape (" + ("; ".join(f"{s} = {SHAPE_HINTS.get(s, s)}" for s in sorted(fx.shapes)) if shape_hints
                     else ", ".join(sorted(fx.shapes))) + "), and bbox_2d "
        "[x1, y1, x2, y2], the tight box around it on a 0-1000 grid (x 0 left to 1000 right, y 0 top to 1000 "
        "bottom)." + (" Look at the outline: a cone and a cylinder both look round from straight above."
                      if shape_hints else ""),
    ]
    if colour_hints:
        lines.append("Yellow is bright lemon yellow; orange is a deeper orange, between yellow and red. In the "
                     "shade of the drawer walls a yellow object stays lemon-coloured, an orange one looks darker. "
                     "Compare the objects with each other before naming yellow or orange.")
    if target is not None:
        lines.append(f"target: the index in objects of the {target}, -1 if there is none in this drawer. "
                     "Colour and shape must both match.")
    lines.append("seen: one short sentence on what is in the drawer.")
    text = f"Looking for: {target}" if target else "List what is in the drawer."
    return _request("\n".join(lines), [image_part(jpeg), {"type": "text", "text": text}], "drawer",
                    drawer_schema(fx, target is not None), model, 400)


def parse_drawer(content: str, fx: Fixture, *, with_target: bool, width: int = 640, height: int = 480) -> Dict[str, Any]:
    """The reply with each object's ``pixel`` centre and ``bbox_px`` added; ``target`` None if not there."""
    r = _json(content)
    if not isinstance(r.get("open"), bool) or not isinstance(r.get("objects"), list):
        raise PlannerError(f"need open and objects, got {content[:200]!r}")
    objects = []
    for o in r["objects"][:MAX_OBJECTS]:
        b = o.get("bbox_2d") if isinstance(o, dict) else None
        if not (isinstance(b, list) and len(b) == 4 and all(_number(v) for v in b)):
            raise PlannerError(f"object needs a bbox_2d of 4 numbers: {str(o)[:200]}")
        px = to_pixels(b, "norm1000", width, height)
        u, v = (px[0] + px[2]) / 2, (px[1] + px[3]) / 2
        if not (0 <= u <= width and 0 <= v <= height):
            raise PlannerError(f"object centre ({u:.0f}, {v:.0f}) is outside the image")
        objects.append({"colour": str(o.get("colour")), "shape": str(o.get("shape")),
                        "bbox_px": [round(x, 1) for x in px], "pixel": [round(u, 1), round(v, 1)]})
    out: Dict[str, Any] = {"seen": str(r.get("seen", "")), "open": r["open"], "objects": objects}
    if with_target:
        t = r.get("target")
        if not isinstance(t, int) or isinstance(t, bool) or not -1 <= t < MAX_OBJECTS:
            raise PlannerError(f"target must be an index or -1, got {t!r}")
        out["target"] = t if 0 <= t < len(objects) else None
    return out


# -- checks --------------------------------------------------------------------


def handover_question(obj: str) -> str:
    # D3 s452: from straight above the blue cylinder was a blue disc on the mat, and the model said
    # the cylinder was not there.
    return (f"The gripper just set the {obj} down on the table below the camera and opened. The camera looks "
            "straight down, so a cylinder, cone or sphere shows as a round disc of its colour and a cube as a "
            f"square. Is the {obj} on the table, out of the gripper? ok is false if the gripper still holds it "
            "or nothing of its colour is there.")


def put_question(obj: str, d: Drawer) -> str:
    return (f"Both images look into drawer {d.id} from the same place, before and after the gripper put the "
            f"{obj} in. Is the {obj} now inside the drawer, on its floor?")
