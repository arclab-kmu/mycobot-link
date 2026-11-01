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
        "the named drawer and is handed over; 'find' = search the drawers for the object and hand it over "
        "(no drawer named).",
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


def cabinet_schema(fx: Fixture, candidates: Optional[Sequence[str]]) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "seen": {"type": "string"},
        "open_drawers": {"type": "array", "items": {"type": "string", "enum": sorted(fx.drawers)},
                         "uniqueItems": True, "maxItems": len(fx.drawers)},
    }
    if candidates is not None:
        props["next"] = {"type": "string", "enum": list(candidates) + ["none"]}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def build_cabinet_request(
    jpeg: bytes,
    fx: Fixture,
    handles: Dict[str, Optional[Tuple[int, int]]],
    *,
    goal: str,
    searched: Optional[Dict[str, str]] = None,
    candidates: Optional[Sequence[str]] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Which drawers are open; with ``candidates``, also which of them to search next.

    ``handles`` maps each drawer to where its handle (shut) is on the 0-1000 grid.
    ``searched`` is what earlier looks found in the drawers already opened.
    """
    lines = [
        "You are the eyes of a small robot arm. The image is the arm's camera looking at the front of a cabinet "
        "of 3x3 small drawers. Rows r0 (bottom) to r2 (top); columns c0 (the robot's left) to c2 (right).",
        "Where each drawer's handle is in the image when the drawer is shut (x, y on a 0-1000 grid, 0,0 top left):",
    ]
    for did in sorted(fx.drawers, key=lambda i: (-fx.drawers[i].row, fx.drawers[i].col)):
        g = handles.get(did)
        where = f"({g[0]}, {g[1]})" if g else "outside the image"
        lines.append(f"  {did} ({fx.drawers[did].name_ko} 서랍): {where}")
    lines.append("open_drawers: every drawer that is pulled out, even a little; [] if all are shut.")
    if candidates is not None:
        lines.append(f"Goal: {goal}.")
        if searched:
            lines.append("Already opened and looked into: " + "; ".join(f"{k}: {v}" for k, v in searched.items()) + ".")
        lines.append("next: the drawer to open and look into next, from: " + ", ".join(candidates)
                     + "; 'none' if no drawer is worth opening.")
    lines.append("seen: one short sentence on the cabinet.")
    content = [image_part(jpeg), {"type": "text", "text": f"Goal: {goal}"}]
    return _request("\n".join(lines), content, "cabinet", cabinet_schema(fx, candidates), model, 200)


def parse_cabinet(content: str, fx: Fixture, candidates: Optional[Sequence[str]]) -> Dict[str, Any]:
    r = _json(content)
    opened = r.get("open_drawers")
    if not isinstance(opened, list) or any(d not in fx.drawers for d in opened):
        raise PlannerError(f"open_drawers must list drawer ids, got {str(opened)[:200]}")
    out: Dict[str, Any] = {"seen": str(r.get("seen", "")), "open_drawers": sorted(set(opened))}
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
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """What is in drawer ``d``, opened; with ``target`` (e.g. "yellow cone"), which one that is.

    ``floor`` is where the drawer floor should be in the image (0-1000 box).
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
        "shape (" + "; ".join(f"{s} = {SHAPE_HINTS.get(s, s)}" for s in sorted(fx.shapes)) + "), and bbox_2d "
        "[x1, y1, x2, y2], the tight box around it on a 0-1000 grid (x 0 left to 1000 right, y 0 top to 1000 "
        "bottom). Look at the outline: a cone and a cylinder both look round from straight above.",
    ]
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
    return (f"The gripper just set the {obj} down on the table below the camera and opened. "
            f"Is the {obj} standing on the table, out of the gripper? ok is false if the gripper still holds it "
            "or it is not there.")


def put_question(obj: str, d: Drawer) -> str:
    return (f"Both images look into drawer {d.id} from the same place, before and after the gripper put the "
            f"{obj} in. Is the {obj} now inside the drawer, on its floor?")
