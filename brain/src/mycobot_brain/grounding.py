"""Ask vLLM where things are in a camera image, and whether a step worked.

The model only answers questions about pictures: where the object to pick and
the place to put it are (a box or a point per object), and yes/no checks
("is it held?", "is it in the box?"). Turning those pixels into metres is
geometry, done in :mod:`mycobot_brain.kinematics`.

Two coordinate conventions are on offer because Qwen generations differ:
Qwen2.5-VL was trained on absolute pixels of the image it was given, Qwen3-VL
on a 0..1000 grid. Which one ``qwen3.6`` answers better is measured
(G1, ``tools/g1_grounding.py``), not assumed.

Property names follow Qwen's own grounding output (``bbox_2d``,
``point_2d``, ``label``) so the schema'd reply is close to what it was trained
to write.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mycobot_brain.planner import DEFAULT_MODEL, PlannerError
from mycobot_brain.vision import image_part

COORDS = ("norm1000", "abs")
SHAPES = ("bbox", "point")
ROLES = {
    "pick": "the object the instruction says to pick up",
    "place": "where the instruction says to put it",
}
NEXT = ("pick", "done", "cannot")


def _coords_text(coords: str, width: int, height: int) -> str:
    if coords == "norm1000":
        return ("as integers on a 0-1000 grid: x is 0 at the left edge and 1000 at the right edge, "
                "y is 0 at the top edge and 1000 at the bottom edge")
    if coords == "abs":
        return f"in pixels of this {width}x{height} image: x from 0 (left) to {width} (right), y from 0 (top) to {height} (bottom)"
    raise ValueError(f"coords is one of {COORDS}, got {coords!r}")


def _object_schema(shape: str) -> Dict[str, Any]:
    n = 4 if shape == "bbox" else 2
    key = "bbox_2d" if shape == "bbox" else "point_2d"
    # Property order is generation order: name it, say whether it is there, then where.
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "visible": {"type": "boolean"},
            key: {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n},
        },
        "required": ["label", "visible", key],
        "additionalProperties": False,
    }


def locate_schema(roles: Sequence[str], shape: str, with_next: bool) -> Dict[str, Any]:
    props: Dict[str, Any] = {"seen": {"type": "string"}}
    for r in roles:
        props[r] = _object_schema(shape)
    if with_next:
        props["next"] = {"type": "string", "enum": list(NEXT)}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def locate_prompt(roles: Sequence[str], coords: str, shape: str, width: int, height: int, with_next: bool,
                  describe: Optional[Dict[str, str]] = None) -> str:
    where = ("bbox_2d [x1, y1, x2, y2], the tight box around it" if shape == "bbox"
             else "point_2d [x, y], the centre of it as seen in the image")
    lines = [
        "You are the eyes of a small robot arm. The image comes from a camera on the arm's gripper, "
        "looking down at a table.",
        "For each object asked for, give label (a few words naming it), visible (false if it is not in the image, "
        f"or only partly), and {where}, {_coords_text(coords, width, height)}.",
        "If an object is not visible, set visible false and give zeros; never guess where something unseen is.",
        "Objects asked for: " + "; ".join(f"{r} = {(describe or ROLES).get(r, ROLES.get(r, r))}" for r in roles) + ".",
        "seen is one short sentence on what is on the table.",
    ]
    if with_next:
        lines.append("next: 'pick' if both are visible and the job still has to be done, 'done' if the object "
                     "is already where it should be, 'cannot' if either is missing.")
    return "\n".join(lines)


def build_locate_request(
    jpeg: bytes,
    instruction: str,
    *,
    roles: Sequence[str] = ("pick", "place"),
    coords: str = "norm1000",
    shape: str = "bbox",
    width: int = 640,
    height: int = 480,
    with_next: bool = True,
    model: str = DEFAULT_MODEL,
    describe: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """``describe`` replaces the generic role texts (``ROLES``), e.g. ``{"pick": "the yellow cone"}``."""
    text = f"Instruction: {instruction}"
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": locate_prompt(roles, coords, shape, width, height, with_next, describe)},
            {"role": "user", "content": [image_part(jpeg), {"type": "text", "text": text}]},
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "locate", "schema": locate_schema(roles, shape, with_next)}},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 300,
    }


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def to_pixels(values: Sequence[float], coords: str, width: int, height: int) -> List[float]:
    """Model coordinates (x, y, x, y, ...) to pixels of the image sent."""
    if coords == "abs":
        return [float(v) for v in values]
    return [float(v) * (width if i % 2 == 0 else height) / 1000.0 for i, v in enumerate(values)]


def parse_locate(
    content: str,
    roles: Sequence[str],
    *,
    coords: str,
    shape: str,
    width: int = 640,
    height: int = 480,
    with_next: bool = True,
) -> Dict[str, Any]:
    """The reply, re-checked, with each visible object's ``pixel`` (u, v) centre added."""
    try:
        reply = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"not JSON: {content[:200]!r}") from exc
    if not isinstance(reply, dict):
        raise PlannerError(f"not an object: {content[:200]!r}")
    out: Dict[str, Any] = {"seen": str(reply.get("seen", ""))}
    key = "bbox_2d" if shape == "bbox" else "point_2d"
    for r in roles:
        obj = reply.get(r)
        if not isinstance(obj, dict) or not isinstance(obj.get("visible"), bool):
            raise PlannerError(f"need {r} with a boolean visible, got {str(obj)[:200]}")
        item: Dict[str, Any] = {"label": str(obj.get("label", "")), "visible": obj["visible"]}
        vals = obj.get(key)
        if obj["visible"]:
            n = 4 if shape == "bbox" else 2
            if not (isinstance(vals, list) and len(vals) == n and all(_number(v) for v in vals)):
                raise PlannerError(f"{r}.{key} needs {n} numbers, got {vals!r}")
            px = to_pixels(vals, coords, width, height)
            if shape == "bbox":
                item["bbox_px"] = [round(v, 1) for v in px]
                u, v = (px[0] + px[2]) / 2, (px[1] + px[3]) / 2
            else:
                u, v = px
            if not (0 <= u <= width and 0 <= v <= height):
                raise PlannerError(f"{r} centre ({u:.0f}, {v:.0f}) is outside the {width}x{height} image")
            item["pixel"] = [round(u, 1), round(v, 1)]
        out[r] = item
    if with_next:
        nxt = reply.get("next")
        if nxt not in NEXT:
            raise PlannerError(f"next must be one of {NEXT}, got {nxt!r}")
        out["next"] = nxt
    return out


CHECK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"seen": {"type": "string"}, "ok": {"type": "boolean"}},
    "required": ["seen", "ok"],
    "additionalProperties": False,
}


def build_check_request(
    question: str,
    images: Sequence[Tuple[str, bytes]],
    *,
    model: str = DEFAULT_MODEL,
    name: str = "check",
) -> Dict[str, Any]:
    """A yes/no question about one or more captioned images: ``{"seen", "ok"}``.
    ``name`` is the schema name, which says in the request which check it is."""
    content: List[Dict[str, Any]] = []
    for caption, jpeg in images:
        content += [{"type": "text", "text": caption}, image_part(jpeg)]
    content.append({"type": "text", "text": question})
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You check the work of a small robot arm from its gripper camera, which looks down past the "
                "gripper fingers. Answer only from what the images show. seen: one sentence on what the "
                "latest image shows that bears on the question. ok: true only if the images clearly show yes.")},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": CHECK_SCHEMA}},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 200,
    }


def parse_check(content: str) -> Dict[str, Any]:
    try:
        reply = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"not JSON: {content[:200]!r}") from exc
    if not isinstance(reply, dict) or not isinstance(reply.get("ok"), bool):
        raise PlannerError(f"need a boolean ok, got {content[:200]!r}")
    return {"seen": str(reply.get("seen", "")), "ok": reply["ok"]}


def prompt_text(request: Dict[str, Any]) -> str:
    """The request's text without the images, for the run log."""
    parts = []
    for m in request["messages"]:
        c = m["content"]
        if isinstance(c, str):
            parts.append(f"[{m['role']}] {c}")
        else:
            parts += [f"[{m['role']}] {p['text']}" for p in c if p.get("type") == "text"]
    return "\n".join(parts)


def held_question(pick_label: str) -> str:
    return (f"The gripper was open above the {pick_label}, went down, closed and lifted. "
            f"Is the {pick_label} now held between the gripper fingers, lifted off the table?")


def held_question_same_view(pick_label: str, gripper_side: str) -> str:
    return (f"Both images are from the same camera position, above where the {pick_label} was lying. "
            f"Between them the gripper went down, closed on the {pick_label} and lifted. "
            f"The gripper fingers are at the {gripper_side} edge of the image. "
            f"Is the {pick_label} gone from the table now, so the gripper is holding it? "
            f"ok is false if the {pick_label} still lies on the table.")


def held_question_from_above(pick_label: str, gripper_side: str) -> str:
    """The second image is from where the grasp lifted, the tool upright: not the first image's camera
    pose, which would roll the wrist with the object in hand (sim-scene 14:03)."""
    return (f"The first image was taken before the grasp, looking down at where the {pick_label} was lying. "
            f"Between them the gripper went down, closed on the {pick_label} and lifted a little; the second "
            f"image is from the camera now, beside the gripper, looking down past its fingers, which are at "
            f"the {gripper_side} edge of the image. Is the {pick_label} gone from where it lay, so the gripper "
            f"is holding it? ok is false if the {pick_label} still lies where it was.")


def final_question(pick_label: str, place_label: str) -> str:
    return f"Is the {pick_label} now inside the {place_label}?"


def object_label(found: Optional[Dict[str, Any]], fallback: str) -> str:
    label = (found or {}).get("label", "").strip()
    return label or fallback
