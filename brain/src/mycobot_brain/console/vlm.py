"""Instruction + camera frames -> one action chunk (or the prepared task meant), from the console VLM.

The old ``act`` loop asked for one bounded joint delta per call: about 2 s of
model time for 5 degrees of motion, then a stop, then a new photo. That is the
choppiness Hojin saw. Here one call returns a *chunk*: up to ``MAX_STEPS``
steps that the console executes as one continuous motion (the QP controller
blends them), while the next call can already be in flight.

The model never produces joint angles for a reach. It points: ``point`` steps
carry a pixel in the top camera (Gemma's native 0-1000 normalised
coordinates), and the console turns that pixel into a 3-D target with the
camera calibration and the aligned depth. ``move`` steps are small Cartesian
nudges in the world frame for "a bit higher / left" corrections. Everything is
re-checked by the console's workspace box and then by the arm server.
"""

from __future__ import annotations

import base64
import json
import math
from typing import Any

from mycobot_brain.planner import PlannerError, chat

#: Gemma 4 26B-A4B served by vLLM on the AGX Orin (``lab/agx-vlm/serve.sh``, 127.0.0.1:8100
#: there; the Spark's ``gemma4-26b-a4b`` container until 2026-11-26), reached from the workstation
#: through ``ssh -L 8100:127.0.0.1:8100 agx@192.168.0.72`` (``tools/console_up.sh``).
DEFAULT_VLM_URL = "http://127.0.0.1:8100/v1"
DEFAULT_VLM_MODEL = "gemma-4-26b-a4b-it"

MAX_STEPS = 6
OPS = ("point", "move", "home", "grip", "wait")
#: Per-step bounds the schema states; the console clamps again.
MAX_NUDGE_MM = 80.0
MAX_HEIGHT_MM = 200.0
MAX_WAIT_S = 3.0

STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": list(OPS)},
        "target": {"type": "string"},
        "x": {"type": "number", "minimum": 0, "maximum": 1000},
        "y": {"type": "number", "minimum": 0, "maximum": 1000},
        "height_mm": {"type": "number", "minimum": 0, "maximum": MAX_HEIGHT_MM},
        "dx_mm": {"type": "number", "minimum": -MAX_NUDGE_MM, "maximum": MAX_NUDGE_MM},
        "dy_mm": {"type": "number", "minimum": -MAX_NUDGE_MM, "maximum": MAX_NUDGE_MM},
        "dz_mm": {"type": "number", "minimum": -MAX_NUDGE_MM, "maximum": MAX_NUDGE_MM},
        "grip": {"type": "integer", "minimum": 0, "maximum": 100},
        "seconds": {"type": "number", "minimum": 0, "maximum": MAX_WAIT_S},
    },
    "required": ["op", "target", "x", "y", "height_mm", "dx_mm", "dy_mm", "dz_mm", "grip", "seconds"],
    "additionalProperties": False,
}

CHUNK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
        "say": {"type": "string"},
        "done": {"type": "boolean"},
        "steps": {"type": "array", "items": STEP_SCHEMA, "minItems": 0, "maxItems": MAX_STEPS},
    },
    "required": ["seen", "say", "done", "steps"],
    "additionalProperties": False,
}


def system_prompt(world_axes: str) -> str:
    return (
        "You drive a small 6-joint robot arm (myCobot 280) with a two-finger gripper over a table. "
        "You get the TOP camera image (looking straight down at the table) and usually a SIDE camera image. "
        "Reply with ONE JSON action chunk: a short list of steps the arm executes as one smooth motion.\n"
        "Step ops (fill every field; unused numbers are 0, unused strings empty):\n"
        "- point: move the gripper tip above the thing at pixel (x, y) of the TOP image, in 0-1000 "
        "normalised coordinates (x right, y down), height_mm above that thing's top surface "
        "(30-80 to hover, 0-10 to touch). Name it in target.\n"
        f"- move: nudge the gripper tip by dx_mm, dy_mm, dz_mm (each within +-{MAX_NUDGE_MM:.0f}) in the "
        f"world frame. {world_axes}\n"
        "- home: return to the safe home pose.\n"
        "- grip: set the gripper, grip 0 = closed, 100 = fully open.\n"
        f"- wait: pause seconds (at most {MAX_WAIT_S:.0f}).\n"
        "Rules: only point at things you can actually see in the TOP image; never aim below the table; "
        "prefer one chunk that finishes the instruction; if the instruction is already satisfied, "
        "return no steps and done=true. seen = one sentence on what is where; say = one short sentence "
        "in the user's language on what you will do."
    )


def image_part(jpeg: bytes) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def build_chunk_request(
    instruction: str,
    images: dict[str, bytes],
    state_text: str,
    world_axes: str,
    model: str = DEFAULT_VLM_MODEL,
    history: list[str] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    for name, jpeg in images.items():
        content.append({"type": "text", "text": f"{name.upper()} camera:"})
        content.append(image_part(jpeg))
    lines = [f"Arm state: {state_text}"]
    if history:
        lines.append("Earlier in this task: " + " | ".join(history[-4:]))
    lines.append(f"Instruction: {instruction}")
    content.append({"type": "text", "text": "\n".join(lines)})
    return {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt(world_axes)}, {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "action_chunk", "schema": CHUNK_SCHEMA}},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 600,
    }


def _num(step: dict[str, Any], key: str, lo: float, hi: float) -> float:
    v = step.get(key, 0)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        raise PlannerError(f"{key} must be a finite number, got {v!r}")
    return float(min(max(v, lo), hi))


def parse_chunk(content: str) -> dict[str, Any]:
    """Schema'd JSON from the model, re-checked and clamped here."""
    text = content.strip()
    if text.startswith("```"):  # a server that ignored response_format
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"not JSON: {content[:200]!r}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("steps"), list):
        raise PlannerError(f"need an object with steps, got {content[:200]!r}")
    steps = []
    for s in raw["steps"][:MAX_STEPS]:
        if not isinstance(s, dict) or s.get("op") not in OPS:
            raise PlannerError(f"bad step {s!r}")
        steps.append(
            {
                "op": s["op"],
                "target": str(s.get("target", ""))[:80],
                "x": _num(s, "x", 0, 1000),
                "y": _num(s, "y", 0, 1000),
                "height_mm": _num(s, "height_mm", 0, MAX_HEIGHT_MM),
                "dx_mm": _num(s, "dx_mm", -MAX_NUDGE_MM, MAX_NUDGE_MM),
                "dy_mm": _num(s, "dy_mm", -MAX_NUDGE_MM, MAX_NUDGE_MM),
                "dz_mm": _num(s, "dz_mm", -MAX_NUDGE_MM, MAX_NUDGE_MM),
                "grip": int(_num(s, "grip", 0, 100)),
                "seconds": _num(s, "seconds", 0, MAX_WAIT_S),
            }
        )
    return {
        "seen": str(raw.get("seen", ""))[:400],
        "say": str(raw.get("say", ""))[:400],
        "done": bool(raw.get("done", False)),
        "steps": steps,
    }


def propose_chunk(
    instruction: str,
    images: dict[str, bytes],
    state_text: str,
    world_axes: str,
    *,
    base_url: str = DEFAULT_VLM_URL,
    model: str = DEFAULT_VLM_MODEL,
    history: list[str] | None = None,
    timeout: float = 60.0,
    opener: Any = None,
) -> dict[str, Any]:
    req = build_chunk_request(instruction, images, state_text, world_axes, model, history)
    return parse_chunk(chat(req, base_url=base_url, timeout=timeout, opener=opener))


# -- which prepared task does the user mean? ------------------------------------------------
#: Answer for "no listed task fits": the console then falls back to free-form action chunks.
TASK_NONE = "none"


def task_schema(names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task": {"type": "string", "enum": [*names, TASK_NONE]},
            "seen": {"type": "string"},
            "say": {"type": "string"},
        },
        "required": ["task", "seen", "say"],
        "additionalProperties": False,
    }


def build_task_request(instruction: str, images: dict[str, bytes], tasks: dict[str, str],
                       model: str = DEFAULT_VLM_MODEL) -> dict[str, Any]:
    """``tasks`` = {exact task name: one-line description}. The model picks one name or none."""
    system = (
        "You read a user's request to a table robot with two small arms (myCobot 280): one holds a two-finger "
        "gripper, the other a spoon. You also get its camera images. The robot has prepared tasks that measure "
        "the table and plan their own motion. If one listed task does what the user asks, answer its exact name "
        f"in task; if none fits, answer {TASK_NONE}. seen = one sentence on what is on the table and say = one short "
        "sentence on what the robot will do, both in the user's language."
    )
    content: list[dict[str, Any]] = []
    for name, jpeg in images.items():
        content.append({"type": "text", "text": f"{name.upper()} camera:"})
        content.append(image_part(jpeg))
    lines = ["Prepared tasks:"] + [f"- {k}: {v}" for k, v in tasks.items()] + [f"User: {instruction}"]
    content.append({"type": "text", "text": "\n".join(lines)})
    return {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "task_choice", "schema": task_schema(list(tasks))}},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 300,
    }


def parse_task(content: str, names: list[str]) -> dict[str, Any]:
    """{"task": a listed name or None, "seen", "say"}; anything else is an error, not a guess."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"not JSON: {content[:200]!r}") from exc
    if not isinstance(raw, dict):
        raise PlannerError(f"need an object, got {content[:200]!r}")
    task = raw.get("task")
    if task != TASK_NONE and task not in names:
        raise PlannerError(f"unknown task {task!r}")
    return {"task": None if task == TASK_NONE else task, "seen": str(raw.get("seen", ""))[:400],
            "say": str(raw.get("say", ""))[:400]}


def choose_task(instruction: str, images: dict[str, bytes], tasks: dict[str, str], *,
                base_url: str = DEFAULT_VLM_URL, model: str = DEFAULT_VLM_MODEL, timeout: float = 60.0,
                opener: Any = None) -> dict[str, Any]:
    req = build_task_request(instruction, images, tasks, model)
    return parse_task(chat(req, base_url=base_url, timeout=timeout, opener=opener), list(tasks))
