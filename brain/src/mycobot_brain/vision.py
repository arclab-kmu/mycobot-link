"""Ask vLLM about a camera frame: describe it, or pick the next small joint step.

``qwen3.6`` (``nvidia/Qwen3.6-35B-A3B-NVFP4``) is a vision-language model: its
config carries a ``vision_config``, so the same server reads images when it is
started without an image limit of 0. Images go in as base64 ``data:`` URLs,
which keeps the brain free of any file server.

A step is a joint *delta*, not a target: the model sees where the camera looks
now, not a map from pixels to angles, so it is asked for a small nudge and
judges the next picture. Which sign moves the image which way is left for it
to read off its previous step (not measured on this arm yet).
"""

from __future__ import annotations

import base64
import json
import math
from typing import Any, Dict, List, Optional, Sequence

from mycobot_brain.planner import DEFAULT_MODEL, DEFAULT_VLLM_URL, Opener, PlannerError, chat

DEFAULT_MAX_STEP_DEG = 5.0

JOINTS = (
    "J1 turns the whole arm about the vertical base axis",
    "J2 (shoulder) tilts the upper arm",
    "J3 (elbow) bends the forearm",
    "J4 pitches the wrist",
    "J5 swings the wrist sideways",
    "J6 rolls the end effector about its own axis",
)

CAMERAS = {
    "wrist": "mounted on the gripper, so every joint step moves the view",
    "fixed": "fixed in the room, looking at the arm and its workspace",
}

DESCRIBE_PROMPT = (
    "Describe this robot camera image in two or three sentences: the main objects, "
    "where each is in the frame (left/right/top/bottom/centre), and any part of the arm that shows."
)


def image_part(jpeg: bytes) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def build_describe_request(jpeg: bytes, question: Optional[str] = None, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": [image_part(jpeg), {"type": "text", "text": question or DESCRIBE_PROMPT}]}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 300,
    }


def describe(
    jpeg: bytes,
    question: Optional[str] = None,
    *,
    base_url: str = DEFAULT_VLLM_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
    opener: Optional[Opener] = None,
) -> str:
    return chat(build_describe_request(jpeg, question, model), base_url=base_url, timeout=timeout, opener=opener).strip()


def step_schema(max_step: float) -> Dict[str, Any]:
    # Property order is generation order: say what is seen before deciding.
    return {
        "type": "object",
        "properties": {
            "seen": {"type": "string"},
            "reason": {"type": "string"},
            "done": {"type": "boolean"},
            "delta_deg": {
                "type": "array",
                "items": {"type": "number", "minimum": -max_step, "maximum": max_step},
                "minItems": 6,
                "maxItems": 6,
            },
        },
        "required": ["seen", "reason", "done", "delta_deg"],
        "additionalProperties": False,
    }


def system_prompt(max_step: float, locked: Sequence[int], camera: str) -> str:
    still = ", ".join(f"J{j}" for j in sorted(locked))
    return (
        f"You steer a myCobot 280 six-joint arm in small steps by looking through a camera {CAMERAS[camera]}. "
        "Each turn you get the camera image now (after the first turn also the image from before your last step), "
        "the joint angles in degrees, the instruction, and your earlier steps. "
        "Joints: " + "; ".join(JOINTS) + ". "
        "Which sign of a joint moves the image which way is not given: work it out from how your previous step "
        "changed the image, and when unsure move a single joint by a few degrees. "
        "Reply with seen (one sentence on what in the image matters for the instruction), reason (why this step), "
        "done (true once the image shows the instruction is met; then all deltas are 0), and delta_deg "
        f"(six changes J1..J6 to apply now, each within +-{max_step:g} degrees)."
        + (f" {still} must stay 0." if still else "")
    )


def build_step_request(
    instruction: str,
    jpeg: bytes,
    angles_deg: Sequence[float],
    *,
    previous_jpeg: Optional[bytes] = None,
    history: Sequence[Dict[str, Any]] = (),
    step: int = 1,
    steps: int = 1,
    max_step: float = DEFAULT_MAX_STEP_DEG,
    locked: Sequence[int] = (),
    camera: str = "wrist",
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    current = ", ".join(f"J{i + 1}={a:+.1f}" for i, a in enumerate(angles_deg))
    earlier = "\n".join(
        f"  step {h['step']}: delta_deg {[round(d, 1) for d in h['delta_deg']]} because {h['reason']}" for h in history
    )
    text = (
        f"Instruction: {instruction}\nThis is step {step} of at most {steps}.\nJoint angles now: {current}.\n"
        + (f"Earlier steps:\n{earlier}\n" if earlier else "No earlier steps.\n")
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    if previous_jpeg is not None:
        content += [{"type": "text", "text": "Image before your last step:"}, image_part(previous_jpeg)]
    content += [{"type": "text", "text": "Image now:"}, image_part(jpeg)]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt(max_step, locked, camera)},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "step", "schema": step_schema(max_step)}},
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 300,
    }


def parse_step(content: str, max_step: float, locked: Sequence[int] = ()) -> Dict[str, Any]:
    """Re-check the schema'd reply here: a step out of bounds is refused, never clipped."""
    try:
        step = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"not JSON: {content[:200]!r}") from exc
    if not isinstance(step, dict) or not isinstance(step.get("done"), bool):
        raise PlannerError(f"need a boolean done, got {content[:200]!r}")
    delta = step.get("delta_deg")
    if not (isinstance(delta, list) and len(delta) == 6):
        raise PlannerError(f"need 6 delta_deg, got {delta!r}")
    if not all(isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d) for d in delta):
        raise PlannerError(f"delta_deg must be finite numbers, got {delta!r}")
    if any(abs(d) > max_step + 1e-9 for d in delta):
        raise PlannerError(f"delta_deg {delta} exceeds +-{max_step:g} deg")
    moved = [j for j in locked if delta[j - 1] != 0]
    if moved and not step["done"]:
        raise PlannerError(f"locked joints moved: {', '.join(f'J{j}' for j in moved)}")
    return {
        "seen": str(step.get("seen", "")),
        "reason": str(step.get("reason", "")),
        "done": step["done"],
        "delta_deg": [0.0] * 6 if step["done"] else [float(d) for d in delta],
    }


def propose_step(
    instruction: str,
    jpeg: bytes,
    angles_deg: Sequence[float],
    *,
    base_url: str = DEFAULT_VLLM_URL,
    timeout: float = 60.0,
    opener: Optional[Opener] = None,
    **request: Any,
) -> Dict[str, Any]:
    """One look → ``{"seen", "reason", "done", "delta_deg"}``. ``request`` goes to :func:`build_step_request`."""
    body = build_step_request(instruction, jpeg, angles_deg, **request)
    max_step = request.get("max_step", DEFAULT_MAX_STEP_DEG)
    return parse_step(chat(body, base_url=base_url, timeout=timeout, opener=opener), max_step, request.get("locked", ()))
