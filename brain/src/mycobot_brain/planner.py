"""Ask the Spark's vLLM for one joint move.

``response_format`` with a JSON schema makes vLLM's decoder emit the
``move_joints`` shape, so the reply always parses. What it cannot do is make
the numbers safe: they are a proposal, and the arm server checks them again
against the measured angles.

2026-10-23 on this Spark (``qwen3.6``, thinking off, one 43-token move): about
555 ms per call once warm, 1027 ms cold.
"""

from __future__ import annotations

import json
import math
import urllib.request
from typing import Any, Callable, Dict, Optional

#: vLLM is published on 127.0.0.1 only (its docker port mapping), which is why
#: the brain runs on the Spark rather than calling it across the network.
DEFAULT_VLLM_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "qwen3.6"

#: Joint limits as the model is told them. The arm server enforces its own
#: copy (``arm-server/src/safety.rs``); these only steer the proposal.
JOINT_LIMITS_DEG = ((-168, 168), (-140, 140), (-150, 150), (-150, 150), (-155, 160), (-180, 180))
MAX_STEP_DEG = 20
MAX_SPEED = 50

MOVE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "angles_deg": {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6},
        "speed": {"type": "integer", "minimum": 1, "maximum": MAX_SPEED},
        "reason": {"type": "string"},
    },
    "required": ["angles_deg", "speed", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You control a myCobot 280 six-joint arm by proposing ONE joint-space target. "
    "Angles are degrees, J1 (base) to J6 (wrist roll). "
    "Limits: " + ", ".join(f"J{i + 1} {lo}..{hi}" for i, (lo, hi) in enumerate(JOINT_LIMITS_DEG)) + ". "
    f"No joint may move more than {MAX_STEP_DEG} degrees from its current angle; "
    "a larger move is refused, so split it and move part of the way. "
    f"speed is 1..{MAX_SPEED}; use 20 unless told otherwise. "
    "Reply with angles_deg, speed and a one-sentence reason."
)

Opener = Callable[..., Any]


class PlannerError(RuntimeError):
    """vLLM did not answer, or answered something that is not one move."""


def build_request(instruction: str, angles_deg: list, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    current = ", ".join(f"J{i + 1}={a:+.1f}" for i, a in enumerate(angles_deg))
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current angles: {current}.\nInstruction: {instruction}"},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "move_joints", "schema": MOVE_SCHEMA}},
        # Qwen3 thinks before answering by default; the 555 ms above is with it off.
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 200,
    }


def parse_move(content: str) -> Dict[str, Any]:
    """The schema'd JSON from the model, re-checked here so a server that
    ignored ``response_format`` cannot slip a malformed move through."""
    try:
        move = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"not JSON: {content[:200]!r}") from exc
    angles = move.get("angles_deg") if isinstance(move, dict) else None
    if not (isinstance(angles, list) and len(angles) == 6):
        raise PlannerError(f"need 6 angles_deg, got {angles!r}")
    if not all(isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a) for a in angles):
        raise PlannerError(f"angles_deg must be finite numbers, got {angles!r}")
    speed = move.get("speed")
    if not (isinstance(speed, int) and not isinstance(speed, bool) and 1 <= speed <= MAX_SPEED):
        raise PlannerError(f"speed must be an integer in 1..{MAX_SPEED}, got {speed!r}")
    return {"angles_deg": [float(a) for a in angles], "speed": speed, "reason": str(move.get("reason", ""))}


def chat(
    request: Dict[str, Any],
    *,
    base_url: str = DEFAULT_VLLM_URL,
    timeout: float = 60.0,
    opener: Optional[Opener] = None,
) -> str:
    """POST one chat completion; return the first choice's text."""
    body = json.dumps(request).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with (opener or urllib.request.urlopen)(req, timeout=timeout) as resp:
            reply = json.load(resp)
    except OSError as exc:
        raise PlannerError(f"vLLM at {base_url}: {exc}") from exc
    try:
        return reply["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlannerError(f"unexpected completion: {str(reply)[:200]}") from exc


def propose_move(
    instruction: str,
    angles_deg: list,
    *,
    base_url: str = DEFAULT_VLLM_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = 60.0,
    opener: Optional[Opener] = None,
) -> Dict[str, Any]:
    """One chat completion → ``{"angles_deg", "speed", "reason"}``."""
    request = build_request(instruction, angles_deg, model)
    return parse_move(chat(request, base_url=base_url, timeout=timeout, opener=opener))
