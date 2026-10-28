"""Look, step, look again: the closed loop between the camera, vLLM and the arm.

Each round takes a frame that arrived after the arm settled, asks vLLM for one
small joint delta, and sends measured angles + delta as a target. Limits here
only stop a wandering model early; the arm server still checks every target
against its own limits and the 20 deg jump rule.

The loop ends when the model says done, after ``steps`` rounds, or at the
first refusal from either side. The gate is closed on the way out regardless.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mycobot_brain.camera import CameraLink
from mycobot_brain.link import ArmLink, LinkError

Propose = Callable[..., Dict[str, Any]]
Emit = Callable[[Dict[str, Any]], None]


class ActStopped(RuntimeError):
    """The loop stopped before done: a proposal broke a limit here."""


def run(
    link: ArmLink,
    camera: CameraLink,
    instruction: str,
    *,
    propose: Propose,
    emit: Emit,
    steps: int = 5,
    max_step: float = 5.0,
    max_total: float = 30.0,
    locked: Sequence[int] = (),
    speed: int = 10,
    settle: float = 1.5,
    frame_timeout: float = 3.0,
    execute: bool = False,
    record: Optional[Path] = None,
    camera_kind: str = "wrist",
) -> Dict[str, Any]:
    """Run the loop; return the last proposal. Without ``execute`` it proposes once and returns."""
    if record is not None:
        record.mkdir(parents=True, exist_ok=True)
    state = link.wait_state()
    if state["angles_deg"] is None:
        raise LinkError("the arm server has no joint reading to start from")
    start = list(state["angles_deg"])
    history: List[Dict[str, Any]] = []
    previous: Optional[bytes] = None
    since = time.monotonic()  # only frames taken after the loop began
    enabled = False
    proposal: Dict[str, Any] = {}
    try:
        for k in range(1, steps + 1):
            frame = camera.wait_frame(after=since, timeout=frame_timeout)
            state = link.wait_state(after_seq=state["seq"])
            angles = state["angles_deg"]
            if angles is None:
                raise LinkError("lost the joint reading")
            t0 = time.monotonic()
            proposal = propose(
                instruction, frame.jpeg, angles, previous_jpeg=previous, history=history, step=k, steps=steps,
                max_step=max_step, locked=locked, camera=camera_kind,
            )
            report = {"step": k, "frame_seq": frame.seq, "frame_age_s": round(frame.age(), 3),
                      "angles_deg": angles, "vllm_s": round(time.monotonic() - t0, 3), **proposal}
            if record is not None:
                (record / f"step-{k:02d}.jpg").write_bytes(frame.jpeg)
                with open(record / "steps.jsonl", "a") as f:
                    f.write(json.dumps(dict(report, instruction=instruction), ensure_ascii=False) + "\n")
            emit(report)
            if proposal["done"]:
                return proposal
            target = [a + d for a, d in zip(angles, proposal["delta_deg"])]
            drift = max(abs(t - s) for t, s in zip(target, start))
            if drift > max_total:
                raise ActStopped(f"step {k} would take a joint {drift:.1f} deg from the start (limit {max_total:g})")
            if not execute:
                return proposal
            if not enabled:
                link.enable()
                enabled = True
            link.move_joints(target, speed)
            emit({"step": k, "sent": {"angles_deg": [round(t, 2) for t in target], "speed": speed}})
            time.sleep(settle)
            since = time.monotonic()
            history.append({"step": k, "delta_deg": proposal["delta_deg"], "reason": proposal["reason"]})
            previous = frame.jpeg
        emit({"done": False, "why": f"no done after {steps} steps"})
        return proposal
    finally:
        if enabled:
            link.disable()
