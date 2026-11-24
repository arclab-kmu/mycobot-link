"""Pre-planned, pre-checked scenarios started from the console's command box.

A scenario is a JSON file made offline (measured objects, cuRobo-checked joint paths). Typing its
``trigger`` text in the command box runs it instead of asking the VLM; the page says so. Each phase
drives one arm through one :meth:`MotionRunner.run` call, so the arm never stops between segments
unless a segment says so (gripper, wait, centring).

Segment kinds in the file::

    {"kind": "joints", "label": "...", "q": [[6 deg], ...], "dt": 0.1, "offset": "none|in|full|out", "ramp": 10}
    {"kind": "grip", "label": "...", "value": 10}
    {"kind": "wait", "label": "...", "seconds": 1.5}
    {"kind": "center", "label": "...", "camera": "top", "template": "bowl.png", "search": [u0, v0, u1, v1],
     "axis_px": [u, v], "px2mm": [[2x2]], "J": [[6x2] deg per mm], "max_err_mm": 25, "deadzone_mm": 2}

``center`` finds a template (e.g. the spoon bowl) in the camera's newest frame, turns its pixel
offset from ``axis_px`` into millimetres, and shifts the following joint segments by
``dq = -J @ err``: "in" ramps the shift in over ``ramp`` samples, "full" adds it, "out" fades it out.
Phase ``backlash`` (deg per joint) and ``v0`` (deg/s) add the deadband feed-forward
``q + b * tanh(qdot / v0)`` to every joint segment of that phase.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mycobot_brain.console.motion import Segment


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    sc = json.loads(path.read_text())
    sc["_dir"] = str(path.parent)
    if not sc.get("trigger") or not sc.get("phases"):
        raise ValueError(f"{path}: needs 'trigger' and 'phases'")
    return sc


def labels(sc: dict[str, Any]) -> list[str]:
    """One line per segment, for the page's step list."""
    out = []
    for ph in sc["phases"]:
        for seg in ph["segments"]:
            out.append(f"{ph['arm']} · {seg.get('label') or seg['kind']}")
    return out


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def _feed_forward(Q: np.ndarray, dt: float, b: np.ndarray | None, v0: float) -> np.ndarray:
    if b is None or len(Q) < 3:
        return Q
    v = np.gradient(Q, dt, axis=0)
    return Q + b * np.tanh(v / v0)


def match_template(gray: np.ndarray, tpl: np.ndarray, box: list[int]) -> tuple[float, float, float]:
    """Best normalised-correlation centre of ``tpl`` inside ``box`` (u0, v0, u1, v1). Returns (corr, u, v)."""
    h, w = tpl.shape
    t = (tpl - tpl.mean()) / (tpl.std() + 1e-6)
    u0, v0, u1, v1 = box
    win = np.lib.stride_tricks.sliding_window_view(gray[v0:v1 + h, u0:u1 + w], (h, w))
    mu = win.mean(axis=(-1, -2))
    sd = win.std(axis=(-1, -2)) + 1e-6
    corr = (np.einsum("ijkl,kl->ij", win, t) / (h * w) - mu * t.mean()) / sd
    j, i = np.unravel_index(int(np.argmax(corr)), corr.shape)
    return float(corr[j, i]), float(u0 + i + w / 2), float(v0 + j + h / 2)


def build(sc: dict[str, Any], phase: dict[str, Any], grab_jpeg: Callable[[str], bytes | None],
          say: Callable[[str], None]) -> list[Segment]:
    """Segments for one phase. ``center`` segments become calls that shift the later joint segments."""
    b = np.array(phase["backlash"], float) if phase.get("backlash") else None
    v0 = float(phase.get("v0", 1.0))
    segs: list[Segment] = []
    for spec in phase["segments"]:
        kind = spec["kind"]
        label = spec.get("label", kind)
        if kind == "joints":
            Q = np.atleast_2d(np.asarray(spec["q"], float))
            dt = float(spec.get("dt", 0.05))
            seg = Segment("joints", label=label, q=_feed_forward(Q, dt, b, v0), q_dt=dt)
            seg.meta = {"offset": spec.get("offset", "none"), "ramp": int(spec.get("ramp", 10))}
            segs.append(seg)
        elif kind == "grip":
            segs.append(Segment("grip", label=label, value=int(spec["value"])))
        elif kind == "wait":
            segs.append(Segment("wait", label=label, seconds=float(spec["seconds"])))
        elif kind == "center":
            segs.append(Segment("call", label=label, meta={"spec": spec}))
        else:
            raise ValueError(f"unknown segment kind {kind!r}")
    for k, seg in enumerate(segs):
        if seg.kind == "call":
            seg.fn = _centre_call(sc, seg.meta["spec"], segs[k + 1:], grab_jpeg, say)
    return segs


def _centre_call(sc: dict[str, Any], spec: dict[str, Any], later: list[Segment],
                 grab_jpeg: Callable[[str], bytes | None], say: Callable[[str], None]) -> Callable[[], Any]:
    from PIL import Image

    tpl = np.asarray(Image.open(Path(sc["_dir"]) / spec["template"]).convert("L"), float)
    axis = np.array(spec["axis_px"], float)
    px2mm = np.array(spec["px2mm"], float)
    J = np.array(spec["J"], float)
    max_err = float(spec.get("max_err_mm", 25.0))
    dead = float(spec.get("deadzone_mm", 2.0))

    def run() -> dict[str, Any]:
        errs = []
        for _ in range(3):  # three fresh frames; the median ignores one bad match
            jpeg = grab_jpeg(spec.get("camera", "top"))
            if jpeg is None:
                continue
            gray = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), float)
            corr, u, v = match_template(gray, tpl, spec["search"])
            if corr >= float(spec.get("min_corr", 0.6)):
                errs.append(px2mm @ (np.array([u, v]) - axis))
        if not errs:
            raise RuntimeError("centring: template not found in the camera frame")
        err = np.median(np.array(errs), axis=0)
        n = float(np.hypot(*err))
        if n > max_err:
            raise RuntimeError(f"centring: offset {n:.1f} mm is over {max_err:.0f} mm; stopping here")
        dq = -J @ err if n > dead else np.zeros(6)
        say(f"centring: offset ({err[0]:+.1f}, {err[1]:+.1f}) mm from {len(errs)} frames -> shift "
            + ", ".join(f"{x:+.1f}" for x in dq) + " deg")
        for seg in later:
            if seg.kind != "joints" or seg.q is None:
                continue
            mode, n_r = seg.meta.get("offset", "none"), max(1, seg.meta.get("ramp", 10))
            idx = np.arange(len(seg.q), dtype=float)
            if mode == "in":
                w = _smooth(idx / n_r)
            elif mode == "full":
                w = np.ones(len(seg.q))
            elif mode == "out":
                w = 1.0 - _smooth((idx + 1) / n_r)
            else:
                continue
            seg.q = seg.q + w[:, None] * dq[None, :]
        return {"err_mm": err.tolist(), "dq_deg": dq.tolist()}

    return run
