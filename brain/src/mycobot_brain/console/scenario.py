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
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mycobot_brain.console.motion import Segment


def load(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    sc = json.loads(path.read_text())
    sc["_dir"] = str(path.parent)
    if not sc.get("trigger") or not (sc.get("phases") or sc.get("generator")):
        raise ValueError(f"{path}: needs 'trigger' and 'phases' (or a 'generator')")
    return sc


def generate(sc: dict[str, Any], say: Callable[[str], None]) -> dict[str, Any]:
    """Run the scenario's generator (measure + plan now) and return the scenario it prints.

    ``"generator": {"cmd": [...], "timeout_s": 60}``: the command prints one JSON scenario on stdout
    (or ``{"error": "..."}``) and a short report on stderr, which goes to the page log line by line.
    """
    import subprocess

    g = sc["generator"]
    r = subprocess.run(g["cmd"], capture_output=True, text=True, timeout=float(g.get("timeout_s", 60)))
    for line in r.stderr.splitlines():
        if line.strip():
            say(line.strip()[:400])
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"generator printed no scenario ({exc})") from exc
    if out.get("error") or r.returncode != 0:
        raise RuntimeError(out.get("error") or f"generator exited {r.returncode}")
    out["_dir"] = sc["_dir"]
    out.setdefault("trigger", sc["trigger"])
    return out


def labels(sc: dict[str, Any]) -> list[str]:
    """One line per segment, for the page's step list."""
    out = []
    for ph in sc.get("phases", []):
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
    H, W = gray.shape
    u0, v0, u1, v1 = box
    # clamp the corners into the frame: a negative v0 would wrap around and search the bottom rows
    u0, v0 = max(int(u0), 0), max(int(v0), 0)
    u1, v1 = min(int(u1), W - w), min(int(v1), H - h)
    if u1 < u0 or v1 < v0:
        return -1.0, float("nan"), float("nan")
    win = np.lib.stride_tricks.sliding_window_view(gray[v0:v1 + h, u0:u1 + w], (h, w))
    mu = win.mean(axis=(-1, -2))
    sd = win.std(axis=(-1, -2)) + 1e-6
    corr = (np.einsum("ijkl,kl->ij", win, t) / (h * w) - mu * t.mean()) / sd
    j, i = np.unravel_index(int(np.argmax(corr)), corr.shape)
    return float(corr[j, i]), float(u0 + i + w / 2), float(v0 + j + h / 2)


def build(sc: dict[str, Any], phase: dict[str, Any], grab_jpeg: Callable[[str], bytes | None],
          say: Callable[[str], None], read_gripper: Callable[[], int] | None = None) -> list[Segment]:
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
        elif kind == "check":
            segs.append(Segment("call", label=label, meta={"check": spec}))
        elif kind == "align":
            segs.append(Segment("call", label=label, meta={"align": spec}))
        elif kind == "grip_check":
            segs.append(Segment("call", label=label, meta={"grip_check": spec}))
        else:
            raise ValueError(f"unknown segment kind {kind!r}")
    for k, seg in enumerate(segs):
        if seg.kind == "call" and "check" in seg.meta:
            seg.fn = _check_call(seg.meta["check"], say)
        elif seg.kind == "call" and "grip_check" in seg.meta:
            seg.fn = _grip_check_call(seg.meta["grip_check"], read_gripper, say)
        elif seg.kind == "call" and "align" in seg.meta:
            seg.fn = _align_call(seg.meta["align"], segs[k + 1:], say)
        elif seg.kind == "call":
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
        _apply_offset(later, dq)
        return {"err_mm": err.tolist(), "dq_deg": dq.tolist()}

    return run


def _apply_offset(later: list[Segment], dq: np.ndarray) -> None:
    """Add dq to the later joint segments by their offset mode: 'in' ramps in, 'full' adds, 'out' fades out.
    Only the first 'in' segment after the call ramps; later 'in' segments carry dq in full (the arm has already
    moved by it), so two corrections in a row stack instead of the second one restarting from zero."""
    ramped = False
    for seg in later:
        if seg.kind != "joints" or seg.q is None:
            continue
        mode, n_r = seg.meta.get("offset", "none"), max(1, seg.meta.get("ramp", 10))
        idx = np.arange(len(seg.q), dtype=float)
        if mode == "in" and ramped:
            w = np.ones(len(seg.q))
        elif mode == "in":
            ramped = True
            w = _smooth(idx / n_r)
        elif mode == "full":
            w = np.ones(len(seg.q))
        elif mode == "out":
            w = 1.0 - _smooth((idx + 1) / n_r)
        else:
            continue
        seg.q = seg.q + w[:, None] * dq[None, :]


def _align_call(spec: dict[str, Any], later: list[Segment], say: Callable[[str], None]) -> Callable[[], Any]:
    """Run spec["cmd"] (it measures the tool-to-target offset in one camera image and prints
    {"err_mm": [x, y], "msg": ...} as its last line), then shift the later segments by dq = -J @ err."""
    import subprocess

    J = np.array(spec["J"], float)
    max_err = float(spec.get("max_err_mm", 25.0))
    dead = float(spec.get("deadzone_mm", 1.5))

    def run() -> dict[str, Any]:
        r = subprocess.run(spec["cmd"], capture_output=True, text=True, timeout=float(spec.get("timeout_s", 15)))
        last = (r.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            out = json.loads(last)
        except json.JSONDecodeError:
            out = {"msg": last}
        if r.returncode != 0 or "err_mm" not in out:
            raise RuntimeError(out.get("msg") or f"align failed ({r.returncode})")
        err = np.array(out["err_mm"], float)
        n = float(np.hypot(*err))
        if not spec.get("apply", True):  # measure-only: log what it saw, move nothing
            say(f"align (측정만, 손은 안 옮김): {out.get('msg', '')}")
            return {"err_mm": err.tolist(), "dq_deg": [0.0] * J.shape[0]}
        if n > max_err:
            raise RuntimeError(f"align: offset {n:.1f} mm is over {max_err:.0f} mm; stopping here")
        if "dq_deg" in out:  # the measuring script solved the joint change itself (exact IK, not -J err)
            dq = np.array(out["dq_deg"], float) if n > dead else np.zeros(J.shape[0])
        else:
            dq = -J @ err if n > dead else np.zeros(J.shape[0])
        say(f"align: {out.get('msg', '')} -> shift " + ", ".join(f"{x:+.1f}" for x in dq) + " deg")
        _apply_offset(later, dq)
        return {"err_mm": err.tolist(), "dq_deg": dq.tolist()}

    return run


def _grip_check_call(spec: dict[str, Any], read_gripper: Callable[[], int] | None,
                     say: Callable[[str], None]) -> Callable[[], Any]:
    """Stop unless the gripper's reported opening is inside [min, max]: closed on a vial it stops at the vial's
    width, closed on nothing it goes further (2026-11-26: runs went on to pour an empty hand)."""
    lo, hi = float(spec["min"]), float(spec["max"])
    what = spec.get("what", "병")

    def run() -> dict[str, Any]:
        if read_gripper is None:
            raise RuntimeError("grip_check: 이 팔의 그리퍼 값을 읽을 수 없어요")
        vals = []
        for _ in range(int(spec.get("reads", 3))):
            vals.append(read_gripper())
            time.sleep(float(spec.get("gap_s", 0.1)))
        v = float(np.median(vals))
        if not lo <= v <= hi:
            why = "닫힘 쪽(빈손)" if v < lo else "열림 쪽"
            raise RuntimeError(f"{what}을 잡지 못했어요: 그리퍼 값 {v:.0f} ({why}; 잡았을 때 {lo:.0f}–{hi:.0f})")
        say(f"grip check: 그리퍼 값 {v:.0f} ({lo:.0f}–{hi:.0f}) — {what}을 잡았어요")
        return {"value": v}

    return run


def _check_call(spec: dict[str, Any], say: Callable[[str], None]) -> Callable[[], Any]:
    """Run spec["cmd"] while the arm holds; a non-zero exit stops the scenario with the command's last line."""
    import subprocess

    def run() -> dict[str, Any]:
        r = subprocess.run(spec["cmd"], capture_output=True, text=True, timeout=float(spec.get("timeout_s", 15)))
        msg = (r.stdout.strip().splitlines() or [""])[-1]
        if r.returncode != 0:
            raise RuntimeError(msg or f"check failed ({r.returncode})")
        say(f"check: {msg}")
        return {"msg": msg}

    return run
