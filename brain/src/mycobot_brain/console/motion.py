"""Turn an action chunk into one continuous joint stream, through the QP controller.

Why this exists: the earlier loops sent one target, waited for the arm to
arrive, then sent the next. myCobot's firmware decelerates to rest at every
``send_angles`` target, so the arm stopped every few degrees ("툭툭 끊김"), and
the arrival wait made it slow. Here a whole chunk (several targets) becomes one
time-parametrised reference; ``mycobot_control.qp.ChunkExecutor`` tracks it at
a fixed rate under joint position/velocity/acceleration limits, and every tick
is streamed to the arm server without waiting for arrival.

Big moves can go through the cuRobo plan server (collision-free joint path);
the QP then tracks that path, so the limits and the streaming are the same
either way. The arm server keeps the last word on every command.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from mycobot_brain.link import ArmLink, LinkError, Rejected


class MotionStopped(RuntimeError):
    """Stop was pressed, or the arm stopped tracking."""


@dataclass
class Segment:
    kind: str  # "cart" | "joints" | "grip" | "wait" | "call"
    label: str = ""
    points_base: list[np.ndarray] = field(default_factory=list)  # tool-point waypoints, arm base frame
    q: np.ndarray | None = None  # "joints": N x 6 deg path (or 1 x 6 goal)
    q_dt: float = 0.05
    R_end: np.ndarray | None = None  # "cart": tool orientation at the end (base frame); None keeps the current one
    value: int = 0
    seconds: float = 0.0
    fn: Callable[[], Any] | None = None  # "call": runs while the arm holds; may edit later segments' q
    meta: dict[str, Any] = field(default_factory=dict)


def _rot_z_to(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, float)


def curobo_plan(api: str, q_start: list[float], goal: dict[str, Any], obstacles: list[dict], dt: float,
                time_scale: float, timeout: float = 20.0) -> dict[str, Any]:
    body = json.dumps({"q_start": q_start, "goal": goal, "obstacles": obstacles, "dt": dt, "time_scale": time_scale}).encode()
    req = urllib.request.Request(api.rstrip("/") + "/plan", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def curobo_check(api: str, Q: np.ndarray, obstacles: list[dict], stride: int = 2, timeout: float = 20.0) -> dict[str, Any]:
    body = json.dumps({"q": np.round(Q, 3).tolist(), "obstacles": obstacles, "stride": stride}).encode()
    req = urllib.request.Request(api.rstrip("/") + "/check", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def curobo_spheres(api: str, q: np.ndarray, timeout: float = 5.0) -> list[list[float]]:
    req = urllib.request.Request(api.rstrip("/") + "/spheres", data=json.dumps({"q": np.round(q, 3).tolist()}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["spheres"]


def curobo_post(api: str, path: str, payload: dict[str, Any], timeout: float = 1.0) -> None:
    req = urllib.request.Request(api.rstrip("/") + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except OSError:
        pass  # the view is optional


class MotionRunner:
    """Streams segments to one arm. One runner per console; one motion at a time."""

    def __init__(self, rate_hz: float = 20.0, speed_param: int = 100, max_track_err_deg: float = 12.0,
                 lookahead_ticks: int = 3) -> None:
        from mycobot_control.qp import ControllerConfig, Kinematics  # noqa: F401 - fail early if missing

        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.speed_param = speed_param
        self.max_track_err_deg = max_track_err_deg
        self.lookahead_ticks = lookahead_ticks
        self.stop_evt = threading.Event()
        #: Per-arm steady-state offset, measured minus commanded, deg. Measured
        #: 2026-11-24 on the .77 arm: J4 settles 1.2 deg past its command, so
        #: "hold where you are" (command = measured) ratcheted J4 up 13 deg over
        #: ten re-enables. Commands go out as q - bias; learned at each run's end.
        self.bias: dict[str, np.ndarray] = {}
        self.progress = 0.0
        self.measured_speed_dps = 0.0
        self._lock = threading.Lock()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def kinematics(tool_mm: np.ndarray):
        from mycobot_control.qp import Kinematics

        T = np.eye(4)
        T[:3, 3] = tool_mm
        return Kinematics(tool=T)

    def controller(self, kin, speed_dps: float, joint_weights: list[float] | None = None,
                   z_min_mm: float | None = None):
        from mycobot_control.qp import ControllerConfig, QPController

        kw: dict[str, Any] = dict(
            dt=self.dt,
            qd_max_deg_s=float(speed_dps),
            # Linear speed scales with the joint cap: at 60 deg/s the tool may go 150 mm/s.
            v_max_mm_s=2.5 * float(speed_dps),
            z_min_mm=z_min_mm,
        )
        if joint_weights is not None:
            kw["joint_weights"] = tuple(float(w) for w in joint_weights)
        return QPController(kin, ControllerConfig(**kw))

    @staticmethod
    def cart_poses(p_from: np.ndarray, points: list[np.ndarray], R: np.ndarray, v_mm_s: float, dt: float,
                   min_T: float = 0.8, R_end: np.ndarray | None = None, w_deg_s: float = 60.0) -> list[np.ndarray]:
        """One min-jerk profile along the whole polyline: no stop at the corners.

        The QP's acceleration limit rounds the corners; a stop only happens at
        the end of the chunk. With ``R_end`` the orientation turns from ``R`` to
        it along the same profile (geodesic).
        """
        from mycobot_control.qp import min_jerk, so3_exp, so3_log

        pts = [np.asarray(p_from, float)] + [np.asarray(p, float) for p in points]
        seg = np.array([np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])])
        L = float(seg.sum())
        rot = np.zeros(3) if R_end is None else so3_log(R.T @ R_end)
        ang = float(np.degrees(np.linalg.norm(rot)))
        # min-jerk peak speed is 1.875 x mean: size T so the peak stays under v_mm_s (and w_deg_s)
        T = max(min_T, 1.875 * L / max(v_mm_s, 1.0), 1.875 * ang / max(w_deg_s, 1.0))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        out = []
        n = max(2, int(np.ceil(T / dt)) + 1)
        for k in range(n):
            s = float(min_jerk(min(1.0, k * dt / T))) * L
            i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
            f = 0.0 if seg[i] < 1e-9 else (s - cum[i]) / seg[i]
            p = pts[i] + (pts[i + 1] - pts[i]) * min(1.0, f)
            Tk = np.eye(4)
            Tk[:3, :3] = R if R_end is None else R @ so3_exp(rot * (s / L if L > 1e-9 else float(min_jerk(min(1.0, k * dt / T)))))
            Tk[:3, 3] = p
            out.append(Tk)
        return out

    def preview(self, kin, segments: list[Segment], q0: np.ndarray, speed_dps: float,
                z_min_mm: float | None = None, joint_weights: list[float] | None = None) -> list[np.ndarray | None]:
        """The joint path each segment will command (no lookahead), for checking and display.

        The QP only keeps the tool point above the table; links are not checked here.
        That is what cuRobo's ``/check`` is for (2026-11-24: turning J1 from the .77
        park pose drove the elbow into the board edge while the tool was in free air).
        """
        from mycobot_control.qp import ChunkExecutor

        ex = ChunkExecutor(self.controller(kin, speed_dps, joint_weights, z_min_mm), q0, lookahead_ticks=0,
                           raise_on_track_error=False)
        out: list[np.ndarray | None] = []
        for seg in segments:
            if seg.kind == "cart":
                T0 = kin.fk(ex.q_cmd)
                ex.set_cartesian_chunk(self.cart_poses(T0[:3, 3], seg.points_base, T0[:3, :3], 2.5 * speed_dps, self.dt,
                                                       R_end=seg.R_end))
                out.append(np.asarray(ex.plan(), float))
            elif seg.kind == "joints":
                Q = np.atleast_2d(np.asarray(seg.q, float))
                if len(Q) == 1:
                    dq = float(np.max(np.abs(Q[0] - ex.q_cmd)))
                    ex.goal_joints(Q[0], max(0.8, 1.875 * dq / speed_dps))
                else:
                    ex.set_joint_chunk(Q, seg.q_dt)
                out.append(np.asarray(ex.plan(), float))
            else:
                out.append(None)
        return out

    # -- execution ---------------------------------------------------------

    def stop(self) -> None:
        self.stop_evt.set()

    def run(self, link: ArmLink, kin, segments: list[Segment], speed_dps: float,
            on_step: Callable[[int], None] = lambda i: None,
            on_tick: Callable[[np.ndarray], None] = lambda q: None,
            joint_weights: list[float] | None = None, arm: str = "", z_min_mm: float | None = None) -> dict[str, Any]:
        """Execute ``segments`` back to back. Returns stats; raises MotionStopped."""
        from mycobot_control.qp import ChunkExecutor

        if not self._lock.acquire(blocking=False):
            raise MotionStopped("another motion is running")
        self.stop_evt.clear()
        self.progress = 0.0
        stats = {"ticks": 0, "fails": 0, "max_err_deg": 0.0, "t": 0.0}
        self._arm = arm
        bias = self.bias.setdefault(arm, np.zeros(6))
        t_start = time.monotonic()
        try:
            st = link.wait_state(timeout=2.0)
            if st.get("angles_deg") is None:
                raise MotionStopped("no joint reading from the arm")
            q_meas = np.array(st["angles_deg"], float)
            ctrl = self.controller(kin, speed_dps, joint_weights, z_min_mm)
            # The arm trails its command by ~0.15-0.2 s (2026-11-24, J6 sine at 20 Hz), and the
            # newest reading is up to 0.1 s old: run 3 ticks ahead so the arm is on schedule
            # and the tracking check compares like with like.
            ex = ChunkExecutor(ctrl, q_meas, lookahead_ticks=self.lookahead_ticks,
                               max_track_err_deg=self.max_track_err_deg)
            link.enable()
            total = max(1, len(segments))
            last_q, last_t = q_meas, time.monotonic()
            for i, seg in enumerate(segments):
                on_step(i)
                if seg.kind == "grip":
                    link.gripper(int(seg.value), 50)
                    self._hold(link, ex, 0.8, stats, on_tick)
                elif seg.kind == "wait":
                    self._hold(link, ex, seg.seconds, stats, on_tick)
                elif seg.kind == "call":
                    self._hold_while(link, ex, seg.fn, stats)
                elif seg.kind == "cart":
                    T0 = kin.fk(ex.q_cmd)
                    poses = self.cart_poses(T0[:3, 3], seg.points_base, T0[:3, :3], 2.5 * speed_dps, self.dt,
                                            R_end=seg.R_end)
                    n = ex.set_cartesian_chunk(poses)
                    self._drive(link, ex, stats, on_tick, i, total, n)
                elif seg.kind == "joints":
                    Q = np.atleast_2d(np.asarray(seg.q, float))
                    if len(Q) == 1:
                        dq = float(np.max(np.abs(Q[0] - ex.q_cmd)))
                        n = ex.goal_joints(Q[0], max(0.8, 1.875 * dq / speed_dps))
                    else:
                        n = ex.set_joint_chunk(Q, seg.q_dt)
                    self._drive(link, ex, stats, on_tick, i, total, n)
                self.progress = (i + 1) / total
            time.sleep(0.6)  # let the servos settle before reading the offset
            s = link.wait_state(timeout=1.0)
            q_end = np.array(s["angles_deg"], float)
            err = q_end - ex.q_cmd
            stats["final_err_deg"] = float(np.max(np.abs(err)))
            # learn the offset only from small, settled errors (a blocked joint is not a bias)
            upd = np.where(np.abs(err) < 3.0, 0.7 * err, 0.0)
            self.bias[arm] = np.clip(bias + upd, -3.0, 3.0)
            stats["bias"] = np.round(self.bias[arm], 2).tolist()
            dtm = time.monotonic() - last_t
            self.measured_speed_dps = float(np.max(np.abs(q_end - last_q))) / max(dtm, 1e-3)
            return stats
        finally:
            stats["t"] = time.monotonic() - t_start
            try:
                if self.stop_evt.is_set():
                    link.stop()
                link.disable()
            except LinkError:
                pass
            self._lock.release()

    def _send(self, link: ArmLink, q_cmd: np.ndarray, stats: dict) -> None:
        for attempt in range(6):
            try:
                q_send = np.asarray(q_cmd, float) - self.bias.get(getattr(self, "_arm", ""), np.zeros(6))
                link.move_joints([round(float(x), 2) for x in q_send], self.speed_param)
                return
            except (Rejected, LinkError) as exc:
                stats["fails"] += 1
                if self.stop_evt.is_set() or attempt == 5 or "20" in str(exc) and "deg" in str(exc):
                    raise MotionStopped(f"arm refused: {exc}") from exc
                time.sleep(0.25)
                try:
                    link.wait_state(timeout=1.0)
                    link.enable()
                except LinkError:
                    pass

    def _measured(self, link: ArmLink) -> np.ndarray | None:
        s = link.latest_state()
        if not s or s.get("angles_deg") is None:
            return None
        return np.array(s["angles_deg"], float)

    def _drive(self, link: ArmLink, ex, stats: dict, on_tick, i: int, total: int, n: int) -> None:
        from mycobot_control.qp import TrackingError

        t_next = time.monotonic()
        prev = None
        while not ex.done:
            if self.stop_evt.is_set():
                raise MotionStopped("stopped")
            try:
                q_cmd = np.asarray(ex.tick(self._measured(link)), float)
            except TrackingError as exc:
                raise MotionStopped(f"tracking: {exc}") from exc
            if ex.last_track_err_deg is not None:
                stats["max_err_deg"] = max(stats["max_err_deg"], ex.last_track_err_deg)
            self._send(link, q_cmd, stats)
            stats["ticks"] += 1
            on_tick(q_cmd)
            if prev is not None:
                self.measured_speed_dps = float(np.max(np.abs(q_cmd - prev))) / self.dt
            prev = q_cmd
            self.progress = (i + min(1.0, 1.0 - ex.remaining_ticks / max(1, n))) / total
            t_next += self.dt
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.monotonic()

    def _hold_while(self, link: ArmLink, ex, fn: Callable[[], Any] | None, stats: dict) -> None:
        """Hold the current command (the arm server's watchdog needs traffic) while ``fn`` runs."""
        if fn is None:
            return
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["out"] = fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised on the motion thread
                box["exc"] = exc

        th = threading.Thread(target=work, name="segment-call", daemon=True)
        th.start()
        while th.is_alive():
            if self.stop_evt.is_set():
                raise MotionStopped("stopped")
            self._send(link, ex.q_cmd, stats)
            th.join(self.dt)
        if "exc" in box:
            exc = box["exc"]
            raise exc if isinstance(exc, MotionStopped) else MotionStopped(f"{type(exc).__name__}: {exc}")

    def _hold(self, link: ArmLink, ex, seconds: float, stats: dict, on_tick) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self.stop_evt.is_set():
                raise MotionStopped("stopped")
            self._send(link, ex.q_cmd, stats)
            time.sleep(self.dt)
