#!/usr/bin/env python3
"""cuRobo planning service + live viser view for the myCobot 280 JN (no hardware).

One process, one warm :class:`mycobot_planner.MycobotPlanner`, two ports on 127.0.0.1:

* viser (``--viser-port``, default 8080): measured arm (solid), last plan as a looping
  ghost (teal) + TCP path, obstacles as translucent boxes, a status panel.
* JSON HTTP API (``--port``, default 8096).

Frames and units (all callers): joints in **pymycobot degrees**; poses and obstacles in the
arm's **pymycobot / DH base frame, mm**. A pose goal is the **flange** frame (orientation =
flange frame in the DH base frame, position = flange origin) unless ``tcp_offset_mm`` (a point
in the flange frame, default [0,0,0]) says the goal refers to a tool point. cuRobo's own frame
(base 16.9 mm below the DH base; its ``tcp`` link 78.9 mm along flange z) never leaves this file.

Endpoints::

    GET  /health
    POST /plan      {"q_start": [6], "goal": {"joints": [6]} | {"pose": {"pos_mm": [3], "quat_wxyz": [4]}},
                     "tcp_offset_mm": [3]?, "obstacles": [...]?, "dt": 0.05?, "time_scale": 1.0?}
    POST /state     {"q": [6]}
    POST /world     {"obstacles": [{"name", "dims_mm": [3], "pose": [x,y,z mm, qw,qx,qy,qz]}, ...]}
    POST /show_plan {"q": [[6]...], "dt": 0.05}

Timing: the planner is built at ``--ceiling-joint-vel-deg``; every plan is re-timed so that
``time_scale`` = 1 means peak joint speed = ``--max-joint-vel-deg``, ``time_scale`` < 1 is faster
(clamped by the ceiling and the accel cap, never beyond), > 1 slower. Time-scaling keeps the path.

Usage:  scripts/run.sh server
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import mycobot_planner as mpl  # noqa: E402
from curobo.logging import setup_logger  # noqa: E402
from curobo.types import ContentPath  # noqa: E402
from curobo.viewer import ViserVisualizer  # noqa: E402
from mycobot_planner import ARM_JOINTS, HOME_Q, ROBOT_YML, Cuboid, MycobotPlanner  # noqa: E402

TEAL = (15, 188, 196)  # #0FBCC4
GREY = (150, 150, 150)
DZ_MM = 16.9  # cuRobo base origin is this far below the DH base origin: z_cu = z_dh + DZ
TCP_FLANGE_MM = 78.9  # cuRobo `tcp` link: along flange z
DH = [(0.0, 139.42, 0.0, math.pi / 2), (-math.pi / 2, 0.0, -110.4, 0.0), (0.0, 0.0, -96.0, 0.0),
      (-math.pi / 2, 63.4, 0.0, math.pi / 2), (math.pi / 2, 75.05, 0.0, -math.pi / 2), (0.0, 45.6, 0.0, 0.0)]


# ---- geometry helpers -------------------------------------------------------------------------
def dh_fk(q_deg) -> np.ndarray:
    """Flange pose (4x4, mm) in the DH base frame, standard DH of the myCobot 280."""
    T = np.eye(4)
    for qi, (off, d, a, al) in zip(np.radians(q_deg), DH):
        th = qi + off
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]])
    return T


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R) -> list:
    w = math.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-3:
        return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
    v = [0.0, 0.0, 0.0]
    v[i] = s / 4
    v[j] = (R[j, i] + R[i, j]) / s
    v[k] = (R[k, i] + R[i, k]) / s
    return [(R[k, j] - R[j, k]) / s] + v


class BadRequest(ValueError):
    pass


def vec(x, n, name) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise BadRequest(f"{name} must be a list of {n} numbers")
    if a.shape[0] != n or not np.all(np.isfinite(a)):
        raise BadRequest(f"{name} must be {n} finite numbers, got {x!r}")
    return a


# ---- the service ------------------------------------------------------------------------------
class Service:
    def __init__(self, args) -> None:
        self.args = args
        self.robot = Path(args.robot)
        self.nominal = math.radians(args.max_joint_vel_deg)
        self.ceiling = max(self.nominal, math.radians(args.ceiling_joint_vel_deg))
        if args.max_accel_deg is not None:
            mpl.MAX_ACCEL = math.radians(args.max_accel_deg)
        self.accel = mpl.MAX_ACCEL
        self.mp = MycobotPlanner(robot_yml=self.robot, speed_fraction=self.ceiling / mpl.URDF_MAX_SPEED)
        self.warm = False
        self.plan_lock = threading.Lock()
        self.world: list[dict] = []
        self._world_sig = None
        self.last_status = "starting"
        self._calibrate_frames()
        self._init_viser()

    # -- frames --
    def _calibrate_frames(self) -> None:
        """Fit the constant cuRobo tcp <-> DH flange transform from FK at random joints; keep residuals."""
        rng = np.random.default_rng(1)
        Q = rng.uniform(-100, 100, (24, 6))
        Rs, A, B = [], [], []
        for q in Q:
            T = dh_fk(q)
            pose = self.mp.fk(np.radians(q))
            pc = pose.position.reshape(-1).cpu().numpy() * 1000
            Rc = quat_to_R(pose.quaternion.reshape(-1).cpu().numpy())
            Rs.append(T[:3, :3].T @ Rc)
            A.append(T[:3, :3])
            B.append(pc - T[:3, 3] - [0, 0, DZ_MM])
        Rs = np.array(Rs)
        # constant rotation (should be identity for the gripper model; recomputed for any yml)
        U, _, Vt = np.linalg.svd(Rs.mean(0))
        self.R_off = U @ Vt
        A, B = np.vstack(A), np.concatenate(B)
        t0 = np.array([0, 0, TCP_FLANGE_MM])
        self.frame_check = {
            "rot_const_spread": float(np.abs(Rs - self.R_off).max()),
            "R_off_is_identity": bool(np.allclose(self.R_off, np.eye(3), atol=1e-3)),
            "pos_rms_mm_with_[0,0,78.9]+dz16.9": float(np.sqrt(np.mean((A @ t0 - B) ** 2))),
            "fit_tcp_offset_flange_mm": np.round(np.linalg.lstsq(A, B, rcond=None)[0], 2).tolist(),
        }

    def flange_goal_to_cu(self, pos_mm, quat, tcp_off) -> tuple[list, list]:
        """Pose of the DH-frame point `flange + R*tcp_off` -> cuRobo tcp (m, wxyz)."""
        R = quat_to_R(quat)
        flange = np.asarray(pos_mm) - R @ tcp_off
        p = flange + R @ np.array([0, 0, TCP_FLANGE_MM]) + [0, 0, DZ_MM]
        return (p / 1000).tolist(), R_to_quat(R @ self.R_off)

    # -- viser --
    def _init_viser(self) -> None:
        a = self.args
        self.viz = ViserVisualizer(content_path=ContentPath(robot_config_file=str(self.robot)),
                                   connect_ip="127.0.0.1", connect_port=a.viser_port,
                                   add_control_frames=False, visualize_robot_spheres=False)
        s = self.server = self.viz._server
        self.viz._viser_urdf.show_visual = False  # viewer.py's default-coloured robot; replaced by the two below
        from viser.extras import ViserUrdf
        s.gui.configure_theme(show_logo=False, show_share_button=False, brand_color=TEAL)
        try:
            s.gui.main_panel.dock_right()
        except Exception:
            pass
        try:
            s.scene.set_background_image(np.full((2, 2, 3), 255, dtype=np.uint8))
        except Exception:
            pass
        s.initial_camera.position = (0.42, -0.42, 0.32)
        s.initial_camera.look_at = (0.08, 0.0, 0.14)
        s.initial_camera.up = (0.0, 0.0, 1.0)
        urdf = self.viz._urdf
        self.measured = ViserUrdf(s, urdf, root_node_name="/measured", mesh_color_override=(0.30, 0.32, 0.35))
        self.ghost = ViserUrdf(s, urdf, root_node_name="/ghost",
                               mesh_color_override=(TEAL[0] / 255, TEAL[1] / 255, TEAL[2] / 255, 0.35))
        self._joint_order = list(self.measured.get_actuated_joint_names())
        if self._joint_order != list(ARM_JOINTS):
            raise RuntimeError(f"viser joint order {self._joint_order} != {ARM_JOINTS}")
        self._viz_lock = threading.Lock()
        self.status_md = s.gui.add_markdown("**cuRobo planner** - warming up")
        self.obs_handles: list = []
        self.path_handle = None
        self.goal_handle = None
        self.anim = {"q": None, "dt": 0.05, "gen": 0}
        self.set_state(np.degrees(HOME_Q))
        self._set_ghost(np.degrees(HOME_Q))
        self.ghost.show_visual = False
        threading.Thread(target=self._anim_loop, daemon=True).start()

    def _cfg(self, urdf_handle, q_deg) -> None:
        urdf_handle.update_cfg(np.radians(np.asarray(q_deg, float)))

    def set_state(self, q_deg) -> None:
        with self._viz_lock:
            self._cfg(self.measured, q_deg)

    def _set_ghost(self, q_deg) -> None:
        with self._viz_lock:
            self._cfg(self.ghost, q_deg)

    def _anim_loop(self) -> None:
        i, gen = 0, -1
        while True:
            q, dt = self.anim["q"], self.anim["dt"]
            if q is None:
                time.sleep(0.1)
                continue
            if gen != self.anim["gen"]:
                gen, i = self.anim["gen"], 0
            self._set_ghost(q[i])
            if i >= len(q) - 1:
                time.sleep(0.8)  # hold at the end, then loop
                i = 0
            else:
                i += 1
            time.sleep(max(dt, 0.02))

    def show_plan(self, q_deg: np.ndarray, dt: float) -> None:
        self.ghost.show_visual = True
        self.anim.update(q=q_deg, dt=dt, gen=self.anim["gen"] + 1)
        pts = np.array([dh_fk(q)[:3, 3] for q in q_deg[:: max(1, len(q_deg) // 150)]] + [dh_fk(q_deg[-1])[:3, 3]])
        pts = (pts + [0, 0, DZ_MM]) / 1000.0
        self._draw_path(pts)

    def _draw_path(self, pts_m: np.ndarray) -> None:
        for h in (self.path_handle, self.goal_handle):
            if h is not None:
                h.remove()
        self.path_handle = self.goal_handle = None
        if len(pts_m) < 2:
            return
        seg = np.stack([pts_m[:-1], pts_m[1:]], axis=1).astype(np.float32)
        self.path_handle = self.server.scene.add_line_segments("/plan/tcp_path", seg, colors=TEAL, line_width=3.0)
        self.goal_handle = self.server.scene.add_icosphere("/plan/goal", radius=0.006, color=TEAL,
                                                           position=tuple(pts_m[-1]))

    def set_status(self, headline: str, *lines: str) -> None:
        self.last_status = headline
        body = "\n".join(f"- {l}" for l in lines)
        self.status_md.content = f"**{headline}**\n\n{body}" if body else f"**{headline}**"

    # -- world --
    def _parse_obstacles(self, obs) -> list[dict]:
        if obs is None:
            return []
        if not isinstance(obs, list) or len(obs) > 15:
            raise BadRequest("obstacles must be a list of at most 15 boxes")
        out = []
        for i, o in enumerate(obs):
            try:
                name = str(o.get("name", f"obs{i}"))
                dims = vec(o["dims_mm"], 3, f"obstacles[{i}].dims_mm")
                pose = vec(o["pose"], 7, f"obstacles[{i}].pose")
            except (KeyError, AttributeError):
                raise BadRequest(f"obstacles[{i}] needs name, dims_mm[3], pose[7]")
            if np.any(dims <= 0):
                raise BadRequest(f"obstacles[{i}].dims_mm must be positive")
            out.append({"name": name, "dims_mm": dims.tolist(), "pose": pose.tolist()})
        return out

    def set_world(self, obs: list[dict]) -> None:
        """Replace the world cuboids (caller holds plan_lock) and redraw them in viser."""
        sig = json.dumps(obs, sort_keys=True)
        if sig == self._world_sig:
            return
        SceneCfg = mpl.SceneCfg
        cubs = []
        for o in obs:
            p = np.asarray(o["pose"], float)
            pos = (p[:3] + [0, 0, DZ_MM]) / 1000.0
            cubs.append(Cuboid(name=o["name"], dims=(np.asarray(o["dims_mm"]) / 1000).tolist(),
                               pose=pos.tolist() + p[3:].tolist()))
        floor = [c for c in self.mp.obstacles if c.name == "floor"]
        self.mp.obstacles = floor + cubs
        self.mp.planner.update_world(SceneCfg(cuboid=list(self.mp.obstacles)))
        self.world, self._world_sig = obs, sig
        for h in self.obs_handles:
            h.remove()
        self.obs_handles = []
        for o in obs:
            p = np.asarray(o["pose"], float)
            self.obs_handles.append(self.server.scene.add_box(
                f"/world/{o['name']}", dimensions=tuple((np.asarray(o["dims_mm"]) / 1000).tolist()),
                position=tuple(((p[:3] + [0, 0, DZ_MM]) / 1000).tolist()), wxyz=tuple(p[3:].tolist()),
                color=GREY, opacity=0.4))

    # -- planning --
    def warmup(self) -> None:
        self.mp.planner.warmup(enable_graph=True, num_warmup_iterations=3)
        with self.plan_lock:  # first real solves compile more paths than warmup() does
            for g in ([0, -20, -40, -20, 0, 0], [20, -50, -50, -30, 10, 0]):
                self.mp.plan_to_joints(np.radians([0, -28.6, -57.3, -34.4, 0, 0]), np.radians(g))
        self.warm = True
        self.set_status("Ready", f"nominal {math.degrees(self.nominal):.0f} deg/s, ceiling "
                        f"{math.degrees(self.ceiling):.0f} deg/s, accel {math.degrees(self.accel):.0f} deg/s^2")

    def plan(self, req: dict) -> dict:
        t0 = time.perf_counter()
        q0 = vec(req.get("q_start"), 6, "q_start")
        goal = req.get("goal")
        if not isinstance(goal, dict) or not (("joints" in goal) ^ ("pose" in goal)):
            raise BadRequest('goal must be {"joints": [6]} or {"pose": {"pos_mm", "quat_wxyz"}}')
        dt = float(req.get("dt", 0.05))
        if not 0.005 <= dt <= 1.0:
            raise BadRequest("dt must be in [0.005, 1] s")
        ts = float(req.get("time_scale", 1.0))
        if not ts > 0:
            raise BadRequest("time_scale must be > 0")
        tcp_off = vec(req.get("tcp_offset_mm", [0, 0, 0]), 3, "tcp_offset_mm")
        obs = self._parse_obstacles(req["obstacles"]) if "obstacles" in req else None
        if "joints" in goal:
            qg = vec(goal["joints"], 6, "goal.joints")
        else:
            pg = goal["pose"]
            pos = vec(pg.get("pos_mm"), 3, "goal.pose.pos_mm")
            quat = vec(pg.get("quat_wxyz"), 4, "goal.pose.quat_wxyz")
            if np.linalg.norm(quat) < 1e-6:
                raise BadRequest("zero quaternion")
        with self.plan_lock:
            if obs is not None:
                self.set_world(obs)
            rep = self.mp.collision_report(np.radians(q0))
            if rep["self"] or rep["obstacles"] or rep["limits"]:
                why = {"self": {"/".join(k): round(v * 1000, 1) for k, v in rep["self"].items()},
                       "obstacles": {"/".join(k): round(v * 1000, 1) for k, v in rep["obstacles"].items()},
                       "limits": rep["limits"]}
                return self._fail("invalid start", "q_start is in collision or outside the joint limits "
                                  "(3 deg margin)", t0, detail=why)
            if "joints" in goal:
                res = self.mp.plan_to_joints(np.radians(q0), np.radians(qg), label="api")
            else:
                p_cu, q_cu = self.flange_goal_to_cu(pos, quat, tcp_off)
                res = self.mp.plan_to_pose(np.radians(q0), p_cu, q_cu, label="api")
                if res.success and req.get("refine", True):
                    # cuRobo's model and the DH model differ by ~1-2 mm (configuration dependent):
                    # shift the cuRobo goal once by the DH-measured end error and re-plan.
                    err = self._dh_end_err(np.degrees(res.q[-1]), pos, quat, tcp_off)
                    dh_err0 = float(np.linalg.norm(err))
                    if dh_err0 > 0.4:
                        res2 = self.mp.plan_to_pose(np.radians(q0), (np.asarray(p_cu) + err / 1000).tolist(), q_cu, label="api")
                        if res2.success and np.linalg.norm(self._dh_end_err(np.degrees(res2.q[-1]), pos, quat, tcp_off)) < dh_err0:
                            res = res2
        if not res.success:
            stage = "ik" if res.stage == "ik" else "trajectory"
            return self._fail(stage, res.reason, t0)
        q = np.degrees(res.q)
        q, k, bound, T = self._retime(q, res.dt, dt, ts)
        out = self._finish(q, dt, T, t0, tcp_off)
        out.update(time_scale_requested=ts, speed_bound_by=bound, stretch_factor=round(k, 3),
                   curobo_solve_ms=round(res.solve_time_s * 1000, 1))
        if "pose" in goal:
            e = self._dh_end_err(q[-1], pos, quat, tcp_off)
            out["goal_error_mm_dh"] = round(float(np.linalg.norm(e)), 2)
        else:
            out["goal_error_deg_max"] = round(float(np.abs(q[-1] - qg).max()), 3)
        self.show_plan(q, dt)
        self.set_status("Plan found", f"{len(q)} points, {T:.2f} s, plan {out['plan_ms']:.0f} ms",
                        f"peak {out['peak_joint_vel_deg_s']:.0f} deg/s (bound: {bound}), time_scale {ts:.2f}")
        return out

    def spheres(self, req: dict) -> dict:
        """The robot's collision spheres at q, in the DH base frame (mm): [[x, y, z, r], ...].

        The console uses them to cut the arm out of the depth camera's obstacle map.
        """
        q = vec(req.get("q"), 6, "q")
        with self.plan_lock:
            sph = self.mp.planner.compute_kinematics(self.mp.js(np.radians(q))).robot_spheres.reshape(-1, 4).cpu().numpy()
        sph = sph[sph[:, 3] > 0] * 1000.0
        sph[:, 2] -= DZ_MM
        return {"ok": True, "spheres": np.round(sph, 1).tolist()}

    def check(self, req: dict) -> dict:
        """Collision/limit check of an externally produced joint path (the console's QP chunks).

        Body: {"q": [[6 deg]...], "obstacles"?: [...] (replaces the world like /plan),
        "stride"?: check every k-th sample (the last one always)}. Returns ok=false at the
        first sample that touches an obstacle, itself, or a joint limit (3 deg margin).
        """
        t0 = time.perf_counter()
        q = np.asarray(req.get("q"), float)
        if q.ndim != 2 or q.shape[1] != 6 or len(q) < 1 or not np.all(np.isfinite(q)):
            raise BadRequest("q must be an [N][6] list of degrees")
        stride = max(1, int(req.get("stride", 1)))
        obs = self._parse_obstacles(req["obstacles"]) if "obstacles" in req else None
        idx = list(range(0, len(q), stride))
        if idx[-1] != len(q) - 1:
            idx.append(len(q) - 1)
        with self.plan_lock:
            if obs is not None:
                self.set_world(obs)
            for i in idx:
                rep = self.mp.collision_report(np.radians(q[i]))
                if rep["self"] or rep["obstacles"] or rep["limits"]:
                    why = {"self": {"/".join(k): round(v * 1000, 1) for k, v in rep["self"].items()},
                           "obstacles": {"/".join(k): round(v * 1000, 1) for k, v in rep["obstacles"].items()},
                           "limits": rep["limits"]}
                    return {"ok": False, "status": "collision", "index": i, "n": len(q), "detail": why,
                            "check_ms": round((time.perf_counter() - t0) * 1e3, 1)}
        return {"ok": True, "status": "clear", "n": len(q), "checked": len(idx),
                "check_ms": round((time.perf_counter() - t0) * 1e3, 1)}

    def _dh_end_err(self, q_deg, pos, quat, tcp_off) -> np.ndarray:
        """target - reached (mm, base frame) of the tool point under the DH model."""
        Tm = dh_fk(q_deg)
        return np.asarray(pos) - (Tm[:3, 3] + Tm[:3, :3] @ tcp_off)

    def _retime(self, q, dt0, dt, ts):
        """Re-time a cuRobo plan (path unchanged) and resample at ``dt`` with a cubic spline.

        Stretch factor k scales time: v/k, a/k^2. k is chosen so the peak joint speed is
        nominal/ts (ts=1 -> --max-joint-vel-deg), then raised until the peak speed <= ceiling
        and the peak acceleration <= the accel cap (2% margin). Returns (q, k, bound, T).
        """
        from scipy.interpolate import CubicSpline
        if len(q) < 2 or dt0 * (len(q) - 1) <= 1e-9:
            return np.vstack([q[0], q[-1]]), 1.0, "none", dt
        # cuRobo pads the plan with stationary frames at both ends: trim them (keep one each)
        moving = np.nonzero(np.abs(q - q[0]).max(axis=1) > 1e-3)[0]
        still_end = np.nonzero(np.abs(q - q[-1]).max(axis=1) > 1e-3)[0]
        if len(moving) and len(still_end):
            q = q[max(moving[0] - 1, 0): still_end[-1] + 2]
        v = np.gradient(np.radians(q), dt0, axis=0)
        a = np.gradient(v, dt0, axis=0)
        vp, ap = float(np.abs(v).max()), float(np.abs(a[1:-1]).max()) if len(a) > 2 else 0.0
        cand = {"time_scale": vp * ts / self.nominal, "ceiling_vel": vp / self.ceiling * 1.02,
                "accel_cap": math.sqrt(ap / self.accel) * 1.02 if ap > 0 else 0.0}
        bound = max(cand, key=cand.get)
        k = cand[bound]
        T = dt0 * (len(q) - 1) * k
        n = max(2, int(math.ceil(T / dt - 1e-9)) + 1)
        t = np.minimum(np.arange(n) * dt, T)
        t[-1] = T
        qs = CubicSpline(np.arange(len(q)) * dt0 * k, q, axis=0)(t)
        qs[0], qs[-1] = q[0], q[-1]
        return qs, k, bound, T

    def _finish(self, q, dt, T, t0, tcp_off) -> dict:
        tcp = []
        for qi in q:
            Tm = dh_fk(qi)
            tcp.append(Tm[:3, 3] + Tm[:3, :3] @ tcp_off)
        tcp = np.array(tcp)
        # the whole polyline is cheap; thin it to <= 200 points for the response
        sel = np.unique(np.r_[np.arange(0, len(tcp), max(1, len(tcp) // 200)), len(tcp) - 1])
        qd = np.gradient(q, dt, axis=0) if len(q) > 1 else np.zeros_like(q)
        qdd = np.gradient(qd, dt, axis=0) if len(q) > 2 else np.zeros_like(q)
        return {"ok": True, "status": "ok", "dt": dt, "q": np.round(q, 3).tolist(), "duration_s": round(T, 3),
                "plan_ms": round((time.perf_counter() - t0) * 1000, 1),
                "tcp_path_mm": np.round(tcp[sel], 2).tolist(),
                "peak_joint_vel_deg_s": round(float(np.abs(qd).max()), 1),
                "peak_joint_acc_deg_s2": round(float(np.abs(qdd[1:-1]).max()) if len(qdd) > 2 else 0.0, 1)}

    def _fail(self, stage, reason, t0, detail=None) -> dict:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        self.set_status(f"Plan failed: {stage}", reason, f"{ms:.0f} ms")
        out = {"ok": False, "status": stage, "reason": reason, "plan_ms": ms}
        if detail:
            out["detail"] = detail
        return out

    def health(self) -> dict:
        import torch
        free, total = torch.cuda.mem_get_info()
        return {"ok": True, "robot": self.robot.name, "variant": "camera" if "camera" in self.robot.name else "gripper",
                "warm": self.warm, "gpu": {"name": torch.cuda.get_device_name(0), "used_mib": (total - free) >> 20,
                                           "total_mib": total >> 20},
                "limits": {"peak_joint_vel_deg_s_at_time_scale_1": round(math.degrees(self.nominal), 1),
                           "ceiling_joint_vel_deg_s": round(math.degrees(self.ceiling), 1),
                           "max_accel_deg_s2": round(math.degrees(self.accel), 1),
                           "joint_limit_margin_deg": mpl.JOINT_LIMIT_MARGIN_DEG},
                "frame_check": self.frame_check, "last_status": self.last_status,
                "world": [o["name"] for o in self.world]}


# ---- HTTP -------------------------------------------------------------------------------------
def make_handler(svc: Service):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep stdout quiet (/state is 5-10 Hz)
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in (("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                         ("Access-Control-Allow-Headers", "Content-Type"), ("Content-Length", "0")):
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            if self.path.split("?")[0] == "/health":
                self._send(200, svc.health())
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(req, dict):
                    raise BadRequest("body must be a JSON object")
                path = self.path.split("?")[0]
                if path == "/plan":
                    out = svc.plan(req)
                elif path == "/check":
                    out = svc.check(req)
                elif path == "/spheres":
                    out = svc.spheres(req)
                elif path == "/state":
                    svc.set_state(vec(req.get("q"), 6, "q"))
                    out = {"ok": True}
                elif path == "/world":
                    obs = svc._parse_obstacles(req.get("obstacles", []))
                    with svc.plan_lock:
                        svc.set_world(obs)
                    out = {"ok": True, "n": len(obs)}
                elif path == "/show_plan":
                    q = np.asarray(req.get("q"), float)
                    if q.ndim != 2 or q.shape[1] != 6 or len(q) < 1 or not np.all(np.isfinite(q)):
                        raise BadRequest("q must be an [N][6] list of degrees")
                    svc.show_plan(q, float(req.get("dt", 0.05)))
                    out = {"ok": True, "n": len(q)}
                else:
                    return self._send(404, {"ok": False, "error": "not found"})
                self._send(200, out)
            except (BadRequest, json.JSONDecodeError) as e:
                self._send(400, {"ok": False, "status": "bad request", "reason": str(e)})
            except Exception as e:  # keep the service up; the caller sees the type
                import traceback
                traceback.print_exc()
                self._send(500, {"ok": False, "status": "error", "reason": f"{type(e).__name__}: {e}"})
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--viser-port", type=int, default=8080)
    ap.add_argument("--robot", default=str(ROBOT_YML), help="cuRobo robot yml (default: gripper model)")
    ap.add_argument("--max-joint-vel-deg", type=float, default=30.0,
                    help="peak joint speed at time_scale 1 (default 30 deg/s = MycobotPlanner's cap; "
                         "note its plans actually peaked at ~15 deg/s, so time_scale 0.5 reproduces that timing)")
    ap.add_argument("--ceiling-joint-vel-deg", type=float, default=90.0,
                    help="fastest joint speed a time_scale < 1 may reach (planner is built here; URDF spec 120)")
    ap.add_argument("--max-accel-deg", type=float, default=None,
                    help="joint accel cap deg/s^2 (default: MycobotPlanner MAX_ACCEL = 4 rad/s^2 = 229)")
    args = ap.parse_args()
    setup_logger("error")
    svc = Service(args)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(svc))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    svc.warmup()
    print(f"plan_server ready: api http://127.0.0.1:{args.port}  viser http://127.0.0.1:{svc.server.get_port()}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping plan_server")
        httpd.shutdown()


if __name__ == "__main__":
    main()
