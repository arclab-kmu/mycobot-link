"""``mycobot-brain console``: one page with the command box, both cameras (RGB + depth), and cuRobo.

Data flow for one command::

    text + TOP/SIDE frames --> Spark VLM --> action chunk (point/move/home/grip/wait)
        --> workspace (pixel + aligned depth -> world mm, safety box)
        --> [cuRobo plan for the reach]  --> QP chunk executor @ 20 Hz --> arm server

Bound to 127.0.0.1 only: the page can move an arm, and it has no login. From
another machine use ``ssh -L 8095:127.0.0.1:8095``. Nothing moves unless the
"실행" box is ticked *and* the arm server was started with ``--live``; without
it the console plans, shows the targets on the top image, and stops there.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
import traceback
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

import numpy as np

from mycobot_brain.console import vlm as vlm_mod
from mycobot_brain.console.feeds import Feed, snapshot
from mycobot_brain.console.workspace import Workspace
from mycobot_brain.link import ArmLink, LinkError
from mycobot_brain.planner import PlannerError

log = logging.getLogger("console")

MAX_ROUNDS = 6


def _quat_wxyz(R: np.ndarray) -> list[float]:
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-3:
        return [float(w), float((R[2, 1] - R[1, 2]) / (4 * w)), float((R[0, 2] - R[2, 0]) / (4 * w)),
                float((R[1, 0] - R[0, 1]) / (4 * w))]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
    v = [0.0, 0.0, 0.0]
    v[i] = s / 4
    v[j] = (R[j, i] + R[i, j]) / s
    v[k] = (R[k, i] + R[i, k]) / s
    return [float((R[k, j] - R[j, k]) / s)] + [float(x) for x in v]


#: Approach tilts below horizontal (deg) tried in this order, then yaw offsets.
#: The .77 arm stands 197 mm below the board, so a vertical tool over the board
#: puts the flange near 420 mm, past its reach (cuRobo IK fails); grasps on
#: 2026-11-22 used 0-60 deg. The first tilt cuRobo can solve wins.
APPROACH_TILTS = (45.0, 30.0, 60.0, 15.0, 75.0, 0.0, 90.0)
APPROACH_YAWS = (0.0, 30.0, -30.0)


def approach_frames(p_base: np.ndarray) -> list[np.ndarray]:
    """Candidate tool orientations (base frame) for reaching ``p_base``: tool z points
    away from the arm's base, tilted down; flange y stays horizontal."""
    r = np.array([p_base[0], p_base[1], 0.0])
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([1.0, 0.0, 0.0])
    out = []
    for yaw in APPROACH_YAWS:
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        h = np.array([c * r[0] - s_ * r[1], s_ * r[0] + c * r[1], 0.0])
        for tilt in APPROACH_TILTS:
            t = np.radians(tilt)
            z = np.array([np.cos(t) * h[0], np.cos(t) * h[1], -np.sin(t)])
            y = np.array([-h[1], h[0], 0.0])
            out.append(np.column_stack([np.cross(y, z), y, z]))
    return out


class Console:
    def __init__(self, ws: Workspace, vlm_url: str, vlm_model: str, rate_hz: float = 20.0, speed_param: int = 100,
                 bind: str = "127.0.0.1") -> None:
        self.ws = ws
        self.vlm_url, self.vlm_model = vlm_url, vlm_model
        self.feeds: dict[str, Feed] = {}
        for cam in ws.cameras.values():
            if cam.depth_set:  # e.g. the D405 needs 70-600 mm, not the viewer's 200-1500 default
                try:
                    snapshot(cam.depth_set, timeout=2.0)
                except OSError:
                    pass
            self.feeds[f"{cam.name}-color"] = Feed(f"{cam.name}-color", cam.color_mjpg).start()
            self.feeds[f"{cam.name}-depth"] = Feed(f"{cam.name}-depth", cam.depth_mjpg).start()
        self.links = {name: ArmLink(host=bind, port=arm.port).start() for name, arm in ws.arms.items()}
        self.start_pose: dict[str, list[float]] = {}
        self.log: collections.deque[str] = collections.deque(maxlen=80)
        self.task: dict[str, Any] | None = None
        self.vlm_last_s = 0.0
        self.vlm_ok = False
        self.curobo_ok = False
        self._fps: dict[str, tuple[int, float, float]] = {}
        self._task_thread: threading.Thread | None = None
        self._extra: list[dict] = []
        self._touch_xy: list[np.ndarray] = []
        self.runner = None
        self.kins: dict[str, Any] = {}
        try:
            from mycobot_brain.console.motion import MotionRunner

            self.runner = MotionRunner(rate_hz=rate_hz, speed_param=speed_param)
            for name, arm in ws.arms.items():
                self.kins[name] = MotionRunner.kinematics(arm.tool)
        except ImportError as exc:
            self.say(f"! motion disabled (mycobot_control.qp missing: {exc})")
        threading.Thread(target=self._health_loop, name="health", daemon=True).start()

    # -- bookkeeping -------------------------------------------------------

    def say(self, line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.append(f"{stamp} {line}")
        log.info(line)

    def _health_loop(self) -> None:
        while True:
            try:
                with urllib.request.urlopen(self.ws.curobo_api.rstrip("/") + "/health", timeout=1.0) as r:
                    self.curobo_ok = bool(json.load(r).get("ok"))
            except Exception:  # noqa: BLE001
                self.curobo_ok = False
            try:
                with urllib.request.urlopen(self.vlm_url.rstrip("/") + "/models", timeout=2.0) as r:
                    ids = [m.get("id") for m in json.load(r).get("data", [])]
                    self.vlm_ok = self.vlm_model in ids
            except Exception:  # noqa: BLE001
                self.vlm_ok = False
            # live arm pose to the cuRobo view (cheap; the view is optional)
            if self.curobo_ok and self.task is not None:
                name = self.task.get("arm")
                st = self.links[name].latest_state() if name in self.links else None
                if st and st.get("angles_deg"):
                    from mycobot_brain.console.motion import curobo_post

                    curobo_post(self.ws.curobo_api, "/state", {"q": st["angles_deg"]})
            time.sleep(1.0)

    def _feed_state(self) -> dict[str, Any]:
        out = {}
        now = time.time()
        for key, f in self.feeds.items():
            n0, t0, fps = self._fps.get(key, (f.frames, now, 0.0))
            if now - t0 >= 1.0:
                fps = (f.frames - n0) / (now - t0)
                n0, t0 = f.frames, now
            self._fps[key] = (n0, t0, fps)
            age = f.age()
            out[key] = {"ok": age is not None and age < 2.0, "fps": fps, "error": f.error}
        return out

    def state(self) -> dict[str, Any]:
        arms = {}
        for name, link in self.links.items():
            st = link.latest_state()
            q = st.get("angles_deg") if st else None
            tcp = None
            if q is not None and name in self.kins:
                tcp = self.ws.arms[name].to_world(self.kins[name].fk(np.array(q))[:3, 3]).round(1).tolist()
            arms[name] = {
                "connected": link.connected and st is not None,
                "live": bool(st and st.get("live")),
                "q": q,
                "tcp": tcp,
                "gate": (("open" if st.get("gate_open") else "closed") + (f" ({st.get('gate_reason')})" if st.get("gate_reason") else "")) if st else None,
                "speed_dps": self.runner.measured_speed_dps if self.runner and self.task and self.task.get("arm") == name else 0.0,
            }
        task = None
        if self.task is not None:
            task = {k: v for k, v in self.task.items() if k not in ("chunk",)}
            if self.runner is not None and task.get("running") and task.get("phase") == "move":
                task["progress"] = self.runner.progress
        return {
            "arms": arms,
            "feeds": self._feed_state(),
            "vlm": {"ok": self.vlm_ok, "model": self.vlm_model, "last_s": self.vlm_last_s},
            "curobo": {"ok": self.curobo_ok},
            "task": task,
            "log": list(self.log)[-40:],
        }

    # -- commands ----------------------------------------------------------

    def command(self, body: dict[str, Any]) -> dict[str, Any]:
        text = str(body.get("text", "")).strip()
        arm = str(body.get("arm", ""))
        if not text:
            return {"ok": False, "error": "empty command"}
        if arm not in self.ws.arms:
            return {"ok": False, "error": f"unknown arm {arm!r}"}
        if self._task_thread is not None and self._task_thread.is_alive():
            return {"ok": False, "error": "a command is still running; press 정지 first"}
        opts = {
            "execute": bool(body.get("execute")),
            "loop": bool(body.get("loop")),
            "curobo": bool(body.get("curobo", True)),
            "speed_dps": float(min(max(float(body.get("speed_dps", 60)), 10.0), 90.0)),
        }
        self.task = {"text": text, "arm": arm, "running": True, "phase": "look", "status": "카메라 프레임을 모으는 중",
                     "say": "", "seen": "", "steps": [], "step": -1, "progress": 0.0, "marks": {}, **opts}
        self._task_thread = threading.Thread(target=self._run_task, args=(text, arm, opts), name="task", daemon=True)
        self._task_thread.start()
        return {"ok": True}

    def stop(self) -> dict[str, Any]:
        if self.runner is not None:
            self.runner.stop()
        if self.task is not None:
            self.task["stop"] = True
        for link in self.links.values():
            if link.connected:
                try:
                    link.stop()
                except LinkError:
                    pass
        self.say("정지")
        return {"ok": True}

    # -- one task ----------------------------------------------------------

    def _images(self) -> tuple[dict[str, bytes], bytes | None]:
        images: dict[str, bytes] = {}
        for name in [self.ws.vlm_camera] + [c for c in self.ws.cameras if c != self.ws.vlm_camera]:
            cam = self.ws.cameras[name]
            feed = self.feeds[f"{name}-color"]
            jpeg, _ = feed.wait_newer(time.time() - 0.05, timeout=1.5)
            if jpeg is None:
                try:
                    jpeg = snapshot(cam.color_jpg)
                except OSError:
                    jpeg = None
            if jpeg is not None:
                images[name] = jpeg
        depth_png = None
        top = self.ws.cameras.get(self.ws.vlm_camera)
        if top is not None and top.depth_png:
            try:
                depth_png = snapshot(top.depth_png)
            except OSError as exc:
                self.say(f"! depth {exc}")
        return images, depth_png

    def _state_text(self, arm_name: str) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        st = self.links[arm_name].latest_state()
        if not st or st.get("angles_deg") is None:
            return "not connected (plan only)", None, None
        q = np.array(st["angles_deg"], float)
        if arm_name in self.kins:
            p = self.ws.arms[arm_name].to_world(self.kins[arm_name].fk(q)[:3, 3])
            return f"gripper tip at world ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm; table top z = {self.ws.table_z:.0f} mm", q, p
        return "joints " + " ".join(f"{x:.0f}" for x in q), q, None

    def _align(self, depth_png: bytes | None) -> np.ndarray | None:
        cam = self.ws.cameras.get(self.ws.vlm_camera)
        if depth_png is None or cam is None or not cam.calibrated or not cam.depth_K:
            return None
        try:
            return cam.aligned_depth(depth_png)
        except Exception as exc:  # noqa: BLE001
            self.say(f"! depth align failed: {exc}")
            return None

    def _chain_world(self, arm_name: str, q: np.ndarray) -> list[np.ndarray]:
        """World points along the arm (DH frame origins + tool), to keep it out of its own obstacle map."""
        from mycobot_control.qp import DH_PARAMS

        arm = self.ws.arms[arm_name]
        T = np.eye(4)
        pts = [arm.to_world(np.zeros(3))]
        for qi, (off, d, a, al) in zip(np.radians(q), DH_PARAMS):
            th = qi + off
            ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
            T = T @ np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]])
            pts.append(arm.to_world(T[:3, 3]))
        pts.append(arm.to_world((T @ np.r_[arm.tool, 1.0])[:3]))
        return pts

    def _resolve(self, chunk: dict[str, Any], arm_name: str, depth: np.ndarray | None, tip_world: np.ndarray | None,
                 start_q: list[float] | None):
        """Steps -> motion segments (arm base frame) + overlay marks on the top image."""
        from mycobot_brain.console.motion import Segment

        arm = self.ws.arms[arm_name]
        cam = self.ws.cameras.get(self.ws.vlm_camera)
        self._touch_xy: list[np.ndarray] = []
        cursor = None if tip_world is None else np.array(tip_world, float)
        segs: list[Segment] = []
        marks: list[dict[str, Any]] = []
        texts: list[str] = []
        pending: list[np.ndarray] = []
        R_end: list[np.ndarray | None] = [None]

        def down_at(p_world: np.ndarray) -> np.ndarray:
            return approach_frames(arm.to_base(p_world))[0]

        def flush() -> None:
            if pending:
                segs.append(Segment("cart", label="reach", points_base=[arm.to_base(p) for p in pending], R_end=R_end[0]))
                pending.clear()
                R_end[0] = None

        for s in chunk["steps"]:
            op = s["op"]
            if op in ("point", "move"):
                if op == "point":
                    if cam is None or not cam.calibrated:
                        texts.append(f"point {s['target']}: top camera not calibrated, skipped")
                        continue
                    u, v = s["x"] / 1000 * cam.size[0], s["y"] / 1000 * cam.size[1]
                    p_obj, frac = cam.pixel_to_world(u, v, depth)
                    how = f"depth {frac:.0%}"
                    if p_obj is None:
                        c, d = cam.pixel_ray(u, v)
                        p_obj = c + d * (self.ws.table_z - c[2]) / d[2]
                        how = "table plane"
                    p_obj[2] = max(p_obj[2], self.ws.table_z)
                    target = p_obj + np.array([0.0, 0.0, s["height_mm"]])
                    if s["height_mm"] < 40:  # touching it: its own depth blob must not block the goal
                        self._touch_xy.append(p_obj[:2].copy())
                    R_end[0] = down_at(target)
                    marks.append({"x": s["x"], "y": s["y"], "label": s["target"][:24]})
                else:
                    if cursor is None:
                        texts.append("move: arm pose unknown, skipped")
                        continue
                    target = cursor + np.array([s["dx_mm"], s["dy_mm"], s["dz_mm"]])
                    how = "nudge"
                target, clipped = self.ws.box.clamp(target)
                horiz = np.linalg.norm(arm.to_base(target)[:2])
                note = (" (clamped to box)" if clipped else "") + (" (beyond reach!)" if horiz > self.ws.box.reach_mm else "")
                texts.append(f"{op} {s['target'] or ''} -> ({target[0]:.0f}, {target[1]:.0f}, {target[2]:.0f}) [{how}]{note}".replace("  ", " "))
                pending.append(target)
                cursor = target
            else:
                flush()
                if op == "home":
                    q_home = arm.home or start_q
                    if q_home is None:
                        texts.append("home: no home pose known, skipped")
                        continue
                    segs.append(Segment("joints", label="home", q=np.array([q_home], float)))
                    texts.append("home " + " ".join(f"{x:.0f}" for x in q_home))
                    cursor = None if tip_world is None else (
                        arm.to_world(self.kins[arm_name].fk(np.array(q_home))[:3, 3]) if arm_name in self.kins else None)
                elif op == "grip":
                    segs.append(Segment("grip", label="grip", value=s["grip"]))
                    texts.append(f"grip {s['grip']}")
                elif op == "wait":
                    segs.append(Segment("wait", label="wait", seconds=s["seconds"]))
                    texts.append(f"wait {s['seconds']:.1f} s")
        flush()
        return segs, marks, texts

    def _curobo_segments(self, segs, arm_name: str, q0: np.ndarray, speed_dps: float):
        """Replace each reach's final target by a cuRobo collision-free joint path."""
        from mycobot_brain.console.motion import Segment, curobo_plan

        arm = self.ws.arms[arm_name]
        kin = self.kins[arm_name]
        q = np.array(q0, float)
        out = []
        for seg in segs:
            if seg.kind == "cart":
                T = kin.fk(q)
                p_end = seg.points_base[-1]
                cands = [T[:3, :3]] if seg.R_end is None else approach_frames(p_end)
                t0 = time.time()
                r: dict[str, Any] = {}
                for k, R in enumerate(cands):
                    flange = p_end - R @ arm.tool
                    goal = {"pose": {"pos_mm": flange.round(2).tolist(), "quat_wxyz": _quat_wxyz(R)}}
                    r = curobo_plan(self.ws.curobo_api, q.round(3).tolist(), goal, self.ws.obstacles_for(arm, self._extra),
                                    self.runner.dt, time_scale=30.0 / max(speed_dps, 10.0))
                    if r.get("ok"):
                        seg.R_end = R
                        break
                if not r.get("ok"):
                    self.say(f"! cuRobo: {r.get('status')} for all {len(cands)} approach angles — QP straight line instead")
                    out.append(seg)
                    continue
                Q = np.array(r["q"], float)
                tilt = float(np.degrees(np.arcsin(np.clip(-seg.R_end[2, 2], -1, 1)))) if seg.R_end is not None else 0.0
                self.say(f"cuRobo {len(Q)} pts, {r.get('duration_s', 0):.1f} s, approach tilt {tilt:.0f} deg "
                         f"(try {k + 1}/{len(cands)}), {(time.time() - t0) * 1e3:.0f} ms")
                out.append(Segment("joints", label="reach (cuRobo)", q=Q, q_dt=float(r.get("dt", self.runner.dt))))
                q = Q[-1]
            elif seg.kind == "joints":
                Qh = np.atleast_2d(seg.q)
                r = curobo_plan(self.ws.curobo_api, q.round(3).tolist(), {"joints": Qh[-1].round(3).tolist()},
                                self.ws.obstacles_for(arm, self._extra), self.runner.dt, time_scale=30.0 / max(speed_dps, 10.0))
                if r.get("ok"):
                    Q = np.array(r["q"], float)
                    out.append(Segment("joints", label="home (cuRobo)", q=Q, q_dt=float(r.get("dt", self.runner.dt))))
                    q = Q[-1]
                else:
                    self.say(f"! cuRobo home: {r.get('status')}")
                    out.append(seg)
                    q = Qh[-1]
            else:
                out.append(seg)
        return out

    def _run_task(self, text: str, arm_name: str, opts: dict[str, Any]) -> None:
        t = self.task
        history: list[str] = []
        try:
            link = self.links[arm_name]
            st0 = link.latest_state()
            start_q = st0.get("angles_deg") if st0 else None
            if start_q is not None and arm_name not in self.start_pose:
                self.start_pose[arm_name] = list(start_q)
            rounds = MAX_ROUNDS if opts["loop"] else 1
            for rnd in range(rounds):
                if t.get("stop"):
                    break
                t.update(phase="look", status="카메라 프레임을 모으는 중")
                images, depth_png = self._images()
                if self.ws.vlm_camera not in images:
                    raise RuntimeError("top camera has no frame (is the viewer tunnel up?)")
                state_text, q, tip = self._state_text(arm_name)
                t.update(phase="think", status=f"VLM에 묻는 중 ({self.vlm_model})")
                t0 = time.time()
                chunk = vlm_mod.propose_chunk(text, images, state_text, self.ws.world_axes_text(),
                                              base_url=self.vlm_url, model=self.vlm_model, history=history)
                self.vlm_last_s = time.time() - t0
                self.say(f"VLM {self.vlm_last_s:.1f} s: {chunk['say']}")
                depth = self._align(depth_png)
                segs, marks, texts = self._resolve(chunk, arm_name, depth, tip, self.start_pose.get(arm_name))
                self._extra = []
                cam = self.ws.cameras.get(self.ws.vlm_camera)
                if depth is not None and q is not None:
                    from mycobot_brain.console.workspace import heightmap_boxes

                    avoid = self._chain_world(arm_name, q)
                    radii = [100.0] * (len(avoid) - 2) + [55.0]  # fallback without cuRobo: links wide, gripper narrow
                    spheres = None
                    if self.curobo_ok:
                        try:
                            from mycobot_brain.console.motion import curobo_spheres

                            sb = np.array(curobo_spheres(self.ws.curobo_api, q), float)
                            arm_ = self.ws.arms[arm_name]
                            spheres = np.column_stack([np.array([arm_.to_world(c) for c in sb[:, :3]]), sb[:, 3]])
                        except Exception as exc:  # noqa: BLE001
                            self.say(f"! spheres: {exc}")
                    boxes = heightmap_boxes(cam, depth, self.ws.table_z, self.ws.box, avoid, avoid_r=radii,
                                            reach_from=self.ws.arms[arm_name].t, spheres=spheres, sphere_margin=35.0)
                    # A box the arm already touches at its current pose is the arm itself (its
                    # wrist camera and cables are not in the sphere model), not an obstacle.
                    if self.curobo_ok and boxes:
                        from mycobot_brain.console.motion import curobo_check

                        for _ in range(4):
                            chk = curobo_check(self.ws.curobo_api, np.array([q]),
                                               self.ws.obstacles_for(self.ws.arms[arm_name], boxes), stride=1)
                            hit = {k.split("/")[0] for k in chk.get("detail", {}).get("obstacles", {})}
                            mine = {b["name"] for b in boxes} & hit
                            if chk.get("ok") or not mine:
                                break
                            boxes = [b for b in boxes if b["name"] not in mine]
                    for xy in self._touch_xy:
                        boxes = [b for b in boxes if not (abs(b["center_mm"][0] - xy[0]) < b["dims_mm"][0] / 2
                                                         and abs(b["center_mm"][1] - xy[1]) < b["dims_mm"][1] / 2)]
                    self._extra = boxes
                    self.say(f"depth obstacles: {len(boxes)} boxes " + ", ".join(
                        f"({b['center_mm'][0]:.0f},{b['center_mm'][1]:.0f}) h{b['dims_mm'][2]:.0f}" for b in boxes[:6]))
                t.update(say=chunk["say"], seen=chunk["seen"], steps=texts, marks={self.ws.vlm_camera: marks}, chunk=chunk)
                for line in texts:
                    self.say("  " + line)
                if chunk["done"] and not segs:
                    t.update(status="완료")
                    break
                if self.runner is None or q is None:
                    t.update(status="계획만 (팔 연결 없음)")
                    break
                z_min = float(self.ws.box.lo[2] - self.ws.arms[arm_name].t[2])
                if opts["curobo"] and self.curobo_ok:
                    t.update(phase="plan", status="cuRobo로 경로 계획 중")
                    segs = self._curobo_segments(segs, arm_name, q, opts["speed_dps"])
                # Every chunk is checked against the links' collision spheres before it runs.
                paths = self.runner.preview(self.kins[arm_name], segs, q, opts["speed_dps"], z_min)
                moving = [Q for Q in paths if Q is not None and len(Q)]
                if moving:
                    from mycobot_brain.console.motion import curobo_check, curobo_post

                    full = np.vstack(moving)
                    if self.curobo_ok:
                        chk = curobo_check(self.ws.curobo_api, full, self.ws.obstacles_for(self.ws.arms[arm_name], self._extra))
                        curobo_post(self.ws.curobo_api, "/show_plan", {"q": np.round(full, 3).tolist(), "dt": self.runner.dt})
                        if not chk.get("ok"):
                            hit = chk.get("detail", {})
                            raise RuntimeError(f"충돌 예상 (경로 {chk.get('index')}/{chk.get('n')}번째): {hit}")
                        self.say(f"cuRobo check clear ({chk.get('checked')} samples, {chk.get('check_ms')} ms)")
                    elif opts["execute"]:
                        raise RuntimeError("cuRobo 서버가 꺼져 있어 링크 충돌을 검사할 수 없어요; 실행하지 않아요")
                    t.update(duration_s=round(len(full) * self.runner.dt, 1))
                if not opts["execute"]:
                    t.update(status="계획만 (실행 꺼짐)")
                    break
                t.update(phase="move", status="움직이는 중")
                stats = self.runner.run(link, self.kins[arm_name], segs, opts["speed_dps"],
                                        on_step=lambda i: t.update(step=i), arm=arm_name, z_min_mm=z_min)
                self.say(f"done {stats['t']:.1f} s, {stats['ticks']} ticks, refused {stats['fails']}, "
                         f"max track err {stats['max_err_deg']:.1f} deg, end err {stats.get('final_err_deg', 0):.1f} deg, bias {stats.get('bias')}")
                history.append(chunk["say"])
                if chunk["done"]:
                    break
            t.update(status=t.get("status") if t.get("status") not in ("움직이는 중",) else "완료")
        except PlannerError as exc:
            self.say(f"! VLM: {exc}")
            t.update(status=f"VLM 오류: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface anything on the page
            self.say(f"! {type(exc).__name__}: {exc}")
            log.debug(traceback.format_exc())
            t.update(status=f"멈춤: {exc}")
        finally:
            t.update(running=False, phase="idle", step=-1)


def page(ws: Workspace) -> bytes:
    cfg = {
        "cameras": [{"name": c.name, "label": c.label} for c in ws.cameras.values()],
        "arms": [{"name": a.name, "label": a.label} for a in ws.arms.values()],
        "curobo_view": ws.curobo_view,
    }
    html = resources.files("mycobot_brain.console").joinpath("page.html").read_text(encoding="utf-8")
    return html.replace("__CONFIG__", json.dumps(cfg, ensure_ascii=False)).encode()


def make_handler(console: Console):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:  # quiet
            pass

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = page(console.ws)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self._json(console.state())
            elif path.startswith("/cam/"):
                parts = path.split("/")  # ['', 'cam', name, 'color.mjpg']
                if len(parts) != 4:
                    return self._json({"error": "bad camera path"}, 404)
                kind, _, ext = parts[3].partition(".")
                feed = console.feeds.get(f"{parts[2]}-{kind}")
                if feed is None:
                    return self._json({"error": "no such camera"}, 404)
                if ext == "jpg":
                    if feed.jpeg is None:
                        return self._json({"error": "no frame yet"}, 503)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(feed.jpeg)))
                    self.end_headers()
                    self.wfile.write(feed.jpeg)
                else:
                    self._mjpeg(feed)
            else:
                self._json({"error": "not found"}, 404)

        def _mjpeg(self, feed: Feed) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            stamp = 0.0
            try:
                while True:
                    jpeg, stamp_new = feed.wait_newer(stamp, timeout=2.0)
                    if jpeg is None or stamp_new == stamp:
                        continue
                    stamp = stamp_new
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
                                     + b"\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"ok": False, "error": "bad json"}, 400)
            if self.path == "/api/command":
                self._json(console.command(body))
            elif self.path == "/api/stop":
                self._json(console.stop())
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def serve(console: Console, host: str = "127.0.0.1", port: int = 8095) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(console))
    httpd.daemon_threads = True
    log.info("console on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    finally:
        console.stop()
        httpd.server_close()
