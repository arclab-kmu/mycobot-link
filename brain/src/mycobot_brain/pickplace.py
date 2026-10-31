"""Pick something up and put it somewhere, from one sentence and a wrist camera.

The split of work: the VLM says where things are in an image and whether a
step worked; this code does the geometry (pixel → camera ray from K and FK of
the *measured* angles → a plane of known height → base xyz → IK) and walks
the arm there in waypoints of at most ``max_step`` degrees per joint, waiting
for the measured angles to arrive before each next one. The arm server still
checks every waypoint (limits, 20 deg from measured, speed, gate).

Sequence: look pose → locate both → camera over the block, look again and
refine → pre-grasp above it → descend → close → lift → "is it held?" →
camera over the box, refine → above it → lower → open → retreat → look pose →
"is it in the box?" → done. A grasp that the VLM does not see held is retried
from the look pose up to ``attempts`` times.

The gate stays open through VLM waits of several seconds because
:class:`ArmLink`'s heartbeat thread pings every 100 ms on its own; nothing
here has to remember to feed the watchdog.

Every step is logged to ``events.jsonl`` (``wall`` unix seconds on each line)
with the images sent to the VLM saved beside it, for the run video.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mycobot_brain import grounding, kinematics
from mycobot_brain.calib import Calib
from mycobot_brain.camera import CameraLink, Frame
from mycobot_brain.kinematics import IKError, Vec
from mycobot_brain.link import ArmLink, LinkError
from mycobot_brain.planner import PlannerError

#: request body → the model's reply text (``planner.chat`` with its URL bound).
VLM = Callable[[Dict[str, Any]], str]
Emit = Callable[[Dict[str, Any]], None]

DOWN: Vec = (0.0, 0.0, -1.0)
TILTS_DEG = (0, 5, 10, 15)


class PickPlaceError(RuntimeError):
    """The sequence cannot go on: nothing to aim at, no IK, or the arm did not arrive."""


@dataclass
class Settings:
    execute: bool = False
    speed: int = 30
    slow_speed: int = 15  # descend, lower, lift
    max_step: float = 8.0  # deg per joint per waypoint; the server refuses > 20 from measured
    pass_tol: float = 3.0  # deg from an intermediate waypoint before sending the next
    arrive_tol: float = 1.0  # deg from the last waypoint of a move
    stall_tol: float = 3.0  # accept a final waypoint this close if the arm stops short of arrive_tol
    arrive_timeout: float = 15.0
    settle: float = 0.5  # s still before a look
    frame_timeout: float = 5.0
    pregrasp_m: float = 0.06  # TCP above the block centre before descending
    close_look_m: float = 0.15  # camera above the target's plane for the close-up look
    carry_clear_m: float = 0.05  # block bottom above the box rim while carrying
    grasp_depth_m: float = 0.0  # TCP below the block centre at the grasp
    finger_clear_m: float = 0.004  # finger pads above the table at the grasp, whatever grasp_depth_m says
    place_clear_m: float = 0.01  # block bottom above the rim at release (negative: inside the box)
    line_step_m: float = 0.01  # straight-line segments for descend / lift / lower / retreat
    gripper_speed: int = 50
    gripper_wait: float = 1.5  # s; the protocol has no gripper reading to wait on
    attempts: int = 2
    coords: str = "norm1000"
    shape: str = "bbox"
    refine: bool = True


class RunLog:
    """``events.jsonl`` + the images the VLM saw, in one directory."""

    def __init__(self, directory: Path, emit: Optional[Emit] = None) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self.emit = emit
        self.t0 = time.monotonic()
        self._images = itertools.count(1)

    def event(self, kind: str, **fields: Any) -> Dict[str, Any]:
        line = {"event": kind, "wall": round(time.time(), 3), "t": round(time.monotonic() - self.t0, 3), **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        if self.emit is not None:
            self.emit(line)
        return line

    def image(self, jpeg: bytes, purpose: str) -> str:
        name = f"vlm-{next(self._images):02d}-{purpose}.jpg"
        (self.dir / name).write_bytes(jpeg)
        return name


def _r(xs: Sequence[float], nd: int = 2) -> List[float]:
    return [round(float(x), nd) for x in xs]


class PickPlace:
    def __init__(self, link: ArmLink, camera: CameraLink, calib: Calib, vlm: VLM, log: RunLog,
                 settings: Optional[Settings] = None, model: Optional[str] = None) -> None:
        self.link, self.camera, self.calib, self.vlm, self.log = link, camera, calib, vlm, log
        self.s = settings or Settings()
        self.model = model
        self.enabled = False
        self._vlm_n = 0
        self._seq: Optional[int] = None

    # -- the sequence --------------------------------------------------------

    def run(self, instruction: str) -> Dict[str, Any]:
        s, c = self.s, self.calib
        self.log.event("command", instruction=instruction, calib=c.source, execute=s.execute,
                       settings=dataclasses.asdict(s))
        try:
            result = self._run(instruction)
        except BaseException as exc:
            if self.enabled:
                try:
                    self.link.stop()  # mid-move failure ends like every other: stop() and a closed gate
                except LinkError:
                    pass
                self.enabled = False
            self.log.event("result", ok=False, why=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if self.enabled:
                self.link.disable()
                self.enabled = False
        self.log.event("result", **result)
        return result

    def _run(self, instruction: str) -> Dict[str, Any]:
        s, c = self.s, self.calib
        state = self.link.wait_state()
        if state["angles_deg"] is None:
            raise LinkError("the arm server has no joint reading to start from")
        if s.execute:
            self.link.enable()
            self.enabled = True
            self._phase("look")
            self._goto(c.look_pose_deg, "look", floor=None)
            self._gripper(c.gripper_open, "open")

        first, frame0, q0 = self._locate(instruction, ("pick", "place"), "look", with_next=True)
        pick_label = grounding.object_label(first["pick"], "object")
        place_label = grounding.object_label(first["place"], "target")
        if first["next"] == "done":
            return {"ok": True, "why": f"the VLM sees the {pick_label} already in the {place_label}"}
        if first["next"] == "cannot" or not (first["pick"]["visible"] and first["place"]["visible"]):
            return {"ok": False, "why": f"the VLM cannot do it: {first['seen']}"}
        block = self._estimate(first["pick"], q0, c.block_plane_z, "pick", "look")
        box = self._estimate(first["place"], q0, c.box_rim_z, "place", "look")

        lift_z = max(c.block_plane_z + s.pregrasp_m, c.box_rim_z + c.block_edge / 2 + s.carry_clear_m)
        if not s.execute:
            here = list(q0)
            plan = {"pregrasp_deg": _r(self._ik(here, (block[0], block[1], c.block_plane_z + s.pregrasp_m))),
                    "above_box_deg": _r(self._ik(c.look_pose_deg, (box[0], box[1], lift_z)))}
            self.log.event("plan", **plan)
            return {"ok": None, "why": "planned only; --execute to move", "pick": _r(block, 4), "place": _r(box, 4),
                    **plan}

        attempt = 0
        while True:
            attempt += 1
            block, before, view = self._grasp(instruction, block, lift_z)
            self._phase("check held")
            if view is not None:
                # Back to the close look: same view as before, so a block still
                # on the table is where it was and a held one is gone. The wrist
                # camera sees the fingers only at the image edge, behind the linkage.
                self._goto(view, "look: held", floor=c.table_z + c.block_edge + 0.02)
                question = grounding.held_question_same_view(pick_label, self._gripper_side())
            else:
                question = grounding.held_question(pick_label)
            check = self._check(question, [("Before the grasp:", before or frame0.jpeg)], "held")
            if check["ok"]:
                break
            self._gripper(c.gripper_open, "open")
            if attempt >= s.attempts:
                return {"ok": False, "why": f"the {pick_label} was not held after {attempt} attempts: {check['seen']}",
                        "attempts": attempt}
            self._phase("look again")
            self._goto(c.look_pose_deg, "look", floor=None)
            again, frame0, q0 = self._locate(instruction, ("pick",), "relook", with_next=False)
            if not again["pick"]["visible"]:
                return {"ok": False, "why": f"lost the {pick_label}: {again['seen']}", "attempts": attempt}
            block = self._estimate(again["pick"], q0, c.block_plane_z, "pick", "relook")

        self._place(instruction, box, lift_z)
        self._phase("final check")
        self._goto(c.look_pose_deg, "look", floor=c.box_rim_z + 0.01)  # empty gripper: only clear the rim
        final = self._check(grounding.final_question(pick_label, place_label),
                            [("At the start:", frame0.jpeg)], "final")
        return {"ok": final["ok"], "why": final["seen"], "attempts": attempt}

    def _grasp(self, instruction: str, block: Vec,
               lift_z: float) -> Tuple[Vec, Optional[bytes], Optional[List[float]]]:
        """Refine, descend, close, lift. Returns the block estimate used, the
        close-up image and the angles it was taken at (None without a close look)."""
        s, c = self.s, self.calib
        clear = c.table_z + c.block_edge + 0.02
        before = view = None
        if s.refine:
            self._phase("close look: pick")
            q = self._camera_over(block, c.block_plane_z, min_tcp_z=clear)
            if q is not None:
                self._goto(q, "close look: pick", floor=clear)
                found, frame, qa = self._locate(instruction, ("pick",), "close-pick", with_next=False)
                before, view = frame.jpeg, qa
                if found["pick"]["visible"]:
                    block = self._estimate(found["pick"], qa, c.block_plane_z, "pick", "close")
        self._phase("pre-grasp")
        # The sim's pads reach 11.4 mm past the TCP: at the 12.5 mm block centre
        # they would graze the table.
        grasp_z = max(c.block_plane_z - s.grasp_depth_m, c.table_z + c.pad_below_tcp + s.finger_clear_m)
        above = (block[0], block[1], c.block_plane_z + s.pregrasp_m)
        approach = self._approach(above, (block[0], block[1], grasp_z), "grasp")
        self._goto(self._ik(self._measured(), above, approach), "pre-grasp", floor=clear)
        self._phase("descend")
        self._line((block[0], block[1], grasp_z), "descend", approach)
        self._gripper(c.gripper_closed, "close")
        self._phase("lift")
        self._line((block[0], block[1], lift_z), "lift", approach)
        return block, before, view

    def _place(self, instruction: str, box: Vec, lift_z: float) -> None:
        s, c = self.s, self.calib
        carry_floor = c.box_rim_z + c.block_edge / 2 + s.carry_clear_m / 2
        if s.refine:
            self._phase("close look: place")
            q = self._camera_over(box, c.box_rim_z, min_tcp_z=lift_z)
            if q is not None:
                self._goto(q, "close look: place", floor=carry_floor)
                found, _, qa = self._locate(instruction, ("place",), "close-place", with_next=False)
                if found["place"]["visible"]:
                    box = self._estimate(found["place"], qa, c.box_rim_z, "place", "close")
        self._phase("above place")
        release = (box[0], box[1], c.box_rim_z + c.block_edge / 2 + s.place_clear_m)
        approach = self._approach((box[0], box[1], lift_z), release, "place")
        self._goto(self._ik(self._measured(), (box[0], box[1], lift_z), approach), "above place", floor=carry_floor)
        self._phase("lower")
        self._line(release, "lower", approach)
        self._gripper(c.gripper_open, "open")
        self._phase("retreat")
        self._line((box[0], box[1], lift_z), "retreat", approach)

    # -- looking -------------------------------------------------------------

    def _look(self) -> Tuple[Frame, List[float]]:
        """A frame that arrived after the arm had been still for ``settle`` s, and the angles then."""
        time.sleep(self.s.settle)
        frame = self.camera.wait_frame(after=time.monotonic(), timeout=self.s.frame_timeout)
        return frame, self._measured()

    def _ask(self, purpose: str, request: Dict[str, Any], images: Sequence[bytes], angles: Sequence[float]) -> str:
        self._vlm_n += 1
        names = [self.log.image(j, purpose) for j in images]
        self.log.event("vlm_request", n=self._vlm_n, purpose=purpose, images=names, angles_deg=_r(angles),
                       prompt=grounding.prompt_text(request))
        t0 = time.monotonic()
        raw = self.vlm(request)
        self._latency = round(time.monotonic() - t0, 3)
        return raw

    def _reply(self, purpose: str, raw: str, parse: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
        try:
            parsed = parse(raw)
        except PlannerError as exc:
            self.log.event("vlm_reply", n=self._vlm_n, purpose=purpose, raw=raw, error=str(exc),
                           latency_s=self._latency)
            raise
        self.log.event("vlm_reply", n=self._vlm_n, purpose=purpose, raw=raw, parsed=parsed, latency_s=self._latency)
        return parsed

    def _locate(self, instruction: str, roles: Sequence[str], purpose: str,
                with_next: bool) -> Tuple[Dict[str, Any], Frame, List[float]]:
        k = self.calib.camera
        frame, angles = self._look()
        opts = dict(roles=roles, coords=self.s.coords, shape=self.s.shape, width=k.width, height=k.height,
                    with_next=with_next)
        request = grounding.build_locate_request(frame.jpeg, instruction, **opts,
                                                 **({"model": self.model} if self.model else {}))
        raw = self._ask(purpose, request, [frame.jpeg], angles)
        found = self._reply(purpose, raw, lambda r: grounding.parse_locate(r, **opts))
        return found, frame, angles

    def _gripper_side(self) -> str:
        """Where in the image the grasp point is, in words: the camera is off the gripper axis."""
        chain, k = self.calib.chain, self.calib.camera
        x, y, z = kinematics.apply(kinematics.inverse(chain.tools["camera"]),
                                   kinematics.position(chain.tools["tcp"]))
        u, v = k.cx + k.fx * x / z, k.cy + k.fy * y / z
        horiz = "left" if u < k.width / 3 else "right" if u > 2 * k.width / 3 else ""
        vert = "top" if v < k.height / 3 else "bottom" if v > 2 * k.height / 3 else ""
        return f"{vert} {horiz}".strip() or "centre"

    def _check(self, question: str, before: Sequence[Tuple[str, bytes]], purpose: str) -> Dict[str, Any]:
        frame, angles = self._look()
        images = list(before) + [("Now:", frame.jpeg)]
        request = grounding.build_check_request(question, images, **({"model": self.model} if self.model else {}))
        raw = self._ask(purpose, request, [j for _, j in images], angles)
        return self._reply(purpose, raw, grounding.parse_check)

    def _estimate(self, found: Dict[str, Any], angles: Sequence[float], plane_z: float, role: str, source: str) -> Vec:
        c = self.calib
        u, v = found["pixel"]
        xyz = kinematics.pixel_to_plane(c.chain, c.camera, angles, u, v, plane_z)
        if xyz is None:
            raise PickPlaceError(f"the ray through {role} pixel ({u:.0f}, {v:.0f}) never meets z = {plane_z:.3f}")
        self.log.event("estimate", role=role, label=found["label"], source=source, pixel=[u, v],
                       plane_z=round(plane_z, 4), xyz=_r(xyz, 4))
        return xyz

    # -- moving --------------------------------------------------------------

    def _phase(self, name: str) -> None:
        self.log.event("phase", name=name)

    def _measured(self) -> List[float]:
        state = self.link.wait_state(after_seq=self._seq)
        self._seq = state["seq"]
        if state["angles_deg"] is None:
            raise LinkError("lost the joint reading")
        return list(state["angles_deg"])

    def _ik(self, seed: Sequence[float], tcp: Sequence[float], approach: Vec = DOWN) -> List[float]:
        try:
            return kinematics.solve_ik(self.calib.chain, seed, tcp, tool="tcp", approach=approach)
        except IKError as exc:
            raise PickPlaceError(str(exc)) from exc

    def _approach(self, top: Vec, bottom: Vec, what: str) -> Vec:
        """Straight down if the arm can, else tipped towards the base a little.

        Close to the base the 280 folds J3 into its limit before a vertical
        gripper reaches the table (r < 0.15 m in the sim); tipping the fingers
        5-15 deg inwards lets the wrist stay further out. A cube does not mind.
        """
        r = math.hypot(top[0], top[1])
        for tilt in TILTS_DEG:
            t = math.radians(tilt)
            a = (-math.sin(t) * top[0] / r, -math.sin(t) * top[1] / r, -math.cos(t))
            try:
                q = kinematics.solve_ik(self.calib.chain, self.calib.look_pose_deg, top, approach=a)
                kinematics.solve_ik(self.calib.chain, q, bottom, approach=a)
            except IKError:
                continue
            if tilt:
                self.log.event("approach", what=what, tilt_deg=tilt)
            return a
        raise PickPlaceError(f"{what}: no IK for z {bottom[2]:.3f}..{top[2]:.3f} at ({top[0]:.3f}, {top[1]:.3f}), "
                             f"even tipped {TILTS_DEG[-1]} deg")

    def _camera_over(self, target: Vec, plane_z: float, min_tcp_z: float) -> Optional[List[float]]:
        """Camera straight above ``target``, looking down, ``close_look_m`` over its plane;
        raised until the TCP (and anything it holds) clears ``min_tcp_z``. None if out of reach."""
        chain, seed = self.calib.chain, self._measured()
        why = ""
        # Near the edge of the 280's reach the full height may be out of reach,
        # and J6 has to turn the off-axis camera towards the base: try lower,
        # then with J6 free.
        for height in (self.s.close_look_m, self.s.close_look_m * 0.8, self.s.close_look_m * 0.65):
            for hold in ((6,), ()):
                h = height
                for _ in range(2):
                    try:
                        q = kinematics.solve_ik(chain, seed, (target[0], target[1], plane_z + h), tool="camera",
                                                 hold=hold)
                    except IKError as exc:
                        why = str(exc)
                        break
                    tcp_z = kinematics.position(chain.pose(q, "tcp"))[2]
                    if tcp_z >= min_tcp_z:
                        return q
                    why = f"TCP at z={tcp_z:.3f} under {min_tcp_z:.3f}"
                    h += min_tcp_z - tcp_z + 0.005
        self.log.event("skip", why=f"no close look: {why}")
        return None

    def _goto(self, goal: Sequence[float], phase: str, floor: Optional[float], speed: Optional[int] = None) -> None:
        """Joint-space line from the measured angles; refuses one whose TCP dips below ``floor``."""
        path = kinematics.joint_path(self._measured(), goal, self.s.max_step)
        if floor is not None:
            low = min(kinematics.position(self.calib.chain.pose(q, "tcp"))[2] for q in path)
            if low < floor - 1e-3:
                raise PickPlaceError(f"{phase}: the joint path takes the TCP down to z={low:.3f} (floor {floor:.3f})")
        self._send(path, phase, speed or self.s.speed)

    def _line(self, goal: Sequence[float], phase: str, approach: Vec = DOWN) -> None:
        try:
            path = kinematics.line_path(self.calib.chain, self._measured(), goal, step_m=self.s.line_step_m,
                                        max_step=self.s.max_step, approach=approach)
        except IKError as exc:
            raise PickPlaceError(f"{phase}: {exc}") from exc
        self._send(path, phase, self.s.slow_speed)

    def _send(self, path: Sequence[Sequence[float]], phase: str, speed: int) -> None:
        for i, q in enumerate(path, 1):
            last = i == len(path)
            self.link.move_joints(q, speed)
            tcp = kinematics.position(self.calib.chain.pose(q, "tcp"))
            self.log.event("move", phase=phase, waypoint=i, of=len(path), angles_deg=_r(q), speed=speed,
                           tcp_xyz=_r(tcp, 4))
            self._arrive(q, self.s.arrive_tol if last else self.s.pass_tol, phase, last)

    def _arrive(self, target: Sequence[float], tol: float, phase: str, last: bool) -> None:
        end = time.monotonic() + self.s.arrive_timeout
        err = float("inf")
        while True:
            state = self.link.wait_state(after_seq=self._seq, timeout=2.0)
            self._seq = state["seq"]
            if state["angles_deg"] is not None:
                err = max(abs(a - t) for a, t in zip(state["angles_deg"], target))
                if err <= tol:
                    return
            if time.monotonic() > end:
                if last and err <= self.s.stall_tol:
                    self.log.event("arrived_approx", phase=phase, err_deg=round(err, 2))
                    return
                raise PickPlaceError(f"{phase}: still {err:.1f} deg from the waypoint after {self.s.arrive_timeout:g} s")

    def _gripper(self, value: int, action: str) -> None:
        self.link.gripper(value, self.s.gripper_speed)
        self.log.event("gripper", action=action, value=value)
        time.sleep(self.s.gripper_wait)
