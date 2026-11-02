"""Drawer tasks: put an object in a drawer, take one out, or search for one.

Same split as :mod:`mycobot_brain.pickplace`, whose motion helpers this
reuses: the VLM reads the command, says which drawers are open, which to
search next, what is in an opened drawer and whether each step worked; code
does the geometry. Drawers are opened by a scripted handle grasp and a
straight pull of ``open_m`` along the calib's pull axis, and shut the same way
backwards; what is inside is found by pixel → ray → the plane at the drawer
floor plus the catalogue's half height for the shape the VLM named.

Commands (Korean in, as Hojin says them):

- put: ``파란 큐브를 가운데 서랍에 넣어`` - the object lies on the table.
- take: ``왼쪽 아래 서랍에 있는 물건을 꺼내 줘`` - out of a named drawer to the handover zone.
- find: ``노란색 원뿔 찾아서 꺼내 줘`` - open → look → found? take it out : close → next;
  every drawer this run opened is shut again.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from mycobot_brain import collision, drawer_vlm, grounding, kinematics
from mycobot_brain.fixture import Drawer, Fixture, HandleStage, Shape
from mycobot_brain.kinematics import IKError, Vec
from mycobot_brain.pickplace import DOWN, TILTS_DEG, PathBlocked, PickPlace, PickPlaceError, Settings, _r

UP: Vec = (0.0, 0.0, 1.0)


@dataclass
class DrawerSettings(Settings):
    handle_speed: int = 15  # pull and push
    view: str = "tilted"  # how to look into an opened drawer: tilted | topdown
    look_distance_m: float = 0.16  # camera from the drawer floor centre when the calib has no look pose
    look_tilt_deg: float = 35.0
    wall_clear_m: float = 0.015  # object bottom above a drawer wall top while carrying
    handover_clear_m: float = 0.005  # object bottom above the table at release
    drawer_clear_m: float = 0.004  # object bottom above a drawer floor at release
    body_clear_m: float = 0.005  # TCP and camera kept out of the cabinet body on joint-space moves
    max_drawers: int = 9
    yaw_drift_deg: float = 30.0  # how far a free-heading pull may turn the gripper
    finger_half_m: float = 0.028  # TCP to the outside of a finger, jaw open (45.6 mm gap + the finger)
    pad_floor_clear_m: float = 0.001  # pad tips above a drawer floor or the table at the grasp
    leave_mat_m: float = 0.12  # TCP over the table before leaving the handover mat for the cabinet


def _add(a: Sequence[float], b: Sequence[float], s: float = 1.0) -> Vec:
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def _in_box(p: Sequence[float], lo: Sequence[float], hi: Sequence[float], m: float) -> bool:
    return all(lo[i] - m <= p[i] <= hi[i] + m for i in range(3))


@dataclass
class HandlePlan:
    """How one stage of a drawer's pull holds the handle: tool frame and IK seed at the stage's two
    ends (``from_m``, ``to_m``), the way onto the handle (None: along the tool axis), and whether the
    heading may drift during the pull."""

    stage: HandleStage
    frames: Dict[float, Tuple[Vec, Optional[Vec]]]
    seeds: Dict[float, List[float]]
    free: bool
    in_axis: Optional[Vec] = None

    def frame_at(self, opening: float) -> Tuple[Vec, Optional[Vec]]:
        """Tool z and x at ``opening``, evenly between the two ends (a top grip tilts on the way)."""
        s0, s1 = self.stage.from_m, self.stage.to_m
        t = (opening - s0) / (s1 - s0)
        (a0, j0), (a1, j1) = self.frames[s0], self.frames[s1]
        mix = [x + (y - x) * t for x, y in zip(a0, a1)]
        n = kinematics.norm(mix)
        return (mix[0] / n, mix[1] / n, mix[2] / n), (j0 if t < 0.5 else j1)

    def in_at(self, opening: float) -> Vec:
        return self.in_axis if self.in_axis is not None else self.frame_at(opening)[0]


class NotHeld(PickPlaceError):
    """The VLM does not see the object in the gripper after a grasp; the gripper is open again."""


class DrawerTask(PickPlace):
    s: DrawerSettings

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.calib.fixture is None:
            raise PickPlaceError(f"{self.calib.source or 'the calib'} has no cabinet")
        self.fx: Fixture = self.calib.fixture
        #: how far the code has pulled each drawer out, as commanded; the VLM says whether it is open.
        self.opening: Dict[str, float] = {d: 0.0 for d in self.fx.drawers}
        self.opened: List[str] = []  # drawers this run opened, in order
        self._plans: Dict[str, List[HandlePlan]] = {}
        #: what the fingers may come close to in the moves now being made (see :meth:`_near`).
        self._ctx: Dict[str, Any] = {}
        self._lifted: Optional[List[float]] = None  # the arm after the last lift, holding the object
        #: how far the open fingers reach out from the TCP axis: the calib's finger capsules, else the setting.
        self._finger_reach = self.s.finger_half_m
        robot = self.calib.robot
        if robot is not None:
            frames = self.calib.chain.frames([0.0] * 6)
            to_tcp = kinematics.inverse(frames["tcp"])
            for c in robot.capsules:
                if c.name in robot.fingers:
                    rel = kinematics.mul(to_tcp, frames[c.frame])
                    for end in (c.a, c.b):
                        e = kinematics.apply(rel, end)
                        self._finger_reach = max(self._finger_reach,
                                                 math.hypot(e[0], e[1]) + c.radius + self.s.robot_margin_m)

    # -- the commands -----------------------------------------------------------

    def _run(self, instruction: str) -> Dict[str, Any]:
        s, fx = self.s, self.fx
        state = self.link.wait_state()
        if state["angles_deg"] is None:
            raise PickPlaceError("the arm server has no joint reading to start from")
        cmd = self._parse(instruction)
        if s.execute:
            self.link.enable()
            self.enabled = True
            self._phase("cabinet look")
            self._goto(fx.cabinet_look_pose_deg, "cabinet look", floor=None)
            self._gripper(self.calib.gripper_open, "open")
        else:
            return self._plan(cmd)
        desc = drawer_vlm.describe(cmd["colour"], cmd["shape"])
        if cmd["task"] == "find":
            out = self._find(cmd, desc)
        elif cmd["task"] == "take":
            out = self._take(cmd, desc)
        else:
            out = self._put(cmd, desc)
        out = {**out, "task": cmd["task"], "drawers_opened": list(self.opened)}
        if out.get("ok") is not False or self.opened:
            out = self._final(out)
        return out

    def _parse(self, instruction: str) -> Dict[str, Any]:
        """The VLM reads the command: put / take / find, which object, which drawer."""
        req = drawer_vlm.build_parse_request(instruction, self.fx, **({"model": self.model} if self.model else {}))
        raw = self._ask("parse", req, [], self._measured())
        return self._reply("parse", raw, lambda r: drawer_vlm.parse_command(r, self.fx))

    def _plan(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        """Without --execute: the command as read, and whether the arm can reach the handles."""
        reach = {}
        for d in self.fx.drawers.values():
            try:
                a, _ = self._handle_frame(d)
                reach[d.id] = round(math.degrees(math.acos(max(-1.0, min(1.0, kinematics.dot(
                    a, self.fx.stages(d)[0].approach_from))))), 1)
            except PickPlaceError:
                reach[d.id] = None
        self.log.event("plan", command=cmd, handle_tilt_deg=reach)
        return {"ok": None, "why": "planned only; --execute to move", "command": cmd, "handle_tilt_deg": reach}

    def _find(self, cmd: Dict[str, Any], desc: str) -> Dict[str, Any]:
        searched: Dict[str, str] = {}
        # A drawer the arm cannot pull all the way is not offered to the VLM at all.
        reachable = []
        for d in sorted(self.fx.drawers):
            if not self.fx.drawers[d].usable:
                self.log.event("unreachable", drawer=d, why="not in use (calib)")
                continue
            try:
                self._handle_plan(self.fx.drawers[d])
                reachable.append(d)
            except PickPlaceError as exc:
                self.log.event("unreachable", drawer=d, why=str(exc))
        for _ in range(self.s.max_drawers):
            candidates = [d for d in reachable if d not in searched]
            if not candidates:
                break
            view = self._cabinet(f"find the {desc} and take it out", searched, candidates)
            d_id = view["next"]
            if d_id is None:
                return {"ok": False, "why": f"the VLM sees no drawer worth opening: {view['seen']}"}
            d = self.fx.drawer(d_id)
            look = self._open_and_look(d, desc)
            searched[d.id] = ", ".join(f"{o['colour']} {o['shape']}" for o in look["objects"]) or "empty"
            if look["target"] is not None:
                obj = look["objects"][look["target"]]
                self._take_out(d, obj, look, desc)
                return {"ok": True, "why": f"the {desc} was in {d.id}", "found_in": d.id}
            self._close(d)
            self._phase("cabinet look")
            self._goto(self.fx.cabinet_look_pose_deg, "cabinet look", floor=None)
        unreachable = sorted(set(self.fx.drawers) - set(reachable))
        return {"ok": False, "why": f"no {desc} in {sorted(searched)}"
                + (f"; could not open {unreachable}" if unreachable else "")}

    def _take(self, cmd: Dict[str, Any], desc: str) -> Dict[str, Any]:
        d = self.fx.drawer(cmd["drawer"])
        if not d.usable:
            return {"ok": False, "why": f"{d.id} is not in use (calib): no plan opens it"}
        look = self._open_and_look(d, desc)
        target = look["target"]
        if target is None and desc == "any object" and look["objects"]:
            target = 0  # the model's pick was not on the floor (a handle, say); anything that is will do
        if target is None:
            self._close(d)
            return {"ok": False, "why": f"no {desc} in {d.id}: {look['seen']}"}
        obj = look["objects"][target]
        self._take_out(d, obj, look, drawer_vlm.describe(obj["colour"], obj["shape"]))
        return {"ok": True, "why": f"took the {obj['colour']} {obj['shape']} out of {d.id}"}

    def _put(self, cmd: Dict[str, Any], desc: str) -> Dict[str, Any]:
        c, fx = self.calib, self.fx
        shape = self._shape(cmd["shape"])
        if not fx.drawer(cmd["drawer"]).usable:
            return {"ok": False, "why": f"{cmd['drawer']} is not in use (calib): no plan opens it"}
        # Find it on the table first: no point opening a drawer for something that is not there.
        self._phase("table look")
        self._goto(c.look_pose_deg, "table look", floor=None)
        found, _, q = self._locate(f"pick up the {desc}", ("pick",), "table", with_next=False,
                                   describe={"pick": f"the {desc}"})
        if not found["pick"]["visible"]:
            return {"ok": False, "why": f"no {desc} on the table: {found['seen']}"}
        start = self._estimate(found["pick"], q, c.table_z + shape.height / 2, "pick", "table")
        d = fx.drawer(cmd["drawer"])
        look = self._open_and_look(d, None)
        occupied = [self._object_xyz(d, o, look["angles"]) for o in look["objects"]]
        spot = fx.free_spot(d, fx.open_m, [p for p in occupied if p is not None], shape.width)
        if spot is None:
            self._close(d)
            return {"ok": False, "why": f"no room for a {desc} in {d.id}"}
        self.log.event("spot", drawer=d.id, xyz=_r(spot, 4), occupied=[_r(p, 4) for p in occupied if p])
        for attempt in range(1, self.s.attempts + 1):
            try:
                self._pick_up(start, c.table_z, shape, desc, "table")
                break
            except NotHeld:
                if attempt == self.s.attempts:
                    raise
            self._goto(c.look_pose_deg, "table look", floor=None)
            found, _, q = self._locate(f"pick up the {desc}", ("pick",), "table", with_next=False,
                                       describe={"pick": f"the {desc}"})
            if not found["pick"]["visible"]:
                raise PickPlaceError(f"lost the {desc} on the table: {found['seen']}")
            start = self._estimate(found["pick"], q, c.table_z + shape.height / 2, "pick", "table")
        self._put_down(spot, d.floor_z, shape, f"into {d.id}", wall_top=self._wall_top(d),
                       clear=self.s.drawer_clear_m, drawer=d)
        self._phase("check put")
        self._goto(look["angles"], f"look: {d.id}", floor=None)
        check = self._check(drawer_vlm.put_question(desc, d), [("Before:", look["jpeg"])], "put")
        self._close(d)
        return {"ok": check["ok"], "why": check["seen"], "spot": _r(spot, 4)}

    def _final(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Back at the cabinet view: every drawer this run opened must be shut."""
        self._phase("final check")
        self._goto(self.fx.cabinet_look_pose_deg, "cabinet look", floor=None)
        view = self._cabinet("check that every drawer is shut", None, None)
        left = [d for d in view["open_drawers"] if d in self.opened]
        if left:
            # Once more for each drawer this run opened that still shows open.
            for d_id in left:
                self._close(self.fx.drawer(d_id), force=True)
            self._goto(self.fx.cabinet_look_pose_deg, "cabinet look", floor=None)
            view = self._cabinet("check that every drawer is shut", None, None)
            left = [d for d in view["open_drawers"] if d in self.opened]
        out = {**out, "open_after": view["open_drawers"]}
        if left:
            out.update(ok=False, why=f"{out.get('why', '')}; still open: {left}")
        return out

    # -- drawers -------------------------------------------------------------------

    def _handle_frame(self, d: Drawer) -> Tuple[Vec, Optional[Vec]]:
        """Approach and jaw axes for grasping ``d``'s handle shut; see :meth:`_handle_plan`."""
        first = self._handle_plan(d)[0]
        return first.frames[first.stage.from_m]

    def _handle_plan(self, d: Drawer) -> List[HandlePlan]:
        """How to hold ``d``'s handle over the whole pull, stage by stage (:meth:`Fixture.stages`),
        checked once before the arm moves.

        Per stage, in order: the calib's frame all the way; then (with a jaw
        axis) the jaw held and the heading left to drift up to ``yaw_drift_deg``
        - pads pinching two faces can turn about the line through them, and
        near the edge of reach the arm needs that; then tipped 5-15 deg towards
        the base, as pick-place tips its grasp (unless that leans the camera
        into the cabinet).
        """
        if d.id in self._plans:
            return self._plans[d.id]
        plans = [self._stage_plan(d, k, stage) for k, stage in enumerate(self.fx.stages(d))]
        self._plans[d.id] = plans
        return plans

    def _stage_plan(self, d: Drawer, k: int, stage: HandleStage) -> HandlePlan:
        fx = self.fx
        a0 = fx.stage_approach(stage, stage.from_m)
        h = fx.stage_tcp(stage, stage.to_m)
        r = math.hypot(h[0], h[1])
        axis = kinematics.cross(a0, (-h[0] / r, -h[1] / r, 0.0))
        options: List[Tuple[str, Optional[kinematics.Mat], bool]] = [("calib", None, False)]
        if stage.jaw is not None:
            options.append(("free heading", None, True))
        if kinematics.norm(axis) > 1e-6:
            for tilt in TILTS_DEG[1:]:
                options.append((f"tipped {tilt} deg", kinematics.axis_angle(axis, math.radians(tilt)), False))
        whys = []
        which = "" if len(fx.stages(d)) == 1 else f" stage {k + 1}"
        for name, rot, free in options:
            try:
                plan = self._try_handle(d, stage, rot, free)
            except (IKError, PickPlaceError) as exc:
                whys.append(f"{name}: {exc}")
                continue
            if name != "calib":
                self.log.event("approach", what=f"handle {d.id}{which}", how=name)
            return plan
        raise PickPlaceError(f"handle of {d.id}{which}: no grasp that holds over the whole pull ({'; '.join(whys)})")

    def _try_handle(self, d: Drawer, stage: HandleStage, rot: Optional[kinematics.Mat], free: bool) -> HandlePlan:
        """The first IK branch, by seed, that holds the grip over the stage's pull and back-off."""
        # The table look pose first (an ordinary elbow-up arm, whatever the cabinet view needs), then
        # the drawer's own look and the cabinet look: one seed's branch can run into a joint limit
        # mid-pull where another's does not (r2c0 on sim-scene's numbers from v1's look pose).
        looks = [q for q in (self.calib.look_pose_deg, d.look_pose_deg, self.fx.cabinet_look_pose_deg) if q]
        err: Exception = PickPlaceError("no seed pose")
        tried = set()
        # v3's grips (sim-scene's search, the whole tool rotation given) sit in arm branches none of the
        # look poses leads to: the wrist turned on J4/J6. Each branch once: seeds that end in the same
        # pre-grasp say nothing new. A grip given by its axes alone finds its branch from the looks.
        more = self._more_seeds(looks) if stage.exact_jaw else iter(())
        for seed in itertools.chain([stage.seed_deg] if stage.seed_deg else [], looks, more):
            fx = self.fx
            a = fx.stage_approach(stage, stage.from_m)
            if rot is not None:
                a = tuple(sum(rot[i][m] * a[m] for m in range(3)) for i in range(3))  # type: ignore[assignment]
            j = stage.jaw if rot is None or stage.jaw is None else \
                tuple(sum(rot[i][m] * stage.jaw[m] for m in range(3)) for i in range(3))
            in0 = (stage.in_axis if rot is None or stage.in_axis is None else
                   tuple(sum(rot[i][m] * stage.in_axis[m] for m in range(3)) for i in range(3))) or a
            try:
                pre = kinematics.solve_ik(self.calib.chain, seed,
                                          _add(fx.stage_tcp(stage, stage.from_m), in0, -fx.stage_standoff(stage, stage.from_m)),
                                          approach=a, jaw=j, either_jaw=not stage.exact_jaw)
            except IKError as exc:
                err = exc if not tried else err  # a pre-grasp no seed reaches; else keep what a reached one hit
                continue
            branch = tuple(round(x / 5.0) for x in pre)
            if branch in tried:
                continue
            tried.add(branch)
            try:
                return self._try_handle_from(d, stage, rot, free, pre)
            except (IKError, PickPlaceError) as exc:
                if len(tried) == 1 or not isinstance(exc, IKError):
                    err = PickPlaceError(f"{len(tried)} arm branch(es) reach the pre-grasp; {exc}")
        raise err

    @staticmethod
    def _more_seeds(looks: Sequence[Sequence[float]]) -> Iterator[List[float]]:
        """The look poses with the wrist turned: J4 in 45 deg steps, J6 in 45 deg steps."""
        for q in looks:
            for j4 in (-90.0, -45.0, 0.0, 45.0):
                for j6 in (-135.0, -45.0, 45.0, 135.0, 180.0):
                    yield [q[0], q[1], q[2], j4, q[4], j6]

    def _try_handle_from(self, d: Drawer, stage: HandleStage, rot: Optional[kinematics.Mat], free: bool,
                         seed: Sequence[float]) -> HandlePlan:
        fx, chain = self.fx, self.calib.chain

        def turn(v: Vec) -> Vec:
            return v if rot is None else tuple(sum(rot[i][m] * v[m] for m in range(3)) for i in range(3))  # type: ignore[return-value]

        s0, s1 = stage.from_m, stage.to_m
        a, a_end = turn(fx.stage_approach(stage, s0)), turn(fx.stage_approach(stage, s1))
        j = None if stage.jaw is None else turn(stage.jaw)
        in_axis = None if stage.in_axis is None else turn(stage.in_axis)
        in0 = in_axis or a
        either = not stage.exact_jaw
        pre = kinematics.solve_ik(chain, seed, _add(fx.stage_tcp(stage, s0), in0, -fx.stage_standoff(stage, s0)),
                                  approach=a, jaw=j, either_jaw=either)
        to_handle = kinematics.line_path(chain, pre, fx.stage_tcp(stage, s0), step_m=0.02, approach=a, jaw=j,
                                         either_jaw=either)
        pull = kinematics.line_path(chain, to_handle[-1], fx.stage_tcp(stage, s1), step_m=self.s.line_step_m,
                                    approach=None if free else a, jaw=j, approach_end=None if free else a_end,
                                    either_jaw=either)
        t = chain.pose(pull[-1], "tcp")
        a1, j1 = kinematics.column(t, 2), kinematics.column(t, 0) if j is not None else None
        drift = math.degrees(math.acos(max(-1.0, min(1.0, kinematics.dot(a_end, a1)))))
        if drift > self.s.yaw_drift_deg:
            raise PickPlaceError(f"the gripper turns {drift:.0f} deg over the pull")
        back = kinematics.line_path(chain, pull[-1],
                                    _add(fx.stage_tcp(stage, s1), in_axis or a1, -fx.stage_standoff(stage, s1)),
                                    step_m=0.02, approach=a1, jaw=j1, either_jaw=either)
        # A tipped or turned gripper leans its back (and the camera) towards the cabinet face. The drawer
        # is at s0 on the way in, moves with the hand on the pull (and the push), and is at s1 on the way out.
        at0, at1 = {**self.opening, d.id: s0}, {**self.opening, d.id: s1}
        c = self.calib
        release = self._jaw_at(stage.release if stage.release is not None else c.gripper_open)
        preshape = self._jaw_at(stage.preshape) if stage.preshape is not None else release
        shut = self._jaw_at(stage.close if stage.close is not None else c.gripper_closed)
        legs: List[Tuple[List[List[float]], Dict[str, Any]]] = [
            ([pre] + to_handle, dict(openings=at0, jaw=preshape)),
            ([to_handle[-1]] + pull, dict(openings=at0, moving=d.id, jaw=shut)),
            ([pull[-1]] + back, dict(openings=at1, jaw=release))]
        for leg, ctx in legs:
            hit = self._path_clear(leg, skip=d.id, at_handle=True, **ctx)
            if hit:
                raise PickPlaceError(f"the {hit}")
        # Joint moves between the cabinet look and the standoffs at both ends must keep the gripper off
        # every handle (D3 s500: rising from a 4 cm back-off it caught the knob above), and the arm off
        # itself: they may go round (:meth:`_route`).
        look = fx.cabinet_look_pose_deg
        if look:
            # The pre-grasp at s0 is also where a push to s0 backs off to.
            for name, q, openings in ((f"standoff at {s0 * 1000:.0f} mm", pre, at0),
                                      (f"standoff at {s1 * 1000:.0f} mm", back[-1], at1)):
                # In with the jaw at preshape, out (after letting go) at release.
                for there, other, jaw in ((look, q, preshape), (q, look, release)):
                    vias, _, why = self._route(there, other, None, openings=openings, jaw=jaw)
                    if vias is None:
                        raise PickPlaceError(f"cabinet look <-> {name}: the {why}")
        return HandlePlan(stage=stage, frames={s0: (a, j), s1: (a1, j1)}, free=free, seeds={s0: pre, s1: back[-1]},
                          in_axis=in_axis)

    def _move_handle(self, d: Drawer, start: float, end: float, verb: str) -> None:
        """Move ``d`` from ``start`` m out to ``end``: per stage of its pull that the move covers (in
        order to open, backwards to shut), grasp the handle, move it, let go, back off."""
        plans = self._handle_plan(d)
        legs: List[Tuple[int, HandlePlan, float, float]] = []
        for k, plan in enumerate(plans):
            lo, hi = max(plan.stage.from_m, min(start, end)), min(plan.stage.to_m, max(start, end))
            if hi > lo + 1e-4:
                legs.append((k, plan, lo, hi) if end > start else (k, plan, hi, lo))
        if end < start:
            legs.reverse()
        self._phase(f"{verb} {d.id}")
        for i, (k, plan, s0, s1) in enumerate(legs):
            tag = f"{verb} {d.id}" + ("" if len(plans) == 1 else f" ({k + 1}/{len(plans)})")
            self._handle_leg(d, plan, s0, s1, tag, first=i == 0)

    def _handle_leg(self, d: Drawer, plan: HandlePlan, start: float, end: float, tag: str, first: bool) -> None:
        """Grasp the handle where the drawer is (``start`` m out), move it to ``end``, let go, back off."""
        fx, c, stage = self.fx, self.calib, plan.stage
        near = min(plan.frames, key=lambda o: abs(o - start))
        a, j = plan.frame_at(start)
        a_end = plan.frame_at(end)[0]
        release = stage.release if stage.release is not None else c.gripper_open
        preshape = stage.preshape if stage.preshape is not None else release
        self._gripper(preshape, "open")
        # Every knob approach starts from the cabinet look, as the reset pose and sim-scene's D1 runs
        # do: in D3 s462 the joint move from the table look to r0c2's pre-grasp swept low across the
        # cabinet front and the arm stalled 6.8 deg short (it hit something our TCP/camera check misses).
        # A later stage of the same pull goes straight on from the back-off (routed).
        here, look = self._measured(), fx.cabinet_look_pose_deg
        if first and look and max(abs(x - y) for x, y in zip(here, look)) > 1.0:
            # Not low over the table, where an object may stand (the handover mat).
            self._goto(look, f"{tag}: cabinet look", floor=self.calib.table_z + self.s.leave_mat_m)
        try:
            q = kinematics.solve_ik(c.chain, plan.seeds[near],
                                    _add(fx.stage_tcp(stage, start), plan.in_at(start), -fx.stage_standoff(stage, start)),
                                    approach=a, jaw=j, either_jaw=not stage.exact_jaw)
        except IKError as exc:
            raise PickPlaceError(f"{tag}: pre-grasp: {exc}") from exc
        self._goto(q, f"{tag}: pre-grasp", floor=None)
        with self._near(skip=d.id, at_handle=True):
            either = not stage.exact_jaw
            self._line(fx.stage_tcp(stage, start), f"{tag}: to handle", a, j, either_jaw=either)
            self._gripper(stage.close if stage.close is not None else c.gripper_closed, "close")
            with self._near(moving=d.id):
                self._line(fx.stage_tcp(stage, end), tag, None if plan.free else a, j, speed=self.s.handle_speed,
                           approach_end=None if plan.free else a_end, either_jaw=either)
            self.opening[d.id] = end
            self._gripper(release, "open")
            # Back off along the tool axis as it is now (with a free heading it has turned on the way), or
            # straight up off a top grip.
            t = c.chain.pose(self._measured(), "tcp")
            a_now, j_now = kinematics.column(t, 2), (kinematics.column(t, 0) if j is not None else None)
            out = plan.in_axis if plan.in_axis is not None else a_now
            self._line(_add(fx.stage_tcp(stage, end), out, -fx.stage_standoff(stage, end)), f"{tag}: back off",
                       a_now, j_now, either_jaw=either)

    def _open(self, d: Drawer, again: bool = False) -> None:
        # ``again``: the VLM does not see it open, so pull from shut whatever the code thinks.
        self._move_handle(d, 0.0 if again else self.opening[d.id], self.fx.open_m, "open")
        if d.id not in self.opened:
            self.opened.append(d.id)

    def _close(self, d: Drawer, force: bool = False) -> None:
        # ``force``: the VLM still sees it open, so push from fully out whatever the code thinks.
        start = self.fx.open_m if force else self.opening[d.id]
        self._move_handle(d, start, 0.0, "close")

    def _look_pose(self, d: Drawer) -> List[float]:
        """The calib's view into the opened drawer, else one aimed at its floor from the robot's side."""
        pose = d.topdown_pose_deg if self.s.view == "topdown" else d.look_pose_deg
        if pose is not None:
            return pose
        c = self.fx.interior_center(d, self.fx.open_m)
        target = (c[0], c[1], c[2] + 0.01)
        tilt = 0.0 if self.s.view == "topdown" else self.s.look_tilt_deg
        dist = self.s.look_distance_m
        why = ""
        # Near the edge of reach the answer depends on the seed and a centimetre or a few degrees.
        for seed in (self._measured(), self.calib.look_pose_deg):
            for dd, dt in ((0.0, 0.0), (0.02, 5.0), (-0.02, -5.0), (0.03, 10.0)):
                try:
                    return kinematics.look_at(self.calib.chain, seed, target, distance=dist + dd,
                                              tilt_deg=max(0.0, tilt + dt) if tilt else 0.0)
                except IKError as exc:
                    why = str(exc)
        raise PickPlaceError(f"no {self.s.view} view into {d.id}: {why}")

    def _open_and_look(self, d: Drawer, target: Optional[str]) -> Dict[str, Any]:
        """Open ``d`` and say what is in it; one more pull if neither view sees it open."""
        self._open(d)
        look = self._look_into(d, target)
        if look["open"] or look["objects"] or self._out_from_cabinet_view(d, look):
            return {**look, "open": True}
        self.log.event("retry", why=f"the VLM does not see {d.id} open", seen=look["seen"])
        self.opening[d.id] = 0.0  # both views say shut: pull from there
        self._open(d, again=True)
        look = self._look_into(d, target)
        if look["open"] or look["objects"] or self._out_from_cabinet_view(d, look):
            return {**look, "open": True}
        self._close(d, force=True)
        raise PickPlaceError(f"{d.id} does not open: {look['seen']}")

    def _out_from_cabinet_view(self, d: Drawer, look: Dict[str, Any]) -> bool:
        """Whether the cabinet view shows ``d`` pulled out. The drawer view called an open, empty drawer
        "closed, I cannot see inside" (D3 s500, r0c1); D2 had objects in every opened drawer it showed."""
        self._goto(self.fx.cabinet_look_pose_deg, f"cabinet look: is {d.id} out", floor=None)
        view = self._cabinet(f"check whether drawer {d.id} is pulled out", None, None)
        out = d.id in view["open_drawers"]
        self.log.event("open_check", drawer=d.id, drawer_view=look["seen"], cabinet_view=view["open_drawers"], out=out)
        return out

    def _look_into(self, d: Drawer, target: Optional[str]) -> Dict[str, Any]:
        self._phase(f"look into {d.id}")
        self._goto(self._look_pose(d), f"look: {d.id}", floor=None)
        k, chain = self.calib.camera, self.calib.chain
        frame, q = self._look()
        floor = drawer_vlm.floor_box(self.fx, d, self.fx.open_m, chain, k, q)
        # D2 (sim, 18 tilted frames): colour hints took colour right 0.938 -> 0.969; the floor box and
        # shape hints made no difference there and stay. The tilted view sees shapes (0.97 vs 0.69).
        req = drawer_vlm.build_drawer_request(frame.jpeg, self.fx, d, floor, target=target, view=self.s.view,
                                              colour_hints=True, **({"model": self.model} if self.model else {}))
        raw = self._ask(f"drawer-{d.id}", req, [frame.jpeg], q)
        look = self._reply(f"drawer-{d.id}", raw, lambda r: drawer_vlm.parse_drawer(
            r, self.fx, with_target=target is not None, width=k.width, height=k.height))
        # Keep what stands on this drawer's floor. In D2 the model twice listed a top-row drawer's own
        # dark handle as a "purple cube"; its ray meets the floor plane in front of the tray.
        kept, target_at = [], None
        for i, o in enumerate(look["objects"]):
            xyz = self._object_xyz(d, o, q)
            on_floor = xyz is not None and self.fx.inside(d, xyz, self.fx.open_m, margin=-0.01)
            self.log.event("estimate", role="object", drawer=d.id, index=i, label=f"{o['colour']} {o['shape']}",
                           source=self.s.view, pixel=o["pixel"], xyz=None if xyz is None else _r(xyz, 4),
                           kept=on_floor)
            if on_floor:
                if look.get("target") == i:
                    target_at = len(kept)
                kept.append(o)
        return {**look, "objects": kept, "target": target_at, "angles": q, "jpeg": frame.jpeg}

    def _object_xyz(self, d: Drawer, o: Dict[str, Any], q: Sequence[float]) -> Optional[Vec]:
        shape = self.fx.shapes.get(o["shape"])
        if shape is None:
            return None
        c = self.calib
        return kinematics.pixel_to_plane(c.chain, c.camera, q, *o["pixel"], d.floor_z + shape.height / 2)

    def _wall_top(self, d: Drawer) -> float:
        return d.wall_top

    def _shape(self, name: str) -> Shape:
        if name not in self.fx.shapes:
            raise PickPlaceError(f"no catalogue entry for shape {name!r}")
        return self.fx.shapes[name]

    def _cabinet(self, goal: str, searched: Optional[Dict[str, str]],
                 candidates: Optional[Sequence[str]]) -> Dict[str, Any]:
        c, fx = self.calib, self.fx
        frame, q = self._look()
        # The model boxes the pulled-out trays and code says whose they are. D2 (sim, 10 cabinet
        # frames, 9 open drawers): naming drawers went by up and down in sim-scene's rolled view
        # (open recall 0.33-0.44, also with handle marks); pointing at open handles 0.67; boxed trays
        # 0.89 with precision 0.89-1.0.
        trays = drawer_vlm.tray_boxes(fx, c.chain, c.camera, q, fx.open_m)
        req = drawer_vlm.build_cabinet_request(frame.jpeg, fx, None, goal=goal, searched=searched,
                                               candidates=candidates, tray=True,
                                               **({"model": self.model} if self.model else {}))
        purpose = "cabinet-next" if candidates is not None else "cabinet-shut"
        raw = self._ask(purpose, req, [frame.jpeg], q)
        return self._reply(purpose, raw, lambda r: drawer_vlm.parse_cabinet(r, fx, candidates, open_boxes=trays))

    # -- objects ----------------------------------------------------------------------

    def _take_out(self, d: Drawer, obj: Dict[str, Any], look: Dict[str, Any], desc: str) -> None:
        for attempt in range(1, self.s.attempts + 1):
            shape = self._shape(obj["shape"])
            xyz = self._object_xyz(d, obj, look["angles"])
            if xyz is None:
                raise PickPlaceError(f"the ray to the {desc} never meets the floor of {d.id}")
            if not self.fx.inside(d, xyz, self.fx.open_m, margin=-0.01):
                raise PickPlaceError(f"the {desc} at {_r(xyz, 3)} is not over the floor of {d.id}")
            others = [p for o in look["objects"] if o is not obj
                      for p in [self._object_xyz(d, o, look["angles"])] if p is not None]
            try:
                self._pick_up(xyz, d.floor_z, shape, desc, d.id, wall_top=self._wall_top(d), drawer=d, others=others)
                break
            except NotHeld:
                if attempt == self.s.attempts:
                    raise
            # Missed: look again from the drawer view, where the object may have been pushed.
            look = self._look_into(d, desc)
            if look["target"] is None:
                raise PickPlaceError(f"lost the {desc} in {d.id}: {look['seen']}")
            obj = look["objects"][look["target"]]
        self._leave_drawer(d, shape)
        h = self.fx.handover_center
        self._put_down((h[0], h[1], self.calib.table_z), self.calib.table_z, shape, "handover",
                       wall_top=None, clear=self.s.handover_clear_m)
        self._phase("check handover")
        q = self._camera_over((h[0], h[1], self.calib.table_z), self.calib.table_z + shape.height / 2,
                              min_tcp_z=self.calib.table_z + shape.height + 0.03)
        if q is not None:
            self._goto(q, "look: handover", floor=self.calib.table_z + shape.height + 0.02)
        check = self._check(drawer_vlm.handover_question(desc), [], "handover")
        # Straight up off the mat before anything else: in D3 s502 (788cb77) the joint move from this
        # low look to the cabinet look swept the gripper through the cone just handed over.
        self._rise(self.calib.table_z + self.s.leave_mat_m, "up from the handover")
        # Shut it either way: the gripper has let go, and a drawer left open fails the task too.
        self._close(d)
        if not check["ok"]:
            raise PickPlaceError(f"the {desc} is not in the handover zone: {check['seen']}")

    def _jaw(self, d: Optional[Drawer], xyz: Sequence[float], others: Sequence[Sequence[float]] = ()) -> Optional[Vec]:
        """In a drawer, the calib's pick jaw axis, or the way with more room: the open
        jaws reach ~23 mm each side, and a neighbour in their path is as bad as a wall."""
        if d is None:
            return None
        fx = self.fx
        if fx.pick_jaw is not None:
            return fx.pick_jaw
        a, b = fx.axes()
        c = fx.interior_center(d, self.opening[d.id])
        along, across = fx.extent(d)
        rel = (xyz[0] - c[0], xyz[1] - c[1], 0.0)
        room = {}
        for axis, extent, side in ((a, along, b), (b, across, a)):
            r = extent / 2 - abs(kinematics.dot(rel, axis))
            for o in others:
                off = (o[0] - xyz[0], o[1] - xyz[1], 0.0)
                if abs(kinematics.dot(off, side)) < 0.025:  # pad half width + a neighbour's half width
                    r = min(r, abs(kinematics.dot(off, axis)) - 0.015)
            room[axis] = r
        return a if room[a] >= room[b] else b

    def _pick_up(self, xyz: Vec, surface_z: float, shape: Shape, desc: str, where: str,
                 wall_top: Optional[float] = None, drawer: Optional[Drawer] = None,
                 others: Sequence[Sequence[float]] = ()) -> None:
        """Close look and refine, then down, close, up, and the VLM says whether it is held."""
        s, c = self.s, self.calib
        plane = surface_z + shape.height / 2
        top = max(surface_z + shape.height, wall_top or -math.inf)
        clear = top + 0.02
        before = view = None
        if s.refine:
            self._phase(f"close look: {where}")
            # In a drawer, sim-scene's straight-down view first: the camera-over pose leaves the gripper's
            # lean free, and over r0c1 it leaned the open jaw towards r2c1's handle and stalled (D3 s503).
            options = [list(drawer.topdown_pose_deg)] if drawer is not None and drawer.topdown_pose_deg else []
            over = self._camera_over(xyz, plane, min_tcp_z=clear)
            options += [over] if over is not None else []
            q = None
            for option in options:
                vias, _, why = self._route(self._measured(), option, None, **self._ctx)
                if vias is None:
                    self.log.event("skip", why=f"no close look from here: {why}")
                else:
                    q = option
                    break
            if q is not None:
                self._goto(q, f"close look: {where}", floor=clear)
                found, frame, qa = self._locate(f"pick up the {desc}", ("pick",), f"close-{where}", with_next=False,
                                                describe={"pick": f"the {desc}"})
                before, view = frame.jpeg, qa
                if found["pick"]["visible"]:
                    xyz = self._estimate(found["pick"], qa, plane, "pick", "close")
        # The catalogue's grasp height is sim-scene's, checked physically (a cone on its 8 mm skirt at
        # 19 mm): only keep the pad tips off the surface. With pick-place's 4 mm clearance the TCP sat
        # at 20.5 mm, the pads barely held the skirt, and three cones slid out on the way (D3, 788cb77).
        grasp_z = max(surface_z + shape.grasp_height, surface_z + c.pad_below_tcp + s.pad_floor_clear_m)
        if self.fx.pick_above_m is not None:
            # The calib's height: a top-row drawer's floor is high enough that ours is out of reach.
            above = (xyz[0], xyz[1], max(clear, surface_z + self.fx.pick_above_m))
        else:
            above = (xyz[0], xyz[1], max(clear, grasp_z + 0.03) + s.pregrasp_m / 2)
        jaw = self._jaw(drawer, xyz, others)
        approach = DOWN if jaw is not None else self._approach(above, (xyz[0], xyz[1], grasp_z), "grasp")
        # The fingers go down among this drawer's walls (or to the table) and come up again: the pad-tip
        # rule above holds them off the floor, the capsules are too coarse for it (calib v3).
        with self._near(**self._grasping(drawer)):
            self._phase(f"pre-grasp: {where}")
            # Turn to the grasp over the object at the height the gripper is at, then straight down: from
            # r0c1's straight-down view a joint move to the pre-grasp turns the wrist on the way down and
            # sweeps a finger over r2c1's handle.
            here_z = kinematics.position(c.chain.pose(self._measured(), "tcp"))[2]
            over = None
            if here_z > above[2] + 0.005:
                try:
                    over = self._ik(self._measured(), (above[0], above[1], here_z), approach, jaw)
                except PickPlaceError:
                    over = None
            if over is not None:
                self._goto(over, f"pre-grasp: {where}: over", floor=clear)
                self._line(above, f"pre-grasp: {where}", approach, jaw)
            else:
                self._goto(self._ik(self._measured(), above, approach, jaw), f"pre-grasp: {where}", floor=clear)
            self._phase("descend")
            self._line((xyz[0], xyz[1], grasp_z), "descend", approach, jaw)
            self._gripper(c.gripper_closed, "close")
            self._phase("lift")
            # Lift until the object's bottom (grasp_height under the TCP) clears the walls or the table,
            # by wall_clear_m where the arm reaches that high, else by 6 mm (sim-scene's figure).
            lift = self._reachable_z(xyz, [max(above[2], top + shape.grasp_height + s.wall_clear_m),
                                           max(above[2], top + shape.grasp_height + 0.006)], approach, jaw)
            self._line((xyz[0], xyz[1], lift), "lift", approach, jaw)
            self._lifted = self._measured()
            self._phase("check held")
            if view is not None:
                # Up to the view's height first, then turn: the way down in reverse (see pre-grasp).
                view_z = kinematics.position(c.chain.pose(view, "tcp"))[2]
                if view_z > lift + 0.005:
                    try:
                        self._line((xyz[0], xyz[1], view_z), "lift: to the look", approach, jaw)
                    except PickPlaceError as exc:
                        self.log.event("skip", why=f"no straight lift to the look: {exc}")
                self._goto(view, "look: held", floor=None, speed=s.slow_speed)
                question = grounding.held_question_same_view(desc, self._gripper_side())
                images = [("Before the grasp:", before)]
            else:
                question, images = grounding.held_question(desc), []
            check = self._check(question, images, "held")
            if not check["ok"]:
                self._gripper(c.gripper_open, "open")
                raise NotHeld(f"the {desc} was not held: {check['seen']}")

    def _put_down(self, xyz: Vec, surface_z: float, shape: Shape, where: str, wall_top: Optional[float],
                  clear: float, drawer: Optional[Drawer] = None) -> None:
        """Carry what is held over ``xyz``, lower until its bottom is ``clear`` over the surface, let go.

        Into an opened ``drawer`` it goes by a point in front of the tray at carry height and then
        straight in over the front panel, and leaves the same way: the arm does not reach a top-row
        carry height over the handover mat (D3 s460), and a joint move may dip on the way over the
        panel. sim-scene's D1 oracle rises and descends only outside the open drawer too.
        """
        s, c = self.s, self.calib
        release_z = surface_z + shape.grasp_height + clear
        least = max(release_z, (wall_top or surface_z) + shape.grasp_height + s.wall_clear_m)
        via = None if drawer is None else self._front_of(drawer, xyz)
        carry = self._reachable_z([xyz] + ([via] if via else []), [least + 0.03, least + 0.015, least], DOWN, None)
        self._phase(f"carry: {where}")
        self._rise(carry, f"raise: {where}")
        floor = surface_z + shape.grasp_height + 0.01
        if via is not None:
            self._carry((via[0], via[1], carry), f"carry: {where}: in front", floor)
        else:
            self._carry((xyz[0], xyz[1], carry), f"carry: {where}", floor)
        with self._near(**self._grasping(drawer)):
            if via is not None:
                self._line((xyz[0], xyz[1], carry), f"carry: {where}", DOWN)
            self._phase(f"lower: {where}")
            self._line((xyz[0], xyz[1], release_z), f"lower: {where}", DOWN)
            self._gripper(c.gripper_open, "open")
            self._phase("retreat")
            self._line((xyz[0], xyz[1], carry), "retreat", DOWN)
            if via is not None:
                self._line((via[0], via[1], carry), "retreat: out of the drawer", DOWN)

    def _carry(self, goal: Vec, phase: str, floor: float) -> None:
        """Holding something, to ``goal``: a joint move where it is clear, else a slow straight line, else
        a slow detour by the cabinet look. D3 s501 (e8eadbb): from in front of r2c2 the joint arc to the
        handover mat swung towards the open tray and the full-speed detour shook the cone out."""
        chain, here = self.calib.chain, self._measured()
        floor = min(floor, kinematics.position(chain.pose(here, "tcp"))[2])
        # A joint move first, when it is clear: short (cones held in the e8eadbb pass), where the slow
        # straight line past the base took 12 s and a cone slid out of the pads on the way (788cb77).
        q = self._ik(here, goal, DOWN)
        joint = kinematics.joint_path(here, q, self.s.max_step)
        why = self._path_clear([here] + joint, **self._ctx) or self._dips([here, q], floor)
        if why is None:
            self._send(joint, phase, self.s.speed)
            return
        try:
            line = kinematics.line_path(chain, here, goal, step_m=self.s.line_step_m, max_step=self.s.max_step,
                                        approach=DOWN)
            why_line = self._path_clear([here] + line, **self._ctx) or self._dips([here] + line, floor)
        except IKError as exc:
            line, why_line = [], str(exc)
        if why_line is None:
            self.log.event("carry", phase=phase, how="straight line", why=f"no joint move: {why}")
            self._send(line, phase, self.s.slow_speed)
            return
        self.log.event("carry", phase=phase, how="detour", why=f"no joint move: {why}; no line: {why_line}")
        self._goto(q, phase, floor=floor, speed=self.s.slow_speed)

    def _leave_drawer(self, d: Drawer, shape: Shape) -> None:
        """With an object out of ``d``: up until it clears the walls, then straight out past the front."""
        top = self._wall_top(d) + shape.grasp_height
        here = kinematics.position(self.calib.chain.pose(self._measured(), "tcp"))
        via = self._front_of(d, here)
        z = self._reachable_z([here, via], [top + self.s.wall_clear_m, top + 0.006], DOWN, None)
        self._phase(f"out of {d.id}")
        with self._near(grasp_in=d.id):
            self._rise(z, f"out of {d.id}: up")
            # From higher up (the held check's view) the line falls to z as it goes, never below it.
            try:
                self._line((via[0], via[1], z), f"out of {d.id}", DOWN)
            except PathBlocked as exc:
                # The straight-down view holds J3 near 0, and IK along the line keeps that elbow: the
                # forearm folds onto the gripper. Back to the grasp's own elbow first, over the floor.
                self.log.event("skip", why=f"no straight line out from here: {exc}")
                here = kinematics.position(self.calib.chain.pose(self._measured(), "tcp"))
                up = self._ik(self._lifted or self._measured(), (here[0], here[1], z), DOWN)
                self._goto(up, f"out of {d.id}: over the floor", floor=top + 0.006)
                self._line((via[0], via[1], z), f"out of {d.id}", DOWN)

    def _front_of(self, d: Drawer, p: Sequence[float]) -> Vec:
        """On the line through ``p`` along the pull, clear in front of the opened tray: past its inner
        front wall by the wall (6 mm), the handle sticking out of it, an open finger and 1 cm. Short of
        the handle, the way down to the handover mat ran a finger over r2c0's open handle."""
        fx = self.fx
        a, _ = fx.axes()
        c = fx.interior_center(d, fx.open_m)
        along, _ = fx.extent(d)
        clear = max(0.045, 0.006 + fx.handle_protrusion_m + self._finger_reach + 0.01)
        s = (c[0] - p[0]) * a[0] + (c[1] - p[1]) * a[1] + along / 2 + clear
        return (p[0] + s * a[0], p[1] + s * a[1], p[2] if len(p) > 2 else 0.0)

    def _rise(self, z: float, phase: str) -> None:
        """Straight up where the TCP is, to ``z`` or as near it as the arm reaches there."""
        here = kinematics.position(self.calib.chain.pose(self._measured(), "tcp"))
        if here[2] >= z - 0.005:
            return
        steps = [z - 0.01 * i for i in range(int((z - here[2]) / 0.01) + 1)]
        try:
            top = self._reachable_z([here], steps, DOWN, None)
        except PickPlaceError:
            return
        if top > here[2] + 0.005:
            self._line((here[0], here[1], top), phase, DOWN)

    def _reachable_z(self, points: Sequence[Sequence[float]], heights: Sequence[float],
                     approach: Optional[Vec], jaw: Optional[Vec]) -> float:
        """The first of ``heights`` (best first) the arm reaches over every one of ``points`` (or over
        the one point ``points``) with that grip."""
        pts = [points] if isinstance(points[0], (int, float)) else points
        for z in heights:
            try:
                for p in pts:
                    kinematics.solve_ik(self.calib.chain, self._measured(), (p[0], p[1], z), tool="tcp",
                                        approach=approach if approach is not None else DOWN, jaw=jaw)
                return z
            except IKError:
                continue
        raise PickPlaceError(f"out of reach over {_r(pts[0], 3)} at every height {_r(heights, 3)}")

    # -- staying out of the cabinet, and off the arm itself ------------------------------

    @contextlib.contextmanager
    def _near(self, **ctx: Any) -> Iterator[None]:
        """Within: the moves' checks know what the fingers may come close to (``grasp_in``: that drawer's
        tray; ``table``; ``at_handle``: anything) and which drawer moves with the hand (``moving``)."""
        old = self._ctx
        self._ctx = {**old, **ctx}
        try:
            yield
        finally:
            self._ctx = old

    @staticmethod
    def _grasping(drawer: Optional[Drawer]) -> Dict[str, Any]:
        return {"grasp_in": drawer.id} if drawer is not None else {"table": True}

    def _path_clear(self, path: Sequence[Sequence[float]], skip: Optional[str] = None, at_handle: bool = False,
                    grasp_in: Optional[str] = None, table: bool = False, moving: Optional[str] = None,
                    openings: Optional[Dict[str, float]] = None, points: bool = True,
                    jaw: Any = "now") -> Optional[str]:
        """None if all along ``path`` (its first entry is where the arm starts; checked every
        ``check_step_deg``) the TCP, the pad tips and the camera stay out of the cabinet and of open
        drawers (but ``skip``), the open fingers off the handles, and, with the calib's capsules, every
        link ``robot_margin_m`` off the listed links, the table, the cabinet body, the handles and the
        open drawers' trays. The cabinet counts its closed handles, except on a move ``at_handle``, which
        goes to one; there the fingers may touch anything, the grip is around it. The fingers may also
        come closer to ``grasp_in``'s tray and (``table``) the table, where they grasp and let go; there
        and at a handle the hand bodies keep ``hand_margin_m`` off the scene. The drawer
        ``moving`` goes with the hand (a pull or a push) and is left out. ``openings`` instead of where
        the drawers are now, for a plan, and ``jaw`` (0..1) for the jaw as it will be then. ``points=False`` leaves out the TCP / pad tip / camera / finger
        points, which were made for joint moves: a straight line in (a descent past the cabinet top, say)
        is planned with its own clearances, and the capsules cover it."""
        fx, chain, m = self.fx, self.calib.chain, self.s.body_clear_m
        hulls = self.calib.robot is not None and bool(self.calib.robot.hulls)
        # With the colliders' hulls the brain checks exactly the shared rule (leader 12:05): the point
        # checks below were stand-ins for a robot the calib did not describe.
        points = points and not hulls
        openings = {k: v for k, v in (self.opening if openings is None else openings).items() if k != moving}
        lo, hi = (fx.box_min, fx.box_max) if at_handle else fx.body_box()
        boxes = [("cabinet", lo, hi)]
        for d_id, opening in openings.items():
            if opening > 1e-3 and d_id != skip:
                d = fx.drawers[d_id]
                c = fx.interior_center(d, opening)
                hx, hy = d.interior_size[0] / 2 + 0.005, d.interior_size[1] / 2 + 0.005
                boxes.append((f"open {d_id}", (c[0] - hx, c[1] - hy, d.floor_z - 0.01),
                              (c[0] + hx, c[1] + hy, d.floor_z + d.interior_size[2])))
        tip, half, knobs = self.calib.pad_below_tcp, self.s.finger_half_m, fx.knob_boxes(openings)
        # The leader's rule (2026-11-02), as sim-scene's D0 checks it: the hand bodies keep 2 mm off the
        # scene while they grasp a handle or pick/place in a drawer or on the mat, 5 mm otherwise.
        grasping = at_handle or grasp_in is not None or table
        scene: List[Tuple[str, Sequence[float], Sequence[float]]] = []
        near: Any = "all" if at_handle else ({"table"} if table else set())
        if at_handle and hulls:
            near = {f"handle of {skip}"}  # the pads pinch that handle; the hulls keep 2 mm off the rest
        if self.calib.robot is not None:
            scene = [("table", (-9.0, -9.0, -9.0), (9.0, 9.0, self.calib.table_z)), ("cabinet", fx.box_min, fx.box_max)]
            scene += [(f"handle of {d_id}", k_lo, k_hi) for d_id, k_lo, k_hi in knobs]
            for d_id, opening in openings.items():
                if opening > 1e-3:
                    parts = fx.tray_parts(fx.drawers[d_id], opening)
                    scene += parts
                    if d_id == grasp_in and near != "all":
                        near |= {name for name, _, _ in parts}
        for q in collision.densify(path, self.s.check_step_deg):
            if not points:
                why = self._robot_clear(q, scene, near, grasping, jaw) if scene else None
                if why is not None:
                    return f"{why} at {_r(q, 1)}"
                continue
            t = chain.pose(q, "tcp")
            p, z, x = kinematics.position(t), kinematics.column(t, 2), kinematics.column(t, 0)
            points = [("tcp", p), ("camera", kinematics.position(chain.pose(q, "camera")))]
            fingers = []
            if not at_handle:
                points.append(("pad tips", _add(p, z, tip)))
                # The fingers either side of the open jaw, at both depths, against the handles: D3 s503
                # stalled with the TCP 2 cm off the r1c1/r2c1 handles and a finger on r2c1's.
                fingers = [_add(_add(p, z, depth), x, side) for side in (half, -half) for depth in (0.0, tip)]
            for tool, pt in points:
                for name, b_lo, b_hi in boxes:
                    if _in_box(pt, b_lo, b_hi, m):
                        return f"{tool} at {_r(pt, 3)} inside the {name}"
            for pt in fingers:
                for d_id, k_lo, k_hi in knobs:
                    if _in_box(pt, k_lo, k_hi, m):
                        return f"a finger at {_r(pt, 3)} on the handle of {d_id}"
            if scene:
                why = self._robot_clear(q, scene, near, grasping, jaw)
                if why is not None:
                    return f"{why} at {_r(q, 1)}"
        return None

    def _dips(self, path: Sequence[Sequence[float]], floor: Optional[float]) -> Optional[str]:
        """Why not, if the TCP goes below ``floor`` anywhere along ``path``."""
        if floor is None:
            return None
        chain = self.calib.chain
        low = min(kinematics.position(chain.pose(q, "tcp"))[2] for q in collision.densify(path, self.s.check_step_deg))
        return f"TCP down to z={low:.3f} (floor {floor:.3f})" if low < floor - 1e-3 else None

    def _check_path(self, path: Sequence[Sequence[float]], phase: str, line: bool = False) -> None:
        why = self._path_clear(path, points=not line, **self._ctx)
        if why is not None:
            raise PathBlocked(f"{phase}: the path would put the {why}")

    def _route(self, start: Sequence[float], goal: Sequence[float], floor: Optional[float],
               **near: Any) -> Tuple[Optional[List[List[float]]], str, str]:
        """A joint move from ``start`` to ``goal`` that is clear (:meth:`_path_clear` with ``near``, and
        the TCP over ``floor``): straight if it can, else by way of a look pose, with J1 turned alone
        first or last, or with the arm folded as a look pose while J1 turns; the least joint travel
        first. Returns the poses to pass on the way, how, and why not straight; poses None if no way is
        clear. Without the base in the check, the wrist camera went through it (BOARD 2026-11-02)."""
        s, g = list(start), list(goal)
        looks = [(n, list(v)) for n, v in (("cabinet look", self.fx.cabinet_look_pose_deg),
                                           ("table look", self.calib.look_pose_deg)) if v]
        ways: List[Tuple[str, List[List[float]]]] = [("straight", [])]
        ways += [(f"by the {n}", [v]) for n, v in looks]
        ways += [("J1 last", [[s[0], *g[1:]]]), ("J1 first", [[g[0], *s[1:]]])]
        ways += [(f"J1 turned folded as the {n}", [[s[0], *v[1:]], [g[0], *v[1:]]]) for n, v in looks]

        def travel(vias: Sequence[Sequence[float]]) -> float:
            pts = [s, *vias, g]
            return sum(max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(pts, pts[1:]))

        straight = ""
        for how, vias in sorted(ways, key=lambda w: travel(w[1])):
            pts = [s, *vias, g]
            why = self._path_clear(pts, **near) or self._dips(pts, floor)
            if why is None:
                return vias, how, straight
            if not vias:
                straight = why
        return None, "", straight

    def _goto(self, goal: Sequence[float], phase: str, floor: Optional[float], speed: Optional[int] = None) -> None:
        """As pick-place's, but round what is in the way (:meth:`_route`), and the floor is no higher than
        where the path starts."""
        here = self._measured()
        if floor is not None:
            # sim-scene's tilted looks into r1c2 and r2c2 hold the TCP beside the drawer below the
            # floor for above it (wall top + 2 cm): the path may not go lower than it starts (D3 s451).
            floor = min(floor, kinematics.position(self.calib.chain.pose(here, "tcp"))[2])
        vias, how, why = self._route(here, goal, floor, **self._ctx)
        if vias is None:
            raise PathBlocked(f"{phase}: no clear way: straight, the path would put the {why}")
        if vias:
            self.log.event("detour", phase=phase, how=how, why=why)
        for i, q in enumerate([*vias, list(goal)]):
            last = i == len(vias)
            self._send(kinematics.joint_path(here if i == 0 else self._measured(), q, self.s.max_step),
                       phase if last else f"{phase}: detour", speed or self.s.speed)
