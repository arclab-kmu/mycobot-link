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
from mycobot_brain.fixture import Drawer, Fixture, HandleStage, Shape, TaughtLeg
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
    plan_buffer_m: float = 0.001  # extra hull clearance asked of a put planned before it is made
    #: a put spot's footprint (catalogue radius, a cube's half diagonal) this far from every other one in the
    #: drawer, for the estimates' error; and the opened pads this far off the walls and the other objects
    #: at the release, at whichever jaw yaw fits (leader 18:5x, after D3 v3.3 s461)
    footprint_gap_m: float = 0.003
    pad_clear_m: float = 0.002


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
        #: the waypoints from the lift up to the held check's view, to go back down the same way
        self._back: Optional[List[List[float]]] = None
        #: sim-scene's verified routes between the views (calib v3.2), played instead of planning
        self._routes: List[Tuple[str, TaughtLeg]] = [(f"{d.id} {name}", leg) for d in self.fx.drawers.values()
                                                     if d.taught for name, leg in d.taught.routes.items()]
        #: objects standing where the brain knows they are - let go of, or seen waiting on the mat for a put:
        #: (name, x, y, support z, shape, drawer id or None, that drawer's opening then). Every move after
        #: is checked against them (:meth:`_known_boxes`); D3 v3.2 s502_e: r0c0's taught close and the move
        #: off it tipped the cone just handed over (leader 16:58).
        self._known: List[Tuple[str, float, float, float, Shape, Optional[str], float]] = []
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
        fk = self.taught_fk_mm()
        if fk is not None:
            self.log.event("taught", drawers=sorted(d for d, x in self.fx.drawers.items() if x.taught),
                           routes=len(self._routes), fk_vs_calib_tcp_max_mm=round(fk, 3))
            # Played as given, so only flagged: the taught legs that reach over the handover mat.
            self.log.event("guard", what="calib", flagged=self._guard_calib())
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
            if d.taught:
                reach[d.id] = 0.0  # played as sim-scene taught it
                continue
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
            if self.fx.drawers[d].taught:
                reachable.append(d)  # sim-scene's verified motions open it
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
        # It waits on the mat while the drawer opens: the same guard as for one let go (leader 16:58).
        waiting = self._leave(shape, (start[0], start[1], c.table_z), None, how="waiting")
        look = self._open_and_look(d, None)
        occupied: List[Tuple[Vec, Shape]] = []
        for o in look["objects"]:
            p = self._object_xyz(d, o, look["angles"])
            if p is None:
                # Not where it is, so not how much room it leaves: nothing goes in (leader 18:5x).
                self._close(d)
                return {"ok": False, "why": f"no room for a {desc} in {d.id}: could not place the "
                                            f"{o['colour']} {o['shape']} in it"}
            occupied.append((p, self._shape(o["shape"]) if o["shape"] in self.fx.shapes else self._widest()))
        spot, margin, put_jaw, why = self._put_spot(d, occupied, shape, start)
        if spot is None:
            # A clean refusal: the object stays on the mat, nothing enters the drawer (leader 18:5x).
            self._close(d)
            return {"ok": False, "why": f"no room for a {desc} in {d.id}: {why}"}
        self.log.event("spot", drawer=d.id, xyz=_r(spot, 4), occupied=[_r(p, 4) for p, _ in occupied],
                       reach_margin_m=round(margin, 3), jaw=_r(put_jaw, 3) if put_jaw else None)
        self._forget(waiting)  # the gripper goes to it now
        for attempt in range(1, self.s.attempts + 1):
            try:
                self._pick_up(start, c.table_z, shape, desc, "table", jaw=self._put_yaw(start, spot, shape, put_jaw))
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
                       clear=self.s.drawer_clear_m, drawer=d, put_jaw=put_jaw)
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
        if d.taught is not None:
            full = (abs(start) < 2e-3 and abs(end - self.fx.open_m) < 2e-3) or \
                (abs(start - self.fx.open_m) < 2e-3 and abs(end) < 2e-3)
            if full:
                self._taught_handle(d, verb)
                return
            self.log.event("skip", why=f"{verb} {d.id} from {start:.3f} to {end:.3f} m: not a taught sequence, planned")
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
            self._gripper(stage.close if stage.close is not None else c.gripper_closed, "close", holds=False)
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

    def _taught_handle(self, d: Drawer, verb: str) -> None:
        """Open or shut ``d`` along sim-scene's verified waypoints (calib v3.2): per segment, the gripper value
        it starts with, then its waypoints - to the standoff, onto the handle, pull (or push), back off, and
        back to the cabinet look."""
        t = d.taught
        assert t is not None
        segs = t.open if verb == "open" else t.close
        n = len({seg.stage for seg in segs if seg.stage is not None})
        self._phase(f"{verb} {d.id}")
        for i, seg in enumerate(segs):
            tag = f"{verb} {d.id}" + (f" ({seg.stage + 1}/{n})" if n > 1 and seg.stage is not None else "")
            if seg.gripper is not None:
                self._gripper(seg.gripper, "close" if seg.phase == "pull" else "open", holds=False)
            label, ctx = {"standoff": (f"{tag}: pre-grasp", {}),
                          "approach": (f"{tag}: to handle", dict(skip=d.id, at_handle=True)),
                          "pull": (tag, dict(skip=d.id, at_handle=True, moving=d.id)),
                          "back_off": (f"{tag}: back off", dict(skip=d.id, at_handle=True)),
                          "return": (f"{verb} {d.id}: cabinet look", {})}[seg.phase]
            with self._near(**ctx):
                self._play(seg.leg, label, speed=self.s.handle_speed if seg.phase == "pull" else None, first=i == 0)
            if seg.opening_after is not None:
                self.opening[d.id] = seg.opening_after

    def _play(self, leg: TaughtLeg, phase: str, speed: Optional[int] = None, first: bool = False) -> None:
        """Send a taught leg as it is. More than 3 deg from where it starts (5 deg from its first waypoint,
        when the start is not given), go there first by a checked move. The brain's checker looks at it
        too, but only to log (leader 2026-11-02 12:24)."""
        here = self._measured()
        start, tol = (leg.start, 3.0) if leg.start is not None else (leg.q[0], 5.0)
        if max(abs(a - b) for a, b in zip(here, start)) > tol:
            # Not low over the table, where an object may stand (the handover mat). Taught routes count: a
            # close after a put starts from the tilted view, and v3.2's way from there to the cabinet look
            # is round by the top-down view (D3 v3.2 D0, put r1c2).
            self._goto(list(start), f"{phase}: to the taught start", floor=self.calib.table_z + self.s.leave_mat_m
                       if first else None, end_tol=tol - 0.5)
            here = self._measured()
        path = [list(q) for q in leg.q]
        why = self._path_clear([here] + path, points=False, **self._ctx)
        if why is not None:
            self.log.event("taught_check", phase=phase, why=why)
        # On its own as well: the scene check above stops at its first finding, which may hide this one.
        hit = self._guard_path([here] + path, self._ctx.get("moving"), self._ctx.get("released"))
        if hit is not None:
            self.log.event("guard", what="taught", phase=phase, why=hit)
        self._send(path, phase, leg.speed or speed or self.s.speed)

    def _taught_route(self, here: Sequence[float], goal: Sequence[float],
                      end_tol: float = 0.5) -> Optional[List[Tuple[str, TaughtLeg]]]:
        """Taught routes, end to end (up to 3), from where the arm is (3 deg) to ``goal`` (``end_tol``):
        v3.2's routes go one way round, cabinet look -> tilted -> top-down -> cabinet look."""
        def near(a: Sequence[float], b: Sequence[float], tol: float) -> bool:
            return max(abs(x - y) for x, y in zip(a, b)) <= tol

        def starts(leg: TaughtLeg) -> Tuple[Tuple[float, ...], float]:
            # Where it begins; a first waypoint is up to 2 deg past the start.
            return (leg.start, 0.5) if leg.start is not None else (leg.q[0], 2.5)

        paths: List[List[Tuple[str, TaughtLeg]]] = [[r] for r in self._routes
                                                    if near(here, starts(r[1])[0], 3.0 + starts(r[1])[1] - 0.5)]
        for _ in range(3):
            for path in paths:
                if near(goal, path[-1][1].q[-1], end_tol):
                    return path
            paths = [path + [r] for path in paths for r in self._routes
                     if r not in path and near(path[-1][1].q[-1], starts(r[1])[0], starts(r[1])[1])]
        return None

    def taught_fk_mm(self) -> Optional[float]:
        """Largest distance, over every taught waypoint that has sim-scene's TCP, between that and this
        chain's FK at the same angles (the two sides agree on the arm); None without any."""
        worst = None
        chain = self.calib.chain
        for d in self.fx.drawers.values():
            if not d.taught:
                continue
            for _, leg in d.taught.legs():
                for q, tcp in zip(leg.q, leg.tcp or ()):
                    e = math.dist(kinematics.position(chain.pose(q, "tcp")), tcp) * 1000
                    worst = e if worst is None else max(worst, e)
        return worst

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
                 others: Sequence[Sequence[float]] = (), jaw: Optional[Vec] = None) -> None:
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
                if self._taught_route(self._measured(), option) is not None:
                    q = option
                    break
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
        jaw = jaw if jaw is not None else self._jaw(drawer, xyz, others)
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
            held = self._jaw_for_width(shape.width)
            if held is not None:
                self._jaw_open = held  # the pads stop on the object, not at the shut setpoint
            self._phase("lift")
            # Lift until the object's bottom (grasp_height under the TCP) clears the walls or the table,
            # by wall_clear_m where the arm reaches that high, else by 6 mm (sim-scene's figure).
            lift = self._reachable_z(xyz, [max(above[2], top + shape.grasp_height + s.wall_clear_m),
                                           max(above[2], top + shape.grasp_height + 0.006)], approach, jaw)
            self._line((xyz[0], xyz[1], lift), "lift", approach, jaw)
            self._lifted = self._measured()
            self._trail = [list(self._lifted)]  # the way up to the held check's view, to retrace
            self._phase("check held")
            if view is not None:
                # Checked from right here, the lift pose: no rise and fall, no roll, with the object in hand.
                # A cone crept out of the pads coming back down from the held check's height (D3 v3.2 s502_c,
                # 3c40c7a; leader 14:26), and before that when the close look's pose rolled the wrist.
                question = grounding.held_question_from_above(desc, self._gripper_side())
                images = [("Before the grasp:", before)]
            else:
                question, images = grounding.held_question(desc), []
            self._back, self._trail = self._trail, None
            check = self._check(question, images, "held")
            if not check["ok"]:
                self._gripper(c.gripper_open, "open")
                raise NotHeld(f"the {desc} was not held: {check['seen']}")

    def _put_down(self, xyz: Vec, surface_z: float, shape: Shape, where: str, wall_top: Optional[float],
                  clear: float, drawer: Optional[Drawer] = None, put_jaw: Optional[Vec] = None) -> None:
        """Carry what is held over ``xyz``, lower until its bottom is ``clear`` over the surface, let go.

        Into an opened ``drawer`` it goes by a point in front of the tray at carry height and then
        straight in over the front panel, and leaves the same way: the arm does not reach a top-row
        carry height over the handover mat (D3 s460), and a joint move may dip on the way over the
        panel. sim-scene's D1 oracle rises and descends only outside the open drawer too.
        """
        s, c = self.s, self.calib
        release_z = surface_z + shape.grasp_height + clear
        least = max(release_z, (wall_top or surface_z) + shape.grasp_height + s.wall_clear_m)
        via = None if drawer is None else self._front_of(drawer, xyz, shape.width)
        carry = self._reachable_z([xyz] + ([via] if via else []), [least + 0.03, least + 0.015, least], DOWN, None)
        self._phase(f"carry: {where}")
        self._rise(carry, f"raise: {where}")
        floor = surface_z + shape.grasp_height + 0.01
        if via is not None:
            self._carry((via[0], via[1], carry), f"carry: {where}: in front", floor)
        else:
            self._carry((xyz[0], xyz[1], carry), f"carry: {where}", floor)
        # Into a drawer with the calib's pick jaw (sim-scene: along x, so the pads and the gripper body stay
        # between the tray's front and back walls), turned to in front of the tray; the yaw left over from
        # the table pick put the gripper body by the handles above (D3 v3.2 D0, put r0c0).
        jaw = None if drawer is None else (put_jaw if put_jaw is not None else self._jaw(drawer, xyz))
        off = 0.0
        if jaw is not None:
            # Gripped on the mat at the yaw that lands here (:meth:`_put_yaw`): if over the spot, J6 held,
            # the jaw is within 10 deg of the pick jaw, go on as it is; else turn, slowly, the one wrist roll
            # with an object in hand (logged, counted by D0).
            try:
                there = kinematics.solve_ik(c.chain, self._measured(), (xyz[0], xyz[1], carry), approach=DOWN,
                                            hold=(6,))
                x = kinematics.column(c.chain.pose(there, "tcp"), 0)
                off = math.degrees(math.acos(min(1.0, abs(kinematics.dot(x, jaw)))))
            except IKError:
                off = 90.0
            if off <= 10.0:
                jaw = None
        with self._near(**self._grasping(drawer)):
            if via is not None:
                if jaw is not None:
                    self.log.event("slow_turn", where=where, deg=round(off, 1))
                    self._line((via[0], via[1], carry), f"carry: {where}: turn", DOWN, jaw, speed=5)
                self._line((xyz[0], xyz[1], carry), f"carry: {where}", DOWN, jaw)
            self._phase(f"lower: {where}")
            self._line((xyz[0], xyz[1], release_z), f"lower: {where}", DOWN, jaw)
            self._gripper(c.gripper_open, "open")
            # From here every move keeps off it; the open fingers only on the way back up, beside it.
            left = self._leave(shape, (xyz[0], xyz[1], surface_z), drawer)
            self._phase("retreat")
            with self._near(released=left):
                self._line((xyz[0], xyz[1], carry), "retreat", DOWN, jaw)
                if via is not None:
                    self._line((via[0], via[1], carry), "retreat: out of the drawer", DOWN, jaw)

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
        if why is not None:
            # Far round the base (v3.3's mat behind it), the solution nearest the seed can be the arm leant
            # back over the base, and the joint move there folds the forearm onto the fingers; from J1 turned
            # to the goal's bearing first it is the ordinary one (D0 v3.3, put r0c1 beside pair0).
            hold = self._held(None)
            for seed in kinematics.bearing_seeds(chain, here, goal, hold=hold):
                try:
                    q2 = kinematics.solve_ik(chain, seed, goal, approach=DOWN, hold=hold)
                except IKError:
                    continue
                path2 = kinematics.joint_path(here, q2, self.s.max_step)
                if (self._path_clear([here] + path2, **self._ctx) or self._dips([here, q2], floor)) is None:
                    self.log.event("carry", phase=phase, how="joint move, J1 turned first in the IK",
                                   why=f"nearest solution: {why}")
                    q, joint, why = q2, path2, None
                    break
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

    def _put_heights(self, surface_z: float, shape: Shape, wall_top: Optional[float], clear: float) -> Tuple[float, float]:
        """Where a put lets go (TCP z) and the lowest carry height over it, as :meth:`_put_down` has them."""
        release_z = surface_z + shape.grasp_height + clear
        return release_z, max(release_z, (wall_top or surface_z) + shape.grasp_height + self.s.wall_clear_m)

    def _reaches(self, p: Sequence[float], z: float) -> bool:
        """The TCP straight down at (``p``, ``z``) is within reach (J6 does not matter: the TCP is on its axis)."""
        chain = self.calib.chain
        for seed in (self._measured(), self.fx.cabinet_look_pose_deg, self.calib.look_pose_deg):
            if not seed:
                continue
            try:
                kinematics.solve_ik(chain, seed, (p[0], p[1], z), approach=DOWN)
                return True
            except IKError:
                continue
        return False

    def _jaw_for_width(self, width: float) -> Optional[float]:
        """How far open (0..1) the jaw is when it stops on something ``width`` wide."""
        g = self.calib.jaw_gap
        if not g or g[1] <= g[0]:
            return None
        return min(max((width - g[0]) / (g[1] - g[0]), 0.0), 1.0)

    def _widest(self) -> Shape:
        """The catalogue's widest shape: for an object the VLM names outside it."""
        return max(self.fx.shapes.values(), key=lambda x: x.width)

    def _put_spot(self, d: Drawer, occupied: Sequence[Tuple[Sequence[float], Shape]], shape: Shape,
                  start: Optional[Sequence[float]] = None) -> Tuple[Optional[Vec], float, Optional[Vec], str]:
        """Where to put an object in ``d``, beside ``occupied`` (where each object there is, and its shape):
        a floor point whose footprint (catalogue radius; a cube's half diagonal) keeps ``footprint_gap_m``
        from every other one, where the opened pads keep ``pad_clear_m`` off the walls and the other objects
        at some jaw yaw (the calib's pick jaw first, then 15 deg steps either way to 90), that the arm
        reaches at the carry height, over it and in front of the tray, and down to where it lets go, and
        whose whole put (in over the front panel, down, let go, up, out) passes the checker as it will be
        made. The most room first (5 mm steps), then the jaw least turned from the pick jaw, then reach to
        spare above the carry height, then the middle.
        Returns (spot, reach to spare, jaw, why not); spot None if none qualifies - a clean refusal (leader
        18:5x: D3 v3.3 s461, a cone set 21 mm from a sphere's centre in a 72 mm tray leant on it at 45 deg)."""
        release_z, least = self._put_heights(d.floor_z, shape, self._wall_top(d), self.s.drawer_clear_m)
        r_new = shape.width / 2
        others = [(f"{o.name} at {_r(q, 3)}", (q[0], q[1], d.floor_z), (q[0], q[1], d.floor_z + o.height),
                   o.width / 2) for q, o in occupied]
        bins: Dict[int, List[Tuple[Vec, float]]] = {}
        why = {}
        for p, _, mid in self.fx.free_spots(d, self.fx.open_m, [q for q, _ in occupied], shape.width, grid=9):
            room = min((math.hypot(p[0] - q[0], p[1] - q[1]) - r_new - o.width / 2 for q, o in occupied), default=1.0)
            if room < self.s.footprint_gap_m:
                why.setdefault("room", f"every spot within {self.s.footprint_gap_m * 1000:.0f} mm of another object")
                continue
            bins.setdefault(int(min(room, 1.0) / 0.005), []).append((p, mid))
        for b in sorted(bins, reverse=True):
            best: Optional[Tuple[Tuple[float, float, float], Vec, Optional[Vec]]] = None
            for p, mid in bins[b]:
                jaw, turn, pads = self._release_jaw(d, p, release_z, others)
                if pads is not None:
                    why.setdefault("pads", f"the opened pads: {pads}")
                    continue
                if best is not None and -turn < best[0][0]:
                    continue
                via = self._front_of(d, p, shape.width)
                if not all(self._reaches(q, z) for q, z in ((p, least), (via, least), (p, (least + release_z) / 2),
                                                             (p, release_z))):
                    why.setdefault("reach", "out of reach at the carry height")
                    continue
                spare = 0.0
                for dz in (0.01, 0.02, 0.03, 0.04, 0.05):
                    if self._reaches(p, least + dz) and self._reaches(via, least + dz):
                        spare = dz
                    else:
                        break
                rank = (-turn, spare, -mid)
                if best is not None and rank <= best[0]:
                    continue
                if start is not None:
                    blocked = self._put_blocked(d, p, via, shape, start, least, release_z, jaw)
                    if blocked:
                        why["plan"] = f"the put would be refused: {blocked}"
                        continue
                best = (rank, p, jaw)
            if best is not None:
                return best[1], best[0][1], best[2], ""
        words = "; ".join(why.values()) or "no floor left for it"
        self.log.event("skip", why=f"no put spot in {d.id}: {words}")
        return None, 0.0, None, words

    def _release_jaw(self, d: Drawer, p: Sequence[float], release_z: float,
                     others: Sequence[Tuple[str, Vec, Vec]]) -> Tuple[Optional[Vec], float, Optional[str]]:
        """The jaw to let go at ``p`` with: the calib's pick jaw, else turned 15 deg at a time either way up to
        90, the first whose opened pads keep ``pad_clear_m`` off ``d``'s walls and ``others``. (jaw, degrees
        turned, None), or (None, 0, why not at the pick jaw)."""
        base = self.fx.pick_jaw or (1.0, 0.0, 0.0)
        first = None
        for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6):
            t = math.radians(15 * k)
            jaw = (base[0] * math.cos(t) - base[1] * math.sin(t), base[0] * math.sin(t) + base[1] * math.cos(t), 0.0)
            why = self._release_clear(d, p, jaw, release_z, others)
            if why is None:
                return jaw, abs(15.0 * k), None
            first = first or why
        return None, 0.0, first

    def _release_clear(self, d: Drawer, p: Sequence[float], jaw: Vec, release_z: float,
                       others: Sequence[Tuple[str, Vec, Vec]]) -> Optional[str]:
        """Why not, if letting go at ``p`` with ``jaw`` puts an opened pad within ``pad_clear_m`` of ``d``'s
        walls (the floor is the margin-free contact rule's) or of ``others``."""
        chain, robot = self.calib.chain, self.calib.robot
        walls = [w for w in self.fx.tray_parts(d, self.fx.open_m) if not w[0].endswith(" floor")]
        boxes = walls + list(others)
        if robot is not None and robot.hulls:
            try:
                q = kinematics.solve_ik(chain, self.fx.cabinet_look_pose_deg, (p[0], p[1], release_z), approach=DOWN,
                                        jaw=jaw)
            except IKError as exc:
                return f"IK: {exc}"
            hit = robot.hull_gaps(chain, q, boxes, self.s.pad_clear_m, lambda name, box: name not in robot.fingers,
                                  jaw=self._jaw_at(self.calib.gripper_open))
            return None if hit is None else f"{hit[0]} {hit[2] * 1000:.1f} mm from the {hit[1]}"
        for side in (1.0, -1.0):  # without hulls: a finger the open reach out along the jaw
            f = (p[0] + side * self._finger_reach * jaw[0], p[1] + side * self._finger_reach * jaw[1], release_z)
            for entry in boxes:
                name, lo, hi = entry[0], entry[1], entry[2]
                rnd = float(entry[3]) if len(entry) > 3 else 0.0  # type: ignore[misc]
                if _in_box(f, lo, hi, self.s.pad_clear_m + rnd):
                    return f"a finger in the {name}"
        return None

    def _put_blocked(self, d: Drawer, spot: Vec, via: Vec, shape: Shape, start: Sequence[float], least: float,
                     release_z: float, put_jaw: Optional[Vec] = None) -> Optional[str]:
        """Why the put at ``spot`` would be refused, planned as :meth:`_put_down` will make it after the grasp
        on the table (:meth:`_put_yaw`): in at the carry height, down, let go, up, out, the jaw ``put_jaw``
        (default the calib's pick jaw) at the spot; None if it passes."""
        chain, c = self.calib.chain, self.calib
        carry = next((z for z in (least + 0.03, least + 0.015, least)
                      if self._reaches(spot, z) and self._reaches(via, z)), least)
        jaw_mat = self._put_yaw(start, spot, shape, put_jaw, log=False)
        try:
            q_mat = kinematics.solve_ik(chain, c.look_pose_deg, (start[0], start[1], c.table_z + shape.grasp_height),
                                        approach=DOWN, jaw=jaw_mat)
            q_via = kinematics.solve_ik(chain, q_mat, (via[0], via[1], carry), approach=DOWN, hold=(6,))
            # As _put_down: J6 held if the jaw lands within 10 deg of the put jaw, else turned to it.
            pick = put_jaw if put_jaw is not None else self._jaw(d, spot)
            jaw, hold = None, (6,)
            if pick is not None:
                there = kinematics.solve_ik(chain, q_via, (spot[0], spot[1], carry), approach=DOWN, hold=(6,))
                x = kinematics.column(chain.pose(there, "tcp"), 0)
                if math.degrees(math.acos(min(1.0, abs(kinematics.dot(x, pick))))) > 10.0:
                    jaw, hold = pick, ()
                    q_via = kinematics.solve_ik(chain, q_via, (via[0], via[1], carry), approach=DOWN, jaw=pick)
            kw = dict(step_m=self.s.line_step_m, max_step=self.s.max_step, approach=DOWN, jaw=jaw, hold=hold)
            inn = kinematics.line_path(chain, q_via, (spot[0], spot[1], carry), **kw)
            down = kinematics.line_path(chain, inn[-1], (spot[0], spot[1], release_z), **kw)
            up = kinematics.line_path(chain, down[-1], (spot[0], spot[1], carry), **kw)
            out = kinematics.line_path(chain, up[-1], (via[0], via[1], carry), **kw)
        except IKError as exc:
            return f"IK: {exc}"
        held, opened = self._jaw_for_width(shape.width), self._jaw_at(c.gripper_open)
        # 1 mm more than the run itself is held to: the grasp's J6 and the spot drift a little from the
        # prediction (D0, put r2c0 cube between two: 2.0 mm predicted, 1.9 run).
        self._plan_buffer = self.s.plan_buffer_m
        try:
            for name, path, jaw_open in (("in", [q_via] + inn, held), ("down", [inn[-1]] + down, held),
                                         ("up", [down[-1]] + up, opened), ("out", [up[-1]] + out, opened)):
                why = self._path_clear(path, grasp_in=d.id, points=False, jaw=jaw_open)
                if why is not None:
                    return f"{name}: {why}"
        finally:
            self._plan_buffer = 0.0
        return None

    def _put_yaw(self, start: Sequence[float], spot: Sequence[float], shape: Shape, put_jaw: Optional[Vec] = None,
                 log: bool = True) -> Optional[Vec]:
        """The jaw to grip with on the table so that, carried with J6 held (no wrist roll with the object in
        hand), it arrives over ``spot`` along ``put_jaw`` (default the calib's pick jaw): tried every 15 deg.
        None: no such jaw, or nothing lands within 10 deg (the put then turns, slowly, in front of the tray)."""
        fx, chain = self.fx, self.calib.chain
        target_jaw = put_jaw if put_jaw is not None else fx.pick_jaw
        if target_jaw is None:
            return None
        grasp = (start[0], start[1], self.calib.table_z + shape.grasp_height)
        target = (spot[0], spot[1], spot[2] + shape.grasp_height + 0.03)
        best: Tuple[float, Optional[Vec]] = (math.inf, None)
        for k in range(12):
            a = math.radians(15 * k)
            j = (math.cos(a), math.sin(a), 0.0)
            try:
                q = kinematics.solve_ik(chain, self.calib.look_pose_deg, grasp, approach=DOWN, jaw=j)
                there = kinematics.solve_ik(chain, q, target, approach=DOWN, hold=(6,))
            except IKError:
                continue
            x = kinematics.column(chain.pose(there, "tcp"), 0)
            err = math.degrees(math.acos(min(1.0, abs(kinematics.dot(x, target_jaw)))))
            if err < best[0]:
                best = (err, j)
        if log:
            self.log.event("put_yaw", deg_off_pick_jaw=None if best[1] is None else round(best[0], 1),
                           jaw=None if best[1] is None else _r(best[1], 3))
        return best[1] if best[0] <= 10.0 else None

    def _leave_drawer(self, d: Drawer, shape: Shape) -> None:
        """With an object out of ``d``: back down to where the grasp lifted, the way the arm came up to the
        held check's view; up until it clears the walls; then straight out past the front.

        D3 v3.2 run 1 (s502, f0f49be): from r0c0's top-down view, hanging over the cabinet edge, the
        straight line out took an elbow branch that folded the forearm onto the gripper, and the joint
        move down to the line's height put the gripper 0.7 mm from the cabinet face. The way up was
        checked on the way up."""
        back, self._back = self._back, None
        top = self._wall_top(d) + shape.grasp_height
        if back and len(back) > 1 and max(abs(a - b) for a, b in zip(self._measured(), back[-1])) <= 3.0:
            lift_z = kinematics.position(self.calib.chain.pose(back[0], "tcp"))[2]
            with self._near(grasp_in=d.id):
                # One joint move back down to the lift pose (leader 13:46: smooth while holding); else the
                # way up, reversed.
                if not self._held_joint(back[0], f"out of {d.id}: back to the lift", floor=min(lift_z, top + 0.006)):
                    path = [list(q) for q in reversed(back[:-1])]
                    self._check_path([self._measured()] + path, f"out of {d.id}: back to the lift")
                    self._send(path, f"out of {d.id}: back to the lift", self.s.slow_speed)
        here = kinematics.position(self.calib.chain.pose(self._measured(), "tcp"))
        via = self._front_of(d, here, shape.width)
        z = self._reachable_z([here, via], [top + self.s.wall_clear_m, top + 0.006], DOWN, None)
        self._phase(f"out of {d.id}")
        with self._near(grasp_in=d.id):
            # One joint move out in front of the tray at this height, the object over the walls and the
            # front panel all the way (leader 13:46: smooth moves while holding); else the line below.
            # Low in front of r0c1 that move folds the arm against itself (joint3 on joint5): higher, as
            # far as the arm reaches, before the line.
            for dz in (0.0, 0.03, 0.06, 0.09):
                try:
                    q_via = self._ik(self._measured(), (via[0], via[1], max(z, here[2]) + dz), DOWN)
                except PickPlaceError:
                    continue
                if self._held_joint(q_via, f"out of {d.id}", floor=top + 0.006):
                    return
            # Low in front of a bottom-row tray the line out folds the arm against itself (r0c1: joint3 on
            # joint5, D3 v3.2 D0): higher, if the arm reaches, before giving up on the line.
            for dz in (0.03, 0.06, 0.09):
                if self._line_clear((via[0], via[1], z)):
                    break
                try:
                    z_up = self._reachable_z([here, via], [z + dz], DOWN, None)
                except PickPlaceError:
                    break
                self.log.event("skip", why=f"out of {d.id}: no clear line at z={z:.3f}; {z_up:.3f}")
                z = z_up
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

    def _line_clear(self, goal: Sequence[float]) -> bool:
        """Whether straight up from here to ``goal``'s height, then the line to ``goal`` (DOWN, jaw free),
        passes the checker as the moves would, in the current context."""
        chain, here = self.calib.chain, self._measured()
        p = kinematics.position(chain.pose(here, "tcp"))
        try:
            up = kinematics.line_path(chain, here, (p[0], p[1], max(p[2], goal[2])), step_m=self.s.line_step_m,
                                      max_step=self.s.max_step, approach=DOWN) if goal[2] > p[2] + 0.005 else []
            start = up[-1] if up else here
            out = kinematics.line_path(chain, start, goal, step_m=self.s.line_step_m, max_step=self.s.max_step,
                                       approach=DOWN)
        except IKError:
            return False
        return self._path_clear([here] + up + out, points=False, **self._ctx) is None

    def _front_of(self, d: Drawer, p: Sequence[float], width: float = 0.0) -> Vec:
        """On the line through ``p`` along the pull, clear in front of the opened tray for something
        ``width`` wide held there: past its inner front wall by the wall (6 mm), the handle sticking out
        of it, and 1 cm, plus half the object; without the arm's hulls in the calib, an open finger
        instead (short of the handle, the way down to the handover mat ran a finger over r2c0's open
        handle). With the hulls the brain checks the fingers itself, and the finger's reach put this
        point for r0c0 by the base, where the line there folded the arm (D3 v3.2 D0, take r0c0)."""
        fx = self.fx
        a, _ = fx.axes()
        c = fx.interior_center(d, fx.open_m)
        along, _ = fx.extent(d)
        hulls = self.calib.robot is not None and bool(self.calib.robot.hulls)
        reach = width / 2 if hulls else self._finger_reach
        clear = max(0.045, 0.006 + fx.handle_protrusion_m + reach + 0.01)
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
                    jaw: Any = "now", released: Optional[str] = None) -> Optional[str]:
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
        # Every link keeps object_margin_m off the objects the brain knows stand somewhere, the fingers
        # excepted from the one just ``released``, which they are still beside.
        objects = self._known_boxes(self.opening if openings is None else openings, moving)
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
        # The one thing the pads may touch: the handle they grip (without hulls, anything at a handle).
        pinched: Any = (near if near == "all" else set(near)) if at_handle else set()
        if self.calib.robot is not None:
            scene = [("table", (-9.0, -9.0, -9.0), (9.0, 9.0, self.calib.table_z)), ("cabinet", fx.box_min, fx.box_max)]
            scene += [(f"handle of {d_id}", k_lo, k_hi) for d_id, k_lo, k_hi in knobs]
            for d_id, opening in openings.items():
                if opening > 1e-3:
                    parts = fx.tray_parts(fx.drawers[d_id], opening)
                    scene += parts
                    if d_id == grasp_in and near != "all":
                        near |= {name for name, _, _ in parts}
            if released is not None and near != "all":
                near = set(near) | {released}
        for q in collision.densify(path, self.s.check_step_deg):
            if not points:
                why = self._robot_clear(q, scene, near, grasping, jaw, objects, pinched) if scene else None
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
                why = self._robot_clear(q, scene, near, grasping, jaw, objects, pinched)
                if why is not None:
                    return f"{why} at {_r(q, 1)}"
        return None

    # -- what the brain has left standing ------------------------------------------------

    def _leave(self, shape: Shape, at: Sequence[float], drawer: Optional[Drawer], how: str = "set-down") -> str:
        """Note that an object of ``shape`` stands at ``at`` (x, y, support z), in ``drawer`` as it is open
        now or on the table; returns its name, as the checker reports it."""
        where = f"in {drawer.id}" if drawer is not None else "on the mat"
        name = f"{how} {shape.name} {where}"
        self._known.append((name, float(at[0]), float(at[1]), float(at[2]), shape,
                            drawer.id if drawer is not None else None,
                            self.opening[drawer.id] if drawer is not None else 0.0))
        self.log.event("guard", what=how, object=name, xyz=_r(at, 4))
        return name

    def _forget(self, name: str) -> None:
        self._known = [k for k in self._known if k[0] != name]

    def _known_boxes(self, openings: Dict[str, float], moving: Optional[str] = None
                     ) -> List[Tuple[str, Vec, Vec]]:
        """Each known object as an upright cylinder of its catalogue radius (a cube's half diagonal), from
        its support to its top: (name, axis bottom, axis top, radius). One in a drawer goes where that drawer
        is in ``openings``, and one in the drawer ``moving`` with the hand is left out (it goes with it). An
        axis-aligned square was too coarse: its corners met the opened pads beside a cube (D0 v3.3)."""
        a = self.fx.pull_axis
        out = []
        for name, x, y, z, shape, d_id, opening0 in self._known:
            if d_id is not None:
                if d_id == moving:
                    continue
                shift = openings.get(d_id, self.opening[d_id]) - opening0
                x, y = x + shift * a[0], y + shift * a[1]
            out.append((name, (x, y, z), (x, y, z + shape.height), shape.width / 2))
        return out  # type: ignore[return-value]

    def _guard_path(self, path: Sequence[Sequence[float]], moving: Optional[str] = None,
                    released: Optional[str] = None) -> Optional[str]:
        """The known objects alone along ``path``: why not, if a link comes within ``object_margin_m``."""
        objects = self._known_boxes(self.opening, moving)
        if not objects or self.calib.robot is None:
            return None
        near = {released} if released is not None else set()
        for q in collision.densify(path, self.s.check_step_deg):
            why = self._object_hit(q, objects, near, "now")
            if why is not None:
                return f"{why} at {_r(q, 1)}"
        return None

    def _guard_calib(self) -> List[Dict[str, Any]]:
        """The taught legs that come within ``object_margin_m`` of the tallest catalogue object standing
        with its centre anywhere on the handover mat (the calib's in-zone rule), pads swept over their
        travel: flagged when the run starts, since taught legs are played as given (leader 16:58)."""
        fx, robot = self.fx, self.calib.robot
        if robot is None or not fx.shapes or not any(d.taught for d in fx.drawers.values()):
            return []
        r = max(s.width for s in fx.shapes.values()) / 2
        top = max(s.height for s in fx.shapes.values())
        (hx, hy), (sx, sy) = fx.handover_center, fx.handover_size
        z = self.calib.table_z
        zone = [("handover zone", (hx - sx / 2 - r, hy - sy / 2 - r, z), (hx + sx / 2 + r, hy + sy / 2 + r, z + top))]
        out = []
        for d in sorted(fx.drawers.values(), key=lambda d: d.id):
            if not d.taught:
                continue
            for name, leg in d.taught.legs():
                path = ([list(leg.start)] if leg.start is not None else []) + [list(q) for q in leg.q]
                for q in collision.densify(path, self.s.check_step_deg):
                    why = self._object_hit(q, zone, (), None)
                    if why is not None:
                        out.append({"leg": f"{d.id} {name}", "why": why})
                        break
        return out

    def _dips(self, path: Sequence[Sequence[float]], floor: Optional[float]) -> Optional[str]:
        """Why not, if the TCP goes below ``floor`` anywhere along ``path``."""
        if floor is None:
            return None
        chain = self.calib.chain
        low = min(kinematics.position(chain.pose(q, "tcp"))[2] for q in collision.densify(path, self.s.check_step_deg))
        return f"TCP down to z={low:.3f} (floor {floor:.3f})" if low < floor - 1e-3 else None

    def _why_blocked(self, path: Sequence[Sequence[float]], line: bool = False) -> Optional[str]:
        return self._path_clear(path, points=not line, **self._ctx)

    def _held_joint(self, goal: Sequence[float], phase: str, floor: float, speed: Optional[int] = None) -> bool:
        """Holding an object, one joint move to ``goal`` (angles) when it passes the checker and keeps the TCP
        over ``floor`` - smooth, where chains of 1 cm line steps shook a cone out of the pads (D3 v3.2
        s502_b). Whether it went."""
        here = self._measured()
        path = kinematics.joint_path(here, goal, self.s.max_step)
        why = self._why_blocked([here] + path) or self._dips([here] + path, floor)
        if why is not None:
            self.log.event("skip", why=f"{phase}: no single joint move: {why}")
            return False
        self.log.event("held_move", phase=phase, how="one joint move", waypoints=len(path))
        self._send(path, phase, speed or self.s.slow_speed)
        return True

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
        if self._holding:
            # No wrist roll with an object in hand (3c40c7a): by a look pose, J6 stays where it is. v3.3's
            # mat behind the base sent a carry from r2c2 by the table look, J6 +35 -> -29 -> +35 (D0).
            looks = [(n, [*v[:5], s[5]]) for n, v in looks]
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

    def _goto(self, goal: Sequence[float], phase: str, floor: Optional[float], speed: Optional[int] = None,
              taught: bool = True, end_tol: float = 0.5) -> None:
        """As pick-place's, but along a taught route where there is one (:meth:`_taught_route`), else round
        what is in the way (:meth:`_route`), and the floor is no higher than where the path starts."""
        here = self._measured()
        route = self._taught_route(here, goal, end_tol) if taught else None
        if route is not None:
            for name, leg in route:
                self.log.event("taught", phase=phase, route=name)
                self._play(leg, phase, speed)
            return
        if floor is not None:
            # sim-scene's tilted looks into r1c2 and r2c2 hold the TCP beside the drawer below the
            # floor for above it (wall top + 2 cm): the path may not go lower than it starts (D3 s451).
            floor = min(floor, kinematics.position(self.calib.chain.pose(here, "tcp"))[2])
        vias, how, why = self._route(here, goal, floor, **self._ctx)
        look = self.fx.cabinet_look_pose_deg
        if vias is None and taught and look:
            # Off a drawer's view the planned moves all start with a joint move the checker refuses (r0c2's
            # and r1c2's tilted views: forearm and camera); v3.2 teaches the way back to the cabinet look
            # (D3 v3.2 D0, put r0c2): take it, and plan from there.
            back = self._taught_route(here, look)
            if back is not None:
                v2, how2, _ = self._route(look, goal, floor, **self._ctx)
                if v2 is not None:
                    self.log.event("detour", phase=phase, how=f"taught back to the cabinet look, then {how2}", why=why)
                    for name, leg in back:
                        self.log.event("taught", phase=phase, route=name)
                        self._play(leg, f"{phase}: detour", speed)
                    here, vias = self._measured(), v2
        if vias is None:
            raise PathBlocked(f"{phase}: no clear way: straight, the path would put the {why}")
        if vias:
            self.log.event("detour", phase=phase, how=how, why=why)
        for i, q in enumerate([*vias, list(goal)]):
            last = i == len(vias)
            self._send(kinematics.joint_path(here if i == 0 else self._measured(), q, self.s.max_step),
                       phase if last else f"{phase}: detour", speed or self.s.speed)
