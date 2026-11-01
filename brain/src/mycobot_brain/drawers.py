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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mycobot_brain import drawer_vlm, grounding, kinematics
from mycobot_brain.fixture import Drawer, Fixture, Shape
from mycobot_brain.kinematics import IKError, Vec
from mycobot_brain.pickplace import DOWN, TILTS_DEG, PickPlace, PickPlaceError, Settings, _r

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


def _add(a: Sequence[float], b: Sequence[float], s: float = 1.0) -> Vec:
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def _in_box(p: Sequence[float], lo: Sequence[float], hi: Sequence[float], m: float) -> bool:
    return all(lo[i] - m <= p[i] <= hi[i] + m for i in range(3))


class NotHeld(PickPlaceError):
    """The VLM does not see the object in the gripper after a grasp; the gripper is open again."""


class PathBlocked(PickPlaceError):
    """A joint-space move would run the TCP or the camera into the cabinet or an open drawer."""


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
        self._frames: Dict[str, Tuple[Vec, Optional[Vec], Dict[float, List[float]]]] = {}

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
                    a, self.fx.handle_frame(d)[0]))))), 1)
            except PickPlaceError:
                reach[d.id] = None
        self.log.event("plan", command=cmd, handle_tilt_deg=reach)
        return {"ok": None, "why": "planned only; --execute to move", "command": cmd, "handle_tilt_deg": reach}

    def _find(self, cmd: Dict[str, Any], desc: str) -> Dict[str, Any]:
        searched: Dict[str, str] = {}
        for _ in range(self.s.max_drawers):
            candidates = [d for d in sorted(self.fx.drawers) if d not in searched]
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
        return {"ok": False, "why": f"no {desc} in {sorted(searched)}"}

    def _take(self, cmd: Dict[str, Any], desc: str) -> Dict[str, Any]:
        d = self.fx.drawer(cmd["drawer"])
        look = self._open_and_look(d, desc)
        if look["target"] is None:
            self._close(d)
            return {"ok": False, "why": f"no {desc} in {d.id}: {look['seen']}"}
        obj = look["objects"][look["target"]]
        self._take_out(d, obj, look, drawer_vlm.describe(obj["colour"], obj["shape"]))
        return {"ok": True, "why": f"took the {obj['colour']} {obj['shape']} out of {d.id}"}

    def _put(self, cmd: Dict[str, Any], desc: str) -> Dict[str, Any]:
        c, fx = self.calib, self.fx
        shape = self._shape(cmd["shape"])
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
                    clear=self.s.drawer_clear_m)
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
        """Approach and jaw axes for ``d``'s handle: the calib's, or tipped 5-15 deg towards the base.

        Near the base a vertical grasp folds J3 into its limit before the
        drawer is fully out (the same reason pick-place tips its grasp); the
        whole gripper turns, jaws with it, so the handle still sits across them.
        Checked along the whole pull, both ways.
        """
        if d.id in self._frames:
            return self._frames[d.id][:2]
        fx, chain = self.fx, self.calib.chain
        a0, j0 = fx.handle_frame(d)
        h = fx.handle(d, fx.open_m)
        r = math.hypot(h[0], h[1])
        inward = (-h[0] / r, -h[1] / r, 0.0)
        axis = kinematics.cross(a0, inward)
        why = ""
        for tilt in TILTS_DEG:
            if tilt and kinematics.norm(axis) < 1e-6:
                break
            rot = kinematics.axis_angle(axis, math.radians(tilt)) if tilt else None
            a = a0 if rot is None else tuple(sum(rot[i][k] * a0[k] for k in range(3)) for i in range(3))
            j = j0 if rot is None or j0 is None else tuple(sum(rot[i][k] * j0[k] for k in range(3)) for i in range(3))
            try:
                # Seeded from the table look pose: an ordinary elbow-up arm, whatever the cabinet view needs.
                pre = kinematics.solve_ik(chain, self.calib.look_pose_deg, _add(fx.handle(d, 0.0), a, -fx.pregrasp_m),
                                          approach=a, jaw=j)
                path = kinematics.line_path(chain, pre, fx.handle(d, 0.0), step_m=0.02, approach=a, jaw=j)
                path += kinematics.line_path(chain, path[-1], fx.handle(d, fx.open_m), step_m=0.02, approach=a, jaw=j)
            except IKError as exc:
                why = str(exc)
                continue
            # A tipped gripper leans its back (and the camera) towards the cabinet face.
            hit = self._path_clear([pre] + path, skip=d.id)
            if hit:
                why = f"tipped {tilt} deg the {hit}"
                continue
            if tilt:
                self.log.event("approach", what=f"handle {d.id}", tilt_deg=tilt)
            # Seeds for the pre-grasp IK at either end, so the arm keeps the branch checked here.
            self._frames[d.id] = (a, j, {0.0: pre, fx.open_m: path[-1]})
            return a, j
        raise PickPlaceError(f"handle of {d.id}: no grasp along the whole pull, even tipped {TILTS_DEG[-1]} deg: {why}")

    def _move_handle(self, d: Drawer, start: float, end: float, verb: str) -> None:
        """Grasp the handle where the drawer is (``start`` m out), move it to ``end``, let go, back off."""
        fx, c = self.fx, self.calib
        a, j = self._handle_frame(d)
        self._phase(f"{verb} {d.id}")
        self._gripper(c.gripper_open, "open")
        seeds = self._frames[d.id][2]
        seed = seeds.get(start, seeds[min(seeds, key=lambda s: abs(s - start))])
        try:
            q = kinematics.solve_ik(c.chain, seed, _add(fx.handle(d, start), a, -fx.pregrasp_m), approach=a, jaw=j)
        except IKError as exc:
            raise PickPlaceError(f"{verb} {d.id}: pre-grasp: {exc}") from exc
        self._goto(q, f"{verb} {d.id}: pre-grasp", floor=None)
        self._line(fx.handle(d, start), f"{verb} {d.id}: to handle", a, j)
        close = fx.handle_close_value if fx.handle_close_value is not None else c.gripper_closed
        self._gripper(close, "close")
        self._line(fx.handle(d, end), f"{verb} {d.id}", a, j, speed=self.s.handle_speed)
        self.opening[d.id] = end
        self._gripper(c.gripper_open, "open")
        self._line(_add(fx.handle(d, end), a, -fx.pregrasp_m), f"{verb} {d.id}: back off", a, j)

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
        """Open ``d`` and say what is in it; one more pull if the VLM does not see it open."""
        self._open(d)
        look = self._look_into(d, target)
        if not look["open"]:
            self.log.event("retry", why=f"the VLM does not see {d.id} open", seen=look["seen"])
            self._open(d, again=True)
            look = self._look_into(d, target)
            if not look["open"]:
                self._close(d, force=True)
                raise PickPlaceError(f"{d.id} does not open: {look['seen']}")
        return look

    def _look_into(self, d: Drawer, target: Optional[str]) -> Dict[str, Any]:
        self._phase(f"look into {d.id}")
        self._goto(self._look_pose(d), f"look: {d.id}", floor=None)
        k, chain = self.calib.camera, self.calib.chain
        frame, q = self._look()
        floor = drawer_vlm.floor_box(self.fx, d, self.fx.open_m, chain, k, q)
        req = drawer_vlm.build_drawer_request(frame.jpeg, self.fx, d, floor, target=target, view=self.s.view,
                                              **({"model": self.model} if self.model else {}))
        raw = self._ask(f"drawer-{d.id}", req, [frame.jpeg], q)
        look = self._reply(f"drawer-{d.id}", raw, lambda r: drawer_vlm.parse_drawer(
            r, self.fx, with_target=target is not None, width=k.width, height=k.height))
        for i, o in enumerate(look["objects"]):
            xyz = self._object_xyz(d, o, q)
            self.log.event("estimate", role="object", drawer=d.id, index=i, label=f"{o['colour']} {o['shape']}",
                           source=self.s.view, pixel=o["pixel"], xyz=None if xyz is None else _r(xyz, 4))
        return {**look, "angles": q, "jpeg": frame.jpeg}

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
        handles = {d.id: drawer_vlm.grid(c.chain, c.camera, q, d.handle_closed) for d in fx.drawers.values()}
        req = drawer_vlm.build_cabinet_request(frame.jpeg, fx, handles, goal=goal, searched=searched,
                                               candidates=candidates,
                                               **({"model": self.model} if self.model else {}))
        purpose = "cabinet-next" if candidates is not None else "cabinet-shut"
        raw = self._ask(purpose, req, [frame.jpeg], q)
        return self._reply(purpose, raw, lambda r: drawer_vlm.parse_cabinet(r, fx, candidates))

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
        h = self.fx.handover_center
        self._put_down((h[0], h[1], self.calib.table_z), self.calib.table_z, shape, "handover",
                    wall_top=None, clear=self.s.handover_clear_m)
        self._phase("check handover")
        q = self._camera_over((h[0], h[1], self.calib.table_z), self.calib.table_z + shape.height / 2,
                              min_tcp_z=self.calib.table_z + shape.height + 0.03)
        if q is not None:
            self._goto(q, "look: handover", floor=self.calib.table_z + shape.height + 0.02)
        check = self._check(drawer_vlm.handover_question(desc), [], "handover")
        if not check["ok"]:
            raise PickPlaceError(f"the {desc} is not in the handover zone: {check['seen']}")
        self._close(d)

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
            q = self._camera_over(xyz, plane, min_tcp_z=clear)
            blocked = q and self._path_clear(kinematics.joint_path(self._measured(), q, s.max_step))
            if blocked:
                self.log.event("skip", why=f"no close look: {blocked}")
            if q is not None and not blocked:
                self._goto(q, f"close look: {where}", floor=clear)
                found, frame, qa = self._locate(f"pick up the {desc}", ("pick",), f"close-{where}", with_next=False,
                                                describe={"pick": f"the {desc}"})
                before, view = frame.jpeg, qa
                if found["pick"]["visible"]:
                    xyz = self._estimate(found["pick"], qa, plane, "pick", "close")
        grasp_z = max(surface_z + shape.grasp_height, surface_z + c.pad_below_tcp + s.finger_clear_m)
        above = (xyz[0], xyz[1], max(clear, grasp_z + 0.03) + s.pregrasp_m / 2)
        jaw = self._jaw(drawer, xyz, others)
        approach = DOWN if jaw is not None else self._approach(above, (xyz[0], xyz[1], grasp_z), "grasp")
        self._phase(f"pre-grasp: {where}")
        self._goto(self._ik(self._measured(), above, approach, jaw), f"pre-grasp: {where}", floor=clear)
        self._phase("descend")
        self._line((xyz[0], xyz[1], grasp_z), "descend", approach, jaw)
        self._gripper(c.gripper_closed, "close")
        self._phase("lift")
        # Lift until the object's bottom (grasp_height under the TCP) clears the walls or the table.
        lift = max(above[2], top + shape.grasp_height + s.wall_clear_m)
        self._line((xyz[0], xyz[1], lift), "lift", approach, jaw)
        self._phase("check held")
        if view is not None:
            self._goto(view, "look: held", floor=None)
            question = grounding.held_question_same_view(desc, self._gripper_side())
            images = [("Before the grasp:", before)]
        else:
            question, images = grounding.held_question(desc), []
        check = self._check(question, images, "held")
        if not check["ok"]:
            self._gripper(c.gripper_open, "open")
            raise NotHeld(f"the {desc} was not held: {check['seen']}")

    def _put_down(self, xyz: Vec, surface_z: float, shape: Shape, where: str, wall_top: Optional[float],
               clear: float) -> None:
        """Carry what is held over ``xyz``, lower until its bottom is ``clear`` over the surface, let go."""
        s, c = self.s, self.calib
        release_z = surface_z + shape.grasp_height + clear
        carry = max(release_z, (wall_top or surface_z) + shape.grasp_height + s.wall_clear_m) + 0.03
        self._phase(f"carry: {where}")
        q = self._ik(self._measured(), (xyz[0], xyz[1], carry), DOWN)
        self._goto(q, f"carry: {where}", floor=surface_z + shape.grasp_height + 0.01)
        self._phase(f"lower: {where}")
        self._line((xyz[0], xyz[1], release_z), f"lower: {where}", DOWN)
        self._gripper(c.gripper_open, "open")
        self._phase("retreat")
        self._line((xyz[0], xyz[1], carry), "retreat", DOWN)

    # -- staying out of the cabinet -------------------------------------------------

    def _path_clear(self, path: Sequence[Sequence[float]], skip: Optional[str] = None) -> Optional[str]:
        """None if the TCP and camera stay out of the cabinet body and of open drawers (but ``skip``)."""
        fx, chain, m = self.fx, self.calib.chain, self.s.body_clear_m
        boxes = [("cabinet", fx.box_min, fx.box_max)]
        for d_id, opening in self.opening.items():
            if opening > 1e-3 and d_id != skip:
                d = fx.drawers[d_id]
                c = fx.interior_center(d, opening)
                hx, hy = d.interior_size[0] / 2 + 0.005, d.interior_size[1] / 2 + 0.005
                boxes.append((f"open {d_id}", (c[0] - hx, c[1] - hy, d.floor_z - 0.01),
                              (c[0] + hx, c[1] + hy, d.floor_z + d.interior_size[2])))
        for q in path:
            for tool in ("tcp", "camera"):
                p = kinematics.position(chain.pose(q, tool))
                for name, lo, hi in boxes:
                    if _in_box(p, lo, hi, m):
                        return f"{tool} at {_r(p, 3)} inside the {name}"
        return None

    def _check_path(self, path: Sequence[Sequence[float]], phase: str) -> None:
        why = self._path_clear(path)
        if why is not None:
            raise PathBlocked(f"{phase}: the joint path would put the {why}")

    def _goto(self, goal: Sequence[float], phase: str, floor: Optional[float], speed: Optional[int] = None) -> None:
        """As pick-place's, but a path through the cabinet goes round by the cabinet look pose."""
        try:
            super()._goto(goal, phase, floor, speed)
        except PathBlocked as exc:
            via = self.fx.cabinet_look_pose_deg
            if list(goal) == list(via):
                raise
            self.log.event("detour", phase=phase, why=str(exc))
            super()._goto(via, f"{phase}: detour", floor=None, speed=speed)
            super()._goto(goal, phase, floor, speed)
