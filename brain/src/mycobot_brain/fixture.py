"""The drawer cabinet as the code knows it, like the table height.

From the calib file: where each handle is when its drawer is shut, which way
and how far a drawer pulls out, each drawer's floor and inner box, where to
look from, the object catalogue (shape names, sizes, grasp heights) and the
handover zone. Never what is in a drawer or whether it is open: that is for
the VLM to say from the camera.

Drawer ids are ``r{row}c{col}``: row 0 is the bottom, column 0 the robot's
left (+y).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mycobot_brain.kinematics import Vec, dot, norm

ROW_KO = ("아래", "가운데", "위")
COL_KO = ("왼쪽", "가운데", "오른쪽")
ROW_EN = ("bottom", "middle", "top")
COL_EN = ("left", "middle", "right")


class FixtureError(ValueError):
    """The calib's cabinet section is missing something or does not add up."""


def _vec(v: Any, n: int = 3) -> Tuple[float, ...]:
    if not (isinstance(v, (list, tuple)) and len(v) == n):
        raise FixtureError(f"need {n} numbers, got {v!r}")
    return tuple(float(x) for x in v)


def _unit(v: Sequence[float]) -> Vec:
    n = norm(v)
    if n < 1e-9:
        raise FixtureError(f"zero vector {v!r}")
    return (v[0] / n, v[1] / n, v[2] / n)


def _pose(v: Any) -> Optional[List[float]]:
    return None if v is None else [float(a) for a in _vec(v, 6)]


@dataclass(frozen=True)
class HandleStage:
    """One grip on a handle over part of the pull: from ``from_m`` to ``to_m`` out.

    sim-scene's v3 recipe (BOARD 2026-11-02) pulls in two: a side pinch to ~50 mm, then a top-down pinch
    to fully open, because a horizontal grip at the end of a full pull puts the wrist in the base. The
    top pinch tilts as the drawer comes out, so the TCP and the tool axis are given at both ends and
    go evenly in between.
    """

    from_m: float
    to_m: float
    tcp_from: Vec  # TCP on the handle with the drawer from_m out
    tcp_to: Vec
    approach_from: Vec  # tool z there
    approach_to: Vec
    jaw: Optional[Vec]  # tool x; None leaves it free
    standoff_m: float  # how far off the handle the grip starts and ends, drawer shut
    backoff_m: Optional[float] = None  # the same with the drawer partly out (None: standoff_m)
    in_axis: Optional[Vec] = None  # the way the TCP goes onto the handle (None: along the tool axis)
    preshape: Optional[int] = None  # gripper to come in with
    close: Optional[int] = None
    release: Optional[int] = None
    #: the jaw's sign is part of the grip (v3 gives the whole tool rotation): the gripper body and the
    #: camera sit off the tool axis, and the plan that was checked had them on this side
    exact_jaw: bool = False
    #: joint angles near the calib's own plan for this grip (its pre-grasp), to start IK from
    seed_deg: Optional[List[float]] = None


@dataclass(frozen=True)
class TaughtLeg:
    """Joint waypoints sim-scene verified (calib v3.2), sent as given, no IK (leader 2026-11-02 12:24):
    ``q`` are the waypoints to send, <= 2 deg apart; ``start`` is where the leg begins, when known (it is
    not sent); ``tcp`` is sim-scene's FK at ``q``, for comparison."""

    q: Tuple[Tuple[float, ...], ...]
    start: Optional[Tuple[float, ...]] = None
    tcp: Optional[Tuple[Vec, ...]] = None
    speed: Optional[int] = None

    @property
    def first(self) -> Tuple[float, ...]:
        return self.start if self.start is not None else self.q[0]


#: a taught open/close is a list of these phases: to the standoff, onto the handle, pull (or push), back
#: off, and back to the cabinet look
PHASES = ("standoff", "approach", "pull", "back_off", "return")


@dataclass(frozen=True)
class TaughtSegment:
    phase: str  # one of PHASES
    leg: TaughtLeg
    stage: Optional[int] = None  # which grip (0-based)
    gripper: Optional[int] = None  # gripper value set before the leg is played
    opening_after: Optional[float] = None  # how far out the drawer is after a pull


@dataclass(frozen=True)
class Taught:
    open: Tuple[TaughtSegment, ...]
    close: Tuple[TaughtSegment, ...]
    #: between the views: "cabinet_to_tilted", "tilted_to_topdown", "topdown_to_cabinet", ...
    routes: Dict[str, TaughtLeg] = field(default_factory=dict)

    def legs(self) -> List[Tuple[str, TaughtLeg]]:
        out = [(f"{verb} {i} {seg.phase}", seg.leg) for verb, segs in (("open", self.open), ("close", self.close))
               for i, seg in enumerate(segs)]
        return out + [(f"route {name}", leg) for name, leg in self.routes.items()]


@dataclass(frozen=True)
class Drawer:
    id: str
    row: int
    col: int
    handle_closed: Vec  # TCP target on the handle with the drawer shut
    floor_z: float  # top of the drawer floor
    interior_center: Tuple[float, float]  # x, y of the inner box with the drawer shut
    interior_size: Vec  # inner box along x, y, z
    look_pose_deg: Optional[List[float]] = None  # tilted view of the opened interior
    topdown_pose_deg: Optional[List[float]] = None
    #: tool z / tool x at this handle when it differs from the cabinet's (a heading per column, say).
    grasp_approach: Optional[Vec] = None
    grasp_jaw: Optional[Vec] = None
    #: highest part of the drawer a carried object must clear (a full-height front panel); None: the walls.
    rim_z: Optional[float] = None
    #: the handle's own centre with the drawer shut (``handle_closed`` is where the TCP goes, short of it)
    knob: Optional[Vec] = None
    #: tray wall height over the floor when the inner box's z is the clearance to the shelf above
    wall_h: Optional[float] = None
    #: the drawer front's bottom and top
    panel_z: Optional[Tuple[float, float]] = None
    #: the calib's grips over the pull, in order; empty: one grip from the cabinet-wide keys
    stages: Tuple[HandleStage, ...] = ()
    #: False: the calib says no plan opens it (sim-scene 13:10: the bottom row with the wrist kept out of
    #: the base); never opened, never offered to the VLM
    usable: bool = True
    #: sim-scene's verified joint waypoints for opening, closing and moving between the views (v3.2)
    taught: Optional["Taught"] = None

    @property
    def knob_closed(self) -> Vec:
        """Where to mark the handle in a picture."""
        return self.knob if self.knob is not None else self.handle_closed

    @property
    def wall_top(self) -> float:
        walls = self.wall_h if self.wall_h is not None else self.interior_size[2]
        return max(self.floor_z + walls, self.rim_z if self.rim_z is not None else -1.0)

    @property
    def name_ko(self) -> str:
        if (self.row, self.col) == (1, 1):
            return "가운데"
        return f"{COL_KO[self.col]} {ROW_KO[self.row]}" if self.col != 1 else f"가운데 {ROW_KO[self.row]}"

    @property
    def name_en(self) -> str:
        return f"{ROW_EN[self.row]} row, {COL_EN[self.col]} column"


@dataclass(frozen=True)
class Shape:
    name: str
    height: float
    width: float  # widest across, for spacing objects in a drawer
    grasp_height: float  # TCP above the surface it stands on, for the grasp


@dataclass
class Fixture:
    drawers: Dict[str, Drawer]
    pull_axis: Vec  # unit, the way a drawer moves when it opens
    open_m: float
    box_min: Vec
    box_max: Vec
    cabinet_look_pose_deg: List[float]
    grasp_approach: Vec  # tool z at the handle
    grasp_jaw: Optional[Vec]  # tool x at the handle; None leaves it free
    pregrasp_m: float
    handle_close_value: Optional[int]
    #: jaw axis for a top-down pick in a drawer; None: whichever way has more room.
    pick_jaw: Optional[Vec]
    colours: List[str]
    shapes: Dict[str, Shape]
    handover_center: Tuple[float, float]
    handover_size: Tuple[float, float]
    #: gripper value to approach and let go of a handle: fully open can hit the handles above and below.
    handle_open_value: Optional[int] = None
    #: TCP over the support for the pre-grasp of a top-down pick, where the calib has checked reach
    pick_above_m: Optional[float] = None
    #: standoff from a handle to start the approach and end the back-off (sim-scene's oracle: 8 cm)
    far_m: Optional[float] = None
    #: how far closed handles stick out of the cabinet body towards the robot
    handle_protrusion_m: float = 0.0
    #: a handle's protrusion, width and height (along the pull, across it, up)
    handle_size: Optional[Vec] = None
    #: a drawer's outside, across the pull and along it (front panel to back wall)
    drawer_outer: Optional[Tuple[float, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- where things are -------------------------------------------------------

    def handle_frame(self, d: Drawer) -> Tuple[Vec, Optional[Vec]]:
        """Approach (tool z) and jaw (tool x) for ``d``'s handle."""
        if d.grasp_approach is not None:
            return d.grasp_approach, d.grasp_jaw
        return self.grasp_approach, self.grasp_jaw

    def drawer(self, drawer_id: str) -> Drawer:
        try:
            return self.drawers[drawer_id]
        except KeyError:
            raise FixtureError(f"no drawer {drawer_id!r}; have {sorted(self.drawers)}") from None

    @property
    def standoff_m(self) -> float:
        """Where a handle approach starts and its back-off ends, off the handle along the tool axis."""
        return max(self.far_m or 0.0, self.pregrasp_m)

    def body_box(self) -> Tuple[Vec, Vec]:
        """The cabinet body grown by the closed handles on the side they stick out of."""
        lo, hi = list(self.box_min), list(self.box_max)
        for i, a in enumerate(self.pull_axis):
            if a < -0.5:
                lo[i] -= self.handle_protrusion_m
            elif a > 0.5:
                hi[i] += self.handle_protrusion_m
        return tuple(lo), tuple(hi)  # type: ignore[return-value]

    def knob_boxes(self, openings: Dict[str, float]) -> List[Tuple[str, Vec, Vec]]:
        """Each handle as a box where its drawer is (``openings``, m out); [] without the handle's size."""
        if self.handle_size is None:
            return []
        prot, width, height = self.handle_size
        a = self.pull_axis
        half = [prot / 2 if abs(a[0]) > 0.5 else width / 2, prot / 2 if abs(a[1]) > 0.5 else width / 2, height / 2]
        out = []
        for d in self.drawers.values():
            c = self.knob(d, openings.get(d.id, 0.0))
            out.append((d.id, tuple(c[i] - half[i] for i in range(3)), tuple(c[i] + half[i] for i in range(3))))
        return out  # type: ignore[return-value]

    def tray_parts(self, d: Drawer, opening: float) -> List[Tuple[str, Vec, Vec]]:
        """The drawer's front panel, floor, side walls and back wall as boxes, ``opening`` m out, for the
        arm's capsules to keep off (the part still in the cabinet is inside its body box anyway)."""
        a, b = self.axes()
        along, across = self.extent(d)
        c = self.interior_center(d, opening)
        s_c, t_c = dot(c, a), dot(c, b)
        width, depth = self.drawer_outer or (across + 0.006, along + 0.012)
        s_front = self.front + opening  # the panel's outer face
        s_in, s_back_in, s_back = s_c + along / 2, s_c - along / 2, s_front - depth
        hw, hw_in = width / 2, across / 2
        floor, wall_top = d.floor_z, d.floor_z + (d.wall_h if d.wall_h is not None else d.interior_size[2])
        bottom, top = d.panel_z if d.panel_z is not None else (floor - 0.005, d.wall_top)
        # (along s, across t, z) ranges, made into boxes in the base frame below.
        parts = [("front", (s_in, s_front), (-hw, hw), (bottom, top)),
                 ("floor", (s_back, s_in), (-hw, hw), (bottom, floor)),
                 ("side wall", (s_back, s_in), (hw_in, hw), (floor, wall_top)),
                 ("side wall", (s_back, s_in), (-hw, -hw_in), (floor, wall_top)),
                 ("back wall", (s_back, s_back_in), (-hw, hw), (floor, wall_top))]
        out = []
        for name, (s0, s1), (t0, t1), (z0, z1) in parts:
            corners = [(s * a[0] + (t_c + t) * b[0], s * a[1] + (t_c + t) * b[1]) for s in (s0, s1) for t in (t0, t1)]
            lo = (min(p[0] for p in corners), min(p[1] for p in corners), z0)
            hi = (max(p[0] for p in corners), max(p[1] for p in corners), z1)
            out.append((f"{d.id} {name}", lo, hi))
        return out

    @staticmethod
    def stages_of(d: Drawer, pull: Vec, open_m: float) -> List[HandleStage]:
        """``stages`` before there is a Fixture: the calib's stages, else one grip over the whole pull."""
        if d.stages:
            return list(d.stages)
        return [HandleStage(0.0, open_m, d.handle_closed,
                            tuple(h + open_m * a for h, a in zip(d.handle_closed, pull)),  # type: ignore[arg-type]
                            (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), None, 0.0)]

    def stages(self, d: Drawer) -> List[HandleStage]:
        """The grips that open ``d``, in order: the calib's, else one grip all the way from
        ``handle_closed_m`` and the cabinet's handle keys."""
        if d.stages:
            return list(d.stages)
        a, j = self.handle_frame(d)
        return [HandleStage(0.0, self.open_m, d.handle_closed, self.handle(d, self.open_m), a, a, j, self.standoff_m,
                            preshape=self.handle_open_value, close=self.handle_close_value,
                            release=self.handle_open_value)]

    @staticmethod
    def _along(stage: HandleStage, opening: float) -> float:
        return (opening - stage.from_m) / (stage.to_m - stage.from_m)

    def stage_tcp(self, stage: HandleStage, opening: float) -> Vec:
        """Where the TCP holds the handle in ``stage`` with the drawer ``opening`` m out."""
        t = self._along(stage, opening)
        return tuple(a + (b - a) * t for a, b in zip(stage.tcp_from, stage.tcp_to))  # type: ignore[return-value]

    def stage_approach(self, stage: HandleStage, opening: float) -> Vec:
        t = self._along(stage, opening)
        return _unit([a + (b - a) * t for a, b in zip(stage.approach_from, stage.approach_to)])

    def stage_in(self, stage: HandleStage, opening: float) -> Vec:
        """The way the TCP moves onto the handle (and back off it, reversed)."""
        return stage.in_axis if stage.in_axis is not None else self.stage_approach(stage, opening)

    @staticmethod
    def stage_standoff(stage: HandleStage, opening: float) -> float:
        if opening > 1e-3 and stage.backoff_m is not None:
            return stage.backoff_m
        return stage.standoff_m

    def knob(self, d: Drawer, opening: float) -> Vec:
        """The handle's centre with the drawer ``opening`` m out, for marking it in a picture."""
        return tuple(h + opening * a for h, a in zip(d.knob_closed, self.pull_axis))  # type: ignore[return-value]

    def handle(self, d: Drawer, opening: float) -> Vec:
        """The handle's grasp point with the drawer pulled out ``opening`` m."""
        return tuple(h + opening * a for h, a in zip(d.handle_closed, self.pull_axis))  # type: ignore[return-value]

    def interior_center(self, d: Drawer, opening: float) -> Vec:
        """Centre of the drawer floor, pulled out ``opening`` m."""
        x, y = d.interior_center
        return (x + opening * self.pull_axis[0], y + opening * self.pull_axis[1], d.floor_z)

    @property
    def front(self) -> float:
        """The cabinet face a drawer comes out of, as a value of ``dot(p, pull_axis)``."""
        corners = [(x, y, z) for x in (self.box_min[0], self.box_max[0]) for y in (self.box_min[1], self.box_max[1])
                   for z in (self.box_min[2], self.box_max[2])]
        return max(dot(c, self.pull_axis) for c in corners)

    def axes(self) -> Tuple[Vec, Vec]:
        """Unit vectors in the floor plane: along the pull, and across it."""
        a = _unit((self.pull_axis[0], self.pull_axis[1], 0.0))
        return a, (-a[1], a[0], 0.0)

    def extent(self, d: Drawer) -> Tuple[float, float]:
        """The inner box's length along and across the pull (the box is axis-aligned)."""
        a, _ = self.axes()
        x, y = d.interior_size[0], d.interior_size[1]
        return (x, y) if abs(a[0]) >= abs(a[1]) else (y, x)

    def exposed(self, d: Drawer, opening: float, margin: float = 0.0) -> Tuple[Vec, float, float]:
        """The part of the opened floor out in the open (not under the cabinet), shrunk by
        ``margin``: its centre and half-lengths along and across the pull. Empty halves are 0."""
        a, _ = self.axes()
        c = self.interior_center(d, opening)
        along, across = self.extent(d)
        half_along, half_across = along / 2, across / 2
        # Along the pull, the floor runs from s_back to s_out (s = dot(p, a)).
        s_c = dot(c, a)
        s_out = s_c + half_along - margin
        s_back = max(s_c - half_along, self.front) + margin
        hw = half_across - margin
        if s_out <= s_back or hw <= 0:
            return c, 0.0, 0.0
        mid = (s_out + s_back) / 2 - s_c
        return (c[0] + mid * a[0], c[1] + mid * a[1], c[2]), (s_out - s_back) / 2, hw

    def inside(self, d: Drawer, p: Sequence[float], opening: float, margin: float = 0.0) -> bool:
        """``p`` (x, y) over the opened drawer's floor, ``margin`` in from its walls."""
        a, b = self.axes()
        c = self.interior_center(d, opening)
        rel = (p[0] - c[0], p[1] - c[1], 0.0)
        along, across = self.extent(d)
        return abs(dot(rel, a)) <= along / 2 - margin and abs(dot(rel, b)) <= across / 2 - margin

    def free_spots(self, d: Drawer, opening: float, occupied: Sequence[Sequence[float]], width: float,
                   margin: float = 0.006, grid: int = 7) -> List[Tuple[Vec, float, float]]:
        """Floor points in the open part of the drawer for an object ``width`` wide, ``margin`` in from
        the walls: (point, room to the nearest of ``occupied`` (inf without any), distance from the
        middle). Empty if it cannot fit."""
        c, ha, hb = self.exposed(d, opening, margin + width / 2)
        if ha <= 0 or hb <= 0:
            return []
        a, b = self.axes()
        out = []
        for i in range(grid):
            for j in range(grid):
                sa = -ha + 2 * ha * i / (grid - 1)
                sb = -hb + 2 * hb * j / (grid - 1)
                p = (c[0] + sa * a[0] + sb * b[0], c[1] + sa * a[1] + sb * b[1], c[2])
                gap = min((math.hypot(p[0] - o[0], p[1] - o[1]) for o in occupied), default=math.inf)
                out.append((p, gap, math.hypot(sa, sb)))
        return out  # type: ignore[return-value]

    def free_spot(self, d: Drawer, opening: float, occupied: Sequence[Sequence[float]], width: float,
                  margin: float = 0.006, grid: int = 7) -> Optional[Vec]:
        """The point of :meth:`free_spots` as far as it can be from ``occupied``; ties towards the middle."""
        best = max(self.free_spots(d, opening, occupied, width, margin, grid),
                   key=lambda s: min(s[1], 1.0) - 1e-3 * s[2], default=None)
        return None if best is None else best[0]

    def in_handover(self, p: Sequence[float], margin: float = 0.0) -> bool:
        return (abs(p[0] - self.handover_center[0]) <= self.handover_size[0] / 2 - margin
                and abs(p[1] - self.handover_center[1]) <= self.handover_size[1] / 2 - margin)

    # -- words ----------------------------------------------------------------

    def drawer_table(self) -> str:
        """One line per drawer for a prompt: id, Korean name, English position."""
        return "\n".join(f"{d.id}: {d.name_ko} 서랍 ({d.name_en})" for d in sorted(
            self.drawers.values(), key=lambda d: (-d.row, d.col)))


def _bounds(parts: Any) -> Dict[str, List[float]]:
    """The box around collision parts given as ``{name: {"center", "size"}}`` (or a list of them)."""
    rows = parts.values() if isinstance(parts, dict) else parts
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for r in rows:
        c, sz = (r["center"], r["size"]) if isinstance(r, dict) else r
        for i in range(3):
            lo[i] = min(lo[i], float(c[i]) - float(sz[i]) / 2)
            hi[i] = max(hi[i], float(c[i]) + float(sz[i]) / 2)
    return {"min": lo, "max": hi}


def _shape(name: str, s: Dict[str, Any]) -> Shape:
    """A catalogue entry: height, how wide it stands, and the TCP height above its support for the grasp."""
    if "diagonal_m" in s:  # a cube stands this wide when turned 45 deg
        width = float(s["diagonal_m"])
    elif "width_m" in s or "diameter_m" in s:
        width = float(s.get("width_m", s.get("diameter_m")))
    elif "radius_m" in s:
        width = 2 * float(s["radius_m"])
    elif "size_m" in s:
        width = math.hypot(*[float(v) for v in s["size_m"][:2]])  # a cube turned 45 deg
    else:
        width = float(s.get("grasp_width_m", 0.03))
    grasp = s.get("grasp_height_m", s.get("grasp_tcp_height_m"))
    if grasp is None:
        raise FixtureError(f"shape {name}: no grasp_height_m")
    return Shape(name, float(s["height_m"]), width, float(grasp))


def _first(d: Dict[str, Any], *keys: str) -> Any:
    return next((d[k] for k in keys if d.get(k) is not None), None)


def _stage(drawer_id: str, k: int, st: Dict[str, Any], cabinet: Dict[str, Any], pull: Vec) -> HandleStage:
    """One entry of a drawer's ``handle_stages``; gripper values default to the cabinet's handle_grasp."""
    if "from_m" not in st or "to_m" not in st:
        raise FixtureError(f"{drawer_id} handle_stages[{k}]: needs from_m and to_m")
    s0, s1 = float(st["from_m"]), float(st["to_m"])
    closed = _first(st, "tcp_closed_m", "handle_closed_m")
    tcp_from, tcp_to = _first(st, "tcp_from_m"), _first(st, "tcp_to_m")
    if closed is not None:  # the TCP target with the drawer shut, carried along the pull
        tcp_from = tcp_from or [c + s0 * a for c, a in zip(closed, pull)]
        tcp_to = tcp_to or [c + s1 * a for c, a in zip(closed, pull)]
    a0 = _first(st, "approach_from", "approach")
    standoff = _first(st, "standoff_m", "pregrasp_m", "far_m")
    if tcp_from is None or tcp_to is None or a0 is None or standoff is None:
        raise FixtureError(f"{drawer_id} handle_stages[{k}]: needs tcp_closed_m (or tcp_from_m/tcp_to_m), "
                           "approach, standoff_m")
    a1 = _first(st, "approach_to") or a0
    jaw, in_axis, backoff = _first(st, "jaw_axis", "jaw"), _first(st, "in_axis"), _first(st, "backoff_m")
    preshape = _first(st, "preshape", "preshape_value")
    preshape = preshape if preshape is not None else _first(cabinet, "preshape_value", "release_value")
    close = _first(st, "close", "close_value")
    close = close if close is not None else cabinet.get("close_value")
    release = _first(st, "release", "release_value")
    release = release if release is not None else _first(cabinet, "release_value", "preshape_value")
    if not s1 > s0 >= 0.0:
        raise FixtureError(f"{drawer_id} handle_stages[{k}]: from_m {s0} to_m {s1}")
    return HandleStage(s0, s1, _vec(tcp_from), _vec(tcp_to), _unit(_vec(a0)), _unit(_vec(a1)),  # type: ignore[arg-type]
                       None if jaw is None else _unit(_vec(jaw)), float(standoff),
                       None if backoff is None else float(backoff), None if in_axis is None else _unit(_vec(in_axis)),
                       None if preshape is None else int(preshape), None if close is None else int(close),
                       None if release is None else int(release),
                       exact_jaw=bool(st.get("exact_jaw", st.get("R_base_tool") is not None)),
                       seed_deg=_pose(_first(st, "q_pre_deg", "q_from_deg", "seed_deg")))


def _side_top(row: Dict[str, Any], cabinet: Dict[str, Any], open_m: float) -> Tuple[HandleStage, ...]:
    """sim-scene's v3 calib as written (write_sim_calib_drawers.py): ``handle_grasp`` = {stage_m, side
    {approach, jaw_axis, tcp_closed_m, tcp_stage_m}, top {jaw_axis, approach_at_stage/at_open,
    tcp_stage_m/open_m}}, with the cabinet's side {far_closed_m, backoff_m, gripper values} and top
    {lift_m, open/close/release values}: the side pinch goes in along the tool, the top one comes down."""
    g = row.get("handle_grasp") or {}
    if not isinstance(g, dict) or not {"side", "top", "stage_m"} <= set(g):
        return ()
    side, top, cs, ct = g["side"], g["top"], cabinet.get("side") or {}, cabinet.get("top") or {}
    stage = float(g["stage_m"])
    lift = float(ct.get("lift_m", 0.03))
    try:
        return (
            HandleStage(0.0, stage, _vec(side["tcp_closed_m"]), _vec(side["tcp_stage_m"]),  # type: ignore[arg-type]
                        _unit(_vec(side["approach"])), _unit(_vec(side["approach"])), _unit(_vec(side["jaw_axis"])),
                        float(cs.get("far_closed_m", 0.05)), float(cs.get("backoff_m", 0.03)), None,
                        cs.get("preshape_value"), cs.get("close_value"), cs.get("release_value")),
            HandleStage(stage, open_m, _vec(top["tcp_stage_m"]), _vec(top["tcp_open_m"]),  # type: ignore[arg-type]
                        _unit(_vec(top["approach_at_stage"])), _unit(_vec(top["approach_at_open"])),
                        _unit(_vec(top["jaw_axis"])), lift, lift, (0.0, 0.0, -1.0),
                        ct.get("open_value"), ct.get("close_value"), ct.get("release_value")),
        )
    except KeyError as exc:
        raise FixtureError(f"{row.get('id')} handle_grasp: missing {exc}") from exc


def _leg(drawer_id: str, what: str, raw: Any, with_start: bool) -> TaughtLeg:
    """A leg: a list of q, or {q_deg, tcp_m, speed, from_q_deg}. ``with_start``: the first q is where the leg
    starts (the layout brain-spark posted), not a waypoint (sim-scene's, which gives from_q_deg instead)."""
    if isinstance(raw, dict):
        q, tcp, speed, start = raw.get("q_deg", raw.get("q")), raw.get("tcp_m"), raw.get("speed"), raw.get("from_q_deg")
    else:
        q, tcp, speed, start = raw, None, None, None
    if not isinstance(q, list) or len(q) < 1 or not all(isinstance(a, (list, tuple)) and len(a) == 6 for a in q):
        raise FixtureError(f"{drawer_id} taught {what}: needs q_deg, a list of 6 joint angles each")
    qs = tuple(tuple(float(x) for x in a) for a in q)
    for a, b in zip(qs, qs[1:]):
        if max(abs(x - y) for x, y in zip(a, b)) > 2.0 + 1e-6:
            raise FixtureError(f"{drawer_id} taught {what}: waypoints {a} -> {b} more than 2 deg apart")
    if tcp is not None and (not isinstance(tcp, list) or len(tcp) != len(qs)):
        raise FixtureError(f"{drawer_id} taught {what}: tcp_m must match q_deg one for one")
    tcps = None if tcp is None else tuple(_vec(t) for t in tcp)
    if with_start and len(qs) > 1:
        return TaughtLeg(qs[1:], qs[0], None if tcps is None else tcps[1:], None if speed is None else int(speed))  # type: ignore[arg-type]
    return TaughtLeg(qs, None if start is None else tuple(float(x) for x in start), tcps,  # type: ignore[arg-type]
                     None if speed is None else int(speed))


#: phase names in sim-scene's taught_motions (to_handle = to the standoff, approach = onto the handle) and in
#: brain-spark's posted layout (in = to the standoff, to_handle = onto the handle), to PHASES
_SIM_PHASES = {"to_handle": "standoff", "approach": "approach", "pull": "pull", "push": "pull", "back_off": "back_off",
               "return": "return"}
_BRAIN_LEGS = (("in", "standoff"), ("to_handle", "approach"), ("pull", "pull"), ("back_off", "back_off"))


def _taught(row: Dict[str, Any], cabinet: Dict[str, Any], stages: Sequence[HandleStage]) -> Optional[Taught]:
    """A drawer's taught motions: sim-scene's v3.2 segment lists ({open: [{stage, phase, gripper_before,
    speed, q_deg, tcp_m}], close: [...], routes: {name: {q_deg, from_q_deg, ...}}}), or the stage layout
    brain-spark posted on BOARD (2026-11-02 12:25)."""
    t = row.get("taught") or row.get("taught_motions")
    if not t:
        return None
    did = row["id"]
    seqs: Dict[str, Tuple[TaughtSegment, ...]] = {}
    for verb in ("open", "close"):
        segs: List[TaughtSegment] = []
        for k, e in enumerate(t.get(verb) or ()):
            if "phase" in e:  # sim-scene's segments
                phase = _SIM_PHASES.get(str(e["phase"]))
                if phase is None:
                    raise FixtureError(f"{did} taught {verb}[{k}]: unknown phase {e['phase']!r}")
                stage = None if e.get("stage") is None else int(e["stage"])
                after = None
                if phase == "pull" and stage is not None and stage < len(stages):
                    after = stages[stage].to_m if verb == "open" else stages[stage].from_m
                g = e.get("gripper_before")
                segs.append(TaughtSegment(phase, _leg(did, f"{verb}[{k}]", e, False), stage,
                                          None if g is None else int(g), after))
            elif set(e) == {"out"}:  # brain-spark's layout
                segs.append(TaughtSegment("return", _leg(did, f"{verb} out", e["out"], True)))
            else:
                if "from_m" not in e or "to_m" not in e:
                    raise FixtureError(f"{did} taught {verb}[{k}]: needs from_m and to_m")
                grip = {name: _first(e, name, name + "_value") for name in ("preshape", "close", "release")}
                for name in grip:
                    if grip[name] is None:
                        grip[name] = _first(cabinet, name + "_value")
                for key, phase in _BRAIN_LEGS:
                    if e.get(key) is None:
                        raise FixtureError(f"{did} taught {verb}[{k}]: no {key} leg")
                    g = {"standoff": grip["preshape"], "pull": grip["close"], "back_off": grip["release"]}.get(phase)
                    after = (float(e["to_m"]) if verb == "open" else float(e["from_m"])) if phase == "pull" else None
                    segs.append(TaughtSegment(phase, _leg(did, f"{verb}[{k}] {key}", e[key], True), k,
                                              None if g is None else int(g), after))
        if not segs:
            raise FixtureError(f"{did} taught: no {verb} motions")
        seqs[verb] = tuple(segs)
    routes = {}
    for n, r in (t.get("routes") or {}).items():
        sim = isinstance(r, dict) and ("phase" in r or "from_q_deg" in r)
        routes[str(n)] = _leg(did, f"route {n}", r, not sim)
    return Taught(seqs["open"], seqs["close"], routes)


def from_dict(d: Dict[str, Any]) -> Fixture:
    """The ``cabinet`` / ``catalogue`` / ``handover`` sections of a calib file."""
    try:
        cab = d["cabinet"]
        grasp = cab.get("handle_grasp", {})
        pull, open_m = _unit(_vec(cab["pull_axis"])), float(cab["open_m"])
        box = cab.get("box_m") or _bounds(cab["parts_m"])
        drawers = {}
        for row in cab["drawers"]:
            inner = row["interior_closed_m"]
            # Per-drawer heading: at the drawer's top level, or inside its own handle_grasp.
            own = {**(row.get("handle_grasp") or {}), **{k: row[k] for k in ("approach", "jaw_axis") if k in row}}
            wall_h = inner.get("wall_height_m", (cab.get("interior_m") or {}).get("wall_height"))
            rim = row.get("front_top_z_m")
            if rim is None and row.get("front_panel_z_m"):
                rim = row["front_panel_z_m"][1]
            dr = Drawer(
                id=str(row["id"]), row=int(row["row"]), col=int(row["col"]),
                # v3 writes null for a drawer no plan opens; its handle centre stands in (it is never opened)
                handle_closed=_vec(_first(row, "handle_closed_m", "handle_center_closed_m")),  # type: ignore[arg-type]
                floor_z=float(row["floor_z_m"]),
                interior_center=_vec(inner["center"][:2], 2),  # type: ignore[arg-type]
                interior_size=_vec(inner["size"]),  # type: ignore[arg-type]
                look_pose_deg=_pose(row.get("look_pose_deg")),
                topdown_pose_deg=_pose(row.get("topdown_pose_deg")),
                grasp_approach=_unit(_vec(own["approach"])) if own.get("approach") else None,
                grasp_jaw=_unit(_vec(own["jaw_axis"])) if own.get("jaw_axis") else None,
                rim_z=None if rim is None else float(rim),
                knob=_vec(row["handle_center_closed_m"]) if row.get("handle_center_closed_m") else None,  # type: ignore[arg-type]
                wall_h=None if wall_h is None else float(wall_h),
                panel_z=_vec(row["front_panel_z_m"], 2) if row.get("front_panel_z_m") else None,  # type: ignore[arg-type]
                stages=tuple(_stage(row["id"], k, st, grasp, pull) for k, st in enumerate(row.get("handle_stages") or ()))
                or _side_top(row, grasp, open_m),
                usable=bool(row.get("usable", row.get("openable", not row.get("disabled", False)))),
            )
            dr = dataclasses.replace(dr, taught=_taught(row, grasp, Fixture.stages_of(dr, pull, open_m)))
            if dr.id != f"r{dr.row}c{dr.col}":
                raise FixtureError(f"drawer id {dr.id!r} does not match row {dr.row}, col {dr.col}")
            drawers[dr.id] = dr
        cat = d["catalogue"]
        shapes = {name: _shape(name, s) for name, s in cat["shapes"].items()}
        hand = d["handover"]
        jaw = grasp.get("jaw_axis")
        pick_jaw = cab.get("pick_jaw_axis")
        if pick_jaw is None and (d.get("pick") or {}).get("jaw_yaw_deg") is not None:
            yaw = math.radians(float(d["pick"]["jaw_yaw_deg"]))
            pick_jaw = (math.cos(yaw), math.sin(yaw), 0.0)
        # No cabinet-wide heading (it differs per column): straight into the face as the fallback.
        approach = grasp.get("approach") or (-pull[0], -pull[1], -pull[2])
        open_value = grasp.get("preshape_value", grasp.get("release_value"))
        pick_above = (d.get("pick") or {}).get("approach_tcp_height_above_support_m")
        outer = cab.get("drawer_outer_m") or {}
        return Fixture(
            drawers=drawers,
            pull_axis=pull,
            open_m=open_m,
            box_min=_vec(box["min"]),  # type: ignore[arg-type]
            box_max=_vec(box["max"]),  # type: ignore[arg-type]
            cabinet_look_pose_deg=_pose(d.get("cabinet_look_pose_deg", cab.get("look_pose_deg"))) or [],
            grasp_approach=_unit(_vec(approach)),
            grasp_jaw=None if jaw is None else _unit(_vec(jaw)),
            pregrasp_m=float(grasp.get("pregrasp_m", 0.04)),
            handle_close_value=None if grasp.get("close_value") is None else int(grasp["close_value"]),
            pick_jaw=None if pick_jaw is None else _unit(_vec(pick_jaw)),
            colours=[str(c) for c in cat["colours"]],
            shapes=shapes,
            handover_center=_vec(hand["center_m"][:2], 2),  # type: ignore[arg-type]
            handover_size=_vec(hand["size_m"][:2], 2),  # type: ignore[arg-type]
            handle_open_value=None if open_value is None else int(open_value),
            pick_above_m=None if pick_above is None else float(pick_above),
            far_m=None if grasp.get("far_m") is None else float(grasp["far_m"]),
            handle_protrusion_m=float((cab.get("handle_m") or {}).get("protrusion", 0.0)),
            handle_size=(None if not {"protrusion", "width", "height"} <= set(cab.get("handle_m") or {}) else
                         tuple(float(cab["handle_m"][k]) for k in ("protrusion", "width", "height"))),  # type: ignore[arg-type]
            drawer_outer=(float(outer["width"]), float(outer["depth"])) if {"width", "depth"} <= set(outer) else None,
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise FixtureError(f"cabinet calib: {exc!r}") from exc
