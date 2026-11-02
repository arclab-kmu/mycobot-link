"""The arm's own body as capsules, against itself and against the scene's boxes.

calib v3's ``robot_collision`` (sim-scene) gives one or more capsules per link,
in the chain's frames, each containing every point of that link's collision
hull, and the pairs of links that can meet (two or more joints apart; the
camera and the gripper against joints 1-3). A capsule is a segment and a
radius, so every check is a segment distance: segment to segment for the arm
against itself, segment to box for the arm against the fixture. D3 on 2eb3e4c
checked only the TCP, pad tips, fingers and camera against the cabinet, and the
wrist camera went through the base (BOARD 2026-11-02 08:40).

Paths are checked every ``CHECK_STEP_DEG`` per joint: the sim moves all joints
linearly to arrive together, so between two checked angles a point of the arm
moves at most a few mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from mycobot_brain.kinematics import Chain, Mat, Vec, apply, inverse, joint_path

CHECK_STEP_DEG = 2.0
#: links that close on what is grasped (a handle, an object on a floor): the parallel gripper's pads.
FINGER_WORDS = ("finger", "pad", "left3", "right3")
#: the hand: what reaches in to a handle or a drawer floor with the pads (gripper body, camera, pads).
HAND_WORDS = ("gripper", "camera")


class CollisionSpecError(ValueError):
    """``robot_collision`` is missing something or names a frame the chain does not have."""


@dataclass(frozen=True)
class Capsule:
    name: str  # the link it belongs to; a link may have several
    frame: str
    a: Vec
    b: Vec
    radius: float


@dataclass(frozen=True)
class Hull:
    """A collider's convex hull: its vertices in a chain frame (v3.1 ``robot_collision.hulls``), and a
    sphere around them, so a hull far from every box costs one point transform."""

    name: str
    frame: str
    vertices: Tuple[Vec, ...]
    center: Vec = (0.0, 0.0, 0.0)
    radius: float = 0.0
    #: a pad: ``vertices`` at the closed jaw, moved ``travel_m`` along ``travel_axis`` (its frame) when open
    travel_axis: Optional[Vec] = None
    travel_m: float = 0.0

    def at(self, jaw: Optional[float]) -> Tuple[List[Vec], Vec, float]:
        """Vertices, sphere centre and radius with the jaw ``jaw`` open (0 shut, 1 open; None: unknown, so
        the hull swept over the whole travel)."""
        if self.travel_axis is None:
            return list(self.vertices), self.center, self.radius
        ax = [a * self.travel_m for a in self.travel_axis]
        if jaw is None:
            moved = [(x + ax[0], y + ax[1], z + ax[2]) for x, y, z in self.vertices]
            mid = tuple(c + a / 2 for c, a in zip(self.center, ax))
            return list(self.vertices) + moved, mid, self.radius + self.travel_m / 2  # type: ignore[return-value]
        k = min(max(jaw, 0.0), 1.0)
        return ([(x + k * ax[0], y + k * ax[1], z + k * ax[2]) for x, y, z in self.vertices],
                tuple(c + k * a for c, a in zip(self.center, ax)), self.radius)  # type: ignore[return-value]


@dataclass(frozen=True)
class Placed:
    """A capsule at one set of joint angles, in the base frame."""

    capsule: Capsule
    a: Vec
    b: Vec


@dataclass
class RobotBody:
    capsules: List[Capsule]
    pairs: List[Tuple[str, str]]
    fingers: FrozenSet[str]
    #: links fixed to the base frame (the base itself): they cannot move into the fixture.
    still: FrozenSet[str]
    #: the hand bodies, which may come closer to the fixture while they grasp (capsule rule, no hulls)
    hand: FrozenSet[str] = frozenset()
    #: exact convex hulls of the colliders: with them the fixture is checked on the hulls (leader 12:05)
    hulls: List[Hull] = field(default_factory=list)
    #: links given as a box instead of capsules (v3's ``base_box``: the base hull's exact bounds; its
    #: capsules overshoot it 10-30 mm): name -> (base frame -> box frame, box lo, box hi). Fixed links only.
    boxes: Dict[str, Tuple[Mat, Vec, Vec]] = field(default_factory=dict)

    def place(self, chain: Chain, q_deg: Sequence[float]) -> List[Placed]:
        frames = chain.frames(q_deg)
        return [Placed(c, apply(frames[c.frame], c.a), apply(frames[c.frame], c.b)) for c in self.capsules]

    def hull_gaps(self, chain: Chain, q_deg: Sequence[float],
                  boxes: Sequence[Tuple[str, Sequence[float], Sequence[float]]], margin: float,
                  skip: Callable[[str, str], bool], jaw: Optional[float] = None) -> Optional[Tuple[str, str, float]]:
        """The first moving collider whose hull at ``q_deg`` (pads with the jaw ``jaw`` open, see
        :meth:`Hull.at`) comes closer than ``margin`` to one of ``boxes`` (name, lo, hi) that
        ``skip(collider, box)`` does not excuse: (collider, box, gap); None if none."""
        frames = chain.frames(q_deg)
        for h in self.hulls:
            if h.name in self.still:
                continue
            t = frames[h.frame]
            r0, r1, r2 = t[0], t[1], t[2]
            verts, center, radius = h.at(jaw)
            c = apply(t, center)
            pts: Optional[List[Vec]] = None
            for box, lo, hi in boxes:
                if skip(h.name, box) or _point_box(c, lo, hi) - radius >= margin:
                    continue
                if pts is None:
                    pts = [(r0[0] * x + r0[1] * y + r0[2] * z + r0[3], r1[0] * x + r1[1] * y + r1[2] * z + r1[3],
                            r2[0] * x + r2[1] * y + r2[2] * z + r2[3]) for x, y, z in verts]
                gap = hull_box_distance(pts, lo, hi, within=margin)
                if gap < margin:
                    return h.name, box, gap
        return None

    def self_hit(self, placed: Sequence[Placed], margin: float) -> Optional[str]:
        """The first listed pair closer than ``margin``, as words; None if all clear."""
        for x, y, gap in self.pair_gaps(placed, margin):
            return f"{x} {gap * 1000:.0f} mm from {y}"
        return None

    def pair_gaps(self, placed: Sequence[Placed], margin: float) -> List[Tuple[str, str, float]]:
        """Every listed pair closer than ``margin``, with its gap (m, negative when they overlap)."""
        by: Dict[str, List[Placed]] = {}
        for p in placed:
            by.setdefault(p.capsule.name, []).append(p)
        out = []
        for x, y in self.pairs:
            if x in self.boxes and y in self.boxes:
                continue  # two fixed boxes never move
            if y in self.boxes:
                x, y = y, x
            if x in self.boxes:
                to_box, lo, hi = self.boxes[x]
                gaps = [box_distance(apply(to_box, r.a), apply(to_box, r.b), lo, hi, within=r.capsule.radius + margin)
                        - r.capsule.radius for r in by.get(y, ())]
            else:
                gaps = [segment_distance(p.a, p.b, r.a, r.b) - p.capsule.radius - r.capsule.radius
                        for p in by.get(x, ()) for r in by.get(y, ())]
            worst = min(gaps, default=math.inf)
            if worst < margin:
                out.append((x, y, worst))
        return out


# -- distances -------------------------------------------------------------------


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def segment_distance(p1: Sequence[float], q1: Sequence[float], p2: Sequence[float], q2: Sequence[float]) -> float:
    """Closest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1, d2, r = _sub(q1, p1), _sub(q2, p2), _sub(p1, p2)
    a, e, f = _dot(d1, d1), _dot(d2, d2), _dot(d2, r)
    eps = 1e-12
    if a <= eps and e <= eps:
        s = t = 0.0
    elif a <= eps:
        s, t = 0.0, _clamp(f / e)
    else:
        c = _dot(d1, r)
        if e <= eps:
            s, t = _clamp(-c / a), 0.0
        else:
            b = _dot(d1, d2)
            denom = a * e - b * b
            s = _clamp((b * f - c * e) / denom) if denom > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                s, t = _clamp(-c / a), 0.0
            elif t > 1.0:
                s, t = _clamp((b - c) / a), 1.0
    c1 = (p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s)
    c2 = (p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t)
    return math.dist(c1, c2)


def _point_box(p: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> float:
    dx = max(lo[0] - p[0], 0.0, p[0] - hi[0])
    dy = max(lo[1] - p[1], 0.0, p[1] - hi[1])
    dz = max(lo[2] - p[2], 0.0, p[2] - hi[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def box_distance(a: Sequence[float], b: Sequence[float], lo: Sequence[float], hi: Sequence[float],
                 within: float = math.inf) -> float:
    """Closest distance from segment a-b to the axis-aligned box lo-hi (0 if they meet).

    The distance from a point on the segment to the box is convex along the segment, so a golden-section
    search finds its minimum. Anything farther than ``within`` may come back as any value above it.
    """
    # Cheap way out: the segment's own bounding box is farther than ``within`` from the box.
    gap = 0.0
    for i in range(3):
        g = max(lo[i] - max(a[i], b[i]), 0.0, min(a[i], b[i]) - hi[i])
        gap += g * g
    if math.sqrt(gap) >= within:
        return math.sqrt(gap)
    d = _sub(b, a)

    def f(t: float) -> float:
        return _point_box((a[0] + d[0] * t, a[1] + d[1] * t, a[2] + d[2] * t), lo, hi)

    g = (math.sqrt(5.0) - 1.0) / 2.0
    x0, x1 = 0.0, 1.0
    u, v = x1 - g * (x1 - x0), x0 + g * (x1 - x0)
    fu, fv = f(u), f(v)
    for _ in range(40):
        if fu <= fv:
            x1, v, fv = v, u, fu
            u = x1 - g * (x1 - x0)
            fu = f(u)
        else:
            x0, u, fu = u, v, fv
            v = x0 + g * (x1 - x0)
            fv = f(v)
        if fu == 0.0 or fv == 0.0:
            return 0.0
    return min(fu, fv, f(0.0), f(1.0))


def _closest_segment(a: Vec, b: Vec) -> Tuple[Vec, List[Vec]]:
    ab = _sub(b, a)
    den = _dot(ab, ab)
    t = -_dot(a, ab) / den if den > 1e-30 else 0.0
    if t <= 0.0:
        return a, [a]
    if t >= 1.0:
        return b, [b]
    return (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]), [a, b]


def _closest_triangle(a: Vec, b: Vec, c: Vec) -> Tuple[Vec, List[Vec]]:
    """The point of triangle abc nearest the origin, and the corners that span it (Ericson 5.1.5)."""
    ab, ac = _sub(b, a), _sub(c, a)
    ap = (-a[0], -a[1], -a[2])
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, [a]
    bp = (-b[0], -b[1], -b[2])
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, [b]
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return (a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2]), [a, b]
    cp = (-c[0], -c[1], -c[2])
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, [c]
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return (a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2]), [a, c]
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        bc = _sub(c, b)
        return (b[0] + w * bc[0], b[1] + w * bc[1], b[2] + w * bc[2]), [b, c]
    den = va + vb + vc
    if abs(den) < 1e-30:  # degenerate: a sliver; its longest edge will do
        return min((_closest_segment(x, y) for x, y in ((a, b), (b, c), (a, c))), key=lambda r: _dot(r[0], r[0]))
    v, w = vb / den, vc / den
    return (a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w), [a, b, c]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _closest_simplex(s: List[Vec]) -> Tuple[Vec, List[Vec]]:
    """The point of the simplex (1-4 points) nearest the origin, and the smallest sub-simplex holding it."""
    if len(s) == 1:
        return s[0], s
    if len(s) == 2:
        return _closest_segment(s[0], s[1])
    if len(s) == 3:
        return _closest_triangle(s[0], s[1], s[2])
    best: Optional[Tuple[Vec, List[Vec]]] = None
    a, b, c, d = s
    for (x, y, z), w in (((a, b, c), d), ((a, c, d), b), ((a, d, b), c), ((b, d, c), a)):
        n = _cross(_sub(y, x), _sub(z, x))
        side_o, side_w = -_dot(x, n), _dot(_sub(w, x), n)
        if side_o * side_w < 0.0 or abs(side_w) < 1e-30:  # the origin is beyond this face (or it is flat)
            r = _closest_triangle(x, y, z)
            if best is None or _dot(r[0], r[0]) < _dot(best[0], best[0]):
                best = r
    if best is None:
        return (0.0, 0.0, 0.0), s  # inside the tetrahedron
    return best


def hull_box_distance(points: Sequence[Vec], lo: Sequence[float], hi: Sequence[float],
                      within: float = math.inf) -> float:
    """Distance from the convex hull of ``points`` to the axis-aligned box lo-hi (0 if they meet), by GJK
    on their Minkowski difference. Anything farther than ``within`` may come back as any value above it."""
    mn = [min(p[i] for p in points) for i in range(3)]
    mx = [max(p[i] for p in points) for i in range(3)]
    gap = math.sqrt(sum(max(lo[i] - mx[i], 0.0, mn[i] - hi[i]) ** 2 for i in range(3)))
    if gap >= within:
        return gap

    def support(d: Sequence[float]) -> Vec:
        best, bd = points[0], -math.inf
        for p in points:
            v = p[0] * d[0] + p[1] * d[1] + p[2] * d[2]
            if v > bd:
                best, bd = p, v
        # minus the box's support in -d
        return (best[0] - (lo[0] if d[0] > 0 else hi[0]), best[1] - (lo[1] if d[1] > 0 else hi[1]),
                best[2] - (lo[2] if d[2] > 0 else hi[2]))

    c = [(lo[i] + hi[i]) / 2 - (mn[i] + mx[i]) / 2 for i in range(3)]
    v = support(c if any(abs(x) > 1e-12 for x in c) else (1.0, 0.0, 0.0))
    simplex = [v]
    for _ in range(64):
        vv = _dot(v, v)
        if vv < 1e-18:
            return 0.0
        w = support((-v[0], -v[1], -v[2]))
        vw = _dot(v, w)
        if vw > 0.0 and vw * vw >= within * within * vv:
            return vw / math.sqrt(vv)  # a lower bound past ``within`` already
        if vv - vw <= 1e-12 + 1e-9 * vv:
            return math.sqrt(vv)
        simplex.append(w)
        v, simplex = _closest_simplex(simplex)
    return math.sqrt(_dot(v, v))


def densify(path: Sequence[Sequence[float]], step_deg: float = CHECK_STEP_DEG) -> List[List[float]]:
    """``path`` (its first entry is where the arm starts) with angles in between, at most ``step_deg``
    apart per joint: what the arm passes through, as the sim moves joints linearly between waypoints."""
    if not path:
        return []
    out = [list(path[0])]
    for q in path[1:]:
        out += joint_path(out[-1], q, step_deg)
    return out


# -- the calib -------------------------------------------------------------------


def _xyz(spec: Dict[str, Any], *keys: str) -> Optional[Vec]:
    for k in keys:
        if k in spec:
            v = spec[k]
            if not (isinstance(v, (list, tuple)) and len(v) == 3):
                raise CollisionSpecError(f"{k}: need 3 numbers, got {v!r}")
            return (float(v[0]), float(v[1]), float(v[2]))
    return None


def _capsule(name: str, spec: Dict[str, Any], frames: Sequence[str]) -> Capsule:
    frame = str(spec.get("frame", spec.get("parent_frame", spec.get("link", name))))
    if frame not in frames:
        raise CollisionSpecError(f"capsule {name}: frame {frame!r} is not one of the chain's {list(frames)}")
    a, b = _xyz(spec, "a_m", "p0_m", "a"), _xyz(spec, "b_m", "p1_m", "b")
    if a is None or b is None:
        ends = spec.get("segment_m", spec.get("points_m"))
        if not (isinstance(ends, list) and len(ends) == 2):
            raise CollisionSpecError(f"capsule {name}: needs a_m and b_m (or segment_m: [a, b])")
        a, b = _xyz({"a": ends[0]}, "a"), _xyz({"b": ends[1]}, "b")
    r = spec.get("radius_m", spec.get("r_m", spec.get("radius")))
    if r is None or float(r) < 0:
        raise CollisionSpecError(f"capsule {name}: needs radius_m >= 0")
    return Capsule(name, frame, a, b, float(r))  # type: ignore[arg-type]


def from_dict(spec: Dict[str, Any], chain: Chain) -> RobotBody:
    """``{"capsules": [{"name", "frame", "a_m", "b_m", "radius_m"}, ...], "pairs": [[name, name], ...]}``.

    ``frame`` is one of the chain's link frames (``joint1`` = the base .. ``joint6_flange``) or a frame
    fixed to them that the calib lists (``tool_mount``, ``tcp``, ``camera`` ...). ``capsules`` may
    instead map a link name to one capsule or a list of them. Without ``pairs``, every two links two
    or more joints apart. ``fingers`` names the links that close on a grasp (else by name: pads);
    ``hand`` the hand bodies (else the fingers and the gripper/camera links).
    ``base_box`` (``{"name", "frame", "center_m", "size_m"}``, a link fixed to the base) replaces that
    link's capsules with its exact box. ``hulls`` (``[{"name", "frame", "vertices_m"}]``) are the
    colliders' convex hulls, for the fixture check.
    """
    frames = chain.frame_names()
    raw = spec.get("capsules")
    rows: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(raw, dict):
        for name, v in raw.items():
            rows += [(str(name), c) for c in (v if isinstance(v, list) else [v])]
    elif isinstance(raw, list):
        rows = [(str(c.get("name", c.get("link", ""))), c) for c in raw]
    else:
        raise CollisionSpecError("needs capsules")
    capsules = [_capsule(n, c, frames) for n, c in rows]
    if any(not c.name for c in capsules):
        raise CollisionSpecError("every capsule needs a name (its link)")
    names = list(dict.fromkeys(c.name for c in capsules))
    depth = {c.name: chain.frame_depth(c.frame) for c in capsules}
    if "pairs" in spec:
        pairs = []
        for p in spec["pairs"]:
            x, y = (p["a"], p["b"]) if isinstance(p, dict) else p
            for n in (x, y):
                if n not in depth:
                    raise CollisionSpecError(f"pair {x}-{y}: no capsule named {n!r}")
            pairs.append((str(x), str(y)))
    else:
        pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:] if abs(depth[x] - depth[y]) >= 2]
    fingers = spec.get("fingers")
    finger_set = frozenset(fingers) if fingers is not None else frozenset(
        n for n in names if any(w in n.lower() for w in FINGER_WORDS))
    still = frozenset(n for n in names if depth[n] == 0)
    boxes: Dict[str, Tuple[Mat, Vec, Vec]] = {}
    for b in [spec["base_box"]] if spec.get("base_box") else []:
        name, frame = str(b.get("name", "joint1")), str(b.get("frame", "joint1"))
        if frame not in frames or chain.frame_depth(frame) != 0:
            raise CollisionSpecError(f"base_box: frame {frame!r} is not fixed to the base")
        c, size = _xyz(b, "center_m"), _xyz(b, "size_m")
        if c is None or size is None:
            raise CollisionSpecError("base_box: needs center_m and size_m")
        lo = (c[0] - size[0] / 2, c[1] - size[1] / 2, c[2] - size[2] / 2)
        hi = (c[0] + size[0] / 2, c[1] + size[1] / 2, c[2] + size[2] / 2)
        boxes[name] = (inverse(chain.frames([0.0] * len(chain.joints))[frame]), lo, hi)  # type: ignore[assignment]
        # The box stands in for the link's capsules; it never moves, so it is never checked against the fixture.
        capsules = [cap for cap in capsules if cap.name != name]
        still = still | {name}
    hand = spec.get("hand")
    hand_set = frozenset(hand) if hand is not None else finger_set | frozenset(
        n for n in names if any(w in n.lower() for w in HAND_WORDS))
    hulls = [_hull(n, h, frames) for n, h in _named(spec.get("hulls"))]
    for h in hulls:
        if chain.frame_depth(h.frame) == 0:
            still = still | {h.name}
    return RobotBody(capsules, pairs, finger_set, still, boxes=boxes, hand=hand_set, hulls=hulls)


def _named(raw: Any) -> List[Tuple[str, Any]]:
    """``[{name, ...}, ...]`` or ``{name: entry or [entries]}`` as (name, entry) pairs."""
    if not raw:
        return []
    if isinstance(raw, dict):
        return [(str(n), e) for n, v in raw.items() for e in (v if isinstance(v, list) and v and isinstance(v[0], dict)
                                                             else [v])]
    return [(str(e.get("name", e.get("link", ""))), e) for e in raw]


def _hull(name: str, spec: Any, frames: Sequence[str]) -> Hull:
    if isinstance(spec, list):  # bare vertices: the link's own frame
        spec = {"vertices_m": spec}
    frame = str(spec.get("frame", spec.get("link", name)))
    if not name or frame not in frames:
        raise CollisionSpecError(f"hull {name!r}: frame {frame!r} is not one of the chain's {list(frames)}")
    pts = spec.get("vertices_m", spec.get("vertices", spec.get("points_m")))
    if not isinstance(pts, list) or not pts or not all(isinstance(v, (list, tuple)) and len(v) == 3 for v in pts):
        raise CollisionSpecError(f"hull {name}: needs vertices_m, a list of [x, y, z]")
    v = tuple((float(p[0]), float(p[1]), float(p[2])) for p in pts)
    c = tuple((min(p[i] for p in v) + max(p[i] for p in v)) / 2 for i in range(3))
    axis = _xyz(spec, "travel_axis")
    travel = float(spec.get("travel_m", spec.get("jaw_travel_m", 0.0)))
    if axis is not None:
        n = math.sqrt(sum(a * a for a in axis))
        if n < 1e-9 or travel <= 0.0:
            raise CollisionSpecError(f"hull {name}: travel_axis needs a length and travel_m > 0")
        axis = (axis[0] / n, axis[1] / n, axis[2] / n)
    return Hull(name, frame, v, c, max(math.dist(p, c) for p in v), axis, travel)  # type: ignore[arg-type]

