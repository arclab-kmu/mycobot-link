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
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from mycobot_brain.kinematics import Chain, Vec, apply, joint_path

CHECK_STEP_DEG = 2.0
#: links that close on what is grasped (a handle, an object on a floor): the parallel gripper's pads.
FINGER_WORDS = ("finger", "pad", "left3", "right3")


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

    def place(self, chain: Chain, q_deg: Sequence[float]) -> List[Placed]:
        frames = chain.frames(q_deg)
        return [Placed(c, apply(frames[c.frame], c.a), apply(frames[c.frame], c.b)) for c in self.capsules]

    def self_hit(self, placed: Sequence[Placed], margin: float) -> Optional[str]:
        """The first listed pair closer than ``margin``, as words; None if all clear."""
        by: Dict[str, List[Placed]] = {}
        for p in placed:
            by.setdefault(p.capsule.name, []).append(p)
        for x, y in self.pairs:
            for p in by.get(x, ()):
                for r in by.get(y, ()):
                    gap = segment_distance(p.a, p.b, r.a, r.b) - p.capsule.radius - r.capsule.radius
                    if gap < margin:
                        return f"{x} {gap * 1000:.0f} mm from {y}"
        return None


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
    or more joints apart. ``fingers`` names the links that close on a grasp (else by name: pads).
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
    return RobotBody(capsules, pairs, finger_set, still)

