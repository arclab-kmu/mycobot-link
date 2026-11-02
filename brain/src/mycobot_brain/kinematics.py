"""Forward and inverse kinematics, and pixels to points on a known plane.

Pure Python (4x4 lists), because the brain stays stdlib-only and six joints
do not need numpy. Joint values crossing this module's boundary are
**firmware degrees**, the numbers ``send_angles`` takes and ``state`` reports;
each joint maps them to its model angle as ``sign * (q - offset)``.

There is no depth image anywhere: a pixel becomes a ray from the camera (K and
the camera pose from FK of the *measured* angles), and the ray meets a plane
whose height is known (table + half a block, or the box rim). The same code
works on the real arm, which has no depth camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec = Tuple[float, float, float]
Mat = List[List[float]]  # 4x4 homogeneous

# -- small linear algebra ------------------------------------------------------


def identity() -> Mat:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def rpy_matrix(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def quat_matrix(x: float, y: float, z: float, w: float) -> List[List[float]]:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def axis_angle(axis: Sequence[float], angle: float) -> List[List[float]]:
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s, v = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return [
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ]


def transform(rot: Optional[Sequence[Sequence[float]]] = None, xyz: Sequence[float] = (0.0, 0.0, 0.0)) -> Mat:
    rot = rot or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return [[*map(float, rot[0]), float(xyz[0])], [*map(float, rot[1]), float(xyz[1])],
            [*map(float, rot[2]), float(xyz[2])], [0.0, 0.0, 0.0, 1.0]]


def mul(a: Mat, b: Mat) -> Mat:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def inverse(t: Mat) -> Mat:
    """Rigid transforms only."""
    r = [[t[j][i] for j in range(3)] for i in range(3)]
    p = [-sum(r[i][k] * t[k][3] for k in range(3)) for i in range(3)]
    return transform(r, p)


def apply(t: Mat, p: Sequence[float]) -> Vec:
    return tuple(t[i][0] * p[0] + t[i][1] * p[1] + t[i][2] * p[2] + t[i][3] for i in range(3))  # type: ignore[return-value]


def rotate(t: Mat, v: Sequence[float]) -> Vec:
    return tuple(t[i][0] * v[0] + t[i][1] * v[1] + t[i][2] * v[2] for i in range(3))  # type: ignore[return-value]


def position(t: Mat) -> Vec:
    return (t[0][3], t[1][3], t[2][3])


def column(t: Mat, j: int) -> Vec:
    return (t[0][j], t[1][j], t[2][j])


def cross(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Sequence[float]) -> float:
    return math.sqrt(dot(a, a))


def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    """Gaussian elimination with partial pivoting; ``a`` is square and well conditioned (damped)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][k] * x[k] for k in range(r + 1, n))) / m[r][r]
    return x


# -- the chain -----------------------------------------------------------------


@dataclass(frozen=True)
class Joint:
    """A revolute joint, URDF style: ``origin`` places the joint frame in its
    parent at zero angle, ``axis`` is in the joint frame. The model angle is
    ``sign * (firmware_deg - offset_deg)``, the sim calib's convention."""

    origin: Mat
    axis: Vec = (0.0, 0.0, 1.0)
    sign: float = 1.0
    offset_deg: float = 0.0

    def model_rad(self, firmware_deg: float) -> float:
        return math.radians(self.sign * (firmware_deg - self.offset_deg))


@dataclass
class Chain:
    joints: Sequence[Joint]
    #: base frame of the joints in the frame xyz answers are given in (the robot base).
    base: Mat = field(default_factory=identity)
    #: last joint frame → flange.
    flange: Mat = field(default_factory=identity)
    #: flange → each named tool frame ("camera" is the optical frame: z forward, x right, y down).
    tools: Dict[str, Mat] = field(default_factory=dict)
    #: link frame names: the base, then the child of each joint (the sim calib's URDF names).
    links: Sequence[str] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint6_flange")
    #: more frames welded to the arm: name → (parent frame, parent → this), parents listed first.
    fixed: Dict[str, Tuple[str, Mat]] = field(default_factory=dict)

    def tool(self, name: str) -> Mat:
        return identity() if name == "flange" else self.tools[name]

    def frames(self, q_deg: Sequence[float]) -> Dict[str, Mat]:
        """Every named frame in the base frame: links, ``base``, ``flange``, tools and fixed frames."""
        if len(q_deg) != len(self.joints):
            raise ValueError(f"need {len(self.joints)} joint angles, got {len(q_deg)}")
        t = self.base
        out = {"base": t, self.links[0]: t}
        for name, joint, q in zip(self.links[1:], self.joints, q_deg):
            t = mul(mul(t, joint.origin), transform(axis_angle(joint.axis, joint.model_rad(q))))
            out[name] = t
        flange = mul(t, self.flange)
        out["flange"] = flange
        for name, tool in self.tools.items():
            out[name] = mul(flange, tool)
        for name, (parent, tf) in self.fixed.items():
            out[name] = mul(out[parent], tf)
        return out

    def frame_names(self) -> List[str]:
        return list(dict.fromkeys(["base", *self.links, "flange", *self.tools, *self.fixed]))

    def frame_depth(self, name: str) -> int:
        """How many joints lie between the base and frame ``name`` (0: the base)."""
        if name == "base":
            return 0
        if name in self.links:
            return list(self.links).index(name)
        if name in self.fixed:
            return self.frame_depth(self.fixed[name][0])
        return len(self.joints)

    def _walk(self, q_deg: Sequence[float]) -> Tuple[List[Tuple[Vec, Vec, float]], Mat]:
        """Each joint's world origin, world axis and sign; then base → flange."""
        if len(q_deg) != len(self.joints):
            raise ValueError(f"need {len(self.joints)} joint angles, got {len(q_deg)}")
        t = self.base
        axes = []
        for joint, q in zip(self.joints, q_deg):
            t = mul(t, joint.origin)
            axes.append((position(t), rotate(t, joint.axis), joint.sign))
            t = mul(t, transform(axis_angle(joint.axis, joint.model_rad(q))))
        return axes, mul(t, self.flange)

    def pose(self, q_deg: Sequence[float], tool: str = "flange") -> Mat:
        return mul(self._walk(q_deg)[1], self.tool(tool))

    def jacobian(self, q_deg: Sequence[float], tool: str = "flange") -> Tuple[Mat, List[List[float]], List[List[float]]]:
        """Tool pose, linear and angular Jacobians (3 x n) per firmware *radian*."""
        axes, flange = self._walk(q_deg)
        t = mul(flange, self.tool(tool))
        p = position(t)
        jv: List[List[float]] = [[], [], []]
        jw: List[List[float]] = [[], [], []]
        for origin, axis, sign in axes:
            w = tuple(sign * a for a in axis)
            v = cross(w, tuple(pi - oi for pi, oi in zip(p, origin)))
            for i in range(3):
                jv[i].append(v[i])
                jw[i].append(w[i])
        return t, jv, jw


#: myCobot 280 JetsonNano as Elephant's ``mycobot_280_jn`` URDF lays it out;
#: firmware degrees equal URDF degrees (the vendor ROS driver sends
#: ``degrees(joint_state)`` to ``send_angles`` unchanged), all zeros is the arm
#: upright. TODO(unverified): numbers recalled from the vendor URDF, not read
#: from a file here; the sim calib file overrides all of it.
MYCOBOT_280_JN = Chain(
    joints=[
        Joint(transform(rpy_matrix(0, 0, 0), (0, 0, 0.13156))),
        Joint(transform(rpy_matrix(0, math.pi / 2, -math.pi / 2), (0, 0, 0))),
        Joint(transform(rpy_matrix(0, 0, 0), (-0.1104, 0, 0))),
        Joint(transform(rpy_matrix(0, 0, -math.pi / 2), (-0.096, 0, 0.06462))),
        Joint(transform(rpy_matrix(math.pi / 2, -math.pi / 2, 0), (0, -0.07318, 0))),
        Joint(transform(rpy_matrix(-math.pi / 2, 0, 0), (0, 0.0456, 0))),
    ],
)

#: Brain-side copy of the arm server's limits less its 3 deg margin and half a
#: degree more, so IK never proposes what the server would refuse.
JOINT_LIMITS_DEG = ((-168, 168), (-140, 140), (-150, 150), (-150, 150), (-155, 160), (-180, 180))
IK_MARGIN_DEG = 3.5


class IKError(RuntimeError):
    """No joint solution near the seed reaches the target."""


def solve_ik(
    chain: Chain,
    seed_deg: Sequence[float],
    target: Sequence[float],
    *,
    tool: str = "tcp",
    approach: Optional[Sequence[float]] = (0.0, 0.0, -1.0),
    jaw: Optional[Sequence[float]] = None,
    either_jaw: bool = True,
    hold: Sequence[int] = (),
    limits: Sequence[Tuple[float, float]] = JOINT_LIMITS_DEG,
    margin: float = IK_MARGIN_DEG,
    tol_m: float = 5e-4,
    tol_deg: float = 0.5,
    iters: int = 400,
    damping: float = 0.03,
) -> List[float]:
    """Damped least squares from ``seed_deg`` to put ``tool`` at ``target``
    with its z axis along ``approach`` (None: position only).

    Rotation about the approach axis is left free and the step is minimum
    norm, so the solution stays near the seed: from the look pose that keeps
    the elbow branch the arm is already in, and J6 roughly where it was.
    ``hold`` lists joints (1-based) kept at the seed: a camera off the flange
    axis would otherwise use J6 to slide itself sideways.

    ``jaw`` (with ``approach``) also fixes the tool x axis, the line the jaws
    close along: a drawer handle has to sit across the jaws, not along them.
    The two jaws are alike, so ``jaw`` and ``-jaw`` are the same grasp: the one
    nearer the seed is tried first, then the other (one can be past J6's limit).
    ``either_jaw=False`` keeps the sign: for the camera, x is the image's right.
    ``jaw`` without ``approach`` fixes only the jaw axis: pads pinching two
    faces may turn about the line through them (a knob gripped top and bottom
    lets the gripper's heading drift as the drawer slides out).
    """
    if jaw is not None and either_jaw:
        first = jaw if dot(column(chain.pose(seed_deg, tool), 0), jaw) >= 0 else tuple(-j for j in jaw)
        kw = dict(tool=tool, approach=approach, hold=hold, limits=limits, margin=margin, tol_m=tol_m,
                  tol_deg=tol_deg, iters=iters, damping=damping)
        try:
            return _solve_ik(chain, seed_deg, target, jaw=first, **kw)
        except IKError:
            if approach is None:
                raise  # jaw only: the minimum-norm step from the seed already turns whichever way is shorter
            return _solve_ik(chain, seed_deg, target, jaw=tuple(-j for j in first), **kw)
    return _solve_ik(chain, seed_deg, target, tool=tool, approach=approach, jaw=jaw, hold=hold, limits=limits,
                     margin=margin, tol_m=tol_m, tol_deg=tol_deg, iters=iters, damping=damping)


def _solve_ik(
    chain: Chain,
    seed_deg: Sequence[float],
    target: Sequence[float],
    *,
    tool: str,
    approach: Optional[Sequence[float]],
    jaw: Optional[Sequence[float]],
    hold: Sequence[int],
    limits: Sequence[Tuple[float, float]],
    margin: float,
    tol_m: float,
    tol_deg: float,
    iters: int,
    damping: float,
) -> List[float]:
    q = [math.radians(a) for a in seed_deg]
    lo = [math.radians(a + margin) for a, _ in limits]
    hi = [math.radians(b - margin) for _, b in limits]
    if approach is None and jaw is not None:
        n = norm(jaw)
        jaw = tuple(j / n for j in jaw)
    elif approach is not None:
        n = norm(approach)
        approach = tuple(a / n for a in approach)
        if jaw is not None:
            # Only the part of jaw across the approach can be met.
            along = dot(jaw, approach)
            jaw = tuple(j - along * a for j, a in zip(jaw, approach))
            n = norm(jaw)
            if n < 1e-6:
                raise ValueError("jaw is parallel to approach")
            jaw = tuple(j / n for j in jaw)
    err_p = err_a = float("inf")
    for _ in range(iters):
        t, jv, jw = chain.jacobian([math.degrees(x) for x in q], tool)
        p = position(t)
        e = [target[i] - p[i] for i in range(3)]
        rows = [jv[0], jv[1], jv[2]]
        err_p = norm(e)
        err_a = 0.0
        if approach is not None and jaw is not None:
            z, x = column(t, 2), column(t, 0)
            ez, ex = cross(z, approach), cross(x, jaw)
            err_a = max(math.atan2(norm(ez), dot(z, approach)), math.atan2(norm(ex), dot(x, jaw)))
            e += [(a + b) / 2 for a, b in zip(ez, ex)]
            rows += [list(r) for r in jw]
        elif jaw is not None:
            x = column(t, 0)
            # As for approach alone, with the jaw axis: spin about it is free.
            ex = cross(x, jaw)
            err_a = math.atan2(norm(ex), dot(x, jaw))
            jw_perp = [[jw[i][k] - x[i] * sum(x[m] * jw[m][k] for m in range(3)) for k in range(len(q))]
                       for i in range(3)]
            e += list(ex)
            rows += jw_perp
        elif approach is not None:
            z = column(t, 2)
            # The rotation that turns z onto approach, minus any spin about z.
            ea = cross(z, approach)
            err_a = math.atan2(norm(ea), dot(z, approach))
            jw_perp = [[jw[i][k] - z[i] * sum(z[m] * jw[m][k] for m in range(3)) for k in range(len(q))]
                       for i in range(3)]
            e += list(ea)
            rows += jw_perp
        if err_p < tol_m and err_a < math.radians(tol_deg):
            return [math.degrees(x) for x in q]
        for j in hold:
            for row in rows:
                row[j - 1] = 0.0
        jjt = [[sum(a * b for a, b in zip(r1, r2)) + (damping ** 2 if i == j else 0.0)
                for j, r2 in enumerate(rows)] for i, r1 in enumerate(rows)]
        y = _solve(jjt, e)
        dq = [sum(rows[i][k] * y[i] for i in range(len(rows))) for k in range(len(q))]
        biggest = max(abs(d) for d in dq)
        if biggest > 0.2:  # keep each iteration in the linear regime
            dq = [d * 0.2 / biggest for d in dq]
        q = [min(max(x + d, lo[k]), hi[k]) for k, (x, d) in enumerate(zip(q, dq))]
    raise IKError(f"IK did not converge: {err_p * 1000:.1f} mm, {math.degrees(err_a):.1f} deg off "
                  f"for {tool} at {[round(c, 3) for c in target]}")


# -- camera geometry -----------------------------------------------------------


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480


def pixel_ray(chain: Chain, k: Intrinsics, q_deg: Sequence[float], u: float, v: float) -> Tuple[Vec, Vec]:
    """Camera centre and unit direction (base frame) through pixel (u, v)."""
    cam = chain.pose(q_deg, "camera")
    d = rotate(cam, ((u - k.cx) / k.fx, (v - k.cy) / k.fy, 1.0))
    n = norm(d)
    return position(cam), (d[0] / n, d[1] / n, d[2] / n)


def ray_plane(origin: Sequence[float], direction: Sequence[float], z: float) -> Optional[Vec]:
    """Where the ray meets the horizontal plane at height ``z``; None if it never does."""
    if abs(direction[2]) < 1e-6:
        return None
    s = (z - origin[2]) / direction[2]
    if s <= 0:
        return None
    return (origin[0] + s * direction[0], origin[1] + s * direction[1], z)


def pixel_to_plane(chain: Chain, k: Intrinsics, q_deg: Sequence[float], u: float, v: float, z: float) -> Optional[Vec]:
    return ray_plane(*pixel_ray(chain, k, q_deg, u, v), z)


def project(chain: Chain, k: Intrinsics, q_deg: Sequence[float], point: Sequence[float]) -> Optional[Tuple[float, float]]:
    """Pixel of a base-frame point, or None if it is behind the camera."""
    x, y, z = apply(inverse(chain.pose(q_deg, "camera")), point)
    if z <= 1e-6:
        return None
    return (k.cx + k.fx * x / z, k.cy + k.fy * y / z)


# -- waypoints -----------------------------------------------------------------


def joint_path(start: Sequence[float], goal: Sequence[float], max_step: float) -> List[List[float]]:
    """Straight line in joint space, no joint moving more than ``max_step`` per waypoint; ends on ``goal``."""
    span = max(abs(g - s) for s, g in zip(start, goal))
    n = max(1, math.ceil(span / max_step - 1e-9))
    return [[s + (g - s) * i / n for s, g in zip(start, goal)] for i in range(1, n + 1)]


def line_path(
    chain: Chain,
    start_deg: Sequence[float],
    goal: Sequence[float],
    *,
    tool: str = "tcp",
    step_m: float = 0.01,
    max_step: float = 8.0,
    approach: Optional[Sequence[float]] = (0.0, 0.0, -1.0),
    jaw: Optional[Sequence[float]] = None,
) -> List[List[float]]:
    """Joint waypoints that take ``tool`` along a straight line to ``goal``
    (descend, lift, pull a drawer), each IK seeded from the one before."""
    p0 = position(chain.pose(start_deg, tool))
    n = max(1, math.ceil(norm([g - p for g, p in zip(goal, p0)]) / step_m))
    q = list(start_deg)
    out: List[List[float]] = []
    for i in range(1, n + 1):
        target = [p + (g - p) * i / n for p, g in zip(p0, goal)]
        nxt = solve_ik(chain, q, target, tool=tool, approach=approach, jaw=jaw)
        out += joint_path(q, nxt, max_step)
        q = nxt
    return out


def look_at(
    chain: Chain,
    seed_deg: Sequence[float],
    target: Sequence[float],
    *,
    distance: float,
    tilt_deg: float,
    toward: Optional[Sequence[float]] = None,
    right: Optional[Sequence[float]] = None,
) -> List[float]:
    """Joint angles that put the camera ``distance`` from ``target``, looking at
    it, ``tilt_deg`` from straight down.

    The camera leans out from the target along ``toward`` (horizontal; default
    towards the base), so a tilted view of an opened drawer looks into it from
    the robot's side and sees a cone's flank, not only its circle.
    ``right`` fixes which way the image's right edge points (else the roll is
    whatever is nearest the seed).
    """
    if toward is None:
        toward = (-target[0], -target[1], 0.0)
    n = math.hypot(toward[0], toward[1])
    h = (toward[0] / n, toward[1] / n, 0.0) if n > 1e-9 else (0.0, 0.0, 0.0)
    t = math.radians(tilt_deg)
    offset = (math.sin(t) * h[0], math.sin(t) * h[1], math.cos(t))
    eye = tuple(p + distance * o for p, o in zip(target, offset))
    return solve_ik(chain, seed_deg, eye, tool="camera", approach=tuple(-o for o in offset), jaw=right,
                    either_jaw=False)
