"""Fast re-planning of the .77 matcha motions (grasp, carry, 140-degree pour, shake, return) for new
vial and cup positions, all in the .77 base frame (mm, deg).

The templates in ``data/templates77.json`` are one grasp and one pour that were checked on the robot.
A new spot is reached in two moves:

1. turn J1 by the change in bearing seen from the .77 base. Everything rotates about the base z axis,
   so the vial / cup relative geometry is exact;
2. walk the Cartesian waypoints by the remaining radial shift with damped least-squares IK seeded
   from the rotated template (same IK branch).

If that fails the checks, the grasp (or the pour) is also turned about the vial (cup) vertical axis
by a few angles. Transits (park <-> pre-grasp, lift <-> pour start) come from the cuRobo plan server.
Every waypoint is checked for joint limits, branch jumps, gripper/cup and vial/table/cup clearance;
the whole path then goes through the server's ``/check`` with the same obstacles the console uses.
"""
from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

import ik_cuda
import kin77 as kin  # table-referenced frame (kin77.BASE_DZ)

HERE = Path(__file__).resolve().parent
TPL = json.loads((HERE / "data" / "templates77.json").read_text())
TLV = np.array(TPL["tlv_pad_mm"])
VIAL_F = np.array(TPL["vial_in_flange_mm"])
PARK = np.array(TPL["park"], float)
TAUGHT_PARK = PARK.copy()
try:  # a room-specific PARK (park_room.py --write): the taught one touched the raised board in the 2026-11-26 room
    import locate as _locate  # noqa: E402
    _pf = _locate.LAB / "park77_q.json"
    if _pf.exists():
        PARK = np.array(json.loads(_pf.read_text())["q"], float)
except Exception:  # noqa: BLE001 - keep the taught PARK
    pass
TABLE = float(TPL["table_z"])
HOUSING = np.array([[x, y, z, 1.0] for x in (-28, 0, 28) for y in (-28, 0, 30) for z in (0, 25, 50, 65)])
FING = np.array([[x, y, z, 1.0] for x in (-22, -12, 12, 22) for y in (-27, -15, -3) for z in (60, 75, 90)])
GPTS = np.vstack([HOUSING, FING])
LIM_MARGIN = 3.0
MAX_OBS = 60  # cuRobo boxes per request (plan_server limit; its OBB cache holds 64)
# cuRobo box for a standing vial (cap 20 mm wide): 20 mm a side. With 28 mm (4 mm a side) the 2026-11-26 approach
# knocked the neighbouring vial over although the plan (and the measured joints) kept 16 mm from it: the camera-to-arm
# map is off by ~10 mm, and the arm model by a few more.
OTHER_VIAL_MM = 60.0
# Reach the pre-grasp from this far straight above it, and leave it the same way, so the low part of every
# approach is a short vertical leg next to the vial instead of a sweep at vial height past its neighbours.
APPROACH_RISE_MM = 70.0
# Every grasp-chain pose (pre-grasp .. lift) keeps the gripper this far from the other vials. The 2026-11-26 pre-grasp
# had its housing 21.5 mm from the neighbour, between two vials 87 mm apart.
NEIGHBOUR_MM = 30.0


class PlanError(RuntimeError):
    pass


# -- small geometry ----------------------------------------------------------------------------
def Rz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def bearing(p) -> float:
    return math.degrees(math.atan2(p[1], p[0]))


def _err(T, Tg):
    e = np.r_[Tg[:3, 3] - T[:3, 3], 0.0, 0.0, 0.0]
    R = Tg[:3, :3] @ T[:3, :3].T
    e[3:] = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 100.0
    return e


def ik(q0, Tg, iters=200, tol=1e-3):
    """DLS on [mm, 100*rad]; stays near the seed's branch. Returns (q, err)."""
    q = np.array(q0, float)
    for _ in range(iters):
        T = kin.fk(q)
        e = _err(T, Tg)
        if np.linalg.norm(e) < tol:
            break
        J = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = 1e-3
            J[:, i] = (e - _err(kin.fk(q + dq), Tg)) / 1e-3
        q = q + np.linalg.solve(J.T @ J + 1e-2 * np.eye(6), J.T @ e)
    return q, float(np.linalg.norm(_err(kin.fk(q), Tg)))


def ik_walk(q0, Tg, step_mm=12.0):
    """Homotopy from fk(q0) to Tg in steps of <= step_mm (and <= 6 deg of rotation)."""
    T0 = kin.fk(q0)
    dp = Tg[:3, 3] - T0[:3, 3]
    Rrel = Tg[:3, :3] @ T0[:3, :3].T
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rrel) - 1) / 2))))
    n = max(1, int(math.ceil(max(np.linalg.norm(dp) / step_mm, ang / 6.0))))
    axis = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-9 else np.array([0, 0, 1.0])
    q = np.array(q0, float)
    e = 0.0
    for k in range(1, n + 1):
        f = k / n
        Tk = np.eye(4)
        Tk[:3, 3] = T0[:3, 3] + f * dp
        Tk[:3, :3] = _axis_angle(axis, f * ang) @ T0[:3, :3]
        q, e = ik(q, Tk)
    return q, e


def _chains(targets, step_fb=12.0, fb_from_prev=False):
    """Solve candidate IK chains, yielding (chain or None, error) per candidate in order.

    ``targets`` = per candidate, a list of (seed q, target T) along its chain. The first pose is reached by
    ik_walk from its seed, each later pose by ik from the previous solution; a pose over 0.3 gets one fallback
    ik_walk (step ``step_fb``, from the previous solution when ``fb_from_prev``, else from its seed) and ends the
    chain if it still fails. With planner/cuda/libikchain.so (built on the AGX Orin) all candidates are solved in
    one GPU launch, ~30-75x faster than this numpy loop there; without it the loop stays lazy, so a search that
    stops at its first good candidate solves only what it needs."""
    def one(tl):
        chain, prev = [], None
        for seed, T in tl:
            qn, e = ik_walk(seed, T) if prev is None else ik(prev, T)
            if e > 0.3:
                qn, e = ik_walk(prev if (fb_from_prev and prev is not None) else seed, T, step_mm=step_fb)
            if e > 0.3:
                return None, e
            chain.append(qn)
            prev = qn
        return chain, 0.0

    # On the AGX one GPU thread walks a chain ~3x slower than numpy does (FP64, one thread), but all candidates run
    # at once. So the first LAZY candidates (usually the answer) go through numpy, and only if the caller asks for
    # more does the rest go to the GPU in one launch (2026-11-26, AGX: easy search 0.25 s, hard one 16 s -> ~1 s).
    lazy = 2 if ik_cuda.available() else len(targets)
    for tl in targets[:lazy]:
        yield one(tl)
    rest = targets[lazy:]
    if not rest:
        return
    S = np.array([[s for s, _ in tl] for tl in rest], float)
    T = np.array([[g for _, g in tl] for tl in rest], float)
    q, e, bad = ik_cuda.chain(T, S, 12.0, step_fb, fb_from_prev)
    for c in range(len(rest)):
        yield (None, float(e[c, bad[c]])) if bad[c] >= 0 else (list(q[c]), 0.0)
    return


def _axis_angle(k, deg):
    a = math.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def turn_about(T, centre_xy, deg):
    """Pose T turned about the vertical line through centre_xy."""
    R = Rz(deg)
    c = np.r_[centre_xy, 0.0]
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = c + R @ (T[:3, 3] - c)
    out[2, 3] = T[2, 3]
    return out


def limit_margin(q) -> float:
    q = np.asarray(q)
    return float(np.min(np.minimum(q - kin.LO, kin.HI - q)))


def pad_pt(q):
    T = kin.fk(q)
    return T[:3, :3] @ TLV + T[:3, 3]


def vial_pt(q):
    T = kin.fk(q)
    return T[:3, :3] @ VIAL_F + T[:3, 3]


# -- clearance (numpy, same rules as the offline pour search) ------------------------------------
class Scene:
    """Cup (centre, rim z, radius) and other vials (xy, top z) in the .77 frame."""

    def __init__(self, cup_xy, rim_z, cup_r, others):
        self.C = np.asarray(cup_xy, float)
        self.top = float(rim_z)
        self.r = float(cup_r)
        self.others = [np.asarray(o, float) for o in others]  # (x, y, top_z)
        self.boxes = [(o[0], o[1], 17.0, 17.0, o[2] + 3.0) for o in self.others]
        self.hand_bias = float(grasp_cfg().get("hand_z_bias_mm", 0.0))

    def grip_clear(self, q) -> float:
        P = (kin.fk(q) @ GPTS.T).T[:, :3]
        # 2026-11-27 exhibit site: the side camera put the real finger centre ~10 mm above the planned one
        # (cal77_grasp.json hand_z_bias_mm), so the model's gripper is that much lower than the real one
        P[:, 2] += self.hand_bias
        dd = np.hypot(P[:, 0] - self.C[0], P[:, 1] - self.C[1])
        low = P[:, 2] < self.top + 12
        m = float(np.min(dd[low] - (self.r + 6))) if low.any() else 99.0
        return min(m, float(P[:, 2].min() - (TABLE + 10)), self.neighbour_clear(P))

    def neighbour_clear(self, P) -> float:
        """Gripper points to the other vials (cylinders r 11 mm up to their tops), minus NEIGHBOUR_MM."""
        m = 99.0
        for x, y, zt in self.others:
            r = np.maximum(np.hypot(P[:, 0] - x, P[:, 1] - y) - 11.0, 0.0)
            m = min(m, float(np.min(np.hypot(r, np.maximum(P[:, 2] - zt, 0.0)))) - NEIGHBOUR_MM)
        return m

    def vial_margin(self, q, cap_inside=False, up_f=None) -> float:
        T = kin.fk(q)
        pad = T[:3, :3] @ TLV + T[:3, 3]
        up = T[:3, :3] @ (UP_F if up_f is None else up_f)
        pts = [pad + s * up for s in np.linspace(-TPL["below_mm"], TPL["above_mm"], 9)]
        VR = TPL["vial_r"]
        m = 1e9
        for i, p in enumerate(pts):
            m = min(m, p[2] - (TABLE + VR))
            r = math.hypot(*(p[:2] - self.C))
            ok_in = cap_inside and i >= len(pts) - 2 and r < self.r - 6 and p[2] > self.top + 8
            if r < self.r + VR + 5 and not ok_in:
                m = min(m, p[2] - (self.top + VR + 5))
            for cx, cy, hx, hy, zt in self.boxes:
                dx, dy = max(abs(p[0] - cx) - hx, 0), max(abs(p[1] - cy) - hy, 0)
                m = min(m, math.sqrt(dx * dx + dy * dy + max(p[2] - zt, 0) ** 2) - VR - 3)
        return m


UP_F = np.array(TPL["grasp"]["R"]).T @ np.array([0, 0, 1.0])  # vial axis in the flange frame (flat grasp)


# -- cuRobo plan server ------------------------------------------------------------------------
def _post(api, path, body, timeout=20.0):
    req = urllib.request.Request(api.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def cu_plan(api, q0, q1, obstacles):
    r = _post(api, "/plan", {"q_start": np.round(q0, 3).tolist(), "goal": {"joints": np.round(q1, 3).tolist()},
                             "obstacles": obstacles, "dt": 0.05, "time_scale": 1.0})
    if not r.get("ok"):
        raise PlanError(f"cuRobo plan failed ({r.get('status')}: {r.get('reason')})")
    return [np.array(q) for q in r["q"]]


def above(q, rise):
    """The same tool pose `rise` mm higher (None if the IK does not close)."""
    T = kin.fk(q)
    T[2, 3] += rise
    qh, e = ik(q, T)
    return qh if e <= 0.3 else None


def path_down_to(api, q0, q1, obstacles, rise=APPROACH_RISE_MM, back_dir=None, back=25.0):
    """q0 -> (cuRobo) -> `rise` mm above a point `back` mm behind q1 (against back_dir, the approach) -> straight
    down -> forward to q1, so the open fingers never come down onto the vial they reach for; the direct path if no
    such poses plan."""
    q1 = np.asarray(q1, float)
    qb = q1
    if back_dir is not None and back > 0:
        T = kin.fk(q1)
        T[:2, 3] -= back * np.asarray(back_dir, float) / np.linalg.norm(back_dir)
        qb, e = ik(q1, T)
        if e > 0.3:
            qb = q1
    qh = above(qb, rise)
    if qh is not None:
        try:
            return cu_plan(api, q0, qh, obstacles) + ([qb] if qb is not q1 else []) + [q1]
        except PlanError:
            pass
    return cu_plan(api, q0, q1, obstacles)


def path_up_from(api, q0, q1, obstacles, rise=APPROACH_RISE_MM):
    """q0 -> straight up `rise` mm -> (cuRobo) -> q1; the direct path if no such pose plans."""
    qh = above(q0, rise)
    if qh is not None:
        try:
            return [np.asarray(q0, float)] + cu_plan(api, qh, q1, obstacles)
        except PlanError:
            pass
    return cu_plan(api, q0, q1, obstacles)


def cu_check(api, Q, obstacles):
    return _post(api, "/check", {"q": np.round(np.asarray(Q), 3).tolist(), "obstacles": obstacles, "stride": 2})


def box(name, cx, cy, z0, z1, dx, dy, yaw=0.0):
    a = math.radians(yaw) / 2
    # z0 / z1 are planner-frame heights (table at TABLE); cuRobo works in the real .77 base frame
    return {"name": name, "dims_mm": [dx, dy, z1 - z0],
            "pose": [cx, cy, (z0 + z1) / 2 + kin.BASE_DZ, math.cos(a), 0.0, 0.0, math.sin(a)]}


def cup_boxes(C, rim, r=40.0):
    R_ = r + 2
    Lw = 2 * R_ * math.tan(math.pi / 8) + 2
    return [box(f"cupw{k}", C[0] + R_ * math.cos(2 * math.pi * k / 8), C[1] + R_ * math.sin(2 * math.pi * k / 8), TABLE,
                rim + 2, 4, Lw, math.degrees(2 * math.pi * k / 8)) for k in range(8)]


# -- grasp -------------------------------------------------------------------------------------
def densify(chain, step=0.5):
    """Joint-space straight lines between waypoints, <= step deg apart (what the streamed path follows)."""
    out = [np.asarray(chain[0], float)]
    for a, b in zip(chain[:-1], chain[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def _grasp_candidates(spins=(0, -10, 10, -20, 20, -30, 30, -45, 45, -60, 60), dzs=(0.0, 5.0), shallow=(0.0, 4.0), pitches=(0.0, 5.0, 8.0, 12.0)):
    """(pitch, dz, shallow, spin) from the template outwards. Costs: turning 1 per 10 deg, raising 1 per 5 mm,
    holding the vial nearer the fingertips 1 per 3 mm (the gripper pulls it in by ~4 mm when it closes),
    tipping the hand down about the closing axis 1 per 4 deg (lifts the wrist off the table; the pour then
    turns the flange back so the vial still follows the checked pour)."""
    c = [(pt, d, h, s_) for pt in pitches for d in dzs for h in shallow for s_ in spins]
    return sorted(c, key=lambda x: x[0] / 4 + abs(x[3]) / 10 + x[1] / 5 + x[2] / 3)


def _template_frames():
    G = TPL["grasp"]
    T0 = kin.fk_taught(np.array(G["q"]))
    a_dir = (pad_pt(G["q"]) - pad_pt(G["qa"]))[:2]
    a_dir = a_dir / np.linalg.norm(a_dir)
    H = np.cross([0, 0, 1.0], np.r_[a_dir, 0.0])  # finger-closing axis (horizontal) of the template grasp
    return T0, a_dir, H / np.linalg.norm(H)


def pitch_local(pitch):
    """Flange-frame transform of a grasp tipped by `pitch` about the closing axis through the pad
    (C = T0^-1 P T0): pour poses are right-multiplied by it so the vial keeps the template's path."""
    T0, _, H = _template_frames()
    h = T0[:3, :3].T @ H
    R = _axis_angle(h, pitch)
    C = np.eye(4)
    C[:3, :3] = R
    C[:3, 3] = TLV - R @ TLV
    return C


def grasp_deeper_mm() -> float:
    """How much deeper (along the template approach) the vial really sits in the hand than the template says, from
    LAB/cal77_grasp.json {"deeper_mm": d}. 2026-11-26: a hand-guided grasp put the vial axis 13.6 mm (vial position
    before the demo) to 21 mm (after) deeper along the tool axis than the template's vial point, and the one
    grasp that worked that day had been shifted 12 mm outward; without it the fingertips closed on the near edge."""
    import os
    p = Path(os.environ.get("MYCOBOT_LAB", "~/Documents/lab-jetson/console")).expanduser() / "cal77_grasp.json"
    try:
        return float(json.loads(p.read_text()).get("deeper_mm", 0.0)) if p.exists() else 0.0
    except (ValueError, OSError):
        return 0.0


def grasp_cfg() -> dict:
    """LAB/cal77_grasp.json as a dict ({} when missing): deeper_mm, dz_mm, hand_z_bias_mm, held_min, held_max."""
    import os
    p = Path(os.environ.get("MYCOBOT_LAB", "~/Documents/lab-jetson/console")).expanduser() / "cal77_grasp.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (ValueError, OSError):
        return {}


def grasp_dz_mm() -> float:
    """A height shift of every grasp candidate (planner mm), LAB/cal77_grasp.json {"dz_mm": d}. 2026-11-27 exhibit
    vials: at the taught height the fingers closed on the cap (gripper 15, below the held range); -4 planned at the
    vial's spot, -8 and -15 found no plan there (the AGX session's GRASP_DZ)."""
    import os
    p = Path(os.environ.get("MYCOBOT_LAB", "~/Documents/lab-jetson/console")).expanduser() / "cal77_grasp.json"
    try:
        return float(json.loads(p.read_text()).get("dz_mm", 0.0)) if p.exists() else 0.0
    except (ValueError, OSError):
        return 0.0


def plan_grasp(vial_xy, scene: Scene, api=None, obs=None, max_tries=140, skip=(), only=None, pitches=(0.0, 5.0, 8.0, 12.0),
               deeper=None):
    G = TPL["grasp"]
    V0 = np.array(G["vial_xy"])
    deeper = grasp_deeper_mm() if deeper is None else deeper
    if deeper:  # the template's vial really stood this much nearer the hand, along its approach
        V0 = V0 - deeper * _template_frames()[1]
    V = np.asarray(vial_xy, float)
    dth = bearing(V) - bearing(V0)
    V0r = (Rz(dth) @ np.r_[V0, 0])[:2]
    T0, a0, H0 = _template_frames()
    gdz = grasp_dz_mm()
    why = []
    cands = _grasp_candidates(pitches=pitches)
    if only is not None:  # e.g. a place pose must hold the vial exactly like the grasp did: same pitch, dz, shallow
        cands = [c for c in cands if all(c[i] == v for i, v in only.items())]
    items = []  # (key, tag, a_dir, [(seed, target)...]) in cost order
    for pitch, dz, shallow, spin in cands[:max_tries]:
        if (pitch, dz, shallow, spin) in skip:
            continue
        tag = f"pitch {pitch:.0f} dz {dz:.0f}{f'{gdz:+.0f}' if gdz else ''} shallow {shallow:.0f} spin {spin}"
        a_dir = Rz(dth + spin)[:2, :2] @ a0
        target = V - shallow * a_dir
        Hs = Rz(dth + spin) @ H0
        Tg = kin.fk_taught(np.r_[G["q"][0] + dth, G["q"][1:]])
        Tg[:3, 3] += np.r_[target - V0r, dz + gdz]
        Tg = turn_about(Tg, target, spin)
        pivot = Tg[:3, :3] @ TLV + Tg[:3, 3]
        P = np.eye(4)
        P[:3, :3] = _axis_angle(Hs, pitch)
        P[:3, 3] = pivot - P[:3, :3] @ pivot
        tl = []
        for q in [G["qa"]] + G["sub"] + [G["q"]] + G["lift"]:
            qr = np.array(q, float)
            qr[0] += dth
            T = kin.fk_taught(qr)  # taught template pose
            T[:3, 3] += np.r_[target - V0r, dz + gdz]
            tl.append((qr, P @ turn_about(T, target, spin)))
        items.append(((pitch, dz, shallow, spin), tag, a_dir, tl))
    for (key, tag, a_dir, _), (chain, e) in zip(items, _chains([it[3] for it in items])):
        pitch, dz, shallow, spin = key
        if chain is None:
            why.append(f"{tag}: IK err {e:.2f}")
            continue
        qa, sub, qg, lift = chain[0], chain[1:4], chain[4], chain[5:]
        lim = min(limit_margin(q) for q in chain)
        jump = max(np.abs(np.diff(np.array(chain), axis=0)).max(1))
        gc = min(scene.grip_clear(q) for q in chain)
        if lim < LIM_MARGIN or jump > 20 or gc < 0:
            why.append(f"{tag}: limit {lim:.1f}, jump {jump:.1f}, grip {gc:.1f}")
            continue
        if api is not None:
            chk = cu_check(api, densify(chain), obs)
            if not chk.get("ok"):
                why.append(f"{tag}: cuRobo {chk.get('detail', {}).get('obstacles') or chk.get('detail')}")
                continue
        return dict(qa=qa, sub=sub, q=qg, lift=lift, spin=spin, dz=dz, shallow=shallow, pitch=pitch, key=(pitch, dz, shallow, spin),
                    d_theta=dth, pad=pad_pt(qg), vial=vial_pt(qg), a_dir=a_dir, deeper=float(deeper))
    raise PlanError("no grasp: " + "; ".join(why[:12]) + (f" (+{len(why) - 12} more)" if len(why) > 12 else ""))


# -- pour --------------------------------------------------------------------------------------
def _pour_candidates(spins=(0, -10, 10, -20, 20, -30, 30), rolls=(0, -10, 10, -20, 20, -30, 30, -45, 45)):
    """(spin about the cup's vertical = which side it pours from, roll of the hand about the vial's own axis =
    same vial path, different wrist / J6)."""
    c = [(sp, ro) for sp in spins for ro in rolls]
    return sorted(c, key=lambda x: abs(x[0]) / 10 + abs(x[1]) / 15)


def plan_pour(cup_xy, scene: Scene, api=None, obs=None, pitch=0.0, max_tries=40):
    P = TPL["pour"]
    C0 = np.array(P["cup_xy"])
    C = np.asarray(cup_xy, float)
    dth = bearing(C) - bearing(C0)
    C0r = (Rz(dth) @ np.r_[C0, 0])[:2]
    TA = P["tilt_angles"]
    Cl = pitch_local(pitch)
    up_f = Cl[:3, :3].T @ UP_F
    v_f = np.linalg.solve(Cl, np.r_[VIAL_F, 1.0])[:3]  # vial centre in the (pitched) flange frame
    why = []
    items = []
    for spin, roll in _pour_candidates()[:max_tries]:
        Rr = np.eye(4)
        Rr[:3, :3] = _axis_angle(up_f, roll)
        Rr[:3, 3] = v_f - Rr[:3, :3] @ v_f
        tl = []
        for q in [P["q40"]] + P["tilt"]:
            qr = np.array(q, float)
            qr[0] += dth
            T = kin.fk_taught(qr)  # taught template pose
            T[:3, 3] += np.r_[C - C0r, 0.0]
            tl.append((qr, turn_about(T, C, spin) @ Cl @ Rr))
        items.append(((spin, roll), tl))
    for ((spin, roll), _), (chain, e) in zip(items, _chains([it[1] for it in items], step_fb=4.0, fb_from_prev=True)):
        if chain is None:
            why.append(f"spin {spin} roll {roll}: IK err {e:.2f}")
            continue
        q40, tilt = chain[0], chain[1:]
        lim = min(limit_margin(q) for q in chain)
        jump = max(np.abs(np.diff(np.array(chain), axis=0)).max(1))
        angs = [40] + TA
        gc = min(scene.grip_clear(q) for q in chain)
        vm = min(scene.vial_margin(q, a >= 90, up_f) for q, a in zip(chain, angs))
        if lim < LIM_MARGIN or jump > 25 or gc < 0 or vm < 0:
            why.append(f"spin {spin} roll {roll}: limit {lim:.1f}, jump {jump:.1f}, grip {gc:.1f}, vial {vm:.1f}")
            continue
        if api is not None:
            chk = cu_check(api, densify(chain), obs)
            if not chk.get("ok"):
                why.append(f"spin {spin} roll {roll}: cuRobo {chk.get('detail')}")
                continue
        return dict(q40=q40, tilt=tilt, tilt_angles=TA, spin=spin, roll=roll, d_theta=dth, grip_clear=gc, vial_margin=vm, up_f=up_f)
    raise PlanError(f"no pour (grasp pitch {pitch:.0f}): " + "; ".join(why[:10]) + (f" (+{len(why) - 10} more)" if len(why) > 10 else ""))


# -- timing (same profile as the recorded fast run) --------------------------------------------
ACC = 100.0


# Per-joint share of a leg's speed cap. 2026-11-27 (AGX, exhibit site): the arm server refused twice ("J3 20.1 / 21.4
# deg from measured") on the first leg from park toward a vial, where J3 swings +30 -> -130 with the arm reaching out
# at the 45 deg/s cap; at ~28 deg/s (a 1.6x slower run) it never tripped. Only J3 is slowed, to 0.62 of the cap.
JOINT_SPEED_SCALE = np.array([1.0, 1.0, 0.62, 1.0, 1.0, 1.0])


def timed(legs, dt=0.05, acc=ACC):
    pts, caps = [np.asarray(legs[0][0][0], float)], []
    for wps, cap in legs:
        for q in wps:
            q = np.asarray(q, float)
            if np.max(np.abs(q - pts[-1])) < 1e-6:
                continue
            n = max(1, int(np.ceil(np.max(np.abs(q - pts[-1])) / 0.25)))
            a = pts[-1]
            for k in range(1, n + 1):
                pts.append(a + (q - a) * k / n)
                caps.append(cap)
    Q = np.array(pts)
    if len(Q) < 2:
        return Q
    ds = np.max(np.abs(np.diff(Q, axis=0)) / JOINT_SPEED_SCALE, axis=1)  # each joint at most scale x cap
    caps = np.array(caps)
    v = np.zeros(len(Q))
    for i in range(1, len(Q)):
        v[i] = min(caps[i - 1], math.sqrt(v[i - 1] ** 2 + 2 * acc * ds[i - 1]))
    v[-1] = 0.0
    for i in range(len(Q) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * acc * ds[i]))
    v[0] = 0.0
    tt = np.r_[0.0, np.cumsum(2 * ds / np.maximum(v[:-1] + v[1:], 1e-3))]
    ts = np.arange(0.0, tt[-1] + 1e-9, dt)
    out = np.column_stack([np.interp(ts, tt, Q[:, j]) for j in range(6)])
    if np.max(np.abs(out[-1] - Q[-1])) > 1e-6:
        out = np.vstack([out, Q[-1]])
    return out


def thin(path, step=1.0):
    """Keep cuRobo path samples at least `step` deg apart (the speed profile re-times them)."""
    out = [path[0]]
    for q in path[1:-1]:
        if np.max(np.abs(q - out[-1])) >= step:
            out.append(q)
    out.append(path[-1])
    return out


# -- the whole .77 phase -----------------------------------------------------------------------
def grasp_jac(qg, step=1.0):
    """deg per mm of the vial point in .77 x / y at the grasp (orientation kept)."""
    T0 = kin.fk(qg)
    J = np.zeros((6, 2))
    for k in range(2):
        Tp, Tm = T0.copy(), T0.copy()
        Tp[k, 3] += step
        Tm[k, 3] -= step
        J[:, k] = (ik(qg, Tp)[0] - ik(qg, Tm)[0]) / (2 * step)
    return J


# Gripper opening (arm server read_gripper, 0 closed .. 100 open) that means "a vial is between the fingers" after
# the close-to-10 command. Measured 2026-11-26 at PARK: closed on nothing 11 (command 10) and 5 (command 0), so an
# empty hand ends within a few units of the command; a vial stops the fingers well above it.
GRIP_CLOSE = 10
GRIP_HELD = (18.0, 80.0)  # old-room vials; the exhibit vials read ~15 when held by the body (2026-11-27)


def grip_held():
    """The gripper reading range that means "holding a vial": cal77_grasp.json held_min / held_max, else GRIP_HELD."""
    c = grasp_cfg()
    return float(c.get("held_min", GRIP_HELD[0])), float(c.get("held_max", GRIP_HELD[1]))
# The pre-grasp alignment measures the finger midline from open/closed difference pixels, which mix parts at
# different heights; seen at an angle by the top camera they can shift the midline sideways by ~10 mm. On
# 2026-11-26 two runs shifted the hand by +9.6 and -9.6 mm and then closed beside the vial (side camera, gripper
# reading 11). Until that is fixed and checked against a grasp, it only measures and logs.
ALIGN_APPLY = False


def pick_segments(g, start, to_qa, label=None, name="말차 병", target="matcha"):
    """Approach and finish the approach to the planned grasp; there, photograph the fingers open / slightly closed,
    move by the vial's offset from the pads, look again and close only if the vial is between the fingers (else
    stop open); check the gripper stopped on the vial, lift (shift faded out), check it still holds the vial and
    that the vial left its spot. Returns (segments, paths)."""
    label = label or f"{name} 쪽으로"
    sub, qg, lift = g["sub"], g["q"], g["lift"]
    a1 = timed([([start] + to_qa[1:], 65.0), ([sub[0]], 25.0)])
    a2 = timed([([sub[0], sub[1]], 25.0), ([sub[2]], 14.0), ([qg], 8.0)])
    up = timed([([qg, lift[0]], 10.0), (lift[1:], 25.0)])
    py, me = sys.executable, str(HERE / "align.py")
    a_dir = g["a_dir"]
    J = grasp_jac(qg)
    R = lambda Q: np.round(Q, 3).tolist()  # noqa: E731
    # Waits: right after a joint stream the gripper starts late; on 2026-11-26 the "closed" photo 1.3 s after the
    # command still showed open fingers (0 px changed), while 1.5 s after it at rest the change was ~2800 px.
    ins = str(HERE / "inside.py")
    meas = [py, ins, "measure", "--q", *[f"{x:.3f}" for x in qg], "--vial", f"{g['vial'][0]:.2f}", f"{g['vial'][1]:.2f}",
            "--approach", f"{a_dir[0]:.4f}", f"{a_dir[1]:.4f}"]
    hold = lambda n: R(np.repeat(np.asarray(qg, float)[None], n, axis=0))  # noqa: E731 - stand still; a shift moves it
    # A look before the last approach, where the fingertips are still ~20 mm behind the vial: 2026-11-26 (vial at
    # r 198-225) the hand stood ~29 mm to the side of the vial and the last approach knocked it over before the look
    # at the grasp pose. Only the sideways error is used there; the hand moves sideways in place, then goes in.
    q_pre = np.asarray(a1[-1], float)
    meas_pre = [py, ins, "measure", "--q", *[f"{x:.3f}" for x in q_pre], "--vial", f"{g['vial'][0]:.2f}", f"{g['vial'][1]:.2f}",
                "--approach", f"{a_dir[0]:.4f}", f"{a_dir[1]:.4f}", "--lateral-only"]
    J_pre = grasp_jac(q_pre)

    def look(tag):  # open / slightly closed photo pair at the grasp pose. 75, not 60: at 60 a pad touched and
        # nudged a vial sitting 10 mm off-centre (2026-11-26); the pads still move enough to be seen.
        return [{"kind": "check", "label": f"사진(열림){tag}", "cmd": [py, ins, "snap", "g_open"]},
                {"kind": "grip", "label": "살짝 오므리기", "value": 75},
                {"kind": "wait", "label": "오므리는 중", "seconds": 1.2},
                {"kind": "check", "label": f"사진(살짝 오므림){tag}", "cmd": [py, ins, "snap", "g_half"]},
                {"kind": "grip", "label": "펴기", "value": 100},
                {"kind": "wait", "label": "펴는 중", "seconds": 0.8}]

    # Hojin 2026-11-26: "그리퍼 내부에 들어오면 오므리자". At the planned grasp pose, look where the vial is relative
    # to the pads, move by that, look again, and close only if the vial is between the fingers; otherwise stop
    # without closing (closing on the vial's near edge had pushed it away three times that day).
    # Open first: 2026-11-26 the gripper was still closed from a hand-guided demo, so the "open" photo showed closed
    # fingers and opening them at the grasp pose swung a pad into the vial (pushed ~13 mm).
    segs = [{"kind": "grip", "label": "그리퍼 열기", "value": 100},
            {"kind": "wait", "label": "여는 중", "seconds": 1.0},
            {"kind": "joints", "label": label, "q": R(a1), "dt": 0.05},
            *look(" 접근 전"),
            {"kind": "align", "label": "접근 전 옆 오차(위 카메라)", "J": np.round(J_pre, 4).tolist(),
             "max_err_mm": 42, "apply": True, "cmd": meas_pre},
            {"kind": "joints", "label": "옆으로 맞추기", "q": R(np.repeat(q_pre[None], 20, axis=0)), "dt": 0.05,
             "offset": "in", "ramp": 20},
            {"kind": "joints", "label": "마지막 접근", "q": R(a2), "dt": 0.05, "offset": "in"},
            *look(""),
            {"kind": "align", "label": "병이 손가락 사이에 있는지(위 카메라)", "J": np.round(J, 4).tolist(),
             "max_err_mm": 42, "apply": True, "cmd": meas},
            {"kind": "joints", "label": "병 쪽으로 맞추기", "q": hold(20), "dt": 0.05, "offset": "in", "ramp": 20},
            # A second correction before the final look: with the vial off to one side the fingertip estimate is
            # poor; once it sits between the fingers the along-the-fingers reading is right (side camera,
            # 2026-11-26: pass 1 read +4 mm along, pass 2 +12.5 mm, and the side view showed the tips ~12 mm short).
            *look(" 다시"),
            {"kind": "align", "label": "다시 재고 맞추기", "J": np.round(J, 4).tolist(),
             "max_err_mm": 42, "apply": True, "cmd": meas},
            {"kind": "joints", "label": "병 쪽으로 더 맞추기", "q": hold(16), "dt": 0.05, "offset": "in", "ramp": 16},
            *look(" 마지막"),
            {"kind": "align", "label": "손가락 사이인지 마지막 확인", "J": np.round(J, 4).tolist(),
             "max_err_mm": 12, "apply": True, "cmd": [*meas, "--final"]},
            {"kind": "joints", "label": "마지막 맞추기", "q": hold(12), "dt": 0.05, "offset": "in", "ramp": 12},
            {"kind": "grip", "label": "잡기", "value": GRIP_CLOSE},
            {"kind": "wait", "label": "잡는 중", "seconds": 1.5},
            # 2026-11-26: a run lifted an empty hand and poured air; the camera check after the lift passed because
            # the pushed vial no longer stood on its old spot. The gripper's own opening says it directly.
            {"kind": "grip_check", "label": "잡았는지 확인(그리퍼)", "min": grip_held()[0], "max": grip_held()[1], "what": name},
            {"kind": "joints", "label": "들기", "q": R(up), "dt": 0.05, "offset": "out", "ramp": len(up)},
            {"kind": "grip_check", "label": "들고 있는지 확인(그리퍼)", "min": grip_held()[0], "max": grip_held()[1], "what": name},
            lift_check(g, target)]
    return segs, [a1, a2, up]


def lift_check(g, target="matcha"):
    """After the lift: no cap of the target vial may still stand at table height near where it stood (top camera
    depth), pushed a little included."""
    import locate  # noqa: E402 - planner module (camera model)
    px = locate.px_of77(g["vial"][:2], locate.CAP_Z)
    return {"kind": "check", "label": "들렸는지 확인(위 카메라)", "timeout_s": 15,
            "cmd": [sys.executable, str(HERE / "verify.py"), "lifted", "--px", f"{px[0]:.1f}", f"{px[1]:.1f}",
                    "--target", target]}



def plan_phase(vial_xy, cup_xy, others, api, ws_obstacles, rim_z=None, cup_r=40.0, place_comp_mm=3.8, q_start=None,
               grasp_test=False, name="말차 병", target="matcha", deeper=None, extra_obs=()):
    """Returns (phase dict for the console scenario, info). ``others`` = [(x, y, top_z)] of the other vials.
    ``deeper``: grasp depth correction (mm) instead of LAB/cal77_grasp.json's (the caller steps it down at the reach
    edge). A grasp test pours nothing, so its hand may tip like a move (MOVE_PITCHES): 2026-11-26 the matcha vial at
    r 248 had no grasp within 12 deg at any depth, and one at 10 mm deeper within 25 deg."""
    t0 = time.perf_counter()
    if rim_z is None:
        import locate  # noqa: E402 - planner module (camera model, object heights)
        rim_z = locate.RIM_Z
    sc = Scene(cup_xy, rim_z, cup_r, others)
    obs = list(ws_obstacles) + cup_boxes(cup_xy, rim_z, cup_r) + [box(f"vial{i}", o[0], o[1], TABLE, o[2] + 3, OTHER_VIAL_MM, OTHER_VIAL_MM)
                                                                   for i, o in enumerate(others)]
    obs = (obs + list(extra_obs))[:MAX_OBS]  # extra_obs: depth boxes (depth77)
    pours, skip, g, p = {}, set(), None, None
    for _ in range(6):
        g = (plan_grasp(vial_xy, sc, api, obs, skip=skip, pitches=MOVE_PITCHES, max_tries=400, deeper=deeper) if grasp_test
             else plan_grasp(vial_xy, sc, api, obs, skip=skip, deeper=deeper))
        if grasp_test:  # pick, lift, put back: no pour, so no pour or carry path to plan (2026-11-26: a grasp test
            break       # failed on the carry path's -2.8 mm vial margin it never drives)
        if g["pitch"] not in pours:
            try:
                pours[g["pitch"]] = plan_pour(cup_xy, sc, api, obs, pitch=g["pitch"])
            except PlanError as exc:
                pours[g["pitch"]] = exc
        p = pours[g["pitch"]]
        if not isinstance(p, PlanError):
            break
        skip |= {k for k in _grasp_candidates() if k[0] == g["pitch"]}
    if isinstance(p, PlanError):
        raise p
    t_ik = time.perf_counter() - t0
    start = PARK if q_start is None else np.asarray(q_start, float)
    to_qa = thin(path_down_to(api, start, g["qa"], obs, back_dir=g["a_dir"]))
    lift_top = g["lift"][-1]
    to_park = thin(path_up_from(api, g["qa"], PARK, obs))
    to_pour = from_pour = None
    if not grasp_test:
        # cuRobo's paths differ from call to call; a carry path that grazes a vial margin gets two more tries
        for nm, a_, b_ in (("to the cup", lift_top, p["q40"]), ("back from the cup", p["q40"], lift_top)):
            for attempt in range(3):
                path = thin(cu_plan(api, a_, b_, obs))
                vm = min(sc.vial_margin(q, False, p["up_f"]) for q in path)
                gc = min(sc.grip_clear(q) for q in path)
                if vm >= 0 and gc >= 0:
                    break
            else:
                raise PlanError(f"carry path {nm}: vial margin {vm:.1f} mm, gripper clearance {gc:.1f} mm (3 tries)")
            if nm == "to the cup":
                to_pour = path
            else:
                from_pour = path
    t_plan = time.perf_counter() - t0
    # the gripper centres the vial on the pad when it closes; place so the vial lands back on its old spot
    qg, sub, lift = g["q"], g["sub"], g["lift"]
    qplace = qg
    if place_comp_mm > 0:
        d = g["vial"][:2] - g["pad"][:2]
        d = d / max(np.linalg.norm(d), 1e-6) * place_comp_mm
        T = kin.fk(qg)
        T[:3, 3] += np.r_[d, 0.0]
        qplace, e = ik(qg, T)
        if e > 0.05:
            qplace = qg
    pick, (a1, a2, up) = pick_segments(g, start, to_qa, name=name, target=target)
    approach = np.vstack([a1, a2])
    leave = timed([([qplace, sub[2]], 12.0), ([sub[1], sub[0], g["qa"]], 30.0), (to_park[1:], 65.0)])
    seg = lambda label, Q: {"kind": "joints", "label": label, "q": np.round(Q, 3).tolist(), "dt": 0.05}  # noqa: E731
    if not grasp_test:
        tilt, TA = p["tilt"], p["tilt_angles"]
        k15 = TA.index(TA[-1] - 15)
        carry = timed([([lift[-1]] + to_pour[1:], 45.0), (tilt, 30.0)])
        pts = [tilt[-1], tilt[k15], tilt[-1], tilt[k15], tilt[-1]]
        shake = np.vstack([timed([([x, y], 45.0)], acc=60.0)[(1 if i else 0):] for i, (x, y) in enumerate(zip(pts, pts[1:]))])
        back = timed([([tilt[-1]] + tilt[::-1][1:] + [p["q40"]], 40.0), (from_pour[1:], 45.0), (lift[::-1][1:], 25.0),
                      ([qplace], 8.0)])
    if grasp_test:  # approach, grip, lift to the top of the lift column, hold, put back, release, leave
        down = timed([([lift[-1]] + lift[::-1][1:], 25.0), ([qplace], 8.0)])
        phase = {"arm": "77", "label": f"{name} 잡기 시험", "speed_dps": 80, "obstacles": obs[len(ws_obstacles):], "segments": [
            *pick,
            seg("내려놓기", down), {"kind": "grip", "label": "놓기", "value": 100}, seg("park", leave)]}
        full = np.vstack([approach, up, down, leave])
        chk = cu_check(api, full, obs)
        if not chk.get("ok"):
            raise PlanError(f"cuRobo check: collision at {chk.get('index')}/{chk.get('n')}: {chk.get('detail')}")
        return phase, dict(t_total_s=round(time.perf_counter() - t0, 2), grasp_pitch=g["pitch"], grasp_spin=g["spin"],
                           grasp_dz=g["dz"], grasp_shallow=g["shallow"], checked=chk.get("checked"),
                           grasp_deeper=grasp_deeper_mm() if deeper is None else deeper)
    phase = {"arm": "77", "label": f"{name} 털어 넣기", "speed_dps": 80, "obstacles": obs[len(ws_obstacles):], "segments": [
        *pick,
        seg(f"컵 위로, {TA[-1]}도 기울이기", carry),
        {"kind": "wait", "label": "붓기", "seconds": 1.0},
        seg("두 번 털기", shake),
        {"kind": "wait", "label": "마저 붓기", "seconds": 1.0},
        seg("세워서 제자리로", back),
        {"kind": "grip", "label": "놓기", "value": 100},
        seg("park", leave)]}
    full = np.vstack([approach, up, carry, shake, back, leave])
    chk = cu_check(api, full, obs)
    if not chk.get("ok"):
        raise PlanError(f"cuRobo check: collision at {chk.get('index')}/{chk.get('n')}: {chk.get('detail')}")
    info = dict(t_ik_s=round(t_ik, 2), t_total_s=round(time.perf_counter() - t0, 2), t_transit_s=round(t_plan - t_ik, 2),
                grasp_spin=g["spin"], grasp_dz=g["dz"], grasp_shallow=g["shallow"], grasp_pitch=g["pitch"], pour_spin=p["spin"], pour_roll=p["roll"], d_theta_grasp=round(g["d_theta"], 1),
                d_theta_pour=round(p["d_theta"], 1), pour_grip_clear=round(p["grip_clear"], 1),
                grasp_deeper=grasp_deeper_mm() if deeper is None else deeper,
                pour_vial_margin=round(p["vial_margin"], 1), checked=chk.get("checked"),
                motion_s=round(len(full) * 0.05 + 0.8 + 0.2 + 1.0 + 1.0 + 0.8, 1))
    return phase, info


MOVE_PITCHES = (0.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0)  # a move needs no pour, so the hand may tip further


def plan_move(vial_xy, to_xy, others, api, ws_obstacles, cup_xy=(600.0, 600.0), rim_z=None, cup_r=40.0, place_comp_mm=3.8,
              extra_obs=(),
              q_start=None, name="말차 병", target="matcha"):
    """Pick the vial at vial_xy and stand it at to_xy (same hold: pitch, height and depth), then park."""
    t0 = time.perf_counter()
    if rim_z is None:
        import locate  # noqa: E402 - planner module (camera model, object heights)
        rim_z = locate.RIM_Z
    sc = Scene(cup_xy, rim_z, cup_r, others)
    obs = (list(ws_obstacles) + cup_boxes(cup_xy, rim_z, cup_r) + [box(f"vial{i}", o[0], o[1], TABLE, o[2] + 3, OTHER_VIAL_MM, OTHER_VIAL_MM)
                                                                   for i, o in enumerate(others)] + list(extra_obs))[:MAX_OBS]
    # a move needs no pour, so the hand may tip further (reaches farther); the place must hold it the same way.
    # The whole timed path is checked again at the end; a grasp that fails there is skipped and the next one tried
    # (2026-11-26: the first tipped grasp touched the table with J4 by 0.6 mm only on the timed approach).
    deep = MOVE_PITCHES
    start = PARK if q_start is None else np.asarray(q_start, float)
    skip, why = set(), []
    for _ in range(6):
        g = plan_grasp(vial_xy, sc, api, obs, pitches=deep, max_tries=220, skip=skip)
        try:
            h = plan_grasp(to_xy, sc, api, obs, only={0: g["pitch"], 1: g["dz"], 2: g["shallow"]}, pitches=deep)
        except PlanError as exc:
            skip.add(g["key"])
            why.append(f"{g['key']}: place {str(exc)[:60]}")
            continue
        to_qa = thin(path_down_to(api, start, g["qa"], obs, back_dir=g["a_dir"]))
        over = thin(cu_plan(api, g["lift"][-1], h["lift"][-1], obs))
        to_park = thin(path_up_from(api, h["qa"], PARK, obs))
        vm = min(sc.vial_margin(q) for q in over)
        if vm < 0:
            skip.add(g["key"])
            why.append(f"{g['key']}: carry path vial margin {vm:.1f} mm")
            continue
        qplace = h["q"]
        if place_comp_mm > 0:
            d = h["vial"][:2] - h["pad"][:2]
            T = kin.fk(h["q"])
            T[:3, 3] += np.r_[d / max(np.linalg.norm(d), 1e-6) * place_comp_mm, 0.0]
            qn, e = ik(h["q"], T)
            qplace = qn if e < 0.3 else h["q"]
        hs = h["sub"]
        pick, (a1, a2, up) = pick_segments(g, start, to_qa, name=name, target=target)
        approach = np.vstack([a1, a2])
        carry = timed([([g["lift"][-1]] + over[1:], 45.0), (h["lift"][::-1][1:], 25.0), ([qplace], 8.0)])
        leave = timed([([qplace, hs[2]], 12.0), ([hs[1], hs[0], h["qa"]], 30.0), (to_park[1:], 65.0)])
        seg = lambda label, Q: {"kind": "joints", "label": label, "q": np.round(Q, 3).tolist(), "dt": 0.05}  # noqa: E731
        phase = {"arm": "77", "label": f"{name} 옮기기", "speed_dps": 80, "obstacles": obs[len(ws_obstacles):], "segments": [
            *pick, seg("새 자리로 옮기기", carry), {"kind": "grip", "label": "놓기", "value": 100}, seg("park", leave)]}
        chk = cu_check(api, np.vstack([approach, up, carry, leave]), obs)
        if chk.get("ok"):
            break
        skip.add(g["key"])
        why.append(f"{g['key']}: collision at {chk.get('index')}/{chk.get('n')}: {chk.get('detail', {}).get('obstacles') or chk.get('detail')}")
    else:
        raise PlanError("cuRobo check: " + "; ".join(why[-3:]))
    return phase, dict(t_total_s=round(time.perf_counter() - t0, 2), pick=g["key"], place=h["key"], checked=chk.get("checked"))
