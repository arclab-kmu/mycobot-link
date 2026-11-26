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

import kin

HERE = Path(__file__).resolve().parent
TPL = json.loads((HERE / "data" / "templates77.json").read_text())
TLV = np.array(TPL["tlv_pad_mm"])
VIAL_F = np.array(TPL["vial_in_flange_mm"])
PARK = np.array(TPL["park"], float)
TABLE = float(TPL["table_z"])
HOUSING = np.array([[x, y, z, 1.0] for x in (-28, 0, 28) for y in (-28, 0, 30) for z in (0, 25, 50, 65)])
FING = np.array([[x, y, z, 1.0] for x in (-22, -12, 12, 22) for y in (-27, -15, -3) for z in (60, 75, 90)])
GPTS = np.vstack([HOUSING, FING])
LIM_MARGIN = 3.0
OTHER_VIAL_MM = 28.0  # cuRobo box for a standing vial (cap 20 mm wide): ~4 mm margin a side


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

    def grip_clear(self, q) -> float:
        P = (kin.fk(q) @ GPTS.T).T[:, :3]
        dd = np.hypot(P[:, 0] - self.C[0], P[:, 1] - self.C[1])
        low = P[:, 2] < self.top + 12
        m = float(np.min(dd[low] - (self.r + 6))) if low.any() else 99.0
        return min(m, float(P[:, 2].min() - (TABLE + 10)))

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


def cu_check(api, Q, obstacles):
    return _post(api, "/check", {"q": np.round(np.asarray(Q), 3).tolist(), "obstacles": obstacles, "stride": 2})


def box(name, cx, cy, z0, z1, dx, dy, yaw=0.0):
    a = math.radians(yaw) / 2
    return {"name": name, "dims_mm": [dx, dy, z1 - z0], "pose": [cx, cy, (z0 + z1) / 2, math.cos(a), 0.0, 0.0, math.sin(a)]}


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
    T0 = kin.fk(np.array(G["q"]))
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


def plan_grasp(vial_xy, scene: Scene, api=None, obs=None, max_tries=140, skip=(), only=None, pitches=(0.0, 5.0, 8.0, 12.0)):
    G = TPL["grasp"]
    V0 = np.array(G["vial_xy"])
    V = np.asarray(vial_xy, float)
    dth = bearing(V) - bearing(V0)
    V0r = (Rz(dth) @ np.r_[V0, 0])[:2]
    T0, a0, H0 = _template_frames()
    why = []
    cands = _grasp_candidates(pitches=pitches)
    if only is not None:  # e.g. a place pose must hold the vial exactly like the grasp did: same pitch, dz, shallow
        cands = [c for c in cands if all(c[i] == v for i, v in only.items())]
    for pitch, dz, shallow, spin in cands[:max_tries]:
        if (pitch, dz, shallow, spin) in skip:
            continue
        tag = f"pitch {pitch:.0f} dz {dz:.0f} shallow {shallow:.0f} spin {spin}"
        a_dir = Rz(dth + spin)[:2, :2] @ a0
        target = V - shallow * a_dir
        Hs = Rz(dth + spin) @ H0
        Tg = kin.fk(np.r_[G["q"][0] + dth, G["q"][1:]])
        Tg[:3, 3] += np.r_[target - V0r, dz]
        Tg = turn_about(Tg, target, spin)
        pivot = Tg[:3, :3] @ TLV + Tg[:3, 3]
        P = np.eye(4)
        P[:3, :3] = _axis_angle(Hs, pitch)
        P[:3, 3] = pivot - P[:3, :3] @ pivot
        chain, prev, bad = [], None, None
        for q in [G["qa"]] + G["sub"] + [G["q"]] + G["lift"]:
            qr = np.array(q, float)
            qr[0] += dth
            T = kin.fk(qr)
            T[:3, 3] += np.r_[target - V0r, dz]
            T = P @ turn_about(T, target, spin)
            qn, e = ik_walk(qr, T) if prev is None else ik(prev, T)
            if e > 0.3:
                qn, e = ik_walk(qr, T)
            if e > 0.3:
                bad = f"IK err {e:.2f}"
                break
            chain.append(qn)
            prev = qn
        if bad:
            why.append(f"{tag}: {bad}")
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
                    d_theta=dth, pad=pad_pt(qg), vial=vial_pt(qg), a_dir=a_dir)
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
    for spin, roll in _pour_candidates()[:max_tries]:
        Rr = np.eye(4)
        Rr[:3, :3] = _axis_angle(up_f, roll)
        Rr[:3, 3] = v_f - Rr[:3, :3] @ v_f
        chain, prev, bad = [], None, None
        for q in [P["q40"]] + P["tilt"]:
            qr = np.array(q, float)
            qr[0] += dth
            T = kin.fk(qr)
            T[:3, 3] += np.r_[C - C0r, 0.0]
            T = turn_about(T, C, spin) @ Cl @ Rr
            qn, e = ik_walk(qr, T) if prev is None else ik(prev, T)
            if e > 0.3:
                qn, e = ik_walk(qr if prev is None else prev, T, step_mm=4.0)
            if e > 0.3:
                bad = f"IK err {e:.2f}"
                break
            chain.append(qn)
            prev = qn
        if bad:
            why.append(f"spin {spin} roll {roll}: {bad}")
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
    ds = np.max(np.abs(np.diff(Q, axis=0)), axis=1)
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


def pick_segments(g, start, to_qa, label="말차 병 쪽으로"):
    """Approach to the pre-grasp, photograph the gripper open / closed, align on the vial in that image, finish
    the approach (shift ramped in), grip, lift (shift faded out), check the lift. Returns (segments, paths)."""
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
    segs = [{"kind": "joints", "label": label, "q": R(a1), "dt": 0.05},
            {"kind": "wait", "label": "멈춤", "seconds": 0.5},
            {"kind": "check", "label": "사진(열림)", "cmd": [py, me, "snap", "open"]},
            {"kind": "grip", "label": "손가락 오므리기", "value": 40},
            {"kind": "wait", "label": "오므리는 중", "seconds": 1.5},
            {"kind": "check", "label": "사진(오므림)", "cmd": [py, me, "snap", "closed"]},
            {"kind": "grip", "label": "손가락 펴기", "value": 100},
            {"kind": "wait", "label": "펴는 중", "seconds": 1.0},
            {"kind": "align", "label": "병에 맞추기(위 카메라)", "J": np.round(J, 4).tolist(), "max_err_mm": 25,
             "cmd": [py, me, "measure", "--q", *[f"{x:.3f}" for x in sub[0]], "--vial", f"{g['vial'][0]:.2f}", f"{g['vial'][1]:.2f}",
                     "--approach", f"{a_dir[0]:.4f}", f"{a_dir[1]:.4f}"]},
            {"kind": "joints", "label": "마지막 접근", "q": R(a2), "dt": 0.05, "offset": "in", "ramp": 20},
            {"kind": "grip", "label": "잡기", "value": 10},
            {"kind": "wait", "label": "잡기 확인", "seconds": 0.2},
            {"kind": "joints", "label": "들기", "q": R(up), "dt": 0.05, "offset": "out", "ramp": len(up)},
            lift_check(g)]
    return segs, [a1, a2, up]


def lift_check(g):
    """After the lift: the matcha cap must no longer stand at table height near where it stood (top camera depth)."""
    import locate  # noqa: E402 - planner module (camera model)
    px = locate.px_of77(g["vial"][:2], 264.0)
    return {"kind": "check", "label": "들렸는지 확인(위 카메라)", "timeout_s": 15,
            "cmd": [sys.executable, str(HERE / "verify.py"), "lifted", "--px", f"{px[0]:.1f}", f"{px[1]:.1f}"]}



def plan_phase(vial_xy, cup_xy, others, api, ws_obstacles, rim_z=286.0, cup_r=40.0, place_comp_mm=3.8, q_start=None,
               grasp_test=False):
    """Returns (phase dict for the console scenario, info). ``others`` = [(x, y, top_z)] of the other vials."""
    t0 = time.perf_counter()
    sc = Scene(cup_xy, rim_z, cup_r, others)
    obs = list(ws_obstacles) + cup_boxes(cup_xy, rim_z, cup_r) + [box(f"vial{i}", o[0], o[1], TABLE, o[2] + 3, OTHER_VIAL_MM, OTHER_VIAL_MM)
                                                                   for i, o in enumerate(others)]
    obs = obs[:15]
    pours, skip, g, p = {}, set(), None, None
    for _ in range(6):
        g = plan_grasp(vial_xy, sc, api, obs, skip=skip)
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
    to_qa = thin(cu_plan(api, start, g["qa"], obs))
    lift_top = g["lift"][-1]
    to_pour = thin(cu_plan(api, lift_top, p["q40"], obs))
    from_pour = thin(cu_plan(api, p["q40"], lift_top, obs))
    to_park = thin(cu_plan(api, g["qa"], PARK, obs))
    for nm, path in (("to the cup", to_pour), ("back from the cup", from_pour)):
        vm = min(sc.vial_margin(q, False, p["up_f"]) for q in path)
        gc = min(sc.grip_clear(q) for q in path)
        if vm < 0 or gc < 0:
            raise PlanError(f"carry path {nm}: vial margin {vm:.1f} mm, gripper clearance {gc:.1f} mm")
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
    tilt, TA = p["tilt"], p["tilt_angles"]
    k15 = TA.index(TA[-1] - 15)
    pick, (a1, a2, up) = pick_segments(g, start, to_qa)
    approach = np.vstack([a1, a2])
    carry = timed([([lift[-1]] + to_pour[1:], 45.0), (tilt, 30.0)])
    pts = [tilt[-1], tilt[k15], tilt[-1], tilt[k15], tilt[-1]]
    shake = np.vstack([timed([([x, y], 45.0)], acc=60.0)[(1 if i else 0):] for i, (x, y) in enumerate(zip(pts, pts[1:]))])
    back = timed([([tilt[-1]] + tilt[::-1][1:] + [p["q40"]], 40.0), (from_pour[1:], 45.0), (lift[::-1][1:], 25.0),
                  ([qplace], 8.0)])
    leave = timed([([qplace, sub[2]], 12.0), ([sub[1], sub[0], g["qa"]], 30.0), (to_park[1:], 65.0)])
    seg = lambda label, Q: {"kind": "joints", "label": label, "q": np.round(Q, 3).tolist(), "dt": 0.05}
    if grasp_test:  # approach, grip, lift to the top of the lift column, hold, put back, release, leave
        down = timed([([lift[-1]] + lift[::-1][1:], 25.0), ([qplace], 8.0)])
        phase = {"arm": "77", "label": "말차 병 잡기 시험", "speed_dps": 80, "obstacles": obs[len(ws_obstacles):], "segments": [
            *pick,
            seg("내려놓기", down), {"kind": "grip", "label": "놓기", "value": 100}, seg("park", leave)]}
        full = np.vstack([approach, up, down, leave])
        chk = cu_check(api, full, obs)
        if not chk.get("ok"):
            raise PlanError(f"cuRobo check: collision at {chk.get('index')}/{chk.get('n')}: {chk.get('detail')}")
        return phase, dict(t_total_s=round(time.perf_counter() - t0, 2), grasp_pitch=g["pitch"], grasp_spin=g["spin"],
                           grasp_dz=g["dz"], grasp_shallow=g["shallow"], checked=chk.get("checked"))
    phase = {"arm": "77", "label": "말차 병 털어 넣기", "speed_dps": 80, "obstacles": obs[len(ws_obstacles):], "segments": [
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
                pour_vial_margin=round(p["vial_margin"], 1), checked=chk.get("checked"),
                motion_s=round(len(full) * 0.05 + 0.8 + 0.2 + 1.0 + 1.0 + 0.8, 1))
    return phase, info


MOVE_PITCHES = (0.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0)  # a move needs no pour, so the hand may tip further


def plan_move(vial_xy, to_xy, others, api, ws_obstacles, cup_xy=(600.0, 600.0), rim_z=286.0, cup_r=40.0, place_comp_mm=3.8,
              q_start=None):
    """Pick the vial at vial_xy and stand it at to_xy (same hold: pitch, height and depth), then park."""
    t0 = time.perf_counter()
    sc = Scene(cup_xy, rim_z, cup_r, others)
    obs = (list(ws_obstacles) + cup_boxes(cup_xy, rim_z, cup_r) + [box(f"vial{i}", o[0], o[1], TABLE, o[2] + 3, OTHER_VIAL_MM, OTHER_VIAL_MM)
                                                                   for i, o in enumerate(others)])[:15]
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
        to_qa = thin(cu_plan(api, start, g["qa"], obs))
        over = thin(cu_plan(api, g["lift"][-1], h["lift"][-1], obs))
        to_park = thin(cu_plan(api, h["qa"], PARK, obs))
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
        pick, (a1, a2, up) = pick_segments(g, start, to_qa)
        approach = np.vstack([a1, a2])
        carry = timed([([g["lift"][-1]] + over[1:], 45.0), (h["lift"][::-1][1:], 25.0), ([qplace], 8.0)])
        leave = timed([([qplace, hs[2]], 12.0), ([hs[1], hs[0], h["qa"]], 30.0), (to_park[1:], 65.0)])
        seg = lambda label, Q: {"kind": "joints", "label": label, "q": np.round(Q, 3).tolist(), "dt": 0.05}  # noqa: E731
        phase = {"arm": "77", "label": "말차 병 옮기기", "speed_dps": 80, "obstacles": obs[len(ws_obstacles):], "segments": [
            *pick, seg("새 자리로 옮기기", carry), {"kind": "grip", "label": "놓기", "value": 100}, seg("park", leave)]}
        chk = cu_check(api, np.vstack([approach, up, carry, leave]), obs)
        if chk.get("ok"):
            break
        skip.add(g["key"])
        why.append(f"{g['key']}: collision at {chk.get('index')}/{chk.get('n')}: {chk.get('detail', {}).get('obstacles') or chk.get('detail')}")
    else:
        raise PlanError("cuRobo check: " + "; ".join(why[-3:]))
    return phase, dict(t_total_s=round(time.perf_counter() - t0, 2), pick=g["key"], place=h["key"], checked=chk.get("checked"))
