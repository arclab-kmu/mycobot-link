"""Motion helpers for the .70 arm with the new base estimate (arm70n): IK along Cartesian lines, obstacle clearance, live moves + photos."""
import os as _os; _P = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, subprocess, urllib.request, time, numpy as np
D = _P; sys.path.insert(0, _os.path.join(_P, "s70")); sys.path.insert(1, _P)
import arm70n as A; sys.modules["arm70"] = A
import kin
PY = "/home/hkim/IsaacLab/.venv/bin/python"
JAW = np.array([0.0, 0.0, 100.0])
STY = dict(c=np.array([237.0, 100.0]), half=17.0, top=331.0); CUP = dict(c=np.array([116.0, 88.0]), r=37.0, top=258.0)
def R77(yaw, pitch, roll):
    a, b = np.radians(yaw), np.radians(pitch); zf = np.array([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), -np.sin(b)])
    xh = np.cross([0, 0, 1.0], zf); xh /= np.linalg.norm(xh); yh = np.cross(zf, xh); r = np.radians(roll)
    xf = np.cos(r) * xh + np.sin(r) * yh; return np.column_stack([xf, np.cross(zf, xf), zf])
def ik(p, Rw, seed, iters=800):
    R70 = A.R77_70.T @ Rw; T = np.eye(4); T[:3, :3] = R70; T[:3, 3] = A.to70(p) - R70 @ JAW
    q, pe, re = kin.ik(T, np.asarray(seed, float), iters=iters)
    return q if (pe < 1.0 and re < 1.0 and np.all(q > kin.LO) and np.all(q < kin.HI)) else None
def body_pts(q, grip=True):
    T = np.eye(4); O = [np.zeros(3)]
    for qi, (off, d, a, al) in zip(np.radians(q), kin.DH):
        T = T @ kin._A(qi + off, d, a, al); O.append(T[:3, 3].copy())
    P = [O[i] + (O[i + 1] - O[i]) * s for i in range(len(O) - 1) for s in np.linspace(0, 1, 6)]
    if grip: P += list((T @ np.array([[x, y, z, 1.0] for x in (-28, 0, 28) for y in (-28, 0, 28) for z in (0, 40, 70)]).T).T[:, :3])
    return np.array([A.to77(p) for p in P])
def clearance(q, ignore_sty=False):
    P = body_pts(q); out = 99.0
    if not ignore_sty:
        dx = np.maximum(np.abs(P[:, :2] - STY["c"]) - STY["half"], 0); dh = np.hypot(dx[:, 0], dx[:, 1]); low = P[:, 2] < STY["top"] + 5
        if low.any(): out = min(out, float(dh[low].min()))
    dc = np.hypot(*(P[:, :2] - CUP["c"]).T) - CUP["r"]; low = P[:, 2] < CUP["top"] + 5
    if low.any(): out = min(out, float(dc[low].min()))
    board = (P[:, 1] < 200) & (P[:, 0] > 0); 
    if board.any(): out = min(out, float(P[board, 2].min() - 205))
    return out
def line(p0, p1, Rw, seed, step=6.0, jump=15.0):
    n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / step))); qs = []; q = np.asarray(seed, float)
    for i in range(1, n + 1):
        qn = ik(np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * i / n, Rw, q)
        if qn is None or np.max(np.abs(qn - q)) > jump: return None
        q = qn; qs.append(np.round(q, 2).tolist())
    return qs
def jpath_clear(a, b, n=25, **kw): return min(clearance(np.asarray(a) + (np.asarray(b) - np.asarray(a)) * s, **kw) for s in np.linspace(0, 1, n))
def arm(acts, timeout=300):
    r = subprocess.run([PY, f"{D}/vlm/arm_step.py", "--port", "30978", "--require-live", "--log", f"{D}/stir/live70.jsonl", json.dumps(acts)], capture_output=True, text=True, timeout=timeout)
    ev = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]; err = [e for e in ev if e.get("event") == "error"]
    if err: print("ARM ERROR", err[-1]); return None
    return ev[-1].get("q") if ev else None
def photo(tag):
    for nm, url in [("_s.jpg", "8092/color.jpg"), ("_sd.png", "8092/depth.png"), ("_t.jpg", "8090/color.jpg"), ("_td.png", "8090/depth.png")]:
        try: open(f"{D}/stir/ph/{tag}{nm}", "wb").write(urllib.request.urlopen(f"http://127.0.0.1:{url}", timeout=5).read())
        except Exception as e: print("photo fail", tag, nm, e)
def jaw77(q): T = kin.fk(q); return A.to77((T @ np.r_[JAW, 1.0])[:3])
_DB = None
def ik2(p, Rw, ref=None, k=25, iters=300):
    global _DB
    if _DB is None: _DB = np.load(f"{D}/data/seeddb.npz")
    dpos = np.linalg.norm(_DB["p"] - p, axis=1); drot = np.linalg.norm(_DB["z"] - Rw[:, 2], axis=1) * 100 + np.linalg.norm(_DB["x"] - Rw[:, 0], axis=1) * 40
    idx = np.argsort(dpos + drot)[:k]; best = None
    seeds = ([ref] if ref is not None else []) + [_DB["q"][i] for i in idx]
    for sd in seeds:
        q = ik(p, Rw, sd, iters=iters)
        if q is None: continue
        cost = 0 if ref is None else float(np.max(np.abs(q - ref)))
        if best is None or cost < best[0]: best = (cost, q)
        if ref is not None and cost < 10: break
    return None if best is None else best[1]
def line2(p0, p1, Rw, seed, step=6.0, jump=12.0):
    n = max(1, int(np.ceil(np.linalg.norm(np.subtract(p1, p0)) / step))); qs = []; q = np.asarray(seed, float)
    for i in range(1, n + 1):
        tgt = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * i / n; qn = ik(tgt, Rw, q, iters=400)
        if qn is None or np.max(np.abs(qn - q)) > jump: return None
        q = qn; qs.append(np.round(q, 2).tolist())
    return qs
from scipy.optimize import least_squares
def _rotvec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-9: return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
def ik(p, Rw, seed, iters=None):
    R70 = A.R77_70.T @ Rw; tp = A.to70(p) - R70 @ JAW
    def f(qd):
        T = kin.fk(qd); return np.r_[T[:3, 3] - tp, 60.0 * _rotvec(R70 @ T[:3, :3].T)]
    s = np.clip(np.asarray(seed, float), kin.LO + 0.5, kin.HI - 0.5)
    r = least_squares(f, s, bounds=(kin.LO, kin.HI), xtol=1e-8, ftol=1e-8, max_nfev=300)
    e = f(r.x); pe, re = np.linalg.norm(e[:3]), np.degrees(np.linalg.norm(e[3:]) / 60.0)
    return r.x if (pe < 1.0 and re < 1.0) else None
def goto_fb(p, Rw, q_meas, speed=10, iters=3, tol=0.8, tag=None):
    """Plan a line to p, move, then correct joint tracking error from the measured q (feed the residual back)."""
    qs = line2(jaw77(q_meas), p, Rw, q_meas, step=3, jump=8)
    if qs is None: print("goto_fb: plan fail"); return None
    qa = arm([["move", q, speed] for q in qs] + [["wait", 1.0]]); qt = np.array(qs[-1]); cmd = qt.copy()
    for k in range(iters):
        if qa is None: return None
        e = jaw77(qa) - p; print(f"  fb{k} meas jaw err mm", np.round(e, 1), "q err", np.round(np.array(qa) - qt, 2))
        if np.linalg.norm(e) < tol: break
        cmd = cmd + (qt - np.array(qa)); qa = arm([["move", np.round(cmd, 2).tolist(), 6], ["wait", 1.2]])
    if tag: photo(tag)
    return qa
