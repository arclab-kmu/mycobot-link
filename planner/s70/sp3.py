import sys, json, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np, mv70, kin
from mv70 import A
mv70.JAW[:] = [0, 0, 85]
QM = [26.01, -23.02, 55.19, -22.32, 2.54, -131.57]
def fr(q): return A.R77_70 @ kin.fk(q)[:3, :3]
OFF = fr(QM).T @ (np.array([201.7, 144.3, 271.0]) - mv70.jaw77(QM))
import os
if os.environ.get('SPOFF'): OFF = np.array([float(x) for x in os.environ['SPOFF'].split(',')])
R0 = fr([1.93, -1.75, 7.38, -13.71, 1.58, -132.45])
def Rz(d):
    t = np.radians(d); c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def Rx(d):
    t = np.radians(d); c, s = np.cos(t), np.sin(t); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Rof(th, yaw=0): return Rz(yaw) @ Rx(th) @ R0
def head_of(q): return mv70.jaw77(q) + fr(q) @ OFF
def pose(h, th, yaw, seed):
    R = Rof(th, yaw); jt = np.asarray(h, float) - R @ OFF; return mv70.ik(jt, R, seed), R, jt
def cross_r(h, R, C, z=256):
    top = h + R @ (-OFF) * 0.8; v = top - h; s = (z - h[2]) / v[2]; p = h + s * v; return np.hypot(*(p[:2] - C))
if __name__ == "__main__":
    C = np.array([141.0, 41.0]); print('OFF', OFF.round(1))
    for yaw in (0, 10, 20, -10):
        for th in (-10, -20, -30, -40):
            q, R, jt = pose(np.r_[C, 225], th, yaw, QM)
            qh, _, _ = pose(np.r_[C, 280], th, yaw, QM if q is None else q)
            if q is not None: print('yaw', yaw, 'tilt', th, 'dip ok, hover', qh is not None, 'cross r', round(cross_r(np.r_[C, 225], R, C), 1), 'clr', round(mv70.clearance(q, ignore_sty=True), 1), np.round(q, 1).tolist())
_SEEDS = [np.array(s) for s in ([13.53,-37.79,21.26,27.68,-9.75,-134.12],[33.1,-56.5,99.5,-29.3,-28.9,-138.8],[16.8,-67.5,90.3,-10.5,-13.0,-134.9],[2.1,-22.4,55.2,-20.9,1.4,-131.8],[1.93,-1.75,7.38,-13.71,1.58,-132.45],[26.7,-77.4,62.6,34.7,-31.6,-142.8])]
def pose_any(h, th, yaw, extra=(), nrand=25, pick='clr'):
    rng = np.random.default_rng(0); best = None
    seeds = list(extra) + _SEEDS + [rng.uniform(kin.LO * 0.8, kin.HI * 0.8) for _ in range(nrand)]
    for sd in seeds:
        q, R, jt = pose(h, th, yaw, sd)
        if q is None: continue
        sc = mv70.clearance(q, ignore_sty=True) - 0.05 * np.abs(q - _SEEDS[0]).sum()
        if best is None or sc > best[0]: best = (sc, q)
    return None if best is None else best[1]
