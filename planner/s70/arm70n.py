"""Frame helpers for the .70 arm: its base pose in the .77 base frame (from the J1 sweep) and tool-tip IK."""
import os as _os; _P = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, subprocess
sys.path.insert(0, _P)
import numpy as np, kin
S = _P
BX, BY, YAW, BZ = 188.0, 313.0, -94.0, -25.0   # 2026-11-23 J1 sweep in D405 (axis +-15 mm), yaw from T0/upright, z rough
# 2026-11-27: in a new room the pose comes from the lab workspace (arms.70 origin_mm / yaw_deg, written by
# recam.py arm70 in the top camera's world); the constants above are the old room's
try:
    _ws = json.load(open(_os.path.join(_os.path.expanduser(_os.environ.get("MYCOBOT_LAB", "~/Documents/lab-jetson/console")),
                                       "workspace.json")))["arms"]["70"]
    BX, BY, BZ = (float(v) for v in _ws["origin_mm"])
    YAW = float(_ws.get("yaw_deg", YAW))
except (OSError, KeyError, ValueError):
    pass
TIP = np.eye(4); TIP[2, 3] = 110.0                                 # closed-gripper tip, estimate
c, s = np.cos(np.radians(YAW)), np.sin(np.radians(YAW))
R77_70 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])               # .70 frame -> .77 frame rotation
def to70(p77): return R77_70.T @ (np.asarray(p77, float) - np.array([BX, BY, BZ]))
def to77(p70): return R77_70 @ np.asarray(p70, float) + np.array([BX, BY, BZ])
def dir70(d77): return R77_70.T @ np.asarray(d77, float)
def tip77(q): return to77((kin.fk(q) @ TIP)[:3, 3])
def ik_tip(p77, approach77, q_seed, roll_ref=None):
    z = dir70(approach77); z = z / np.linalg.norm(z)
    up = np.array([0, 0, 1.0]); x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.column_stack([x, y, z]); T[:3, 3] = to70(p77)
    best = None
    for roll in (0, 90, -90, 180):
        a = np.radians(roll); Rr = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        TT = T.copy(); TT[:3, :3] = T[:3, :3] @ Rr
        q, pe, re = kin.ik(TT, q_seed, tool=TIP, iters=1500)
        if pe < 1.5 and re < 1.5:
            d = abs(q[5] - (roll_ref if roll_ref is not None else q_seed[5]))
            if best is None or d < best[0]: best = (d, q)
    return None if best is None else best[1]
