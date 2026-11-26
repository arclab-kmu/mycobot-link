"""Stir plan for the .70 spoon arm at a measured cup (port of the 2026-11-24 smooth stir planner).

World = the top camera's calib_top3 frame (where the .70 base estimate lives). Motion:
home -> lift (out of the rest pose) -> start above the cup -> min-jerk arc down to the dip -> spiral out,
N circles, spiral in -> arc back up -> start -> home. Checks: spoon inside the cup wall ring, the white block
next to the cup (bright top-camera pixels 45-110 mm from the cup centre), arm clearance, joint steps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = Path(__import__("os").environ.get("MYCOBOT_LAB", "~/Documents/lab-jetson/console")).expanduser()
# spoon-head offset in the flange frame found on 2026-11-24 (the taped spoon sits 19 mm off the old estimate)
import os  # noqa: E402

os.environ.setdefault("SPOFF", "-89.0,-78.3,23.1")
sys.path.insert(0, str(HERE / "s70"))
sys.path.insert(0, str(HERE))
import mv70  # noqa: E402
from sp3 import head_of, pose, pose_any  # noqa: E402

# Rest pose while the .77 works. Until 2026-11-26 it was [1.93, -1.75, 7.38, -13.71, 1.58, -132.45]; the .77's
# 140 deg pour then reached into the .70's wrist (spheres -12 mm, and Hojin saw them touch on run 5), so the shoulder
# leans 20 deg back: +82 mm from today's pours, the upper arm ~33 mm further from the table.
HOME = np.array([1.93, 18.25, 7.38, -13.71, 1.58, -132.45])
LIFT = [np.array(q) for q in json.loads((HERE / "data" / "lift_c3.json").read_text())["lift"]]


def smooth(x):
    return x * x * x * (10 - 15 * x + 6 * x * x)


def bias70():
    """Where to aim the spoon relative to the measured cup (world mm). Each run's centring result updates it."""
    p = LAB / "cal70.json"
    d = json.loads(p.read_text()) if p.exists() else {"aim_offset_mm": [7.6, -10.6], "_src": "2026-11-25 three centring runs"}
    return np.array(d["aim_offset_mm"], float)


class StirPlanner:
    def __init__(self, C, rim, blk_xy, ZD=260.0, TH1=-15.0, RS=8.0, NC=5, PER=2.0, DT=0.1, WALLR=34.0, blk_centre=None):
        self.C = np.asarray(C, float)
        self.RIM, self.ZD, self.TH1, self.RS, self.NC, self.PER, self.DT, self.WALLR = rim, ZD, TH1, RS, NC, PER, DT, WALLR
        mv70.CUP.update({"c": self.C, "r": 39.0, "top": rim + 1})
        B = np.asarray(blk_xy, float).reshape(-1, 2)
        bc = self.C if blk_centre is None else np.asarray(blk_centre, float)
        r = np.hypot(*(B - bc).T) if len(B) else np.zeros(0)
        self.BLK = B[(r > 45) & (r < 110)] if len(B) else B

    def spoon_pts(self, q, n=12):
        h = head_of(q)
        j = mv70.jaw77(q)
        top = h + (j - h) * 0.75
        return np.array([h + (top - h) * f for f in np.linspace(0, 1, n)])

    def rim_margin(self, q):
        S_ = self.spoon_pts(q)
        rr = np.hypot(*(S_[:, :2] - self.C).T)
        low = S_[:, 2] < self.RIM + 4
        m_in = (self.WALLR - rr[low]).min() if low.any() else 99.0
        out = S_[:, 2] < 250
        m_blk = 99.0
        if out.any() and len(self.BLK):
            m_blk = min(np.hypot(*(self.BLK - p[:2]).T).min() for p in S_[out])
        return float(m_in), float(m_blk)

    def plan(self, yaw, TH0, Z0, T_arc=10.0):
        C, DT = self.C, self.DT
        qd = pose_any(np.r_[C, self.ZD], self.TH1, yaw)
        if qd is None:
            return None, "dip IK"
        n = int(T_arc / DT)
        arc = [qd]
        q = qd
        for k in range(1, n + 1):
            f = 1 - smooth(1 - k / n)
            s = 1 - f
            h = np.r_[C, self.ZD + (Z0 - self.ZD) * (1 - s)]
            th = self.TH1 + (TH0 - self.TH1) * (1 - s)
            qn, _, _ = pose(h, th, yaw, q)
            if qn is None or np.abs(qn - q).max() > 4:
                return None, f"arc z {h[2]:.0f} " + ("IK" if qn is None else "jump")
            mi, mb = self.rim_margin(qn)
            if (h[2] < self.RIM + 30 and mi < 6) or mb < 12 or mv70.clearance(qn, ignore_sty=True) < 5:
                return None, f"arc z {h[2]:.0f} rim {mi:.1f} blk {mb:.1f}"
            arc.append(qn)
            q = qn
        down = arc[::-1]
        NC, PER, RS = self.NC, self.PER, self.RS
        T = (NC + 1) * PER
        ts = np.arange(DT, T + 1e-9, DT)
        w = 2 * np.pi / PER

        def phi(t):
            a = PER / 2
            if t < a:
                return w * t * t / (2 * a)
            if t > T - a:
                tt = T - t
                return w * (T - a) - w * tt * tt / (2 * a)
            return w * (t - a / 2)

        def rad(t):
            if t < PER:
                return RS * smooth(t / PER)
            if t > T - PER:
                return RS * smooth((T - t) / PER)
            return RS

        stir = []
        q = qd
        worst = [99.0, 99.0, 0.0]
        for t in ts:
            p = C + rad(t) * np.r_[np.cos(phi(t)), np.sin(phi(t))]
            qn, _, _ = pose(np.r_[p, self.ZD], self.TH1, yaw, q)
            if qn is None or np.abs(qn - q).max() > 4:
                return None, f"stir t {t:.1f} " + ("IK" if qn is None else "jump")
            mi, mb = self.rim_margin(qn)
            worst = [min(worst[0], mi), min(worst[1], mb), max(worst[2], float(np.abs(qn - q).max()))]
            if mi < 5 or mb < 12:
                return None, f"stir t {t:.1f} rim {mi:.1f} blk {mb:.1f}"
            stir.append(qn)
            q = qn
        return dict(yaw=yaw, TH0=TH0, Z0=Z0, start=down[0], down=down[1:], stir=stir, up=arc[1:], worst=worst), "ok"

    def best(self, yaws=(20, 10, 30, 0), starts=((-50, 322), (-55, 325), (-45, 320), (-60, 325))):
        tried = []
        for yaw in yaws:
            for TH0, Z0 in starts:
                P, why = self.plan(yaw, TH0, Z0)
                tried.append(f"yaw {yaw} TH0 {TH0} Z0 {Z0}: {why}")
                if P is not None:
                    return P, tried
        return None, tried


def transit(start, vpeak=45.0):
    """home -> lift -> start as joint waypoints (the caller re-times them)."""
    QL = LIFT[-1]
    W = [HOME] + LIFT + [QL + (np.asarray(start) - QL) * f for f in np.linspace(0.1, 1, 10)]
    return [w for i, w in enumerate(W) if i == 0 or np.linalg.norm(w - W[i - 1]) > 1e-6]


def axis_and_px2mm(at, world_to_pixel, rim_px, rim_z, ZD):
    """Pixel where the spoon bowl should appear at the dip, and the pixel -> mm map there (top camera)."""
    Cw = at(*rim_px, rim_z)
    ax = np.asarray(rim_px) + (np.array(world_to_pixel(Cw + [0, 0, ZD - rim_z])) - np.array(world_to_pixel(Cw)))
    px2mm = np.column_stack([(at(ax[0] + 1, ax[1], ZD) - at(ax[0] - 1, ax[1], ZD))[:2] / 2,
                             (at(ax[0], ax[1] + 1, ZD) - at(ax[0], ax[1] - 1, ZD))[:2] / 2])
    return ax, px2mm


def jac_mm(q, TH1, yaw):
    """deg per mm of spoon-head shift in world x / y at q, keeping the spoon tilt and yaw (IK differences)."""
    q = np.asarray(q, float)
    h0 = head_of(q)
    J = np.zeros((6, 2))
    for k in range(2):
        d = np.zeros(3)
        d[k] = 1.5
        qp, _, _ = pose(h0 + d, TH1, yaw, q)
        qm, _, _ = pose(h0 - d, TH1, yaw, q)
        if qp is None or qm is None:
            raise RuntimeError("centring Jacobian: IK failed")
        J[:, k] = (qp - qm) / 3.0
    return J
