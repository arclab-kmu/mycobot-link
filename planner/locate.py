"""Pixels from detect.py -> object positions in the .77 base frame and in the .70 planning world.

The top camera model is the 2026-11-24 'before-recal' workspace camera (calib_top3); a ray is cut at the
object's known height (cap top 264 mm, cup rim 286 mm in .77 z). ``cal77.json`` maps that world to the
.77 base frame (2-D similarity, refit by ``cal77.py`` from vial placements); the .70 stir planner works
in the camera world directly.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "brain" / "src"))
from mycobot_brain.console import workspace  # noqa: E402

LAB = Path(os.environ.get("MYCOBOT_LAB", "~/Documents/lab-jetson/console")).expanduser()
REPO_LAB = HERE.parent / "lab" / "console"


def lab_file(name: str) -> Path:
    """A lab file from MYCOBOT_LAB, or the snapshot in the repository's lab/console/ if it is missing there."""
    p = LAB / name
    return p if p.exists() else REPO_LAB / name


CAM = workspace.load(str(lab_file("workspace.before-recal-20261124.json"))).cameras["top"]
CAP_Z, RIM_Z = 264.0, 286.0


def cal77():
    p = LAB / "cal77.json"
    if not p.exists():
        p = HERE / "data" / "cal77_default.json"
    return json.loads(p.read_text())


def at(u, v, z):
    c, d = CAM.pixel_ray(u, v)
    return c + d * (z - c[2]) / d[2]


def to77(w_xy, cal=None):
    cal = cal or cal77()
    return np.array(cal["R2"]) @ np.asarray(w_xy)[:2] + np.array(cal["T"])


def from77(p_xy, cal=None):
    cal = cal or cal77()
    return np.array(cal["R2"]).T @ (np.asarray(p_xy)[:2] - np.array(cal["T"]))


def px_of77(p77_xy, z, cal=None):
    w = from77(p77_xy, cal)
    return np.array(CAM.world_to_pixel(np.r_[w, z]))


def heights(depth_png: bytes, pts, r=3):
    dep = CAM.aligned_depth(depth_png)
    out = []
    for u, v in pts:
        p, _ = CAM.pixel_to_world(u, v, dep, r=r)
        out.append(None if p is None else float(p[2]))
    return out


def rim_px_radius(u, v, r_mm=37.2, z=RIM_Z):
    """Pixel radius of a horizontal circle of r_mm at height z centred under pixel (u, v)."""
    c = at(u, v, z)
    ps = [np.array(CAM.world_to_pixel(c + [r_mm * np.cos(a), r_mm * np.sin(a), 0.0])) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    return float(np.mean([np.hypot(*(p - [u, v])) for p in ps]))
