"""pathcheck.py: clearance of a planned .77 scenario to what the top depth sees, under map errors. Read-only gate.

  pathcheck.py SCEN.json [--d 15] [--yaw 10] [--skip X77 Y77 R] [--min 30] [--frames 5] [--depth FILE.png]

From the AGX session's lab tool path_vs_depth.py (2026-11-27), which caught the black pole and the board edge before the
first calibration motion at the exhibit site. Same idea, faster: the arm spheres come from the local DH sphere fit
(spheres.py, numpy, ~2 mm from cuRobo /spheres) instead of one HTTP call per sample, and both the sweep and the depth
points are thinned to 8 mm voxels (the AGX tool took 4-5 min for 27 maps x 250 samples).

- Points: median of --frames top depth frames, 200 mm under to 450 mm over the table, in the camera world.
- The .77 axis and yaw come from cal77.json and are perturbed by +-d mm (x, y) and +-yaw deg: 27 maps.
- Self points (the arm at its start pose: within 40 mm of its spheres, inside r 75 of the axis, or within 90 mm of the
  spheres inside r 150 above the table) are dropped per map. --skip drops the target vial (world-fixed, below +90).
- Clearance = min over the sweep of (distance to the nearest point - sphere radius), with and without the table-top
  points (+-15 mm of the table: the board box covers those).
Exit 1 when the worst clearance without the table top is under --min.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kin77  # noqa: E402
import locate  # noqa: E402
import plan77  # noqa: E402
import spheres  # noqa: E402

DEPTH_URL = "http://127.0.0.1:8090/depth.png"


def world_points(frames):
    from PIL import Image
    F = np.stack([np.asarray(Image.open(io.BytesIO(f))).astype(float) for f in frames])
    F[F == 0] = np.nan
    D = np.nanmedian(F, 0) if len(F) > 1 else F[0]
    cam = locate.CAM
    h, w = D.shape
    vv, uu = np.mgrid[0:h, 0:w]
    K = cam.depth_K
    X = np.stack([(uu - K["ppx"]) / K["fx"] * D, (vv - K["ppy"]) / K["fy"] * D, D], -1).reshape(-1, 3)
    X = X[np.isfinite(X[:, 2])]
    Xc = X @ np.asarray(cam.R_dc).T + np.asarray(cam.t_dc)
    return (Xc - np.asarray(cam.t)) @ np.asarray(cam.R)


def voxel(P, size, r=None):
    """Thin points (or spheres, keeping the largest radius) to one per `size` mm voxel."""
    k = np.floor(P[:, :3] / size).astype(np.int64)
    _, idx, inv = np.unique(k, axis=0, return_index=True, return_inverse=True)
    if r is None:
        return P[idx]
    rmax = np.zeros(len(idx))
    np.maximum.at(rmax, inv.ravel(), r)
    return np.column_stack([P[idx, :3], rmax])


def nearest(A, B):
    """Distance from each row of A to the nearest row of B (3-D)."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree(B).query(A)[0]
    except ImportError:
        out = np.full(len(A), np.inf)
        for i in range(0, len(A), 512):
            a = A[i:i + 512]
            for j in range(0, len(B), 8192):
                d = np.sqrt(((a[:, None, :] - B[None, j:j + 8192, :]) ** 2).sum(-1)).min(1)
                out[i:i + 512] = np.minimum(out[i:i + 512], d)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--d", type=float, default=15.0, help="axis error to try, +-mm in x and y")
    ap.add_argument("--yaw", type=float, default=10.0, help="yaw error to try, +-deg")
    ap.add_argument("--skip", nargs=3, type=float, metavar=("X77", "Y77", "R"), help="the target vial (.77 frame)")
    ap.add_argument("--min", type=float, default=30.0, help="required clearance without the table top (mm)")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--depth", help="a saved top depth PNG instead of live frames")
    a = ap.parse_args()
    t0 = time.perf_counter()
    scen = json.loads(Path(a.scenario).read_text())
    Q = [np.array(sg["q"], float) for ph in scen["phases"] if str(ph.get("arm")) == "77"
         for sg in ph["segments"] if sg["kind"] == "joints"]
    if not Q:
        print("시나리오에 .77 관절 구간이 없어요")
        return 1
    Q = np.vstack(Q)[::2]
    frames = [Path(a.depth).read_bytes()] if a.depth else \
        [urllib.request.urlopen(DEPTH_URL, timeout=5).read() for _ in range(a.frames)]
    W = world_points(frames)
    t = plan77.TABLE
    W = W[(W[:, 2] > t - 200.0) & (W[:, 2] < t + 450.0)]
    cal = locate.cal77()
    R2, T2 = np.array(cal["R2"]), np.array(cal["T"])
    if a.skip:
        sw = R2.T @ (np.array(a.skip[:2]) - T2)
        W = W[~((np.hypot(W[:, 0] - sw[0], W[:, 1] - sw[1]) < a.skip[2]) & (W[:, 2] < t + 90.0))]
    W = voxel(W, 8.0)
    axis = -R2.T @ T2
    yaw0 = math.degrees(math.atan2(R2[0, 1], R2[0, 0]))  # base -> world rotation is R2^T
    S_all = np.vstack([spheres.spheres(q) for q in Q])  # real .77 base frame
    S = voxel(S_all, 8.0, S_all[:, 3])
    S0 = spheres.spheres(Q[0])
    z_tab = t + kin77.BASE_DZ  # the table in the real base frame
    res = []
    for dx in (-a.d, 0.0, a.d):
        for dy in (-a.d, 0.0, a.d):
            for dw in (-a.yaw, 0.0, a.yaw):
                c, s = math.cos(math.radians(yaw0 + dw)), math.sin(math.radians(yaw0 + dw))
                xy = W[:, :2] - (axis + [dx, dy])
                P = np.column_stack([c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1], W[:, 2] + kin77.BASE_DZ])
                d0 = np.concatenate([(np.linalg.norm(P[i:i + 8192, None, :] - S0[None, :, :3], axis=2) - S0[None, :, 3]).min(1)
                                     for i in range(0, len(P), 8192)])  # to the start pose's spheres
                rr = np.hypot(P[:, 0], P[:, 1])
                obj = P[~((d0 <= 40.0) | (rr < 75.0) | ((d0 < 90.0) & (rr < 150.0) & (P[:, 2] > z_tab + 15.0)))]
                top = np.abs(obj[:, 2] - z_tab) < 15.0
                c_all = float((nearest(S[:, :3], obj) - S[:, 3]).min()) if len(obj) else float("inf")
                o2 = obj[~top]
                if len(o2):
                    dd = nearest(S[:, :3], o2) - S[:, 3]
                    k = int(np.argmin(dd))
                    near = o2[int(np.argmin(np.linalg.norm(o2 - S[k, :3], axis=1)))]
                    c_obj = float(dd[k])
                else:
                    c_obj, near = float("inf"), None
                res.append((c_all, c_obj, dx, dy, dw, near))
    nom = next(r for r in res if r[2] == 0 and r[3] == 0 and r[4] == 0)
    worst = min(res, key=lambda r: r[1])
    msg = (f"경로-깊이 여유: 지도 그대로 {nom[1]:.0f} mm (테이블 면 포함 {nom[0]:.0f}), 27개 지도 중 최악 {worst[1]:.0f} mm "
           f"(축 {worst[2]:+.0f}/{worst[3]:+.0f} mm, yaw {worst[4]:+.0f}°, 테이블 면 포함 최악 {min(r[0] for r in res):.0f})")
    if worst[5] is not None:
        p = worst[5]
        msg += (f"; 가장 가까운 점 .77 반지름 {math.hypot(p[0], p[1]):.0f} mm, 방위 {math.degrees(math.atan2(p[1], p[0])):+.0f}°, "
                f"테이블 {p[2] - z_tab:+.0f} mm")
    msg += f"; 표본 {len(Q)}, 구 {len(S)}, 점 {len(W)}, {time.perf_counter() - t0:.1f} s"
    ok = worst[1] >= a.min
    print(msg + ("" if ok else f": {a.min:.0f} mm 미만이라 통과 못 해요"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
