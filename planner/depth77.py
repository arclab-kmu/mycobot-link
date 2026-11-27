"""depth77.py: things standing on the table, from the top camera's depth, as cuRobo boxes for the .77.

Hojin (2026-11-26 night): "curobo나 경로 검사 시에는 depth를 이용하고 ... 테이블 위의 물건 위치는 매번 바뀔 거야".
Every plan builds its obstacles from a fresh depth frame instead of a list of known objects.

The depth pixels go straight through the depth intrinsics and the depth->colour extrinsics into the camera world.
There is no re-projection onto the colour grid: ~10 ms on the AGX instead of ~170 ms (measured there 2026-11-26).
Then:
- keep points 12-250 mm above the table, and move them into the .77 frame (xy: the cal77.json map; z: the
  table-referenced planner frame, table at 197, the same as the camera world);
- drop points that belong to the .77 arm (its collision spheres at the current joints + 50 mm), the vial about to be
  grasped, or the cup's inside (the planner's cup walls cover the cup; a depth box over it would forbid the pour);
- bin the rest on a 20 mm grid; connected cells become boxes, cut into tiles <= 60 mm so one blob cannot wall off free
  table. Each box runs from the table to the blob's highest point + 10 mm and is 10 mm larger than its cells.

  depth77.py [--q Q6] [--skip X Y R]...   prints the boxes and the time taken (live top depth)
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kin77  # noqa: E402
import locate  # noqa: E402
import plan77  # noqa: E402

DEPTH_URL = "http://127.0.0.1:8090/depth.png"
# Depth points of the .77 itself sat 4-42 mm (median 22) outside its collision spheres at PARK (servo housings,
# cable, map error ~6 mm; old-room frame 2026-11-26), so 25 mm left the arm in as a box next to the grasp.
ARM_MARGIN = 50.0


def points77(depth_png: bytes, stride: int = 2) -> np.ndarray:
    """Depth pixels -> points (N, 3) in the .77 planner frame (xy .77, z table-referenced)."""
    from PIL import Image
    cam = locate.CAM
    d = np.asarray(Image.open(io.BytesIO(depth_png)))[::stride, ::stride].astype(np.float64)
    vv, uu = np.nonzero(d > 0)
    z = d[vv, uu]
    uu, vv = uu * stride, vv * stride
    D = cam.depth_K
    X = np.stack([(uu - D["ppx"]) / D["fx"] * z, (vv - D["ppy"]) / D["fy"] * z, z], 1)
    Xc = X @ np.asarray(cam.R_dc).T + np.asarray(cam.t_dc)
    W = (Xc - np.asarray(cam.t)) @ np.asarray(cam.R)  # R^T (p - t), row-wise: camera world
    cal = locate.cal77()
    R2, T2 = np.array(cal["R2"]), np.array(cal["T"])
    xy = W[:, :2] @ R2.T + T2
    return np.column_stack([xy, W[:, 2]])


def boxes(depth_png: bytes, q77=None, skip=(), cup=None, cup_r=45.0, max_boxes=30, cell=20.0, tile=60.0,
          reach=420.0, prefix="dep", low=12.0, pad=0.0, base_r=85.0):
    """cuRobo boxes (real .77 frame, via plan77.box) of what stands on the table. `skip`: [(x, y, r)] to leave out.
    low < 0 also takes what stands lower than the table top (the .77's own lower desk: router, keyboard), down to
    table + low; the table top itself (+-12 mm) is never a box. pad grows every box sideways and upward (map error)."""
    P = points77(depth_png)
    t = plan77.TABLE
    # cells where the table top itself is seen: an object whose cells mostly show the table too is held up off it
    # (2026-11-27: a black pole ~140-180 mm above the table became a wall from the table up and blocked every grasp)
    on_tab = P[np.abs(P[:, 2] - t) < 12.0]
    tab_cells = {tuple(k) for k in np.unique(np.floor(on_tab[:, :2] / cell).astype(int), axis=0)} if len(on_tab) else set()
    # above the table, inside base_r of the .77 axis is the arm's own column and cable (2026-11-27: a chunk at r ~70
    # survived the 50 mm sphere margin at the room park and blocked joint 4 by 0.5 mm). Below the table top (the
    # lower desk, low < 0) things next to the base stay: a router stood there that morning.
    rxy = np.hypot(P[:, 0], P[:, 1])
    keep = (P[:, 2] > t + low) & (P[:, 2] < t + 250.0) & (rxy < reach) & ((rxy > base_r) | (P[:, 2] < t))
    if low < 12.0:
        keep &= np.abs(P[:, 2] - t) >= 12.0
    P = P[keep]
    if q77 is not None and len(P):
        import spheres
        S = spheres.spheres(np.asarray(q77, float)).copy()
        S[:, 2] -= kin77.BASE_DZ  # real base frame -> planner frame
        near = np.zeros(len(P), bool)
        for c0 in range(0, len(S), 32):
            blk = S[c0:c0 + 32]
            dd = np.linalg.norm(P[:, None, :] - blk[None, :, :3], axis=2) - blk[None, :, 3]
            near |= (dd < ARM_MARGIN).any(1)
        P = P[~near]
    for x, y, r in skip:
        P = P[np.hypot(P[:, 0] - x, P[:, 1] - y) > r]
    if cup is not None:
        P = P[np.hypot(P[:, 0] - cup[0], P[:, 1] - cup[1]) > cup_r]
    if not len(P):
        return []
    ij = np.floor(P[:, :2] / cell).astype(int)
    keys, inv, cnt = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    top = np.full(len(keys), -1e9)
    np.maximum.at(top, inv, P[:, 2])
    bot = np.full(len(keys), 1e9)
    np.minimum.at(bot, inv, P[:, 2])
    occ = {tuple(k): float(h) for k, h, n in zip(keys, top, cnt) if n >= 3}  # a stray pixel or two is not an object
    under = {tuple(k): float(b) for k, b in zip(keys, bot)}
    per = max(1, int(round(tile / cell)))
    seen, out = set(), []
    for k in sorted(occ, key=lambda k_: -occ[k_]):
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.append(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in occ and n not in seen:
                    seen.add(n)
                    stack.append(n)
        tiles = {}
        for c in comp:
            tiles.setdefault((c[0] // per, c[1] // per), []).append(c)
        floating = (min(under[c] for c in comp) > t + 100.0  # vials (+65) and the cup (+83) never float
                    and np.mean([c in tab_cells for c in comp]) > 0.6)
        for cs in tiles.values():
            a = np.array(cs)
            lo, hi = a.min(0) * cell - 10.0 - pad, (a.max(0) + 1) * cell + 10.0 + pad
            zt = max(occ[c] for c in cs) + 10.0 + pad
            zb = min(under[c] for c in cs)
            if floating:  # held above the table (table seen under/beside it): from its lowest seen point down by the pad
                z0 = max(t, zb - 10.0 - pad - 20.0)  # + 20: the top camera sees the top, not the underside
            else:  # on the table: from the table up; below the table top: from its own bottom
                z0 = t if zb >= t - 12.0 else zb - 10.0 - pad
            out.append((zt, lo, hi, z0))
    out.sort(key=lambda b: -b[0])  # tallest first if there are too many
    return [plan77.box(f"{prefix}{i}", (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z0, zt, hi[0] - lo[0], hi[1] - lo[1])
            for i, (zt, lo, hi, z0) in enumerate(out[:max_boxes])]


if __name__ == "__main__":
    import argparse
    import urllib.request
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", nargs=6, type=float)
    ap.add_argument("--skip", nargs=3, type=float, action="append", default=[])
    ap.add_argument("--depth", help="a saved depth PNG instead of the live viewer")
    ap.add_argument("--low", type=float, default=12.0)
    ap.add_argument("--pad", type=float, default=0.0)
    a = ap.parse_args()
    png = Path(a.depth).read_bytes() if a.depth else urllib.request.urlopen(DEPTH_URL, timeout=5).read()
    t0 = time.perf_counter()
    bx = boxes(png, a.q, a.skip, low=a.low, pad=a.pad, max_boxes=60)
    dt = time.perf_counter() - t0
    for b in bx:
        print(b["name"], [round(v) for v in b["pose"][:3]], [round(v) for v in b["dims_mm"]])
    print(f"{len(bx)} boxes in {dt * 1000:.0f} ms")
