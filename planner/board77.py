"""board77.py: the table edge next to the .77, from the top camera's depth, as the cuRobo "board" obstacle.

  board77.py fit [--write] [--depth FILE.png]

The .77 stands on a lower desk beside the table it reaches over (old room: 197 mm under the table top; the
2026-11-26 exhibit room: ~222 mm). In the depth image the table top is the big flat area near the centre and the
lower desk reads >= 150 mm deeper. A table pixel with a lower-desk pixel within 6 px is an edge pixel. The ones within
350 mm of the .77 base are mapped to the .77 frame (locate: camera world at table height, then cal77.json), and a line
is fitted to them (least squares, then again without the worst third).

The board box keeps its old size: 600 mm into the table, 900 mm along the edge, 214 mm tall. Its near face sits on
that line and its top is the table (real .77 z = workspace table_z_mm). --write replaces the "board" entry in
$MYCOBOT_LAB/workspace.json and keeps the old file as workspace.before-board-<time>.json.

Run it after `cal77.py fit2d` in a new room: the edge is only as good as the camera map.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import locate  # noqa: E402
import plan77  # noqa: E402

DEPTH_URL = "http://127.0.0.1:8090/depth.png"
NEAR_MM = 350.0  # edge pixels this close to the .77 base (xy) are used
STEP_MM = 150.0  # the lower desk is at least this much deeper than the table top
DIMS = [600.0, 900.0, 214.0]


def _shift_or(m, r):
    out = m.copy()
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            if du * du + dv * dv > r * r:
                continue
            out |= np.roll(np.roll(m, dv, 0), du, 1)
    return out


def fit(depth_png: bytes):
    D = locate.CAM.aligned_depth(depth_png)
    H, W = D.shape
    c = D[H // 3:2 * H // 3, W // 3:2 * W // 3]
    d_tab = float(np.median(c[c > 0]))
    table = (D > 0) & (np.abs(D - d_tab) < 20)
    lower = D > d_tab + STEP_MM
    edge = table & _shift_or(lower, 6)
    b = 12  # the re-projected depth has a band of stray values along the image border
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = False, False, False, False
    vv, uu = np.nonzero(edge)
    if len(uu) < 30:
        return None, f"모서리 픽셀이 너무 적어요 ({len(uu)})"
    P = np.array([locate.to77(locate.at(float(u), float(v), plan77.TABLE)) for u, v in zip(uu, vv)])
    P = P[np.hypot(P[:, 0], P[:, 1]) < NEAR_MM]
    if len(P) < 30:
        return None, f".77 근처({NEAR_MM:.0f} mm) 모서리 픽셀이 너무 적어요 ({len(P)})"
    # the dominant straight piece (RANSAC, 6 mm), then least squares on its inliers: a corner or a stray blob next
    # to the arm must not tilt the line
    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        i, j = rng.choice(len(P), 2, replace=False)
        d = P[j] - P[i]
        if np.linalg.norm(d) < 30:
            continue
        t = d / np.linalg.norm(d)
        n = np.array([-t[1], t[0]])
        inl = np.abs((P - P[i]) @ n) < 6.0
        if best is None or inl.sum() > best.sum():
            best = inl
    keep = best
    p0 = P[keep].mean(0)
    _, _, Vt = np.linalg.svd(P[keep] - p0)
    t = Vt[0]
    n = np.array([-t[1], t[0]])
    rms = float(np.sqrt(np.mean(((P[keep] - p0) @ n) ** 2)))
    span = float(np.ptp((P[keep] - p0) @ t))
    # the normal points into the table: toward the table pixels near the .77
    tv, tu = np.nonzero(table)
    pick = np.linspace(0, len(tu) - 1, min(len(tu), 3000)).astype(int)
    T = np.array([locate.to77(locate.at(float(tu[i]), float(tv[i]), plan77.TABLE)) for i in pick])
    T = T[np.hypot(T[:, 0], T[:, 1]) < NEAR_MM]
    foot = p0 + (-p0 @ t) * t  # nearest point of the edge line to the .77 base axis
    if len(T) and (T.mean(0) - foot) @ n < 0:
        n = -n
    dist = float(foot @ n)  # the base on the outer side (-n of the edge) -> positive
    return dict(p0=p0, t=t, n=n, foot=foot, dist=dist, rms=rms, span=span, n_pts=int(keep.sum()), d_tab=d_tab), None


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("fit")
    p.add_argument("--write", action="store_true")
    p.add_argument("--depth", help="a saved top depth PNG instead of the live viewer")
    a = ap.parse_args()
    png = Path(a.depth).read_bytes() if a.depth else urllib.request.urlopen(DEPTH_URL, timeout=5).read()
    r, err = fit(png)
    if r is None:
        print(f"테이블 모서리 실패: {err}")
        return 1
    ws_path = locate.lab_file("workspace.json")
    ws = json.loads(Path(ws_path).read_text())
    table_z = float(ws.get("table_z_mm", 197.0))
    c = r["foot"] + r["n"] * DIMS[0] / 2
    yaw = math.degrees(math.atan2(r["n"][1], r["n"][0]))
    board = {"name": "board", "center_mm": [round(float(c[0]), 1), round(float(c[1]), 1), round(table_z - DIMS[2] / 2, 1)],
             "dims_mm": DIMS, "yaw_deg": round(yaw, 2), "frame": "77",
             "_src": f"board77.py fit {time.strftime('%Y-%m-%d %H:%M')}: top depth edge, {r['n_pts']} px, line rms {r['rms']:.1f} mm; "
                     f"top = table_z {table_z:.0f}"}
    msg = (f"테이블 모서리: .77 중심에서 {r['dist']:.0f} mm, 안쪽 방향 {yaw:+.1f}°, 직선 잔차 {r['rms']:.1f} mm, "
           f"길이 {r['span']:.0f} mm ({r['n_pts']} px, 테이블 깊이 {r['d_tab']:.0f} mm)")
    old = next((o for o in ws.get("obstacles", []) if o.get("name") == "board"), None)
    if old:
        msg += f"; 이전 board 중심 {old['center_mm'][:2]}, yaw {old.get('yaw_deg', 0):+.1f}"
    if r["dist"] < 40 or r["span"] < 120:
        msg += ": 모서리가 .77 중심에 너무 가깝거나 보이는 직선이 짧아요(지도 확인 필요), 쓰지 않아요"
        print(msg)
        print(json.dumps(board, ensure_ascii=False))
        return 1
    if a.write:
        p_ws = locate.LAB / "workspace.json"
        if p_ws.exists():
            p_ws.rename(p_ws.with_name(f"workspace.before-board-{time.strftime('%Y%m%d-%H%M')}.json"))
        ws["obstacles"] = [board if o.get("name") == "board" else o for o in ws.get("obstacles", [])]
        if not any(o.get("name") == "board" for o in ws["obstacles"]):
            ws["obstacles"].insert(0, board)
        p_ws.write_text(json.dumps(ws, indent=1, ensure_ascii=False))
        msg += f" -> {p_ws}"
    print(msg)
    print(json.dumps(board, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
