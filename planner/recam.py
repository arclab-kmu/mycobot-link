"""recam.py: after the top camera was moved, register it again from its depth alone, without moving an arm.

Hojin (2026-11-27 morning) plugged both cameras into the AGX and re-aimed the top one. The view check there found the
same tilt (0.2 deg from the stored pose) but the camera ~40 mm higher and yawed ~30-38 deg, so the stored camera pose
and cal77.json (camera world -> .77) no longer hold. Three steps, each a separate run:

  recam.py plane [--depth F]           the table plane in the colour-camera frame: tilt and height vs the stored pose
  recam.py pose  [--depth F] [--write] a new top-camera pose. World z = the table normal with the table at plan77.TABLE
                                       (the table-referenced planner frame); world x = the old world x projected onto
                                       the table; the camera centre keeps its old xy. The in-plane yaw and xy are
                                       arbitrary: the next step (and later fit2d) map them to the .77.
                                       --write: $MYCOBOT_LAB/topcam.json (locate.py reads it) and workspace.json
                                       cameras.top (the console), old files kept as *.before-recam-<time>.json
  recam.py arm77 [--q Q6] [--near X Y | --near-px U V] [--depth F] [--png OUT] [--write]
                                       where the .77 stands in that world, from its own depth silhouette: the cuRobo
                                       spheres at joints q (default: the console's live q) rendered as a top-view
                                       height map, matched to the observed height map over x, y and yaw.
                                       --write: cal77.json + workspace.json arms.77 (cal77.save_map), if the fit is clear

The arm77 map is a provisional one (about 5-10 mm, the sphere model is not the real shell). It is good enough for the
cal77.py hover plan (keep-out/fence margins >= 40 mm), and cal77.py fit2d replaces it with the measured map.
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

DEPTH_URL = "http://127.0.0.1:8090/depth.png"
COLOR_URL = "http://127.0.0.1:8090/color.jpg"
CONSOLE = "http://127.0.0.1:8095"
J1_CAP_BELOW_TABLE = 95.0  # 2026-11-27 view check: the .77 J1 cap reads ~95 mm under the table top in depth


def _get(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=5).read()


def cam_points(depth_png: bytes, stride: int = 2) -> np.ndarray:
    """Depth pixels -> points (N, 3) in the colour-camera frame (mm)."""
    from PIL import Image
    cam = locate.CAM
    d = np.asarray(Image.open(io.BytesIO(depth_png)))[::stride, ::stride].astype(np.float64)
    vv, uu = np.nonzero(d > 0)
    z = d[vv, uu]
    uu, vv = uu * stride, vv * stride
    D = cam.depth_K
    X = np.stack([(uu - D["ppx"]) / D["fx"] * z, (vv - D["ppy"]) / D["fy"] * z, z], 1)
    return X @ np.asarray(cam.R_dc).T + np.asarray(cam.t_dc)


def fit_plane(Xc: np.ndarray, iters: int = 400, tol: float = 6.0, max_tilt_deg: float = 30.0):
    """The largest plane facing the camera (RANSAC, then least squares on its inliers).
    Returns n (unit, pointing from the table toward the camera), h (camera centre height above it), rms, inliers."""
    rng = np.random.default_rng(0)
    P = Xc[rng.choice(len(Xc), min(len(Xc), 20000), replace=False)]
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if math.degrees(math.acos(min(1.0, abs(n[2])))) > max_tilt_deg:
            continue
        k = int((np.abs((P - a) @ n) < tol).sum())
        if k > best_n:
            best, best_n = (a, n), k
    a, n = best
    inl = np.abs((Xc - a) @ n) < tol
    p0 = Xc[inl].mean(0)
    n = np.linalg.svd(Xc[inl] - p0, full_matrices=False)[2][2]
    if n @ (-p0) < 0:  # toward the camera centre (origin of the camera frame)
        n = -n
    rms = float(np.sqrt(np.mean(((Xc[inl] - p0) @ n) ** 2)))
    return n, float(n @ (-p0)), rms, int(inl.sum())


def stored_view():
    """Stored pose: world z in the camera frame, and the camera height above plan77.TABLE."""
    R, t = np.asarray(locate.CAM.R), np.asarray(locate.CAM.t)
    c = -R.T @ t
    return R[:, 2], float(c[2] - plan77.TABLE), c


def new_pose(n: np.ndarray, h: float):
    R_old, t_old = np.asarray(locate.CAM.R), np.asarray(locate.CAM.t)
    x = R_old[:, 0] - (R_old[:, 0] @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.column_stack([x, y, n])  # world -> camera: its columns are the world axes in camera coordinates
    c_old = -R_old.T @ t_old
    c = np.array([c_old[0], c_old[1], plan77.TABLE + h])
    return R, -R @ c


def cmd_plane(a) -> int:
    Xc = cam_points(_depth(a))
    n, h, rms, k = fit_plane(Xc)
    z_old, h_old, _ = stored_view()
    tilt = math.degrees(math.acos(min(1.0, float(n @ z_old))))
    vert = math.degrees(math.acos(min(1.0, float(-n[2]))))
    print(f"테이블 평면: 기울기 차이 {tilt:.2f}° (저장된 자세 대비), 수직에서 {vert:.2f}°, 높이 {h:.1f} mm "
          f"(저장 {h_old:.1f}, 차이 {h - h_old:+.1f}), 잔차 rms {rms:.1f} mm, {k} 점")
    return 0


def cmd_pose(a) -> int:
    Xc = cam_points(_depth(a))
    n, h, rms, k = fit_plane(Xc)
    R, t = new_pose(n, h)
    W = (Xc - t) @ R
    near = np.abs(W[:, 2] - plan77.TABLE) < 6.0
    msg = (f"새 위 카메라 자세: 높이 {h:.1f} mm, 평면 잔차 {rms:.1f} mm ({k} 점); 새 자세로 본 테이블 점 {near.sum()}개의 "
           f"z 중앙값 {np.median(W[near, 2]):.1f} (목표 {plan77.TABLE:.0f})")
    pose = {"R": R.round(8).tolist(), "t": t.round(3).tolist(), "height_mm": round(h, 1), "plane_rms_mm": round(rms, 2),
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "_src": "recam.py pose: table plane from the top depth; in-plane yaw/xy kept from the old pose (arbitrary)"}
    if a.write:
        stamp = time.strftime("%Y%m%d-%H%M")
        p = locate.LAB / "topcam.json"
        if p.exists():
            p.rename(p.with_name(f"topcam.before-recam-{stamp}.json"))
        p.write_text(json.dumps(pose, indent=1))
        pw = locate.LAB / "workspace.json"
        if pw.exists():
            cfg = json.loads(pw.read_text())
            top = cfg.get("cameras", {}).get("top")
            if top is not None:
                top["R"], top["t"] = pose["R"], pose["t"]
                top["_cal"] = f"recam.py pose {pose['time']} (same as topcam.json)"
                pw.rename(pw.with_name(f"workspace.before-recam-{stamp}.json"))
                pw.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
        msg += f" -> {p} + workspace.json cameras.top; cal77.json is now stale: run `recam.py arm77` next"
    print(msg)
    print(json.dumps({k_: v for k_, v in pose.items() if k_ in ("R", "t")}))
    return 0


def _depth(a) -> bytes:
    return Path(a.depth).read_bytes() if a.depth else _get(DEPTH_URL)


def _live_q():
    try:
        q = json.loads(_get(CONSOLE + "/api/state"))["arms"]["77"]["q"]
        return None if q is None else np.array(q, float)
    except Exception:  # noqa: BLE001
        return None


def model_map(q, cell: float):
    """Top-view height map of the .77 spheres at q, in the .77 frame (table-referenced z): cell centres (M, 2), tops (M,)."""
    import spheres
    S = spheres.spheres(np.asarray(q, float)).copy()
    S[:, 2] -= kin77.BASE_DZ
    lo, hi = (S[:, :2] - S[:, 3:4]).min(0), (S[:, :2] + S[:, 3:4]).max(0)
    gx = np.arange(lo[0], hi[0] + cell, cell)
    gy = np.arange(lo[1], hi[1] + cell, cell)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    top = np.full(len(G), -1e9)
    for s in S:
        d2 = ((G - s[:2]) ** 2).sum(1)
        m = d2 < s[3] ** 2
        top[m] = np.maximum(top[m], s[2] + np.sqrt(s[3] ** 2 - d2[m]))
    keep = top > plan77.TABLE - 150.0  # lower parts hide behind the table edge from most of the image
    return G[keep], top[keep], S


def obs_map(W: np.ndarray, centre, radius: float, cell: float):
    """Observed height map (max z per cell) around `centre`: grid origin, the map (NaN = no depth)."""
    m = np.hypot(W[:, 0] - centre[0], W[:, 1] - centre[1]) < radius
    P = W[m]
    o = np.asarray(centre, float) - radius
    n = int(math.ceil(2 * radius / cell)) + 1
    ij = np.floor((P[:, :2] - o) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    H = np.full((n, n), -np.inf)
    np.maximum.at(H, (ij[ok, 0], ij[ok, 1]), P[ok, 2])
    H[np.isinf(H)] = np.nan
    return o, H


def pose_costs(G, top, o, H, cell, yaws, offs, clip=40.0, nan_cost=25.0):
    """Mean clipped |observed - model| height over the model cells for every (yaw, offset). Returns (len(yaws), len(offs))."""
    n = H.shape[0]
    out = np.empty((len(yaws), len(offs)))
    for i, th in enumerate(yaws):
        c, s = math.cos(th), math.sin(th)
        Gw = G @ np.array([[c, s], [-s, c]])  # rows: Rz(th) @ g
        ij = np.floor((Gw[None, :, :] + offs[:, None, :] - o) / cell).astype(int)
        ok = (ij >= 0).all(2) & (ij < n).all(2)
        ijc = np.clip(ij, 0, n - 1)
        h = H[ijc[..., 0], ijc[..., 1]]
        h[~ok] = np.nan
        d = np.abs(h - top[None, :])
        d = np.where(np.isnan(d), nan_cost, np.minimum(d, clip))
        out[i] = d.mean(1)
    return out


def guess_base(W):
    """Clusters of depth points 30-200 mm under the table top (the .77 base and lower arm stand on the lower desk)."""
    t = plan77.TABLE
    P = W[(W[:, 2] < t - 30.0) & (W[:, 2] > t - 200.0)]
    if not len(P):
        return []
    ij = np.floor(P[:, :2] / 20.0).astype(int)
    keys, cnt = np.unique(ij, axis=0, return_counts=True)
    occ = {tuple(k): c for k, c in zip(keys, cnt) if c >= 3}
    seen, out = set(), []
    for k in occ:
        if k in seen:
            continue
        st, comp = [k], []
        seen.add(k)
        while st:
            c = st.pop()
            comp.append(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nb = (c[0] + d[0], c[1] + d[1])
                if nb in occ and nb not in seen:
                    seen.add(nb)
                    st.append(nb)
        w = np.array([occ[c] for c in comp], float)
        xy = (np.array(comp, float) + 0.5) * 20.0
        out.append(((w[:, None] * xy).sum(0) / w.sum(), int(w.sum())))
    return sorted(out, key=lambda r: -r[1])


def cmd_arm77(a) -> int:
    q = np.array(a.q, float) if a.q else _live_q()
    if q is None:
        print(".77 관절값을 콘솔에서 읽지 못했어요: --q 로 주세요 (팔 서버가 켜져 있어야 live q 가 있어요)")
        return 1
    W = (cam_points(_depth(a)) - np.asarray(locate.CAM.t)) @ np.asarray(locate.CAM.R)
    if a.near:
        c0 = np.array(a.near, float)
    elif a.near_px:
        c0 = locate.at(a.near_px[0], a.near_px[1], plan77.TABLE - J1_CAP_BELOW_TABLE)[:2]
    else:
        cands = guess_base(W)
        for (xy, n) in cands[:4]:
            print(f"  테이블 아래 덩어리: ({xy[0]:.0f}, {xy[1]:.0f}), {n} 점")
        if not cands:
            print("테이블 아래 30-200 mm 에 점이 없어요: --near 또는 --near-px 로 .77 축 근처를 주세요")
            return 1
        c0 = cands[0][0]
    cell = 5.0
    G, top, S = model_map(q, cell)
    reach = float(np.hypot(G[:, 0], G[:, 1]).max())
    o, H = obs_map(W, c0, reach + a.search + 40.0, cell)
    t0 = time.perf_counter()
    yaws = np.radians(np.arange(0.0, 360.0, 3.0))
    g = np.arange(-a.search, a.search + 0.1, 10.0)
    offs = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) + c0
    C = pose_costs(G, top, o, H, cell, yaws, offs)
    i, j = np.unravel_index(np.argmin(C), C.shape)
    th0, c1 = yaws[i], offs[j]
    # the best pose at a clearly different yaw (> 25 deg away): how unambiguous the silhouette is
    dyaw = np.abs((np.degrees(yaws - th0) + 180.0) % 360.0 - 180.0)
    alt = C[dyaw > 25.0].min()
    ya = np.radians(np.degrees(th0) + np.arange(-3.0, 3.01, 0.5))
    gf = np.arange(-10.0, 10.1, 2.0)
    of = np.stack(np.meshgrid(gf, gf, indexing="ij"), -1).reshape(-1, 2) + c1
    Cf = pose_costs(G, top, o, H, cell, ya, of)
    i, j = np.unravel_index(np.argmin(Cf), Cf.shape)
    th, c = float(ya[i]), of[j]
    best = float(Cf[i, j])
    # fraction of the model cells whose observed top is within 15 mm
    cs, sn = math.cos(th), math.sin(th)
    Gw = G @ np.array([[cs, sn], [-sn, cs]]) + c
    ij = np.floor((Gw - o) / cell).astype(int)
    okk = (ij >= 0).all(1) & (ij < H.shape[0]).all(1)
    h = np.full(len(G), np.nan)
    h[okk] = H[ij[okk, 0], ij[okk, 1]]
    match = float(np.mean(np.abs(h - top) < 15.0))
    seen = float(np.mean(~np.isnan(h)))
    dt = time.perf_counter() - t0
    R2 = np.array([[cs, sn], [-sn, cs]])  # Rz(-th): world -> .77
    yaw_w = (math.degrees(th) + 180.0) % 360.0 - 180.0  # the .77 x axis in the world, (-180, 180]
    T = -R2 @ c
    msg = (f".77 위치(새 카메라 world): 축 ({c[0]:.1f}, {c[1]:.1f}), yaw {yaw_w:+.1f}°; 높이 차 평균 {best:.1f} mm "
           f"(다른 yaw 최선 {alt:.1f}, 차이 {alt - best:.1f}), 15 mm 안에 맞은 칸 {match:.0%}, 깊이 있는 칸 {seen:.0%}, "
           f"시작점 ({c0[0]:.0f}, {c0[1]:.0f}), q {np.round(q, 1).tolist()}, {dt:.1f} s")
    clear = best < 20.0 and alt - best > 4.0 and match > 0.3 and seen > 0.4  # real shell vs spheres: looser than synthetic
    if a.png:
        _draw(a.png, S, th, c)
        msg += f"; 그림 {a.png}"
    if not clear:
        print(msg + ": 맞춤이 분명하지 않아 쓰지 않아요 (그림을 보고 --near 로 다시)")
        return 1
    out = {"yaw_deg": round(-yaw_w, 4), "R2": R2.round(8).tolist(), "T": T.round(3).tolist(),
           "time": time.strftime("%Y-%m-%d %H:%M"), "provisional": True,
           "fit": {"mean_dz_mm": round(best, 2), "alt_yaw_mm": round(float(alt), 2), "match": round(match, 3)},
           "_src": "recam.py arm77: .77 sphere model at q vs the top depth height map (no motion); refine with fit2d"}
    if a.write:
        import cal77
        msg += cal77.save_map(out, R2, T, "recam.py arm77")
    print(msg)
    print(json.dumps({k: out[k] for k in ("yaw_deg", "R2", "T")}))
    return 0


def _draw(path, S, th, c):
    """The fitted sphere model drawn on the live top colour image (sphere tops, real size at their height)."""
    from PIL import Image, ImageDraw
    img = Image.open(io.BytesIO(_get(COLOR_URL))).convert("RGB")
    dr = ImageDraw.Draw(img)
    cs, sn = math.cos(th), math.sin(th)
    K = locate.CAM.K
    for s in S:
        w = np.array([cs * s[0] - sn * s[1] + c[0], sn * s[0] + cs * s[1] + c[1], s[2] + s[3]])
        u, v = locate.CAM.world_to_pixel(w)
        zc = (np.asarray(locate.CAM.R) @ w + np.asarray(locate.CAM.t))[2]
        r = max(1.5, K["fx"] * s[3] / max(zc, 1.0))
        dr.ellipse([u - r, v - r, u + r, v + r], outline=(0, 200, 200))
    img.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plane", "pose", "arm77"):
        p = sp.add_parser(name)
        p.add_argument("--depth", help="a saved top depth PNG instead of the live viewer")
        if name != "plane":
            p.add_argument("--write", action="store_true")
    p = sp.choices["arm77"]
    p.add_argument("--q", nargs=6, type=float)
    p.add_argument("--near", nargs=2, type=float, help="the .77 axis near (x, y) in the camera world (mm)")
    p.add_argument("--near-px", nargs=2, type=float, help="... or near the J1 cap at this colour pixel")
    p.add_argument("--search", type=float, default=80.0, help="xy search radius around the start (mm)")
    p.add_argument("--png", help="draw the fitted model on the live colour image")
    a = ap.parse_args()
    return {"plane": cmd_plane, "pose": cmd_pose, "arm77": cmd_arm77}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
