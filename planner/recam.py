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
  recam.py arm77 [--q Q6] [--near X Y | --near-px U V | --near-depth-px U V] [--depth F] [--png OUT] [--write]
                                       where the .77 stands in that world, from its own depth silhouette: the cuRobo
                                       spheres at joints q (default: the console's live q) rendered as a top-view
                                       height map, matched to the observed height map over x, y and yaw.
                                       --write: cal77.json + workspace.json arms.77 (cal77.save_map), if the fit is clear

  recam.py arm70 [same options] [--bz LO HI]
                                       the .70 (spoon arm) the same way, its spoon included, also searching its base
                                       height. --write: workspace.json arms.70 origin_mm/yaw_deg (s70/arm70n.py reads
                                       it). Needs a good part of the .70 inside the top depth view.

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


def _live_q(arm: str):
    try:
        q = json.loads(_get(CONSOLE + "/api/state"))["arms"][arm]["q"]
        return None if q is None else np.array(q, float)
    except Exception:  # noqa: BLE001
        return None


def model_map(S, cell: float, zmin=None):
    """Top-view height map of spheres S (N, 4) in their arm frame: cell centres (M, 2), tops (M,). zmin drops lower cells."""
    lo, hi = (S[:, :2] - S[:, 3:4]).min(0), (S[:, :2] + S[:, 3:4]).max(0)
    gx = np.arange(lo[0], hi[0] + cell, cell)
    gy = np.arange(lo[1], hi[1] + cell, cell)
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    top = np.full(len(G), -1e9)
    for s in S:
        d2 = ((G - s[:2]) ** 2).sum(1)
        m = d2 < s[3] ** 2
        top[m] = np.maximum(top[m], s[2] + np.sqrt(s[3] ** 2 - d2[m]))
    keep = top > (-1e8 if zmin is None else zmin)
    return G[keep], top[keep]


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


def fit_pose(S, W, c0, search, dzs=(0.0,), zmin=None, yaw_step=3.0):
    """Place spheres S (arm frame) in the world as Rz(th) p + (c, dz) so their top-view height map matches the depth.
    Coarse on 10 mm cells (xy step 10, yaw `yaw_step`, every dz), then fine on 5 mm cells around the best."""
    t0 = time.perf_counter()
    reach = float(np.hypot(S[:, 0], S[:, 1]).max() + S[:, 3].max())
    dzs = np.asarray(dzs, float)
    res = {}
    for cell, stage in ((10.0, "coarse"), (5.0, "fine")):
        G, top = model_map(S, cell, zmin)
        o, H = obs_map(W, c0, reach + search + 40.0, cell)
        if stage == "coarse":
            yaws = np.radians(np.arange(0.0, 360.0, yaw_step))
            g = np.arange(-search, search + 0.1, 10.0)
            offs = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) + c0
            dz_set = dzs
        else:
            yaws = np.radians(np.degrees(res["th"]) + np.arange(-yaw_step, yaw_step + 0.01, 0.5))
            g = np.arange(-10.0, 10.1, 2.0)
            offs = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) + res["c"]
            dz_set = [res["dz"]] if len(dzs) == 1 else res["dz"] + np.arange(-6.0, 6.1, 2.0)
        best = None
        for dz in dz_set:
            C = pose_costs(G, top + dz, o, H, cell, yaws, offs)
            i, j = np.unravel_index(np.argmin(C), C.shape)
            if best is None or C[i, j] < best[0]:
                best = (float(C[i, j]), float(yaws[i]), offs[j].copy(), float(dz), C, yaws)
        cost, th, c, dz, C, ys = best
        if stage == "coarse":  # the best pose at a clearly different yaw (> 25 deg away): how unambiguous it is
            dyaw = np.abs((np.degrees(ys - th) + 180.0) % 360.0 - 180.0)
            res["alt"] = float(C[dyaw > 25.0].min())
        res.update(th=th, c=c, dz=dz, best=cost)
    # how many model cells have depth at all, and how many of those are within 15 mm of the model top
    cs, sn = math.cos(res["th"]), math.sin(res["th"])
    Gw = G @ np.array([[cs, sn], [-sn, cs]]) + res["c"]
    ij = np.floor((Gw - o) / cell).astype(int)
    okk = (ij >= 0).all(1) & (ij < H.shape[0]).all(1)
    h = np.full(len(G), np.nan)
    h[okk] = H[ij[okk, 0], ij[okk, 1]]
    seen = ~np.isnan(h)
    res["seen"] = float(seen.mean())
    res["match"] = float(np.mean(np.abs(h[seen] - top[seen] - res["dz"]) < 15.0)) if seen.any() else 0.0
    res["yaw_w"] = (math.degrees(res["th"]) + 180.0) % 360.0 - 180.0
    res["dt"] = time.perf_counter() - t0
    return res


def _world_pts(a):
    return (cam_points(_depth(a)) - np.asarray(locate.CAM.t)) @ np.asarray(locate.CAM.R)


def _start(a, W, z_guess):
    if a.near:
        return np.array(a.near, float)
    if a.near_px:
        return locate.at(a.near_px[0], a.near_px[1], z_guess)[:2]
    if a.near_depth_px:
        from PIL import Image
        d = np.asarray(Image.open(io.BytesIO(_depth(a)))).astype(float)
        u, v = int(a.near_depth_px[0]), int(a.near_depth_px[1])
        win = d[max(0, v - 3):v + 4, max(0, u - 3):u + 4]
        z = float(np.median(win[win > 0]))
        D = locate.CAM.depth_K
        X = np.array([(u - D["ppx"]) / D["fx"] * z, (v - D["ppy"]) / D["fy"] * z, z])
        Xc = np.asarray(locate.CAM.R_dc) @ X + np.asarray(locate.CAM.t_dc)
        return (np.asarray(locate.CAM.R).T @ (Xc - np.asarray(locate.CAM.t)))[:2]
    return None


def _report(name, r, c0, q, extra=""):
    return (f"{name} 위치(새 카메라 world): 축 ({r['c'][0]:.1f}, {r['c'][1]:.1f}){extra}, yaw {r['yaw_w']:+.1f}°; "
            f"높이 차 평균 {r['best']:.1f} mm (다른 yaw 최선 {r['alt']:.1f}, 차이 {r['alt'] - r['best']:.1f}), "
            f"깊이 있는 칸 {r['seen']:.0%} 중 15 mm 안 {r['match']:.0%}, 시작점 ({c0[0]:.0f}, {c0[1]:.0f}), "
            f"q {np.round(q, 1).tolist()}, {r['dt']:.1f} s")


def cmd_arm77(a) -> int:
    import spheres
    q = np.array(a.q, float) if a.q else _live_q("77")
    if q is None:
        print(".77 관절값을 콘솔에서 읽지 못했어요: --q 로 주세요 (팔 서버가 켜져 있어야 live q 가 있어요)")
        return 1
    W = _world_pts(a)
    c0 = _start(a, W, plan77.TABLE - J1_CAP_BELOW_TABLE)
    if c0 is None:
        cands = guess_base(W)
        for (xy, n) in cands[:4]:
            print(f"  테이블 아래 덩어리: ({xy[0]:.0f}, {xy[1]:.0f}), {n} 점")
        if not cands:
            print("테이블 아래 30-200 mm 에 점이 없어요: --near 또는 --near-px 로 .77 축 근처를 주세요")
            return 1
        c0 = cands[0][0]
    S = spheres.spheres(q).copy()
    S[:, 2] -= kin77.BASE_DZ  # table-referenced, = camera world z
    r = fit_pose(S, W, c0, a.search, zmin=plan77.TABLE - 150.0)  # lower parts hide behind the table edge
    th, c = r["th"], r["c"]
    cs, sn = math.cos(th), math.sin(th)
    R2 = np.array([[cs, sn], [-sn, cs]])  # Rz(-th): world -> .77
    T = -R2 @ c
    msg = _report(".77", r, c0, q)
    # the absolute height error depends on occlusion and on the sphere model vs the real shell (synthetic: 14 mm for
    # the right pose); the gap to the best other yaw and the matched share are what tell a clear fit
    clear = r["best"] < 30.0 and r["alt"] - r["best"] > 4.0 and r["seen"] > 0.4 and r["match"] > 0.5
    if a.png:
        _draw(a.png, S, th, c)
        msg += f"; 그림 {a.png}"
    if not clear:
        print(msg + ": 맞춤이 분명하지 않아 쓰지 않아요 (그림을 보고 --near 로 다시)")
        return 1
    out = {"yaw_deg": round(-r["yaw_w"], 4), "R2": R2.round(8).tolist(), "T": T.round(3).tolist(),
           "time": time.strftime("%Y-%m-%d %H:%M"), "provisional": True,
           "fit": {k: round(float(r[k]), 3) for k in ("best", "alt", "seen", "match")},
           "_src": "recam.py arm77: .77 sphere model at q vs the top depth height map (no motion); refine with fit2d"}
    if a.write:
        import cal77
        msg += cal77.save_map(out, R2, T, "recam.py arm77")
    print(msg)
    print(json.dumps({k: out[k] for k in ("yaw_deg", "R2", "T")}))
    return 0


def cmd_arm70(a) -> int:
    """The .70 (spoon arm): same fit, plus its base height (BZ, unknown in a new room). Writes workspace.json arms.70
    (origin_mm = [BX, BY, BZ], yaw_deg = YAW), which s70/arm70n.py reads."""
    import kin
    import spheres
    q = np.array(a.q, float) if a.q else _live_q("70")
    if q is None:
        print(".70 관절값을 콘솔에서 읽지 못했어요: --q 로 주세요")
        return 1
    W = _world_pts(a)
    c0 = _start(a, W, plan77.TABLE)
    if c0 is None:
        print(".70 축 근처를 --near, --near-px 또는 --near-depth-px 로 주세요")
        return 1
    ws = json.loads(locate.lab_file("workspace.json").read_text())
    tool = np.array(ws["arms"]["70"].get("tool_mm", [-70.8, -98.7, 21.5]), float)
    T = kin.fk(q)  # the .70's own base frame (no table reference)
    spoon = [np.r_[T[:3, :3] @ (tool * f) + T[:3, 3], 12.0] for f in np.linspace(0.15, 1.0, 7)]
    S = np.vstack([spheres.spheres(q), spoon])
    r = fit_pose(S, W, c0, a.search, dzs=np.arange(a.bz[0], a.bz[1] + 0.1, 10.0), yaw_step=4.0)
    msg = _report(".70", r, c0, q, extra=f", 받침 z {r['dz']:.0f}")
    clear = r["best"] < 30.0 and r["alt"] - r["best"] > 4.0 and r["seen"] > 0.25 and r["match"] > 0.5
    if a.png:
        S2 = S.copy()
        S2[:, 2] += r["dz"]
        _draw(a.png, S2, r["th"], r["c"])
        msg += f"; 그림 {a.png}"
    if not clear:
        print(msg + ": 맞춤이 분명하지 않아 쓰지 않아요 (.70이 화면에 더 들어와야 할 수 있어요)")
        return 1
    if a.write:
        pw = locate.LAB / "workspace.json"
        cfg = json.loads(pw.read_text())
        a70 = cfg["arms"]["70"]
        old = (a70.get("origin_mm"), a70.get("yaw_deg"))
        a70["origin_mm"] = [round(float(r["c"][0]), 1), round(float(r["c"][1]), 1), round(float(r["dz"]), 1)]
        a70["yaw_deg"] = round(r["yaw_w"], 2)
        a70["_cal"] = (f"recam.py arm70 {time.strftime('%Y-%m-%d %H:%M')}: depth silhouette, provisional "
                       f"(before: {old[0]}, {old[1]})")
        pw.rename(pw.with_name(f"workspace.before-arm70-{time.strftime('%Y%m%d-%H%M')}.json"))
        pw.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
        msg += f" -> {pw} arms.70 (restart the console)"
    print(msg)
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
    for name in ("plane", "pose", "arm77", "arm70"):
        p = sp.add_parser(name)
        p.add_argument("--depth", help="a saved top depth PNG instead of the live viewer")
        if name != "plane":
            p.add_argument("--write", action="store_true")
        if name.startswith("arm"):
            p.add_argument("--q", nargs=6, type=float)
            p.add_argument("--near", nargs=2, type=float, help="the arm's axis near (x, y) in the camera world (mm)")
            p.add_argument("--near-px", nargs=2, type=float, help="... or near this colour pixel (J1 cap height)")
            p.add_argument("--near-depth-px", nargs=2, type=float, help="... or near the point seen at this depth pixel")
            p.add_argument("--search", type=float, default=80.0, help="xy search radius around the start (mm)")
            p.add_argument("--png", help="draw the fitted model on the live colour image")
    sp.choices["arm70"].add_argument("--bz", nargs=2, type=float, default=[-100.0, 60.0],
                                     help="range of the .70 base height (camera world z, table at 197)")
    a = ap.parse_args()
    return {"plane": cmd_plane, "pose": cmd_pose, "arm77": cmd_arm77, "arm70": cmd_arm70}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
