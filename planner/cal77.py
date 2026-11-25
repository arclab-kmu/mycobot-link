"""Top camera <-> .77 calibration from gripper open/close differences (no marker, nothing touched).

plan   : print a console scenario: the empty hand hovers at ~12 spots over the table (two heights); at each one
         the gripper opens, a top photo is taken, it closes to 40, another photo, it opens again.
snap   : save one top photo (called by the scenario while the arm holds).
fit    : centroid of the changed pixels in each open/closed pair <-> FK of a point in the flange frame;
         fits the camera pose in the .77 base frame (6) and that flange point (3) by Gauss-Newton on pixels.
         Writes /home/hkim/Documents/lab-jetson/console/cal77_cam.json, used by locate.py from then on.
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
import kin  # noqa: E402
import locate  # noqa: E402
import plan77  # noqa: E402

OUT = locate.LAB / "cal77_cam.json"
SNAPS = locate.LAB / "cal77_snaps"
GRIP_CLOSED = 40


def _initial_cam():
    """.77 -> camera from the old chain (cal77 2-D similarity + the calib_top3 camera)."""
    cal = locate.cal77()
    R2, T2 = np.array(cal["R2"]), np.array(cal["T"])
    A = np.eye(3)
    A[:2, :2] = R2.T  # world_xy = R2^T (p77_xy - T2)
    b = np.r_[-R2.T @ T2, 0.0]
    R, t = np.array(locate.CAM.R), np.array(locate.CAM.t)
    return R @ A, R @ b + t


def project(Rc, tc, P):
    K = locate.CAM.K
    c = (Rc @ np.atleast_2d(P).T).T + tc
    return np.column_stack([K["fx"] * c[:, 0] / c[:, 2] + K["ppx"], K["fy"] * c[:, 1] / c[:, 2] + K["ppy"]])


def hover_pose(xy, z_pad, scene, api, obs):
    """Template grasp turned to the bearing of xy, moved so the pad sits at (xy, z_pad)."""
    G = plan77.TPL["grasp"]
    V0 = np.array(G["vial_xy"])
    dth = plan77.bearing(xy) - plan77.bearing(V0)
    qr = np.array(G["q"], float)
    qr[0] += dth
    T = kin.fk(qr)
    pad = T[:3, :3] @ plan77.TLV + T[:3, 3]
    T[:3, 3] += np.r_[np.asarray(xy) - pad[:2], z_pad - pad[2]]
    q, e = plan77.ik_walk(qr, T)
    if e > 0.3 or plan77.limit_margin(q) < plan77.LIM_MARGIN:
        return None
    if scene.grip_clear(q) < 5:
        return None
    if not plan77.cu_check(api, np.array([q]), obs).get("ok"):
        return None
    return q


def cmd_plan(args):
    from mycobot_brain.console import workspace

    ws = workspace.load(str(locate.LAB / "workspace.json"))
    api = ws.curobo_api
    # keep well clear of the cup and the vials: their positions are only as good as the map being fixed here
    import matcha_dyn
    _, d = matcha_dyn.measure()
    cup = locate.to77(locate.at(d["cup"]["u"], d["cup"]["v"], locate.RIM_Z)).tolist()
    caps = ([d["matcha"]] if d["matcha"] else []) + d["others"]
    others = [locate.to77(locate.at(c["u"], c["v"], locate.CAP_Z)).tolist() for c in caps]
    scene = plan77.Scene(cup, 286.0 + 25.0, 40.0 + 25.0, [[o[0], o[1], 262.5 + 25.0] for o in others])
    obs = (ws.obstacles_for(ws.arms["77"]) + plan77.cup_boxes(cup, 286.0 + 25.0, 40.0 + 25.0) +
           [plan77.box(f"vial{i}", o[0], o[1], 197.0, 300.0, 80, 80) for i, o in enumerate(others)])[:15]
    spots = []
    grid = [(x, y) for y in range(-220, 61, 40) for x in (range(100, 241, 35) if (y // 40) % 2 else range(240, 99, -35))]
    for k, (x, y) in enumerate(grid):
        heights = (300.0, 330.0, 360.0) if k % 2 == 0 else (360.0, 330.0, 300.0)
        for z in heights:
            q = hover_pose([x, y], z, scene, api, obs)
            if q is not None:
                spots.append((x, y, z, q))
                break
        if len(spots) >= 14:
            break
    if len(spots) < 6:
        raise SystemExit(f"only {len(spots)} reachable hover spots")
    import matcha_dyn
    q_now = matcha_dyn.arm_q("77")
    seq = [plan77.PARK if q_now is None else q_now] + [s[3] for s in spots] + [plan77.PARK]
    segs = []
    py = sys.executable
    me = str(Path(__file__).resolve())
    for i in range(1, len(seq)):
        path = plan77.thin(plan77.cu_plan(api, seq[i - 1], seq[i], obs))
        segs.append({"kind": "joints", "label": f"자리 {i}" if i < len(seq) - 1 else "park",
                     "q": np.round(plan77.timed([(path, 45.0)]), 3).tolist(), "dt": 0.05})
        if i < len(seq) - 1:
            segs += [{"kind": "grip", "label": "열기", "value": 100},
                     {"kind": "wait", "label": "멈춤", "seconds": 0.6},
                     {"kind": "check", "label": "사진(열림)", "cmd": [py, me, "snap", f"{i:02d}_open"]},
                     {"kind": "grip", "label": "닫기", "value": GRIP_CLOSED},
                     {"kind": "wait", "label": "멈춤", "seconds": 0.6},
                     {"kind": "check", "label": "사진(닫힘)", "cmd": [py, me, "snap", f"{i:02d}_closed"]}]
    segs.append({"kind": "grip", "label": "열기", "value": 100})
    segs.append({"kind": "check", "label": "보정 계산", "cmd": [py, me, "fit"], "timeout_s": 60})
    meta = [{"i": i + 1, "xy": [s[0], s[1]], "z_pad": s[2], "q": np.round(s[3], 3).tolist()} for i, s in enumerate(spots)]
    SNAPS.mkdir(parents=True, exist_ok=True)
    for f in SNAPS.glob("*.jpg"):
        f.unlink()
    (SNAPS / "spots.json").write_text(json.dumps(meta))
    scen = {"trigger": "카메라 보정", "title": "위 카메라 ↔ .77 보정",
            "say": f".77이 빈손으로 {len(spots)}곳에서 그리퍼를 열고 닫아, 위 카메라 사진 차이로 손가락 위치를 찾고 카메라 위치를 다시 맞춥니다.",
            "seen": "병과 컵은 건드리지 않습니다.", "phases": [{"arm": "77", "label": "카메라 보정", "speed_dps": 80,
                                                          "obstacles": obs[len(ws.obstacles_for(ws.arms['77'])):], "segments": segs}]}
    print(json.dumps(scen, ensure_ascii=False))


def cmd_snap(args):
    SNAPS.mkdir(parents=True, exist_ok=True)
    time.sleep(0.2)
    err = None
    for _ in range(4):
        try:
            data = urllib.request.urlopen("http://127.0.0.1:8090/color.jpg", timeout=5).read()
            (SNAPS / f"{args.name}.jpg").write_bytes(data)
            print(f"saved {args.name}")
            return
        except Exception as exc:  # noqa: BLE001 - the viewer can be slow while the page records
            err = exc
            time.sleep(0.5)
    print(f"사진 저장 실패 ({args.name}): {err}")
    sys.exit(1)


def _centroids():
    from PIL import Image
    meta = json.loads((SNAPS / "spots.json").read_text())
    Rc, tc = _initial_cam()
    out = []
    for m in meta:
        fo, fc = SNAPS / f"{m['i']:02d}_open.jpg", SNAPS / f"{m['i']:02d}_closed.jpg"
        if not (fo.exists() and fc.exists()):
            continue
        a = np.asarray(Image.open(fo).convert("L"), float)
        b = np.asarray(Image.open(fc).convert("L"), float)
        d = np.abs(a - b)
        T = kin.fk(np.array(m["q"]))
        guess = project(Rc, tc, T[:3, :3] @ plan77.TLV + T[:3, 3])[0]
        vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        win = np.hypot(uu - guess[0], vv - guess[1]) < 120
        mask = (d > 35) & win
        if mask.sum() < 30:
            continue
        w = d[mask]
        u, v = float((uu[mask] * w).sum() / w.sum()), float((vv[mask] * w).sum() / w.sum())
        out.append(dict(i=m["i"], q=m["q"], px=[u, v], n=int(mask.sum()), guess=guess.round(1).tolist()))
    return out


def _rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def cmd_fit(args):
    obs = _centroids()
    if len(obs) < 6:
        print(f"보정 실패: 쓸 만한 열림/닫힘 차이가 {len(obs)}곳뿐이에요")
        sys.exit(1)
    R0, t0 = _initial_cam()
    Ts, px = [], None

    def resid(x):
        Rc = _rodrigues(x[:3]) @ R0
        tc = t0 + x[3:6]
        P = np.array([T[:3, :3] @ (plan77.TLV + x[6:9]) + T[:3, 3] for T in Ts])
        return (project(Rc, tc, P) - px).ravel()

    obs = [o for o in obs if o["n"] >= 1500]  # a hand at the image edge shows only part of the fingers
    Ts = [kin.fk(np.array(o["q"])) for o in obs]
    px = np.array([o["px"] for o in obs])

    def full(x):  # pixel residuals + a weak prior that the finger centre is near the pad point (1 px per 5 mm)
        return np.r_[resid(x), x[6:9] / 5.0]

    x = np.zeros(9)
    lam = 1e-2
    for _ in range(200):
        r = full(x)
        J = np.zeros((len(r), 9))
        for j in range(9):
            dx = np.zeros(9)
            dx[j] = 1e-5 if j < 3 else 1e-3
            J[:, j] = (full(x + dx) - r) / dx[j]
        A = J.T @ J
        g = J.T @ r
        step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
        if (full(x + step) ** 2).sum() < (r ** 2).sum():
            x, lam = x + step, lam * 0.3
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            lam *= 10
    r = resid(x).reshape(-1, 2)
    Rc, tc = _rodrigues(x[:3]) @ R0, t0 + x[3:6]
    rms = float(np.sqrt((r ** 2).sum(1).mean()))
    # what the old map said vs the new one, at the cap height, over the table
    probe = [(u, v) for u in (220, 300, 380, 460) for v in (100, 180, 260)]
    old = np.array([locate.to77(locate.at(u, v, 264.0)) for u, v in probe])
    tmp = {"R": Rc.tolist(), "t": tc.tolist()}
    new = np.array([ray77(tmp, u, v, 264.0)[:2] for u, v in probe])
    shift = np.linalg.norm(new - old, axis=1)
    OUT.write_text(json.dumps({"R77_to_cam": Rc.round(6).tolist(), "t_mm": tc.round(3).tolist(), "flange_point_mm": (plan77.TLV + x[6:9]).round(2).tolist(),
                               "rms_px": round(rms, 2), "n": len(obs), "time": time.strftime("%Y-%m-%d %H:%M"),
                               "residuals_px": np.round(r, 2).tolist()}, indent=1))
    print(f"보정 완료: {len(obs)}곳, 재투영 rms {rms:.2f} px, 이전 지도와 차이 평균 {shift.mean():.1f} mm (최대 {shift.max():.1f})")


def ray77(cal, u, v, z):
    """Pixel (u, v) -> point at height z in the .77 frame, with a camera pose given in the .77 frame."""
    K = locate.CAM.K
    R, t = np.array(cal["R"]), np.array(cal["t"])
    d = R.T @ np.array([(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0])
    c = -R.T @ t
    return c + d * (z - c[2]) / d[2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--scene", help="json with cup77 and vials77 (to keep the hover spots clear of them)")
    p = sp.add_parser("snap")
    p.add_argument("name")
    sp.add_parser("fit")
    a = ap.parse_args()
    {"plan": cmd_plan, "snap": cmd_snap, "fit": cmd_fit}[a.cmd](a)
