"""Top camera <-> .77 calibration from gripper open/close differences (no marker, nothing touched).

plan   : print a console scenario: the empty hand hovers at ~12 spots over the table (two heights); at each one
         the gripper opens, a top photo is taken, it closes to 40, another photo, it opens again.
snap   : save one top photo (called by the scenario while the arm holds).
fit    : centroid of the changed pixels in each open/closed pair <-> FK of a point in the flange frame;
         fits the camera pose in the .77 base frame (6) and that flange point (3) by Gauss-Newton on pixels.
         Writes $MYCOBOT_LAB/cal77_cam.json (not used by locate.py: the 2026-11-25 fit was rejected).
fit2d  : the map locate.py does use: camera world -> .77 as a 2-D rotation + translation ($MYCOBOT_LAB/cal77.json),
         from the same photos. The finger centre's offset in the hand (along / across the fingers) is fitted with it:
         it turns with the hand from spot to spot, the map does not. For a new room (2026-11-26 the exhibit setup moved
         off the lab LAN), before any grasp. --dry-run prints without writing.
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
import kin77 as kin  # noqa: E402 - table-referenced frame (kin77.BASE_DZ)
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
    T = kin.fk_taught(qr)  # taught template pose
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

    ws = workspace.load(str(locate.lab_file("workspace.json")))
    api = ws.curobo_api
    # keep well clear of the cup and the vials: their positions are only as good as the map being fixed here
    import matcha_dyn
    if args.clear:  # a new room: the old map cannot place the cup or vials, so the table must be empty
        cup, others = [2000.0, 2000.0], []
    else:
        _, d = matcha_dyn.measure()
        cup = locate.to77(locate.at(d["cup"]["u"], d["cup"]["v"], locate.RIM_Z)).tolist()
        caps = ([d["matcha"]] if d["matcha"] else []) + d["others"]
        others = [locate.to77(locate.at(c["u"], c["v"], locate.CAP_Z)).tolist() for c in caps]
    scene = plan77.Scene(cup, locate.RIM_Z + 25.0, 40.0 + 25.0, [[o[0], o[1], locate.CAP_Z - 1.5 + 25.0] for o in others])
    obs = ws.obstacles_for(ws.arms["77"])
    if not args.clear:
        obs = (obs + plan77.cup_boxes(cup, locate.RIM_Z + 25.0, 40.0 + 25.0) +
               [plan77.box(f"vial{i}", o[0], o[1], 197.0, 300.0, 80, 80) for i, o in enumerate(others)])[:plan77.MAX_OBS]
    if args.keepout:
        # Objects stay on the table but the map that would place them is what this run measures (2026-11-26 night,
        # Hojin: "지금 상태에서 진행해 봐 ... 병 넘어트리지마"). Distances from the .77 axis do not depend on the
        # unknown yaw, and every object stood >= ~135 mm from it (vials 145, cup 264, depth heights 67-83 mm). So the
        # arm is kept above table + H everywhere beyond an octagon of apothem R around its axis (4 boxes); inside it
        # (where the upper arm rises past the table edge) it may go lower.
        r_in, h = args.keepout
        for k, ang in enumerate((0, 45, 90, -45)):
            a = math.radians(ang)
            c = r_in + 350.0
            obs.append(plan77.box(f"keepout{k}", c * math.cos(a), c * math.sin(a), plan77.TABLE, plan77.TABLE + h,
                                  700.0, 1400.0, ang))
    if args.fence:
        # tall fixtures (camera gooseneck 180 mm, tripod ~140 mm above the table) stood >= 270 mm from the .77 axis
        # (AGX depth, 2026-11-26): a fence of the same octagon shape beyond R, up to table + H, all around
        r_f, h_f = args.fence
        for k, ang in enumerate((0, 45, 90, 135, 180, -135, -90, -45)):
            a = math.radians(ang)
            c = r_f + 350.0
            obs.append(plan77.box(f"fence{k}", c * math.cos(a), c * math.sin(a), plan77.TABLE - 30.0, plan77.TABLE + h_f,
                                  700.0, 1400.0, ang))
    import matcha_dyn
    q_now = matcha_dyn.arm_q("77")
    start = plan77.PARK if q_now is None else q_now
    if args.board_margin:
        # the board box sits on the depth edge mapped by a map this run is about to fix: move its near face toward the
        # .77 by M (its local +x points into the table, board77.py)
        for o in obs:
            if o["name"] == "board":
                yaw = 2.0 * math.atan2(o["pose"][6], o["pose"][3])
                o["dims_mm"][0] += args.board_margin
                o["pose"][0] -= math.cos(yaw) * args.board_margin / 2
                o["pose"][1] -= math.sin(yaw) * args.board_margin / 2
    if args.depth_obs is not None:
        # what stands near the arm, from the top depth at this moment, grown by PAD for the map error (2026-11-27:
        # the AGX cross-check found the black pole at r 241-278 inside the fence and desk items near the base)
        import depth77
        png = urllib.request.urlopen(depth77.DEPTH_URL, timeout=5).read()
        dep = depth77.boxes(png, q77=start, low=args.depth_low, pad=args.depth_obs,
                            max_boxes=max(0, plan77.MAX_OBS - len(obs)))
        print(f"cal77 plan: {len(dep)} depth boxes (pad {args.depth_obs:.0f} mm, from table {args.depth_low:+.0f})",
              file=sys.stderr)
        obs = obs + dep
    spots = []
    grid = [(x, y) for y in range(-220, 61, 40) for x in (range(100, 241, 35) if (y // 40) % 2 else range(240, 99, -35))]
    if args.ring:
        # A new room with fixtures the planner cannot see yet (2026-11-26: the top camera's gooseneck tip and the side
        # camera's tripod, both ~180-230 mm from the .77 base): hover only on rings close to the base, over a wide
        # spread of bearings (the spread is what pins the map's rotation).
        r_in, r_out = args.ring
        grid = [(r * math.cos(math.radians(b)), r * math.sin(math.radians(b)))
                for i, r in enumerate((r_in, (r_in + r_out) / 2, r_out))
                for b in (range(-70, 71, 20) if i % 2 == 0 else range(70, -71, -20))]
    if args.polar:  # radii x bearings (deg) of the .77 frame, nearest ring first
        b0, b1, db = args.bearings
        grid = []
        for i, r in enumerate(args.polar):
            bs = list(np.arange(b0, b1 + 0.1, db))
            grid += [(r * math.cos(math.radians(b)), r * math.sin(math.radians(b))) for b in (bs if i % 2 == 0 else bs[::-1])]
    if args.box:  # a rectangle of the .77 frame known to be over the table and away from fixtures
        x0, x1, y0, y1 = args.box
        xs, ys = np.linspace(x0, x1, 4), np.linspace(y0, y1, 4)
        grid = [(float(x), float(y)) for j, y in enumerate(ys) for x in (xs if j % 2 == 0 else xs[::-1])]
    for k, (x, y) in enumerate(grid):
        heights = (300.0, 330.0, 360.0) if k % 2 == 0 else (360.0, 330.0, 300.0)
        if args.heights:
            heights = tuple(args.heights) if k % 2 == 0 else tuple(args.heights[::-1])
        for z in heights:
            q = hover_pose([x, y], z, scene, api, obs)
            if q is not None:
                spots.append((x, y, z, q))
                break
        if len(spots) >= args.max_spots:
            break
    if len(spots) < 6:
        raise SystemExit(f"only {len(spots)} reachable hover spots")
    # --end-here: come back to where the arm stood (2026-11-26: the taught PARK touched the raised board in cuRobo)
    seq = [start] + [s[3] for s in spots] + [start if args.end_here else plan77.PARK]
    segs = []
    py = sys.executable
    me = str(Path(__file__).resolve())
    for i in range(1, len(seq)):
        path = plan77.thin(plan77.cu_plan(api, seq[i - 1], seq[i], obs))
        segs.append({"kind": "joints", "label": f"자리 {i}" if i < len(seq) - 1 else "park",
                     "q": np.round(plan77.timed([(path, args.speed)]), 3).tolist(), "dt": 0.05})
        if i < len(seq) - 1:
            segs += [{"kind": "grip", "label": "열기", "value": 100},
                     {"kind": "wait", "label": "멈춤", "seconds": 0.6},
                     {"kind": "check", "label": "사진(열림)", "cmd": [py, me, "snap", f"{i:02d}_open"]},
                     {"kind": "grip", "label": "닫기", "value": GRIP_CLOSED},
                     {"kind": "wait", "label": "멈춤", "seconds": 0.6},
                     {"kind": "check", "label": "사진(닫힘)", "cmd": [py, me, "snap", f"{i:02d}_closed"]}]
    segs.append({"kind": "grip", "label": "열기", "value": 100})
    segs.append({"kind": "check", "label": "보정 계산(2-D 지도)", "cmd": [py, me, "fit2d"], "timeout_s": 60})
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
    t0 = time.time()  # only a fresh frame (the top viewer lagged 0.5-5 s per photo on 2026-11-26)
    while time.time() - t0 < 3.0:
        try:
            age = json.loads(urllib.request.urlopen("http://127.0.0.1:8090/stats", timeout=2).read()).get("color", {}).get("age_s")
            if age is not None and age < 0.3:
                break
        except Exception:  # noqa: BLE001 - an older viewer without /stats
            break
        time.sleep(0.1)
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


def _centroids(win_px=120.0, min_n=30):
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
        if win_px:
            win = np.hypot(uu - guess[0], vv - guess[1]) < win_px
        else:  # no usable guess (new room): start at the median of every changed pixel, then shift a 120 px window
            ch = d > 35
            ch[:8], ch[-8:], ch[:, :8], ch[:, -8:] = False, False, False, False
            if ch.sum() < min_n:
                continue
            c = np.array([np.median(uu[ch]), np.median(vv[ch])])
            for _ in range(4):
                win = np.hypot(uu - c[0], vv - c[1]) < 120
                if (ch & win).sum() < min_n:
                    break
                c = np.array([np.median(uu[ch & win]), np.median(vv[ch & win])])
            win = np.hypot(uu - c[0], vv - c[1]) < 120
        mask = (d > 35) & win
        if mask.sum() < min_n:
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


def cmd_fit2d(args):
    import align  # noqa: E402 - FINGER_POINT: flange-frame centre of the open/close difference (fit 2026-11-25)
    obs = [o for o in _centroids(win_px=args.win, min_n=800)]
    if len(obs) < 6:
        print(f"보정 실패: 쓸 만한 열림/닫힘 차이가 {len(obs)}곳뿐이에요")
        sys.exit(1)
    rows, rhs, info = [], [], []
    for o in obs:
        T = kin.fk(np.array(o["q"]))
        p = T[:3, :3] @ align.FINGER_POINT + T[:3, 3]
        f = (T[:3, :3] @ align.FINGER_POINT)[:2]
        f = f / np.linalg.norm(f)  # along the fingers (horizontal)
        l = np.array([-f[1], f[0]])
        w = locate.at(o["px"][0], o["px"][1], p[2])[:2]  # camera world at the finger point's height
        # unknowns c, s, tx, ty, oa, ol:  [c -s; s c] w + t = p + oa f + ol l
        rows += [[w[0], -w[1], 1, 0, -f[0], -l[0]], [w[1], w[0], 0, 1, -f[1], -l[1]]]
        rhs += [p[0], p[1]]
        info.append((o["i"], w, p[:2], f, l))
    A, b = np.array(rows, float), np.array(rhs, float)
    prior = np.zeros((2, 6))
    prior[0, 4] = prior[1, 5] = 1.0 / 10.0  # finger-centre offset ~ +-10 mm
    x = np.linalg.lstsq(np.vstack([A, prior]), np.r_[b, 0, 0], rcond=None)[0]
    scale = float(np.hypot(x[0], x[1]))
    th = math.atan2(x[1], x[0])
    c, s_ = math.cos(th), math.sin(th)
    R2 = np.array([[c, -s_], [s_, c]])
    # translation and offset again with a pure rotation (locate.from77 inverts R2 by transposing)
    A2 = np.array([[1, 0, -it[3][0], -it[4][0]] for it in info] + [[0, 1, -it[3][1], -it[4][1]] for it in info], float)
    b2 = np.r_[[it[2][0] - (R2 @ it[1])[0] for it in info], [it[2][1] - (R2 @ it[1])[1] for it in info]]
    y = np.linalg.lstsq(np.vstack([A2, [[0, 0, 0.1, 0], [0, 0, 0, 0.1]]]), np.r_[b2, 0, 0], rcond=None)[0]
    t2, oa, ol = y[:2], float(y[2]), float(y[3])
    res = np.array([R2 @ it[1] + t2 - (it[2] + oa * it[3] + ol * it[4]) for it in info])
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    probe = [(u, v) for u in (220, 300, 380, 460) for v in (100, 180, 260)]
    old = np.array([locate.to77(locate.at(u, v, 264.0)) for u, v in probe])
    new = np.array([R2 @ locate.at(u, v, 264.0)[:2] + t2 for u, v in probe])
    shift = np.linalg.norm(new - old, axis=1)
    out = {"yaw_deg": round(math.degrees(th), 4), "R2": R2.round(8).tolist(), "T": t2.round(3).tolist(),
           "finger_offset_mm": [round(oa, 1), round(ol, 1)], "scale_free_fit": round(scale, 4), "rms_mm": round(rms, 2),
           "n": len(info), "time": time.strftime("%Y-%m-%d %H:%M"),
           "residuals_mm": {str(it[0]): np.round(r_, 1).tolist() for it, r_ in zip(info, res)},
           "_src": "cal77.py fit2d: empty-hand hovers, open/close difference centroids"}
    msg = (f"2-D 지도: {len(info)}곳, 잔차 rms {rms:.1f} mm, 회전 {math.degrees(th):+.2f}° (이전 "
           f"{locate.cal77().get('yaw_deg', float('nan')):+.2f}°), 이전 지도와 차이 평균 {shift.mean():.1f} mm (최대 {shift.max():.1f}), "
           f"손가락 중심 오프셋 앞 {oa:+.1f} / 옆 {ol:+.1f} mm, 축척 {scale:.3f}")
    if abs(scale - 1.0) > 0.03:
        msg += " (축척이 3% 넘게 달라요: 카메라 높이가 바뀌었을 수 있어요)"
    if rms > 10.0 or len(info) < 8:  # 2026-11-25 old-room photos: 12 spots, 5.9 mm, map within 5.5 mm of the old one
        print(msg + ": 잔차가 크거나 자리가 적어 지도를 바꾸지 않아요")
        sys.exit(1)
    if not args.dry_run:
        msg += save_map(out, R2, t2, "cal77.py fit2d")
    print(msg)


def save_map(out: dict, R2, t2, who: str = "cal77.py fit2d") -> str:
    """Write the camera-world -> .77 map to cal77.json and the same .77 pose into workspace.json (old files kept)."""
    R2, t2 = np.asarray(R2, float), np.asarray(t2, float)
    p = locate.LAB / "cal77.json"
    if p.exists():
        p.rename(p.with_name(f"cal77.before-{time.strftime('%Y%m%d-%H%M')}.json"))
    p.write_text(json.dumps(out, indent=1))
    msg = f" -> {p}"
    # the console places world-frame things (depth heightmap, world boxes) with workspace.json's .77 pose: make it
    # the same map. base -> world: xy = R2^T (p - T); z: camera world = planner frame = real .77 z - BASE_DZ
    pw = locate.LAB / "workspace.json"
    if pw.exists():
        ws_cfg = json.loads(pw.read_text())
        a77 = ws_cfg.get("arms", {}).get("77")
        if a77 is not None:
            Rb = np.eye(3)
            Rb[:2, :2] = R2.T
            a77["R_base"] = Rb.round(8).tolist()
            a77["origin_mm"] = [*np.round(R2.T @ (-t2), 2).tolist(), round(-kin.BASE_DZ, 2)]
            a77["yaw_deg"] = 0.0
            a77["_cal"] = f"{who} {time.strftime('%Y-%m-%d %H:%M')} (same map as cal77.json)"
            pw.rename(pw.with_name(f"workspace.before-cal77-{time.strftime('%Y%m%d-%H%M')}.json"))
            pw.write_text(json.dumps(ws_cfg, indent=1, ensure_ascii=False))
            msg += " (workspace.json .77 pose too; restart the console)"
    return msg


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
    p.add_argument("--clear", action="store_true", help="the table is empty: do not measure the cup and vials")
    p.add_argument("--speed", type=float, default=45.0, help="deg/s between spots (30 in a new room)")
    p.add_argument("--ring", type=float, nargs=2, metavar=("R_IN", "R_OUT"),
                   help="hover on three rings between these radii (mm) at bearings -70..70 instead of the grid")
    p.add_argument("--end-here", action="store_true", help="return to the start pose instead of PARK")
    p.add_argument("--keepout", type=float, nargs=2, metavar=("R_MM", "H_MM"),
                   help="objects left in place: keep the arm above table + H beyond R mm from the .77 axis")
    p.add_argument("--fence", type=float, nargs=2, metavar=("R_MM", "H_MM"),
                   help="keep the arm inside R mm of the .77 axis below table + H (tall fixtures farther out)")
    p.add_argument("--depth-obs", type=float, metavar="PAD_MM",
                   help="add the top depth's boxes (depth77) around the arm, grown by PAD_MM for the map error")
    p.add_argument("--depth-low", type=float, default=-205.0,
                   help="with --depth-obs: also take things this far under the table top (the .77's lower desk)")
    p.add_argument("--board-margin", type=float, default=0.0, metavar="MM",
                   help="move the board's near face this much toward the .77 (map error at the table edge)")
    p.add_argument("--polar", type=float, nargs="+", metavar="R", help="hover radii (mm), with --bearings")
    p.add_argument("--bearings", type=float, nargs=3, default=[-20.0, 70.0, 15.0], metavar=("FROM", "TO", "STEP"))
    p.add_argument("--max-spots", type=int, default=14)
    p.add_argument("--heights", type=float, nargs="+", help="pad heights (planner z, table 197) to try per spot")
    p.add_argument("--box", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"),
                   help="hover on a 4 x 4 grid over this rectangle of the .77 frame (mm) instead of the default grid")
    p = sp.add_parser("snap")
    p.add_argument("name")
    sp.add_parser("fit")
    p = sp.add_parser("fit2d")
    p.add_argument("--win", type=float, default=0.0,
                   help="search radius (px) around the old map's guess; 0 = whole image (a new room)")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    {"plan": cmd_plan, "snap": cmd_snap, "fit": cmd_fit, "fit2d": cmd_fit2d}[a.cmd](a)
