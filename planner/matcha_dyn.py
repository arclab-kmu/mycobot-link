"""matcha_dyn.py [--out scen.json] [--photo-dir DIR]: measure, plan and check the matcha scenario now.

1. top camera colour + depth -> cup rim, matcha vial (greenest cap), other vials (caps at vial height)
2. .77: grasp / carry / 140-degree pour with two shakes / return (plan77, templates + J1 turn + IK).
   A vial too far for the flat grasp + pour is first carried (tipped grasp) to the nearest free spot
   of NEAR_SPOTS from which the whole pour plans (relocate)
3. .70: stir at the measured cup (plan70) with the top-camera centring step at the dip
4. the .70 starts during the .77 return; the delay is the smallest one whose two-arm clearance
   (cuRobo spheres + vial + spoon, +-1.5 s timing skew) stays above the margins
Prints the scenario JSON (console format) on stdout and a short report on stderr.
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
import detect  # noqa: E402
import kin77 as kin  # noqa: E402 - table-referenced frame (kin77.BASE_DZ)
import locate  # noqa: E402
import plan70  # noqa: E402
import plan77  # noqa: E402
from mycobot_brain.console import workspace  # noqa: E402

WS = workspace.load(str(locate.lab_file("workspace.json")))
API = WS.curobo_api
CONSOLE = "http://127.0.0.1:8095"
BACKLASH70 = [1.4, 1.0, 1.0, 1.0, 1.1, 0.7]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _get(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def arm_q(name):
    try:
        st = json.loads(_get(CONSOLE + "/api/state"))
        q = st["arms"][name]["q"]
        return None if q is None else np.array(q, float)
    except Exception:  # noqa: BLE001 - planning still works without the console
        return None


# -- two-arm clearance on the time axis -------------------------------------------------------
VIAL = [(np.array(plan77.VIAL_F), 45.0)]
SPOON = [(np.array([-70.8, -98.7, 21.5]) * f, 12.0) for f in np.linspace(0.15, 1.0, 7)]
_sph_cache = {}


def _spheres(q):
    """cuRobo's collision spheres from the local DH fit (spheres.py; within ~2 mm of /spheres)."""
    import spheres
    return spheres.spheres(np.asarray(q, float))


_CAL = locate.cal77()
_R2, _T2 = np.array(_CAL["R2"]), np.array(_CAL["T"])


def world77(p77):
    """.77 base frame -> calib_top3 world (the inverse of the cal77 map; z is shared)."""
    p = np.atleast_2d(p77)
    return np.column_stack([(p[:, :2] - _T2) @ _R2, p[:, 2]])


def world70(p70):
    import arm70n as A70  # noqa: E402 - plan70 put s70 on the path (same module object mv70 uses)
    return np.atleast_2d(p70) @ A70.R77_70.T + np.array([A70.BX, A70.BY, A70.BZ])


def arm_world(arm, q, extra):
    s = _spheres(q).copy()
    if arm == "77":
        s[:, 2] -= kin.BASE_DZ  # spheres come in the real base frame; the .77's world here is table-referenced
    # the spoon/tool spheres ride on the same frame as the arm spheres: table-referenced for the .77, raw for the .70
    T = kin.fk(np.asarray(q, float)) if arm == "77" else kin.fk_taught(np.asarray(q, float))
    if extra:
        s = np.vstack([s, [np.r_[T[:3, :3] @ p + T[:3, 3], r] for p, r in extra]])
    c = world77(s[:, :3]) if arm == "77" else world70(s[:, :3])
    return np.column_stack([c, s[:, 3]])


def timeline(segs, from_step=0, hold_call=1.0, held_until=None):
    out, t = [], 0.0
    q = None
    for k, sg in enumerate(segs):
        if k < from_step:
            continue
        held = held_until is not None and k < held_until
        if sg["kind"] == "joints":
            Q = np.array(sg["q"])
            for i in range(0, len(Q), 3):
                out.append((t + i * sg["dt"], Q[i], held))
            t += len(Q) * sg["dt"]
            q = Q[-1]
        else:
            sec = 0.8 if sg["kind"] == "grip" else sg.get("seconds", hold_call)
            out.append((t, q, held))
            t += sec
    out.append((t, q, False))
    return out, t


def clearance(tl77, tl70, delay, skew):
    """Smallest sphere-surface gap between the arms when the .70 starts `delay` s after .77 step 7 and the
    .77 runs `skew` s ahead. Sphere sets are computed once per timeline sample and cached on the lists."""
    ts77 = np.array([x[0] for x in tl77])
    best = 1e9
    for t70, b in tl70:
        t = t70 + delay + skew
        i = int(np.clip(np.searchsorted(ts77, t, side="right") - 1, 0, len(tl77) - 1))
        a = tl77[i][1]
        d = np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=-1) - a[:, None, 3] - b[None, :, 3]
        best = min(best, float(d.min()))
    return best


REST_GAP_MM = 30.0  # the .77's whole work (vial in hand) against the .70 standing at plan70.HOME


def rest_gap(phases77, q70):
    """Smallest sphere gap between every .77 sample (with the vial once gripped) and the .70 at rest at q70.
    choose_delay only covers the time after the .70 starts; this covers the pour it waits through."""
    B = arm_world("70", np.asarray(q70, float), SPOON)
    best = 1e9
    for ph in phases77:
        held = False
        for s in ph["segments"]:
            if s["kind"] == "grip":
                held = s["value"] < 50
            if s["kind"] != "joints":
                continue
            for q in s["q"][::3]:
                A = arm_world("77", np.asarray(q, float), VIAL if held else [])
                d = np.linalg.norm(A[:, None, :3] - B[None, :, :3], axis=-1) - A[:, None, 3] - B[None, :, 3]
                best = min(best, float(d.min()))
    return best


def to_rest70():
    """A first .70 phase that moves it to plan70.HOME when it stands elsewhere nearby (e.g. the old rest pose)."""
    q70 = arm_q("70")
    if q70 is None or np.max(np.abs(q70 - plan70.HOME)) <= 2.0:
        return None
    if np.max(np.abs(q70 - plan70.HOME)) > 30.0:
        raise plan77.PlanError(f".70이 기본 자세에서 {np.max(np.abs(q70 - plan70.HOME)):.0f}도 떨어져 있어요. 먼저 기본 자세로 돌려 주세요.")
    Q = plan77.timed([([q70, plan70.HOME], 15.0)])
    return {"arm": "70", "label": ".70 기본 자세로", "speed_dps": 30, "workspace_obstacles": False,
            "segments": [{"kind": "joints", "label": ".70 기본 자세로", "q": np.round(Q, 3).tolist(), "dt": 0.05}]}


def choose_delay(ph77, ph70, step=7, held_until=8, delays=np.arange(2.0, 12.01, 0.5), nominal=40.0, skewed=20.0):
    tl77, _ = timeline(ph77["segments"], from_step=step, held_until=held_until)
    tl70, _ = timeline(ph70["segments"])
    tl77 = [(t, arm_world("77", q, VIAL if held else [])) for t, q, held in tl77]
    tl70 = [(t, arm_world("70", q, SPOON)) for t, q, _ in tl70]
    for d in delays:
        c0 = clearance(tl77, tl70, d, 0.0)
        if c0 < nominal:
            continue
        cs = min(clearance(tl77, tl70, d, -1.5), clearance(tl77, tl70, d, +1.5))
        if cs >= skewed:
            return float(d), c0, cs
    return None, None, None


# -- the whole scenario ----------------------------------------------------------------------
def measure(photo_dir=None):
    rgb_b = _get("http://127.0.0.1:8090/color.jpg")
    dep_b = _get("http://127.0.0.1:8090/depth.png")
    from PIL import Image
    rgb = np.asarray(Image.open(io.BytesIO(rgb_b)).convert("RGB")).astype(float)
    if photo_dir:
        Path(photo_dir).mkdir(parents=True, exist_ok=True)
        (Path(photo_dir) / "top.jpg").write_bytes(rgb_b)
        (Path(photo_dir) / "top_depth.png").write_bytes(dep_b)
    d = detect.detect(rgb, lambda pts: locate.heights(dep_b, pts), r_fix=locate.rim_px_radius, cup_ok=locate.cup_ok)
    d["depth_caps"] = []
    if d["matcha"] is None or not d["others"]:
        # caps the colour finder misses (silver caps, 2026-11-27): from the depth, minus anything within 120 mm of the
        # .77 axis (its own gripper at park made a cap-height blob at r 89)
        try:
            dc = detect.caps_by_depth(dep_b, locate.CAM, plan77.TABLE, d["cup"])
            d["depth_caps"] = [c for c in dc
                               if np.hypot(*locate.to77(locate.at(c["u"], c["v"], locate.CAP_Z))) >= 120.0]
        except Exception as exc:  # noqa: BLE001 - the colour result still stands
            log(f"caps_by_depth failed: {exc}")
    global LAST_DEPTH
    LAST_DEPTH = dep_b  # the same frame's depth, for depth77.boxes()
    return rgb, d


LAST_DEPTH = None


def depth_obstacles(q77, vial77, cup77):
    """What stands on the table now, from the measured frame's depth (Hojin 2026-11-26: "curobo나 경로 검사 시에는
    depth를 이용하고"), minus the vial to grasp, the cup's inside and the .77 itself."""
    if LAST_DEPTH is None:
        return []
    import depth77
    t0 = time.perf_counter()
    bx = depth77.boxes(LAST_DEPTH, plan77.PARK if q77 is None else q77, skip=[(float(vial77[0]), float(vial77[1]), 30.0)],
                       cup=cup77, cup_r=55.0)
    log(f"depth obstacles: {len(bx)} boxes in {(time.perf_counter() - t0) * 1000:.0f} ms")
    return bx


#: Spots in the .77 frame (mm) from which the flat grasp + 140 deg pour planned on 2026-11-25 (reach map);
#: a vial that cannot be poured from where it stands is moved to the nearest free one first.
NEAR_SPOTS = [(157.0, -164.0), (150.0, -140.0), (165.0, -185.0), (140.0, -120.0), (170.0, -140.0), (130.0, -150.0)]
# centre distances (mm) from the cup and the other vials; a vial needs ~100 mm around it for the gripper to keep
# plan77.NEIGHBOUR_MM from its neighbours (2026-11-26: 87 mm apart left no grasp, 100 mm planned)
SPOT_CLEAR_CUP, SPOT_CLEAR_VIAL = 95.0, 100.0
NEIGHBOUR_HINT_MM = 100.0
RELOCATE_BUDGET_S = 40.0


#: Which vial a run pours. Same vial type and cap height; the matcha one has a green cap, the coffee one does not.
VIALS = {
    "matcha": {"name": "말차 병", "drink": "말차", "trigger": "컵에 말차 가루를 타 줘", "title": "말차 타기 (지금 위치로 계획)"},
    "coffee": {"name": "커피 병", "drink": "커피", "trigger": "커피 타 줘", "title": "커피 타기 (지금 위치로 계획)"},
}


def bias77():
    """Radial shift (mm, away from the .77 base) added to a measured vial before planning its grasp, from
    LAB/cal77_bias.json {"radial_mm": r}. 2026-11-26: three grasps closed just short of the vial along the
    approach (the cap ended at the fingertips), i.e. the vial stood farther out than the camera map put it."""
    p = locate.LAB / "cal77_bias.json"
    try:
        return float(json.loads(p.read_text()).get("radial_mm", 0.0)) if p.exists() else 0.0
    except (ValueError, OSError):
        return 0.0


def _by_position(caps, vial):
    """Silver caps: the lab says where the matcha vial stands in the top image (workspace.json
    "vial_by_position": {"matcha": "right" | "left"}, right = larger colour u); coffee is the other one."""
    rule = json.loads(locate.lab_file("workspace.json").read_text()).get("vial_by_position", {}).get("matcha")
    if rule not in ("right", "left") or len(caps) < 1:
        raise plan77.PlanError("뚜껑 색으로 병을 구별하지 못했어요: 어느 쪽이 말차 병인지 workspace.json "
                               "vial_by_position 으로 알려 주세요")
    order = sorted(caps, key=lambda c: c["u"], reverse=(rule == "right"))
    t = order[0] if vial == "matcha" else (order[1] if len(order) > 1 else None)
    if t is None:
        raise plan77.PlanError("깊이로 찾은 병이 하나뿐이라 커피 병을 고를 수 없어요")
    log(f"{vial} vial by position ({rule} = matcha) from {len(caps)} depth caps")
    return t, [c for c in caps if c is not t]


def pick_target(d, vial):
    """(target cap, other caps) from detect(): matcha = the green cap; coffee = the least green of the others.
    With no colour caps (silver), the depth caps and the lab's position rule (_by_position)."""
    if d["matcha"] is None:  # no green cap: colour cannot tell the vials apart (both silver), so position decides
        return _by_position(d.get("depth_caps") or list(d["others"]), vial)
    caps = ([d["matcha"]] if d["matcha"] is not None else []) + list(d["others"])
    if vial == "matcha":
        t = d["matcha"]
        if t is None:
            raise plan77.PlanError("말차 병(초록 뚜껑)을 위 카메라에서 찾지 못했어요")
    else:
        cand = [c for c in caps if c["green"] <= 3.0]
        if not cand:
            raise plan77.PlanError("커피 병(초록이 아닌 뚜껑)을 위 카메라에서 찾지 못했어요")
        if len(cand) > 1:
            log(f"커피 병 후보 {len(cand)}개: 초록이 가장 적은 뚜껑을 고릅니다")
        t = min(cand, key=lambda c: c["green"])
    return t, [c for c in caps if c is not t]


def relocate(vial77, cup77, others77, start77, t0, name="말차 병", target="matcha"):
    """(move phase, pour phase, info, spot): carry the vial to a NEAR_SPOTS spot, then plan the pour from there."""
    others = [o.tolist() for o in others77]
    free = [s for s in NEAR_SPOTS
            if math.dist(s, cup77) >= SPOT_CLEAR_CUP and all(math.dist(s, o[:2]) >= SPOT_CLEAR_VIAL for o in others77)
            and math.dist(s, vial77) >= 15.0]
    free.sort(key=lambda s: math.dist(s, vial77))
    tried = []
    for s in free:
        if time.perf_counter() - t0 > RELOCATE_BUDGET_S:
            break
        try:
            ph_mv, info_mv = plan77.plan_move(vial77, np.array(s), others, API, WS.obstacles_for(WS.arms["77"]), cup_xy=cup77,
                                              q_start=start77, name=name, target=target)
        except plan77.PlanError as exc:
            tried.append(f"move to {s}: {str(exc)[:80]}")
            continue
        q_end = np.asarray(ph_mv["segments"][-1]["q"][-1], float)
        q_next = None if np.max(np.abs(q_end - plan77.PARK)) < 5 else q_end
        try:
            ph77, info77 = plan77.plan_phase(np.array(s), cup77, others, API, WS.obstacles_for(WS.arms["77"]), q_start=q_next,
                                             name=name, target=target)
        except plan77.PlanError as exc:
            tried.append(f"pour from {s}: {str(exc)[:80]}")
            continue
        log(f"relocate: carry to ({s[0]:.0f}, {s[1]:.0f}) planned in {info_mv['t_total_s']:.2f} s (pick {info_mv['pick']}), "
            f"then pour from there")
        info77 = dict(info77, moved_to=list(s), move_pick=info_mv["pick"])
        return ph_mv, ph77, info77, s
    near = min((math.dist(vial77, o[:2]) for o in others77), default=1e9)
    if near < NEIGHBOUR_HINT_MM:
        raise plan77.PlanError(f"{name} 옆의 다른 병이 {near / 10:.1f} cm로 가까워서 손가락이 들어갈 자리가 없어요. "
                               f"다른 병을 {name}에서 {NEIGHBOUR_HINT_MM / 10 + 2:.0f} cm 이상 떨어뜨려 주세요.")
    raise plan77.PlanError(f"{name}을 붓기 좋은 자리로 옮기는 계획도 찾지 못했어요: " + "; ".join(tried[-3:] or ["빈 자리 없음"]))


def build(photo_dir=None, only=None, move_to=None, vial="matcha"):
    t0 = time.perf_counter()
    V = VIALS[vial]
    name = V["name"]
    rgb, d = measure(photo_dir)
    t_meas = time.perf_counter() - t0
    cup = d["cup"]
    m, other_caps = pick_target(d, vial)
    # A grasp test or a move does not use the cup. In the 2026-11-26 exhibit room the cup stood outside the colour
    # view, and a colour "cup" found elsewhere would wrongly clear the depth boxes around it; the real cup is a depth
    # box like everything else there.
    no_cup = only == "grasp" or move_to is not None
    if not no_cup and (cup is None or cup["n_edge"] < 40):
        raise plan77.PlanError("컵 테두리를 위 카메라에서 찾지 못했어요")
    cal = locate.cal77()
    vial77 = locate.to77(locate.at(m["u"], m["v"], locate.CAP_Z), cal)
    rb = bias77()
    if rb:
        vial77 = vial77 + rb * vial77 / np.linalg.norm(vial77)
        log(f"vial position shifted {rb:+.0f} mm outward (cal77_bias.json)")
    if no_cup:
        cup77 = np.array([2000.0, 2000.0])  # far away: no cup walls, no depth hole
        cup_w = locate.from77(cup77, cal)
    else:
        cup_w = locate.at(cup["u"], cup["v"], locate.RIM_Z)
        cup77 = locate.to77(cup_w, cal)
    others77 = [np.r_[locate.to77(locate.at(c["u"], c["v"], locate.CAP_Z), cal), locate.CAP_Z - 1.5] for c in other_caps]
    log(f"measured in {t_meas:.2f} s: {vial} vial .77 ({vial77[0]:.1f}, {vial77[1]:.1f}), cup .77 ({cup77[0]:.1f}, {cup77[1]:.1f}),"
        f" other vials {[tuple(float(x) for x in np.round(o[:2], 1)) for o in others77]}")
    # .77
    q77 = arm_q("77")
    start77 = None if q77 is None or np.max(np.abs(q77 - plan77.PARK)) < 5 else q77
    if move_to is not None:
        ph, info = plan77.plan_move(vial77, move_to, [o.tolist() for o in others77], API, WS.obstacles_for(WS.arms["77"]),
                                    cup_xy=cup77, q_start=start77, name=name, target=vial,
                                    extra_obs=depth_obstacles(q77, vial77, cup77))
        total = time.perf_counter() - t0
        log(f"move planned in {info['t_total_s']:.2f} s: {info}")
        return {"trigger": f"{name} 옮기기", "title": f"{name} 옮기기 (시험)",
                "say": f"{name}을 ({vial77[0]:.0f}, {vial77[1]:.0f})에서 ({move_to[0]:.0f}, {move_to[1]:.0f}) mm로 옮깁니다. 계획 {total:.1f}초.",
                "seen": "위 카메라로 병·컵 위치를 재고 경로는 cuRobo로 검사했습니다.", "phases": [ph]}
    moved, ph_mv = None, None
    # The calibrated grasp depth first, then 3 mm shallower steps: at the reach edge the deeper hand puts J4 on the
    # board (2026-11-26, matcha vial at r 248: 13 mm none, 10 mm found). The inside check measures what is left.
    dep_obs = depth_obstacles(q77, vial77, cup77)
    d0 = plan77.grasp_deeper_mm()
    depths = [d0] + [d for d in (d0 - 3.0, d0 - 6.0, d0 - 9.0) if d >= 0.0]
    try:
        for i, d in enumerate(depths):
            try:
                ph77, info77 = plan77.plan_phase(vial77, cup77, [o.tolist() for o in others77], API, WS.obstacles_for(WS.arms["77"]),
                                                 q_start=start77, grasp_test=(only == "grasp"), name=name, target=vial, deeper=d,
                                                 extra_obs=dep_obs)
                if i:
                    log(f"grasp depth {d0:.0f} mm had no path; planned at {d:.0f} mm deeper")
                break
            except plan77.PlanError as exc:
                if i == len(depths) - 1:
                    raise
                log(f"grasp {d:.0f} mm deeper: {str(exc)[:120]}")
    except plan77.PlanError as exc:
        if only == "grasp":
            raise
        log(f"no flat grasp + pour from ({vial77[0]:.0f}, {vial77[1]:.0f}): {str(exc)[:160]}")
        ph_mv, ph77, info77, moved = relocate(vial77, cup77, others77, start77, t0, name=name, target=vial)
    if only == "grasp":
        total = time.perf_counter() - t0
        return {"trigger": f"{name} 잡기 시험", "title": f"{name} 잡고 들었다 놓기 (시험)",
                "say": f"위 카메라로 {name}을 찾고 {total:.1f}초 만에 계획했습니다. 잡고 들었다가 같은 자리에 놓습니다.",
                "seen": f"{name} ({vial77[0]:.0f}, {vial77[1]:.0f}) mm (.77 기준)", "phases": [ph77],
                "_plan": {"vial77": vial77.round(1).tolist(), "info77": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                                                       for k, v in info77.items()}}}
    log(f".77 planned in {info77['t_total_s']:.2f} s: " + ", ".join(f"{k} {float(v):g}" if isinstance(v, (int, float, np.floating)) else f"{k} {v}"
                                                               for k, v in info77.items()))
    # .70
    t1 = time.perf_counter()
    aim = cup_w[:2] + plan70.bias70()
    im = rgb.astype(int)
    x0, x1 = int(max(cup["u"] - 160, 0)), int(min(cup["u"] + 160, 640))
    y0, y1 = int(max(cup["v"] - 160, 0)), int(min(cup["v"] + 160, 480))
    vv, uu = np.nonzero(im[y0:y1, x0:x1].min(2) > 170)
    blk = np.array([locate.at(a + x0, b + y0, 240.0)[:2] for a, b in zip(uu[::2], vv[::2])]) if len(uu) else np.zeros((0, 2))
    SP = plan70.StirPlanner(aim, locate.RIM_Z, blk, blk_centre=cup_w[:2])
    P, tried = SP.best()
    if P is None:
        raise plan77.PlanError(".70 젓기 경로를 찾지 못했어요: " + "; ".join(tried[-3:]))
    up = plan77.timed([(plan70.transit(P["start"]), 45.0)])
    down_home = up[::-1]
    ax, px2mm = plan70.axis_and_px2mm(locate.at, locate.CAM.world_to_pixel, (cup["u"], cup["v"]), locate.RIM_Z, SP.ZD)
    J = plan70.jac_mm(P["down"][-1], SP.TH1, P["yaw"])
    H = 14
    au, av = ax
    centre = {"kind": "center", "label": "컵 중심 확인(위 카메라)", "camera": "top", "template": str(HERE / "data" / "bowl.png"),
              # top-left corners of the 2H template, kept inside the 640 x 480 frame (a cup at the frame edge pushed v0 below 0)
              "search": [max(int(au - 45 - H), 0), max(int(av - 40 - H), 0), min(int(au + 45 - H), 640 - 2 * H), min(int(av + 40 - H), 480 - 2 * H)],
              "axis_px": np.round(ax, 2).tolist(),
              "px2mm": np.round(px2mm, 5).tolist(), "J": np.round(J, 4).tolist(), "max_err_mm": 22, "deadzone_mm": 2, "min_corr": 0.6}
    R3 = lambda Q: np.round(np.asarray(Q), 3).tolist()  # noqa: E731
    ph70 = {"arm": "70", "label": "스푼으로 젓기", "speed_dps": 80, "workspace_obstacles": False, "backlash": BACKLASH70, "v0": 1.0,
            "segments": [
                {"kind": "joints", "label": "컵 위로", "q": R3(up), "dt": 0.05},
                {"kind": "joints", "label": "컵 안으로", "q": R3([P["start"]] + P["down"]), "dt": 0.05},
                centre,
                {"kind": "joints", "label": f"{SP.NC}바퀴 젓기", "q": R3([P["down"][-1]] + P["stir"]), "dt": 0.1, "offset": "in", "ramp": 10},
                {"kind": "joints", "label": "올라오기", "q": R3(P["up"]), "dt": 0.05, "offset": "out", "ramp": 25},
                {"kind": "joints", "label": "홈으로", "q": R3(down_home), "dt": 0.05}]}
    t_70 = time.perf_counter() - t1
    log(f".70 planned in {t_70:.2f} s: aim ({aim[0]:.1f}, {aim[1]:.1f}) world, yaw {P['yaw']}, worst rim {P['worst'][0]:.1f} mm")
    t2 = time.perf_counter()
    labels = [sg.get("label") for sg in ph77["segments"]]
    k_back, k_rel = labels.index("세워서 제자리로"), labels.index("놓기")
    delay, c0, cs = choose_delay(ph77, ph70, step=k_back, held_until=k_rel)
    t_pair = time.perf_counter() - t2
    if delay is None:
        log("no overlap delay passed; the .70 waits for the .77 to finish")
    else:
        ph70["start_with"] = {"step": k_back, "delay_s": delay}
        log(f"two-arm overlap: .70 starts {delay:.1f} s into the .77 return (clearance {c0:.0f} mm, {cs:.0f} mm with 1.5 s skew), "
            f"checked in {t_pair:.2f} s")
    rg = rest_gap([p for p in (ph_mv, ph77) if p], plan70.HOME)
    log(f".70 at rest vs the .77's work: {rg:.0f} mm")
    if rg < REST_GAP_MM:
        raise plan77.PlanError(f".77이 붓는 동안 쉬고 있는 .70과 {rg:.0f} mm까지 가까워져요 (기준 {REST_GAP_MM:.0f} mm)")
    pre70 = to_rest70()
    total = time.perf_counter() - t0
    TA = plan77.TPL["pour"]["tilt_angles"]
    scen = {"trigger": V["trigger"], "title": V["title"],
            "say": f"위 카메라로 컵과 병을 찾고 {total:.1f}초 만에 계획했습니다. "
                   + (f"병이 멀어서 먼저 숙여 잡아 ({moved[0]:.0f}, {moved[1]:.0f}) mm로 옮긴 뒤, " if moved else "")
                   + f".77이 {name}을 잡아 {TA[-1]}도까지 기울여 두 번 털고, 병을 제자리로 가져가는 동안 .70이 컵으로 와서 스푼으로 젓습니다.",
            "seen": f"{name} ({vial77[0]:.0f}, {vial77[1]:.0f}) mm, 컵 ({cup77[0]:.0f}, {cup77[1]:.0f}) mm (.77 기준), "
                    f"다른 병 {len(others77)}개. 경로는 cuRobo로, 두 팔이 겹치는 구간은 시간축 거리로 검사했습니다.",
            "phases": ([pre70] if pre70 else []) + ([ph_mv] if ph_mv else []) + [ph77, ph70],
            "_plan": {"t_measure_s": round(t_meas, 2), "t_total_s": round(total, 2), "vial77": vial77.round(1).tolist(),
                      "moved_to": None if moved is None else list(moved),
                      "cup77": cup77.round(1).tolist(), "cup_px": [round(cup["u"], 1), round(cup["v"], 1)],
                      "vial_px": [round(m["u"], 1), round(m["v"], 1)], "others77": [o.round(1).tolist() for o in others77],
                      "info77": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info77.items()},
                      "aim70": aim.round(1).tolist(), "delay_s": delay, "clearance_mm": None if c0 is None else [round(c0, 1), round(cs, 1)]}}
    log(f"total {total:.2f} s")
    return scen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--photo-dir")
    ap.add_argument("--only", choices=["grasp"])
    ap.add_argument("--move-to", help="x,y in the .77 frame: stand the vial there")
    ap.add_argument("--vial", choices=sorted(VIALS), default="matcha", help="which vial to pour (matcha = green cap)")
    a = ap.parse_args()
    try:
        sc = build(a.photo_dir, a.only, None if not a.move_to else np.array([float(v) for v in a.move_to.split(",")]), vial=a.vial)
    except plan77.PlanError as exc:
        log(f"PLAN FAILED: {exc}")
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        sys.exit(2)
    txt = json.dumps(sc, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
