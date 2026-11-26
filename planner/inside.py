"""inside.py: is the vial between the fingers at the grasp pose? One top-camera image pair, measured at the pads.

Hojin (2026-11-26): "그리퍼 내부에 들어오면 오므리자" - close only once the vial is inside the gripper. Closing at the
planned pose had closed the fingertips on the vial's near edge three times that day (the vial stood ~12 mm farther
out than the camera map put it) and pushed it away each time.

At the planned grasp pose the gripper is photographed open (100) and slightly closed (60, still wider than the
vial). The changed pixels are the moving finger parts. Projected onto the plane at pad height they give
  - across the approach: two groups (one per finger); their midline is the finger centre line;
  - along the approach: the far end (95th percentile) is the fingertips; the pad centre is PAD_BACK mm behind it.
The vial cap is found near its planned pixel and placed at cap height. The hand should move by the cap's offset
from the pad centre; the console shifts the rest of the grasp by it (dq = -J err).

  inside.py snap NAME
  inside.py measure --q Q6 --vial X Y --approach AX AY [--final]

Prints {"err_mm": [x, y], "inside": bool, "lat_mm", "along_mm", "msg"} as its last line. Exit 1 when nothing could
be measured, or with --final when the vial is still not inside (the scenario then stops without closing).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align  # noqa: E402 - snap() and the photo folder
import detect  # noqa: E402
import kin  # noqa: E402
import locate  # noqa: E402
import plan77  # noqa: E402

PAD_BACK = 10.0  # mm from the fingertips back to the pad centre (pads ~20 mm long)
LAT_OK, ALONG_LO, ALONG_HI = 6.0, -6.0, 8.0  # cap offset from the pad centre that the fingers still wrap
MAX_STEP = 20.0  # a measured correction above this is treated as a bad measurement


def measure(q, vial, approach, final=False):
    from PIL import Image
    a = np.asarray(Image.open(align.DIR / "g_open.jpg").convert("RGB")).astype(float)
    b = np.asarray(Image.open(align.DIR / "g_half.jpg").convert("RGB")).astype(float)
    T = kin.fk(np.asarray(q, float))
    pad = T[:3, :3] @ plan77.TLV + T[:3, 3]
    pred_pad = locate.px_of77(pad[:2], pad[2])
    pred_cap = locate.px_of77(np.asarray(vial, float), locate.CAP_Z)
    d = np.abs(a.mean(2) - b.mean(2))
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    mask = (d > 30) & (np.hypot(uu - pred_pad[0], vv - pred_pad[1]) < 100)
    if mask.sum() < 150:
        return {"msg": f"손가락 움직임을 찾지 못했어요 ({int(mask.sum())} px)"}, 1
    ax = np.asarray(approach, float)
    ax = ax / np.linalg.norm(ax)
    lat = np.array([-ax[1], ax[0]])
    pts = np.array([locate.to77(locate.at(float(u), float(v), pad[2])) for u, v in zip(uu[mask][::3], vv[mask][::3])])
    rel = pts - pad[:2]
    s_a, s_c = rel @ ax, rel @ lat
    mid = float(np.median(s_c))
    for _ in range(10):
        left, right = s_c[s_c < mid], s_c[s_c >= mid]
        if len(left) < 15 or len(right) < 15:
            return {"msg": f"손가락 두 개를 나누지 못했어요 ({len(left)} / {len(right)} 점)"}, 1
        mid = float((np.median(left) + np.median(right)) / 2)
    tips = float(np.percentile(s_a, 95))
    # The cap: the round candidate nearest its planned pixel that is also found in the second photo (a pad, which
    # moves, is not). 2026-11-26: a changed-pixel filter dropped the right cap (the closing pad sweeps next to it);
    # a 6.5 px same-spot test dropped it too once the photo close (then 60) nudged an off-centre vial by ~6 mm.
    cb = detect.find_caps(b, None, n=40, gray_min=100.0, wood_min=0.0)
    near = sorted((c for c in detect.find_caps(a, None, n=40, gray_min=100.0, wood_min=0.0)
                   if np.hypot(c["u"] - pred_cap[0], c["v"] - pred_cap[1]) < 45),
                  key=lambda c_: np.hypot(c_["u"] - pred_cap[0], c_["v"] - pred_cap[1]))
    caps = [c for c in near if any(np.hypot(c["u"] - k["u"], c["v"] - k["v"]) < 10.0 for k in cb)]
    if not caps:
        return {"msg": "예상 위치 근처에서 병뚜껑을 찾지 못했어요"}, 1
    c = caps[0]
    cap = locate.to77(locate.at(c["u"], c["v"], locate.CAP_Z)) - pad[:2]
    lat_mm = float(cap @ lat - mid)
    along_mm = float(cap @ ax - (tips - PAD_BACK))
    inside = abs(lat_mm) <= LAT_OK and ALONG_LO <= along_mm <= ALONG_HI
    dv = lat_mm * lat + along_mm * ax
    out = {"lat_mm": round(lat_mm, 1), "along_mm": round(along_mm, 1), "inside": bool(inside),
           "err_mm": (-dv).round(2).tolist(),
           "msg": f"병이 패드 중심에서 옆으로 {lat_mm:+.1f} mm, 앞으로 {along_mm:+.1f} mm"
                  + (" (손가락 사이)" if inside else " (손가락 밖)")}
    # The joint change for this hand shift, by IK from where the arm is now (the console's -J err is linear and
    # was ~2x off next to the base, 2026-11-26: a 15 mm sideways fix moved the hand ~27 mm).
    import matcha_dyn  # noqa: E402 - arm_q() reads the console
    q_now = matcha_dyn.arm_q("77")
    q_now = np.asarray(q, float) if q_now is None else np.asarray(q_now, float)
    Tn = kin.fk(q_now)
    Tt = Tn.copy()
    Tt[:2, 3] += dv
    q_t, e_ik = plan77.ik(q_now, Tt)
    if e_ik < 0.3:
        out["dq_deg"] = np.round(q_t - q_now, 3).tolist()
    if np.hypot(*dv) > MAX_STEP:
        out["msg"] += f": 보정 {np.hypot(*dv):.0f} mm는 너무 커서 측정이 틀린 것으로 봅니다"
        return out, 1
    if final and not inside:
        out["msg"] += ": 잡지 않고 멈춥니다"
        return out, 1
    return out, 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("snap")
    p.add_argument("name")
    p = sp.add_parser("measure")
    p.add_argument("--q", nargs=6, type=float, required=True)
    p.add_argument("--vial", nargs=2, type=float, required=True)
    p.add_argument("--approach", nargs=2, type=float, required=True)
    p.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.cmd == "snap":
        align.snap(a.name)
        # the side camera (D405 on the .77) too: it sees the fingers and the vial from the side, along the approach
        import urllib.request
        for ext, url in (("jpg", "http://127.0.0.1:8092/color.jpg"), ("png", "http://127.0.0.1:8092/depth.png")):
            try:
                (align.DIR / f"side_{a.name}.{ext}").write_bytes(urllib.request.urlopen(url, timeout=5).read())
            except Exception:  # noqa: BLE001 - the side view is extra evidence, not required
                pass
    else:
        out, code = measure(a.q, a.vial, a.approach, a.final)
        (align.DIR / "inside_last.json").write_text(json.dumps(out, ensure_ascii=False))
        print(json.dumps(out, ensure_ascii=False))
        sys.exit(code)
