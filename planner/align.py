"""Pre-grasp alignment in one top-camera image (relative measurement, so camera-to-arm map errors cancel).

At the pre-grasp pose the gripper is photographed open (A) and closed (B). The centre of the changed pixels is
the finger midpoint; the vial cap is found near its planned pixel. Both go to the .77 plane at their own heights
and are compared with the planned finger-to-vial vector. The vial's offset (lateral fully, along the approach
clamped to +-8 mm) is printed as the tool error for the console's "align" step.

  align.py snap NAME
  align.py measure --q Q6 --vial X Y --approach AX AY
"""
from __future__ import annotations

import argparse
import json
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

DIR = locate.LAB / "plans" / "align"
FINGER_POINT = np.array([13.1, 6.2, 87.3])  # flange frame: centre of the open/close difference (fit 2026-11-25)


def snap(name):
    DIR.mkdir(parents=True, exist_ok=True)
    time.sleep(0.2)
    # Only a fresh frame: on 2026-11-26 the top viewer lagged (0.5-5 s per photo) and two snaps 2.5 s apart came
    # back identical (0 px changed). Wait until its colour frame is younger than 0.3 s (up to 3 s).
    t0 = time.time()
    while time.time() - t0 < 3.0:
        try:
            st = json.loads(urllib.request.urlopen("http://127.0.0.1:8090/stats", timeout=2).read())
            age = st.get("color", {}).get("age_s")
            if age is not None and age < 0.3:
                break
        except Exception:  # noqa: BLE001 - an older viewer without /stats: just take the photo
            break
        time.sleep(0.1)
    for _ in range(4):
        try:
            (DIR / f"{name}.jpg").write_bytes(urllib.request.urlopen("http://127.0.0.1:8090/color.jpg", timeout=5).read())
            print(f"saved {name}")
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    print(f"사진 저장 실패 ({name})")
    sys.exit(1)


def measure(q, vial, approach):
    from PIL import Image
    a = np.asarray(Image.open(DIR / "open.jpg").convert("RGB")).astype(float)
    b = np.asarray(Image.open(DIR / "closed.jpg").convert("RGB")).astype(float)
    T = kin.fk(np.asarray(q, float))
    fp77 = T[:3, :3] @ FINGER_POINT + T[:3, 3]
    pred_f = locate.px_of77(fp77[:2], fp77[2])
    pred_v = locate.px_of77(vial, locate.CAP_Z)
    d = np.abs(a.mean(2) - b.mean(2))
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    mask = (d > 35) & (np.hypot(uu - pred_f[0], vv - pred_f[1]) < 110)
    if mask.sum() < 300:
        return {"msg": f"손가락 차이를 찾지 못했어요 ({int(mask.sum())} px)"}, 1
    # With the white gripper in view the camera's auto exposure / white balance shift: on 2026-11-26 the cap read
    # gray 139 and its wood ring 0, so the table-scene thresholds dropped it. Here the cap's pixel is predicted, so
    # search near it with loose thresholds, and skip a candidate that sits on the fingers. "On the fingers" means a
    # real share of its 25 x 25 window changed: on 2026-11-26 a correct cap 8 px from its prediction was dropped
    # because 4 of those 625 pixels had changed.
    moved = np.zeros_like(d, dtype=bool)
    moved[mask] = True
    near_finger = lambda c: moved[max(int(c["v"]) - 12, 0):int(c["v"]) + 13, max(int(c["u"]) - 12, 0):int(c["u"]) + 13].sum() > 60  # noqa: E731
    caps = [c for c in detect.find_caps(a, None, n=40, gray_min=100.0, wood_min=0.0)
            if np.hypot(c["u"] - pred_v[0], c["v"] - pred_v[1]) < 60 and not near_finger(c)]
    if not caps:
        return {"msg": "예상 위치 근처에서 병뚜껑을 찾지 못했어요"}, 1
    c = max(caps, key=lambda c_: c_["green"])
    cv = np.array([c["u"], c["v"]])
    # Only the sideways error is measured. The changed pixels cover both whole finger linkages (about 110 px long at
    # a flat grasp), so their centre along the fingers depends on the view and on 2026-11-26 sat 37 px (~29 mm) from
    # the FINGER_POINT fit of 2026-11-25. Sideways the two fingers are mirror images, so the median of the changed
    # pixels across the approach splits into the two fingers, and the midpoint of their two medians is the finger
    # midline; the plan puts the vial axis on that midline (offset 0). (One median of all pixels leaned 13 mm to the
    # finger with more visible pixels plus a cable; the two-group midpoint matched the pads read by eye to 1.5 mm.)
    ax = np.asarray(approach, float) / np.linalg.norm(approach)
    lat_ax = np.array([-ax[1], ax[0]])
    pts = np.array([locate.to77(locate.at(float(u), float(v), fp77[2])) for u, v in zip(uu[mask][::4], vv[mask][::4])])
    s = pts @ lat_ax
    mid = float(np.median(s))
    for _ in range(10):
        left, right = s[s < mid], s[s >= mid]
        if len(left) < 20 or len(right) < 20:
            return {"msg": f"손가락 두 개를 나누지 못했어요 ({len(left)} / {len(right)} 점)"}, 1
        mid = float((np.median(left) + np.median(right)) / 2)
    Pv = locate.to77(locate.at(cv[0], cv[1], locate.CAP_Z))
    lat = float(Pv @ lat_ax) - mid
    dv = lat * lat_ax
    cf = np.array([np.median(uu[mask]), np.median(vv[mask])])
    msg = (f"병이 손가락 가운데에서 옆으로 {lat:+.1f} mm (앞뒤는 계획대로; "
           f"손가락 {cf.round(0).tolist()} px, 뚜껑 {cv.round(0).tolist()} px)")
    return {"err_mm": (-dv).round(2).tolist(), "vial_offset_mm": dv.round(2).tolist(), "msg": msg}, 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("snap")
    p.add_argument("name")
    p = sp.add_parser("measure")
    p.add_argument("--q", nargs=6, type=float, required=True)
    p.add_argument("--vial", nargs=2, type=float, required=True)
    p.add_argument("--approach", nargs=2, type=float, required=True)
    a = ap.parse_args()
    if a.cmd == "snap":
        snap(a.name)
    else:
        out, code = measure(a.q, a.vial, a.approach)
        (DIR / "last.json").write_text(json.dumps(out, ensure_ascii=False))
        print(json.dumps(out, ensure_ascii=False))
        sys.exit(code)
