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
import kin  # noqa: E402
import locate  # noqa: E402

DIR = locate.LAB / "plans" / "align"
FINGER_POINT = np.array([13.1, 6.2, 87.3])  # flange frame: centre of the open/close difference (fit 2026-11-25)


def snap(name):
    DIR.mkdir(parents=True, exist_ok=True)
    time.sleep(0.2)
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
    w = d[mask]
    cf = np.array([(uu[mask] * w).sum() / w.sum(), (vv[mask] * w).sum() / w.sum()])
    # With the white gripper in view the camera's auto exposure / white balance shift: on 2026-11-26 the cap read
    # gray 139 and its wood ring 0, so the table-scene thresholds dropped it. Here the cap's pixel is predicted, so
    # search near it with loose thresholds, and skip anything on the pixels the fingers changed.
    moved = np.zeros_like(d, dtype=bool)
    moved[mask] = True
    near_finger = lambda c: moved[max(int(c["v"]) - 12, 0):int(c["v"]) + 13, max(int(c["u"]) - 12, 0):int(c["u"]) + 13].any()  # noqa: E731
    caps = [c for c in detect.find_caps(a, None, n=40, gray_min=100.0, wood_min=0.0)
            if np.hypot(c["u"] - pred_v[0], c["v"] - pred_v[1]) < 60 and not near_finger(c)]
    if not caps:
        return {"msg": "예상 위치 근처에서 병뚜껑을 찾지 못했어요"}, 1
    c = max(caps, key=lambda c_: c_["green"])
    cv = np.array([c["u"], c["v"]])
    Pf = locate.to77(locate.at(cf[0], cf[1], fp77[2]))
    Pv = locate.to77(locate.at(cv[0], cv[1], locate.CAP_Z))
    dvec = (Pv - Pf) - (np.asarray(vial) - fp77[:2])  # where the vial is, relative to where the hand expects it
    ax = np.asarray(approach, float) / np.linalg.norm(approach)
    along = float(dvec @ ax)
    lat = dvec - along * ax
    dv = lat + np.clip(along, -8.0, 8.0) * ax
    msg = (f"병이 예상보다 옆으로 {np.linalg.norm(lat):.1f} mm, 앞뒤로 {along:+.1f} mm "
           f"(손가락 {cf.round(0).tolist()} px, 뚜껑 {cv.round(0).tolist()} px)")
    return {"err_mm": (-dv).round(2).tolist(), "vial_offset_mm": dvec.round(2).tolist(), "msg": msg}, 0


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
