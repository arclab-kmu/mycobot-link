"""verify.py lifted --px U V: exit 1 if a matcha cap still stands at table height near pixel (U, V)."""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect  # noqa: E402
import locate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["lifted"])
    ap.add_argument("--px", nargs=2, type=float, required=True)
    ap.add_argument("--radius", type=float, default=35.0)
    a = ap.parse_args()
    from PIL import Image
    rgb_b = urllib.request.urlopen("http://127.0.0.1:8090/color.jpg", timeout=3).read()
    dep_b = urllib.request.urlopen("http://127.0.0.1:8090/depth.png", timeout=3).read()
    rgb = np.asarray(Image.open(io.BytesIO(rgb_b)).convert("RGB")).astype(float)
    caps = detect.find_caps(rgb, None, n=8)
    zs = locate.heights(dep_b, [(c["u"], c["v"]) for c in caps]) if caps else []
    near = [(c, z) for c, z in zip(caps, zs) if np.hypot(c["u"] - a.px[0], c["v"] - a.px[1]) < a.radius and c["green"] > 3.0]
    on_table = [(c, z) for c, z in near if z is not None and z < 282.0]
    if on_table:
        c, z = on_table[0]
        print(f"말차 병이 아직 테이블에 있어요 (뚜껑 높이 {z:.0f} mm, 픽셀 {c['u']:.0f},{c['v']:.0f}): 잡기 실패로 멈춥니다")
        sys.exit(1)
    lifted = [z for _, z in near if z is not None]
    print("병이 들렸어요" + (f" (뚜껑 높이 {max(lifted):.0f} mm)" if lifted else " (제자리에 뚜껑 없음)"))


if __name__ == "__main__":
    main()
