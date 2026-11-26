"""park70.py: move the .70 spoon arm from where it stands (near its rest pose) to plan70.HOME; print a one-phase scenario.

The console's generator step for ".70 기본 자세로" (e.g. after plan70.HOME changed on 2026-11-26).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matcha_dyn  # noqa: E402
import plan70  # noqa: E402
import plan77  # noqa: E402


def main() -> int:
    q = matcha_dyn.arm_q("70")
    if q is None:
        print(json.dumps({"error": ".70 팔 자세를 읽지 못했어요 (팔 서버 연결?)"}, ensure_ascii=False))
        return 1
    try:
        ph = matcha_dyn.to_rest70()
    except plan77.PlanError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    if ph is None:  # already there: a two-point hold keeps the scenario well-formed
        ph = {"arm": "70", "label": ".70 기본 자세로", "speed_dps": 30, "workspace_obstacles": False,
              "segments": [{"kind": "joints", "label": ".70 기본 자세로", "q": np.round([q, plan70.HOME], 3).tolist(), "dt": 0.05}]}
    print(f"park70: {len(ph['segments'][0]['q'])} points from {np.round(q, 1).tolist()}", file=sys.stderr)
    print(json.dumps({"trigger": ".70 기본 자세로", "title": ".70 기본 자세로", "say": ".70을 기본 자세(어깨 20도 뒤)로 옮깁니다.",
                      "seen": "", "phases": [ph]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
