"""park77.py: plan the .77 from where it stands now back to PARK (cuRobo) and print a one-phase scenario.

The console's generator step for ".77 출발 자세로": after a stopped run the arm can be anywhere (often at a
pre-grasp with the gripper hiding the vial from the top camera), and the measuring planners expect it parked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matcha_dyn  # noqa: E402
import plan77  # noqa: E402


def main() -> int:
    q = matcha_dyn.arm_q("77")
    if q is None:
        print(json.dumps({"error": ".77 팔 자세를 읽지 못했어요 (팔 서버 연결?)"}, ensure_ascii=False))
        return 1
    phase_extra = {}
    escape = []
    obs0 = matcha_dyn.WS.obstacles_for(matcha_dyn.WS.arms["77"])
    chk = plan77.cu_check(matcha_dyn.API, [q, q], obs0)
    if not chk.get("ok") and chk.get("detail", {}).get("self") and not chk.get("detail", {}).get("obstacles"):
        # 2026-11-26: after a hand-guided demo the arm was locked folded onto itself (fingers 4-5 mm into the
        # J2 / J3 spheres) and no path could start. Unfold by the smallest joint change that clears it, slowly.
        import itertools
        best = None
        for dj in itertools.product([0], [0, -10, -20], [0, 10, 20, 30], [0, 15, 30, -15], [0, 20, -20], [0]):
            if not any(dj):
                continue
            qe = q + np.array(dj, float)
            if plan77.limit_margin(qe) < 3 or not plan77.cu_check(matcha_dyn.API, [qe, qe], obs0).get("ok"):
                continue
            if best is None or np.abs(dj).sum() < np.abs(best).sum():
                best = np.array(dj, float)
        if best is None:
            print(json.dumps({"error": f"자기 접촉에서 빠져나올 자세를 찾지 못했어요: {chk.get('detail')}"}, ensure_ascii=False))
            return 1
        best = best * 2.0 if np.abs(best).max() <= 10 else best  # a little more than just clear
        escape = [q, q + best]
        print(f"park: unfold {np.round(best, 0).tolist()} out of self-contact first", file=sys.stderr)
        q = q + best
    if np.max(np.abs(q - plan77.PARK)) < 2.0:
        Q = np.vstack([q, plan77.PARK])
    else:
        obs = matcha_dyn.WS.obstacles_for(matcha_dyn.WS.arms["77"])
        legs, start, why = [], q, ""
        # A stop near the table can leave the arm inside cuRobo's margin (a pre-grasp read 0.x mm into it), and cuRobo
        # refuses such a start. Lift the tool straight up first (away from everything below it), then plan from there.
        # cuRobo does not know the vials here, and on 2026-11-26 the direct path from a pre-grasp swept the fingers at
        # vial height 30 mm past the matcha vial, so a gripper that is low (below the vial tops) always lifts 60 mm first.
        low = float(((plan77.kin.fk(q) @ plan77.GPTS.T).T[:, 2]).min()) < plan77.TABLE + 80.0
        for lift in ((60.0, 45.0, 30.0) if low else (0.0, 15.0, 30.0, 45.0)):
            if lift:
                T = plan77.kin.fk(q)
                T[2, 3] += lift
                ql, e = plan77.ik(q, T)
                if e > 0.3:
                    continue
                start = ql
            try:
                path = plan77.cu_plan(matcha_dyn.API, start, plan77.PARK, obs)
            except plan77.PlanError as exc:
                why = str(exc)
                continue
            if lift:
                legs.append(([q, start], 15.0))
                phase_extra = {"workspace_obstacles": False}  # the start itself is inside the margin
            legs.append((plan77.thin(path), 40.0))
            print(f"park: lifted {lift:.0f} mm first" if lift else "park: direct", file=sys.stderr)
            break
        else:
            print(json.dumps({"error": f"출발 자세로 가는 경로를 찾지 못했어요: {why}"}, ensure_ascii=False))
            return 1
        Q = plan77.timed(legs)
    if escape:
        E = plan77.timed([(escape, 10.0)])
        phase_extra["unchecked_start"] = len(E)
        Q = np.vstack([E, Q[1:]])
    print(f"park: {len(Q)} points from {np.round(Q[0], 1).tolist()}", file=sys.stderr)
    print(json.dumps({"trigger": ".77 출발 자세로", "title": ".77 출발 자세(PARK)로", "say": ".77을 지금 자세에서 출발 자세로 돌립니다.",
                      "seen": "", "phases": [{"arm": "77", "label": "출발 자세로", "speed_dps": 40, **phase_extra,
                                              "segments": [{"kind": "joints", "label": "출발 자세로", "q": np.round(Q, 3).tolist(), "dt": 0.05}]}]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
