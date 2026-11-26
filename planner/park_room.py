"""park_room.py: a PARK for this room, the smallest change of the taught PARK that cuRobo finds clear.

  park_room.py [--write]

The taught PARK (data/templates77.json) was clear in the old room. In the 2026-11-26 exhibit room, with the board
raised to the new table height, cuRobo put the .77's joint4 8 mm into it. This searches J2..J5 changes (10 deg steps,
up to 30) for the smallest one that is clear of the workspace obstacles with joint-limit margin, and prints it.
--write saves it to $MYCOBOT_LAB/park77_q.json, which plan77 then uses as PARK (delete the file to go back).
Run it after board77.py has placed the table edge: a wrong board makes a wrong PARK.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import locate  # noqa: E402
import plan77  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    from mycobot_brain.console import workspace
    ws = workspace.load(str(locate.lab_file("workspace.json")))
    obs = ws.obstacles_for(ws.arms["77"])
    q0 = plan77.TAUGHT_PARK
    chk = plan77.cu_check(ws.curobo_api, [q0, q0], obs)
    if chk.get("ok"):
        print(f"taught PARK is clear: {q0.tolist()}")
        best = np.zeros(6)
    else:
        print(f"taught PARK: {chk.get('detail')}")
        best = None
        steps = [0, -10, 10, -20, 20, -30, 30]
        for dj in itertools.product([0], steps, steps, steps, [0, -15, 15], [0]):
            if not any(dj):
                continue
            d = np.array(dj, float)
            if best is not None and np.abs(d).sum() >= np.abs(best).sum():
                continue
            q = q0 + d
            if plan77.limit_margin(q) < 5:
                continue
            if plan77.cu_check(ws.curobo_api, [q, q], obs).get("ok"):
                best = d
        if best is None:
            print("no clear PARK within 30 deg of the taught one")
            return 1
    q = q0 + best
    T = plan77.kin.fk(q)
    print(f"PARK {np.round(q, 2).tolist()} (change {best.tolist()}), flange at planner z {T[2, 3]:.0f} "
          f"(table {plan77.TABLE:.0f})")
    if a.write and np.any(best):
        p = locate.LAB / "park77_q.json"
        p.write_text(json.dumps({"q": np.round(q, 3).tolist(), "taught": q0.tolist(), "change_deg": best.tolist(),
                                 "time": time.strftime("%Y-%m-%d %H:%M"), "_src": "park_room.py"}, indent=1))
        print(f"-> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
