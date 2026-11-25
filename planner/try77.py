"""try77.py: plan the .77 matcha phase for given positions and print timing/margins (no motion)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "brain" / "src"))
import plan77
from mycobot_brain.console import workspace

import locate  # noqa: E402
WS = workspace.load(str(locate.lab_file('workspace.json')))
OBS = WS.obstacles_for(WS.arms['77'])
API = WS.curobo_api

def run(vial, cup, others, show=True):
    t0 = time.perf_counter()
    try:
        ph, info = plan77.plan_phase(vial, cup, others, API, OBS)
        if show: print("OK", vial, cup, info)
        return True, info
    except plan77.PlanError as exc:
        if show: print("FAIL", vial, cup, f"{time.perf_counter() - t0:.2f}s", str(exc)[:300])
        return False, str(exc)

if __name__ == "__main__":
    run([153.9, -166.1], [238.5, 41.3], [[203.5, -100.4, 262.5]])
