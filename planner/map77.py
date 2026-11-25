"""map77.py out.json: where the .77 can grasp the vial (flat grasp) and pour into the cup (140 deg), on a 15 mm grid."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from try77 import plan77, OBS, API

res = {"vial": [], "cup": []}
t0 = time.time()
for x in range(90, 261, 20):
    for y in range(-260, 21, 20):
        sc = plan77.Scene([600, 600], 286, 40, [])
        obs = OBS
        try:
            g = plan77.plan_grasp([x, y], sc, API, obs, max_tries=40); res["vial"].append([x, y, 1, g["pitch"], g["spin"], g["shallow"]])
        except plan77.PlanError:
            res["vial"].append([x, y, 0])
for x in range(140, 321, 20):
    for y in range(-80, 161, 20):
        sc = plan77.Scene([x, y], 286, 40, [])
        obs = OBS + plan77.cup_boxes([x, y], 286)
        try:
            p = plan77.plan_pour([x, y], sc, API, obs, max_tries=25); res["cup"].append([x, y, 1, p["spin"], p["roll"]])
        except plan77.PlanError:
            res["cup"].append([x, y, 0])
res["sec"] = round(time.time() - t0, 1)
json.dump(res, open(sys.argv[1], "w"))
print("vial ok", sum(r[2] for r in res["vial"]), "/", len(res["vial"]), "cup ok", sum(r[2] for r in res["cup"]), "/", len(res["cup"]), "sec", res["sec"])
