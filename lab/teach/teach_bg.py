#!/usr/bin/env python3
"""Release torque, log joints at ~15 Hz until /tmp/teach_stop exists (or max 600 s), then lock all servos."""
import csv, os, sys, time
from pymycobot import MyCobot280
out = sys.argv[1]; STOP = "/tmp/teach_stop"
if os.path.exists(STOP): os.remove(STOP)
mc = MyCobot280("/dev/ttyTHS1", 1000000); time.sleep(0.5)
print("start angles", mc.get_angles(), flush=True)
mc.release_all_servos(); time.sleep(0.2); print("RELEASED", flush=True)
t0 = time.time(); n = 0
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t", "j1", "j2", "j3", "j4", "j5", "j6"])
    while not os.path.exists(STOP) and time.time() - t0 < 600:
        a = mc.get_angles()
        if isinstance(a, list) and len(a) == 6:
            w.writerow([round(time.time() - t0, 3)] + a); n += 1
            if n % 30 == 0: f.flush(); print("%.1f %s" % (time.time() - t0, a), flush=True)
        time.sleep(0.05)
for sid in range(1, 7):
    mc.focus_servo(sid); time.sleep(0.05)
print("LOCKED rows=%d final=%s" % (n, mc.get_angles()), flush=True)
