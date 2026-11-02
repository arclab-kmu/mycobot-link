"""Brain-side D0: every drawer x object slot x catalogue shape, planned by the brain with no motion.

Leader 2026-11-02 13:10, before D3 runs again: the brain's own planner and checker (the commit under
test) run the full take chain - open (taught), look, close look, pick, held check, out of the drawer,
carry to the handover zone, back, close (taught) - and the put chain (handover zone -> drawer -> close)
for every case, against the fake arm (it teleports to every waypoint the brain sends), the fake camera
and the fake VLM that answers from a world following the run (cabinet_scene.py). Nothing moves; a case
is "planned" when the chain finishes with the object where the command wants it and every drawer shut,
"refused" when the brain's checker or IK stops it (with the reason), "other" for anything else.

Slots per drawer: the calib's object_slots_open_m (single, pair - where the resets put objects) and the
interior's four corners, inset by the object's half width + 6 mm.

    uv run python tests/d0_brain.py --calib /home/hkim/mycobot-sim/sim_calib_drawers.json --out DIR [--jobs 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SETTINGS = dict(execute=True, settle=0.02, gripper_wait=0.02, arrive_timeout=3.0)


def cases(calib_path: str):
    from mycobot_brain import calib

    d = json.loads(Path(calib_path).read_text())
    c = calib.from_dict(d)
    fx = c.fixture
    open_m, a = fx.open_m, fx.pull_axis
    out = []
    rows = {r["id"]: r for r in d["cabinet"]["drawers"]}
    for did, dr in sorted(fx.drawers.items()):
        if not dr.usable:
            continue
        row = rows[did]
        slots = {}
        s = row.get("object_slots_open_m") or {}
        if s.get("single"):
            slots["single"] = s["single"]
        for i, p in enumerate(s.get("pair") or ()):
            slots[f"pair{i}"] = p
        for shape_name, shape in sorted(fx.shapes.items()):
            # closed-drawer frame: the open slots minus the pull
            sl = {k: [p[0] - open_m * a[0], p[1] - open_m * a[1]] for k, p in slots.items()}
            along, across = fx.extent(dr)
            ax, bx = fx.axes()
            cx, cy = dr.interior_center
            ia, ib = along / 2 - shape.width / 2 - 0.006, across / 2 - shape.width / 2 - 0.006
            for sa, sb, name in ((1, 1, "corner front-left"), (1, -1, "corner front-right"),
                                 (-1, 1, "corner back-left"), (-1, -1, "corner back-right")):
                sl[name] = [cx + sa * ia * ax[0] + sb * ib * bx[0], cy + sa * ia * ax[1] + sb * ib * bx[1]]
            z = dr.floor_z + shape.height / 2
            for slot, xy in sl.items():
                out.append({"kind": "take", "drawer": did, "slot": slot, "shape": shape_name,
                            "object": {"colour": "yellow", "shape": shape_name, "drawer": did, "xy": xy, "z": z}})
            h = fx.handover_center
            out.append({"kind": "put", "drawer": did, "slot": "handover centre", "shape": shape_name,
                        "object": {"colour": "yellow", "shape": shape_name, "drawer": None, "xy": list(h),
                                   "z": c.table_z + shape.height / 2}})
    return out


def run_case(calib_path: str, case: dict, out_dir: str) -> dict:
    import cabinet_scene as cs
    from conftest import FakeArm
    from test_camera import FakeCamera

    from mycobot_brain import calib, drawers, pickplace
    from mycobot_brain.camera import CameraLink
    from mycobot_brain.link import ArmLink

    name = f"{case['kind']}_{case['drawer']}_{case['shape']}_{case['slot'].replace(' ', '-')}"
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    scene = calib.load(calib_path)
    instruction = f"{case['kind']} {case['shape']} {case['drawer']}"
    cmd = ({"task": "take", "colour": "any", "shape": "any", "drawer": case["drawer"]} if case["kind"] == "take"
           else {"task": "put", "colour": "yellow", "shape": case["shape"], "drawer": case["drawer"]})
    start = list(scene.fixture.cabinet_look_pose_deg)
    t0 = time.monotonic()
    rec = {"case": name, **{k: case[k] for k in ("kind", "drawer", "slot", "shape")}}
    with ArmLink("127.0.0.1", 0, ping_interval=0.02) as link, CameraLink("127.0.0.1", 0) as cam:
        arm = FakeArm(link.port)
        link.wait_connected(2.0)
        stop = threading.Event()

        def teleport():
            seq = 0
            while not stop.is_set():
                moves = [m for m in list(arm.received) if m["type"] == "move_joints"]
                seq += 1
                try:
                    arm.state(seq, moves[-1]["angles_deg"] if moves else start)
                except OSError:
                    return
                time.sleep(0.01)

        threading.Thread(target=teleport, daemon=True).start()
        fake_cam = FakeCamera(cam.port).stream()
        cam.wait_connected(2.0)
        try:
            world = cs.World(scene, {"obj": case["object"]})
            world.angles = start
            vlm = cs.CabinetVLM(world, run_dir, commands={instruction: cmd}, order=[case["drawer"]])
            task = drawers.DrawerTask(link, cam, scene, vlm, pickplace.RunLog(run_dir),
                                      drawers.DrawerSettings(**SETTINGS))
            try:
                result = task.run(instruction)
            except Exception as exc:  # noqa: BLE001 - the case's outcome, whatever it is
                result = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
            world.sync(run_dir)
            o = world.objects["obj"]
            where_ok = world.in_handover("obj") if case["kind"] == "take" else (
                o["drawer"] == case["drawer"] and not o["held"])
            ev = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
            rec.update(ok=result.get("ok"), why=result.get("why"), object_where_wanted=where_ok,
                       all_shut=world.all_shut(), seconds=round(time.monotonic() - t0, 1),
                       taught_checks=[e["why"] for e in ev if e["event"] == "taught_check"],
                       detours=[e.get("how") for e in ev if e["event"] == "detour"],
                       moves=sum(1 for e in ev if e["event"] == "move"))
        finally:
            stop.set()
            fake_cam.close()
            arm.close()
    why = rec.get("why") or ""
    if rec.get("ok") and rec["object_where_wanted"] and rec["all_shut"]:
        rec["outcome"] = "planned"
    elif any(w in why for w in ("PathBlocked", "no clear way", "would put", "IK", "out of reach", "no grasp")):
        rec["outcome"] = "refused"
    else:
        rec["outcome"] = "other"
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--calib", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--only", help="run only cases whose name contains this")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    todo = cases(args.calib)
    if args.only:
        todo = [c for c in todo if args.only in f"{c['kind']}_{c['drawer']}_{c['shape']}_{c['slot']}"]
    print(f"{len(todo)} cases on {args.calib}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_case, args.calib, c, str(args.out)): c for c in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001
                c = futs[f]
                r = {"case": f"{c['kind']}_{c['drawer']}_{c['shape']}_{c['slot']}", "outcome": "other",
                     "why": f"harness: {exc!r}"}
            rows.append(r)
            print(f"{len(rows)}/{len(todo)} {r['case']}: {r['outcome']} {r.get('seconds', '')}s "
                  f"{'' if r['outcome'] == 'planned' else (r.get('why') or '')[:200]}", flush=True)
    rows.sort(key=lambda r: r["case"])
    (args.out / "d0_brain.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    counts = {k: sum(1 for r in rows if r["outcome"] == k) for k in ("planned", "refused", "other")}
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
