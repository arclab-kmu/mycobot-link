"""Replay a run's commanded joint path through the brain's capsule check (a cross-check, not the audit).

Reads the ``move`` events of each run's ``events.jsonl`` in order, starts at the first, checks every
2 deg between waypoints for the listed link pairs and every moving link against the table and the
cabinet body (no handles or trays: the run's drawer openings are not replayed), and prints each
colliding stretch: phase, first and last waypoint, the worst pair and its gap. sim-scene's audit of
the recorded frames with the full hulls is the one that scores.

    uv run --project brain python tools/d3_collision_replay.py --calib CALIB [--capsules caps.json] RUN_DIR...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mycobot_brain import calib as calib_mod
from mycobot_brain import collision


def worst(c: calib_mod.Calib, q: List[float], margin: float) -> Optional[Tuple[str, float]]:
    """The closest pair or link-to-scene gap under ``margin`` at ``q``: (words, gap m)."""
    robot = c.robot
    assert robot is not None
    placed = robot.place(c.chain, q)
    best: Optional[Tuple[str, float]] = None
    for x, y, gap in robot.pair_gaps(placed, margin):
        if best is None or gap < best[1]:
            best = (f"{x}-{y}", gap)
    scene = [("table", (-9.0, -9.0, -9.0), (9.0, 9.0, c.table_z))]
    if c.fixture is not None:
        scene.append(("cabinet", c.fixture.box_min, c.fixture.box_max))
    for p in placed:
        if p.capsule.name in robot.still:
            continue
        for name, lo, hi in scene:
            if name == "table" and p.capsule.name in robot.fingers:
                continue  # the pads go to the table at a grasp
            gap = collision.box_distance(p.a, p.b, lo, hi, within=p.capsule.radius + margin) - p.capsule.radius
            if gap < margin and (best is None or gap < best[1]):
                best = (f"{p.capsule.name}-{name}", gap)
    return best


def replay(c: calib_mod.Calib, run: Path, margin: float) -> List[Dict[str, Any]]:
    moves = [e for e in map(json.loads, (run / "events.jsonl").read_text().splitlines()) if e["event"] == "move"]
    out: List[Dict[str, Any]] = []
    for prev, e in zip(moves, moves[1:]):
        for q in collision.densify([prev["angles_deg"], e["angles_deg"]])[1:]:
            hit = worst(c, q, margin)
            if hit is None:
                continue
            last = out[-1] if out else None
            if last and last["phase"] == e["phase"] and last["to"] in (e["waypoint"], e["waypoint"] - 1):
                last["to"] = e["waypoint"]
                if hit[1] < last["gap_m"]:
                    last.update(pair=hit[0], gap_m=hit[1], at=[round(a, 1) for a in q])
            else:
                out.append({"phase": e["phase"], "from": e["waypoint"], "to": e["waypoint"], "of": e["of"],
                            "pair": hit[0], "gap_m": hit[1], "at": [round(a, 1) for a in q]})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--calib", required=True, type=Path)
    p.add_argument("--capsules", type=Path, help="robot_collision JSON to use instead of the calib's")
    p.add_argument("--margin", type=float, default=0.005)
    p.add_argument("runs", nargs="+", type=Path)
    args = p.parse_args()
    d = json.loads(args.calib.read_text())
    if args.capsules:
        d["robot_collision"] = json.loads(args.capsules.read_text())
    c = calib_mod.from_dict(d, str(args.calib))
    if c.robot is None:
        raise SystemExit(f"{args.calib} has no robot_collision; pass --capsules")
    for run in args.runs:
        hits = replay(c, run, args.margin)
        print(f"{run.name}: {len(hits)} stretches under {args.margin * 1000:.0f} mm")
        for h in hits:
            print(f"  {h['phase']} wp {h['from']}-{h['to']}/{h['of']}: {h['pair']} {h['gap_m'] * 1000:.0f} mm at {h['at']}")


if __name__ == "__main__":
    main()
