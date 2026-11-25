#!/usr/bin/env python3
"""Which link pairs, obstacles or joint limits a configuration violates (MycobotPlanner.collision_report)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curobo.logging import setup_logger  # noqa: E402

from mycobot_planner import HOME_Q, ROBOT_YML, MycobotPlanner  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("q", nargs="*", type=float, help="6 joint angles in rad (default: home)")
    ap.add_argument("--deg", action="store_true", help="the angles are in degrees")
    ap.add_argument("--robot", default=str(ROBOT_YML))
    args = ap.parse_args()
    setup_logger("error")
    q = np.radians(args.q) if (args.q and args.deg) else np.array(args.q or HOME_Q, dtype=float)
    rep = MycobotPlanner(robot_yml=Path(args.robot)).collision_report(q)
    print(f"q = {np.round(np.degrees(q), 1).tolist()} deg")
    print("self-collision:" if rep["self"] else "no self-collision")
    for (a, b), v in sorted(rep["self"].items(), key=lambda kv: -kv[1]):
        print(f"  {a:<16s} {b:<16s} {v * 1000:6.1f} mm")
    print("obstacle contact:" if rep["obstacles"] else "no obstacle contact")
    for (ob, link), v in sorted(rep["obstacles"].items(), key=lambda kv: -kv[1]):
        print(f"  {ob:<10s} {link:<16s} {v * 1000:6.1f} mm")
    print(f"outside joint limits: {rep['limits']}" if rep["limits"] else "inside joint limits (3 deg margin)")
    print(f"lowest sphere bottom: {rep['lowest'][0] * 1000:.1f} mm ({rep['lowest'][1]})")


if __name__ == "__main__":
    main()
