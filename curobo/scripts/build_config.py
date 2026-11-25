#!/usr/bin/env python3
"""URDF -> cuRobo v2 robot config for the myCobot 280 JN (gripper welded open).

Adapted from ``nero_curobo/scripts/build_nero_config.py``. Differences, and why:

1. **Convex-hull collision meshes.** The vendor meshes are open shells and the
   arm links have no ``<collision>`` at all; ``build_urdf.py`` gives every link
   the convex hull of its visual, and spheres are fitted to that.
2. **Gripper links ignored against each other.** They are welded (fixed
   joints), so any contact between them is geometry, not motion.
3. **Base spheres clamped above the mounting plane.** A sphere fit to the
   base overhangs its flat bottom (the lowest reached -11 mm), and the planner's
   floor slab then reports every posture in collision. Same fix, and the same
   reason not to use cuRobo's ``clip_links``, as nero_curobo's ``CLIP_LINKS``.
4. **Retract = arc-mycobot's HOME_POSE** (elbow bent, flange forward at
   ~(0.19, -0.07, 0.18) m). A sim pose, not measured on this arm.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from curobo.config_io import load_yaml, write_yaml
from curobo.logging import setup_logger
from curobo.robot_builder import RobotBuilder

HERE = Path(__file__).resolve().parents[1]

GRIPPER_LINKS = [
    "gripper_base", "gripper_left1", "gripper_left2", "gripper_left3",
    "gripper_right1", "gripper_right2", "gripper_right3",
]
#: Welded together, and all rigidly attached to the flange. ``camera_flange``
#: exists only in the ``--camera-flange`` URDF; absent links are skipped.
WELDED_GROUP = ["joint6", "joint6_flange", "camera_flange"] + GRIPPER_LINKS

PROTRUSION_WEIGHT = 50.0  # same reasoning as nero: fewer false self-collisions
MIN_COVERAGE = 0.75
RETRY_SPHERE_COUNTS = (4, 6, 8, 12, 16)

#: (axis index, plane, buffer): keep spheres on the +axis side of the plane.
CLIP_LINKS = {"joint1": (2, 0.0, 0.002)}

DEFAULT_JOINT_POSITION = {
    "joint2_to_joint1": 0.0,
    "joint3_to_joint2": -0.5,
    "joint4_to_joint3": -1.0,
    "joint5_to_joint4": -0.6,
    "joint6_to_joint5": 0.0,
    "joint6output_to_joint6": 0.0,
}


def fmt(link, m) -> str:
    return f"  {link:<16s} {m.num_spheres:4d} {m.coverage * 100:6.1f}% {m.protrusion * 100:6.1f}%"


def clamp_spheres_to_plane(builder: RobotBuilder) -> None:
    for link, (axis, offset, buffer) in CLIP_LINKS.items():
        kept = []
        for sphere in builder.collision_spheres.get(link, []):
            distance = sphere["center"][axis] - offset
            if distance <= buffer:
                continue
            sphere["radius"] = min(sphere["radius"], distance - buffer)
            kept.append(sphere)
        print(f"  {link}: clamped above plane, kept {len(kept)} of {len(builder.collision_spheres[link])}")
        builder.collision_spheres[link] = kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=str(HERE / "urdf/mycobot_280_jn.urdf"))
    parser.add_argument("--output", default=str(HERE / "configs/mycobot_280_jn.yml"))
    parser.add_argument("--tool-frame", default="tcp")
    parser.add_argument("--sphere-density", type=float, default=1.5)
    parser.add_argument("--num-collision-samples", type=int, default=2000)
    parser.add_argument("--log-level", default="error")
    args = parser.parse_args()
    setup_logger(args.log_level)

    builder = RobotBuilder(urdf_path=args.urdf, asset_path=str(Path(args.urdf).parent),
                           tool_frames=[args.tool_frame])
    builder.fit_collision_spheres(sphere_density=args.sphere_density, use_collision_mesh=True,
                                  compute_metrics=True, protrusion_weight=PROTRUSION_WEIGHT)
    print(f"  {'link':<16s} {'n':>4s} {'cover':>7s} {'protr':>7s}")
    for link, m in builder.link_metrics.items():
        print(fmt(link, m))

    for link, m in list(builder.link_metrics.items()):
        if m.coverage >= MIN_COVERAGE:
            continue
        best = (m.coverage, builder.collision_spheres[link], m)
        for count in RETRY_SPHERE_COUNTS:
            spheres = builder.refit_link_spheres(link, num_spheres=count, use_collision_mesh=True,
                                                 compute_metrics=True, protrusion_weight=PROTRUSION_WEIGHT)
            cm = builder.link_metrics[link]
            if cm.coverage > best[0]:
                best = (cm.coverage, spheres, cm)
            if best[0] >= MIN_COVERAGE:
                break
        builder.collision_spheres[link], builder.link_metrics[link] = best[1], best[2]
        print(fmt(link, best[2]) + ("   refit ok" if best[0] >= MIN_COVERAGE else "   STILL POOR"))

    clamp_spheres_to_plane(builder)
    present = [l for l in WELDED_GROUP if l in builder.collision_spheres]
    ignore = {}
    for a, b in combinations(present, 2):
        ignore.setdefault(a, []).append(b)
    builder.compute_collision_matrix(num_samples=args.num_collision_samples, custom_ignore=ignore)
    print(f"\nspheres: {builder.num_spheres} over {len(builder.collision_link_names)} links; "
          f"ignore entries: {len(builder.collision_matrix)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    builder.save(builder.build(), str(out))

    # Same two fixes nero_curobo applies after save(): the robot_cfg wrapper,
    # and a retract pose that is not simply the middle of every range.
    data = load_yaml(str(out))
    if "robot_cfg" not in data:
        data = {"robot_cfg": data}
    cspace = data["robot_cfg"]["kinematics"]["cspace"]
    names = cspace["joint_names"]
    for name, value in DEFAULT_JOINT_POSITION.items():
        cspace["default_joint_position"][names.index(name)] = value
    write_yaml(data, str(out))
    print(f"joints : {names}\nretract: {cspace['default_joint_position']}\nwrote  : {out}")


if __name__ == "__main__":
    main()
