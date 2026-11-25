#!/usr/bin/env python3
"""Interactive cuRobo viewer for the myCobot 280 JN, in the browser.

Adapted from ``nero_curobo/scripts/nero_viewer.py``: every action writes its
outcome - and for a failure, which stage failed (IK or trajectory) - into a
status panel. Planning goes through :class:`mycobot_planner.MycobotPlanner`,
so the speed cap, the 3 deg joint margin and the floor slab are the same as for
``plan_demo.py`` and the JSON export.

* Drag the gizmo at the gripper and press **Move** (TCP goal, orientation held).
* Or set the **joint sliders** (degrees, pymycobot's unit) and press **Plan to joints**.
* **Save plan** writes the last successful plan to ``plans/viewer-*.json``.

Usage::

    scripts/run.sh viewer          # then open http://localhost:8080
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from curobo.logging import setup_logger  # noqa: E402
from curobo.types import ContentPath, JointState  # noqa: E402
from curobo.viewer import ViserVisualizer  # noqa: E402

from mycobot_planner import ARM_JOINTS, HOME_Q, ROBOT_YML, MycobotPlanner  # noqa: E402

PLAYBACK_SPEEDUP = 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--robot", default=str(ROBOT_YML), help="cuRobo robot yml")
    ap.add_argument("--speed-fraction", type=float, default=0.25)
    args = ap.parse_args()
    setup_logger("error")

    mp = MycobotPlanner(robot_yml=Path(args.robot), speed_fraction=args.speed_fraction)
    viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=args.robot),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )
    server = viz._server
    # viser's default camera is framed for a metre-scale robot; the myCobot is 0.3 m tall.
    server.initial_camera.position = (0.42, -0.42, 0.32)
    server.initial_camera.look_at = (0.08, 0.0, 0.14)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    tool = mp.tool_frame
    limits = mp.limits_deg()

    status = server.gui.add_markdown("**Warming up the planner...**")
    with server.gui.add_folder("TCP goal"):
        target_text = server.gui.add_text("target (m)", initial_value="-", disabled=True)
        snap_button = server.gui.add_button("Snap target to TCP")
        move_button = server.gui.add_button("Move")
    with server.gui.add_folder("Joint goal (deg)"):
        sliders = [
            server.gui.add_slider(f"J{i + 1}", min=float(limits[0][i]), max=float(limits[1][i]), step=0.5,
                                  initial_value=round(math.degrees(HOME_Q[i]), 1))
            for i in range(6)
        ]
        joints_button = server.gui.add_button("Plan to joints")
    with server.gui.add_folder("Plan"):
        home_button = server.gui.add_button("Home")
        replay_button = server.gui.add_button("Replay last plan")
        save_button = server.gui.add_button("Save plan (JSON)")
        plan_text = server.gui.add_text("current joints (deg)", initial_value="-", disabled=True)

    current = np.array(HOME_Q, dtype=float)
    last = {"plan": None}
    busy = {"on": False}

    def set_status(headline: str, *lines: str) -> None:
        body = "\n".join(f"- {line}" for line in lines)
        status.content = f"**{headline}**\n\n{body}" if body else f"**{headline}**"

    def show(q) -> None:
        viz.set_joint_state(mp.js(q).squeeze(0))
        plan_text.value = " ".join(f"{v:+.1f}" for v in np.degrees(q))

    def refresh_target() -> None:
        p = viz.get_control_frame_pose()[tool].position.flatten().cpu().numpy()
        target_text.value = f"[{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]"

    def snap() -> None:
        pose = mp.fk(current)
        handle = viz._control_frames[tool]
        handle.position = pose.position.flatten().cpu().numpy()
        handle.wxyz = pose.quaternion.flatten().cpu().numpy()
        refresh_target()

    def play(result) -> None:
        nonlocal current
        for q in result.q:
            show(q)
            time.sleep(result.dt / PLAYBACK_SPEEDUP)
        current = result.q[-1].copy()
        for s, v in zip(sliders, np.degrees(current)):
            s.value = float(round(v, 1))
        snap()

    def report(result, what: str) -> None:
        if not result.success:
            hint = {
                "ik": "Rotate the gizmo back toward the gripper's current orientation, or move it closer "
                      "to the base. *Snap target to TCP* recovers a lost gizmo.",
                "trajopt": "The goal posture collides (floor, self), or the path is blocked. Try two shorter moves.",
                "cspace": "Start or goal is in collision, or within 3 deg of a joint limit.",
            }.get(result.stage, "")
            set_status(f"{what}: failed at {result.stage}", result.reason, hint)
            return
        last["plan"] = result
        set_status(f"{what}: plan found", result.summary(), "Press *Save plan* to export it.")
        play(result)

    def background(action):
        def wrapper(_=None):
            if busy["on"]:
                return

            def body():
                busy["on"] = True
                try:
                    action()
                except Exception as exc:
                    set_status(f"Error: {type(exc).__name__}", str(exc))
                finally:
                    busy["on"] = False

            threading.Thread(target=body, daemon=True).start()
        return wrapper

    def do_move() -> None:
        refresh_target()
        pose = viz.get_control_frame_pose()[tool]
        xyz = pose.position.flatten().cpu().numpy()
        quat = pose.quaternion.flatten().cpu().numpy()
        set_status("Planning...", f"TCP -> {target_text.value}")
        report(mp.plan_to_pose(current, xyz, quat, label="viewer-move"), "Move")

    def do_joints() -> None:
        goal = np.radians([s.value for s in sliders])
        set_status("Planning...", "joints -> " + " ".join(f"{s.value:+.1f}" for s in sliders))
        report(mp.plan_to_joints(current, goal, label="viewer-joints"), "Plan to joints")

    def do_home() -> None:
        report(mp.plan_to_joints(current, np.array(HOME_Q), label="viewer-home"), "Home")

    def do_replay() -> None:
        nonlocal current
        if last["plan"] is None:
            set_status("Nothing to replay yet")
            return
        current = last["plan"].q[0].copy()
        play(last["plan"])

    def do_save() -> None:
        if last["plan"] is None:
            set_status("Nothing to save yet")
            return
        path = HERE / "plans" / time.strftime(f"viewer-%Y%m%d-%H%M%S-{last['plan'].label}.json")
        last["plan"].to_json(path)
        set_status("Saved", str(path))

    move_button.on_click(background(do_move))
    joints_button.on_click(background(do_joints))
    home_button.on_click(background(do_home))
    replay_button.on_click(background(do_replay))
    save_button.on_click(background(do_save))
    snap_button.on_click(lambda _=None: snap())

    mp.planner.warmup(enable_graph=True, num_warmup_iterations=3)
    show(current)
    snap()
    set_status("Ready", "Drag the gizmo and press **Move**, or set the joint sliders and press **Plan to joints**.",
               f"speed cap {math.degrees(mp.target_speed):.0f} deg/s (unverified on hardware)")
    print(f"myCobot viewer ready at http://localhost:{server.get_port()}", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping viewer")


if __name__ == "__main__":
    main()
