#!/usr/bin/env python3
"""Keyboard end-effector teleop of the simulated myCobot 280, shown in viser.

The keys are mycobot-control's, imported from it (``mycobot_control.teleop``),
so the layout practised here is the layout ``mycobot-teleop-keyboard`` uses on
the real arm - it cannot drift. That import is the allowed direction:
mycobot-control must never import cuRobo, not the other way round.

viser has no keyboard events, so keys are read from **this terminal** (cbreak
mode, like the real teleop) while the arm is drawn in the **browser**. The
browser panel also has jog buttons for the same moves.

Each motion key, like the real teleop, builds its target from the arm's
*current* (displayed) state, never from an accumulated target:

* ``cartesian``: TCP pose + one step (translation along base axes; rotation
  about base or tool axes, composed as rotations). Solved with
  :meth:`MycobotPlanner.step_ik` - collision-free, nearest IK branch, refused
  if any joint would jump more than ``--max-jump`` degrees (default: the same
  20 deg per command that mycobot-control's ``check_command_delta`` enforces).
  Near a stretched arm or a wrist singularity the joint motion per 5 mm grows
  quickly (3.5 -> 15 deg over 35 mm of +Z from home); that is the arm, not a bug.
* ``joint``: one joint + one step, refused at the 3 deg limit margin or in collision.

The displayed arm then moves to the solution at ``--display-speed`` deg/s, so
held keys look like a jog instead of a teleport.

    w/s a/d r/f   +-X +-Y +-Z (base)       u/j i/k o/l   +-rx +-ry +-rz
    joint mode:   w e r t y u  = +J1..+J6  s d f g h j = -J1..-J6
    [ ]  step x0.5 / x2      m  cartesian <-> joint     n  rotation frame base <-> tool
    z    plan home (cuRobo)  space  stop where the arm is now     ?  help   q / Esc  quit

TODO(unverified): these Cartesian axes are the URDF's (base ``joint1``). The
firmware coords mycobot-control's cartesian mode commands are rotated 90 deg
about z from them, per the vendor's follow_display.py - so +X here is not
necessarily +X there. Joint mode is the same in both.

Usage::

    VARIANT=camera scripts/run.sh teleop     # open http://localhost:8080, keep this terminal focused
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
CONTROL_SRC = Path.home() / "manipulation/mycobot-control/src"
sys.path.insert(0, str(CONTROL_SRC))

try:
    from mycobot_control.safety.limits import (
        DEFAULT_ANGULAR_STEP_DEG,
        DEFAULT_JOINT_STEP_DEG,
        DEFAULT_LINEAR_STEP_MM,
        MAX_ANGULAR_STEP_DEG,
        MAX_COMMAND_DELTA_DEG,
        MAX_JOINT_STEP_DEG,
        MAX_LINEAR_STEP_MM,
    )
    from mycobot_control.teleop.keyboard import QUIT_KEYS, KeyReader
    from mycobot_control.teleop.keymap import CARTESIAN_BINDINGS, GRIPPER_KEYS, JOINT_BINDINGS
except ImportError as exc:  # pragma: no cover - depends on the checkout
    raise SystemExit(f"mycobot-control not importable from {CONTROL_SRC}: {exc}")

from curobo.config_io import load_yaml  # noqa: E402
from curobo.logging import setup_logger  # noqa: E402
from curobo.types import ContentPath  # noqa: E402
from curobo.viewer import ViserVisualizer  # noqa: E402

from mycobot_planner import HOME_Q, ROBOT_YML, MycobotPlanner  # noqa: E402
from plan_demo import UrdfDrawer  # noqa: E402

#: Sim-only keys. Checked against every key mycobot-control binds, below.
SIM_KEYS = {"m": "toggle cartesian / joint mode", "n": "toggle rotation frame base / tool", "z": "plan home"}
for _table in (CARTESIAN_BINDINGS, JOINT_BINDINGS, GRIPPER_KEYS):
    assert not set(SIM_KEYS) & set(_table), "sim key shadows a mycobot-control key"


class SimTeleop:
    def __init__(self, mp: MycobotPlanner, mode: str, max_jump_deg: float) -> None:
        self.mp = mp
        self.mode = mode
        self.rotation_frame = "base"
        self.linear_mm = DEFAULT_LINEAR_STEP_MM
        self.angular_deg = DEFAULT_ANGULAR_STEP_DEG
        self.joint_deg = DEFAULT_JOINT_STEP_DEG
        self.max_jump_deg = max_jump_deg
        self.limits = np.radians(mp.limits_deg())
        self.q_cmd = np.array(HOME_Q, dtype=float)
        self.q_show = self.q_cmd.copy()
        self.lock = threading.Lock()
        #: Set while a cuRobo plan (z = home) plays. Motion keys are refused
        #: meanwhile - otherwise each playback frame overwrote the jog target
        #: and the jog overwrote the playback (found by the nero-control session
        #: building the NERO version). Space cancels the playback.
        self.playing = threading.Event()
        self.cancel = threading.Event()
        self.last = "ready"
        self.accepted = 0
        self.refused = 0

    # -- helpers --------------------------------------------------------------
    def tcp(self, q):
        pose = self.mp.fk(q)
        xyz = pose.position.reshape(-1).cpu().numpy().astype(float)
        wxyz = pose.quaternion.reshape(-1).cpu().numpy().astype(float)
        return xyz, wxyz

    def _accept(self, q_new, label: str, extra: str = "") -> str:
        with self.lock:
            self.q_cmd = np.asarray(q_new, dtype=float)
        self.accepted += 1
        return f"{label:<6} ok {extra}"

    def _refuse(self, label: str, reason: str) -> str:
        self.refused += 1
        return f"{label:<6} REFUSED: {reason}"

    # -- the one entry point, for terminal keys and browser buttons alike -------
    def handle(self, key: str) -> str:
        with self.lock:
            q_now = self.q_show.copy()  # the "measured" state: where the arm is drawn now
        if key == " ":
            was_playing = self.playing.is_set()
            self.cancel.set()
            with self.lock:
                self.q_cmd = self.q_show.copy()
            return "stop: plan cancelled, holding here" if was_playing else "stop: holding the current posture"
        if self.playing.is_set() and key not in ("[", "]", "m", "n"):
            return f"{key!r} ignored: a plan is playing (space cancels it)"
        if key in ("[", "]"):
            f = 0.5 if key == "[" else 2.0
            lin, ang, jnt = self.linear_mm * f, self.angular_deg * f, self.joint_deg * f
            if not (0 < lin <= MAX_LINEAR_STEP_MM and 0 < ang <= MAX_ANGULAR_STEP_DEG and 0 < jnt <= MAX_JOINT_STEP_DEG):
                return "step unchanged: at mycobot-control's per-key cap"
            self.linear_mm, self.angular_deg, self.joint_deg = lin, ang, jnt
            return f"step: {lin:.2f} mm / {ang:.2f} deg / joint {jnt:.2f} deg"
        if key == "m":
            self.mode = "joint" if self.mode == "cartesian" else "cartesian"
            return f"mode: {self.mode}"
        if key == "n":
            self.rotation_frame = "tool" if self.rotation_frame == "base" else "base"
            return f"rotation frame: {self.rotation_frame}"
        if key == "z":
            result = self.mp.plan_to_joints(q_now, np.array(HOME_Q), label="home")
            if not result.success:
                return self._refuse("home", result.reason)
            self.cancel.clear()
            self.playing.set()
            threading.Thread(target=self._play, args=(result,), daemon=True).start()
            return f"home: {result.summary()}"
        if key in GRIPPER_KEYS:
            return "gripper: welded open in this model (nothing to move)"

        table = JOINT_BINDINGS if self.mode == "joint" else CARTESIAN_BINDINGS
        binding = table.get(key)
        if binding is None:
            return f"unbound key {key!r} in {self.mode} mode - press ? for help"

        if self.mode == "joint":
            q_new = q_now.copy()
            q_new[binding.index] += binding.sign * math.radians(self.joint_deg)
            lo, hi = self.limits[:, binding.index]
            if not lo <= q_new[binding.index] <= hi:
                return self._refuse(binding.label, "within 3 deg of the joint limit")
            if self.mp.in_collision(q_new):
                rep = self.mp.collision_report(q_new)
                what = [f"{a}-{b}" for a, b in rep["self"]] + [f"{link} in {ob}" for ob, link in rep["obstacles"]]
                return self._refuse(binding.label, "that posture collides: " + ", ".join(what[:3]))
            return self._accept(q_new, binding.label)

        xyz, wxyz = self.tcp(q_now)
        if binding.is_linear:
            xyz = xyz.copy()
            xyz[binding.index] += binding.sign * self.linear_mm / 1000.0
        else:
            axis = np.zeros(3)
            axis[binding.index - 3] = binding.sign * math.radians(self.angular_deg)
            r = Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
            d = Rotation.from_rotvec(axis)
            r = d * r if self.rotation_frame == "base" else r * d
            x, y, zq, w = r.as_quat()
            wxyz = np.array([w, x, y, zq])
        q_new, reason, jump = self.mp.step_ik(q_now, xyz, wxyz, max_jump_deg=self.max_jump_deg)
        if q_new is None:
            return self._refuse(binding.label, reason)
        return self._accept(q_new, binding.label, f"(largest joint move {jump:.1f} deg)")

    def _play(self, result) -> None:
        try:
            for q in result.q:
                if self.cancel.is_set():
                    return
                with self.lock:
                    self.q_cmd = q.copy()
                    self.q_show = q.copy()
                time.sleep(result.dt)
        finally:
            self.playing.clear()

    def advance(self, dt: float, speed_deg: float) -> None:
        """Move the displayed arm toward the commanded posture at ``speed_deg`` deg/s (all joints together)."""
        with self.lock:
            d = self.q_cmd - self.q_show
            worst = np.abs(d).max()
            step = math.radians(speed_deg) * dt
            self.q_show = self.q_cmd.copy() if worst <= step else self.q_show + d * (step / worst)


def help_text(sim: SimTeleop) -> str:
    return "\n".join([
        f"mode {sim.mode}, rotation frame {sim.rotation_frame}, step {sim.linear_mm:.1f} mm / "
        f"{sim.angular_deg:.1f} deg / joint {sim.joint_deg:.1f} deg",
        "  cartesian: w/s +-X  a/d +-Y  r/f +-Z   u/j +-rx  i/k +-ry  o/l +-rz",
        "  joint    : w e r t y u = +J1..+J6   s d f g h j = -J1..-J6",
        "  [ ] step   m mode   n rot frame   z home   space stop   ? help   q quit",
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=str(ROBOT_YML))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mode", choices=("cartesian", "joint"), default="cartesian")
    ap.add_argument("--max-jump", type=float, default=MAX_COMMAND_DELTA_DEG,
                    help="refuse IK steps moving any joint more than this (deg). Default: mycobot-control's "
                         f"per-command limit, {MAX_COMMAND_DELTA_DEG:g} deg, so a step the sim accepts is one "
                         "the real teleop would also send")
    ap.add_argument("--display-speed", type=float, default=90.0, help="how fast the drawn arm follows (deg/s)")
    args = ap.parse_args()
    setup_logger("error")

    print("loading cuRobo ...", flush=True)
    mp = MycobotPlanner(robot_yml=Path(args.robot))
    sim = SimTeleop(mp, args.mode, args.max_jump)
    urdf = Path(load_yaml(args.robot)["robot_cfg"]["kinematics"]["urdf_path"])
    drawer = UrdfDrawer(urdf, [n for n in mp.planner.joint_names])
    has_camera = "camera_link" in drawer.link_transforms(sim.q_show)

    viz = ViserVisualizer(content_path=ContentPath(robot_config_file=args.robot), connect_ip="0.0.0.0",
                          connect_port=args.port, add_control_frames=False, visualize_robot_spheres=False)
    server = viz._server
    # viser's default camera is framed for a metre-scale robot; the myCobot is 0.3 m tall.
    server.initial_camera.position = (0.42, -0.42, 0.32)
    server.initial_camera.look_at = (0.08, 0.0, 0.14)
    server.initial_camera.up = (0.0, 0.0, 1.0)
    tcp_frame = server.scene.add_frame("/tcp", axes_length=0.04, axes_radius=0.002)
    cam_frame = server.scene.add_frame("/camera_link", axes_length=0.03, axes_radius=0.0015,
                                       origin_color=(221, 132, 82), visible=has_camera)

    status = server.gui.add_markdown("**loading**")
    with server.gui.add_folder("Mode"):
        mode_dd = server.gui.add_dropdown("mode", ("cartesian", "joint"), initial_value=sim.mode)
        frame_dd = server.gui.add_dropdown("rotation frame", ("base", "tool"), initial_value="base")
    with server.gui.add_folder("Cartesian jog"):
        cart_groups = {axis: server.gui.add_button_group(axis, (f"-{axis}", f"+{axis}"))
                       for axis in ("X", "Y", "Z", "rx", "ry", "rz")}
    with server.gui.add_folder("Joint jog"):
        joint_groups = {j: server.gui.add_button_group(f"J{j}", (f"-J{j}", f"+J{j}")) for j in range(1, 7)}
    with server.gui.add_folder("Actions"):
        home_btn = server.gui.add_button("Home (plan)")
        stop_btn = server.gui.add_button("Stop")

    label_to_key = {}
    for key, b in CARTESIAN_BINDINGS.items():
        label_to_key[("cartesian", b.label)] = key
    for key, b in JOINT_BINDINGS.items():
        label_to_key[("joint", b.label)] = key
    busy = threading.Lock()

    def run_key(key: str, mode=None) -> None:
        if not busy.acquire(blocking=False):
            return
        try:
            if mode is not None and sim.mode != mode:
                sim.mode = mode
                mode_dd.value = mode
            msg = sim.handle(key)
            sim.last = msg
            print("  " + msg, flush=True)
        finally:
            busy.release()

    def cart_click(event) -> None:
        run_key(label_to_key[("cartesian", event.target.value)], "cartesian")

    def joint_click(event) -> None:
        run_key(label_to_key[("joint", event.target.value)], "joint")

    for g in cart_groups.values():
        g.on_click(cart_click)
    for g in joint_groups.values():
        g.on_click(joint_click)
    home_btn.on_click(lambda _=None: threading.Thread(target=run_key, args=("z",), daemon=True).start())
    stop_btn.on_click(lambda _=None: run_key(" "))
    mode_dd.on_update(lambda _=None: setattr(sim, "mode", mode_dd.value))
    frame_dd.on_update(lambda _=None: setattr(sim, "rotation_frame", frame_dd.value))

    stop = threading.Event()

    # The render thread must not touch the GPU. cuRobo runs CUDA graphs, and a
    # compute_kinematics() here racing an IK solve in the key thread crashed
    # the process with CUDA_ERROR_ILLEGAL_INSTRUCTION. So frames come from the
    # numpy URDF FK (plan_demo.UrdfDrawer) and joint values go to viser as
    # numpy; every cuRobo call happens in run_key(), serialised by `busy`.
    viser_names = list(viz._viser_joint_names)
    order = [list(mp.planner.joint_names).index(n) for n in viser_names]

    def render() -> None:
        period, n = 1.0 / 30.0, 0
        while not stop.is_set():
            sim.advance(period, args.display_speed)
            q = sim.q_show.copy()
            viz._viser_urdf.update_cfg(q[order])
            T = drawer.link_transforms(q)
            x, y, zq, w = Rotation.from_matrix(T["tcp"][:3, :3]).as_quat()
            tcp_frame.position, tcp_frame.wxyz = T["tcp"][:3, 3], (w, x, y, zq)
            if has_camera:
                x, y, zq, w = Rotation.from_matrix(T["camera_link"][:3, :3]).as_quat()
                cam_frame.position, cam_frame.wxyz = T["camera_link"][:3, 3], (w, x, y, zq)
            if n % 6 == 0:
                xyz = T["tcp"][:3, 3]
                rpy = Rotation.from_matrix(T["tcp"][:3, :3]).as_euler("xyz", degrees=True)
                status.content = (
                    f"**{sim.mode} mode**, rotation frame {sim.rotation_frame}  \n"
                    f"step {sim.linear_mm:.1f} mm / {sim.angular_deg:.1f} deg / joint {sim.joint_deg:.1f} deg  \n"
                    f"TCP `[{xyz[0]:+.3f} {xyz[1]:+.3f} {xyz[2]:+.3f}]` m, rpy "
                    f"`[{rpy[0]:+.0f} {rpy[1]:+.0f} {rpy[2]:+.0f}]` deg  \n"
                    f"joints `{' '.join(f'{v:+.1f}' for v in np.degrees(q))}` deg  \n"
                    f"last: {sim.last}  \n"
                    f"accepted {sim.accepted}, refused {sim.refused}  \n"
                    "*keys in the terminal, or the buttons below*"
                )
            n += 1
            time.sleep(period)

    mp.planner.warmup(enable_graph=True, num_warmup_iterations=3)
    xyz, wxyz = sim.tcp(sim.q_cmd)
    mp.step_ik(sim.q_cmd, xyz, wxyz)  # warm the jog IK path so the first key is not slow
    mp.in_collision(sim.q_cmd)  # and the collision checker joint mode uses
    threading.Thread(target=render, daemon=True).start()
    print(f"\nmyCobot sim teleop at http://localhost:{server.get_port()}  (robot: {Path(args.robot).name})")
    print(help_text(sim), flush=True)

    try:
        with KeyReader() as reader:
            while True:
                key = reader.read(0.05)
                if key is None:
                    continue
                if key in QUIT_KEYS:
                    break
                if key == "?":
                    print(help_text(sim), flush=True)
                    continue
                run_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        print(f"\nbye - {sim.accepted} steps accepted, {sim.refused} refused")


if __name__ == "__main__":
    main()
