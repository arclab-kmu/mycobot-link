#!/usr/bin/env python3
"""Plan myCobot 280 trajectories with cuRobo and look at them - no robot needed.

For each plan this writes, under ``plans/``:

* ``<name>.json``        the ``mycobot-joint-trajectory/1`` hand-off (rad and deg)
* ``<name>.gif``         the arm moving, meshes drawn from the URDF, TCP path traced
* ``<name>.joints.png``  joint angle and speed vs time, with the limits drawn

``--video out.mp4`` also renders every successful plan, in order and in real
time, into one video (3D view + joint angles with a moving cursor; needs
``ffmpeg`` on PATH).

Goals (combine freely; they run in sequence, each starting where the last ended)::

    --demo                     four sample moves (the default when nothing else is given)
    --joints J1 .. J6          joint goal in DEGREES (pymycobot's unit)
    --pose X Y Z               TCP goal in metres, base frame, keeping the current orientation
    --start J1 .. J6           start in degrees (default: arc-mycobot's home pose)

Examples::

    python scripts/plan_demo.py
    python scripts/plan_demo.py --joints 45 -30 -60 0 0 0
    python scripts/plan_demo.py --pose 0.15 0.10 0.12 --name reach-left
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

# ----------------------------------------------------------------------------
# A small URDF forward kinematics for drawing: every link's world transform.
# cuRobo reports only tool frames; drawing needs all fourteen links.
# ----------------------------------------------------------------------------
def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
        T[:3, :3] = _rpy(*[float(v) for v in el.get("rpy", "0 0 0").split()])
    return T


def _axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    return T


class UrdfDrawer:
    def __init__(self, urdf: Path, joint_names: Sequence[str]) -> None:
        import trimesh

        root = ET.parse(urdf).getroot()
        self.joint_names = list(joint_names)
        self.joints = []
        for j in root.findall("joint"):
            axis = j.find("axis")
            self.joints.append((j.get("name"), j.get("type"), j.find("parent").get("link"),
                                j.find("child").get("link"), _origin(j.find("origin")),
                                np.array([float(v) for v in axis.get("xyz").split()]) if axis is not None else None))
        self.root_link = "joint1"
        self.meshes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for link in root.findall("link"):
            # The convex hulls build_urdf.py wrote for collision: closed, a few
            # hundred faces each, and what the planner's spheres were fitted to -
            # so the picture shows the volume the plan actually avoids with.
            mesh_el = link.find("collision/geometry/mesh")
            if mesh_el is None:
                continue
            m = trimesh.load(mesh_el.get("filename"), force="mesh")
            T = _origin(link.find("collision/origin"))
            v = (T[:3, :3] @ m.vertices.T).T + T[:3, 3]
            self.meshes[link.get("name")] = (v, np.asarray(m.faces))

    def link_transforms(self, q: Sequence[float]) -> Dict[str, np.ndarray]:
        values = dict(zip(self.joint_names, q))
        poses = {self.root_link: np.eye(4)}
        pending = list(self.joints)
        while pending:
            rest = []
            for name, jtype, parent, child, T0, axis in pending:
                if parent not in poses:
                    rest.append((name, jtype, parent, child, T0, axis))
                    continue
                T = poses[parent] @ T0
                if jtype in ("revolute", "continuous"):
                    T = T @ _axis_rot(axis, values.get(name, 0.0))
                poses[child] = T
            if len(rest) == len(pending):
                break
            pending = rest
        return poses

    def draw(self, ax, q: Sequence[float], color="#4c72b0", alpha=0.85) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for link, T in self.link_transforms(q).items():
            if link not in self.meshes:
                continue
            v, f = self.meshes[link]
            w = (T[:3, :3] @ v.T).T + T[:3, 3]
            shade = "#c44e52" if link.startswith("gripper") else "#4d4d4d" if link.startswith("camera") else color
            ax.add_collection3d(Poly3DCollection(w[f], facecolor=shade, edgecolor=(0, 0, 0, 0.08),
                                                 linewidths=0.2, alpha=alpha))


def _axes(ax, title: str) -> None:
    ax.set_xlim(-0.25, 0.3)
    ax.set_ylim(-0.3, 0.25)
    ax.set_zlim(-0.02, 0.5)
    ax.set_box_aspect((0.55, 0.55, 0.52))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title, fontsize=9)
    xx, yy = np.meshgrid([-0.25, 0.3], [-0.3, 0.25])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="#dddddd", alpha=0.3)


def render_gif(drawer: UrdfDrawer, result, path: Path, fps: int = 15, view=(25, -60)) -> None:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = max(1, int(round(1.0 / (fps * result.dt))))
    idx = list(range(0, len(result.q), step)) + [len(result.q) - 1]
    frames = []
    fig = plt.figure(figsize=(5, 5), dpi=90)
    for i in idx:
        fig.clf()
        ax = fig.add_subplot(projection="3d")
        ax.view_init(*view)
        _axes(ax, f"{result.label}   t = {result.t[i]:.2f} / {result.duration_s:.2f} s")
        drawer.draw(ax, result.q[i])
        ax.plot(*result.tcp[: i + 1].T, color="#dd8452", lw=2)
        ax.scatter(*result.tcp[-1], color="#55a868", s=30, depthshade=False)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)


def render_joints(result, limits_deg: np.ndarray, speed_cap_deg: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    qd = np.degrees(result.q)
    for j in range(6):
        axes[0].plot(result.t, qd[:, j], label=f"J{j + 1}")
        axes[1].plot(result.t, np.degrees(result.qd[:, j]))
    axes[0].set_ylabel("angle [deg]")
    axes[0].legend(ncol=6, fontsize=8)
    lo, hi = limits_deg
    span = qd.min(), qd.max()
    for j in range(6):
        for lim in (lo[j], hi[j]):
            if span[0] - 20 < lim < span[1] + 20:
                axes[0].axhline(lim, color=f"C{j}", ls=":", lw=0.8)
    axes[1].axhline(speed_cap_deg, color="k", ls="--", lw=0.8)
    axes[1].axhline(-speed_cap_deg, color="k", ls="--", lw=0.8)
    axes[1].set_ylabel("speed [deg/s]")
    axes[1].set_xlabel("t [s]")
    fig.suptitle(result.label + "   (dotted: joint limits after margin, dashed: speed cap)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def render_video(drawer: UrdfDrawer, results, path: Path, fps: int = 30) -> None:
    """All plans back to back, real time, 3D view on the left and joint angles on the right."""
    import shutil
    import subprocess
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    t_offset, frames_dir = 0.0, Path(tempfile.mkdtemp(prefix="mycobot-video-"))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    n = 0
    t_total = sum(r.duration_s for r in results)
    for k, result in enumerate(results):
        hold = int(fps * 0.5)  # half a second still at the start of each move
        times = [0.0] * hold + list(np.arange(0.0, result.duration_s, 1.0 / fps)) + [result.duration_s] * hold
        for t in times:
            i = min(int(round(t / result.dt)), len(result.q) - 1)
            fig.clf()
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.view_init(22, -55 + 25 * (t_offset + t) / max(t_total, 1e-6))  # slow orbit
            _axes(ax, "")
            drawer.draw(ax, result.q[i])
            ax.plot(*result.tcp[: i + 1].T, color="#dd8452", lw=2.5)
            ax.scatter(*result.tcp[-1], color="#55a868", s=40, depthshade=False)
            aj = fig.add_subplot(1, 2, 2)
            deg = np.degrees(result.q)
            for j in range(6):
                aj.plot(result.t, deg[:, j], color=f"C{j}", label=f"J{j + 1}")
                aj.plot(result.t[i], deg[i, j], "o", color=f"C{j}")
            aj.axvline(result.t[i], color="k", lw=0.8)
            aj.set_xlabel("t [s]")
            aj.set_ylabel("joint angle [deg]")
            aj.legend(ncol=3, fontsize=9, loc="best")
            aj.grid(alpha=0.3)
            fig.suptitle(
                f"myCobot 280 JN  |  cuRobo plan {k + 1}/{len(results)}: {result.label}  |  "
                f"t = {result.t[i]:.2f} / {result.duration_s:.2f} s\n"
                "simulation only - timing uses an unmeasured 30 deg/s cap",
                fontsize=11,
            )
            fig.savefig(frames_dir / f"{n:05d}.png")
            n += 1
        t_offset += result.duration_s
    plt.close(fig)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
        check=True,
    )
    shutil.rmtree(frames_dir)


def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=None,
                    help="cuRobo robot yml (default configs/mycobot_280_jn.yml); "
                         "configs/mycobot_280_jn_camera_gripper.yml for the camera flange + gripper")
    ap.add_argument("--start", nargs=6, type=float, metavar="J", help="start, degrees")
    ap.add_argument("--joints", nargs=6, type=float, metavar="J", action="append", help="joint goal, degrees")
    ap.add_argument("--pose", nargs=3, type=float, metavar="M", action="append", help="TCP goal x y z, metres")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--name", default=None, help="file stem for a single plan")
    ap.add_argument("--speed-fraction", type=float, default=0.25, help="of 120 deg/s (default 0.25)")
    ap.add_argument("--out", default=str(HERE / "plans"))
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--video", default=None, metavar="MP4", help="also render all plans into one video")
    args = ap.parse_args(argv)

    from curobo.logging import setup_logger

    from curobo.config_io import load_yaml

    from mycobot_planner import ARM_JOINTS, HOME_Q, ROBOT_YML, MycobotPlanner

    setup_logger("error")
    robot = Path(args.robot) if args.robot else ROBOT_YML
    planner = MycobotPlanner(robot_yml=robot, speed_fraction=args.speed_fraction)
    drawer = UrdfDrawer(Path(load_yaml(str(robot))["robot_cfg"]["kinematics"]["urdf_path"]), ARM_JOINTS)
    limits = planner.limits_deg()
    cap = math.degrees(planner.target_speed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q = np.radians(args.start) if args.start else np.array(HOME_Q)
    home_tcp = planner.fk_xyz(np.array(HOME_Q))
    goals: List[Tuple[str, str, np.ndarray]] = []
    for j in args.joints or []:
        goals.append(("joints", "joints-" + "_".join(f"{v:g}" for v in j), np.radians(j)))
    for p in args.pose or []:
        goals.append(("pose", "pose-" + "_".join(f"{v:g}" for v in p), np.array(p)))
    if args.demo or not goals:
        goals = [
            ("joints", "demo1-zero-to-home", None),
            ("pose", "demo2-home-left-10cm", home_tcp + [0.0, 0.10, 0.0]),
            ("pose", "demo3-down-5cm", home_tcp + [0.0, 0.10, -0.05]),
            ("joints", "demo4-back-home", np.array(HOME_Q)),
        ]
        q = np.zeros(6)
        goals[0] = ("joints", goals[0][1], np.array(HOME_Q))
    if args.name and len(goals) == 1:
        goals[0] = (goals[0][0], args.name, goals[0][2])

    failures = 0
    done = []
    for kind, name, goal in goals:
        if kind == "joints":
            result = planner.plan_to_joints(q, goal, label=name)
        else:
            result = planner.plan_to_pose(q, goal, label=name)
        print(f"[{name}] {result.summary()}")
        if not result.success:
            failures += 1
            continue
        stem = out / name
        result.to_json(stem.with_suffix(".json"))
        render_joints(result, limits, cap, stem.with_suffix(".joints.png"))
        if not args.no_gif:
            render_gif(drawer, result, stem.with_suffix(".gif"))
        print(f"    -> {stem}.json / .gif / .joints.png")
        done.append(result)
        q = result.q[-1]
    if args.video and done:
        render_video(drawer, done, Path(args.video))
        print(f"video -> {args.video}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
