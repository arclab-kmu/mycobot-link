"""cuRobo v2 planner for the myCobot 280 JN - the boundary mycobot-control reads.

Six joint angles in (rad, URDF order = pymycobot order), a goal in, a
time-parameterised ``(N, 6)`` joint trajectory out, with a success flag and a
reason on failure. The file hand-off is ``mycobot-joint-trajectory/1`` JSON
(:meth:`PlanResult.to_json`): mycobot-control must never import torch or
cuRobo, so only that file crosses (same rule as nero_curobo / nero-control).

What is imposed on every plan, and where the numbers come from:

* **Joint position limits** from the URDF, shrunk by ``JOINT_LIMIT_MARGIN_DEG``
  (3 deg) - the same margin mycobot-control's ``check_joint_target`` refuses
  inside, so a plan never ends where the executor would reject it.
* **Joint speed** = ``speed_fraction`` x 2.0944 rad/s (the spec-sheet 120 deg/s
  the repaired URDF carries). Default 0.25, i.e. 30 deg/s.
  TODO(unverified): no speed has been measured on this arm, and how pymycobot's
  1-100 ``speed`` maps to deg/s is undocumented. Treat the timing of a plan as
  a shape, not a promise, until mycobot-record has measured it.
* **A floor slab** under the base (top at z = -2 mm), so no plan sweeps the
  gripper through the table the arm stands on.

Self-check (GPU, no hardware)::

    source ~/manipulation/curobo/.venv/bin/activate
    python mycobot_planner.py                                        # gripper only
    python mycobot_planner.py configs/mycobot_280_jn_camera_gripper.yml  # camera flange + gripper
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from curobo.config_io import load_yaml
from curobo.logging import setup_logger
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
try:  # public path in newer cuRobo v2 builds, private in this checkout (as nero_planner.py)
    from curobo.geom import Cuboid, SceneCfg
except ImportError:
    from curobo._src.geom.types import Cuboid, SceneCfg
from curobo.types import GoalToolPose, JointState, Pose

HERE = Path(__file__).resolve().parent
ROBOT_YML = HERE / "configs/mycobot_280_jn.yml"

ARM_JOINTS = (
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
)
#: arc-mycobot HOME_POSE: elbow bent, flange forward. A sim pose, not measured.
HOME_Q = (0.0, -0.5, -1.0, -0.6, 0.0, 0.0)
URDF_MAX_SPEED = 2.0944
JOINT_LIMIT_MARGIN_DEG = 3.0
MAX_ACCEL = 4.0  # rad/s^2. TODO(unverified): nothing measured; a gentle cap.
SCHEMA = "mycobot-joint-trajectory/1"


@dataclass
class PlanResult:
    success: bool
    reason: str = ""
    stage: str = ""
    dt: float = 0.0
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    qd: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    tcp: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    solve_time_s: float = 0.0
    goal_error_mm: float = float("nan")
    goal_error_deg: float = float("nan")
    label: str = ""
    #: The cuRobo robot yml this was planned with - which end-effector stack.
    robot: str = ""

    @property
    def duration_s(self) -> float:
        return float(self.t[-1]) if len(self.t) else 0.0

    def summary(self) -> str:
        if not self.success:
            return f"FAILED at {self.stage}: {self.reason}"
        travel = np.degrees(np.abs(self.q[-1] - self.q[0])).max()
        return (
            f"{len(self.q)} points @ {self.dt * 1000:.0f} ms = {self.duration_s:.2f} s, "
            f"max travel {travel:.1f} deg, peak speed {np.degrees(np.abs(self.qd).max()):.1f} deg/s, "
            f"goal error {self.goal_error_mm:.2f} mm / {self.goal_error_deg:.2f} deg, "
            f"solve {self.solve_time_s:.3f} s"
        )

    def to_json(self, path) -> Path:
        """Write the ``mycobot-joint-trajectory/1`` hand-off. Joints in rad *and* deg."""
        if not self.success:
            raise ValueError("refusing to export a failed plan")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema": SCHEMA,
            "label": self.label,
            "robot": self.robot,
            "tool_frame": "tcp",
            "joint_names": list(ARM_JOINTS),
            "dt": self.dt,
            "t": [round(float(v), 6) for v in self.t],
            "q_rad": np.round(self.q, 6).tolist(),
            "q_deg": np.round(np.degrees(self.q), 3).tolist(),
            "tcp_m": np.round(self.tcp, 5).tolist(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "planner": "cuRobo v2 via mycobot_curobo/mycobot_planner.py",
            "limits": {"joint_margin_deg": JOINT_LIMIT_MARGIN_DEG, "max_accel_rads2": MAX_ACCEL},
            "unverified": "joint speeds / timing are not measured on hardware",
        }
        path.write_text(json.dumps(doc, indent=1))
        return path


class MycobotPlanner:
    def __init__(
        self,
        robot_yml: Path = ROBOT_YML,
        tool_frame: str = "tcp",
        speed_fraction: float = 0.25,
        floor: bool = True,
        num_ik_seeds: int = 64,
    ) -> None:
        if not 0.0 < speed_fraction <= 1.0:
            raise ValueError("speed_fraction must be in (0, 1]")
        self.tool_frame = tool_frame
        self.target_speed = URDF_MAX_SPEED * speed_fraction
        self._robot_yml = Path(robot_yml)
        self._num_ik_seeds = num_ik_seeds
        self.obstacles = []
        if floor:
            self.obstacles.append(Cuboid(name="floor", dims=[2.0, 2.0, 0.2], pose=[0, 0, -0.102, 1, 0, 0, 0]))
        self.planner = self._build()

    # -- construction -------------------------------------------------------
    def _robot_dict(self, velocity_scale: float) -> dict:
        robot = load_yaml(str(self._robot_yml))
        kin = robot["robot_cfg"]["kinematics"]
        kin["tool_frames"] = [self.tool_frame]
        kin["cspace"]["position_limit_clip"] = math.radians(JOINT_LIMIT_MARGIN_DEG)
        n = len(kin["cspace"]["joint_names"])
        kin["cspace"]["velocity_scale"] = [velocity_scale] * n
        kin["cspace"]["max_acceleration"] = [MAX_ACCEL] * n
        return robot

    def _make(self, velocity_scale: float) -> MotionPlanner:
        cfg = MotionPlannerCfg.create(
            robot=self._robot_dict(velocity_scale),
            scene_model=None,
            collision_cache={"obb": 16},
            num_ik_seeds=self._num_ik_seeds,
        )
        planner = MotionPlanner(cfg)
        if list(planner.joint_names) != list(ARM_JOINTS):
            raise RuntimeError(f"expected joints {ARM_JOINTS}, got {planner.joint_names}")
        # The checker reads the typed lists; SceneCfg(objects=...) registers nothing.
        planner.update_world(SceneCfg(cuboid=list(self.obstacles)))
        return planner

    def _build(self) -> MotionPlanner:
        # nero_curobo found this cuRobo checkout applies velocity_scale twice,
        # so try sqrt(target/urdf) first and keep whichever the model reports.
        ratio = self.target_speed / URDF_MAX_SPEED
        for scale in (math.sqrt(ratio), ratio):
            planner = self._make(scale)
            got = planner.kinematics.get_joint_limits().velocity[1].cpu().numpy()
            if np.allclose(got, self.target_speed, rtol=1e-3):
                return planner
        raise RuntimeError(f"could not impose {self.target_speed:.3f} rad/s; model reports {got.tolist()}")

    # -- helpers --------------------------------------------------------------
    @property
    def dt(self) -> float:
        return float(self.planner.trajopt_solver.config.interpolation_dt)

    def js(self, q: Sequence[float]) -> JointState:
        t = torch.tensor([list(map(float, q))], device="cuda", dtype=torch.float32)
        if t.shape[-1] != 6:
            raise ValueError(f"need 6 joint angles, got {t.shape[-1]}")
        return JointState.from_position(t, joint_names=list(ARM_JOINTS))

    def fk(self, q: Sequence[float]) -> Pose:
        return self.planner.compute_kinematics(self.js(q)).tool_poses.get_link_pose(self.tool_frame)

    def fk_xyz(self, q: Sequence[float]) -> np.ndarray:
        return self.fk(q).position.reshape(-1).cpu().numpy()

    def limits_deg(self) -> np.ndarray:
        return np.degrees(self.planner.kinematics.get_joint_limits().position.cpu().numpy())

    def _sphere_tables(self):
        """Which link owns each collision sphere, and which link pairs are ignored (from the yml)."""
        if getattr(self, "_owner", None) is None:
            kin = load_yaml(str(self._robot_yml))["robot_cfg"]["kinematics"]
            owner = []
            for link in kin["collision_link_names"]:
                owner += [link] * len(kin["collision_spheres"][link])
            ignore = kin.get("self_collision_ignore", {})
            self._owner = np.array(owner)
            n = len(owner)
            skip = np.zeros((n, n), dtype=bool)
            for i in range(n):
                a = owner[i]
                for j in range(n):
                    b = owner[j]
                    skip[i, j] = a == b or b in ignore.get(a, []) or a in ignore.get(b, [])
            self._pair_ok = ~skip & np.triu(np.ones((n, n), dtype=bool), 1)
        return self._owner, self._pair_ok

    def collision_report(self, q: Sequence[float]) -> dict:
        """Self-collision pairs, obstacle contacts and joint-limit violations at ``q``.

        Uses this planner's own kinematics for the sphere positions and does the
        geometry in numpy. Deliberately *not* cuRobo's ``RobotCollisionChecker``:
        building one in the same process as a MotionPlanner broke the IK solver
        (every later ``solve_pose`` died with CUDA_ERROR_ILLEGAL_INSTRUCTION,
        reproduced 2026-09-23), and a zero-length ``plan_cspace(q, q)`` - the
        version before that - can fail for reasons other than collision.
        Obstacles are the planner's cuboids (the floor by default).
        """
        owner, pair_ok = self._sphere_tables()
        sph = self.planner.compute_kinematics(self.js(q)).robot_spheres.reshape(-1, 4).cpu().numpy()
        if len(sph) != len(owner):
            raise RuntimeError(f"{len(sph)} spheres from kinematics, {len(owner)} in the yml")
        active = sph[:, 3] > 0
        c, r = sph[:, :3], sph[:, 3]
        pen = r[:, None] + r[None, :] - np.linalg.norm(c[:, None] - c[None, :], axis=-1)
        hit = pair_ok & (pen > 0) & active[:, None] & active[None, :]
        pairs = {}
        for i, j in zip(*np.nonzero(hit)):
            key = tuple(sorted((owner[i], owner[j])))
            pairs[key] = max(pairs.get(key, 0.0), float(pen[i, j]))
        contacts = {}
        for ob in self.obstacles:
            p = np.asarray(ob.pose[:3], dtype=float)
            w, x, y, z = ob.pose[3:]
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            local = (c - p) @ R  # sphere centres in the box frame
            half = np.asarray(ob.dims, dtype=float) / 2
            dist = np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)
            inside = np.all(np.abs(local) <= half, axis=1)
            to_face = np.min(half - np.abs(local), axis=1)  # centre inside: distance to the nearest face
            depth = np.where(inside, r + to_face, r - dist)
            k = np.nonzero(active & (depth > 0))[0]
            for i in k:
                contacts[(ob.name, owner[i])] = max(contacts.get((ob.name, owner[i]), 0.0), float(depth[i]))
        lim = self.planner.kinematics.get_joint_limits().position.cpu().numpy()
        qq = np.asarray(q, dtype=float)
        limits = [ARM_JOINTS[i] for i in range(6) if not lim[0, i] - 1e-6 <= qq[i] <= lim[1, i] + 1e-6]
        low = int(np.argmin(np.where(active, c[:, 2] - r, np.inf)))
        return {"self": pairs, "obstacles": contacts, "limits": limits,
                "lowest": (float(c[low, 2] - r[low]), str(owner[low]))}

    def in_collision(self, q: Sequence[float]) -> bool:
        """Self- or obstacle collision at ``q``, or outside the margin-shrunk joint limits."""
        rep = self.collision_report(q)
        return bool(rep["self"] or rep["obstacles"] or rep["limits"])

    # -- jogging --------------------------------------------------------------
    def step_ik(
        self,
        q_now: Sequence[float],
        xyz: Sequence[float],
        quat_wxyz: Sequence[float],
        max_jump_deg: float = 10.0,
        noise_rad: float = 0.02,
        position_tol_mm: float = 1.0,
        orientation_tol_deg: float = 1.0,
    ):
        """One jog step: the collision-free IK solution nearest ``q_now`` that reaches the pose.

        Same method as nero_curobo's ``trace_poses``: every seed starts at
        ``q_now`` plus a little noise (seed 0 exact), because cuRobo on its own
        seeds only one from the current state and the rest at random - which is
        how a 5 mm step ends up on another IK branch. Accuracy is re-measured
        with this planner's own FK, and a solution that moves any joint more
        than ``max_jump_deg`` is refused: between two nearby poses that is a
        branch change or a singularity, not a jog.

        Returns ``(q_new or None, reason, jump_deg)``.
        """
        q_prev = np.asarray(q_now, dtype=float).reshape(-1)
        goal = Pose.from_list(list(map(float, xyz)) + list(map(float, quat_wxyz)))
        goal_tool = GoalToolPose.from_poses({self.tool_frame: goal}, num_goalset=1)
        prev_js = self.js(q_prev)
        ik_solver = self.planner.ik_solver
        n_seeds = int(getattr(ik_solver.config, "num_seeds", 32))
        base = prev_js.position.reshape(1, 1, -1)
        noise = torch.randn(1, n_seeds, base.shape[-1], device=base.device, dtype=base.dtype) * noise_rad
        noise[:, 0] = 0.0
        ik = ik_solver.solve_pose(goal_tool, current_state=prev_js, seed_config=(base + noise).contiguous(),
                                  return_seeds=n_seeds)
        sol_t = ik.solution.reshape(-1, len(ARM_JOINTS))
        ok = ik.success.reshape(-1).detach().cpu().numpy().astype(bool)
        sol = sol_t.detach().cpu().numpy().astype(float)
        fk = self.planner.compute_kinematics(
            JointState.from_position(sol_t.detach().clone(), joint_names=list(ARM_JOINTS))
        ).tool_poses.get_link_pose(self.tool_frame)
        d_mm = np.linalg.norm(fk.position.reshape(-1, 3).cpu().numpy() - np.asarray(xyz, dtype=float), axis=1) * 1000
        g = np.asarray(quat_wxyz, dtype=float)
        g = g / np.linalg.norm(g)
        d_rad = 2.0 * np.arccos(np.clip(np.abs(fk.quaternion.reshape(-1, 4).cpu().numpy() @ g), 0.0, 1.0))
        lim = self.planner.kinematics.get_joint_limits().position.cpu().numpy()
        ok &= (d_mm < position_tol_mm) & (d_rad < math.radians(orientation_tol_deg))
        ok &= np.all((sol >= lim[0] - 1e-6) & (sol <= lim[1] + 1e-6), axis=1)
        if not ok.any():
            return None, (f"no collision-free IK solution within {position_tol_mm:.0f} mm / "
                          f"{orientation_tol_deg:.0f} deg inside the joint limits (closest {d_mm.min():.1f} mm)"), float("nan")
        jump = np.abs(sol - q_prev).max(axis=1)
        jump[~ok] = np.inf
        k = int(np.argmin(jump))
        jump_deg = math.degrees(jump[k])
        if jump_deg > max_jump_deg:
            joint = int(np.abs(sol[k] - q_prev).argmax())
            return None, (f"nearest IK solution jumps {jump_deg:.1f} deg on J{joint + 1} (limit {max_jump_deg:.0f}): "
                          "an IK branch change or a singular posture"), jump_deg
        return sol[k], "", jump_deg

    # -- planning -------------------------------------------------------------
    def plan_to_joints(self, q_now: Sequence[float], q_goal: Sequence[float], label: str = "") -> PlanResult:
        t0 = time.time()
        start = self.planner.kinematics.get_active_js(self.js(q_now))
        goal = self.planner.kinematics.get_active_js(self.js(q_goal))
        result = self.planner.plan_cspace(goal, start, max_attempts=3)
        if result is None or not bool(result.success.any()):
            return PlanResult(False, "no collision-free joint-space trajectory (start/goal in collision, "
                              "or goal within 3 deg of a limit)", stage="cspace", label=label)
        return self._finish(result, self.fk(q_goal), label)

    def plan_to_pose(
        self,
        q_now: Sequence[float],
        xyz: Sequence[float],
        quat_wxyz: Optional[Sequence[float]] = None,
        label: str = "",
    ) -> PlanResult:
        """Plan the TCP to ``xyz`` (m, base frame). Orientation defaults to the start's."""
        start = self.planner.kinematics.get_active_js(self.js(q_now))
        here = self.fk(q_now)
        quat = here.quaternion.reshape(-1).tolist() if quat_wxyz is None else list(quat_wxyz)
        goal_pose = Pose.from_list(list(map(float, xyz)) + list(map(float, quat)))
        goal = GoalToolPose.from_poses({self.tool_frame: goal_pose}, num_goalset=1)
        ik = self.planner.ik_solver.solve_pose(goal, return_seeds=1, current_state=start)
        if not bool(ik.success.any()):
            return PlanResult(
                False,
                f"no IK solution: closest is {ik.position_error.min().item() * 1000:.1f} mm / "
                f"{math.degrees(ik.rotation_error.min().item()):.1f} deg away",
                stage="ik", label=label,
            )
        result = self.planner.plan_pose(goal, start, use_implicit_goal=True, max_attempts=3)
        if result is None or not bool(result.success.any()):
            return PlanResult(False, "IK solved but no collision-free trajectory", stage="trajopt", label=label)
        return self._finish(result, goal_pose, label)

    def _finish(self, result, goal: Pose, label: str) -> PlanResult:
        plan = result.get_interpolated_plan()
        names = list(plan.joint_names) if plan.joint_names is not None else list(ARM_JOINTS)
        q = plan.position.reshape(-1, len(names))[:, [names.index(n) for n in ARM_JOINTS]]
        last = getattr(result, "interpolated_last_tstep", None)
        if last is not None:
            n = int(torch.as_tensor(last).reshape(-1)[0].item()) + 1
            if 0 < n < q.shape[0]:
                q = q[:n]
        tcp = self.planner.compute_kinematics(
            JointState.from_position(q.contiguous(), joint_names=list(ARM_JOINTS))
        ).tool_poses.get_link_pose(self.tool_frame).position.reshape(-1, 3).cpu().numpy()
        q = q.detach().cpu().numpy().astype(float)
        dt = self.dt
        end = self.fk(q[-1])
        return PlanResult(
            True,
            dt=dt,
            t=np.arange(len(q)) * dt,
            q=q,
            qd=np.gradient(q, dt, axis=0) if len(q) > 1 else np.zeros_like(q),
            tcp=tcp,
            solve_time_s=float(result.total_time),
            goal_error_mm=float(goal.linear_distance(end).item() * 1000),
            goal_error_deg=float(math.degrees(goal.angular_distance(end).item())),
            label=label,
            robot=self._robot_yml.name,
        )


def _selfcheck(robot_yml: Path = ROBOT_YML) -> int:
    setup_logger("error")
    p = MycobotPlanner(robot_yml=robot_yml)
    print(f"robot: {robot_yml}")
    print(f"joint speed cap {math.degrees(p.target_speed):.1f} deg/s, dt {p.dt * 1000:.0f} ms")
    print("limits (deg, after 3 deg margin):")
    for name, lo, hi in zip(ARM_JOINTS, *p.limits_deg()):
        print(f"  {name:<24s} {lo:+7.1f} .. {hi:+7.1f}")
    home = np.array(HOME_Q)
    flange = p.planner.compute_kinematics(p.js(home)).tool_poses
    print(f"home TCP  {np.round(p.fk_xyz(home), 4)} m")
    print(f"home in collision: {p.in_collision(home)}")
    failures = 0
    tcp0 = p.fk_xyz(home)
    for label, fn in [
        ("zero -> home", lambda: p.plan_to_joints([0.0] * 6, home, "zero-to-home")),
        ("home +10 cm y", lambda: p.plan_to_pose(home, tcp0 + [0.0, 0.10, 0.0], label="home-y+10cm")),
        ("home -5 cm z", lambda: p.plan_to_pose(home, tcp0 + [0.0, 0.0, -0.05], label="home-z-5cm")),
        ("home J1 +60 deg", lambda: p.plan_to_joints(home, home + [math.radians(60), 0, 0, 0, 0, 0], "j1+60")),
    ]:
        r = fn()
        print(f"[{label}] {r.summary()}")
        failures += not r.success
    del flange
    return failures


if __name__ == "__main__":
    import sys

    raise SystemExit(_selfcheck(Path(sys.argv[1]) if len(sys.argv) > 1 else ROBOT_YML))
