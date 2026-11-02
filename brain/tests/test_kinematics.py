import math
import random

import pytest
import scene

from mycobot_brain import kinematics as K


@pytest.fixture
def calib():
    return scene.load()


def test_all_zeros_is_the_arm_upright(calib):
    flange = K.position(calib.chain.pose([0] * 6))
    assert flange[2] > 0.40 and math.hypot(flange[0], flange[1]) < 0.09


def test_the_jacobian_matches_finite_differences(calib):
    q = [10, -25, -50, -15, 20, 30]
    t, jv, jw = calib.chain.jacobian(q, "camera")
    for k in range(6):
        dq = list(q)
        dq[k] += math.degrees(1e-6)
        moved = K.position(calib.chain.pose(dq, "camera"))
        num = [(a - b) / 1e-6 for a, b in zip(moved, K.position(t))]
        assert num == pytest.approx([jv[i][k] for i in range(3)], abs=1e-4)


def test_a_sign_flipped_joint_turns_the_other_way(calib):
    flipped = K.Chain([K.Joint(j.origin, j.axis, -1.0 if i == 0 else 1.0) for i, j in enumerate(calib.chain.joints)],
                      tools=calib.chain.tools)
    q = [20, -30, -40, -10, 0, 0]
    a = K.position(calib.chain.pose(q, "tcp"))
    b = K.position(flipped.pose([-20] + q[1:], "tcp"))
    assert a == pytest.approx(b, abs=1e-9)


def test_ik_reaches_random_downward_targets_from_the_look_pose(calib):
    rng = random.Random(4)
    for _ in range(20):
        target = (rng.uniform(0.12, 0.22), rng.uniform(-0.12, 0.08), rng.uniform(0.02, 0.12))
        q = K.solve_ik(calib.chain, calib.look_pose_deg, target)
        t = calib.chain.pose(q, "tcp")
        assert math.dist(K.position(t), target) < 5e-4
        assert K.column(t, 2)[2] < -math.cos(math.radians(0.5))
        lo_hi = K.JOINT_LIMITS_DEG
        assert all(lo + 3 <= a <= hi - 3 for a, (lo, hi) in zip(q, lo_hi))


def test_ik_holds_a_joint_at_the_seed(calib):
    q = K.solve_ik(calib.chain, calib.look_pose_deg, (0.17, -0.02, 0.18), tool="camera", hold=(6,))
    assert q[5] == calib.look_pose_deg[5]


def test_ik_out_of_reach_raises(calib):
    with pytest.raises(K.IKError, match="did not converge"):
        K.solve_ik(calib.chain, calib.look_pose_deg, (0.6, 0.0, 0.05))


def test_a_pixel_goes_back_to_the_point_it_came_from(calib):
    q = calib.look_pose_deg
    for p in (scene.BLOCK, scene.BOX, (0.19, -0.05, 0.0)):
        u, v = K.project(calib.chain, calib.camera, q, p)
        assert K.pixel_to_plane(calib.chain, calib.camera, q, u, v, p[2]) == pytest.approx(p, abs=1e-9)


def test_the_principal_point_of_a_downward_camera_is_straight_below_it(calib):
    q = K.solve_ik(calib.chain, calib.look_pose_deg, (0.18, -0.04, 0.2), tool="camera")
    k = calib.camera
    x, y, z = K.pixel_to_plane(calib.chain, k, q, k.cx, k.cy, 0.0)
    assert (x, y, z) == pytest.approx((0.18, -0.04, 0.0), abs=1e-3)


def test_a_ray_parallel_to_or_away_from_the_plane_has_no_hit():
    assert K.ray_plane((0, 0, 0.2), (1, 0, 0), 0.0) is None
    assert K.ray_plane((0, 0, 0.2), (0, 0, 1), 0.0) is None


def test_joint_path_steps_are_bounded_and_end_on_the_goal():
    path = K.joint_path([0, 0, 0, 0, 0, 0], [20, -3, 0, 0, 0, 7], 8.0)
    assert len(path) == 3 and path[-1] == [20, -3, 0, 0, 0, 7]
    prev = [0] * 6
    for q in path:
        assert max(abs(a - b) for a, b in zip(q, prev)) <= 8.0
        prev = q
    assert K.joint_path([1] * 6, [1] * 6, 8.0) == [[1] * 6]


def test_line_path_keeps_the_tcp_on_the_line(calib):
    start = K.solve_ik(calib.chain, calib.look_pose_deg, (0.2, -0.08, 0.08))
    path = K.line_path(calib.chain, start, (0.2, -0.08, 0.015), step_m=0.01, max_step=8.0)
    prev = start
    for q in path:
        x, y, _ = K.position(calib.chain.pose(q, "tcp"))
        assert math.hypot(x - 0.2, y + 0.08) < 1e-3
        assert max(abs(a - b) for a, b in zip(q, prev)) <= 8.0 + 1e-9
        prev = q
    assert K.position(calib.chain.pose(path[-1], "tcp")) == pytest.approx((0.2, -0.08, 0.015), abs=5e-4)


def test_ik_goes_round_the_base_from_a_seed_far_off_in_j1(calib):
    """Calib v3.3 puts the handover mat behind the base (bearing ~120 deg) and drawers on the right (~-40):
    from a seed at one, the damped steps stall short of the other; the retry turns J1 to the target's
    bearing first. J6 held, as a carry holds it."""
    chain = calib.chain
    seed = K.solve_ik(chain, calib.look_pose_deg, (-0.10, 0.17, 0.02), approach=(0, 0, -1))
    target = (0.10, -0.10, 0.18)
    kw = dict(tool="tcp", approach=(0, 0, -1), jaw=None, either_jaw=True, hold=(6,), limits=K.JOINT_LIMITS_DEG,
              margin=K.IK_MARGIN_DEG, tol_m=5e-4, tol_deg=0.5, iters=400, damping=0.03)
    with pytest.raises(K.IKError):
        K._solve_seeded(chain, seed, target, **kw)
    q = K.solve_ik(chain, seed, target, approach=(0, 0, -1), hold=(6,))
    assert K.position(chain.pose(q, "tcp")) == pytest.approx(target, abs=1e-3)
    assert q[5] == pytest.approx(seed[5], abs=1e-6)
    assert K.bearing_seeds(chain, seed, target, hold=(1,)) == []  # J1 held: no retry
