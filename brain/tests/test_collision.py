"""The arm's own links as capsules (calib v3 ``robot_collision``), against each other and the scene.

The capsules here are made up in the format on the BOARD (2026-11-02): one or more capsules per link in
the chain's frames, and the link pairs to check. sim-scene's numbers come with calib v3.
"""

import math

import cabinet_scene as cs
import pytest
from test_pickplace import START, camera, sent, teleporting  # noqa: F401 (fixtures)

from mycobot_brain import calib, collision, drawers, kinematics, pickplace

# The demo (s510) pulling r0c1 open, as commanded by 2eb3e4c: the wrist camera was in the base.
DEMO_R0C1_PULL = [-36.5, -68.2, -134.8, 23.0, 103.5, -135.0]


def task_with_capsules(tmp_path, capsules=None):
    c = cs.load_sim_like_with_capsules(capsules)
    return drawers.DrawerTask(None, None, c, None, pickplace.RunLog(tmp_path / "run"), drawers.DrawerSettings())


# -- distances ---------------------------------------------------------------------


def test_segment_distance():
    d = collision.segment_distance
    assert d((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)) == pytest.approx(1.0)  # parallel
    assert d((0, 0, 0), (1, 0, 0), (0.5, -1, 1), (0.5, 1, 1)) == pytest.approx(1.0)  # crossing, 1 apart
    assert d((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)) == pytest.approx(1.0)  # in line, end to end
    assert d((0, 0, 0), (0, 0, 0), (0, 3, 4), (0, 3, 4)) == pytest.approx(5.0)  # two points
    assert d((0, 0, 0), (1, 1, 1), (0, 1, 0), (1, 0, 1)) == pytest.approx(0.0, abs=1e-12)  # they cross


def test_box_distance():
    lo, hi = (0, 0, 0), (1, 1, 1)
    d = collision.box_distance
    assert d((-1, 0.5, 0.5), (2, 0.5, 0.5), lo, hi) == 0.0  # through it
    assert d((-1, 0.5, 1.2), (2, 0.5, 1.2), lo, hi) == pytest.approx(0.2)  # over the top
    # Past a corner on the diagonal: closest at the corner (1, 1, 0.5), sqrt(2) * 0.1 away.
    assert d((1.2, 0.9, 0.5), (0.9, 1.2, 0.5), lo, hi) == pytest.approx(math.sqrt(0.02) / 2, abs=1e-6)
    assert d((5, 5, 5), (6, 6, 6), lo, hi, within=0.1) > 0.1  # far away: any value over ``within``


def test_paths_are_checked_every_2_deg_from_the_start():
    path = collision.densify([[0] * 6, [10, 0, 0, 0, 0, -7], [10, 5, 0, 0, 0, -7]])
    assert path[0] == [0] * 6 and path[-1] == [10, 5, 0, 0, 0, -7]
    assert max(max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(path, path[1:])) <= 2.0 + 1e-9
    assert [10, 0, 0, 0, 0, -7] in path


# -- the calib -----------------------------------------------------------------------


def test_chain_frames_are_the_links_of_the_sim_urdf():
    c = calib.from_dict(cs.SIM)
    q = [12.0, -30.0, -60.0, 20.0, 15.0, 40.0]
    f = c.chain.frames(q)
    assert c.chain.links == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint6_flange"]
    for name, tool in (("tcp", "tcp"), ("camera", "camera"), ("joint6_flange", "flange")):
        assert kinematics.position(f[name]) == pytest.approx(kinematics.position(c.chain.pose(q, tool)))
    assert f["joint1"] == kinematics.identity()
    assert kinematics.position(f["joint2"]) == pytest.approx([0, 0, 0.13156 + 0.026], abs=1e-9)
    # The fixed frames from the flange to the camera and the TCP are there too.
    assert kinematics.position(f["camera_link"]) == pytest.approx(kinematics.position(f["camera"]), abs=0.02)
    assert c.chain.frame_depth("joint1") == 0 and c.chain.frame_depth("joint4") == 3
    assert c.chain.frame_depth("tool_mount") == 6 == c.chain.frame_depth("tcp")


def test_robot_collision_in_the_board_format():
    c = cs.load_sim_like_with_capsules()
    body = c.robot
    assert len(body.capsules) == 7 and ("joint1", "camera_flange") in body.pairs
    assert body.fingers == {"gripper_left3", "gripper_right3"}
    assert body.still == {"joint1"}  # the base cannot move into anything


def test_robot_collision_with_several_capsules_per_link_aliases_and_default_pairs():
    caps = {
        "capsules": {
            "joint1": [{"frame": "joint1", "a_m": [0, -0.05, 0.05], "b_m": [0, 0.05, 0.05], "radius_m": 0.05},
                       {"frame": "joint1", "segment_m": [[0, 0, 0.02], [0, 0, 0.1]], "radius_m": 0.04}],
            "joint3": {"frame": "joint3", "p0_m": [-0.11, 0, 0.06], "p1_m": [0, 0, 0.06], "r_m": 0.035},
            "joint2": {"frame": "joint2", "a_m": [0, 0, 0], "b_m": [0, 0, -0.03], "radius_m": 0.03},
            "camera_flange": {"frame": "camera_mount", "a_m": [0.04, 0, 0.01], "b_m": [0, 0, 0.01],
                              "radius_m": 0.02},
        },
        "fingers": [],
    }
    body = cs.load_sim_like_with_capsules(caps).robot
    assert [c.name for c in body.capsules].count("joint1") == 2
    # No pairs given: every two links two or more joints apart (joint1-joint2 and joint2-joint3 are not).
    assert set(body.pairs) == {("joint1", "joint3"), ("joint1", "camera_flange"), ("joint3", "camera_flange"),
                               ("joint2", "camera_flange")}
    assert body.fingers == frozenset()


@pytest.mark.parametrize("bad, words", [
    ({"capsules": [{"name": "x", "frame": "elbow", "a_m": [0, 0, 0], "b_m": [0, 0, 0], "radius_m": 0.01}]},
     "frame 'elbow'"),
    ({"capsules": [{"name": "x", "frame": "joint3", "a_m": [0, 0, 0], "b_m": [0, 0, 0], "radius_m": 0.01}],
      "pairs": [["x", "joint9"]]}, "no capsule named 'joint9'"),
    ({"capsules": [{"name": "x", "frame": "joint3", "a_m": [0, 0, 0], "radius_m": 0.01}]}, "a_m and b_m"),
])
def test_a_robot_collision_the_brain_cannot_read_stops_the_load(bad, words):
    with pytest.raises(calib.CalibError, match=words):
        cs.load_sim_like_with_capsules(bad)


# -- the checks ------------------------------------------------------------------------


def test_the_demo_pull_of_r0c1_had_the_camera_in_the_base(tmp_path):
    """BOARD 2026-11-02 08:40: Hojin saw the wrist camera pass through the base in demo s510; D3 checked
    only the TCP, pads and camera against the cabinet."""
    task = task_with_capsules(tmp_path)
    c = task.calib
    placed = {p.capsule.name: p for p in c.robot.place(c.chain, DEMO_R0C1_PULL)}
    base, cam = placed["joint1"], placed["camera_flange"]
    gap = collision.segment_distance(base.a, base.b, cam.a, cam.b) - base.capsule.radius - cam.capsule.radius
    assert gap < -0.02  # 2 cm and more into the base
    assert c.robot.self_hit(list(placed.values()), 0.005).startswith("joint1")
    assert c.robot.self_hit(c.robot.place(c.chain, task.fx.cabinet_look_pose_deg), 0.005) is None
    assert "camera_flange" in task._path_clear([task.fx.cabinet_look_pose_deg, DEMO_R0C1_PULL])


def test_the_margin_is_5_mm(tmp_path):
    task = task_with_capsules(tmp_path)
    q = task.calib.look_pose_deg
    gb = next(p for p in task.calib.robot.place(task.calib.chain, q) if p.capsule.name == "gripper_base")
    end = min(gb.a, gb.b, key=lambda e: e[2])
    low = end[2] - gb.capsule.radius
    # A post right under the gripper body's lowest point, its top 6 mm and then 4 mm below it.
    lo = (end[0] - 0.01, end[1] - 0.01, 0.0)
    assert task._robot_clear(q, [("post", lo, (end[0] + 0.01, end[1] + 0.01, low - 0.006))], "all") is None
    why = task._robot_clear(q, [("post", lo, (end[0] + 0.01, end[1] + 0.01, low - 0.004))], "all")
    assert why == "gripper_base 4 mm from the post"


def test_straight_lines_are_checked_and_the_pads_meet_the_table_only_at_a_grasp(tmp_path):
    """Lines (descend, lift, pull) went unchecked in 2eb3e4c. The pads' capsules are coarser than the
    pad-tip rule that holds them 1 mm over the table at a grasp, so there they may come closer."""
    task = task_with_capsules(tmp_path)
    c = task.calib
    over = kinematics.solve_ik(c.chain, c.look_pose_deg, (0.10, 0.19, 0.08), approach=pickplace.DOWN)
    sent = []
    task._measured = lambda: list(over)
    task._send = lambda path, phase, speed: sent.append(phase)
    with pytest.raises(pickplace.PathBlocked, match="gripper_(left|right)3 .* from the table"):
        task._line((0.10, 0.19, 0.019), "descend")
    with task._near(table=True):
        task._line((0.10, 0.19, 0.019), "descend")
    assert sent == ["descend"]


def test_a_move_that_would_pass_a_pad_over_the_base_goes_round(tmp_path):
    """From r0c0's drawer look to r2c1's pre-grasp (both D3 s501 poses) the straight joint move passes the
    right pad within 4 mm of the base; by the table look it is clear."""
    task = task_with_capsules(tmp_path)
    look_r0c0, pre_r2c1 = [95.5, -56.2, -22.8, -51.1, -41.9, 12.2], [-44.71, -34.09, -93.83, -51.92, 95.05, -135.14]
    vias, how, why = task._route(look_r0c0, pre_r2c1, None)
    assert why.startswith("joint1") and "gripper_" in why
    assert vias and how != "straight"
    assert task._path_clear([look_r0c0, *vias, pre_r2c1]) is None
    sent = []
    task._measured = lambda: list(look_r0c0)
    task._send = lambda path, phase, speed: sent.append((phase, [round(a, 2) for a in path[-1]]))
    task._goto(pre_r2c1, "open r2c1: pre-grasp", floor=None)
    assert [p for p, _ in sent] == ["open r2c1: pre-grasp: detour"] * len(vias) + ["open r2c1: pre-grasp"]
    assert sent[-1][1] == pytest.approx(pre_r2c1)
    detour = [e for e in cs_events(tmp_path) if e["event"] == "detour"]
    assert detour and detour[0]["how"] == how


def test_a_move_into_the_base_has_no_way_round(tmp_path):
    task = task_with_capsules(tmp_path)
    vias, _, why = task._route(task.fx.cabinet_look_pose_deg, DEMO_R0C1_PULL, None)
    assert vias is None and "camera_flange" in why
    task._measured = lambda: list(task.fx.cabinet_look_pose_deg)
    task._send = lambda path, phase, speed: pytest.fail("sent a blocked move")
    with pytest.raises(pickplace.PathBlocked, match="no clear way"):
        task._goto(DEMO_R0C1_PULL, "open r0c1", floor=None)


def test_a_pulled_drawer_is_boxes_the_arm_keeps_off(tmp_path):
    """r0c0 fully out, as in sim-scene's calib: interior x 0.153-0.225, y 0.045-0.117, front panel 8 mm."""
    fx = cs.load_sim_like().fixture
    parts = {name: (lo, hi) for name, lo, hi in fx.tray_parts(fx.drawers["r0c0"], 0.1)}
    lo, hi = parts["r0c0 front"]
    assert lo[0] == pytest.approx(0.145) and hi[0] == pytest.approx(0.153)
    assert (lo[2], hi[2]) == pytest.approx((0.0185, 0.0535))
    lo, hi = parts["r0c0 floor"]
    assert (lo[0], hi[0]) == pytest.approx((0.153, 0.229)) and hi[2] == pytest.approx(0.023)
    walls = sorted(((lo, hi) for name, lo, hi in fx.tray_parts(fx.drawers["r0c0"], 0.1) if "side wall" in name),
                   key=lambda w: w[0][1])
    assert walls[0][1][1] == pytest.approx(0.045) and walls[1][0][1] == pytest.approx(0.117)


def cs_events(tmp_path):
    import json
    return [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]


def test_a_find_with_the_capsules_on_checks_every_move_and_still_hands_the_cone_over(
        link, camera, teleporting, tmp_path, monkeypatch):
    """The whole find (handle, look, grasp among the walls, handover on the table, push shut) with every
    joint move and line checked every 2 deg against the made-up capsules."""
    from test_drawers import SIM_OBJECTS, events, run
    monkeypatch.setattr(cs, "load", cs.load_sim_like_with_capsules)
    # r2c1 holds the cone: with these made-up capsules some other drawers have no way in within 5 mm
    # (the gripper body by the closed handle above r1c0's tray, say); calib v3's numbers decide that.
    cone = dict(SIM_OBJECTS["yellow_cone"], drawer="r2c1", xy=[0.289, 0.0])
    objects = {"yellow_cone": cone, "yellow_cube": SIM_OBJECTS["yellow_cube"]}
    result, world, _ = run(link, camera, teleporting, tmp_path, objects=objects, order=["r2c2", "r2c1"])
    assert result["ok"] is True, result
    assert result["drawers_opened"] == ["r2c2", "r2c1"]
    assert world.in_handover("yellow_cone") and world.all_shut() and not world.moved("yellow_cube")
    # What was sent, replayed every 2 deg: no listed pair of links within the margin.
    c = cs.load_sim_like_with_capsules()
    path = [m["angles_deg"] for m in sent(teleporting, ("move_joints",))]
    for q in collision.densify([START] + path):
        assert c.robot.self_hit(c.robot.place(c.chain, q), 0.005) is None, q
    assert all(e["event"] != "skip" or "no close look" not in e["why"] for e in events(tmp_path))


def test_v3_base_box_stands_in_for_the_base_capsules():
    """v3 (sim-scene 10:41): the base hull is a box; its capsules overshoot it 10-30 mm, so the box
    (exact bounds, the base never moves) replaces them in the pairs."""
    caps = dict(cs.SYNTH_CAPSULES, base_box={"name": "joint1", "frame": "joint1", "center_m": [0, 0, 0.055],
                                             "size_m": [0.11, 0.15, 0.11]})
    c = cs.load_sim_like_with_capsules(caps)
    body = c.robot
    assert "joint1" not in [cap.name for cap in body.capsules] and "joint1" in body.still
    # A pad capsule 10 mm off the box's +x face, then 3 mm off it.
    pad = collision.Capsule("gripper_left3", "joint6_flange", (0, 0, 0), (0, 0, 0), 0.01)
    for x, expect in ((0.055 + 0.02, None), (0.055 + 0.013, 0.003)):
        placed = [collision.Placed(pad, (x, 0.0, 0.05), (x, 0.0, 0.07))]
        gaps = body.pair_gaps(placed, 0.005)
        if expect is None:
            assert gaps == []
        else:
            assert gaps[0][:2] == ("joint1", "gripper_left3") and gaps[0][2] == pytest.approx(expect, abs=1e-6)
    with pytest.raises(calib.CalibError, match="not fixed to the base"):
        cs.load_sim_like_with_capsules(dict(caps, base_box=dict(caps["base_box"], frame="joint3")))


def test_the_hand_keeps_2_mm_off_the_scene_only_while_it_grasps(tmp_path):
    """Leader's rule (2026-11-02), the same in sim-scene's check: self pairs 5 mm; the scene 5 mm in free
    moves; the hand bodies (gripper body, pads, camera) 2 mm at a handle and in the drawer they grasp in."""
    task = task_with_capsules(tmp_path)
    assert task.calib.robot.hand == {"gripper_base", "camera_flange", "gripper_left3", "gripper_right3"}
    q = task.calib.look_pose_deg
    gb = next(p for p in task.calib.robot.place(task.calib.chain, q) if p.capsule.name == "gripper_base")
    end = min(gb.a, gb.b, key=lambda e: e[2])
    post = [("post", (end[0] - 0.01, end[1] - 0.01, 0.0), (end[0] + 0.01, end[1] + 0.01, end[2] - gb.capsule.radius - 0.003))]
    assert task._robot_clear(q, post, "all") == "gripper_base 3 mm from the post"
    assert task._robot_clear(q, post, "all", grasping=True) is None
    # Which moves count as grasping: at a handle, in the drawer or on the mat it grasps on; not a free move.
    seen = []
    task._robot_clear = lambda q, boxes, near, grasping=False, jaw="now": seen.append(grasping)
    path = [q, q]
    for kw, want in ((dict(), False), (dict(at_handle=True, skip="r1c1"), True), (dict(grasp_in="r1c1"), True),
                     (dict(table=True), True)):
        seen.clear()
        task._path_clear(path, **kw)
        assert set(seen) == {want}, kw


# -- the hull rule (leader 2026-11-02 12:05) ----------------------------------------------------------


def test_hull_to_box_distance_by_gjk():
    def cube(c, h):
        return [(c[0] + sx * h, c[1] + sy * h, c[2] + sz * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    lo, hi = (0, 0, 0), (1, 1, 1)
    d = collision.hull_box_distance
    assert d(cube((2.5, 0.5, 0.5), 0.5), lo, hi) == pytest.approx(1.0)
    assert d(cube((1.3, 1.4, 0.5), 0.1), lo, hi) == pytest.approx(math.hypot(0.2, 0.3))  # past an edge
    assert d(cube((1.2, 1.2, 1.2), 0.1), lo, hi) == pytest.approx(math.sqrt(3) * 0.1)  # past a corner
    assert d(cube((0.5, 0.5, 0.5), 0.1), lo, hi) == 0.0 and d(cube((1.05, 0.5, 0.5), 0.1), lo, hi) == 0.0
    # A tilted plank whose corner, not a face, is nearest: 45 deg about z, corner at x = 1.1.
    r = 0.1 / math.sqrt(2)
    plank = [(1.2 + sx * r - sy * r, 0.5 + sx * r + sy * r, 0.5 + sz * 0.05) for sx in (-1, 1) for sy in (-1, 1)
             for sz in (-1, 1)]
    assert d(plank, lo, hi) == pytest.approx(min(p[0] for p in plank) - 1.0)


def test_with_hulls_the_fixture_is_checked_on_them_at_2_mm_and_the_capsules_only_keep_the_arm_off_itself(tmp_path):
    c = cs.load_sim_like_with_hulls()
    task = drawers.DrawerTask(None, None, c, None, pickplace.RunLog(tmp_path / "run"), drawers.DrawerSettings())
    assert [h.name for h in c.robot.hulls][:3] == ["joint1", "joint3", "joint4"] and "joint1" in c.robot.still
    q = c.look_pose_deg
    t = c.chain.frames(q)["joint6_flange"]
    gb = [kinematics.apply(t, v) for v in next(h for h in c.robot.hulls if h.name == "gripper_base").vertices]
    low = min(gb, key=lambda p: p[2])
    post = lambda gap: [("post", (low[0] - 0.003, low[1] - 0.003, 0.0), (low[0] + 0.003, low[1] + 0.003, low[2] - gap))]
    assert task._robot_clear(q, post(0.0025), "all") is None
    assert task._robot_clear(q, post(0.0015), "all") == "gripper_base 1.5 mm from the post"


def test_with_hulls_the_pads_skip_only_the_handle_they_pinch(tmp_path):
    c = cs.load_sim_like_with_hulls()
    task = drawers.DrawerTask(None, None, c, None, pickplace.RunLog(tmp_path / "run"), drawers.DrawerSettings())
    seen = []
    task._robot_clear = lambda q, boxes, near, grasping=False, jaw="now": seen.append(near)
    q = c.look_pose_deg
    task._path_clear([q, q], skip="r1c1", at_handle=True)
    assert seen[-1] == {"handle of r1c1"}
    task.opening["r1c1"] = 0.1
    task._path_clear([q, q], grasp_in="r1c1")
    assert seen[-1] == {"r1c1 front", "r1c1 floor", "r1c1 side wall", "r1c1 back wall"}
    task._path_clear([q, q], table=True)
    assert seen[-1] == {"table"}


def test_a_find_with_hulls_and_capsules_hands_the_cone_over(link, camera, teleporting, tmp_path, monkeypatch):
    """The whole find on the hull rule. Without the pads' hulls: the v2 grip in this scene opens the pads
    above and below a handle, 0.3 mm from the next row's handle (why v3 grips differently); and in r2c2,
    one of the drawers whose v2 grip keeps the hand 2 mm off the table and the base."""
    from test_drawers import SIM_OBJECTS, run
    no_pads = [h for h in cs.SYNTH_HULLS if h["name"] not in ("gripper_left3", "gripper_right3")]
    monkeypatch.setattr(cs, "load", lambda: cs.load_sim_like_with_capsules(dict(cs.SYNTH_CAPSULES, hulls=no_pads)))
    cone = dict(SIM_OBJECTS["yellow_cone"], drawer="r2c2", xy=[0.289, -0.081])
    result, world, _ = run(link, camera, teleporting, tmp_path, objects={"yellow_cone": cone}, order=["r2c2"])
    assert result["ok"] is True, result
    assert world.in_handover("yellow_cone") and world.all_shut()


def test_a_pad_hull_sits_where_the_jaw_is_open_to(tmp_path):
    """v3's grips hold the jaw at preshape 56 (33.8 mm) by the handles 16 mm apart: a pad hull swept over
    the whole travel touches the next handle, the pad itself does not. A pad given shut plus its travel
    sits at the opening last commanded, swept only while that is unknown."""
    pad = {"name": "gripper_left3", "frame": "joint6_flange", "vertices_m": [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0.02]],
           "travel_axis": [-2.0, 0, 0], "travel_m": 0.0131}
    caps = dict(cs.SYNTH_CAPSULES, hulls=[pad])
    c = cs.load_sim_like_with_capsules(caps)
    (h,) = c.robot.hulls
    assert h.travel_axis == (-1.0, 0.0, 0.0) and h.travel_m == 0.0131
    shut, half, swept = h.at(0.0)[0], h.at(0.5)[0], h.at(None)[0]
    assert shut[1] == (0.01, 0.0, 0.0) and half[1] == pytest.approx((0.01 - 0.00655, 0.0, 0.0))
    assert len(swept) == 6 and min(p[0] for p in swept) == pytest.approx(-0.0131)
    task = drawers.DrawerTask(None, None, c, None, pickplace.RunLog(tmp_path / "run"), drawers.DrawerSettings())
    assert task._jaw_open is None and task._jaw_at(56) == pytest.approx(0.56) and task._jaw_at(100) == 1.0
    q = c.look_pose_deg
    t = c.chain.frames(q)["joint6_flange"]
    tip = kinematics.apply(t, (-0.0131, 0.0, 0.0))  # the vertex at (0, 0, 0), fully open
    post = [("post", tuple(x - 0.001 for x in tip), tuple(x + 0.001 for x in tip))]
    assert "gripper_left3" in task._robot_clear(q, post, (), jaw=1.0)
    assert task._robot_clear(q, post, (), jaw=0.0) is None
    assert "gripper_left3" in task._robot_clear(q, post, ())  # unknown: swept
