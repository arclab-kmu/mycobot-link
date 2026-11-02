"""Cabinet geometry from the calib, the drawer prompts' parsers, and IK with the jaw fixed."""

import json
import math

import cabinet_scene as cs
import pytest

from mycobot_brain import calib, drawer_vlm, fixture, kinematics
from mycobot_brain.planner import PlannerError


def fx():
    return cs.load().fixture


def test_the_calib_reads_the_cabinet_and_needs_no_block_or_box():
    c = cs.load()
    assert c.block_edge is None and c.box_rim_z is None and c.fixture is not None
    assert sorted(c.fixture.drawers) == [f"r{r}c{k}" for r in range(3) for k in range(3)]
    assert c.fixture.pull_axis == (-1.0, 0.0, 0.0) and c.fixture.open_m == cs.OPEN_M
    with pytest.raises(calib.CalibError, match="no block"):
        _ = c.block_plane_z


def test_a_drawer_id_that_does_not_match_its_row_and_column_is_refused():
    d = cs.scene_dict()
    d["cabinet"]["drawers"][0]["id"] = "r2c2"
    with pytest.raises(calib.CalibError, match="does not match"):
        calib.from_dict(d)


def test_korean_names_follow_row_and_column():
    f = fx()
    assert f.drawers["r0c0"].name_ko == "왼쪽 아래"
    assert f.drawers["r1c1"].name_ko == "가운데"
    assert f.drawers["r2c2"].name_ko == "오른쪽 위"
    assert f.drawers["r0c1"].name_ko == "가운데 아래"
    table = f.drawer_table().splitlines()
    assert table[0].startswith("r2c0: 왼쪽 위 서랍 (top row, left column)") and len(table) == 9


def test_the_handle_and_floor_move_out_along_the_pull_axis():
    f = fx()
    d = f.drawers["r1c2"]
    assert f.handle(d, 0.05) == pytest.approx((d.handle_closed[0] - 0.05, d.handle_closed[1], d.handle_closed[2]))
    c = f.interior_center(d, f.open_m)
    assert c == pytest.approx((d.interior_center[0] - f.open_m, d.interior_center[1], d.floor_z))


def test_only_the_part_of_the_floor_out_of_the_cabinet_is_exposed():
    f = fx()
    d = f.drawers["r0c1"]
    c, half_along, half_across = f.exposed(d, f.open_m)
    # Floor runs from the front panel's inside (front + 6 mm - open_m) out to the cabinet face.
    inner_front = cs.FRONT_X + 0.006 - f.open_m
    assert c[0] - half_along == pytest.approx(inner_front)
    assert c[0] + half_along == pytest.approx(cs.FRONT_X)
    assert half_across == pytest.approx(cs.INNER[1] / 2)
    assert f.exposed(d, 0.0)[1] == 0.0  # shut: nothing out


def test_free_spot_keeps_away_from_what_is_there_and_from_the_walls():
    f = fx()
    d = f.drawers["r1c1"]
    c, ha, hb = f.exposed(d, f.open_m)
    taken = (c[0] + ha - 0.02, c[1], c[2])
    spot = f.free_spot(d, f.open_m, [taken], width=0.03)
    assert math.hypot(spot[0] - taken[0], spot[1] - taken[1]) > 0.03
    assert f.inside(d, spot, f.open_m, margin=0.015)
    assert f.free_spot(d, 0.0, [], width=0.03) is None  # shut: no room


def test_parse_command_checks_its_fields():
    f = fx()
    ok = json.dumps({"task": "find", "colour": "yellow", "shape": "cone", "drawer": "none"})
    assert drawer_vlm.parse_command(ok, f) == {"task": "find", "colour": "yellow", "shape": "cone", "drawer": None}
    with pytest.raises(PlannerError, match="needs a drawer"):
        drawer_vlm.parse_command(json.dumps({"task": "take", "colour": "any", "shape": "any", "drawer": "none"}), f)
    # D3: "노란색 원뿔 찾아서 꺼내 줘" came back as take with no drawer; a named object with none is a search.
    took = json.dumps({"task": "take", "colour": "yellow", "shape": "cone", "drawer": "none"})
    assert drawer_vlm.parse_command(took, f)["task"] == "find"
    with pytest.raises(PlannerError, match="needs an object"):
        drawer_vlm.parse_command(json.dumps({"task": "find", "colour": "any", "shape": "any", "drawer": "none"}), f)
    with pytest.raises(PlannerError, match="out of range"):
        drawer_vlm.parse_command(json.dumps({"task": "find", "colour": "pink", "shape": "cone", "drawer": "none"}), f)


def test_parse_request_lists_the_drawers_with_korean_names():
    req = drawer_vlm.build_parse_request("왼쪽 아래 서랍에 있는 물건을 꺼내 줘", fx())
    assert "r0c0: 왼쪽 아래 서랍" in req["messages"][0]["content"]
    assert req["response_format"]["json_schema"]["schema"]["properties"]["drawer"]["enum"][-1] == "none"


def test_parse_drawer_turns_the_grid_into_pixels_and_a_missing_target_into_none():
    f = fx()
    raw = json.dumps({"seen": "two", "open": True, "objects": [
        {"colour": "red", "shape": "cube", "bbox_2d": [100, 100, 200, 300]},
        {"colour": "yellow", "shape": "cone", "bbox_2d": [500, 500, 600, 700]}], "target": 1})
    out = drawer_vlm.parse_drawer(raw, f, with_target=True)
    assert out["objects"][0]["pixel"] == [96.0, 96.0] and out["target"] == 1
    out = drawer_vlm.parse_drawer(raw.replace('"target": 1', '"target": -1'), f, with_target=True)
    assert out["target"] is None
    out = drawer_vlm.parse_drawer(raw.replace('"target": 1', '"target": 3'), f, with_target=True)
    assert out["target"] is None  # an index past the list is no target
    with pytest.raises(PlannerError, match="outside"):
        drawer_vlm.parse_drawer(raw.replace("[500, 500, 600, 700]", "[1500, 500, 1600, 700]"), f, with_target=True)


def test_cabinet_parse_holds_next_to_the_candidates():
    f = fx()
    raw = json.dumps({"seen": "x", "open_drawers": ["r1c1"], "next": "r0c0"})
    assert drawer_vlm.parse_cabinet(raw, f, ["r0c0", "r2c2"]) == {"seen": "x", "open_drawers": ["r1c1"], "next": "r0c0"}
    with pytest.raises(PlannerError, match="next must be"):
        drawer_vlm.parse_cabinet(raw, f, ["r2c2"])
    assert drawer_vlm.parse_cabinet(raw.replace('"r0c0"', '"none"'), f, ["r2c2"])["next"] is None
    twice = json.dumps({"seen": "x", "open_drawers": ["r1c1", "r1c1"]})
    assert drawer_vlm.parse_cabinet(twice, f, None)["open_drawers"] == ["r1c1"]


def test_the_schemas_use_no_keys_the_vllm_grammar_rejects():
    # vLLM answered HTTP 400 "Unimplemented keys: uniqueItems" to every cabinet request in D2.
    f = fx()
    for schema in (drawer_vlm.cabinet_schema(f, None), drawer_vlm.cabinet_schema(f, ["r0c0"]),
                   drawer_vlm.drawer_schema(f, True), drawer_vlm.parse_schema(f)):
        assert "uniqueItems" not in json.dumps(schema)


def test_the_cabinet_prompt_says_where_each_handle_is_in_the_image():
    c = cs.load()
    q = c.fixture.cabinet_look_pose_deg
    handles = {d.id: drawer_vlm.grid(c.chain, c.camera, q, d.handle_closed) for d in c.fixture.drawers.values()}
    assert all(handles.values())  # the test cabinet look sees every handle
    req = drawer_vlm.build_cabinet_request(b"\xff\xd8\xff\xd9", c.fixture, handles, goal="find it",
                                           searched={"r0c0": "red cube"}, candidates=["r0c1"])
    text = req["messages"][0]["content"]
    g = handles["r2c2"]
    assert f"r2c2 (오른쪽 위 서랍): ({g[0]}, {g[1]})" in text and "r0c0: red cube" in text


def test_ik_fixes_the_jaw_axis_either_way_round():
    c = cs.load()
    for jaw in ((1, 0, 0), (-1, 0, 0), (0, 1, 0)):
        q = kinematics.solve_ik(c.chain, c.look_pose_deg, (0.2, 0.0, 0.06), approach=(0, 0, -1), jaw=jaw)
        x = kinematics.column(c.chain.pose(q, "tcp"), 0)
        assert abs(kinematics.dot(x, jaw)) > 0.999


def test_look_at_points_the_camera_at_the_target_from_the_base_side():
    c = cs.load()
    target = (0.21, 0.0, 0.02)
    q = kinematics.look_at(c.chain, c.look_pose_deg, target, distance=0.16, tilt_deg=35)
    u, v = kinematics.project(c.chain, c.camera, q, target)
    assert math.hypot(u - c.camera.cx, v - c.camera.cy) < 6  # IK stops within 0.5 deg = 4.8 px
    eye = kinematics.position(c.chain.pose(q, "camera"))
    assert eye[0] < target[0] and math.dist(eye, target) == pytest.approx(0.16, abs=0.001)


def test_fixture_errors_name_what_is_missing():
    d = cs.scene_dict()
    del d["handover"]
    with pytest.raises(calib.CalibError, match="handover"):
        calib.from_dict(d)
    assert issubclass(fixture.FixtureError, ValueError)


def test_the_loader_takes_the_sim_scene_s_keys_too():
    """Collision parts instead of a box, a cabinet-wide heading or radius/size shapes."""
    d = cs.scene_dict()
    cab = d["cabinet"]
    del cab["box_m"]
    cab["parts_m"] = {"left": {"center": [0.3, 0.1, 0.05], "size": [0.1, 0.02, 0.1]},
                      "right": {"center": [0.3, -0.1, 0.06], "size": [0.1, 0.02, 0.12]}}
    cab["pick_jaw_axis"] = [-1, 0, 0]
    cab["drawers"][0]["handle_grasp"] = {"approach": [0.94, -0.34, 0], "jaw_axis": [0, 0, 1]}
    cab["drawers"][0]["front_top_z_m"] = 0.04
    d["catalogue"]["shapes"] = {"cone": {"radius_m": 0.015, "height_m": 0.026, "grasp_tcp_height_m": 0.0205},
                                "cube": {"size_m": [0.025, 0.025, 0.025], "height_m": 0.025,
                                         "grasp_tcp_height_m": 0.0205}}
    f = calib.from_dict(d).fixture
    assert f.box_min == pytest.approx((0.25, -0.11, 0.0)) and f.box_max == pytest.approx((0.35, 0.11, 0.12))
    assert f.pick_jaw == (-1.0, 0.0, 0.0)
    a, j = f.handle_frame(f.drawers["r0c0"])
    assert a == pytest.approx((0.94 / math.hypot(0.94, 0.34), -0.34 / math.hypot(0.94, 0.34), 0)) and j == (0, 0, 1)
    assert f.handle_frame(f.drawers["r0c1"]) == (f.grasp_approach, f.grasp_jaw)
    assert f.drawers["r0c0"].wall_top == 0.04 and f.drawers["r0c1"].wall_top == pytest.approx(0.005 + cs.INNER[2])
    assert f.shapes["cone"].width == pytest.approx(0.03) and f.shapes["cone"].grasp_height == 0.0205
    assert f.shapes["cube"].width == pytest.approx(0.025 * math.sqrt(2))


def test_the_loader_reads_sim_scene_s_v2_calib():
    """The layout of sim_calib_drawers.json v2: no cabinet-wide heading, approach and jaw at each
    drawer's top level, a preshape value, the handle's centre apart from the TCP target, tray walls
    lower than the shelf clearance, the pick section, a cube's diagonal."""
    f = cs.load_sim_like().fixture
    assert f.grasp_approach == (1.0, 0.0, 0.0)  # null in the file: straight into the face
    r2c0, r1c1 = f.drawers["r2c0"], f.drawers["r1c1"]
    a, j = f.handle_frame(r2c0)
    assert a == pytest.approx((math.cos(math.radians(-20)), math.sin(math.radians(-20)), 0), abs=1e-6)
    assert j == (0.0, 0.0, 1.0) and f.handle_frame(r1c1)[0][1] > 0.6  # middle column at +40 deg
    assert f.handle_open_value == 55 and f.handle_close_value == 0
    assert r2c0.knob_closed == (0.236, 0.081, 0.116) and r2c0.handle_closed == (0.2226, 0.081, 0.116)
    assert f.knob(r2c0, f.open_m) == pytest.approx((0.136, 0.081, 0.116))
    assert r1c1.rim_z == pytest.approx(0.0935) and r1c1.wall_top == pytest.approx(0.0935)  # the front, not the shelf
    assert f.pick_jaw == pytest.approx((-1.0, 0.0, 0.0)) and f.pick_above_m == 0.057
    assert f.shapes["cube"].width == 0.0354 and f.shapes["cone"].grasp_height == 0.019
    assert f.far_m == 0.08 and f.standoff_m == 0.08
    assert f.body_box()[0] == pytest.approx((0.245 - 0.018, -0.1245, 0.0))  # handles stick out towards the robot


def test_boxed_trays_are_matched_to_drawers_by_where_they_would_be_fully_out():
    f = fx()
    c = cs.load()
    q = f.cabinet_look_pose_deg
    full = drawer_vlm.tray_boxes(f, c.chain, c.camera, q, f.open_m)
    assert all(full.values())
    b = full["r1c2"]
    nudged = [b[0] + 15, b[1] - 10, b[2] + 10, b[3] - 5]  # the model's box is never exact
    out = drawer_vlm.assign_trays([nudged, [0, 0, 40, 40]], full)
    assert out["open_drawers"] == ["r1c2"] and out["dropped"][0]["box"] == [0, 0, 40, 40]
    raw = json.dumps({"seen": "x", "open_trays": [{"bbox_2d": [b[2], b[3], b[0], b[1]]}]})  # corners swapped
    assert drawer_vlm.parse_cabinet(raw, f, None, open_boxes=full)["open_drawers"] == ["r1c2"]
    with pytest.raises(PlannerError, match="open_trays"):
        drawer_vlm.parse_cabinet(json.dumps({"seen": "x", "open_drawers": []}), f, None, open_boxes=full)


def test_pointed_handles_on_a_shut_handle_or_off_every_track_open_nothing():
    tracks = {"r2c0": ((390, 500), (340, 820)), "r1c0": ((440, 575), (390, 880)), "r0c0": ((470, 640), (440, 930))}
    out = drawer_vlm.assign_handles([(438, 578), (345, 800), (900, 100)], tracks)
    assert out["open_drawers"] == ["r2c0"]
    assert out["dropped"][0]["near"] == "a shut handle" and out["dropped"][1]["near"]["off"] > 70


def test_handle_stages_from_the_calib_or_one_from_the_old_keys():
    old = cs.load_sim_like().fixture
    (one,) = old.stages(old.drawers["r1c1"])
    assert (one.from_m, one.to_m, one.standoff_m, one.preshape, one.close) == (0.0, 0.1, 0.08, 55, 0)
    assert one.tcp_closed == old.drawers["r1c1"].handle_closed
    new = cs.load_sim_like_two_stage().fixture
    side, top = new.stages(new.drawers["r1c1"])
    assert (side.from_m, side.to_m, top.from_m, top.to_m) == (0.0, 0.05, 0.05, 0.1)
    assert top.approach == (0.0, 0.0, -1.0) and top.jaw == (0.0, 1.0, 0.0) and top.standoff_m == 0.03
    assert top.release == 55 and top.close == 0  # the cabinet's handle_grasp values
    assert new.stage_tcp(top, 0.05) == pytest.approx((0.236 - 0.05, 0.0, top.tcp_closed[2]))


def test_a_handle_stage_without_its_numbers_stops_the_load():
    d = cs.with_two_stage_handles(cs.sim_like_dict(), ids={"r0c0"})
    del d["cabinet"]["drawers"][0]["handle_stages"][1]["standoff_m"]
    with pytest.raises(calib.CalibError, match=r"r0c0 handle_stages\[1\]"):
        calib.from_dict(d)
