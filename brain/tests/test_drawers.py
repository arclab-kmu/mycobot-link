import json
import math

import cabinet_scene as cs
import pytest
from test_pickplace import START, camera, sent, teleporting  # noqa: F401 (fixtures)

from mycobot_brain import drawers, kinematics, pickplace

FAST = dict(settle=0.02, gripper_wait=0.02, arrive_timeout=3.0)
FIND = "노란색 원뿔 찾아서 꺼내 줘"
COMMANDS = {
    FIND: {"task": "find", "colour": "yellow", "shape": "cone", "drawer": "none"},
    "왼쪽 아래 서랍에 있는 물건을 꺼내 줘": {"task": "take", "colour": "any", "shape": "any", "drawer": "r0c0"},
    "파란 큐브를 가운데 서랍에 넣어": {"task": "put", "colour": "blue", "shape": "cube", "drawer": "r1c1"},
}


def _obj(colour, shape, drawer, dx, dy):
    """In ``drawer``, (dx, dy) from its floor centre, standing on the floor."""
    if drawer is None:
        return {"colour": colour, "shape": shape, "drawer": None, "xy": [dx, dy],
                "z": cs.SHAPES[shape]["height_m"] / 2}
    r, c = int(drawer[1]), int(drawer[3])
    x0 = cs.FRONT_X + 0.006 + cs.INNER[0] / 2
    return {"colour": colour, "shape": shape, "drawer": drawer, "xy": [x0 + dx, cs.COLS_Y[c] + dy],
            "z": cs.FLOORS[r] + cs.SHAPES[shape]["height_m"] / 2}


OBJECTS = {
    "red_cube": _obj("red", "cube", "r0c0", 0.015, 0.013),
    "yellow_cyl": _obj("yellow", "cylinder", "r0c0", -0.02, -0.012),
    "yellow_cone": _obj("yellow", "cone", "r1c2", -0.01, 0.007),
    "green_cone": _obj("green", "cone", "r2c1", 0.005, 0.0),
    "blue_cube": _obj("blue", "cube", None, 0.15, 0.10),
}


def events(run_dir):
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def run(link, camera, arm, tmp_path, instruction=FIND, objects=OBJECTS, slip=None, **kw):
    scene = cs.load()
    world = cs.World(scene, objects, slip)
    vlm_kw = {k: kw.pop(k) for k in ("order", "open_ok", "overrides") if k in kw}
    vlm = cs.CabinetVLM(world, tmp_path, commands=COMMANDS, **vlm_kw)
    s = drawers.DrawerSettings(**{**dict(execute=True), **FAST, **kw})
    task = drawers.DrawerTask(link, camera, scene, vlm, pickplace.RunLog(tmp_path), s)
    result = task.run(instruction)
    world.sync(tmp_path)
    return result, world, vlm


def test_find_opens_drawers_until_the_cone_then_hands_it_over_and_shuts_them(link, camera, teleporting, tmp_path):
    result, world, vlm = run(link, camera, teleporting, tmp_path, order=["r0c0", "r1c2", "r2c1"])
    assert result["ok"] is True, result
    assert result["drawers_opened"] == ["r0c0", "r1c2"] and result["found_in"] == "r1c2"
    assert world.in_handover("yellow_cone")
    assert not any(world.moved(o) for o in OBJECTS if o != "yellow_cone")
    assert world.all_shut() and result["open_after"] == []


def test_no_waypoint_jumps_and_no_move_puts_the_tcp_or_camera_in_the_cabinet(link, camera, teleporting, tmp_path):
    run(link, camera, teleporting, tmp_path, order=["r0c0", "r1c2"], max_step=8.0)
    scene = cs.load()
    fx = scene.fixture
    q = START
    for m in sent(teleporting, ("move_joints",)):
        assert max(abs(a - b) for a, b in zip(m["angles_deg"], q)) <= 8.0 + 1e-6
        q = m["angles_deg"]
        for tool in ("tcp", "camera"):
            p = kinematics.position(scene.chain.pose(q, tool))
            inside = all(fx.box_min[i] + 0.001 < p[i] < fx.box_max[i] - 0.001 for i in range(3))
            assert not inside, (tool, p)


def test_the_pull_is_a_straight_line_along_the_pull_axis_with_the_jaws_across_the_handle(link, camera, teleporting,
                                                                                          tmp_path):
    run(link, camera, teleporting, tmp_path, order=["r1c2"])
    scene = cs.load()
    pull = [e for e in events(tmp_path) if e["event"] == "move" and e["phase"] == "open r1c2"]
    assert len(pull) >= 5
    h = scene.fixture.drawers["r1c2"].handle_closed
    for e in pull:
        assert abs(e["tcp_xyz"][1] - h[1]) < 0.001 and abs(e["tcp_xyz"][2] - h[2]) < 0.001
        jaw = kinematics.column(scene.chain.pose(e["angles_deg"], "tcp"), 0)
        assert abs(jaw[0]) > 0.99  # closing along x: the bar (along y) sits across the jaws
    assert pull[-1]["tcp_xyz"][0] == pytest.approx(h[0] - cs.OPEN_M, abs=0.001)


def test_take_hands_over_the_object_in_the_named_drawer(link, camera, teleporting, tmp_path):
    result, world, _ = run(link, camera, teleporting, tmp_path, instruction="왼쪽 아래 서랍에 있는 물건을 꺼내 줘")
    assert result["ok"] is True and result["task"] == "take" and result["drawers_opened"] == ["r0c0"]
    taken = [o for o in OBJECTS if world.in_handover(o)]
    assert len(taken) == 1 and OBJECTS[taken[0]]["drawer"] == "r0c0"
    assert world.all_shut()


def test_put_takes_the_object_off_the_table_into_the_named_drawer_and_shuts_it(link, camera, teleporting, tmp_path):
    result, world, _ = run(link, camera, teleporting, tmp_path, instruction="파란 큐브를 가운데 서랍에 넣어")
    assert result["ok"] is True and result["task"] == "put" and result["drawers_opened"] == ["r1c1"]
    o = world.objects["blue_cube"]
    assert o["drawer"] == "r1c1" and not o["held"] and o["fell"] < 0.008  # set down, not dropped from high
    assert world.all_shut()
    assert not any(world.moved(k) for k in OBJECTS if k != "blue_cube")


def test_put_finds_room_beside_what_is_already_in_the_drawer(link, camera, teleporting, tmp_path):
    objects = dict(OBJECTS, purple=_obj("purple", "sphere", "r1c1", -0.025, 0.0))
    result, world, _ = run(link, camera, teleporting, tmp_path, instruction="파란 큐브를 가운데 서랍에 넣어",
                           objects=objects)
    assert result["ok"] is True
    a, b = world.pos("blue_cube"), world.pos("purple")
    assert math.hypot(a[0] - b[0], a[1] - b[1]) > 0.03


def test_find_that_finds_nothing_shuts_every_drawer_it_opened(link, camera, teleporting, tmp_path):
    objects = {k: v for k, v in OBJECTS.items() if k != "yellow_cone"}
    result, world, vlm = run(link, camera, teleporting, tmp_path, objects=objects)
    assert result["ok"] is False and "no yellow cone" in result["why"]
    assert sorted(result["drawers_opened"]) == sorted(cs.load().fixture.drawers)
    assert world.all_shut() and not any(world.moved(k) for k in objects)


def test_a_drawer_the_vlm_does_not_see_open_is_pulled_again(link, camera, teleporting, tmp_path):
    result, world, _ = run(link, camera, teleporting, tmp_path, order=["r1c2"], open_ok={"r1c2": [False]})
    assert result["ok"] is True
    assert [e["name"] for e in events(tmp_path) if e["event"] == "phase"].count("open r1c2") == 2
    assert any(e["event"] == "retry" for e in events(tmp_path))
    assert world.all_shut()


def test_a_drawer_left_open_at_the_end_is_pushed_shut_again(link, camera, teleporting, tmp_path):
    """The VLM says r1c2 is still open after the push: one more push, then it checks again."""
    shut_once = json.dumps({"seen": "one open", "open_drawers": ["r1c2"]})
    scene = cs.load()
    w = cs.World(scene, OBJECTS)
    vlm = cs.CabinetVLM(w, tmp_path / "c", commands=COMMANDS, order=["r1c2"])
    calls = {"n": 0}

    def flaky(request):
        name = request["response_format"]["json_schema"]["name"]
        if name == "cabinet" and "next" not in request["response_format"]["json_schema"]["schema"]["properties"]:
            calls["n"] += 1
            if calls["n"] == 1:
                return shut_once
        return vlm(request)

    s = drawers.DrawerSettings(execute=True, **FAST)
    result = drawers.DrawerTask(link, camera, scene, flaky, pickplace.RunLog(tmp_path / "c"), s).run(FIND)
    assert result["ok"] is True and calls["n"] == 2
    assert [e["name"] for e in events(tmp_path / "c") if e["event"] == "phase"].count("close r1c2") == 2


def test_without_execute_it_reads_the_command_and_checks_the_handles_only(link, camera, teleporting, tmp_path):
    result, world, vlm = run(link, camera, teleporting, tmp_path, execute=False)
    assert result["ok"] is None and result["command"]["task"] == "find"
    assert all(t is not None for t in result["handle_tilt_deg"].values())
    assert not sent(teleporting, ("enable", "move_joints", "gripper", "stop"))


def test_a_reply_that_does_not_parse_stops_the_arm(link, camera, teleporting, tmp_path):
    with pytest.raises(Exception, match="not JSON"):
        run(link, camera, teleporting, tmp_path, order=["r1c2"], overrides={"drawer": ["the cone is there"]})
    assert len(sent(teleporting, ("stop",))) == 1
    assert events(tmp_path)[-1]["ok"] is False


def test_a_grasp_that_closes_on_air_is_retried_after_a_new_look(link, camera, teleporting, tmp_path):
    result, world, _ = run(link, camera, teleporting, tmp_path, order=["r1c2"], slip={"yellow_cone": 1})
    assert result["ok"] is True and world.in_handover("yellow_cone")
    purposes = [e["purpose"] for e in events(tmp_path) if e["event"] == "vlm_request"]
    assert purposes.count("held") == 2 and purposes.count("drawer-r1c2") == 2


def test_put_retries_a_table_grasp_that_missed(link, camera, teleporting, tmp_path):
    result, world, _ = run(link, camera, teleporting, tmp_path, instruction="파란 큐브를 가운데 서랍에 넣어",
                           slip={"blue_cube": 1})
    assert result["ok"] is True and world.objects["blue_cube"]["drawer"] == "r1c1"
    purposes = [e["purpose"] for e in events(tmp_path) if e["event"] == "vlm_request"]
    assert purposes.count("table") == 2 and purposes.count("held") == 2
