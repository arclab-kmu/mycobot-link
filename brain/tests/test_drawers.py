import json
import math

import cabinet_scene as cs
import pytest
from test_pickplace import START, camera, sent, teleporting  # noqa: F401 (fixtures)

from mycobot_brain import calib, drawer_vlm, drawers, kinematics, pickplace

FAST = dict(settle=0.02, gripper_wait=0.02, arrive_timeout=3.0)
FIND = "노란색 원뿔 찾아서 꺼내 줘"
COMMANDS = {
    FIND: {"task": "find", "colour": "yellow", "shape": "cone", "drawer": "none"},
    "왼쪽 아래 서랍에 있는 물건을 꺼내 줘": {"task": "take", "colour": "any", "shape": "any", "drawer": "r0c0"},
    "오른쪽 가운데 서랍에 있는 물건을 꺼내 줘": {"task": "take", "colour": "any", "shape": "any", "drawer": "r1c2"},
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


def test_a_handle_listed_as_an_object_is_left_out_and_a_real_one_taken(link, camera, teleporting, tmp_path):
    """D2: the model boxed a top-row drawer's own handle as a "purple cube" and could name it the one."""
    scene = cs.load()
    w = cs.World(scene, OBJECTS)
    vlm = cs.CabinetVLM(w, tmp_path, commands=COMMANDS)

    def handle_too(request):
        raw = vlm(request)
        if request["response_format"]["json_schema"]["name"] != "drawer":
            return raw
        r = json.loads(raw)
        g = drawer_vlm.grid(scene.chain, scene.camera, w.angles, scene.fixture.knob(scene.fixture.drawer("r0c0"),
                                                                                   scene.fixture.open_m))
        assert g is not None
        r["objects"].insert(0, {"colour": "purple", "shape": "cube", "bbox_2d": [g[0] - 30, g[1] - 30, g[0] + 30, g[1] + 30]})
        r["target"] = 0
        return json.dumps(r)

    s = drawers.DrawerSettings(execute=True, **FAST)
    result = drawers.DrawerTask(link, camera, scene, handle_too, pickplace.RunLog(tmp_path), s).run(
        "왼쪽 아래 서랍에 있는 물건을 꺼내 줘")
    w.sync(tmp_path)
    assert result["ok"] is True, result
    assert [o for o in OBJECTS if w.in_handover(o)] and w.all_shut()
    dropped = [e for e in events(tmp_path) if e["event"] == "estimate" and e.get("kept") is False]
    assert [e["label"] for e in dropped] == ["purple cube"]


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


def test_a_drawer_neither_view_sees_open_is_pulled_again(link, camera, teleporting, tmp_path):
    first = json.dumps({"seen": "all shut", "open_trays": [], "next": "r1c2"})
    shut = json.dumps({"seen": "all shut", "open_trays": []})
    result, world, _ = run(link, camera, teleporting, tmp_path, order=["r1c2"], open_ok={"r1c2": [False]},
                           overrides={"cabinet": [first, shut]})
    assert result["ok"] is True
    assert [e["name"] for e in events(tmp_path) if e["event"] == "phase"].count("open r1c2") == 2
    assert any(e["event"] == "retry" for e in events(tmp_path))
    assert world.all_shut()


def test_an_open_empty_drawer_the_drawer_view_calls_shut_is_not_pulled_again(link, camera, teleporting, tmp_path):
    """D3 s500: r0c1, open and empty, came back "closed, I cannot see inside"; the code re-pulled from
    shut through the open tray. The cabinet view (tray boxes) says it is out: on to the next drawer."""
    objects = {k: v for k, v in OBJECTS.items() if v["drawer"] != "r0c1"}
    result, world, _ = run(link, camera, teleporting, tmp_path, objects=objects, order=["r0c1", "r1c2"],
                           open_ok={"r0c1": [False]})
    assert result["ok"] is True and result["drawers_opened"] == ["r0c1", "r1c2"]
    ev = events(tmp_path)
    assert [e["name"] for e in ev if e["event"] == "phase"].count("open r0c1") == 1
    assert [e["out"] for e in ev if e["event"] == "open_check"] == [True]
    assert world.in_handover("yellow_cone") and world.all_shut()


def test_a_drawer_left_open_at_the_end_is_pushed_shut_again(link, camera, teleporting, tmp_path):
    """The VLM says r1c2 is still open after the push: one more push, then it checks again."""
    scene = cs.load()
    fx = scene.fixture
    box = drawer_vlm.tray_boxes(fx, scene.chain, scene.camera, fx.cabinet_look_pose_deg, fx.open_m)["r1c2"]
    shut_once = json.dumps({"seen": "one open", "open_trays": [{"bbox_2d": box}]})
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


SIM_OBJECTS = {  # sim-scene's pair slots, 17.5 mm either side of the bay's centre line
    "yellow_cone": {"colour": "yellow", "shape": "cone", "drawer": "r2c0", "xy": [0.289, 0.0985], "z": 0.103 + 0.013},
    "orange_cone": {"colour": "orange", "shape": "cone", "drawer": "r2c0", "xy": [0.289, 0.0635], "z": 0.103 + 0.013},
    "yellow_cube": {"colour": "yellow", "shape": "cube", "drawer": "r2c2", "xy": [0.289, -0.081], "z": 0.103 + 0.0125},
    "blue_sphere": {"colour": "blue", "shape": "sphere", "drawer": "r0c0", "xy": [0.289, 0.081], "z": 0.023 + 0.013},
}


def test_on_sim_scene_numbers_the_knob_is_pinched_from_the_side_and_the_cone_found(link, camera, teleporting,
                                                                                    tmp_path, monkeypatch):
    """sim-scene's v2 calib: knob gripped with the tool level and the jaw vertical, the jaw only
    preshaped to 55 around it, a 100 mm pull, the cone in the top-left drawer beside an orange one."""
    monkeypatch.setattr(cs, "load", cs.load_sim_like)
    result, world, vlm = run(link, camera, teleporting, tmp_path, objects=SIM_OBJECTS, order=["r2c2", "r2c0"])
    assert result["ok"] is True, result
    assert result["drawers_opened"] == ["r2c2", "r2c0"]
    assert world.in_handover("yellow_cone") and world.all_shut()
    assert not any(world.moved(o) for o in SIM_OBJECTS if o != "yellow_cone")
    ev = events(tmp_path)
    phase, at_knob = "", []
    for e in ev:
        phase = e["name"] if e["event"] == "phase" else phase
        if e["event"] == "gripper" and phase.startswith(("open r", "close r")):
            at_knob.append(e["value"])
    assert at_knob and set(at_knob) == {55, 0}  # never 100 at a knob: it hits the handles above and below
    scene = cs.load_sim_like()
    pull = [e for e in ev if e["event"] == "move" and e["phase"] == "open r2c0"]
    assert pull
    for e in pull:
        jaw = kinematics.column(scene.chain.pose(e["angles_deg"], "tcp"), 0)
        assert abs(jaw[2]) > 0.99  # jaw vertical the whole pull: the pads stay on the knob's top and bottom


def test_on_sim_scene_numbers_a_take_starts_its_close_look_from_a_low_side_view(link, camera, teleporting,
                                                                                  tmp_path, monkeypatch):
    """D3 s451: the calib's look into r1c2 holds the TCP at z 0.112, under the close look's floor
    (wall top + 2 cm, 0.114); the path up from there was refused as dipping below it."""
    monkeypatch.setattr(cs, "load", cs.load_sim_like)
    objects = dict(SIM_OBJECTS, red_cone={"colour": "red", "shape": "cone", "drawer": "r1c2",
                                          "xy": [0.289, -0.081], "z": 0.063 + 0.013})
    result, world, _ = run(link, camera, teleporting, tmp_path, instruction="오른쪽 가운데 서랍에 있는 물건을 꺼내 줘",
                           objects=objects)
    assert result["ok"] is True, result
    assert world.in_handover("red_cone") and world.all_shut()


def test_joint_moves_keep_the_pad_tips_off_the_handles(tmp_path):
    """D3 s500: backed off 4 cm from r0c1's handle, the joint move up to the cabinet look caught the
    handle above (r1c1 came out 8 mm) and stalled. The cabinet now counts its handles and the pad
    tips are checked; the approach and back-off stand 8 cm off (sim-scene's far_m)."""
    scene = cs.load_sim_like()
    task = drawers.DrawerTask(None, None, scene, None, pickplace.RunLog(tmp_path), drawers.DrawerSettings())
    back_off_4cm = [-27.71, -65.94, -97.77, -16.48, 112.29, -135.08]  # measured in s500
    path = [back_off_4cm] + kinematics.joint_path(back_off_4cm, scene.fixture.cabinet_look_pose_deg, 8.0)
    why = task._path_clear(path)
    assert ("pad tips" in why or "a finger" in why) and task._path_clear(path, at_handle=True) is None
    assert scene.fixture.standoff_m == 0.08
    for d in scene.fixture.drawers.values():
        task._handle_plan(d)  # every handle: look <-> standoff, shut and open, clear of the others


def test_on_sim_scene_numbers_a_cone_is_gripped_on_its_skirt_and_carried_without_a_detour(
        link, camera, teleporting, tmp_path, monkeypatch):
    """D3: from r2c2 the carry once went round by the cabinet look at full speed (s501, e8eadbb), and
    gripped 1.5 mm above the catalogue's skirt height three cones slid out on a long carry (788cb77)."""
    monkeypatch.setattr(cs, "load", cs.load_sim_like)
    objects = {"yellow_cone": dict(SIM_OBJECTS["yellow_cone"], drawer="r2c2", xy=[0.289, -0.0985])}
    result, world, _ = run(link, camera, teleporting, tmp_path, objects=objects, order=["r2c2"])
    assert result["ok"] is True, result
    ev = events(tmp_path)
    assert not any(e["event"] == "detour" for e in ev)
    assert not any(e["event"] == "carry" and e["how"] == "detour" for e in ev)
    grasp = [e for e in ev if e["event"] == "move" and e["phase"] == "descend"][-1]
    assert grasp["tcp_xyz"][2] == pytest.approx(0.103 + 0.019, abs=0.001)  # floor + the cone's grasp height
    # s502 (788cb77): leaving the low handover look for the cabinet swept through the cone just set down.
    i = max(k for k, e in enumerate(ev) if e["event"] == "vlm_reply" and e["purpose"] == "handover")
    after = [e for e in ev[i:] if e["event"] == "move" and "pre-grasp" not in e["phase"]]
    away = [e for e in after if e["phase"].endswith("cabinet look")]
    assert after[0]["phase"] == "up from the handover" and away
    assert min(e["tcp_xyz"][2] for e in away) >= 0.12 - 0.002
    assert world.in_handover("yellow_cone") and world.all_shut()


def test_on_sim_scene_numbers_the_close_look_into_a_drawer_is_the_calib_s_straight_down_view(
        link, camera, teleporting, tmp_path, monkeypatch):
    """D3 s503: the camera-over close look into r0c1 leaned the open jaw into r2c1's handle and stalled.
    That path is now refused (a finger in the cabinet), and the calib's top-down view is used first."""
    monkeypatch.setattr(cs, "load", cs.load_sim_like)
    scene = cs.load_sim_like()
    task = drawers.DrawerTask(None, None, scene, None, pickplace.RunLog(tmp_path / "t"), drawers.DrawerSettings())
    s503 = kinematics.joint_path(scene.fixture.drawer("r0c1").look_pose_deg, [5.07, -21.54, -77.05, 7.23, 4.21, -8.5], 8.0)
    assert "a finger" in task._path_clear(s503)
    objects = {"yellow_cone": dict(SIM_OBJECTS["yellow_cone"], drawer="r0c1", xy=[0.289, 0.004], z=0.023 + 0.013)}
    result, world, _ = run(link, camera, teleporting, tmp_path, objects=objects, order=["r0c1"])
    assert result["ok"] is True, result
    close = [e for e in events(tmp_path) if e["event"] == "move" and e["phase"] == "close look: r0c1"]
    assert close[-1]["angles_deg"] == pytest.approx(scene.fixture.drawer("r0c1").topdown_pose_deg, abs=0.01)
    assert world.in_handover("yellow_cone") and world.all_shut()


@pytest.mark.parametrize("layout", ["handle_stages", "side/top"])
def test_a_two_stage_handle_pulls_halfway_from_the_side_then_from_above(link, camera, teleporting, tmp_path,
                                                                        monkeypatch, layout):
    """sim-scene's v3 recipe: a horizontal grip at the end of a full pull puts the wrist in the base, so the
    pull goes to 50 mm by a side pinch and on to open by a top-down pinch; shutting runs it backwards.
    In the side/top layout the top grip leans 10 deg towards the robot by full open."""
    load = cs.load_sim_like_two_stage if layout == "handle_stages" else cs.load_sim_like_side_top
    monkeypatch.setattr(cs, "load", load)
    cone = dict(SIM_OBJECTS["yellow_cone"], drawer="r1c1", xy=[0.289, 0.0], z=0.063 + 0.013)
    result, world, _ = run(link, camera, teleporting, tmp_path, objects={"yellow_cone": cone}, order=["r1c1"])
    assert result["ok"] is True, result
    assert world.in_handover("yellow_cone") and world.all_shut()
    scene = load()
    phase, grips = "", []
    for e in events(tmp_path):
        phase = e["phase"] if e["event"] == "move" else phase
        if e["event"] == "gripper" and e["action"] == "close" and phase.endswith("to handle"):
            t = scene.chain.pose(last_angles, "tcp")
            grips.append((phase, round(kinematics.column(t, 2)[2], 2)))
        if e["event"] == "move":
            last_angles = e["angles_deg"]
    lean = -1.0 if layout == "handle_stages" else -math.cos(math.radians(10))  # gripped again at full open to push
    assert [g[0] for g in grips] == ["open r1c1 (1/2): to handle", "open r1c1 (2/2): to handle",
                                     "close r1c1 (2/2): to handle", "close r1c1 (1/2): to handle"]
    assert [g[1] for g in grips] == pytest.approx([0.0, -1.0, lean, 0.0], abs=0.011)
    pulls = [e for e in events(tmp_path) if e["event"] == "move" and e["phase"] == "open r1c1 (2/2)"]
    assert pulls[-1]["tcp_xyz"][0] == pytest.approx(0.236 - 0.1, abs=0.002)  # the knob, fully out


def test_a_drawer_the_calib_takes_out_of_use_is_never_opened(link, camera, teleporting, tmp_path, monkeypatch):
    """sim-scene 13:10: with the arm kept out of the base, no plan opens the bottom row; the calib can mark
    a drawer ``usable: false``, and the brain neither offers it to the VLM nor touches its handle."""
    d = cs.sim_like_dict()
    for row in d["cabinet"]["drawers"]:
        row["usable"] = row["row"] != 0
    monkeypatch.setattr(cs, "load", lambda: calib.from_dict(d))
    cone = dict(SIM_OBJECTS["yellow_cone"], drawer="r2c1", xy=[0.289, 0.0])
    result, world, vlm = run(link, camera, teleporting, tmp_path, objects={"yellow_cone": cone},
                             order=["r0c0", "r0c1", "r2c1"])
    assert result["ok"] is True and result["drawers_opened"] == ["r2c1"], result
    out = {e["drawer"] for e in events(tmp_path) if e["event"] == "unreachable"}
    assert out == {"r0c0", "r0c1", "r0c2"}
    result, _, _ = run(link, camera, teleporting, tmp_path / "take", instruction="왼쪽 아래 서랍에 있는 물건을 꺼내 줘",
                       objects={"yellow_cone": cone})
    assert result["ok"] is False and "r0c0 is not in use" in result["why"] and result["drawers_opened"] == []


def test_out_of_a_drawer_goes_back_down_to_the_lift_first(link, camera, teleporting, tmp_path, monkeypatch):
    """D3 v3.2 run 1 (s502, f0f49be): leaving r0c0 from its top-down view, over the cabinet edge, the line
    out took a folding elbow branch and the joint move down came 0.7 mm from the cabinet face. Now the arm
    first goes back down to where the grasp lifted, then leaves at the lift height."""
    monkeypatch.setattr(cs, "load", cs.load_sim_like)
    objects = {"yellow_cone": dict(SIM_OBJECTS["yellow_cone"], drawer="r2c2", xy=[0.289, -0.0985])}
    result, world, _ = run(link, camera, teleporting, tmp_path, objects=objects, order=["r2c2"])
    assert result["ok"] is True, result
    ev = [e for e in events(tmp_path) if e["event"] == "move"]
    phases = [e["phase"] for e in ev]
    lifted = [e for e in ev if e["phase"] == "lift"][-1]["angles_deg"]
    # Where the line out starts: back at the lift pose (retraced, if the held check rose above it).
    before_out = ev[phases.index("out of r2c2") - 1]["angles_deg"]
    assert before_out == pytest.approx(lifted, abs=0.05)
    assert world.in_handover("yellow_cone") and world.all_shut()


def _between_grasp_and_release(ev):
    """The moves while an object is held: from the gripper's close on it to the next open."""
    out, holding = [], False
    for e in ev:
        if e["event"] == "gripper":
            holding = e["action"] == "close" and e["value"] == 0 and not e.get("handle")
        if holding and e["event"] == "move":
            out.append(e)
    return out


def test_holding_an_object_the_wrist_does_not_roll_and_moves_go_smooth(link, camera, teleporting, tmp_path,
                                                                       monkeypatch):
    """sim-scene 14:03 / leader 14:04: a cone gripped a little off centre crept out of the pads when the held
    check's view rolled J6 82 -> 47 -> 82 with it in hand. From the grasp to the release J6 stays put; the
    held check looks from straight above; the way down and out go as one joint move each where they pass."""
    monkeypatch.setattr(cs, "load", cs.load_sim_like)
    objects = {"yellow_cone": dict(SIM_OBJECTS["yellow_cone"], drawer="r2c2", xy=[0.289, -0.0985])}
    result, world, _ = run(link, camera, teleporting, tmp_path, objects=objects, order=["r2c2"])
    assert result["ok"] is True, result
    ev = events(tmp_path)
    held = [e for e in ev if e["event"] == "move" and e["phase"] in (
        "lift", "lift: to the look", "out of r2c2: back to the lift", "out of r2c2", "carry: handover",
        "lower: handover")]
    j6 = {round(e["angles_deg"][5], 1) for e in held}
    assert max(j6) - min(j6) < 0.2, j6
    assert not any(e["event"] == "move" and e["phase"] == "look: held" for e in ev)
    assert "out of r2c2" in {e["phase"] for e in ev if e["event"] == "held_move"}
    assert world.in_handover("yellow_cone") and world.all_shut()


def test_a_put_grips_on_the_mat_at_the_yaw_that_arrives_along_the_pick_jaw(link, camera, teleporting, tmp_path,
                                                                           monkeypatch):
    """No wrist roll with the object in hand on a put either: the table grasp takes the yaw that, carried
    with J6 held, lands along the calib's pick jaw over the drawer; a slow turn only if none does."""
    monkeypatch.setattr(cs, "load", cs.load_sim_like)
    cone = {"colour": "yellow", "shape": "cone", "drawer": None, "xy": [0.10, 0.19], "z": 0.013}
    cmd = "노란색 원뿔을 왼쪽 위 서랍에 넣어"
    COMMANDS[cmd] = {"task": "put", "colour": "yellow", "shape": "cone", "drawer": "r2c0"}
    try:
        result, world, _ = run(link, camera, teleporting, tmp_path, instruction=cmd, objects={"yellow_cone": cone})
    finally:
        COMMANDS.pop(cmd)
    assert result["ok"] is True, result
    ev = events(tmp_path)
    yaw = [e for e in ev if e["event"] == "put_yaw"]
    assert yaw and yaw[0]["deg_off_pick_jaw"] <= 10.0
    assert not any(e["event"] == "slow_turn" for e in ev)
    carried = [e for e in ev if e["event"] == "move" and e["phase"].startswith(("lift", "raise", "carry", "lower"))]
    j6 = [e["angles_deg"][5] for e in carried if not e["phase"].startswith("lift: to")]
    assert max(j6) - min(j6) < 0.2, (min(j6), max(j6))
    assert world.objects["yellow_cone"]["drawer"] == "r2c0" and world.all_shut()
