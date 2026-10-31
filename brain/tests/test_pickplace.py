import json
import math
import threading
import time
from pathlib import Path

import pytest
import scene
from conftest import wait_until
from test_camera import FakeCamera

from mycobot_brain import calib, kinematics, pickplace
from mycobot_brain.camera import CameraLink
from mycobot_brain.planner import PlannerError

START = [0.0, 0.0, -30.0, 0.0, 0.0, 0.0]
FAST = dict(settle=0.02, gripper_wait=0.02, arrive_timeout=3.0)


@pytest.fixture
def camera():
    with CameraLink("127.0.0.1", 0) as cam:
        fake = FakeCamera(cam.port).stream()
        cam.wait_connected(2.0)
        yield cam
        fake.close()


@pytest.fixture
def teleporting(arm):
    """States at 100 Hz; a move_joints the fake arm accepted is where it is next."""
    stop = threading.Event()

    def run():
        seq = 0
        while not stop.is_set():
            moves = [m for m in list(arm.received) if m["type"] == "move_joints"]
            seq += 1
            try:
                arm.state(seq, moves[-1]["angles_deg"] if moves else START)
            except OSError:
                return
            time.sleep(0.01)

    threading.Thread(target=run, daemon=True).start()
    yield arm
    stop.set()


def sent(arm, kind):
    return [m for m in arm.received if m["type"] in kind]


def where(arm):
    moves = sent(arm, ("move_joints",))
    return moves[-1]["angles_deg"] if moves else START


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]


def run(link, camera, arm, tmp_path, vlm=None, calib=None, **settings):
    calib = calib or scene.load()
    vlm = vlm or scene.SceneVLM(calib, lambda: where(arm))
    log = pickplace.RunLog(tmp_path)
    s = pickplace.Settings(**{**dict(execute=True), **FAST, **settings})
    return pickplace.PickPlace(link, camera, calib, vlm, log, s).run("빨간 블록을 잡아서 갈색 상자 안에 넣어"), vlm


def tcp(q, calib=None):
    return kinematics.position((calib or scene.load()).chain.pose(q, "tcp"))


def at_gripper(arm, n):
    """Arm angles when the n-th gripper command (1-based) was sent."""
    angles, k = START, 0
    for m in arm.received:
        if m["type"] == "move_joints":
            angles = m["angles_deg"]
        if m["type"] == "gripper":
            k += 1
            if k == n:
                return angles
    raise AssertionError(f"only {k} gripper commands")


def test_full_run_grasps_the_block_centre_and_releases_over_the_box(link, camera, teleporting, tmp_path):
    result, vlm = run(link, camera, teleporting, tmp_path)
    assert result["ok"] is True and result["attempts"] == 1
    assert [g["value"] for g in sent(teleporting, ("gripper",))] == [80, 15, 80]
    assert math.dist(tcp(at_gripper(teleporting, 2)), scene.BLOCK) < 0.002
    release = tcp(at_gripper(teleporting, 3))
    assert math.dist(release[:2], scene.BOX[:2]) < 0.002
    assert release[2] == pytest.approx(0.05 + 0.0125 + 0.01, abs=0.001)  # block bottom 1 cm over the rim
    assert len(sent(teleporting, ("enable",))) == 1
    assert wait_until(lambda: len(sent(teleporting, ("disable",))) == 1)
    assert not sent(teleporting, ("stop",))


def test_no_waypoint_moves_a_joint_more_than_max_step(link, camera, teleporting, tmp_path):
    run(link, camera, teleporting, tmp_path, max_step=8.0)
    q = START
    for m in sent(teleporting, ("move_joints",)):
        assert max(abs(a - b) for a, b in zip(m["angles_deg"], q)) <= 8.0 + 1e-6
        q = m["angles_deg"]
    assert all(m["speed"] in (30, 15) for m in sent(teleporting, ("move_joints",)))


def test_descent_goes_straight_down_with_the_gripper_pointing_down(link, camera, teleporting, tmp_path):
    run(link, camera, teleporting, tmp_path)
    descend = [e for e in events(tmp_path) if e["event"] == "move" and e["phase"] == "descend"]
    assert len(descend) >= 5
    for e in descend:
        assert math.dist(e["tcp_xyz"][:2], scene.BLOCK[:2]) < 0.002
        z_axis = kinematics.column(scene.load().chain.pose(e["angles_deg"], "tcp"), 2)
        assert z_axis[2] < -0.999


def test_events_have_wall_time_and_every_vlm_image_is_saved(link, camera, teleporting, tmp_path):
    run(link, camera, teleporting, tmp_path)
    ev = events(tmp_path)
    assert all(isinstance(e["wall"], float) and e["wall"] > 1.7e9 for e in ev)
    kinds = {e["event"] for e in ev}
    assert {"command", "vlm_request", "vlm_reply", "move", "gripper", "result", "estimate"} <= kinds
    assert ev[0]["event"] == "command" and ev[0]["instruction"] == "빨간 블록을 잡아서 갈색 상자 안에 넣어"
    assert ev[-1] == {**ev[-1], "event": "result", "ok": True}
    requests = [e for e in ev if e["event"] == "vlm_request"]
    assert [r["purpose"] for r in requests] == ["look", "close-pick", "held", "close-place", "final"]
    for r in requests:
        assert all((tmp_path / name).read_bytes().startswith(b"\xff\xd8") for name in r["images"])
    replies = [e for e in ev if e["event"] == "vlm_reply"]
    assert all("raw" in r and "parsed" in r and r["latency_s"] >= 0 for r in replies)
    assert replies[0]["parsed"]["pick"]["label"] == "red block"
    assert len(requests[2]["images"]) == 2  # held: before the grasp and now


def test_the_close_look_corrects_a_first_look_that_was_off(link, camera, teleporting, tmp_path):
    result, _ = run(link, camera, teleporting, tmp_path / "a",
                    vlm=scene.SceneVLM(scene.load(), lambda: where(teleporting), bias=(25, -20)))
    assert result["ok"]
    est = [e for e in events(tmp_path / "a") if e["event"] == "estimate" and e["role"] == "pick"]
    assert math.dist(est[0]["xyz"], scene.BLOCK) > 0.01  # far look: a centimetre or more off
    assert math.dist(est[1]["xyz"], scene.BLOCK) < 0.001  # close look: on it
    assert math.dist(tcp(at_gripper(teleporting, 2)), scene.BLOCK) < 0.002


@pytest.mark.parametrize("coords,shape", [("abs", "bbox"), ("norm1000", "point"), ("abs", "point")])
def test_every_coordinate_format_lands_on_the_block(link, camera, teleporting, tmp_path, coords, shape):
    vlm = scene.SceneVLM(scene.load(), lambda: where(teleporting), coords=coords, shape=shape)
    result, _ = run(link, camera, teleporting, tmp_path, vlm=vlm, coords=coords, shape=shape)
    assert result["ok"]
    assert math.dist(tcp(at_gripper(teleporting, 2)), scene.BLOCK) < 0.002


def test_a_grasp_not_seen_held_is_retried_from_the_look_pose(link, camera, teleporting, tmp_path):
    vlm = scene.SceneVLM(scene.load(), lambda: where(teleporting), held=(False, True))
    result, _ = run(link, camera, teleporting, tmp_path, vlm=vlm)
    assert result["ok"] and result["attempts"] == 2
    assert [g["value"] for g in sent(teleporting, ("gripper",))] == [80, 15, 80, 15, 80]
    purposes = [e["purpose"] for e in events(tmp_path) if e["event"] == "vlm_request"]
    assert purposes == ["look", "close-pick", "held", "relook", "close-pick", "held", "close-place", "final"]


def test_gives_up_with_the_gripper_open_after_the_last_attempt(link, camera, teleporting, tmp_path):
    vlm = scene.SceneVLM(scene.load(), lambda: where(teleporting), held=(False,))
    result, _ = run(link, camera, teleporting, tmp_path, vlm=vlm, attempts=2)
    assert result["ok"] is False and "not held after 2 attempts" in result["why"]
    assert sent(teleporting, ("gripper",))[-1]["value"] == 80
    assert events(tmp_path)[-1]["ok"] is False
    assert wait_until(lambda: len(sent(teleporting, ("disable",))) == 1)


def test_cannot_ends_at_the_look_pose_without_touching_anything(link, camera, teleporting, tmp_path):
    vlm = scene.SceneVLM(scene.load(), lambda: where(teleporting), next="cannot")
    result, _ = run(link, camera, teleporting, tmp_path, vlm=vlm)
    assert result["ok"] is False
    assert where(teleporting) == pytest.approx(scene.SCENE["look_pose_deg"])
    assert [g["value"] for g in sent(teleporting, ("gripper",))] == [80]


def test_without_execute_it_plans_from_where_the_arm_is_and_sends_nothing(link, camera, teleporting, tmp_path):
    # Start at the look pose, as a dry plan would be run.
    global START
    old, START = START, list(scene.SCENE["look_pose_deg"])
    try:
        result, _ = run(link, camera, teleporting, tmp_path, execute=False)
    finally:
        START = old
    assert result["ok"] is None
    assert math.dist(result["pick"], scene.BLOCK) < 0.001 and math.dist(result["place"], scene.BOX) < 0.001
    assert math.dist(tcp(result["pregrasp_deg"]), (*scene.BLOCK[:2], 0.0725)) < 0.002
    assert not sent(teleporting, ("enable", "move_joints", "gripper", "stop"))


def test_a_vlm_reply_that_does_not_parse_stops_the_arm(link, camera, teleporting, tmp_path):
    good = scene.SceneVLM(scene.load(), lambda: where(teleporting))
    calls = []

    def flaky(request):
        calls.append(request)
        return good(request) if len(calls) == 1 else "I think the block is on the left"

    with pytest.raises(PlannerError, match="not JSON"):
        run(link, camera, teleporting, tmp_path, vlm=flaky)
    assert wait_until(lambda: len(sent(teleporting, ("stop",))) == 1)
    last = events(tmp_path)
    assert last[-2]["event"] == "vlm_reply" and "error" in last[-2]
    assert last[-1]["event"] == "result" and last[-1]["ok"] is False


def test_an_arm_that_does_not_arrive_stops_the_run(link, camera, arm, tmp_path):
    """States keep coming but never move: the first waypoint times out."""
    stop = threading.Event()

    def frozen():
        seq = 0
        while not stop.is_set():
            seq += 1
            try:
                arm.state(seq, START)
            except OSError:
                return
            time.sleep(0.01)

    threading.Thread(target=frozen, daemon=True).start()
    try:
        with pytest.raises(pickplace.PickPlaceError, match="look: still"):
            run(link, camera, arm, tmp_path, arrive_timeout=0.3)
        assert wait_until(lambda: len(sent(arm, ("stop",))) == 1)
        assert len(sent(arm, ("move_joints",))) == 1
    finally:
        stop.set()


def test_sim_calib_near_the_base_tips_the_fingers_in_and_keeps_them_off_the_table(link, camera, teleporting, tmp_path):
    """At r = 0.147 m a vertical grasp folds J3 past its limit in the sim scene."""
    sim = calib.load(Path(__file__).parent / "data" / "sim_calib.json")
    block, box = (0.14, 0.045, sim.block_plane_z), (0.2, -0.04, sim.box_rim_z)
    vlm = scene.SceneVLM(sim, lambda: where(teleporting), block=block, box=box)
    result, _ = run(link, camera, teleporting, tmp_path, vlm=vlm, calib=sim)
    assert result["ok"]
    assert [e["tilt_deg"] for e in events(tmp_path) if e["event"] == "approach"] == [5]
    grasp = tcp(at_gripper(teleporting, 2), sim)
    assert math.dist(grasp[:2], block[:2]) < 0.002
    assert grasp[2] == pytest.approx(0.0114 + 0.004, abs=0.0005)  # pads 4 mm over the table, not at the block centre
    assert math.dist(tcp(at_gripper(teleporting, 3), sim)[:2], box[:2]) < 0.002
