import json
import threading
import time

import pytest
from conftest import wait_until
from test_camera import FakeCamera

from mycobot_brain import act
from mycobot_brain.camera import CameraLink

START = [10.0, 0.0, 0.0, 0.0, 0.0, -45.0]


@pytest.fixture
def camera():
    with CameraLink("127.0.0.1", 0) as cam:
        fake = FakeCamera(cam.port).stream()
        cam.wait_connected(2.0)
        yield cam
        fake.close()


@pytest.fixture
def teleporting(arm):
    """States at 50 Hz; a move_joints the fake arm accepted is where it is next."""
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
            time.sleep(0.02)

    threading.Thread(target=run, daemon=True).start()
    yield arm
    stop.set()


class Script:
    """Stands in for vision.propose_step: hands out canned steps, records what it saw."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []

    def __call__(self, instruction, jpeg, angles, **kw):
        self.calls.append(dict(kw, instruction=instruction, jpeg=jpeg, angles=list(angles)))
        delta, done = self.steps.pop(0)
        return {"seen": "x", "reason": f"step {len(self.calls)}", "done": done, "delta_deg": delta}


def sent(arm, kind):
    return [m for m in arm.received if m["type"] == kind]


def test_without_execute_it_proposes_once_and_never_opens_the_gate(link, camera, teleporting, tmp_path):
    script = Script(([3, 0, 0, 0, 0, 0], False))
    out = []
    act.run(link, camera, "centre the cup", propose=script, emit=out.append, record=tmp_path)
    assert len(script.calls) == 1 and script.calls[0]["angles"] == START
    assert script.calls[0]["previous_jpeg"] is None
    assert not sent(teleporting, "enable") and not sent(teleporting, "move_joints")
    assert out[0]["step"] == 1 and out[0]["delta_deg"] == [3, 0, 0, 0, 0, 0]
    assert (tmp_path / "step-01.jpg").read_bytes().startswith(b"\xff\xd8")
    assert json.loads((tmp_path / "steps.jsonl").read_text())["instruction"] == "centre the cup"


def test_execute_steps_from_measured_angles_until_done(link, camera, teleporting):
    script = Script(([3, 0, 0, 0, 0, 0], False), ([0, -2, 0, 1, 0, 0], False), ([0] * 6, True))
    out = []
    last = act.run(link, camera, "centre the cup", propose=script, emit=out.append, execute=True, settle=0.05)
    assert last["done"]
    moves = [m["angles_deg"] for m in sent(teleporting, "move_joints")]
    assert moves == [[13.0, 0.0, 0.0, 0.0, 0.0, -45.0], [13.0, -2.0, 0.0, 1.0, 0.0, -45.0]]
    assert script.calls[1]["angles"] == moves[0]  # planned from where the arm reported, not the plan
    assert script.calls[1]["previous_jpeg"] == script.calls[0]["jpeg"]
    assert [h["step"] for h in script.calls[2]["history"]] == [1, 2]
    assert len(sent(teleporting, "enable")) == 1
    assert wait_until(lambda: len(sent(teleporting, "disable")) == 1)
    assert all(s["speed"] == 10 for s in sent(teleporting, "move_joints"))


def test_each_look_waits_for_a_frame_newer_than_the_settled_arm(link, camera, teleporting):
    script = Script(([3, 0, 0, 0, 0, 0], False), ([0] * 6, True))
    act.run(link, camera, "x", propose=script, emit=lambda r: None, execute=True, settle=0.1)
    first, second = (int.from_bytes(c["jpeg"][2:6], "big") for c in script.calls)
    assert second > first + 2  # frames come every 20 ms; settle was 100 ms


def test_drift_past_max_total_stops_and_closes_the_gate(link, camera, teleporting):
    script = Script(([5, 0, 0, 0, 0, 0], False), ([5, 0, 0, 0, 0, 0], False), ([5, 0, 0, 0, 0, 0], False))
    with pytest.raises(act.ActStopped, match="15.0 deg from the start"):
        act.run(link, camera, "x", propose=script, emit=lambda r: None, execute=True, settle=0.05, max_total=12)
    assert len(sent(teleporting, "move_joints")) == 2
    assert wait_until(lambda: len(sent(teleporting, "disable")) == 1)


def test_step_limit_ends_without_done(link, camera, teleporting):
    script = Script(([1, 0, 0, 0, 0, 0], False), ([1, 0, 0, 0, 0, 0], False))
    out = []
    act.run(link, camera, "x", propose=script, emit=out.append, execute=True, settle=0.05, steps=2)
    assert out[-1] == {"done": False, "why": "no done after 2 steps"}
    assert wait_until(lambda: len(sent(teleporting, "disable")) == 1)


def test_frame_age_is_taken_before_vllm_answers(link, camera, teleporting):
    def slow(*a, **kw):
        time.sleep(0.4)
        return {"seen": "x", "reason": "x", "done": True, "delta_deg": [0.0] * 6}

    out = []
    act.run(link, camera, "x", propose=slow, emit=out.append)
    assert out[0]["vllm_s"] >= 0.4 and out[0]["frame_age_s"] < 0.2
