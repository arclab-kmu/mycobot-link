import base64
import json

import pytest
from test_planner import FakeOpener

from mycobot_brain import vision
from mycobot_brain.planner import PlannerError

JPEG = b"\xff\xd8now\xff\xd9"
BEFORE = b"\xff\xd8before\xff\xd9"


def images(content):
    return [base64.b64decode(p["image_url"]["url"].split(",", 1)[1]) for p in content if p["type"] == "image_url"]


def step(**kw):
    return json.dumps(dict({"seen": "a cup left of centre", "reason": "pan left", "done": False,
                            "delta_deg": [3, 0, 0, 0, 0, 0]}, **kw))


def test_describe_sends_the_frame_as_a_data_url_with_thinking_off():
    opener = FakeOpener("  A white tape roll in the centre.  ")
    text = vision.describe(JPEG, base_url="http://spark:8000/v1", opener=opener)
    assert text == "A white tape roll in the centre."
    url, body = opener.requests[0]
    assert url == "http://spark:8000/v1/chat/completions"
    content = body["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert images(content) == [JPEG]
    assert content[1]["text"] == vision.DESCRIBE_PROMPT
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_step_request_shows_before_then_now_and_bounds_the_schema():
    opener = FakeOpener(step())
    history = [{"step": 1, "delta_deg": [3.0, 0, 0, 0, 0, 0], "reason": "pan left"}]
    out = vision.propose_step(
        "centre the cup", JPEG, [10.0, 0, 0, 0, 0, -45.0], previous_jpeg=BEFORE, history=history,
        step=2, steps=5, max_step=4, locked=[2, 3], opener=opener,
    )
    assert out == {"seen": "a cup left of centre", "reason": "pan left", "done": False,
                   "delta_deg": [3.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    _, body = opener.requests[0]
    system, user = body["messages"]
    assert "J2, J3 must stay 0" in system["content"] and "+-4 degrees" in system["content"]
    assert images(user["content"]) == [BEFORE, JPEG]  # the order the text labels promise
    text = user["content"][0]["text"]
    assert "step 2 of at most 5" in text and "J1=+10.0" in text and "J6=-45.0" in text and "pan left" in text
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["delta_deg"]["items"] == {"type": "number", "minimum": -4, "maximum": 4}
    assert list(schema["properties"]) == ["seen", "reason", "done", "delta_deg"]


def test_first_step_has_one_image():
    opener = FakeOpener(step())
    vision.propose_step("x", JPEG, [0.0] * 6, opener=opener)
    _, body = opener.requests[0]
    assert images(body["messages"][1]["content"]) == [JPEG]
    assert "No earlier steps" in body["messages"][1]["content"][0]["text"]


def test_done_zeroes_the_deltas():
    out = vision.parse_step(step(done=True, delta_deg=[1, 0, 0, 0, 0, 0]), max_step=5)
    assert out["done"] and out["delta_deg"] == [0.0] * 6


@pytest.mark.parametrize("content, why", [
    ("nope", "not JSON"),
    (step(done="yes"), "boolean done"),
    (step(delta_deg=[1, 2, 3]), "6 delta_deg"),
    (step(delta_deg=[1, 0, 0, 0, 0, True]), "finite"),
    (step(delta_deg=[5.5, 0, 0, 0, 0, 0]), "exceeds"),
    (step(delta_deg=[0, 1, 0, 0, 0, 0]), "locked joints moved: J2"),
])
def test_bad_steps_are_refused_not_clipped(content, why):
    with pytest.raises(PlannerError, match=why):
        vision.parse_step(content, max_step=5, locked=[2])
