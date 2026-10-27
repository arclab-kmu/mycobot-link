import io
import json

import pytest

from mycobot_brain import planner


class FakeOpener:
    """Stands in for urllib.request.urlopen: records the request, returns ``content``."""

    def __init__(self, content):
        self.content = content
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append((req.full_url, json.loads(req.data)))
        body = {"choices": [{"message": {"content": self.content}}]}
        return io.BytesIO(json.dumps(body).encode())


def test_request_forces_the_move_schema_and_turns_thinking_off():
    opener = FakeOpener(json.dumps({"angles_deg": [5, 0, 0, 0, 0, 0], "speed": 20, "reason": "nudge J1"}))
    move = planner.propose_move("nudge J1", [0.0] * 6, base_url="http://spark:8000/v1/", opener=opener)
    assert move == {"angles_deg": [5.0, 0.0, 0.0, 0.0, 0.0, 0.0], "speed": 20, "reason": "nudge J1"}
    url, body = opener.requests[0]
    assert url == "http://spark:8000/v1/chat/completions"
    assert body["model"] == "qwen3.6"
    assert body["response_format"]["json_schema"]["schema"] == planner.MOVE_SCHEMA
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "J1=+0.0" in body["messages"][1]["content"]


@pytest.mark.parametrize("content", [
    "not json",
    json.dumps({"angles_deg": [0, 0, 0], "speed": 20}),
    json.dumps({"angles_deg": [0, 0, 0, 0, 0, "x"], "speed": 20}),
    json.dumps({"angles_deg": [0, 0, 0, 0, 0, True], "speed": 20}),
    json.dumps({"angles_deg": [0] * 6, "speed": 90}),
    json.dumps([0] * 6),
])
def test_malformed_moves_are_refused_here_too(content):
    with pytest.raises(planner.PlannerError):
        planner.propose_move("x", [0.0] * 6, opener=FakeOpener(content))


def test_unreachable_vllm_is_a_planner_error():
    def opener(req, timeout):
        raise OSError("connection refused")

    with pytest.raises(planner.PlannerError, match="connection refused"):
        planner.propose_move("x", [0.0] * 6, opener=opener)
