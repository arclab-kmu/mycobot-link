import json

import pytest
from test_vision import images

from mycobot_brain import grounding
from mycobot_brain.planner import PlannerError

JPEG = b"\xff\xd8look\xff\xd9"


def reply(**kw):
    base = {"seen": "a red block and a brown box",
            "pick": {"label": "red block", "visible": True, "bbox_2d": [400, 300, 600, 500]},
            "place": {"label": "brown box", "visible": True, "bbox_2d": [100, 100, 200, 300]},
            "next": "pick"}
    return json.dumps({**base, **kw})


def test_locate_request_is_schema_constrained_with_thinking_off():
    body = grounding.build_locate_request(JPEG, "빨간 블록을 잡아서 갈색 상자 안에 넣어")
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0
    schema = body["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["seen", "pick", "place", "next"]
    assert schema["properties"]["pick"]["properties"]["bbox_2d"]["minItems"] == 4
    user = body["messages"][1]["content"]
    assert images(user) == [JPEG] and "빨간 블록" in user[1]["text"]
    assert "0-1000" in body["messages"][0]["content"]


def test_abs_coordinates_name_the_image_size_and_points_ask_for_two_numbers():
    body = grounding.build_locate_request(JPEG, "x", roles=("pick",), coords="abs", shape="point", with_next=False)
    assert "640x480" in body["messages"][0]["content"]
    schema = body["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["seen", "pick"]
    assert schema["properties"]["pick"]["properties"]["point_2d"]["maxItems"] == 2


def test_norm1000_boxes_become_pixel_centres():
    found = grounding.parse_locate(reply(), ("pick", "place"), coords="norm1000", shape="bbox")
    assert found["pick"]["pixel"] == [320.0, 192.0]
    assert found["pick"]["bbox_px"] == [256.0, 144.0, 384.0, 240.0]
    assert found["place"]["pixel"] == [96.0, 96.0] and found["next"] == "pick"


def test_abs_points_are_already_pixels():
    text = json.dumps({"seen": "", "pick": {"label": "b", "visible": True, "point_2d": [500, 100]}})
    found = grounding.parse_locate(text, ("pick",), coords="abs", shape="point", with_next=False)
    assert found["pick"]["pixel"] == [500.0, 100.0]


def test_an_invisible_object_has_no_pixel_whatever_numbers_came_with_it():
    text = reply(place={"label": "brown box", "visible": False, "bbox_2d": [0, 0, 0, 0]}, next="cannot")
    found = grounding.parse_locate(text, ("pick", "place"), coords="norm1000", shape="bbox")
    assert "pixel" not in found["place"] and found["next"] == "cannot"


@pytest.mark.parametrize("text,match", [
    ("the block is left", "not JSON"),
    (reply(pick={"label": "b", "visible": "yes", "bbox_2d": [1, 2, 3, 4]}), "boolean visible"),
    (reply(pick={"label": "b", "visible": True, "bbox_2d": [1, 2, 3]}), "needs 4 numbers"),
    (reply(pick={"label": "b", "visible": True, "bbox_2d": [1100, 0, 1300, 10]}), "outside"),
    (reply(next="maybe"), "next must be"),
])
def test_bad_locate_replies_are_refused(text, match):
    with pytest.raises(PlannerError, match=match):
        grounding.parse_locate(text, ("pick", "place"), coords="norm1000", shape="bbox")


def test_check_sends_captioned_images_in_order_and_parses_ok():
    body = grounding.build_check_request("Is it held?", [("Before:", b"\xff\xd8a\xff\xd9"), ("Now:", JPEG)])
    content = body["messages"][1]["content"]
    assert [p.get("text") for p in content if p["type"] == "text"] == ["Before:", "Now:", "Is it held?"]
    assert images(content) == [b"\xff\xd8a\xff\xd9", JPEG]
    assert grounding.parse_check('{"seen": "block between fingers", "ok": true}') == {
        "seen": "block between fingers", "ok": True}
    with pytest.raises(PlannerError, match="boolean ok"):
        grounding.parse_check('{"seen": "x", "ok": "yes"}')


def test_prompt_text_drops_the_images():
    body = grounding.build_locate_request(JPEG, "pick it")
    text = grounding.prompt_text(body)
    assert "base64" not in text and text.endswith("[user] Instruction: pick it")
