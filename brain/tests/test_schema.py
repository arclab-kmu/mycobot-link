"""The schema, its examples, and what this side actually sends must agree.
The Rust side checks the same examples in arm-server/tests/schema_examples.rs."""

import json

import pytest
from jsonschema import Draft202012Validator

from conftest import SCHEMA_DIR, wait_until

SCHEMA = json.loads((SCHEMA_DIR / "messages.schema.json").read_text())
EXAMPLES = json.loads((SCHEMA_DIR / "examples.json").read_text())


def validator(part):
    """The whole schema, narrowed to one direction."""
    return Draft202012Validator({**SCHEMA, "oneOf": [{"$ref": f"#/$defs/{part}"}]})


def test_schema_is_valid():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("part", ["brain_to_arm", "arm_to_brain"])
def test_examples_validate_in_their_direction_only(part):
    other = "arm_to_brain" if part == "brain_to_arm" else "brain_to_arm"
    for ex in EXAMPLES[part]:
        validator(part).validate(ex)
        assert not validator(other).is_valid(ex), ex


@pytest.mark.parametrize("bad", [
    {"type": "move_joints", "id": 1, "angles_deg": [0, 0, 0, 0, 0], "speed": 20},
    {"type": "move_joints", "id": 1, "angles_deg": [0, 0, 0, 0, 0, 0], "speed": 20.5},
    {"type": "move_joints", "id": 1, "angle_deg": [0, 0, 0, 0, 0, 0], "speed": 20},
    {"type": "gripper", "id": 1, "value": 101, "speed": 50},
    {"type": "ping", "id": 1},
    {"type": "reply", "id": 1, "ok": False},
    {"type": "reply", "id": 1, "ok": True, "error": "contradiction"},
])
def test_schema_refuses(bad):
    assert not Draft202012Validator(SCHEMA).is_valid(bad)


def test_everything_the_brain_sends_validates(link, arm):
    link.enable()
    link.move_joints([0, -10.5, 20, 0, 0, 45], speed=20)
    link.gripper(40, speed=50)
    link.stop()
    link.disable()
    assert wait_until(lambda: any(m["type"] == "ping" for m in arm.received))
    v = validator("brain_to_arm")
    for msg in arm.received:
        v.validate(msg)
