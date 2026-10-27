"""The real Rust arm server, dry-run, against the real ArmLink on localhost.

Skipped unless the binary exists: ``cargo build`` in arm-server/ first, or
point MYCOBOT_ARM_BIN at one. Every line the server sends is validated against
the schema, so this is also the check that both sides speak the same protocol.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from conftest import SCHEMA_DIR, wait_until
from mycobot_brain.link import ArmLink, Rejected

BIN = Path(os.environ.get("MYCOBOT_ARM_BIN", SCHEMA_DIR.parent / "arm-server/target/debug/mycobot-arm"))
pytestmark = pytest.mark.skipif(not BIN.exists(), reason=f"no arm server binary at {BIN}")

SCHEMA = json.loads((SCHEMA_DIR / "messages.schema.json").read_text())
ARM_TO_BRAIN = Draft202012Validator({**SCHEMA, "oneOf": [{"$ref": "#/$defs/arm_to_brain"}]})


class Recording(ArmLink):
    """ArmLink that keeps every line the arm sent, for schema validation."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []

    def _on_line(self, raw, conn):
        self.seen.append(json.loads(raw))
        super()._on_line(raw, conn)


@pytest.fixture
def served():
    link = Recording("127.0.0.1", 0, ping_interval=0.05).start()
    proc = subprocess.Popen(
        [str(BIN), "--brain", f"127.0.0.1:{link.port}", "--watchdog-ms", "300", "--state-hz", "20"],
        stderr=subprocess.PIPE, text=True,
    )
    try:
        hello = link.wait_connected(10.0)
        yield link, hello, proc
    finally:
        proc.terminate()  # SIGTERM: no Ctrl-C handler path, just make sure it is gone
        _, err = proc.communicate(timeout=5)
        link.close()
        print(err)
    for msg in link.seen:
        ARM_TO_BRAIN.validate(msg)


def test_hello_says_dry_run(served):
    link, hello, _ = served
    assert hello["live"] is False
    assert hello["watchdog_ms"] == 300
    assert hello["state_hz"] == pytest.approx(20.0)


def test_move_within_the_jump_limit_lands_and_a_jump_is_refused(served):
    link, _, _ = served
    state = link.wait_state(timeout=2.0)
    assert state["angles_deg"] == [0.0] * 6 and state["gate_open"] is False
    with pytest.raises(Rejected, match="gate closed"):
        link.move_joints([5, 0, 0, 0, 0, 0])
    link.enable()
    link.move_joints([10, -5, 0, 0, 0, 0], speed=20)
    assert wait_until(lambda: (link.latest_state() or {}).get("angles_deg") == [10.0, -5.0, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(Rejected, match="J1 is 25.0 deg"):
        link.move_joints([35, -5, 0, 0, 0, 0])
    with pytest.raises(Rejected, match="outside limits"):
        link.move_joints([10, -5, 0, 0, 158, 0])
    with pytest.raises(Rejected, match="speed 80"):
        link.move_joints([12, -5, 0, 0, 0, 0], speed=80)
    link.stop()
    assert wait_until(lambda: (link.latest_state() or {}).get("gate_reason") == "stopped by brain")


def test_watchdog_closes_the_gate_when_the_brain_goes_quiet(served):
    link, _, _ = served
    link.wait_state(timeout=2.0)
    link.enable()
    link.pinging.clear()
    t0 = time.monotonic()
    assert wait_until(lambda: (link.latest_state() or {}).get("gate_reason", "").startswith("watchdog"), timeout=2.0)
    assert time.monotonic() - t0 >= 0.3
    assert link.latest_state()["gate_open"] is False
    with pytest.raises(Rejected, match="gate closed: watchdog"):
        link.move_joints([1, 0, 0, 0, 0, 0])


def test_server_reconnects_after_the_brain_drops_it(served):
    link, _, proc = served
    with link._cv:
        link._drop_locked("test")
    assert link.wait_connected(5.0)["server"].startswith("mycobot-arm/")
    assert proc.poll() is None
