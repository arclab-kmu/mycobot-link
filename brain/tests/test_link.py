import threading

import pytest

from conftest import FakeArm, wait_until
from mycobot_brain.link import ArmLink, LinkError, Rejected


def test_request_gets_its_own_reply(link, arm):
    link.move_joints([1, 2, 3, 4, 5, 6], speed=20)
    sent = [m for m in arm.received if m["type"] == "move_joints"]
    assert sent == [{"type": "move_joints", "angles_deg": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "speed": 20, "id": sent[0]["id"]}]


def test_concurrent_requests_do_not_cross(link, arm):
    errors = []

    def run(i):
        try:
            link.gripper(i, speed=50)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    ids = [m["id"] for m in arm.received if m["type"] == "gripper"]
    assert len(set(ids)) == 20


def test_refusal_raises_with_the_arm_reason(link):
    fake = FakeArm(link.port, refuse={"enable"})
    try:
        wait_until(lambda: link.connected)
        with pytest.raises(Rejected, match="refused by fake"):
            link.enable()
    finally:
        fake.close()


def test_missing_reply_times_out(link, arm):
    arm.silent = True
    with pytest.raises(LinkError, match="no reply"):
        link.request({"type": "stop"}, timeout=0.1)


def test_heartbeat_flows_and_can_be_paused(link, arm):
    assert wait_until(lambda: sum(m["type"] == "ping" for m in arm.received) >= 3)
    link.pinging.clear()
    wait_until(lambda: False, timeout=0.05)  # let an in-flight ping land
    n = len(arm.received)
    wait_until(lambda: False, timeout=0.1)
    assert len(arm.received) == n


def test_latest_state_and_wait_state(link, arm):
    arm.state(1)
    first = link.wait_state(timeout=1.0)
    arm.state(2, angles=(5, 0, 0, 0, 0, 0))
    second = link.wait_state(after_seq=first["seq"], timeout=1.0)
    assert second["angles_deg"][0] == 5
    assert link.latest_state()["seq"] == 2


def test_new_connection_replaces_the_old_and_fails_its_waiters(link, arm):
    arm.silent = True
    result = {}

    def waiter():
        try:
            link.request({"type": "stop"}, timeout=5.0)
        except LinkError as exc:
            result["err"] = str(exc)

    t = threading.Thread(target=waiter)
    t.start()
    assert wait_until(lambda: any(m["type"] == "stop" for m in arm.received))
    second = FakeArm(link.port)
    try:
        t.join(2.0)
        assert "dropped" in result.get("err", "")
        assert link.wait_connected(1.0)["server"] == "fake-arm"
        link.stop()  # goes to the new connection
        assert any(m["type"] == "stop" for m in second.received)
    finally:
        second.close()


def test_wrong_protocol_is_dropped(link):
    fake = FakeArm(link.port, hello={"type": "hello", "protocol": 2, "server": "future", "live": False,
                                     "watchdog_ms": 500, "state_hz": 10.0})
    try:
        with pytest.raises(LinkError):
            link.wait_connected(0.3)
    finally:
        fake.close()


def test_request_without_arm_fails_fast():
    with ArmLink("127.0.0.1", 0) as lk:
        with pytest.raises(LinkError, match="no arm"):
            lk.stop()
