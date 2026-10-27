import json
import socket
import threading
import time
from pathlib import Path

import pytest

from mycobot_brain.link import ArmLink

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schema"

HELLO = {"type": "hello", "protocol": 1, "server": "fake-arm", "live": False, "watchdog_ms": 500, "state_hz": 10.0}


def wait_until(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


class FakeArm:
    """Plays the arm server over a real socket. Replies ok to every request
    unless ``refuse`` names its type; records every line the brain sends."""

    def __init__(self, port, hello=HELLO, refuse=()):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.received = []
        self.refuse = set(refuse)
        self.silent = False
        self.send(hello)
        threading.Thread(target=self._read, daemon=True).start()

    def send(self, msg):
        self.sock.sendall((json.dumps(msg) + "\n").encode())

    def state(self, seq, angles=(0.0,) * 6, gate_open=False):
        self.send({"type": "state", "seq": seq, "angles_deg": list(angles), "read_age_ms": 1,
                   "gate_open": gate_open, "gate_reason": "test", "live": False})

    def _read(self):
        buf = b""
        while True:
            try:
                chunk = self.sock.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                msg = json.loads(raw)
                self.received.append(msg)
                if "id" in msg and not self.silent:
                    if msg["type"] in self.refuse:
                        self.send({"type": "reply", "id": msg["id"], "ok": False, "error": "refused by fake"})
                    else:
                        self.send({"type": "reply", "id": msg["id"], "ok": True})

    def close(self):
        self.sock.close()


@pytest.fixture
def link():
    with ArmLink("127.0.0.1", 0, ping_interval=0.02) as lk:
        yield lk


@pytest.fixture
def arm(link):
    fake = FakeArm(link.port)
    link.wait_connected(2.0)
    yield fake
    fake.close()
