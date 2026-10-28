import contextlib
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from mycobot_brain.camera import CameraError, CameraLink

CAMERA_PUSH = Path(__file__).resolve().parents[2] / "camera-push" / "camera_push.py"
HELLO = {"type": "hello", "protocol": 1, "server": "fake-camera", "source": "test", "rate_hz": 20}


def jpeg(n):
    return b"\xff\xd8" + n.to_bytes(4, "big") + b"\xff\xd9"


class FakeCamera:
    """Plays camera_push.py over a real socket: ``send(n)`` pushes frame n."""

    def __init__(self, port, hello=HELLO):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.sock.sendall((json.dumps(hello) + "\n").encode())

    def send(self, n, payload=None):
        body = jpeg(n) if payload is None else payload
        header = {"type": "frame", "seq": n, "jpeg_bytes": len(body), "sent_unix_ms": 0}
        self.sock.sendall((json.dumps(header) + "\n").encode() + body)

    def stream(self, period=0.02):
        """Keep sending frames in the background until closed."""
        def run():
            n = 0
            while True:
                n += 1
                try:
                    self.send(n)
                except OSError:
                    return
                time.sleep(period)

        threading.Thread(target=run, daemon=True).start()
        return self

    def close(self):
        self.sock.close()


@pytest.fixture
def camera():
    with CameraLink("127.0.0.1", 0) as cam:
        yield cam


def test_newest_frame_wins_and_after_filters_old_ones(camera):
    fake = FakeCamera(camera.port)
    assert camera.wait_connected(2.0)["server"] == "fake-camera"
    fake.send(1)
    assert camera.wait_frame(timeout=2.0).seq == 1
    mark = time.monotonic()
    with pytest.raises(CameraError, match="no new camera frame"):
        camera.wait_frame(after=mark, timeout=0.2)
    fake.send(2)
    frame = camera.wait_frame(after=mark, timeout=2.0)
    assert (frame.seq, frame.jpeg) == (2, jpeg(2))
    assert frame.age() < 1.0
    fake.close()


def test_a_new_connection_replaces_the_old_and_its_frame(camera):
    old = FakeCamera(camera.port)
    camera.wait_connected(2.0)
    old.send(7)
    camera.wait_frame(timeout=2.0)
    new = FakeCamera(camera.port, dict(HELLO, server="second"))
    deadline = time.monotonic() + 2.0
    while camera.wait_connected(2.0)["server"] != "second" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert camera.latest() is None  # frame 7 belonged to the dropped connection
    with contextlib.suppress(OSError):
        old.send(8)  # the old socket is closed on our side; nothing it says counts
    new.send(1)
    assert camera.wait_frame(timeout=2.0).seq == 1
    old.close()
    new.close()


@pytest.mark.parametrize("payload", [b"not a jpeg", b""])
def test_a_bad_frame_drops_the_connection(camera, payload):
    fake = FakeCamera(camera.port)
    camera.wait_connected(2.0)
    fake.send(1, payload)
    with pytest.raises(CameraError):
        camera.wait_frame(timeout=0.5)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            fake.send(2)
        except OSError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the brain kept a connection that sent a bad frame")
    fake.close()


def test_wrong_protocol_is_refused(camera):
    fake = FakeCamera(camera.port, dict(HELLO, protocol=2))
    with pytest.raises(CameraError):
        camera.wait_connected(0.5)
    fake.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_camera_push_streams_whole_jpegs_from_ffmpeg(camera):
    proc = subprocess.Popen(
        [sys.executable, str(CAMERA_PUSH), "--brain", f"127.0.0.1:{camera.port}", "--test-pattern",
         "--rate", "10", "--once"],
        stderr=subprocess.PIPE,
    )
    try:
        hello = camera.wait_connected(10.0)
        assert (hello["server"], hello["source"], hello["rate_hz"]) == ("camera-push", "testsrc", 10.0)
        first = camera.wait_frame(timeout=10.0)
        t0 = time.monotonic()
        frames = [first]
        while len(frames) < 6:
            frames.append(camera.wait_frame(after=frames[-1].received, timeout=2.0))
        elapsed = time.monotonic() - t0
        assert [f.seq for f in frames] == list(range(first.seq, first.seq + 6))
        for f in frames:
            assert f.jpeg.startswith(b"\xff\xd8") and f.jpeg.endswith(b"\xff\xd9")
            assert f.jpeg.count(b"\xff\xd8") == 1  # one image per frame, not two glued together
        assert 0.25 < elapsed < 1.5  # five periods at 10 Hz
    finally:
        camera.close()  # --once: camera_push exits when this connection ends
        try:
            proc.wait(10)
        finally:
            proc.kill()
    assert proc.returncode == 0, proc.stderr.read().decode()
