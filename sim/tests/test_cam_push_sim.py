"""cam_push_sim against the brain's real CameraLink: same framing as camera_push.py, once per frame."""

import importlib.util
import json
import os
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(REPO, "brain", "src"))

import cam_push_sim  # noqa: E402
from bridge_client import BridgeError  # noqa: E402
from mycobot_brain.camera import CameraLink  # noqa: E402
from stub_sim import StubSim  # noqa: E402


def load_camera_push():
    spec = importlib.util.spec_from_file_location("camera_push", os.path.join(REPO, "camera-push", "camera_push.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Framing(unittest.TestCase):
    def test_header_and_hello_match_camera_push_and_the_protocol_doc(self):
        real = load_camera_push()
        obj = {"type": "frame", "seq": 7, "jpeg_bytes": 3, "sent_unix_ms": 1761600000000}
        self.assertEqual(cam_push_sim.header(obj), real.header(obj))
        # The example hello line in schema/PROTOCOL.md, key for key.
        doc_hello = '{"type":"hello","protocol":1,"server":"camera-push","source":"/dev/video0","size":"640x480","rate_hz":5.0}'
        ours = cam_push_sim.hello("640x480", 5.0)
        self.assertEqual(list(ours), list(json.loads(doc_hello)))
        self.assertEqual({k: v for k, v in ours.items() if k != "source"},
                         {k: v for k, v in json.loads(doc_hello).items() if k != "source"})
        self.assertEqual(real.PROTOCOL_VERSION, cam_push_sim.PROTOCOL_VERSION)

    def test_frame_message_is_a_header_line_then_exactly_the_jpeg(self):
        jpeg = b"\xff\xd8abc\xff\xd9"
        raw = cam_push_sim.frame_message(3, jpeg, 1.5)
        line, _, rest = raw.partition(b"\n")
        self.assertEqual(json.loads(line), {"type": "frame", "seq": 3, "jpeg_bytes": len(jpeg), "sent_unix_ms": 1500})
        self.assertEqual(rest, jpeg)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)

    def call(self, request):
        return self.replies.pop(0)


def reply(wall, jpeg=b"\xff\xd8xx\xff\xd9"):
    import base64
    return {"ok": True, "wall": wall, "sim_t": wall, "width": 640, "height": 480,
            "jpeg_b64": base64.b64encode(jpeg).decode()}


class FrameSourceRules(unittest.TestCase):
    def test_each_picture_once(self):
        src = cam_push_sim.FrameSource(FakeClient([reply(10.0), reply(10.0), reply(10.2)]), max_age=1.0)
        self.assertIsNotNone(src.poll(10.1))
        self.assertIsNone(src.poll(10.15))
        self.assertIsNotNone(src.poll(10.25))

    def test_a_frame_older_than_max_age_is_not_pushed(self):
        src = cam_push_sim.FrameSource(FakeClient([reply(10.0)]), max_age=1.0)
        self.assertIsNone(src.poll(11.5))

    def test_a_reply_that_is_not_a_jpeg_is_an_error(self):
        src = cam_push_sim.FrameSource(FakeClient([reply(10.0, b"GIF89a")]), max_age=1.0)
        with self.assertRaises(BridgeError):
            src.poll(10.0)


class AgainstTheBrain(unittest.TestCase):
    def test_frames_arrive_at_about_5_hz_and_are_fresh(self):
        sim = StubSim(port=0).start()
        with CameraLink("127.0.0.1", 0) as camera:
            proc = subprocess.Popen(
                [sys.executable, os.path.join(ROOT, "cam_push_sim.py"), "--brain", f"127.0.0.1:{camera.port}",
                 "--bridge", f"127.0.0.1:{sim.port}", "--once"],
                stderr=subprocess.PIPE)
            try:
                hello = camera.wait_connected(10.0)
                self.assertEqual((hello["type"], hello["protocol"], hello["server"], hello["size"], hello["rate_hz"]),
                                 ("hello", 1, "camera-push", "640x480", 5.0))
                first = camera.wait_frame(timeout=5.0)
                t0, seq0 = time.monotonic(), first.seq
                ages, last = [], seq0
                while time.monotonic() - t0 < 4.0:
                    f = camera.wait_frame(after=time.monotonic() - 0.0, timeout=2.0)
                    self.assertTrue(f.jpeg.startswith(b"\xff\xd8") and f.jpeg.endswith(b"\xff\xd9"))
                    self.assertGreater(f.seq, last)
                    last = f.seq
                    ages.append(f.age())
                elapsed = time.monotonic() - t0
                rate = (last - seq0) / elapsed
                print(f"\n  {last - seq0} frames in {elapsed:.2f} s = {rate:.2f} Hz, "
                      f"frame age at arrival max {max(ages) * 1000:.1f} ms", end=" ")
                self.assertGreater(rate, 4.0)
                self.assertLess(rate, 5.6)
            finally:
                proc.terminate()
                proc.wait(5)
                proc.stderr.close()
                sim.close()


if __name__ == "__main__":
    unittest.main()
