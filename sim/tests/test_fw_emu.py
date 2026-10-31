"""fw_emu over a real pty against the stub sim: what the arm-server's SerialArm would see."""

import os
import select
import statistics
import sys
import threading
import time
import tty
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fw_emu  # noqa: E402
from bridge_client import BridgeClient, BridgeError  # noqa: E402
from fw_emu import CommandQueue, FrameReader  # noqa: E402
from stub_sim import StubSim  # noqa: E402


def wait_for(cond, timeout=3.0, step=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


class Port:
    """The arm-server's end: the pty slave, raw, like serialport leaves it."""

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        self.reader = FrameReader()

    def close(self):
        os.close(self.fd)

    def query(self, request, timeout=0.5):
        """Write ``request``; the first frame with its command byte, or ``None`` after ``timeout``."""
        os.write(self.fd, request)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if select.select([self.fd], [], [], 0.005)[0]:
                self.reader.push(os.read(self.fd, 256))
            while (f := self.reader.next_frame()) is not None:
                if f.cmd == request[3]:
                    return f
        return None


class OverAPty(unittest.TestCase):
    def setUp(self):
        self.sim = StubSim(port=0, start_deg=[10.0, -5.0, 0.0, 0.0, 0.0, 0.0]).start()
        self.service = fw_emu.Service(("127.0.0.1", self.sim.port), link=None).start()
        self.port = Port(self.service.path)
        self.assertTrue(wait_for(lambda: self.service.cache.get(0.2) is not None), "no state from the stub")

    def tearDown(self):
        self.port.close()
        self.service.close()
        self.sim.close()

    def test_get_angles_comes_from_the_sims_state(self):
        f = self.port.query(fw_emu.frame(0x20))
        self.assertEqual(f.data, fw_emu.angles_reply([10.0, -5.0, 0, 0, 0, 0])[4:-1])

    def test_get_angles_is_far_inside_the_arm_servers_50_ms(self):
        times = []
        for _ in range(100):
            t0 = time.monotonic()
            self.assertIsNotNone(self.port.query(fw_emu.frame(0x20)))
            times.append((time.monotonic() - t0) * 1000)
        print(f"\n  get_angles round trip over the pty: median {statistics.median(times):.2f} ms, "
              f"max {max(times):.2f} ms (100 queries)", end=" ")
        self.assertLess(max(times), 25.0)

    def test_send_angles_reaches_the_sim_as_move_joints_and_the_arm_follows(self):
        os.write(self.port.fd, fw_emu.frame(0x22, b"".join(v.to_bytes(2, "big", signed=True) for v in
                                                          (2000, -500, 0, 0, 0, 0)) + bytes([40])))
        self.assertTrue(wait_for(lambda: self.sim.moves))
        self.assertEqual(self.sim.moves[-1], {"angles_deg": [20.0, -5.0, 0.0, 0.0, 0.0, 0.0], "speed": 40})
        # 10 deg at 40 deg/s with a first-order tail: well inside 2 s.
        self.assertTrue(wait_for(lambda: abs(self.sim.q[0] - 20.0) < 0.2, timeout=2.0), self.sim.q)
        f = self.port.query(fw_emu.frame(0x20))
        self.assertAlmostEqual(FrameReaderAngles(f)[0], 20.0, delta=0.5)

    def test_gripper_and_stop_reach_the_sim(self):
        os.write(self.port.fd, fw_emu.frame(0x67, bytes([0, 50])))
        self.assertTrue(wait_for(lambda: self.sim.grips))
        self.assertEqual(self.sim.grips[-1], {"value": 0.0, "speed": 50})
        self.assertTrue(wait_for(lambda: self.sim.gripper == 0.0, timeout=2.0), self.sim.gripper)
        self.assertTrue(wait_for(lambda: self.port.query(fw_emu.frame(0x65)).data == bytes([0, 0])))
        os.write(self.port.fd, fw_emu.frame(0x29))
        self.assertTrue(wait_for(lambda: self.sim.stops == 1))

    def test_stop_holds_the_arm_where_it_is(self):
        os.write(self.port.fd, fw_emu.frame(0x22, (5000).to_bytes(2, "big") + bytes(10) + bytes([10])))  # 50 deg, 10 deg/s
        self.assertTrue(wait_for(lambda: self.sim.q[0] > 11.0))
        os.write(self.port.fd, fw_emu.frame(0x29))
        self.assertTrue(wait_for(lambda: self.sim.stops == 1))
        held = self.sim.q[0]
        time.sleep(0.3)
        self.assertAlmostEqual(self.sim.q[0], held, delta=0.5)
        self.assertLess(held, 20.0)

    def test_silence_when_the_sim_goes_away_and_speech_when_it_is_back(self):
        port = self.sim.port
        self.sim.close()
        self.assertTrue(wait_for(lambda: self.service.cache.get(0.2) is None, timeout=2.0))
        self.assertIsNone(self.port.query(fw_emu.frame(0x20), timeout=0.15), "must stay silent on stale state")
        self.sim = StubSim(port=port, start_deg=[1.0] * 6).start()
        self.assertTrue(wait_for(lambda: self.service.cache.get(0.2) is not None, timeout=3.0))
        f = self.port.query(fw_emu.frame(0x20), timeout=0.5)
        self.assertEqual(FrameReaderAngles(f)[0], 1.0)

    def test_the_port_can_be_closed_and_reopened_while_the_emulator_runs(self):
        self.port.close()
        self.port = Port(self.service.path)
        self.assertIsNotNone(self.port.query(fw_emu.frame(0x20)))


def FrameReaderAngles(f):
    return [int.from_bytes(f.data[i:i + 2], "big", signed=True) / 100.0 for i in range(0, 12, 2)]


class SlowClient:
    """A bridge client whose ``call`` blocks until released, to test the queue order."""

    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        self.entered = threading.Event()

    def call(self, request):
        self.calls.append(request["op"] + (":" + str(request["angles_deg"][0]) if "angles_deg" in request else ""))
        self.entered.set()
        self.release.wait(2.0)
        return {"ok": True}


class QueueOrder(unittest.TestCase):
    def test_stop_goes_first_and_drops_what_waited(self):
        client = SlowClient()
        q = CommandQueue(client)
        q.start()
        q.move_joints([1] * 6, 20)
        self.assertTrue(client.entered.wait(1.0))  # the worker holds move 1 inside call()
        q.move_joints([2] * 6, 20)
        q.move_joints([3] * 6, 20)
        q.stop()
        q.move_joints([4] * 6, 20)  # after the stop: allowed through
        client.release.set()
        q.close()
        q.join(2.0)
        self.assertEqual(client.calls, ["move_joints:1", "stop", "move_joints:4"])

    def test_speed_is_clamped_to_the_bridges_range(self):
        client = SlowClient()
        client.release.set()
        seen = []
        client.call = lambda req: seen.append(req) or {"ok": True}
        q = CommandQueue(client)
        q.start()
        q.move_joints([0] * 6, 0)
        q.gripper(40, 200)
        q.close()
        q.join(2.0)
        self.assertEqual([r["speed"] for r in seen], [1, 100])


class Bridge(unittest.TestCase):
    def test_an_error_reply_raises_and_a_dead_bridge_raises(self):
        sim = StubSim(port=0).start()
        client = BridgeClient(("127.0.0.1", sim.port), timeout=0.5)
        with self.assertRaises(BridgeError):
            client.call({"op": "nope"})
        self.assertTrue(client.call({"op": "state"})["ok"])
        sim.close()
        with self.assertRaises(BridgeError):
            client.call({"op": "state"})


if __name__ == "__main__":
    unittest.main()
