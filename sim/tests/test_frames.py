"""fw_emu's serial codec held to the bytes pymycobot 4.0.7 wrote (``arm-server/tests/golden``)."""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import fw_emu  # noqa: E402
from fw_emu import Frame, FrameReader, FwEmu, StateCache  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(ROOT), "arm-server", "tests", "golden", "pymycobot-4.0.7.json")


class Commands:
    def __init__(self):
        self.calls = []

    def move_joints(self, angles, speed):
        self.calls.append(("move_joints", angles, speed))

    def gripper(self, value, speed):
        self.calls.append(("gripper", value, speed))

    def stop(self):
        self.calls.append(("stop",))


def emu_with_state(angles=None, gripper=100.0, age=0.0):
    clock = [100.0]
    cache = StateCache(clock=lambda: clock[0])
    if angles is not None:
        cache.update({"angles_deg": angles, "gripper": gripper})
    clock[0] += age
    commands = Commands()
    return FwEmu(cache, commands, max_state_age=0.2), commands


def parse_one(raw):
    reader = FrameReader()
    reader.push(raw)
    f = reader.next_frame()
    assert reader.next_frame() is None and not reader.buf, raw.hex()
    return f


class GoldenRequests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(GOLDEN) as fh:
            cls.golden = json.load(fh)

    def test_every_pymycobot_request_is_understood(self):
        requests = self.golden["requests"]
        self.assertGreaterEqual(len(requests), 10)
        for r in requests:
            with self.subTest(call=r["call"], args=r["args"]):
                raw = bytes.fromhex(r["frame"])
                f = parse_one(raw)
                self.assertEqual(fw_emu.frame(f.cmd, f.data), raw)  # emit(parse(x)) == x
                emu, commands = emu_with_state([0.0] * 6)
                reply = emu.handle(f)
                getattr(self, "check_" + r["call"])(r["args"], f, reply, commands)

    def check_get_angles(self, args, f, reply, commands):
        self.assertEqual(reply, fw_emu.angles_reply([0.0] * 6))
        self.assertEqual(commands.calls, [])

    def check_send_angles(self, args, f, reply, commands):
        self.assertIsNone(reply)
        (name, angles, speed), = commands.calls
        self.assertEqual(name, "move_joints")
        # pymycobot sends int(angle * 100): the firmware only ever sees hundredths, truncated.
        self.assertEqual(angles, [int(a * 100) / 100.0 for a in args[0]])
        self.assertEqual(speed, args[1])

    def check_stop(self, args, f, reply, commands):
        self.assertIsNone(reply)
        self.assertEqual(commands.calls, [("stop",)])

    def check_set_fresh_mode(self, args, f, reply, commands):
        self.assertIsNone(reply)
        self.assertEqual(commands.calls, [])

    def check_set_gripper_value(self, args, f, reply, commands):
        self.assertIsNone(reply)
        self.assertEqual(commands.calls, [("gripper", args[0], args[1])])

    def check_get_gripper_value(self, args, f, reply, commands):
        self.assertEqual(parse_one(reply), Frame(0x65, bytes([0, 100])))

    def check_is_power_on(self, args, f, reply, commands):
        self.assertEqual(parse_one(reply), Frame(0x12, b"\x01"))

    def check_get_error_information(self, args, f, reply, commands):
        self.assertEqual(parse_one(reply), Frame(0x07, b"\x00"))


class GoldenReply(unittest.TestCase):
    def test_get_angles_reply_is_byte_identical_to_the_recorded_one(self):
        with open(GOLDEN) as fh:
            replies = json.load(fh)["replies"]
        self.assertTrue(replies)
        for r in replies:
            self.assertEqual(fw_emu.angles_reply(r["result"]).hex(), r["reply"])

    def test_the_reply_goes_through_the_arm_servers_decoder_rules(self):
        # frame.rs decode_angles: 12 data bytes, and a vector holding exactly -1.00 deg is "no reading".
        f = parse_one(fw_emu.angles_reply([12.34, -45.68, 90.0, -0.01, 150.0, -179.99]))
        self.assertEqual((f.cmd, len(f.data)), (0x20, 12))

    def test_angles_outside_i16_are_clamped_not_wrapped(self):
        f = parse_one(fw_emu.angles_reply([400.0, -400.0, 0, 0, 0, 0]))
        self.assertEqual(f.data[:4], bytes([0x7F, 0xFF, 0x80, 0x00]))


class Reader(unittest.TestCase):
    def test_skips_noise_and_split_frames_like_frame_rs(self):
        r = FrameReader()
        r.push(bytes([0x00, 0xFE, 0xFE, 0x63, 0x01, 0xFE]))  # bad length, then half a header
        self.assertIsNone(r.next_frame())
        r.push(bytes([0xFE, 0x02, 0x29]))
        self.assertIsNone(r.next_frame())
        r.push(bytes([0xFA]))
        self.assertEqual(r.next_frame(), Frame(0x29, b""))
        self.assertIsNone(r.next_frame())

    def test_drops_a_frame_without_footer(self):
        r = FrameReader()
        r.push(bytes([0xFE, 0xFE, 0x02, 0x29, 0x00]))
        r.push(fw_emu.frame(0x29))
        self.assertEqual(r.next_frame(), Frame(0x29, b""))

    def test_three_resends_are_three_frames(self):
        r = FrameReader()
        r.push(fw_emu.frame(0x29) * 3)
        self.assertEqual([r.next_frame() for _ in range(4)], [Frame(0x29, b"")] * 3 + [None])

    def test_a_split_byte_stream_gives_the_same_frames(self):
        raw = bytes.fromhex("fefe0f2200000000000000000000000014fa") + bytes.fromhex("fefe0229fa")
        r = FrameReader()
        got = []
        for b in raw:
            r.push(bytes([b]))
            while (f := r.next_frame()) is not None:
                got.append(f)
        self.assertEqual([f.cmd for f in got], [0x22, 0x29])


class State(unittest.TestCase):
    def test_no_state_no_reply(self):
        emu, _ = emu_with_state(None)
        self.assertIsNone(emu.handle(Frame(0x20, b"")))

    def test_state_older_than_the_limit_is_not_served(self):
        emu, _ = emu_with_state([1.0] * 6, age=0.25)
        self.assertIsNone(emu.handle(Frame(0x20, b"")))
        self.assertIsNone(emu.handle(Frame(0x65, b"")))

    def test_fresh_state_is_served(self):
        emu, _ = emu_with_state([1.5, 2, 3, 4, 5, 6], age=0.1)
        self.assertEqual(emu.handle(Frame(0x20, b"")), fw_emu.angles_reply([1.5, 2, 3, 4, 5, 6]))

    def test_malformed_commands_do_nothing(self):
        emu, commands = emu_with_state([0.0] * 6)
        self.assertIsNone(emu.handle(Frame(0x22, b"\x00" * 5)))
        self.assertIsNone(emu.handle(Frame(0x67, b"\x28")))
        self.assertIsNone(emu.handle(Frame(0x7F, b"")))
        self.assertEqual(commands.calls, [])


if __name__ == "__main__":
    unittest.main()
