"""One PC, the whole chain: brain -> Rust arm-server (--live on a pty) -> fw_emu -> stub sim.

    brain CLI / ArmLink --TCP--> mycobot-arm --serial <pty> <--> fw_emu --TCP--> stub_sim

The arm-server binary is the unchanged one; every check it makes (limits, 20 deg from measured,
freshness, watchdog) runs for real. ``--live`` is only ever given a pty this test created.
Skipped when ``arm-server/target/{debug,release}/mycobot-arm`` is not built or the vendor flock
``/tmp/mycobot_lock`` is taken (two arm-servers cannot run at once, by design).
"""

import json
import os
import random
import signal
import socket
import statistics
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(REPO, "brain", "src"))

import fw_emu  # noqa: E402
from stub_sim import StubSim  # noqa: E402

try:
    from mycobot_brain.link import ArmLink, Rejected  # noqa: E402
except Exception as exc:  # brain needs Python 3.10+
    ArmLink = None
    IMPORT_ERROR = exc

ARM_BIN = next((p for p in (os.environ.get("MYCOBOT_ARM", ""),
                            os.path.join(REPO, "arm-server/target/release/mycobot-arm"),
                            os.path.join(REPO, "arm-server/target/debug/mycobot-arm")) if p and os.path.exists(p)), None)

START = [0.0] * 6


def free_port():
    """A free port *below* the ephemeral range. The arm-server redials the brain every second, and
    to an ephemeral port nobody listens on a loopback connect can pair with itself (TCP
    simultaneous open) and then sit on the port the brain wants to bind."""
    with open("/proc/sys/net/ipv4/ip_local_port_range") as fh:
        low = int(fh.read().split()[0])
    for _ in range(100):
        port = random.randrange(20000, min(low, 30000))
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("no free port")


def wait_for(cond, timeout=3.0, step=0.01):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


@unittest.skipIf(ARM_BIN is None, "build the arm-server first: cd arm-server && cargo build")
@unittest.skipIf(ArmLink is None, "the brain package needs Python 3.10+")
class Chain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = StubSim(port=0, start_deg=START).start()
        cls.emu = fw_emu.Service(("127.0.0.1", cls.sim.port), link=None).start()
        # The one place --live is given a path: the pty we just made.
        assert cls.emu.path.startswith("/dev/pts/"), cls.emu.path
        cls.brain_port = free_port()
        cls.arm = subprocess.Popen(
            [ARM_BIN, "--brain", f"127.0.0.1:{cls.brain_port}", "--live", "--assume-yes", "--serial", cls.emu.path],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        time.sleep(0.3)
        if cls.arm.poll() is not None:
            err = cls.arm.stderr.read()
            cls.emu.close()
            cls.sim.close()
            if "mycobot_lock" in err:
                raise unittest.SkipTest("another arm-server holds /tmp/mycobot_lock: " + err.strip())
            raise RuntimeError("arm-server did not start: " + err)

    @classmethod
    def tearDownClass(cls):
        cls.arm.send_signal(signal.SIGINT)
        try:
            cls.arm.wait(5)
        except subprocess.TimeoutExpired:
            cls.arm.kill()
        cls.log = cls.arm.stderr.read()
        cls.arm.stderr.close()
        cls.emu.close()
        cls.sim.close()

    def setUp(self):
        with self.sim._lock:
            self.sim.q, self.sim.target, self.sim.rate = list(START), list(START), 0.0
            self.sim.gripper = self.sim.gripper_target = 100.0
            self.sim.moves.clear()
            self.sim.grips.clear()
        self.stops0 = self.sim.stops

    def cli(self, *args, timeout=40):
        env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "brain", "src"), PYTHONUNBUFFERED="1")
        p = subprocess.run(
            [sys.executable, "-m", "mycobot_brain.cli", "--host", "127.0.0.1", "--port", str(self.brain_port),
             "--wait", "30", *args], env=env, capture_output=True, text=True, timeout=timeout)
        lines = [json.loads(line) for line in p.stdout.splitlines() if line.startswith("{")]
        return p.returncode, lines, p.stderr

    def link(self):
        link = ArmLink("127.0.0.1", self.brain_port).start()
        self.addCleanup(self.close_link, link)
        hello = link.wait_connected(30.0)
        self.assertEqual((hello["protocol"], hello["live"]), (1, True))
        link.wait_state()
        return link

    def close_link(self, link):
        """``ArmLink.close()`` closes its listener while a thread sits in accept(), and Linux keeps the
        port listening until that accept returns; poke it so the next ``bind`` on the port works."""
        link.close()
        try:
            socket.create_connection(("127.0.0.1", self.brain_port), timeout=1.0).close()
        except OSError:
            pass
        time.sleep(0.05)

    # --- the brain CLI

    def test_status_shows_live_hello_and_the_sims_angles(self):
        rc, lines, err = self.cli("status", "--seconds", "2")
        self.assertEqual(rc, 0, err)
        hello, states = lines[0], [m for m in lines[1:] if m.get("type") == "state"]
        self.assertEqual((hello["type"], hello["protocol"], hello["live"]), ("hello", 1, True))
        self.assertGreaterEqual(len(states), 15)
        for s in states:
            self.assertEqual(s["angles_deg"], START)
            self.assertFalse(s["gate_open"])
            self.assertTrue(s["live"])
        ages = [s["read_age_ms"] for s in states]
        print(f"\n  status: {len(states)} states in 2 s, read_age_ms median {statistics.median(ages)} max {max(ages)}",
              end=" ")
        self.assertLess(max(ages), 100)

    def test_move_joints_10_degrees_is_tracked_and_the_exit_stops_the_arm(self):
        rc, lines, err = self.cli("move-joints", "10", "0", "0", "0", "0", "0", "--settle", "2.5")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.sim.moves, [{"angles_deg": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0], "speed": 20}])
        states = [m for m in lines if m.get("type") == "state"]
        after = states[next(i for i, m in enumerate(lines) if "sent" in m) - 1:]  # states after the send
        j1 = [s["angles_deg"][0] for s in after if s["angles_deg"]]
        self.assertAlmostEqual(j1[-1], 10.0, delta=0.3)
        self.assertTrue(any(1.0 < v < 9.0 for v in j1), f"no in-between sample: {j1}")
        self.assertTrue(any(s["gate_open"] for s in after))
        print(f"\n  move-joints 10: measured J1 {[round(v, 1) for v in j1]}", end=" ")
        # The CLI ends: the arm-server loses the brain and calls stop() on the way out.
        self.assertTrue(wait_for(lambda: self.sim.stops > self.stops0, timeout=4.0))

    def test_a_25_degree_jump_is_rejected_and_a_19_degree_one_is_not(self):
        with self.sim._lock:
            self.sim.q = self.sim.target = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rc, lines, err = self.cli("move-joints", "35", "0", "0", "0", "0", "0", "--settle", "0.5")
        self.assertEqual(rc, 1)
        self.assertIn("J1 is 25.0 deg from the measured angle", err)
        self.assertEqual(self.sim.moves, [], "a refused target must never reach the sim")
        rc, lines, err = self.cli("move-joints", "29", "0", "0", "0", "0", "0", "--settle", "0.5")
        self.assertEqual(rc, 0, err)
        self.assertEqual([m["angles_deg"][0] for m in self.sim.moves], [29.0])

    def test_speed_over_the_arm_servers_cap_is_rejected(self):
        rc, _, err = self.cli("move-joints", "5", "0", "0", "0", "0", "0", "--speed", "80", "--settle", "0.5")
        self.assertEqual(rc, 1)
        self.assertIn("speed 80", err)
        self.assertEqual(self.sim.moves, [])

    # --- the library the CLI is built on: gripper, watchdog, link loss

    def test_gripper_goes_through_the_gate_and_the_checks(self):
        link = self.link()
        with self.assertRaises(Rejected) as closed:
            link.gripper(0, 50)
        self.assertIn("gate closed", str(closed.exception))
        link.enable()
        link.gripper(0, 50)
        self.assertTrue(wait_for(lambda: self.sim.gripper == 0.0, timeout=3.0), self.sim.gripper)
        with self.assertRaises(Rejected):
            link.gripper(101, 50)
        link.gripper(100, 50)
        self.assertTrue(wait_for(lambda: self.sim.gripper == 100.0, timeout=3.0), self.sim.gripper)
        self.assertEqual([g["value"] for g in self.sim.grips], [0.0, 100.0])

    def test_watchdog_stops_the_arm_and_closes_the_gate_until_enabled_again(self):
        link = self.link()
        link.enable()
        link.move_joints([15, 0, 0, 0, 0, 0], speed=10)  # 1.5 s of travel at 10 deg/s
        self.assertTrue(wait_for(lambda: self.sim.q[0] > 2.0))
        t0 = time.monotonic()
        link.pinging.clear()  # the brain goes quiet on a live connection
        self.assertTrue(wait_for(lambda: not link.latest_state()["gate_open"], timeout=3.0))
        t_gate = time.monotonic() - t0
        self.assertTrue(wait_for(lambda: self.sim.stops > self.stops0, timeout=1.0))
        t_stop = time.monotonic() - t0
        reason = link.latest_state()["gate_reason"]
        self.assertIn("watchdog", reason)
        held = self.sim.q[0]
        time.sleep(0.4)
        print(f"\n  watchdog: gate closed {t_gate * 1000:.0f} ms and sim stop {t_stop * 1000:.0f} ms after the last "
              f"ping (500 ms setting); arm held at J1 {held:.1f} deg of 15", end=" ")
        self.assertGreater(t_gate, 0.4)
        self.assertLess(t_gate, 1.2)
        self.assertLess(held, 14.0)
        self.assertAlmostEqual(self.sim.q[0], held, delta=0.3)
        # Quiet is not an open door: moves are refused until the brain enables again.
        link.pinging.set()
        with self.assertRaises(Rejected) as closed:
            link.move_joints([5, 0, 0, 0, 0, 0])
        self.assertIn("gate closed", str(closed.exception))
        link.enable()

    def test_losing_the_brain_stops_the_arm(self):
        link = self.link()
        link.enable()
        link.move_joints([15, 0, 0, 0, 0, 0], speed=10)
        self.assertTrue(wait_for(lambda: self.sim.q[0] > 2.0))
        link.close()
        self.assertTrue(wait_for(lambda: self.sim.stops > self.stops0, timeout=2.0))
        held = self.sim.q[0]
        time.sleep(0.4)
        self.assertAlmostEqual(self.sim.q[0], held, delta=0.3)
        self.assertLess(held, 14.0)

    def test_zz_a_dead_sim_closes_the_gate_and_the_chain_recovers(self):
        link = self.link()
        link.enable()
        port = self.sim.port
        self.sim.close()  # the bridge dies; fw_emu stops answering get_angles
        self.assertTrue(wait_for(lambda: not link.latest_state()["gate_open"], timeout=3.0))
        self.assertIn("stale", link.latest_state()["gate_reason"])
        with self.assertRaises(Rejected):
            link.move_joints([5, 0, 0, 0, 0, 0])
        type(self).sim = StubSim(port=port, start_deg=START).start()  # the sim is back
        self.assertTrue(wait_for(lambda: link.latest_state()["angles_deg"] == START, timeout=3.0))
        link.enable()  # re-opening is explicit


if __name__ == "__main__":
    unittest.main()
