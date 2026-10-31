"""G2 link check, brain side: run this where the brain runs, with the sim side up.

    ssh spark 'cd ~/manipulation/mycobot-link/brain && ~/.local/bin/uv run python -' < sim/g2_check.py

It binds the arm port like ``mycobot-brain`` does, waits for the arm-server (``sim/run_sim_link.sh``
on the sim host dials in), and walks the gate: ``enable``, a 10 deg move whose measured angles it
watches, a 25 deg jump that must be refused, the gripper, and the watchdog (heartbeats stopped on a
live connection). Prints one JSON object per step and a summary; exit code 1 if a step failed.
The arm-server and fw_emu logs on the sim host are the independent record of what reached the sim.

Only the brain package's public ``ArmLink`` is used; nothing in the checkout is written.
"""

import json
import statistics
import sys
import time

from mycobot_brain.link import ArmLink, LinkError, Rejected

HOST, PORT = "0.0.0.0", 47900
results = []


def step(name, ok, **info):
    results.append(ok)
    print(json.dumps({"step": name, "ok": bool(ok), **info}), flush=True)


def watch(link, seconds):
    """(seconds since start, state) for every state message in the next ``seconds``."""
    t0, seq, out = time.monotonic(), None, []
    while time.monotonic() - t0 < seconds:
        try:
            state = link.wait_state(after_seq=seq, timeout=1.0)
        except LinkError:
            break
        seq = state["seq"]
        out.append((round(time.monotonic() - t0, 2), state))
    return out


def main():
    with ArmLink(HOST, PORT) as link:
        print("waiting for the arm server on port", PORT, flush=True)
        hello = link.wait_connected(60.0)
        state = link.wait_state()
        q0 = state["angles_deg"]
        step("hello", hello["protocol"] == 1 and hello["live"] is True and q0 is not None, hello=hello, angles_deg=q0)

        # Request/reply round trip across the network, through the arm-server's loop.
        rtts = []
        for _ in range(20):
            t0 = time.monotonic()
            link.enable()
            link.disable()
            rtts.append((time.monotonic() - t0) * 1000 / 2)
        step("request_reply_ms", True, median=round(statistics.median(rtts), 2), max=round(max(rtts), 2), n=len(rtts))

        # States arrive at state_hz.
        states = watch(link, 3.0)
        gaps = [b[0] - a[0] for a, b in zip(states, states[1:])]
        step("state_rate", 8 <= len(states) / 3.0 <= 12, hz=round(len(states) / 3.0, 1),
             median_gap_ms=round(statistics.median(gaps) * 1000), max_read_age_ms=max(s["read_age_ms"] for _, s in states))

        link.enable()
        target = [q0[0] + 10.0] + q0[1:]
        link.move_joints(target, speed=20)
        trace = [(t, s["angles_deg"][0]) for t, s in watch(link, 2.5)]
        final = trace[-1][1]
        moving = any(q0[0] + 1 < v < q0[0] + 9 for _, v in trace)
        step("move_10_deg_tracks", abs(final - target[0]) < 0.5 and moving,
             target=target[0], final=final, trace=[(t, round(v, 1)) for t, v in trace[::2]])

        measured = link.wait_state()["angles_deg"]
        try:
            link.move_joints([measured[0] + 25.0] + measured[1:], speed=20)
            step("jump_25_deg_rejected", False, error="accepted")
        except Rejected as exc:
            time.sleep(0.5)
            after = link.wait_state()["angles_deg"]
            step("jump_25_deg_rejected", "from the measured angle" in str(exc) and abs(after[0] - measured[0]) < 0.5,
                 error=str(exc), measured_before=measured[0], measured_after=after[0])

        try:
            link.gripper(0, 50)
            time.sleep(1.5)
            link.gripper(100, 50)
            step("gripper_0_then_100_accepted", True)
        except LinkError as exc:
            step("gripper_0_then_100_accepted", False, error=str(exc))
        try:
            link.gripper(101, 50)
            step("gripper_101_rejected", False, error="accepted")
        except Rejected as exc:
            step("gripper_101_rejected", True, error=str(exc))

        # Watchdog: a slow 12 deg move, then the brain falls silent on a live connection.
        before = link.wait_state()["angles_deg"]
        link.move_joints([before[0] - 12.0] + before[1:], speed=10)
        time.sleep(0.4)
        t0 = time.monotonic()
        link.pinging.clear()
        deadline, closed_after, reason = t0 + 3.0, None, None
        while time.monotonic() < deadline:
            s = link.latest_state()
            if s is not None and not s["gate_open"]:
                closed_after, reason = time.monotonic() - t0, s["gate_reason"]
                break
            time.sleep(0.01)
        held_a = link.latest_state()["angles_deg"][0]
        time.sleep(0.5)
        held_b = link.latest_state()["angles_deg"][0]
        link.pinging.set()
        step("watchdog_closes_gate_and_stops_arm",
             closed_after is not None and "watchdog" in reason and abs(held_a - held_b) < 0.5 and held_b > before[0] - 11.5,
             gate_closed_after_ms=None if closed_after is None else round(closed_after * 1000), reason=reason,
             held_at=held_b, started_at=before[0], would_have_reached=before[0] - 12.0)
        try:
            link.move_joints([held_b + 1.0] + before[1:], speed=10)
            step("move_refused_until_enable", False, error="accepted")
        except Rejected as exc:
            step("move_refused_until_enable", "gate closed" in str(exc), error=str(exc))
        link.enable()
        link.disable()
        step("enable_again", True)
    print(json.dumps({"summary": f"{sum(results)}/{len(results)} steps ok"}), flush=True)
    return 0 if all(results) else 1


sys.exit(main())
