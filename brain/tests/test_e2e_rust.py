"""The real Rust arm server, dry-run, against the real ArmLink on localhost.

Skipped unless the binary exists: ``cargo build`` in arm-server/ first, or
point MYCOBOT_ARM_BIN at one. Every line the server sends is validated against
the schema, so this is also the check that both sides speak the same protocol.
"""

import http.server
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from conftest import SCHEMA_DIR, wait_until
from mycobot_brain import cli
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


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeVLLM(http.server.ThreadingHTTPServer):
    """Answers every chat completion with the next canned step; keeps the requests."""

    def __init__(self, steps):
        self.steps, self.requests = list(steps), []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"message": {"content": json.dumps(outer.steps.pop(0))}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        super().__init__(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_cli_act_drives_the_rust_server_from_camera_frames(tmp_path, capsys):
    """mycobot-brain act --execute end to end: camera_push (ffmpeg test card) and
    the Rust dry-run server dial in, a fake vLLM sees real JPEGs and steps twice."""
    arm_port, cam_port = _free_port(), _free_port()
    vllm = FakeVLLM([
        {"seen": "test card", "reason": "pan", "done": False, "delta_deg": [4, 0, 0, 0, 0, 0]},
        {"seen": "test card", "reason": "tilt", "done": False, "delta_deg": [0, 0, 0, -3, 0, 0]},
        {"seen": "test card", "reason": "there", "done": True, "delta_deg": [0, 0, 0, 0, 0, 0]},
    ])
    procs = [
        subprocess.Popen([str(BIN), "--brain", f"127.0.0.1:{arm_port}"], stderr=subprocess.DEVNULL),
        subprocess.Popen([sys.executable, str(SCHEMA_DIR.parent / "camera-push/camera_push.py"),
                          "--brain", f"127.0.0.1:{cam_port}", "--test-pattern", "--rate", "10"],
                         stderr=subprocess.DEVNULL),
    ]
    try:
        code = cli.main(["--host", "127.0.0.1", "--port", str(arm_port), "--camera-port", str(cam_port),
                         "--wait", "15", "act", "look at the card", "--execute", "--settle", "0.3",
                         "--record", str(tmp_path), "--vllm-url", f"http://127.0.0.1:{vllm.server_port}/v1"])
    finally:
        for p in procs:
            p.terminate()
            p.wait(5)
        vllm.shutdown()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0, lines
    assert lines[0]["live"] is False and lines[1]["server"] == "camera-push"
    sent = [line["sent"]["angles_deg"] for line in lines if "sent" in line]
    assert sent == [[4.0, 0, 0, 0, 0, 0], [4.0, 0, 0, -3.0, 0, 0]]  # dry-run arm starts at zero
    steps = [line for line in lines if "step" in line and "done" in line]
    assert [s["done"] for s in steps] == [False, False, True]
    assert steps[2]["angles_deg"] == [4.0, 0, 0, -3.0, 0, 0]
    images = [p for p in vllm.requests[1]["messages"][1]["content"] if p["type"] == "image_url"]
    assert len(images) == 2 and all(p["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/") for p in images)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["step-01.jpg", "step-02.jpg", "step-03.jpg", "steps.jsonl"]


def test_a_closed_stderr_does_not_kill_the_server():
    """`mycobot-arm … | tee log` + Ctrl-C kills tee first. The server used to
    panic (exit 101) on its next log line, the "session ended" one, before
    on_disconnect could send stop(). Logging must not be able to stop that."""
    link = ArmLink("127.0.0.1", 0, ping_interval=0.05).start()
    proc = subprocess.Popen([str(BIN), "--brain", f"127.0.0.1:{link.port}"], stderr=subprocess.PIPE)
    try:
        link.wait_connected(10.0)
        proc.stderr.close()  # the reader is gone
        link.close()  # the session ends and the server logs it
        time.sleep(0.5)
        assert proc.poll() is None, f"server died with {proc.returncode}"
        proc.send_signal(signal.SIGINT)
        assert proc.wait(5) == 0
    finally:
        proc.kill()


class AnsweringVLLM(http.server.ThreadingHTTPServer):
    """Answers each chat completion with ``answer(request)``, after ``delay`` seconds."""

    def __init__(self, answer, delay=0.0):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                time.sleep(delay)
                body = json.dumps({"choices": [{"message": {"content": answer(request)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        super().__init__(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()


def _camera_when_bound(port, stop):
    """FakeCamera once the brain listens (cli.main binds it); frames until ``stop``."""
    from test_camera import FakeCamera

    end = time.monotonic() + 15
    while not stop.is_set() and time.monotonic() < end:
        try:
            cam = FakeCamera(port)
            break
        except OSError:
            time.sleep(0.05)
    else:
        return
    n = 0
    while not stop.is_set():
        n += 1
        try:
            cam.send(n)
        except OSError:
            return
        time.sleep(0.05)
    cam.close()


def test_cli_pick_place_runs_a_whole_pick_on_the_rust_server(tmp_path, capsys):
    """mycobot-brain pick-place --execute: the Rust dry-run server takes every
    waypoint, a fake vLLM that sees the scene answers each look 0.8 s later
    (over the 300 ms watchdog, so the heartbeat has to hold the gate), and the
    block is grasped at its centre and released over the box."""
    import scene

    arm_port, cam_port = _free_port(), _free_port()
    run_dir = tmp_path / "run"

    def angles():  # where the brain says the camera was for this request
        lines = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
        return [e for e in lines if e["event"] == "vlm_request"][-1]["angles_deg"]

    vllm = AnsweringVLLM(scene.SceneVLM(scene.load(), angles), delay=0.8)
    server = subprocess.Popen([str(BIN), "--brain", f"127.0.0.1:{arm_port}", "--watchdog-ms", "300",
                               "--state-hz", "50"], stderr=subprocess.PIPE, text=True)
    stop = threading.Event()
    threading.Thread(target=_camera_when_bound, args=(cam_port, stop), daemon=True).start()
    try:
        code = cli.main(["--host", "127.0.0.1", "--port", str(arm_port), "--camera-port", str(cam_port),
                         "--wait", "15", "pick-place", "빨간 블록을 잡아서 갈색 상자 안에 넣어",
                         "--calib", str(scene.write(tmp_path / "calib.json")), "--execute", "--record", str(run_dir),
                         "--settle", "0.1", "--gripper-wait", "0.1",
                         "--vllm-url", f"http://127.0.0.1:{vllm.server_port}/v1"])
    finally:
        stop.set()
        server.terminate()
        _, err = server.communicate(timeout=5)
        vllm.shutdown()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0, (lines[-3:], err[-2000:])
    assert lines[0]["live"] is False
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "result" and events[-1]["ok"] is True
    replies = [e for e in events if e["event"] == "vlm_reply"]
    assert len(replies) == 5 and all(r["latency_s"] >= 0.8 for r in replies)
    moves_and_grips = [e for e in events if e["event"] in ("move", "gripper")]
    grips = [i for i, e in enumerate(moves_and_grips) if e["event"] == "gripper"]
    assert [moves_and_grips[i]["action"] for i in grips] == ["open", "close", "open"]
    grasp, release = moves_and_grips[grips[1] - 1], moves_and_grips[grips[2] - 1]
    assert grasp["phase"] == "descend" and release["phase"] == "lower"
    assert math.dist(grasp["tcp_xyz"], scene.BLOCK) < 0.002
    assert math.dist(release["tcp_xyz"][:2], scene.BOX[:2]) < 0.002
    assert "watchdog: no line from the brain" not in err


def test_cli_drawer_find_runs_on_the_rust_server(tmp_path, capsys):
    """mycobot-brain drawer --execute "노란색 원뿔 찾아서 꺼내 줘": every waypoint through the
    Rust dry-run server (watchdog 300 ms, vLLM answers 0.5 s late), and the
    world the run's own events drive ends with the cone handed over and the
    drawers shut."""
    import cabinet_scene as cs
    from test_drawers import COMMANDS, OBJECTS

    arm_port, cam_port = _free_port(), _free_port()
    run_dir = tmp_path / "run"
    scene = cs.load()
    world = cs.World(scene, OBJECTS)
    vllm = AnsweringVLLM(cs.CabinetVLM(world, run_dir, commands=COMMANDS, order=["r0c0", "r1c2"]), delay=0.5)
    server = subprocess.Popen([str(BIN), "--brain", f"127.0.0.1:{arm_port}", "--watchdog-ms", "300",
                               "--state-hz", "50"], stderr=subprocess.PIPE, text=True)
    stop = threading.Event()
    threading.Thread(target=_camera_when_bound, args=(cam_port, stop), daemon=True).start()
    try:
        code = cli.main(["--host", "127.0.0.1", "--port", str(arm_port), "--camera-port", str(cam_port),
                         "--wait", "15", "drawer", "노란색 원뿔 찾아서 꺼내 줘",
                         "--calib", str(cs.write(tmp_path / "calib.json")), "--execute", "--record", str(run_dir),
                         "--settle", "0.1", "--gripper-wait", "0.1",
                         "--vllm-url", f"http://127.0.0.1:{vllm.server_port}/v1"])
    finally:
        stop.set()
        server.terminate()
        _, err = server.communicate(timeout=5)
        vllm.shutdown()
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 0, (lines[-3:], err[-2000:])
    world.sync(run_dir)
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "result" and events[-1]["ok"] is True
    assert events[-1]["drawers_opened"] == ["r0c0", "r1c2"]
    assert world.in_handover("yellow_cone") and world.all_shut()
    assert not any(world.moved(o) for o in OBJECTS if o != "yellow_cone")
    assert "watchdog: no line from the brain" not in err
