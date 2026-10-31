#!/usr/bin/env python3
"""A stand-in for the simulation's sim-bridge, so the link can be tested without Isaac Lab.

Speaks the same line protocol (``schema/SIM.md``, CONTRACT.md "sim-bridge v0") on 127.0.0.1:47950:
``state``, ``move_joints``, ``gripper``, ``stop``, ``frame``, ``reset``, ``truth``, ``record``.

Physics is deliberately crude and written down so the real sim can be compared against it:

* joints: first-order tracking, ``v = (target - q) / TAU``, then all six joint velocities are scaled
  together so the fastest one does not exceed ``speed`` deg/s (``schema/SIM.md``: the speed -> rate
  map fw_emu assumes). The joints therefore arrive together on a straight line in joint space.
* gripper: linear at ``2 * speed`` units/s towards the target; nothing blocks it.
* ``sim_t`` is wall-clock seconds since start: this stub never runs slower than real time.
* ``frame``: ffmpeg's ``testsrc`` card, a new picture every 1/5 s (the stub's "render rate"), so two
  requests in the same 200 ms return the same ``wall`` like a real renderer would. Without ffmpeg
  it falls back to one flat grey-blue picture.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import socketserver
import subprocess
import sys
import threading
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence

from bridge_client import DEFAULT_BRIDGE

TAU_S = 0.1
GRIPPER_RATE_PER_SPEED = 2.0
RENDER_HZ = 5.0
MOVING_EPS_DEG = 0.05
MOVING_EPS_GRIPPER = 0.5
SOI, EOI = b"\xff\xd8", b"\xff\xd9"

# One 640x480 flat picture, zlib + base64, for hosts without ffmpeg.
_FLAT_JPEG_Z = (
    "eNr7f+P/AwYBLzdPNwZGJgYGRiBk+P+PQcAnsSzZzEDP2FDP0MCI4f9tBmcGDjs7TzvPUDBIiU3JyMhIAQIgVVBQ0NzcDCRBzJqa"
    "5ub+yf3t7c3tkyfPnj17165Nm27efPAfCI4w+DAwAs1HAWyMGEKsAugiDIIYIv8PMAhyMD5gamBmVGJgEmRgFmT4f4uBh5kRyAGy"
    "7RlmMkaaMIyCUTAKRsEoGAUjFsj/vwkA7eU3AA=="
)


def log(msg: str) -> None:
    print(f"stub-sim: {msg}", file=sys.stderr, flush=True)


def testsrc_frames(n: int = 25) -> List[bytes]:
    """``n`` JPEGs of ffmpeg's moving test card, or one flat picture if ffmpeg is missing."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi", "-i",
             f"testsrc=size=640x480:rate={RENDER_HZ:g}", "-frames:v", str(n), "-c:v", "mjpeg", "-q:v", "5",
             "-f", "image2pipe", "pipe:1"],
            capture_output=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return [zlib.decompress(base64.b64decode(_FLAT_JPEG_Z))]
    frames, pos = [], 0
    while (start := out.find(SOI, pos)) >= 0 and (end := out.find(EOI, start + 2)) >= 0:
        frames.append(out[start : end + 2])  # no thumbnails, every FF stuffed: first EOI ends a frame
        pos = end + 2
    return frames or [zlib.decompress(base64.b64decode(_FLAT_JPEG_Z))]


class StubSim:
    """The simulated arm plus the line server. ``start()`` returns once it listens."""

    def __init__(self, host: str = DEFAULT_BRIDGE[0], port: int = DEFAULT_BRIDGE[1],
                 start_deg: Sequence[float] = (0.0,) * 6, hz: float = 200.0) -> None:
        self.start_deg = list(start_deg)
        self.hz = hz
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._frames: Optional[List[bytes]] = None
        self.q = list(self.start_deg)
        self.target = list(self.start_deg)
        self.rate = 0.0
        self.gripper = 100.0
        self.gripper_target = 100.0
        self.gripper_rate = 0.0
        # Everything the bridge was asked, for tests.
        self.moves: List[Dict[str, Any]] = []
        self.grips: List[Dict[str, Any]] = []
        self.stops = 0
        self.requests: List[str] = []
        self._stopping = threading.Event()
        self._clients: List[socket.socket] = []
        sim = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                with sim._lock:
                    sim._clients.append(self.request)
                try:
                    for raw in self.rfile:
                        try:
                            reply = sim.dispatch(json.loads(raw))
                        except (ValueError, TypeError, KeyError) as exc:
                            reply = {"ok": False, "error": f"bad request: {exc}"}
                        self.wfile.write(json.dumps(reply, separators=(",", ":")).encode() + b"\n")
                except OSError:
                    pass  # close() shut this connection down

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server((host, port), Handler)
        self.address = self._server.server_address

    @property
    def port(self) -> int:
        return self.address[1]

    def start(self) -> "StubSim":
        threading.Thread(target=self._server.serve_forever, name="stub-sim-server", daemon=True).start()
        threading.Thread(target=self._physics, name="stub-sim-physics", daemon=True).start()
        return self

    def close(self) -> None:
        """Stop listening and drop every client, as a sim that died would."""
        self._stopping.set()
        self._server.shutdown()
        self._server.server_close()
        with self._lock:
            clients, self._clients = self._clients, []
        for sock in clients:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    # --- physics

    def _physics(self) -> None:
        last = time.monotonic()
        while not self._stopping.wait(1.0 / self.hz):
            now = time.monotonic()
            self._step(now - last)
            last = now

    def _step(self, dt: float) -> None:
        with self._lock:
            v = [(t - q) / TAU_S for q, t in zip(self.q, self.target)]
            fastest = max(abs(x) for x in v)
            if fastest > self.rate > 0:
                v = [x * self.rate / fastest for x in v]
            elif self.rate <= 0:
                v = [0.0] * 6
            self.q = [q + x * dt for q, x in zip(self.q, v)]
            g_err = self.gripper_target - self.gripper
            g_step = self.gripper_rate * dt
            self.gripper += max(-g_step, min(g_step, g_err))

    def _moving(self) -> bool:
        return (max(abs(t - q) for q, t in zip(self.q, self.target)) > MOVING_EPS_DEG
                or abs(self.gripper_target - self.gripper) > MOVING_EPS_GRIPPER)

    # --- protocol

    def dispatch(self, req: Dict[str, Any]) -> Dict[str, Any]:
        op = req["op"]
        with self._lock:
            self.requests.append(op)
        if op == "state":
            with self._lock:
                return {"ok": True, "wall": time.time(), "sim_t": time.time() - self._t0,
                        "angles_deg": [round(q, 4) for q in self.q], "gripper": round(self.gripper, 2),
                        "moving": self._moving()}
        if op == "move_joints":
            angles = [float(a) for a in req["angles_deg"]]
            if len(angles) != 6:
                return {"ok": False, "error": "angles_deg needs 6 values"}
            speed = _clamp_speed(req["speed"])
            with self._lock:
                self.target, self.rate = angles, float(speed)
                self.moves.append({"angles_deg": angles, "speed": speed})
            return {"ok": True}
        if op == "gripper":
            value, speed = float(req["value"]), _clamp_speed(req["speed"])
            if not 0 <= value <= 100:
                return {"ok": False, "error": "value must be 0..100"}
            with self._lock:
                self.gripper_target, self.gripper_rate = value, GRIPPER_RATE_PER_SPEED * speed
                self.grips.append({"value": value, "speed": speed})
            return {"ok": True}
        if op == "stop":
            with self._lock:
                self.target, self.gripper_target = list(self.q), self.gripper
                self.stops += 1
            return {"ok": True}
        if op == "frame":
            return self._frame()
        if op == "reset":
            with self._lock:
                self.q, self.target, self.rate = list(self.start_deg), list(self.start_deg), 0.0
                self.gripper = self.gripper_target = 100.0
            return {"ok": True}
        if op == "truth":
            return {"ok": True, "red_block": {"pos_m": [0.2, 0.0, 0.02], "quat_wxyz": [1, 0, 0, 0]},
                    "brown_box": {"pos_m": [0.2, 0.15, 0.0]}, "in_box": False, "block_speed_mps": 0.0}
        if op == "record":
            return {"ok": True, "dir": "/dev/null"}
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _frame(self) -> Dict[str, Any]:
        if self._frames is None:
            self._frames = testsrc_frames()
        now = time.time()
        tick = int((now - self._t0) * RENDER_HZ)
        jpeg = self._frames[tick % len(self._frames)]
        return {"ok": True, "wall": self._t0 + tick / RENDER_HZ, "sim_t": tick / RENDER_HZ, "width": 640,
                "height": 480, "jpeg_b64": base64.b64encode(jpeg).decode()}


def _clamp_speed(speed: Any) -> int:
    return max(1, min(100, int(speed)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="stub_sim", description=__doc__.split("\n")[0])
    p.add_argument("--host", default=DEFAULT_BRIDGE[0])
    p.add_argument("--port", type=int, default=DEFAULT_BRIDGE[1])
    p.add_argument("--start", type=float, nargs=6, default=[0.0] * 6, metavar="DEG", help="initial joint angles")
    args = p.parse_args(argv)
    sim = StubSim(args.host, args.port, args.start).start()
    log(f"listening on {sim.address[0]}:{sim.port}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
