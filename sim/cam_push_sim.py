#!/usr/bin/env python3
"""Sim side of the camera link: pull the wrist-camera frame from sim-bridge, push it to the brain.

The brain cannot tell this from the Jetson's ``camera-push/camera_push.py``: same connection
direction (this side dials the brain's camera port and redials every second), same ``hello`` line,
same ``frame`` header line + JPEG bytes (``schema/PROTOCOL.md``, "카메라 연결"):

    python3 cam_push_sim.py --brain 210.123.42.157:47901 [--bridge 127.0.0.1:47950] [--rate 5]

Two differences from the real camera, both because the brain measures a frame's age from the moment
its last byte *arrives*:

* a frame is pushed once, when the bridge renders a new one (a changed ``wall``/``sim_t``), never
  again - a repeated picture would look as fresh as a new one;
* a frame the bridge rendered more than ``--max-age`` seconds ago is not pushed, for the same
  reason. The sim and this process share a host, so ``wall`` is comparable to ``time.time()``.

Stdlib only, Python 3.8.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import signal
import socket
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from bridge_client import DEFAULT_BRIDGE, BridgeClient, BridgeError, parse_addr

PROTOCOL_VERSION = 1
MAX_JPEG = 4 * 1024 * 1024  # the brain refuses bigger frames
SEND_TIMEOUT_S = 3.0  # a brain that stops reading costs at most this
SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def log(msg: str) -> None:
    print(f"cam-push-sim: {msg}", file=sys.stderr, flush=True)


def header(obj: Dict[str, Any]) -> bytes:
    """One JSON line, serialised exactly as ``camera_push.header`` does."""
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def hello(size: str, rate: float) -> Dict[str, Any]:
    return {"type": "hello", "protocol": PROTOCOL_VERSION, "server": "camera-push",
            "source": "sim-bridge", "size": size, "rate_hz": rate}


def frame_message(seq: int, jpeg: bytes, now_s: float) -> bytes:
    return header({"type": "frame", "seq": seq, "jpeg_bytes": len(jpeg),
                   "sent_unix_ms": int(now_s * 1000)}) + jpeg


class FrameSource:
    """Asks the bridge for its latest frame; hands out each picture once, and only while fresh."""

    def __init__(self, client: BridgeClient, max_age: float) -> None:
        self.client, self.max_age = client, max_age
        self._last: Optional[Tuple[float, float]] = None
        self.size = "640x480"

    def poll(self, now_s: float) -> Optional[bytes]:
        """A new, fresh JPEG, or ``None`` (nothing new, too old, or an unusable reply)."""
        reply = self.client.call({"op": "frame"})
        key = (reply.get("wall", 0.0), reply.get("sim_t", 0.0))
        if key == self._last:
            return None
        self._last = key
        try:
            jpeg = base64.b64decode(reply["jpeg_b64"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error):
            raise BridgeError("frame reply has no valid jpeg_b64")
        if not (jpeg.startswith(SOI) and jpeg.endswith(EOI)) or len(jpeg) > MAX_JPEG:
            raise BridgeError(f"frame reply is not a JPEG of at most {MAX_JPEG} bytes ({len(jpeg)} bytes)")
        self.size = f"{reply.get('width', 640)}x{reply.get('height', 480)}"
        if now_s - float(reply.get("wall", now_s)) > self.max_age:
            return None
        return jpeg


def run_connection(sock: socket.socket, source: FrameSource, rate: float, stop: list,
                   clock=time.time, sleep=time.sleep) -> None:
    """Send hello, then every new frame at ``rate`` Hz, until the socket fails or ``stop``."""
    sock.sendall(header(hello(source.size, rate)))
    seq, period, stalled = 0, 1.0 / rate, False
    next_at = last_sent = time.monotonic()
    while not stop:
        error = None
        try:
            jpeg = source.poll(clock())
        except BridgeError as exc:
            jpeg, error = None, str(exc)
        now = time.monotonic()
        if jpeg is not None:
            if stalled:
                log("frames back")
                stalled = False
            last_sent = now
            seq += 1
            sock.sendall(frame_message(seq, jpeg, clock()))
        elif not stalled and now - last_sent > 2.0:
            log("no new frame from the sim for 2 s" + (f" ({error})" if error else ""))
            stalled = True
        next_at += period
        if next_at > now:
            sleep(next_at - now)
        else:
            next_at = now  # skip missed periods rather than bursting


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="cam_push_sim", description=__doc__.split("\n")[0])
    p.add_argument("--brain", type=parse_addr, required=True, help="the Spark brain's camera port, HOST:PORT")
    p.add_argument("--bridge", type=parse_addr, default=DEFAULT_BRIDGE, help="sim-bridge HOST:PORT (default 127.0.0.1:47950)")
    p.add_argument("--rate", type=float, default=5.0, help="frames per second to send (default 5)")
    p.add_argument("--max-age", type=float, default=1.0, help="drop a frame rendered longer ago than this, s (default 1)")
    p.add_argument("--once", action="store_true", help="exit when the first connection ends (tests)")
    args = p.parse_args(argv)
    if not 0 < args.rate <= 30:
        p.error("--rate must be in (0, 30]")

    stop: list = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    source = FrameSource(BridgeClient(args.bridge), args.max_age)
    try:
        while not stop:
            try:
                sock = socket.create_connection(args.brain, timeout=3.0)
            except OSError as exc:
                log(f"{args.brain[0]}:{args.brain[1]}: {exc}; retrying in 1 s")
                time.sleep(1.0)
                continue
            sock.settimeout(SEND_TIMEOUT_S)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            log(f"connected to {args.brain[0]}:{args.brain[1]}, {args.rate:g} Hz from sim-bridge "
                f"{args.bridge[0]}:{args.bridge[1]}")
            try:
                run_connection(sock, source, args.rate, stop)
            except OSError as exc:
                log(f"link dropped: {exc}")
            finally:
                sock.close()
            if args.once:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
