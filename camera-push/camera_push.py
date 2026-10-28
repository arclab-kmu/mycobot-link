#!/usr/bin/env python3
"""Jetson side of the camera link: grab JPEG frames with ffmpeg, push them to the brain.

ffmpeg reads the USB camera (MJPG only, 30/25 fps), keeps every 1/rate-th
frame and re-encodes it as a full JPEG, so each frame carries its own Huffman
tables and decodes anywhere. The newest frame goes to the Spark, over a
connection this side opens for the same reason the arm server does: the
myCobot is behind the lab router's NAT (``schema/PROTOCOL.md``, "카메라 연결").

Standard library plus the ``ffmpeg`` binary, Python 3.8 (the Jetson's python3):

    python3 camera_push.py --brain 210.123.42.157:47901

It never opens the arm's serial port, so it runs beside the arm server.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import List, Optional, Sequence, Tuple

PROTOCOL_VERSION = 1
DEFAULT_DEVICE = "/dev/video0"  # the wrist camera on the lab's myCobot (2026-10-28)
MAX_JPEG = 4 * 1024 * 1024  # the brain refuses bigger frames
SEND_TIMEOUT_S = 3.0  # a brain that stops reading costs at most this
SOI, EOI = b"\xff\xd8", b"\xff\xd9"


def log(msg: str) -> None:
    print(f"camera-push: {msg}", file=sys.stderr, flush=True)


def parse_addr(text: str) -> Tuple[str, int]:
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {text!r}")
    return host, int(port)


def ffmpeg_command(device: str, size: str, rate: float, quality: int, test_pattern: bool) -> List[str]:
    if test_pattern:  # no camera: ffmpeg's own moving test card, paced in real time
        source = ["-re", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30"]
    else:
        source = ["-f", "v4l2", "-input_format", "mjpeg", "-video_size", size, "-framerate", "30", "-i", device]
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *source,
            "-vf", f"fps={rate:g}", "-c:v", "mjpeg", "-q:v", str(quality), "-f", "image2pipe", "pipe:1"]


class Grabber:
    """Runs ffmpeg and keeps only the newest whole JPEG; restarts ffmpeg if it dies."""

    def __init__(self, command: List[str]) -> None:
        self.command = command
        self._cv = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._count = 0
        self._stopped = False
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> "Grabber":
        threading.Thread(target=self._run, name="grabber", daemon=True).start()
        return self

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            proc = self._proc
            self._cv.notify_all()
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def wait_newer(self, count: int, timeout: float) -> Tuple[int, Optional[bytes]]:
        """The newest frame once there is one past ``count``; ``(count, None)`` on timeout."""
        with self._cv:
            self._cv.wait_for(lambda: self._count > count or self._stopped, timeout)
            if self._count > count:
                return self._count, self._jpeg
            return count, None

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stopped:
                    return
                self._proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc = self._proc
            log("ffmpeg started: " + " ".join(self.command))
            errors: List[bytes] = []  # drained on the side so a chatty ffmpeg never blocks on a full pipe
            drain = threading.Thread(target=lambda: errors.extend(proc.stderr), daemon=True)  # type: ignore[arg-type]
            drain.start()
            self._read(proc)
            proc.wait()
            drain.join(1.0)
            err = b"".join(errors[-5:]).decode(errors="replace").strip()
            with self._cv:
                if self._stopped:
                    return
            log(f"ffmpeg exited {proc.returncode}: {err[-300:] or '(no message)'}; restarting in 1 s")
            time.sleep(1.0)

    def _read(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        buf = b""
        while True:
            chunk = proc.stdout.read1(65536)  # type: ignore[attr-defined]
            if not chunk:
                return
            buf += chunk
            while True:
                start = buf.find(SOI)
                if start < 0:
                    buf = b""
                    break
                # ffmpeg's encoder writes no thumbnail and stuffs every 0xFF in
                # the scan, so the first EOI after SOI ends the frame.
                end = buf.find(EOI, start + 2)
                if end < 0:
                    buf = buf[start:]
                    if len(buf) > MAX_JPEG:
                        buf = b""  # not a JPEG stream we understand; resync
                    break
                jpeg, buf = buf[start:end + 2], buf[end + 2:]
                with self._cv:
                    self._jpeg = jpeg
                    self._count += 1
                    self._cv.notify_all()


def header(obj: dict) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode()


def push(sock: socket.socket, grabber: Grabber, hello: dict, stop: list) -> None:
    """Send hello, then every new frame, until the socket fails or ``stop``."""
    sock.sendall(header(hello))
    count, seq, stalled = 0, 0, False
    while not stop:
        count, jpeg = grabber.wait_newer(count, timeout=2.0)
        if jpeg is None:
            if not stalled:
                log("no new frame from ffmpeg for 2 s")
                stalled = True
            continue
        if stalled:
            log("frames back")
            stalled = False
        seq += 1
        sock.sendall(header({"type": "frame", "seq": seq, "jpeg_bytes": len(jpeg),
                             "sent_unix_ms": int(time.time() * 1000)}) + jpeg)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="camera_push", description=__doc__.split("\n")[0])
    p.add_argument("--brain", type=parse_addr, required=True, help="the Spark brain's camera port, HOST:PORT")
    p.add_argument("--device", default=DEFAULT_DEVICE, help=f"V4L2 device (default {DEFAULT_DEVICE})")
    p.add_argument("--size", default="640x480", help="capture size the camera offers in MJPG (default 640x480)")
    p.add_argument("--rate", type=float, default=5.0, help="frames per second to send (default 5)")
    p.add_argument("--quality", type=int, default=5, help="ffmpeg -q:v, 2 (best) .. 31 (default 5)")
    p.add_argument("--test-pattern", action="store_true", help="ffmpeg's test card instead of a camera")
    p.add_argument("--once", action="store_true", help="exit when the first connection ends (tests)")
    args = p.parse_args(argv)
    if not 0 < args.rate <= 30:
        p.error("--rate must be in (0, 30]")
    if not 2 <= args.quality <= 31:
        p.error("--quality must be in 2..31")

    source = "testsrc" if args.test_pattern else args.device
    hello = {"type": "hello", "protocol": PROTOCOL_VERSION, "server": "camera-push",
             "source": source, "size": args.size, "rate_hz": args.rate}
    grabber = Grabber(ffmpeg_command(args.device, args.size, args.rate, args.quality, args.test_pattern)).start()
    stop: list = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
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
            log(f"connected to {args.brain[0]}:{args.brain[1]}, {args.rate:g} Hz from {source}")
            try:
                push(sock, grabber, hello, stop)
            except OSError as exc:
                log(f"link dropped: {exc}")
            finally:
                sock.close()
            if args.once:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
