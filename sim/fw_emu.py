#!/usr/bin/env python3
"""Atom firmware stand-in: a pty that speaks the myCobot 280 serial protocol, backed by sim-bridge.

The Rust arm-server opens the pty as if it were ``/dev/ttyTHS1`` (``--live --serial <pty>``), so its
real checks (limits, 20 deg from *measured* angles, freshness, watchdog) run in every sim test:

    python3 fw_emu.py                       # prints the slave path, links /tmp/mycobot-sim-tty
    mycobot-arm --live --assume-yes --serial /tmp/mycobot-sim-tty --brain <spark>:47900

What it answers (frames as in ``arm-server/src/frame.rs``, held to pymycobot 4.0.7's bytes by
``arm-server/tests/golden/``): ``get_angles`` from a cached sim ``state`` (the arm-server waits
50 ms), ``send_angles`` -> bridge ``move_joints``, ``set_gripper_value`` -> ``gripper``, ``stop`` ->
``stop``, plus the three queries pymycobot makes. Nothing else answers or moves anything. Like the
Atom, it never acknowledges a command (TODO(unverified): the real firmware's acknowledgements).
Speed -> rate map and gripper direction: ``schema/SIM.md``.

Silence means stop: with no sim state newer than ``--max-state-age-ms`` the emulator stays quiet, so
the arm-server's freshness check closes the gate instead of acting on a frozen reading.
"""

from __future__ import annotations

import argparse
import collections
import errno
import os
import select
import signal
import struct
import sys
import threading
import time
import tty
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from bridge_client import DEFAULT_BRIDGE, BridgeClient, BridgeError, parse_addr

HEADER = 0xFE
FOOTER = 0xFA
#: pymycobot's reader drops a length byte outside 2..30 as noise (same bound as frame.rs).
MAX_LEN = 30

#: Command bytes, pymycobot 4.0.7 ``ProtocolCode`` (the ones ``frame.rs`` knows).
GET_ERROR_INFO = 0x07
IS_POWER_ON = 0x12
SET_FRESH_MODE = 0x16
GET_ANGLES = 0x20
SEND_ANGLES = 0x22
STOP = 0x29
GET_GRIPPER_VALUE = 0x65
SET_GRIPPER_VALUE = 0x67

DEFAULT_LINK = "/tmp/mycobot-sim-tty"


def log(msg: str) -> None:
    print(f"fw-emu: {msg}", file=sys.stderr, flush=True)


# --- codec ------------------------------------------------------------------


def frame(cmd: int, data: Sequence[int] = ()) -> bytes:
    """``FE FE LEN CMD DATA.. FA`` with ``LEN = len(DATA) + 2`` (the 280 has no CRC)."""
    return bytes([HEADER, HEADER, len(data) + 2, cmd, *data, FOOTER])


def angles_reply(angles_deg: Sequence[float]) -> bytes:
    """A ``get_angles`` reply: six big-endian i16 hundredths of a degree.

    Rounded, not truncated: this is a measurement, and ``golden/`` holds it to the bytes pymycobot's
    reader turns back into the same hundredths. A firmware reading cannot leave i16.
    """
    data = b"".join(struct.pack(">h", max(-32768, min(32767, round(a * 100)))) for a in angles_deg)
    return frame(GET_ANGLES, data)


def decode_send_angles(data: bytes) -> Optional[Tuple[List[float], int]]:
    """``(angles_deg, speed)`` from a ``send_angles`` payload, or ``None`` if it is not one."""
    if len(data) != 13:
        return None
    ints = struct.unpack(">6h", data[:12])
    return [v / 100.0 for v in ints], data[12]


class Frame:
    __slots__ = ("cmd", "data")

    def __init__(self, cmd: int, data: bytes) -> None:
        self.cmd, self.data = cmd, data

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Frame) and (self.cmd, self.data) == (other.cmd, other.data)

    def __repr__(self) -> str:
        return f"Frame(0x{self.cmd:02x}, {self.data.hex()})"


class FrameReader:
    """Incremental parser: push what the port returned, pull frames.

    Resynchronises like ``frame.rs`` and pymycobot: a bad length or a missing footer discards one
    byte and scans for the next ``FE FE``.
    """

    def __init__(self) -> None:
        self.buf = bytearray()

    def push(self, data: bytes) -> None:
        self.buf += data

    def next_frame(self) -> Optional[Frame]:
        while True:
            i = self.buf.find(b"\xfe\xfe")
            if i < 0:
                # Keep a trailing FE: it may be the first half of a header.
                keep = 1 if self.buf.endswith(b"\xfe") else 0
                del self.buf[: len(self.buf) - keep]
                return None
            del self.buf[:i]
            if len(self.buf) < 3:
                return None
            n = self.buf[2]
            if not 2 <= n <= MAX_LEN:
                del self.buf[:1]
                continue
            total = n + 3
            if len(self.buf) < total:
                return None
            if self.buf[total - 1] != FOOTER:
                del self.buf[:1]
                continue
            out = Frame(self.buf[3], bytes(self.buf[4 : total - 1]))
            del self.buf[:total]
            return out


# --- sim side ---------------------------------------------------------------


class StateCache:
    """The newest sim ``state`` and when it arrived; readers never touch the network."""

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._state: Optional[Dict[str, Any]] = None
        self._at = 0.0

    def update(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self._state, self._at = state, self._clock()

    def get(self, max_age: float) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._state is None or self._clock() - self._at > max_age:
                return None
            return self._state


class StatePoller(threading.Thread):
    """Polls ``state`` at ``hz`` on its own connection, so a slow command never delays a reading."""

    def __init__(self, client: BridgeClient, cache: StateCache, hz: float) -> None:
        super().__init__(name="state-poller", daemon=True)
        self.client, self.cache, self.period = client, cache, 1.0 / hz
        self.stopping = threading.Event()
        self.failures = 0

    def run(self) -> None:
        down_since: Optional[float] = None
        while not self.stopping.is_set():
            t0 = time.monotonic()
            try:
                self.cache.update(self.client.call({"op": "state"}))
                if down_since is not None:
                    log("sim-bridge back")
                    down_since = None
            except BridgeError as exc:
                self.failures += 1
                if down_since is None:
                    down_since = t0
                    log(f"sim-bridge: {exc}; staying silent until it answers")
                self.stopping.wait(0.2)
                continue
            self.stopping.wait(max(0.0, self.period - (time.monotonic() - t0)))


class CommandQueue(threading.Thread):
    """Forwards firmware commands to the bridge in order, on a connection of its own.

    ``stop`` goes to the front and drops whatever still waits: a move queued behind it would
    start the arm again after the stop.
    """

    def __init__(self, client: BridgeClient) -> None:
        super().__init__(name="commands", daemon=True)
        self.client = client
        self._cv = threading.Condition()
        self._queue: Deque[Dict[str, Any]] = collections.deque()
        self._stopping = False

    def move_joints(self, angles_deg: Sequence[float], speed: int) -> None:
        self._put({"op": "move_joints", "angles_deg": list(angles_deg), "speed": _speed(speed)})

    def gripper(self, value: int, speed: int) -> None:
        self._put({"op": "gripper", "value": int(value), "speed": _speed(speed)})

    def stop(self) -> None:
        self._put({"op": "stop"}, front=True)

    def close(self) -> None:
        with self._cv:
            self._stopping = True
            self._cv.notify_all()

    def _put(self, request: Dict[str, Any], front: bool = False) -> None:
        with self._cv:
            if front:
                self._queue.clear()
                self._queue.appendleft(request)
            else:
                self._queue.append(request)
            self._cv.notify()

    def run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._stopping)
                if self._stopping and not self._queue:
                    return
                request = self._queue.popleft()
            try:
                self.client.call(request)
            except BridgeError as exc:
                log(f"dropped {request['op']}: {exc}")


def _speed(speed: int) -> int:
    return max(1, min(100, int(speed)))


# --- the firmware -----------------------------------------------------------


class FwEmu:
    """Frame in, reply bytes (or ``None``) out. No I/O of its own, so tests drive it directly."""

    def __init__(self, cache: StateCache, commands: Any, max_state_age: float = 0.2) -> None:
        self.cache, self.commands, self.max_state_age = cache, commands, max_state_age
        self.counts: Dict[int, int] = collections.Counter()

    def handle(self, f: Frame) -> Optional[bytes]:
        self.counts[f.cmd] += 1
        if f.cmd == GET_ANGLES:
            state = self.cache.get(self.max_state_age)
            angles = state.get("angles_deg") if state else None
            if not isinstance(angles, list) or len(angles) != 6:
                return None
            return angles_reply(angles)
        if f.cmd == SEND_ANGLES:
            decoded = decode_send_angles(f.data)
            if decoded is None:
                log(f"ignored send_angles with {len(f.data)} data bytes")
                return None
            angles, speed = decoded
            log(f"send_angles {angles} speed {speed}")
            self.commands.move_joints(angles, speed)
        elif f.cmd == SET_GRIPPER_VALUE:
            if len(f.data) < 2:
                log(f"ignored set_gripper_value with {len(f.data)} data bytes")
                return None
            log(f"set_gripper_value {f.data[0]} speed {f.data[1]}")
            self.commands.gripper(f.data[0], f.data[1])
        elif f.cmd == STOP:
            log("stop")
            self.commands.stop()
        elif f.cmd == GET_GRIPPER_VALUE:
            state = self.cache.get(self.max_state_age)
            if not state or not isinstance(state.get("gripper"), (int, float)):
                return None
            # pymycobot reads a two-byte payload as one big-endian i16.
            return frame(GET_GRIPPER_VALUE, struct.pack(">h", max(0, min(100, round(state["gripper"])))))
        elif f.cmd == IS_POWER_ON:
            return frame(IS_POWER_ON, [1])
        elif f.cmd == GET_ERROR_INFO:
            return frame(GET_ERROR_INFO, [0])
        elif f.cmd == SET_FRESH_MODE:
            pass  # the sim always runs the newest command
        else:
            log(f"unknown command 0x{f.cmd:02x}, {len(f.data)} data bytes: ignored")
        return None


# --- the pty ----------------------------------------------------------------


class PtyPort:
    """A raw pty. We keep a slave fd open ourselves so the master survives the arm-server
    closing and reopening the port (it reconnects to the brain, not to the UART, but a crash
    and restart must not turn every read into EIO)."""

    def __init__(self, link: Optional[str]) -> None:
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        os.set_blocking(self.master, False)
        self.path = os.ttyname(self.slave)
        self.link = link
        if link:
            tmp = f"{link}.{os.getpid()}"
            os.symlink(self.path, tmp)
            os.replace(tmp, link)  # atomic, replaces the previous run's link

    def close(self) -> None:
        if self.link:
            try:
                if os.readlink(self.link) == self.path:
                    os.unlink(self.link)
            except OSError:
                pass
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass

    def serve(self, emu: FwEmu, stopping: threading.Event) -> None:
        reader = FrameReader()
        while not stopping.is_set():
            ready, _, _ = select.select([self.master], [], [], 0.1)
            if not ready:
                continue
            try:
                data = os.read(self.master, 4096)
            except BlockingIOError:
                continue
            except OSError as exc:
                if exc.errno == errno.EIO:  # every slave end closed; ours is open, so only at exit
                    return
                raise
            reader.push(data)
            while (f := reader.next_frame()) is not None:
                reply = emu.handle(f)
                if reply is not None:
                    try:
                        os.write(self.master, reply)
                    except BlockingIOError:
                        pass  # nobody is reading the port; a get_angles reply is not worth waiting for


class Service:
    """Everything wired together: the pty, the sim connections, the firmware.

    ``serve()`` blocks (the command line); ``start()`` runs it on a thread (tests).
    """

    def __init__(self, bridge: Tuple[str, int] = DEFAULT_BRIDGE, link: Optional[str] = DEFAULT_LINK,
                 state_hz: float = 50.0, max_state_age: float = 0.2) -> None:
        self.port = PtyPort(link)
        self.cache = StateCache()
        self.poller = StatePoller(BridgeClient(bridge), self.cache, state_hz)
        self.commands = CommandQueue(BridgeClient(bridge))
        self.emu = FwEmu(self.cache, self.commands, max_state_age)
        self.stopping = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def path(self) -> str:
        return self.port.path

    def start(self) -> "Service":
        self._thread = threading.Thread(target=self.serve, name="fw-emu", daemon=True)
        self._thread.start()
        return self

    def serve(self) -> None:
        self.poller.start()
        self.commands.start()
        try:
            self.port.serve(self.emu, self.stopping)
        finally:
            # The arm-server's port is about to vanish; leave the sim holding still, as a stop would.
            self.commands.stop()
            self.commands.close()
            self.commands.join(1.0)  # drains the queue first, so the stop gets out
            self.poller.stopping.set()
            self.poller.join(1.0)
            self.poller.client.close()
            self.commands.client.close()
            self.port.close()

    def close(self) -> None:
        self.stopping.set()
        if self._thread is not None:
            self._thread.join(3.0)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="fw_emu", description=__doc__.split("\n")[0])
    p.add_argument("--bridge", type=parse_addr, default=DEFAULT_BRIDGE, help="sim-bridge HOST:PORT (default 127.0.0.1:47950)")
    p.add_argument("--link", default=DEFAULT_LINK, help=f"symlink to the pty slave (default {DEFAULT_LINK}); '' for none")
    p.add_argument("--state-hz", type=float, default=50.0, help="how often to poll the sim's state (default 50)")
    p.add_argument("--max-state-age-ms", type=float, default=200.0, help="older state is not served (default 200)")
    args = p.parse_args(argv)
    if not 0 < args.state_hz <= 500:
        p.error("--state-hz must be in (0, 500]")

    service = Service(args.bridge, args.link or None, args.state_hz, args.max_state_age_ms / 1000.0)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: service.stopping.set())
    print(service.path, flush=True)
    log(f"pty slave {service.path}" + (f", linked at {args.link}" if args.link else "")
        + f"; sim-bridge {args.bridge[0]}:{args.bridge[1]}")
    service.serve()
    log("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
