"""Client for the simulation's sim-bridge: one JSON object per line, one reply line per request.

Shared by ``fw_emu.py`` and ``cam_push_sim.py``; stdlib only, Python 3.8.
The bridge is loopback-only on the sim host (``schema/SIM.md``), so every user of this
client runs on that host.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
from typing import Any, Dict, Optional, Tuple

DEFAULT_BRIDGE = ("127.0.0.1", 47950)
#: A ``frame`` reply carries a base64 JPEG; 16 MiB is far above 640x480.
MAX_LINE = 16 * 1024 * 1024


class BridgeError(RuntimeError):
    """The bridge is unreachable, answered garbage, or answered ``ok: false``."""


def parse_addr(text: str) -> Tuple[str, int]:
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {text!r}")
    return host, int(port)


class BridgeClient:
    """One TCP connection, opened on first use and again after any failure.

    ``call`` is serialised by a lock, so a thread that wants state while another
    waits on a slow ``move_joints`` should use its own client.
    """

    def __init__(self, addr: Tuple[str, int] = DEFAULT_BRIDGE, timeout: float = 2.0) -> None:
        self.addr = addr
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._close()

    def call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        line = json.dumps(request, separators=(",", ":")).encode() + b"\n"
        with self._lock:
            try:
                if self._sock is None:
                    self._connect()
                assert self._sock is not None
                self._sock.sendall(line)
                reply = json.loads(self._readline())
            except (OSError, ValueError) as exc:
                self._close()
                raise BridgeError(f"{self.addr[0]}:{self.addr[1]}: {exc}") from exc
        if not isinstance(reply, dict) or not reply.get("ok"):
            error = reply.get("error") if isinstance(reply, dict) else reply
            raise BridgeError(f"{request.get('op')}: {error or 'not ok'}")
        return reply

    def _connect(self) -> None:
        sock = socket.create_connection(self.addr, timeout=self.timeout)
        sock.settimeout(self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock, self._buf = sock, b""

    def _readline(self) -> bytes:
        assert self._sock is not None
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise OSError("bridge closed the connection")
            self._buf += chunk
            if len(self._buf) > MAX_LINE:
                raise OSError("bridge line too long")
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def _close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock, self._buf = None, b""
