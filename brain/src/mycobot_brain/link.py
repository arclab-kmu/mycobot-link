"""Brain side of the link: listen for the arm server, send requests, keep the latest state.

The arm server connects to us, not the other way round: the myCobot sits
behind the lab router's NAT and the Spark does not (``schema/PROTOCOL.md``).
This side only proposes. Every limit is checked again, and with the final
say, by the arm server on the Jetson.
"""

from __future__ import annotations

import itertools
import json
import logging
import socket
import threading
import time
from typing import Any, Dict, Optional, Sequence

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
DEFAULT_PORT = 47900
MAX_LINE = 64 * 1024


class LinkError(RuntimeError):
    """No arm connected, the link dropped, or a reply did not come in time."""


class Rejected(LinkError):
    """The arm server refused the request. ``str()`` is its reason."""


class ArmLink:
    """One listening socket, at most one arm connection.

    A new connection replaces the old one: after a Jetson reboot the stale
    socket may not have noticed yet, and the arm that just dialled in is the
    real one. Requests waiting on the old connection fail with
    :class:`LinkError` rather than hanging.

    Args:
        host: interface to bind. ``0.0.0.0`` so the Jetson can reach it.
        port: ``0`` picks a free port (tests); read it back from :attr:`port`.
        ping_interval: seconds between heartbeats. Must stay well under the
            arm's ``watchdog_ms`` (500 by default): the heartbeat is what keeps
            the gate open while this process waits on vLLM.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_PORT, ping_interval: float = 0.1) -> None:
        self._addr = (host, port)
        self.ping_interval = ping_interval
        #: Tests clear this to let the arm's watchdog fire on a live connection.
        self.pinging = threading.Event()
        self.pinging.set()
        self._cv = threading.Condition()
        self._send_lock = threading.Lock()
        self._listener: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._generation = 0  # bumped whenever the connection changes
        self._hello: Optional[Dict[str, Any]] = None
        self._state: Optional[Dict[str, Any]] = None
        self._replies: Dict[int, Optional[Dict[str, Any]]] = {}
        self._ids = itertools.count(1)
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "ArmLink":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self._addr)
        sock.listen(1)
        self._listener = sock
        threading.Thread(target=self._accept_loop, name="arm-accept", daemon=True).start()
        threading.Thread(target=self._ping_loop, name="arm-ping", daemon=True).start()
        log.info("listening on %s:%d", self._addr[0], self.port)
        return self

    @property
    def port(self) -> int:
        if self._listener is None:
            raise LinkError("not started")
        return self._listener.getsockname()[1]

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._drop_locked("closed")
            self._cv.notify_all()
        if self._listener is not None:
            self._listener.close()

    def __enter__(self) -> "ArmLink":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- what the arm told us ----------------------------------------------

    def wait_connected(self, timeout: float = 30.0) -> Dict[str, Any]:
        """Block until an arm has connected and said hello; return the hello."""
        with self._cv:
            if not self._cv.wait_for(lambda: self._hello is not None or self._closed, timeout):
                raise LinkError(f"no arm server connected within {timeout:.0f} s")
            if self._hello is None:
                raise LinkError("link closed")
            return dict(self._hello)

    @property
    def connected(self) -> bool:
        with self._cv:
            return self._hello is not None

    def latest_state(self) -> Optional[Dict[str, Any]]:
        with self._cv:
            return dict(self._state) if self._state is not None else None

    def wait_state(self, after_seq: Optional[int] = None, timeout: float = 2.0) -> Dict[str, Any]:
        """The next state with ``seq`` above ``after_seq`` (any state if None)."""

        def ready() -> bool:
            return self._state is not None and (after_seq is None or self._state["seq"] > after_seq)

        with self._cv:
            if not self._cv.wait_for(ready, timeout):
                raise LinkError(f"no state from the arm within {timeout:.1f} s")
            return dict(self._state)  # type: ignore[arg-type]

    # -- requests ----------------------------------------------------------

    def request(self, msg: Dict[str, Any], timeout: float = 2.0) -> Dict[str, Any]:
        """Send ``msg`` with a fresh ``id``; return the reply, or raise.

        ``ok: true`` means the arm server sent it to the arm. Whether the arm
        got there is for :meth:`wait_state` to show.
        """
        with self._cv:
            if self._conn is None or self._hello is None:
                raise LinkError("no arm server connected")
            rid = next(self._ids)
            generation = self._generation
            self._replies[rid] = None
        try:
            self._send(dict(msg, id=rid))
            with self._cv:
                done = self._cv.wait_for(
                    lambda: self._replies.get(rid) is not None or self._generation != generation, timeout
                )
                reply = self._replies.get(rid)
            if reply is None:
                raise LinkError("link dropped before the reply" if done else f"no reply within {timeout:.1f} s")
        finally:
            with self._cv:
                self._replies.pop(rid, None)
        if not reply.get("ok"):
            raise Rejected(reply.get("error", "rejected without a reason"))
        return reply

    def enable(self) -> None:
        self.request({"type": "enable"})

    def disable(self) -> None:
        self.request({"type": "disable"})

    def stop(self) -> None:
        self.request({"type": "stop"})

    def move_joints(self, angles_deg: Sequence[float], speed: int = 20) -> None:
        self.request({"type": "move_joints", "angles_deg": [float(a) for a in angles_deg], "speed": int(speed)})

    def gripper(self, value: int, speed: int = 50) -> None:
        self.request({"type": "gripper", "value": int(value), "speed": int(speed)})

    # -- internals ---------------------------------------------------------

    def _send(self, msg: Dict[str, Any]) -> None:
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._cv:
            conn = self._conn
        if conn is None:
            raise LinkError("no arm server connected")
        with self._send_lock:
            try:
                conn.sendall(line)
            except OSError as exc:
                raise LinkError(f"send failed: {exc}") from exc

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._closed:
            try:
                conn, peer = self._listener.accept()
            except OSError:
                return  # listener closed
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._cv:
                if self._closed:
                    conn.close()
                    return
                self._drop_locked("replaced by a new connection")
                self._conn = conn
                self._generation += 1
                generation = self._generation
                self._cv.notify_all()
            log.info("arm server connected from %s:%d", *peer)
            threading.Thread(target=self._read_loop, args=(conn, generation), name="arm-read", daemon=True).start()

    def _read_loop(self, conn: socket.socket, generation: int) -> None:
        buf = b""
        why = "arm server closed the connection"
        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    if raw.strip():
                        self._on_line(raw, conn)
                if len(buf) > MAX_LINE:
                    why = f"line longer than {MAX_LINE} bytes"
                    break
        except (OSError, ValueError) as exc:
            why = str(exc)
        with self._cv:
            if self._generation == generation:
                self._drop_locked(why)
                self._cv.notify_all()

    def _on_line(self, raw: bytes, conn: socket.socket) -> None:
        msg = json.loads(raw)
        kind = msg.get("type")
        with self._cv:
            if conn is not self._conn:
                return
            if kind == "hello":
                if msg.get("protocol") != PROTOCOL_VERSION:
                    raise ValueError(f"arm speaks protocol {msg.get('protocol')!r}, this brain {PROTOCOL_VERSION}")
                self._hello = msg
                log.info("hello: %s live=%s", msg.get("server"), msg.get("live"))
            elif kind == "state":
                self._state = msg
            elif kind == "reply":
                if msg.get("id") in self._replies:
                    self._replies[msg["id"]] = msg
                elif not msg.get("ok"):
                    log.warning("arm rejected an unattributed line: %s", msg.get("error"))
            else:
                log.warning("ignoring unknown message type %r", kind)
            self._cv.notify_all()

    def _drop_locked(self, why: str) -> None:
        if self._conn is None:
            return
        log.info("arm link dropped: %s", why)
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._conn.close()
        self._conn = None
        self._hello = None
        self._state = None
        self._generation += 1

    def _ping_loop(self) -> None:
        while not self._closed:
            time.sleep(self.ping_interval)
            if not self.pinging.is_set():
                continue
            with self._cv:
                ready = self._conn is not None and self._hello is not None
            if ready:
                try:
                    self._send({"type": "ping"})
                except LinkError:
                    pass  # the read loop notices the drop
