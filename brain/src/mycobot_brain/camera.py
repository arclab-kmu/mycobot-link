"""Brain side of the camera link: listen for ``camera_push.py``, keep the newest JPEG.

A second TCP connection, separate from the arm's: a 40 KB frame on the arm
link would sit in front of a ``stop`` reply. Each frame is a JSON header line
naming its size, then that many JPEG bytes (``schema/PROTOCOL.md``, "카메라 연결").

Frame age is measured on this side, from when the last byte arrived, so the
Jetson's clock does not matter. The snapshot it fetched was at most 0.5 s old
there (``mycobot-camera``'s freshness rule), so add that to the age here.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
DEFAULT_CAMERA_PORT = 47901
MAX_HEADER = 4096
MAX_JPEG = 4 * 1024 * 1024


class CameraError(RuntimeError):
    """No camera connected, or no new frame in time."""


@dataclass(frozen=True)
class Frame:
    seq: int
    jpeg: bytes
    received: float  # time.monotonic() when its last byte arrived

    def age(self) -> float:
        return time.monotonic() - self.received


class CameraLink:
    """One listening socket; the newest connection's frames win."""

    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_CAMERA_PORT) -> None:
        self._addr = (host, port)
        self._cv = threading.Condition()
        self._listener: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._hello: Optional[Dict[str, Any]] = None
        self._frame: Optional[Frame] = None
        self._closed = False

    def start(self) -> "CameraLink":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(self._addr)
        sock.listen(1)
        self._listener = sock
        threading.Thread(target=self._accept_loop, name="camera-accept", daemon=True).start()
        log.info("camera: listening on %s:%d", self._addr[0], self.port)
        return self

    @property
    def port(self) -> int:
        if self._listener is None:
            raise CameraError("not started")
        return self._listener.getsockname()[1]

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._drop_locked()
            self._cv.notify_all()
        if self._listener is not None:
            self._listener.close()

    def __enter__(self) -> "CameraLink":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def wait_connected(self, timeout: float = 30.0) -> Dict[str, Any]:
        with self._cv:
            if not self._cv.wait_for(lambda: self._hello is not None or self._closed, timeout):
                raise CameraError(f"no camera connected within {timeout:.0f} s")
            if self._hello is None:
                raise CameraError("camera link closed")
            return dict(self._hello)

    def latest(self) -> Optional[Frame]:
        with self._cv:
            return self._frame

    def wait_frame(self, after: Optional[float] = None, timeout: float = 3.0) -> Frame:
        """The newest frame that arrived after monotonic time ``after`` (any if None).

        Pass the moment the arm settled to get a picture of where it is now,
        not of where it was going.
        """

        def ready() -> bool:
            return self._frame is not None and (after is None or self._frame.received > after)

        with self._cv:
            if not self._cv.wait_for(ready, timeout):
                raise CameraError(f"no new camera frame within {timeout:.1f} s")
            return self._frame  # type: ignore[return-value]

    # -- internals ---------------------------------------------------------

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._closed:
            try:
                conn, peer = self._listener.accept()
            except OSError:
                return
            with self._cv:
                if self._closed:
                    conn.close()
                    return
                self._drop_locked()
                self._conn = conn
            log.info("camera connected from %s:%d", *peer)
            threading.Thread(target=self._read_loop, args=(conn,), name="camera-read", daemon=True).start()

    def _read_loop(self, conn: socket.socket) -> None:
        why = "camera closed the connection"
        reader = conn.makefile("rb")
        try:
            while True:
                line = reader.readline(MAX_HEADER + 1)
                if not line:
                    break
                if len(line) > MAX_HEADER or not line.endswith(b"\n"):
                    why = f"header line over {MAX_HEADER} bytes"
                    break
                msg = json.loads(line)
                kind = msg.get("type")
                if kind == "hello":
                    if msg.get("protocol") != PROTOCOL_VERSION:
                        why = f"camera speaks protocol {msg.get('protocol')!r}, this brain {PROTOCOL_VERSION}"
                        break
                    with self._cv:
                        if conn is not self._conn:
                            return
                        self._hello = msg
                        self._cv.notify_all()
                elif kind == "frame":
                    size = msg.get("jpeg_bytes")
                    if not isinstance(size, int) or not 0 < size <= MAX_JPEG:
                        why = f"bad jpeg_bytes {size!r}"
                        break
                    jpeg = reader.read(size)
                    if len(jpeg) < size:
                        break
                    if not jpeg.startswith(b"\xff\xd8"):
                        why = "frame is not a JPEG"
                        break
                    with self._cv:
                        if conn is not self._conn:
                            return
                        self._frame = Frame(int(msg.get("seq", 0)), jpeg, time.monotonic())
                        self._cv.notify_all()
                else:
                    log.warning("camera: ignoring %r", kind)
        except (OSError, ValueError) as exc:
            why = str(exc)
        with self._cv:
            if conn is self._conn:
                log.info("camera link dropped: %s", why)
                self._drop_locked()
                self._cv.notify_all()

    def _drop_locked(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._conn.close()
        self._conn = None
        self._hello = None
        self._frame = None  # from a connection that is gone; never hand it out as current
