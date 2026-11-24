"""Camera feeds for the console: one upstream reader per stream, any number of browsers.

The Jetsons already serve MJPEG (``rs_web_viewer.py`` on 127.0.0.1:8090, reached
through ``ssh -L``). Opening that stream once here and fanning the newest frame
out keeps the Jetson's encoder load the same whether one or five tabs are open,
and gives the VLM the same frame the person is looking at.

Frames are cut out of the byte stream at JPEG start/end markers instead of by
parsing multipart headers: the viewer's boundary string and header spacing are
then irrelevant, and a reconnect resynchronises on the next start marker.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from dataclasses import dataclass, field

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
#: A JPEG larger than this means the markers were lost; drop the buffer.
MAX_FRAME_BYTES = 4 * 1024 * 1024


def split_jpegs(buf: bytearray) -> list[bytes]:
    """Remove and return every complete JPEG in ``buf`` (in place).

    Bytes before the first start marker are discarded. An incomplete trailing
    frame stays in ``buf`` for the next read.
    """
    out: list[bytes] = []
    while True:
        start = buf.find(SOI)
        if start < 0:
            # Keep a trailing 0xff: it may be the first half of a start marker.
            keep = 1 if buf.endswith(b"\xff") else 0
            del buf[: len(buf) - keep]
            return out
        if start:
            del buf[:start]
        end = buf.find(EOI, 2)
        if end < 0:
            if len(buf) > MAX_FRAME_BYTES:
                buf.clear()
            return out
        out.append(bytes(buf[: end + 2]))
        del buf[: end + 2]


@dataclass
class Feed:
    """The newest JPEG of one upstream MJPEG URL, refreshed by a daemon thread."""

    name: str
    url: str
    jpeg: bytes | None = None
    stamp: float = 0.0  # time.time() of the newest frame
    frames: int = 0
    error: str = ""
    _cond: threading.Condition = field(default_factory=threading.Condition, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def start(self) -> "Feed":
        self._thread = threading.Thread(target=self._loop, name=f"feed-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def age(self) -> float | None:
        return None if self.jpeg is None else time.time() - self.stamp

    def wait_newer(self, stamp: float, timeout: float = 2.0) -> tuple[bytes | None, float]:
        """Block until a frame newer than ``stamp`` arrives (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self.stamp > stamp or self._stop.is_set(), timeout)
            return self.jpeg, self.stamp

    def _loop(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=5.0) as resp:
                    self.error = ""
                    backoff = 0.5
                    buf = bytearray()
                    while not self._stop.is_set():
                        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                        if not chunk:
                            raise OSError("stream ended")
                        buf += chunk
                        for jpeg in split_jpegs(buf):
                            with self._cond:
                                self.jpeg, self.stamp = jpeg, time.time()
                                self.frames += 1
                                self._cond.notify_all()
            except Exception as exc:  # noqa: BLE001 - any failure means "reconnect"
                self.error = f"{type(exc).__name__}: {exc}"[:200]
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)


def snapshot(url: str, timeout: float = 3.0) -> bytes:
    """One still frame (the viewers' ``/color.jpg``)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()
