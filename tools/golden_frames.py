"""Record what pymycobot writes for each call the arm server makes, and how it
reads a get_angles reply. ``arm-server/tests/golden.rs`` holds the Rust frames
to this file.

pymycobot's ``MyCobot280`` opens one end of a pty; this script plays the arm on
the other end. No arm, no Jetson. Run from the repo root:

    uv run --no-project --with pymycobot==4.0.7 python tools/golden_frames.py

Calls that expect a reply get none (except get_angles), so pymycobot waits
0.5 s and re-sends, three times: ``writes`` records that, and the run takes
about 15 s.
"""

from __future__ import annotations

import argparse
import json
import os
import pty
import select
import struct
import threading
import time
from importlib.metadata import version
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "arm-server/tests/golden/pymycobot-4.0.7.json"

#: Canned get_angles reply: J1..J6 as i16 hundredths of a degree.
ANGLES_REPLY_INTS = [1234, -4568, 9000, -1, 15000, -17999]

REQUESTS = [
    ("get_angles", []),
    ("send_angles", [[0, 0, 0, 0, 0, 0], 20]),
    # 0.29 and -45.678 are the ones where int(a * 100) truncates below the decimal value.
    ("send_angles", [[0.29, -45.678, 90, -0.005, 150, -179.99], 50]),
    ("send_angles", [[-165, 137, -147, 147, -152, 177], 1]),
    ("stop", []),
    ("set_fresh_mode", [1]),
    ("set_fresh_mode", [0]),
    ("set_gripper_value", [40, 50]),
    ("get_gripper_value", []),
    ("is_power_on", []),
    ("get_error_information", []),
]


def split_frames(buf: bytes) -> list[bytes]:
    out, i = [], 0
    while i + 3 <= len(buf):
        if buf[i] == 0xFE and buf[i + 1] == 0xFE:
            end = i + buf[i + 2] + 3
            if end <= len(buf) and buf[end - 1] == 0xFA:
                out.append(buf[i:end])
                i = end
                continue
        i += 1
    return out


def angles_reply() -> bytes:
    data = b"".join(struct.pack(">h", v) for v in ANGLES_REPLY_INTS)
    return bytes([0xFE, 0xFE, len(data) + 2, 0x20]) + data + bytes([0xFA])


def run_call(master: int, fn, args, reply: bytes | None):
    """Run ``fn(*args)`` in a thread while reading the arm's end of the pty."""
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("value", fn(*args)))
    t.start()
    buf, replied = b"", False
    deadline = time.monotonic() + 5.0
    while (t.is_alive() or select.select([master], [], [], 0.05)[0]) and time.monotonic() < deadline:
        if select.select([master], [], [], 0.01)[0]:
            buf += os.read(master, 4096)
        if reply is not None and not replied and split_frames(buf):
            os.write(master, reply)
            replied = True
    t.join()
    return result.get("value"), split_frames(buf)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    from pymycobot import MyCobot280

    master, slave = pty.openpty()
    arm = MyCobot280(os.ttyname(slave), 1000000)  # sleeps 1.5 s, as on the real 280
    requests, replies = [], []
    for name, call_args in REQUESTS:
        reply = angles_reply() if name == "get_angles" else None
        value, frames = run_call(master, getattr(arm, name), call_args, reply)
        if not frames or any(f != frames[0] for f in frames):
            raise SystemExit(f"{name}{call_args}: expected repeats of one frame, got {[f.hex() for f in frames]}")
        requests.append({"call": name, "args": call_args, "frame": frames[0].hex(), "writes": len(frames)})
        if reply is not None:
            replies.append({"call": name, "reply": reply.hex(), "result": value})
        print(f"{name}{call_args}: {frames[0].hex()} x{len(frames)}" + (f" -> {value}" if reply else ""))
    arm.close()
    os.close(slave)
    os.close(master)

    golden = {
        "generated_by": "tools/golden_frames.py",
        "pymycobot": version("pymycobot"),
        "pyserial": version("pyserial"),
        "class": "MyCobot280",
        "requests": requests,
        "replies": replies,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
