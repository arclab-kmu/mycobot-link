#!/usr/bin/env python3
"""Find the RealSense V4L2 nodes on this machine: per camera, the Z16 depth node and the YUYV colour node.

Since 2026-11-27 both cameras of the exhibit plug into the AGX Orin instead of the Jetsons, and the /dev/videoN numbers
depend on the plug order. The camera is told apart by its USB product name (sysfs), not by the node number:
the D435i is the top camera, the D405 the side camera.

  rs_nodes.py        every RealSense node with its formats, and the pick per camera
  rs_nodes.py env    shell lines for tools/console_up.sh, e.g.  : "${TOP_DEPTH:=/dev/video0}"
                     (a variable already set in the environment wins)
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys

ROLE = {"435": "TOP", "405": "SIDE"}  # product name contains "Depth Camera 435i" / "Depth Camera 405"


def _read(d: str, f: str) -> str:
    try:
        with open(os.path.join(d, f)) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _usb_device(sysdir: str) -> str:
    """From /sys/class/video4linux/videoN up to the USB device (the directory with idProduct)."""
    p = os.path.realpath(os.path.join(sysdir, "device"))
    while p != "/" and not os.path.exists(os.path.join(p, "idProduct")):
        p = os.path.dirname(p)
    return p


def cameras() -> dict:
    cams: dict = {}
    nodes = sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda s: int(re.sub(r"\D", "", os.path.basename(s))))
    for sd in nodes:
        u = _usb_device(sd)
        prod = _read(u, "product")
        if "RealSense" not in prod:
            continue
        dev = "/dev/" + os.path.basename(sd)
        out = subprocess.run(["v4l2-ctl", "-d", dev, "--list-formats"], capture_output=True, text=True).stdout
        fmts = [f.strip() for f in re.findall(r"(?:\[\d+\]: |Pixel Format: )'(.{4})'", out)]
        c = cams.setdefault(_read(u, "serial") or u, {"product": prod, "usb": os.path.basename(u), "depth": None,
                                                      "color": None, "nodes": []})
        c["nodes"].append((dev, fmts))
        if c["depth"] is None and "Z16" in fmts:
            c["depth"] = dev
        if c["color"] is None and "YUYV" in fmts:
            c["color"] = dev
    return cams


def role(product: str) -> str | None:
    return next((r for k, r in ROLE.items() if f"Camera {k}" in product), None)


def main() -> int:
    cams = cameras()
    if len(sys.argv) > 1 and sys.argv[1] == "env":
        for c in cams.values():
            r = role(c["product"])
            if r and c["depth"] and c["color"]:
                print(f': "${{{r}_DEPTH:={c["depth"]}}}"')
                print(f': "${{{r}_COLOR:={c["color"]}}}"')
        return 0
    if not cams:
        print("RealSense 카메라가 없어요 (USB 연결, v4l2-ctl 설치 확인)")
        return 1
    for serial, c in cams.items():
        print(f"{c['product']}  serial {serial}  usb {c['usb']}  -> {role(c['product']) or '?'}: "
              f"depth {c['depth']}, color {c['color']}")
        for dev, fmts in c["nodes"]:
            print(f"   {dev}: {' '.join(fmts) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
