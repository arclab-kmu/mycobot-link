"""``mycobot-brain``: listen for the arm server and drive it by hand or through vLLM.

Every subcommand binds, waits for the Jetson to dial in, then acts: the arm
server on ``--port``, ``camera_push.py`` on ``--camera-port`` (``look`` needs
only the camera, ``act`` both). Whether a command moves anything is the arm
server's choice (``--live`` there), not this side's: against a dry-run server
these are all safe to try.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from mycobot_brain import act, planner, vision
from mycobot_brain.camera import DEFAULT_CAMERA_PORT, CameraError, CameraLink
from mycobot_brain.link import DEFAULT_PORT, ArmLink, LinkError


def _print(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _watch(link: ArmLink, seconds: float) -> None:
    seq = None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        state = link.wait_state(after_seq=seq, timeout=2.0)
        seq = state["seq"]
        _print(state)


def _move(link: ArmLink, angles: List[float], speed: int, settle: float) -> None:
    """enable → move_joints → watch → disable. Disabled again even on a refusal."""
    link.wait_state()
    link.enable()
    try:
        link.move_joints(angles, speed)
        _print({"sent": {"angles_deg": angles, "speed": speed}})
        _watch(link, settle)
    finally:
        link.disable()


def _joints(text: str) -> List[int]:
    joints = [int(j) for j in text.split(",") if j.strip()]
    if not all(1 <= j <= 6 for j in joints):
        raise argparse.ArgumentTypeError(f"joints are 1..6, got {text!r}")
    return joints


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="mycobot-brain", description=__doc__.split("\n")[0])
    p.add_argument("--host", default="0.0.0.0", help="interface to bind (default: all)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--camera-port", type=int, default=DEFAULT_CAMERA_PORT)
    p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the arm server")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="print hello and states")
    s.add_argument("--seconds", type=float, default=3.0)

    sub.add_parser("stop", help="stop() and close the gate")

    m = sub.add_parser("move-joints", help="one joint target, degrees (at most 20 deg from the measured angles)")
    m.add_argument("angles", nargs=6, type=float, metavar="J")
    m.add_argument("--speed", type=int, default=20)
    m.add_argument("--settle", type=float, default=2.0, help="seconds of state to print after sending")

    a = sub.add_parser("ask", help="let vLLM propose one move; --execute to send it")
    a.add_argument("instruction")
    a.add_argument("--vllm-url", default=planner.DEFAULT_VLLM_URL)
    a.add_argument("--model", default=planner.DEFAULT_MODEL)
    a.add_argument("--execute", action="store_true", help="send the proposal (the arm server still checks it)")
    a.add_argument("--settle", type=float, default=2.0)

    lk = sub.add_parser("look", help="save one camera frame and let vLLM describe it (no arm needed)")
    lk.add_argument("question", nargs="?", help="ask this instead of the default description")
    lk.add_argument("--out", type=Path, help="where to save the JPEG (default runs/look-<time>.jpg)")
    lk.add_argument("--no-vlm", action="store_true", help="only save the frame")
    lk.add_argument("--vllm-url", default=planner.DEFAULT_VLLM_URL)
    lk.add_argument("--model", default=planner.DEFAULT_MODEL)

    c = sub.add_parser("act", help="camera -> vLLM -> small joint step, repeated; --execute to move")
    c.add_argument("instruction")
    c.add_argument("--steps", type=int, default=5, help="at most this many look-and-step rounds (default 5)")
    c.add_argument("--max-step", type=float, default=vision.DEFAULT_MAX_STEP_DEG,
                   help="largest per-joint change per round, deg (default 5, at most 20)")
    c.add_argument("--max-total", type=float, default=30.0,
                   help="stop before any joint goes this far from where the loop started, deg (default 30)")
    c.add_argument("--lock", type=_joints, default=[], metavar="J,J", help="joints that must not move, e.g. 2,3")
    c.add_argument("--speed", type=int, default=10)
    c.add_argument("--settle", type=float, default=1.5, help="seconds to wait after each step before looking")
    c.add_argument("--camera", choices=sorted(vision.CAMERAS), default="wrist")
    c.add_argument("--record", type=Path, help="frames + steps.jsonl here (default runs/act-<time>)")
    c.add_argument("--execute", action="store_true", help="send the steps (the arm server still checks them)")
    c.add_argument("--vllm-url", default=planner.DEFAULT_VLLM_URL)
    c.add_argument("--model", default=planner.DEFAULT_MODEL)

    k = sub.add_parser("console", help="web console: command box, both cameras (RGB + depth), cuRobo view; "
                       "VLM action chunks executed as one QP stream")
    k.add_argument("--workspace", type=Path, required=True, help="cameras, arms, calibration (JSON)")
    k.add_argument("--http-port", type=int, default=8095)
    k.add_argument("--vlm-url", default="http://127.0.0.1:8100/v1", help="OpenAI-compatible VLM (Spark via ssh -L)")
    k.add_argument("--vlm-model", default="gemma-4-26b-a4b-it")
    k.add_argument("--rate", type=float, default=20.0, help="QP / streaming rate, Hz")
    k.add_argument("--speed-param", type=int, default=100,
                   help="send_angles speed for every streamed tick; the arm server caps it with --max-speed")

    args = p.parse_args(argv)
    if args.cmd == "act" and not (0 < args.max_step <= 20 and args.steps >= 1 and 1 <= args.speed <= planner.MAX_SPEED):
        p.error(f"act needs 0 < --max-step <= 20, --steps >= 1 and --speed in 1..{planner.MAX_SPEED}")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    if args.cmd == "console":
        from mycobot_brain.console import server, workspace

        ws = workspace.load(args.workspace)
        con = server.Console(ws, args.vlm_url, args.vlm_model, rate_hz=args.rate, speed_param=args.speed_param)
        print(f"console on http://127.0.0.1:{args.http_port}  (arms: "
              + ", ".join(f"{a.name} <- 127.0.0.1:{a.port}" for a in ws.arms.values()) + ")", file=sys.stderr)
        try:
            server.serve(con, "127.0.0.1", args.http_port)
        except KeyboardInterrupt:
            pass
        return 0

    with contextlib.ExitStack() as stack:
        try:
            link = camera = None
            if args.cmd in ("look", "act"):
                camera = stack.enter_context(CameraLink(args.host, args.camera_port))
            if args.cmd != "look":
                link = stack.enter_context(ArmLink(args.host, args.port))
                print(f"waiting for the arm server on {args.host}:{link.port} ...", file=sys.stderr)
                _print(link.wait_connected(args.wait))
            if camera is not None:
                print(f"waiting for the camera on {args.host}:{camera.port} ...", file=sys.stderr)
                _print(camera.wait_connected(args.wait))

            if args.cmd == "status":
                _watch(link, args.seconds)
            elif args.cmd == "stop":
                link.stop()
                _print({"stopped": True})
            elif args.cmd == "move-joints":
                _move(link, args.angles, args.speed, args.settle)
            elif args.cmd == "ask":
                state = link.wait_state()
                if state["angles_deg"] is None:
                    raise LinkError("the arm server has no joint reading to plan from")
                t0 = time.monotonic()
                move = planner.propose_move(
                    args.instruction, state["angles_deg"], base_url=args.vllm_url, model=args.model
                )
                _print({"proposal": move, "vllm_s": round(time.monotonic() - t0, 3)})
                if args.execute:
                    _move(link, move["angles_deg"], move["speed"], args.settle)
            elif args.cmd == "look":
                frame = camera.wait_frame()
                out = args.out or Path("runs") / f"look-{_stamp()}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(frame.jpeg)
                report = {"frame_seq": frame.seq, "jpeg_bytes": len(frame.jpeg), "frame_age_s": round(frame.age(), 3),
                          "saved": str(out)}
                if not args.no_vlm:
                    t0 = time.monotonic()
                    report["description"] = vision.describe(
                        frame.jpeg, args.question, base_url=args.vllm_url, model=args.model
                    )
                    report["vllm_s"] = round(time.monotonic() - t0, 3)
                _print(report)
            elif args.cmd == "act":
                act.run(
                    link, camera, args.instruction,
                    propose=functools.partial(vision.propose_step, base_url=args.vllm_url, model=args.model),
                    emit=_print, steps=args.steps, max_step=args.max_step, max_total=args.max_total,
                    locked=args.lock, speed=args.speed, settle=args.settle, execute=args.execute,
                    record=args.record or Path("runs") / f"act-{_stamp()}", camera_kind=args.camera,
                )
        except (LinkError, CameraError, planner.PlannerError, act.ActStopped) as exc:
            print(f"mycobot-brain: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
