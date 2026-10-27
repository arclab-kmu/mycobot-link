"""``mycobot-brain``: listen for the arm server and drive it by hand or through vLLM.

Every subcommand binds, waits for the Jetson to dial in, then acts. Whether a
command moves anything is the arm server's choice (``--live`` there), not this
side's: against a dry-run server these are all safe to try.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import List, Optional

from mycobot_brain import planner
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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="mycobot-brain", description=__doc__.split("\n")[0])
    p.add_argument("--host", default="0.0.0.0", help="interface to bind (default: all)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
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

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    with ArmLink(args.host, args.port) as link:
        print(f"waiting for the arm server on {args.host}:{link.port} ...", file=sys.stderr)
        try:
            _print(link.wait_connected(args.wait))
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
        except (LinkError, planner.PlannerError) as exc:
            print(f"mycobot-brain: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
