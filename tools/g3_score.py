"""Score G3 runs from their raw files alone.

Pass = the sim's truth after the run says ``in_box``, the block is at rest
(``block_speed_mps`` < 0.005) and the gripper reads open (>= 90). The brain's
own ``result`` (the VLM's "is it in the box?") is printed beside it, not used.

    python3 tools/g3_score.py [--runs /home/hkim/mycobot-sim/runs]
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def score(d: Path) -> dict:
    truth = json.loads((d / "truth.json").read_text())
    state = json.loads((d / "state_after.json").read_text())
    ev = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()]
    result = [e for e in ev if e["event"] == "result"][-1]
    latency = [e["latency_s"] for e in ev if e["event"] == "vlm_reply"]
    rim = truth["brown_box"]["rim_center_m"]
    # The box does not move during a run, so its truth after the run scores the brain's estimates.
    box_mm = {e["source"]: round(1000 * math.hypot(e["xyz"][0] - rim[0], e["xyz"][1] - rim[1]), 1)
              for e in ev if e["event"] == "estimate" and e["role"] == "place"}
    return {
        "run": d.name,
        "pass": truth["in_box"] and truth["block_speed_mps"] < 0.005 and state["gripper"] >= 90,
        "in_box": truth["in_box"], "block_speed_mps": truth["block_speed_mps"], "gripper": state["gripper"],
        "brain_ok": result.get("ok"), "why": result.get("why"), "seconds": result["t"],
        "attempts": result.get("attempts"), "vlm_calls": len(latency),
        "vlm_latency_median_s": statistics.median(latency) if latency else None,
        "box_estimate_err_mm": box_mm,
        "tilted": [e["tilt_deg"] for e in ev if e["event"] == "approach"],
        "approx_arrivals": sum(e["event"] == "arrived_approx" for e in ev),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=Path, default=Path("/home/hkim/mycobot-sim/runs"))
    p.add_argument("--glob", default="g3_s[0-9]*")
    args = p.parse_args()
    rows = []
    for d in sorted(x for x in args.runs.glob(args.glob) if x.is_dir()):
        try:
            rows.append(score(d))
        except (FileNotFoundError, json.JSONDecodeError, IndexError) as exc:
            print(f"{d.name}: incomplete ({exc})")
            continue
        r = rows[-1]
        print(f"{r['run']}: {'PASS' if r['pass'] else 'FAIL'} in_box={r['in_box']} v={r['block_speed_mps']:.4f} "
              f"gripper={r['gripper']:.0f} brain_ok={r['brain_ok']} {r['seconds']:.1f} s, "
              f"{r['vlm_calls']} VLM calls (median {r['vlm_latency_median_s']:.2f} s), attempts={r['attempts']}, "
              f"box err mm {r['box_estimate_err_mm']}")
    print(f"{sum(r['pass'] for r in rows)}/{len(rows)} pass")


if __name__ == "__main__":
    main()
