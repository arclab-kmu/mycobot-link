"""Score D3 drawer runs from their raw files alone (CONTRACT-drawers.md, D3).

Pass, from the sim's truth before and after the run:

- every drawer shut (``open_m`` < SHUT_M);
- the moved object at rest (``speed_mps`` < REST_MPS) where the command wants it:
  find, the yellow cone in the handover zone; take, one object that was in the
  named drawer, in the handover zone; put, the object that stood in the
  handover zone, in the named drawer;
- every other object in the drawer it started in.

The brain's own ``result`` is printed beside it, not used.

    python3 tools/d3_score.py [--runs /home/hkim/mycobot-sim/runs] [--glob 'd3_*']
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

SHUT_M = 0.003  # D1 shut its drawers to <= 0.7 mm
REST_MPS = 0.02  # the bridge's at_rest: resting cylinders and cones chatter near 0.01 m/s


def _moved(before: Dict[str, Any], after: Dict[str, Any], kind: str, drawer: str) -> Optional[str]:
    """The object the command is about, from the truth before the run."""
    objs = before["objects"]
    if kind == "find":
        return next((i for i, o in objs.items() if (o["color"], o["shape"]) == ("yellow", "cone")), None)
    if kind == "put":
        return next((i for i, o in objs.items() if o["in_handover"]), None)
    # take: whichever of the named drawer's objects left it
    left = [i for i, o in objs.items() if o["in_drawer"] == drawer and after["objects"][i]["in_drawer"] != drawer]
    return left[0] if len(left) == 1 else None


def score(d: Path) -> Dict[str, Any]:
    meta = json.loads((d / "meta.json").read_text())
    before = json.loads((d / "truth_before.json").read_text())
    after = json.loads((d / "truth.json").read_text())
    ev = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()]
    result = [e for e in ev if e["event"] == "result"][-1]
    kind = meta["run"].split("_")[1]
    drawer = meta.get("drawer") or ""
    oid = _moved(before, after, kind, drawer)
    o = after["objects"].get(oid) if oid else None
    if kind in ("find", "take"):
        placed = bool(o and o["in_handover"])
    else:
        placed = bool(o and o["in_drawer"] == drawer)
    at_rest = bool(o and o["speed_mps"] < REST_MPS)
    open_after = {k: v["open_m"] for k, v in after["drawers"].items() if v["open_m"] >= SHUT_M}
    strays = [i for i, b in before["objects"].items()
              if i != oid and after["objects"][i]["in_drawer"] != b["in_drawer"]]
    latency = [e["latency_s"] for e in ev if e["event"] == "vlm_reply"]
    return {
        "run": d.name, "kind": kind, "seed": meta["seed"], "drawer": drawer or None,
        "pass": placed and at_rest and not open_after and not strays,
        "object": oid, "object_after": o, "placed": placed, "at_rest": at_rest,
        "open_after": open_after, "max_open_m": max(v["open_m"] for v in after["drawers"].values()),
        "strays": {i: [before["objects"][i]["in_drawer"], after["objects"][i]["in_drawer"]] for i in strays},
        "brain_ok": result.get("ok"), "why": result.get("why"), "seconds": result["t"],
        "drawers_opened": result.get("drawers_opened"), "vlm_calls": len(latency),
        "vlm_latency_median_s": statistics.median(latency) if latency else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=Path, default=Path("/home/hkim/mycobot-sim/runs"))
    p.add_argument("--glob", default="d3_*")
    p.add_argument("--json", type=Path, help="also write the rows here")
    args = p.parse_args()
    rows: List[Dict[str, Any]] = []
    for d in sorted(x for x in args.runs.glob(args.glob) if x.is_dir()):
        try:
            rows.append(score(d))
        except (FileNotFoundError, json.JSONDecodeError, IndexError, KeyError) as exc:
            print(f"{d.name}: incomplete ({exc!r})")
            continue
        r = rows[-1]
        o = r["object_after"] or {}
        print(f"{r['run']}: {'PASS' if r['pass'] else 'FAIL'} {r['object']} -> in_drawer={o.get('in_drawer')} "
              f"in_handover={o.get('in_handover')} v={o.get('speed_mps')} | max open {r['max_open_m']:.4f} m, "
              f"strays {r['strays'] or 'none'} | brain_ok={r['brain_ok']} opened {r['drawers_opened']} "
              f"{r['seconds']:.1f} s, {r['vlm_calls']} VLM calls")
    print(f"{sum(r['pass'] for r in rows)}/{len(rows)} pass")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
