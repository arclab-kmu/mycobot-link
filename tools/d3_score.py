"""Score D3 drawer runs from their raw files alone (CONTRACT-drawers.md, D3).

Pass, from the sim's truth before and after the run:

- every drawer shut (``open_m`` < SHUT_M);
- the moved object at rest (``speed_mps`` < REST_MPS) where the command wants it:
  find, the yellow cone in the handover zone; take, one object that was in the
  named drawer, in the handover zone; put, the object that stood in the
  handover zone, in the named drawer;
- that object upright: its axis within UPRIGHT_DEG of vertical, from its truth quaternion (leader 16:58,
  after D3 v3.2 s502_e: a taught close tipped the cone just handed over). A sphere has no up and is
  not held to it (its tilt is printed);
- every other object in the drawer it started in;
- with ``--audit`` (sim-scene's ``collision_audit.py --out`` JSON, any number of them): zero colliding
  segments for the run, the arm against itself and the scene (BOARD 2026-11-02). A run the audits do
  not cover fails.

The brain's own ``result`` is printed beside it, not used.

    python3 tools/d3_score.py [--runs /home/hkim/mycobot-sim/runs] [--glob 'd3_*'] [--audit audit.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

SHUT_M = 0.003  # D1 shut its drawers to <= 0.7 mm
REST_MPS = 0.02  # the bridge's at_rest: resting cylinders and cones chatter near 0.01 m/s
UPRIGHT_DEG = 15.0


def tilt_deg(quat_wxyz: List[float]) -> float:
    """Angle between the body's z axis and the world's, from a unit quaternion (w, x, y, z)."""
    w, x, y, z = quat_wxyz
    return math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))


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


def score(d: Path, audit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = json.loads((d / "meta.json").read_text())
    before = json.loads((d / "truth_before.json").read_text())
    after = json.loads((d / "truth.json").read_text())
    ev = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()]
    result = [e for e in ev if e["event"] == "result"][-1]
    kind = meta["run"].split("_")[1]
    kind = "find" if kind == "demo" else kind  # the demo is a find on its own seed
    drawer = meta.get("drawer") or ""
    oid = _moved(before, after, kind, drawer)
    o = after["objects"].get(oid) if oid else None
    if kind in ("find", "take"):
        placed = bool(o and o["in_handover"])
    else:
        placed = bool(o and o["in_drawer"] == drawer)
    at_rest = bool(o and o["speed_mps"] < REST_MPS)
    tilt = tilt_deg(o["quat_wxyz"]) if o and o.get("quat_wxyz") else None
    upright = bool(o) and (o["shape"] == "sphere" or (tilt is not None and tilt < UPRIGHT_DEG))
    open_after = {k: v["open_m"] for k, v in after["drawers"].items() if v["open_m"] >= SHUT_M}
    strays = [i for i, b in before["objects"].items()
              if i != oid and after["objects"][i]["in_drawer"] != b["in_drawer"]]
    latency = [e["latency_s"] for e in ev if e["event"] == "vlm_reply"]
    collided: Optional[int] = None
    if audit is not None:
        row = audit.get(d.name)
        collided = None if row is None else int(row["segments"])
    clean = audit is None or collided == 0
    return {
        "run": d.name, "kind": kind, "seed": meta["seed"], "drawer": drawer or None,
        "pass": placed and at_rest and upright and not open_after and not strays and clean,
        "collision_segments": collided, "audit_row": None if audit is None else audit.get(d.name),
        "object": oid, "object_after": o, "placed": placed, "at_rest": at_rest, "upright": upright,
        "tilt_deg": None if tilt is None else round(tilt, 1),
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
    p.add_argument("--audit", type=Path, action="append", help="sim-scene's collision audit JSON (repeatable)")
    args = p.parse_args()
    audit: Optional[Dict[str, Any]] = None
    if args.audit:
        audit = {}
        for a in args.audit:
            audit.update(json.loads(a.read_text())["runs"])
    rows: List[Dict[str, Any]] = []
    # Run directories only: archived passes (d3_try*) sit beside them.
    runs = [x for x in args.runs.glob(args.glob)
            if x.is_dir() and x.name.split("_")[1:2] in (["put"], ["take"], ["find"], ["demo"])]
    for d in sorted(runs):
        try:
            rows.append(score(d, audit))
        except (FileNotFoundError, json.JSONDecodeError, IndexError, KeyError) as exc:
            print(f"{d.name}: incomplete ({exc!r})")
            continue
        r = rows[-1]
        o = r["object_after"] or {}
        print(f"{r['run']}: {'PASS' if r['pass'] else 'FAIL'} {r['object']} -> in_drawer={o.get('in_drawer')} "
              f"in_handover={o.get('in_handover')} v={o.get('speed_mps')} tilt={r['tilt_deg']} deg"
              f"{'' if r['upright'] else ' NOT UPRIGHT'} | max open {r['max_open_m']:.4f} m, "
              f"strays {r['strays'] or 'none'} | brain_ok={r['brain_ok']} opened {r['drawers_opened']} "
              f"{r['seconds']:.1f} s, {r['vlm_calls']} VLM calls"
              + ("" if audit is None else f" | collision segments {r['collision_segments']}"))
    print(f"{sum(r['pass'] for r in rows)}/{len(rows)} pass")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
