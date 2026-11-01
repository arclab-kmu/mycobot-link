"""D2: does the VLM see what is in an opened drawer, and which drawers are open?

Sends each ground frame through the same requests the drawer task uses
(``mycobot_brain.drawer_vlm``) once per prompt variant and scores the answers
against ``truth.json``:

- drawer views (top-down and tilted): detections matched to the true objects
  by pixel; colour, shape and both right among the matched; recall and
  precision of the detections; yellow-cone precision/recall of the "target"
  answer, per frame, with yellow non-cones and non-yellow cones as distractors.
- cabinet views: each drawer's open/closed from ``open_drawers`` against the
  true opening.

Truth is read here and only here; the brain never sees it.

    uv run --project brain python tools/d2_grounding.py \\
        --ground /home/hkim/mycobot-sim/ground_drawers --calib /home/hkim/mycobot-sim/sim_calib_drawers.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from mycobot_brain import calib as calib_mod
from mycobot_brain import drawer_vlm, planner

TARGET = ("yellow", "cone")
OPEN_M = 0.02  # a drawer out more than this is open
#: drawer-view prompt variants: (floor hint, shape hints, colour hints)
DRAWER_VARIANTS = {"full": (True, True, False), "nofloor": (False, True, False), "plain": (False, False, False),
                   "colours": (True, True, True)}
#: cabinet-view variants: which handle points the prompt gives (shut, shut and fully open, none)
CABINET_VARIANTS = {"handles": "shut", "both": "both", "plain": None, "point": "point", "tray": "tray"}


def _get(o: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in o:
            return o[k]
    return default


def frames_from_truth(truth: Any, ground: Path) -> List[Dict[str, Any]]:
    """truth.json → [{image, view, drawer, angles_deg, objects: [...], open: {id: m}}].

    ``objects`` may be a list or a dict by id; ``colour``/``color`` and
    ``center_px``/``pixel`` are both read. An object counts as visible when it is
    in the image and, if truth says so, some of its colour shows (``color_px``
    is 0 behind a front panel or another object). In a drawer view the truth
    set is the visible objects of that drawer; ``view`` is topdown, tilted or
    cabinet.
    """
    out = []
    for r in truth["frames"]:
        view = _get(r, "view", "kind")
        raw = _get(r, "objects", default=[]) or []
        objects = []
        for o in (raw.values() if isinstance(raw, dict) else raw):
            shows = bool(_get(o, "in_image", "visible", default=True)) and o.get("color_px", 1) > 0
            if view != "cabinet" and r.get("drawer") and o.get("in_drawer", r["drawer"]) != r["drawer"]:
                shows = False
            objects.append({
                "id": _get(o, "id", default=""),
                "colour": _get(o, "colour", "color"),
                "shape": o["shape"],
                "pixel": [float(v) for v in _get(o, "center_px", "pixel")],
                "bbox": [float(v) for v in o["bbox_px"]] if o.get("bbox_px") else None,
                "visible": shows,
            })
        opening = {}
        for did, v in (_get(r, "drawers", "open_m", default={}) or {}).items():
            opening[did] = float(v["open_m"] if isinstance(v, dict) else v)
        out.append({"image": ground / r["file"], "view": view, "drawer": r.get("drawer"),
                    "angles_deg": r.get("angles_deg"), "objects": objects, "open": opening})
    return out


def ask(frame: Dict[str, Any], variant: str, scene: calib_mod.Calib, chat: Callable[[Dict[str, Any]], str]) -> Dict[str, Any]:
    fx, k = scene.fixture, scene.camera
    jpeg = frame["image"].read_bytes()
    q = frame["angles_deg"]
    if frame["view"] == "cabinet":
        mode = CABINET_VARIANTS[variant]
        handles = handles_open = tracks = boxes = None
        if mode in ("shut", "both") and q is not None:
            handles = {d.id: drawer_vlm.grid(scene.chain, k, q, d.knob_closed) for d in fx.drawers.values()}
        if mode == "both" and q is not None:
            handles_open = {d.id: drawer_vlm.grid(scene.chain, k, q, fx.knob(d, fx.open_m)) for d in fx.drawers.values()}
        if mode in ("point", "tray"):
            if q is None:
                return {"variant": variant, "error": "no angles_deg: nothing to match against", "latency_s": 0.0}
            if mode == "point":
                tracks = drawer_vlm.handle_tracks(fx, scene.chain, k, q)
            else:
                boxes = drawer_vlm.tray_boxes(fx, scene.chain, k, q, fx.open_m)
        req = drawer_vlm.build_cabinet_request(jpeg, fx, handles, goal="say which drawers are open",
                                               handles_open=handles_open, point=mode == "point",
                                               tray=mode == "tray")
        parse = lambda r: drawer_vlm.parse_cabinet(r, fx, None, tracks, boxes)  # noqa: E731
    else:
        floor_hint, shape_hints, colour_hints = DRAWER_VARIANTS[variant]
        d = fx.drawer(frame["drawer"])
        floor = drawer_vlm.floor_box(fx, d, fx.open_m, scene.chain, k, q) if floor_hint and q is not None else None
        req = drawer_vlm.build_drawer_request(jpeg, fx, d, floor, target=" ".join(TARGET), view=frame["view"],
                                              shape_hints=shape_hints, colour_hints=colour_hints)
        parse = lambda r: drawer_vlm.parse_drawer(r, fx, with_target=True, width=k.width, height=k.height)  # noqa: E731
    t0 = time.monotonic()
    try:
        raw = chat(req)
    except planner.PlannerError as exc:
        return {"variant": variant, "error": str(exc), "latency_s": round(time.monotonic() - t0, 3)}
    latency = round(time.monotonic() - t0, 3)
    try:
        return {"variant": variant, "raw": raw, "found": parse(raw), "latency_s": latency}
    except planner.PlannerError as exc:
        return {"variant": variant, "raw": raw, "error": str(exc), "latency_s": latency}


def _inside(p: Sequence[float], box: Optional[Sequence[float]], pad: float) -> bool:
    return box is not None and box[0] - pad <= p[0] <= box[2] + pad and box[1] - pad <= p[1] <= box[3] + pad


def match(dets: List[Dict[str, Any]], truth: List[Dict[str, Any]], max_px: float = 40.0) -> List[tuple]:
    """Greedy nearest pairs (detection index, truth index): the detection's centre within
    ``max_px`` of the true centre, or inside the true box (10 px slack)."""
    pairs = sorted((math.dist(d["pixel"], t["pixel"]), i, j) for i, d in enumerate(dets) for j, t in enumerate(truth))
    used_d, used_t, out = set(), set(), []
    for dist, i, j in pairs:
        if i in used_d or j in used_t:
            continue
        if dist <= max_px or _inside(dets[i]["pixel"], truth[j]["bbox"], 10.0):
            out.append((i, j))
            used_d.add(i)
            used_t.add(j)
    return out


def score_drawer(frame: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    truth = [o for o in frame["objects"] if o["visible"]]
    has_target = any((o["colour"], o["shape"]) == TARGET for o in truth)
    if "found" not in answer:
        return {"error": True, "has_target": has_target}
    dets = answer["found"]["objects"]
    pairs = match(dets, truth)
    row: Dict[str, Any] = {"n_true": len(truth), "n_det": len(dets), "matched": len(pairs),
                           "colour_ok": 0, "shape_ok": 0, "both_ok": 0, "confusions": [],
                           "has_target": has_target, "open": answer["found"]["open"]}
    det_to_truth = dict(pairs)
    for i, j in pairs:
        c_ok = dets[i]["colour"] == truth[j]["colour"]
        s_ok = dets[i]["shape"] == truth[j]["shape"]
        row["colour_ok"] += c_ok
        row["shape_ok"] += s_ok
        row["both_ok"] += c_ok and s_ok
        if not (c_ok and s_ok):
            row["confusions"].append(f"{truth[j]['colour']} {truth[j]['shape']} -> {dets[i]['colour']} {dets[i]['shape']}")
    t = answer["found"]["target"]
    said = t is not None
    right = said and t in det_to_truth and (truth[det_to_truth[t]]["colour"], truth[det_to_truth[t]]["shape"]) == TARGET
    row["target"] = "tp" if right else "fp" if said else "fn" if has_target else "tn"
    return row


def score_cabinet(frame: Dict[str, Any], answer: Dict[str, Any]) -> Dict[str, Any]:
    if "found" not in answer:
        return {"error": True}
    said = set(answer["found"]["open_drawers"])
    cells = {d: (m > OPEN_M, d in said) for d, m in frame["open"].items()}
    return {"cells": {d: {"true": t, "said": s} for d, (t, s) in cells.items()},
            "exact": all(t == s for t, s in cells.values())}


def _pct(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 3)


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 3) if b else None


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in sorted({(r["view"], r["variant"]) for r in rows}):
        view, variant = key
        rs = [r for r in rows if (r["view"], r["variant"]) == key]
        lat = [r["latency_s"] for r in rs if "latency_s" in r]
        s: Dict[str, Any] = {"frames": len(rs), "errors": sum(1 for r in rs if r["score"].get("error")),
                             "latency_median_s": _pct(lat, 0.5), "latency_p90_s": _pct(lat, 0.9)}
        sc = [r["score"] for r in rs if not r["score"].get("error")]
        if view == "cabinet":
            cells = [c for x in sc for c in x["cells"].values()]
            tp = sum(c["true"] and c["said"] for c in cells)
            fp = sum(c["said"] and not c["true"] for c in cells)
            fn = sum(c["true"] and not c["said"] for c in cells)
            s.update(drawers=len(cells), accuracy=_ratio(sum(c["true"] == c["said"] for c in cells), len(cells)),
                     open_precision=_ratio(tp, tp + fp), open_recall=_ratio(tp, tp + fn),
                     frames_exact=_ratio(sum(x["exact"] for x in sc), len(sc)))
        else:
            matched = sum(x["matched"] for x in sc)
            t = [x["target"] for x in sc]
            tp, fp, fn = t.count("tp"), t.count("fp"), t.count("fn")
            confusions: Dict[str, int] = {}
            for x in sc:
                for c in x["confusions"]:
                    confusions[c] = confusions.get(c, 0) + 1
            s.update(objects=sum(x["n_true"] for x in sc), detections=sum(x["n_det"] for x in sc), matched=matched,
                     recall=_ratio(matched, sum(x["n_true"] for x in sc)),
                     precision=_ratio(matched, sum(x["n_det"] for x in sc)),
                     colour_acc=_ratio(sum(x["colour_ok"] for x in sc), matched),
                     shape_acc=_ratio(sum(x["shape_ok"] for x in sc), matched),
                     both_acc=_ratio(sum(x["both_ok"] for x in sc), matched),
                     cone_tp=tp, cone_fp=fp, cone_fn=fn, cone_tn=t.count("tn"),
                     cone_precision=_ratio(tp, tp + fp), cone_recall=_ratio(tp, tp + fn),
                     said_open=_ratio(sum(1 for x in sc if x["open"]), len(sc)),
                     confusions=dict(sorted(confusions.items(), key=lambda kv: -kv[1])[:8]))
        out.setdefault(view, {})[variant] = s
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ground", type=Path, required=True, help="frames + truth.json")
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--drawer-variants", default=",".join(DRAWER_VARIANTS))
    p.add_argument("--cabinet-variants", default=",".join(CABINET_VARIANTS))
    p.add_argument("--views", default="topdown,tilted,cabinet", help="which frames to ask about")
    p.add_argument("--out", type=Path, help="default <ground>/d2-<time>")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--vllm-url", default=planner.DEFAULT_VLLM_URL)
    p.add_argument("--model", default=planner.DEFAULT_MODEL)
    args = p.parse_args(argv)

    scene = calib_mod.load(args.calib)
    if scene.fixture is None:
        p.error(f"{args.calib} has no cabinet")
    frames = frames_from_truth(json.loads((args.ground / "truth.json").read_text()), args.ground)
    out = args.out or args.ground / f"d2-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    def chat(request):
        request["model"] = args.model
        return planner.chat(request, base_url=args.vllm_url)

    # One job per frame, its variants in turn: vLLM 0.28 failed with "Expected a cached item for
    # mm_hash" (HTTP 500) when the same image was in two requests at once.
    views = args.views.split(",")
    jobs = [(f, (args.cabinet_variants if f["view"] == "cabinet" else args.drawer_variants).split(","))
            for f in frames if f["view"] in views]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        answers = pool.map(lambda job: [ask(job[0], v, scene, chat) for v in job[1]], jobs)
        for (f, _), per_frame in zip(jobs, answers):
            for answer in per_frame:
                score = score_cabinet(f, answer) if f["view"] == "cabinet" else score_drawer(f, answer)
                rows.append({"image": f["image"].name, "view": f["view"], "drawer": f["drawer"], **answer,
                             "score": score})
    with open(out / "answers.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    for view, variants in summary.items():
        for variant, s in variants.items():
            keys = (("accuracy", "open_precision", "open_recall", "frames_exact") if view == "cabinet" else
                    ("recall", "precision", "colour_acc", "shape_acc", "both_acc", "cone_precision", "cone_recall"))
            print(f"{view:8} {variant:8} n={s['frames']:3} err={s['errors']} "
                  + " ".join(f"{k}={s[k]}" for k in keys) + f" lat={s['latency_median_s']}")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
