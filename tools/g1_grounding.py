"""G1: how far off are the VLM's pixels for the block and the box?

Sends each ground frame through the same locate request pick-place uses
(``mycobot_brain.grounding``), once per coordinate variant, and scores the
answer against ``truth.json``: pixel error, and the error on the plane in cm
(VLM pixel → camera ray → the plane pick-place would put it on, against the
true centre). Truth is read here and only here; the brain never sees it.

    uv run --project brain python tools/g1_grounding.py \\
        --ground /home/hkim/mycobot-sim/ground --calib /home/hkim/mycobot-sim/sim_calib.json

Writes ``answers.jsonl`` (one line per frame x variant) and ``summary.json``
to ``--out`` and prints the summary table.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mycobot_brain import calib as calib_mod
from mycobot_brain import grounding, kinematics, planner

INSTRUCTION = "빨간 블록을 잡아서 갈색 상자 안에 넣어"
ROLE_OF = {"block": "pick", "box": "place"}
VARIANTS = ("norm1000:bbox", "abs:bbox", "norm1000:point", "abs:point")


def _image_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _mime(data: bytes) -> str:
    return "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"


def frames_from_truth(truth: Any, ground: Path) -> List[Dict[str, Any]]:
    """sim-scene's truth.json → [{image, kind, angles_deg, objects: {block|box: {pixel, xyz, visible}}}].

    The block's truth pixel is its 3D centroid (``center_px``), the box's the
    centre at rim height (``rim_center_px``): the points pick-place puts on
    its two planes. ``pos_m`` is the block centre and the box's outer bottom
    centre; only x, y of it are compared.
    """
    out = []
    for r in truth["frames"]:
        objects = {}
        for name, key, px in (("block", "red_block", "center_px"), ("box", "brown_box", "rim_center_px")):
            o = r.get(key)
            if o:
                objects[name] = {"pixel": [float(v) for v in o[px]], "xyz": [float(v) for v in o["pos_m"]],
                                 "visible": bool(o.get("in_image", True))}
        out.append({"image": ground / r["file"], "kind": "look" if r["kind"] == "look" else "close",
                    "sub": r["kind"], "angles_deg": r["angles_deg"], "objects": objects})
    return out


def ask(frame: Dict[str, Any], variant: str, k: kinematics.Intrinsics, chat) -> Dict[str, Any]:
    coords, shape = variant.split(":")
    look = frame["kind"] == "look"
    seen = [n for n, o in frame["objects"].items() if o["visible"]]
    # As pick-place asks: both at the look pose; at a close look only the one it is over.
    roles = ("pick", "place") if look else tuple(ROLE_OF[n] for n in seen)
    data = _image_bytes(frame["image"])
    request = grounding.build_locate_request(data, INSTRUCTION, roles=roles, coords=coords, shape=shape,
                                             width=k.width, height=k.height, with_next=look)
    if _mime(data) != "image/jpeg":
        url = f"data:{_mime(data)};base64," + base64.b64encode(data).decode()
        request["messages"][1]["content"][0]["image_url"]["url"] = url
    t0 = time.monotonic()
    try:
        raw = chat(request)
    except planner.PlannerError as exc:
        return {"variant": variant, "roles": roles, "error": str(exc), "latency_s": round(time.monotonic() - t0, 3)}
    latency = round(time.monotonic() - t0, 3)
    try:
        found = grounding.parse_locate(raw, roles, coords=coords, shape=shape, width=k.width, height=k.height,
                                       with_next=look)
    except planner.PlannerError as exc:
        return {"variant": variant, "roles": roles, "raw": raw, "error": str(exc), "latency_s": latency}
    return {"variant": variant, "roles": roles, "raw": raw, "found": found, "latency_s": latency}


def score(frame: Dict[str, Any], answer: Dict[str, Any], scene: calib_mod.Calib) -> Dict[str, Any]:
    """Per object: px and cm error, or why there is none."""
    out = {}
    for name, truth in frame["objects"].items():
        role = ROLE_OF[name]
        if role not in answer["roles"]:
            continue
        if "found" not in answer:
            out[name] = {"miss": "error"}
            continue
        got = answer["found"][role]
        if not got["visible"]:
            out[name] = {"miss": "said not visible" if truth["visible"] else "agreed not visible"}
            continue
        plane = scene.block_plane_z if name == "block" else scene.box_rim_z
        q = frame["angles_deg"]
        row: Dict[str, Any] = {"pixel": got["pixel"]}
        true_px = truth["pixel"]
        if true_px is None and truth["xyz"] is not None and q is not None:
            true_px = kinematics.project(scene.chain, scene.camera, q, truth["xyz"])
        if true_px is not None:
            row["px"] = math.dist(got["pixel"], true_px)
        if q is not None:
            est = kinematics.pixel_to_plane(scene.chain, scene.camera, q, *got["pixel"], plane)
            ref = truth["xyz"] or (true_px and kinematics.pixel_to_plane(scene.chain, scene.camera, q, *true_px, plane))
            if est is not None and ref:
                row["cm"] = 100 * math.hypot(est[0] - ref[0], est[1] - ref[1])
        out[name] = row
    return out


def _pct(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = math.floor(i), math.ceil(i)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 2)


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for variant in sorted({r["variant"] for r in rows}):
        for kind in ("look", "close", "all"):
            for name in ("block", "box"):
                cells = [r["score"][name] for r in rows if r["variant"] == variant and name in r["score"]
                         and (kind == "all" or r["kind"] == kind)]
                if not cells:
                    continue
                px = [c["px"] for c in cells if "px" in c]
                cm = [c["cm"] for c in cells if "cm" in c]
                summary.setdefault(variant, {}).setdefault(kind, {})[name] = {
                    "n": len(cells), "scored": len(px), "missed": sum(1 for c in cells if "miss" in c),
                    "px_median": _pct(px, 0.5), "px_p90": _pct(px, 0.9),
                    "cm_median": _pct(cm, 0.5), "cm_p90": _pct(cm, 0.9),
                }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ground", type=Path, required=True, help="frames + truth.json")
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--variants", default=",".join(VARIANTS), help=f"comma list of coords:shape (default {VARIANTS})")
    p.add_argument("--out", type=Path, help="default <ground>/g1-<time>")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--vllm-url", default=planner.DEFAULT_VLLM_URL)
    p.add_argument("--model", default=planner.DEFAULT_MODEL)
    args = p.parse_args(argv)

    scene = calib_mod.load(args.calib)
    frames = frames_from_truth(json.loads((args.ground / "truth.json").read_text()), args.ground)
    out = args.out or args.ground / f"g1-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    def chat(request):
        request["model"] = args.model
        return planner.chat(request, base_url=args.vllm_url)

    jobs = [(f, v) for f in frames for v in args.variants.split(",")]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for (f, v), answer in zip(jobs, pool.map(lambda job: ask(job[0], job[1], scene.camera, chat), jobs)):
            rows.append({"image": f["image"].name, "kind": f["kind"], "sub": f["sub"], **answer,
                         "score": score(f, answer, scene)})
    with open(out / "answers.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{'variant':16} {'kind':5} {'obj':5} {'n':>3} {'miss':>4} {'px med':>7} {'px p90':>7} {'cm med':>7} {'cm p90':>7}")
    for variant, kinds in summary.items():
        for kind, objs in kinds.items():
            for name, s in objs.items():
                print(f"{variant:16} {kind:5} {name:5} {s['n']:3} {s['missed']:4} "
                      + " ".join(f"{s[k] if s[k] is not None else '-':>7}"
                                 for k in ("px_median", "px_p90", "cm_median", "cm_p90")))
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
