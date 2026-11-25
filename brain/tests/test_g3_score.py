"""tools/g3_score.py passes a run on the sim's truth, not on what the brain said."""

import importlib.util
import json

from conftest import SCHEMA_DIR

spec = importlib.util.spec_from_file_location("g3", SCHEMA_DIR.parent / "tools" / "g3_score.py")
g3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g3)


def run_dir(tmp_path, *, in_box, speed, gripper, brain_ok):
    d = tmp_path / "g3_s00"
    d.mkdir()
    (d / "truth.json").write_text(json.dumps({
        "ok": True, "in_box": in_box, "block_speed_mps": speed,
        "red_block": {"pos_m": [0.19, -0.04, 0.0164]},
        "brown_box": {"pos_m": [0.188, -0.039, 0.0], "rim_center_m": [0.188, -0.039, 0.045]}}))
    (d / "state_after.json").write_text(json.dumps({"ok": True, "gripper": gripper}))
    events = [
        {"event": "estimate", "role": "place", "source": "look", "xyz": [0.191, -0.035, 0.045]},
        {"event": "vlm_reply", "latency_s": 1.0},
        {"event": "vlm_reply", "latency_s": 2.0},
        {"event": "result", "t": 35.0, "ok": brain_ok, "why": "in the box", "attempts": 1},
    ]
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return d


def test_truth_decides_and_the_box_estimate_is_scored(tmp_path):
    r = g3.score(run_dir(tmp_path, in_box=True, speed=0.0, gripper=100, brain_ok=True))
    assert r["pass"] and r["vlm_calls"] == 2 and r["vlm_latency_median_s"] == 1.5
    assert r["box_estimate_err_mm"] == {"look": 5.0}  # (3, 4) mm off


def test_the_vlm_saying_yes_does_not_pass_a_block_still_rolling(tmp_path):
    r = g3.score(run_dir(tmp_path, in_box=True, speed=0.02, gripper=100, brain_ok=True))
    assert not r["pass"] and r["brain_ok"] is True


def test_a_closed_gripper_fails(tmp_path):
    assert not g3.score(run_dir(tmp_path, in_box=True, speed=0.0, gripper=20, brain_ok=True))["pass"]
