"""tools/g1_grounding.py scores a fake VLM whose answers are off by a known amount."""

import importlib.util
import json

import pytest
import scene
from conftest import SCHEMA_DIR

from mycobot_brain import kinematics

spec = importlib.util.spec_from_file_location("g1", SCHEMA_DIR.parent / "tools" / "g1_grounding.py")
g1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1)


def ground(tmp_path):
    c = scene.load()
    frames = []
    for i in range(3):
        q = list(c.look_pose_deg)
        q[0] += i  # three slightly different looks
        name = f"look_{i:02d}.jpg"
        (tmp_path / name).write_bytes(b"\xff\xd8" + bytes([i]) + b"\xff\xd9")
        frames.append({"file": name, "kind": "look", "angles_deg": q,
                       "red_block": {"center_px": list(kinematics.project(c.chain, c.camera, q, scene.BLOCK)),
                                     "pos_m": list(scene.BLOCK), "in_image": True},
                       "brown_box": {"rim_center_px": list(kinematics.project(c.chain, c.camera, q, scene.BOX)),
                                     "pos_m": [*scene.BOX[:2], 0.0], "in_image": True}})
    (tmp_path / "truth.json").write_text(json.dumps({"frames": frames}))
    return tmp_path, frames


def test_errors_are_measured_in_pixels_and_on_the_plane(tmp_path, monkeypatch, capsys):
    g, frames = ground(tmp_path)
    by_image = {f["file"]: f for f in frames}

    def fake_chat(request, base_url):
        jpeg = g1.base64.b64decode(request["messages"][1]["content"][0]["image_url"]["url"].split(",")[1])
        q = by_image[f"look_{jpeg[2]:02d}.jpg"]["angles_deg"]
        vlm = scene.SceneVLM(scene.load(), lambda: q, bias=(3, 4),
                             coords="abs" if "pixels of this" in request["messages"][0]["content"] else "norm1000")
        return vlm(request)

    monkeypatch.setattr(g1.planner, "chat", fake_chat)
    assert g1.main(["--ground", str(g), "--calib", str(scene.write(tmp_path / "c.json")),
                    "--variants", "norm1000:bbox,abs:bbox", "--out", str(tmp_path / "out")]) == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    for variant in ("norm1000:bbox", "abs:bbox"):
        block = summary[variant]["look"]["block"]
        assert block["n"] == 3 and block["missed"] == 0
        assert block["px_median"] == pytest.approx(5.0, abs=0.05)  # (3, 4) off
        assert 0.1 < block["cm_median"] < 0.5  # 5 px at ~0.2 m and f = 500
    rows = (tmp_path / "out" / "answers.jsonl").read_text().splitlines()
    assert len(rows) == 6
    assert "norm1000:bbox" in capsys.readouterr().out


def test_a_frame_without_angles_still_gets_a_pixel_score(tmp_path):
    g, frames = ground(tmp_path)
    f = g1.frames_from_truth({"frames": [dict(frames[0], angles_deg=None)]}, g)[0]
    answer = {"roles": ("pick", "place"), "found": {
        "pick": {"visible": True, "pixel": [f["objects"]["block"]["pixel"][0] + 6, f["objects"]["block"]["pixel"][1]]},
        "place": {"visible": False}}}
    s = g1.score(f, answer, scene.load())
    assert s["block"] == {"pixel": answer["found"]["pick"]["pixel"], "px": pytest.approx(6.0)}
    assert s["box"] == {"miss": "said not visible"}
