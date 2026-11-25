"""tools/d2_grounding.py scores a fake VLM whose drawer answers have known mistakes."""

import base64
import importlib.util
import json

import cabinet_scene as cs
import pytest
from conftest import SCHEMA_DIR

from mycobot_brain import kinematics

spec = importlib.util.spec_from_file_location("d2", SCHEMA_DIR.parent / "tools" / "d2_grounding.py")
d2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d2)


def ground(tmp_path):
    """Three tilted drawer views (cone in r1c2 only) and one cabinet view with r1c2 open."""
    c = cs.load()
    fx = c.fixture
    contents = {"r0c0": [("red", "cube", 0.0, 0.015), ("yellow", "cylinder", 0.0, -0.015)],
                "r1c2": [("yellow", "cone", 0.0, 0.0)],
                "r2c1": [("green", "cone", 0.0, 0.0)]}
    frames = []
    for i, (did, objs) in enumerate(contents.items()):
        d = fx.drawers[did]
        centre = fx.interior_center(d, fx.open_m)
        q = kinematics.look_at(c.chain, c.look_pose_deg, (centre[0], centre[1], centre[2] + 0.01),
                               distance=0.18, tilt_deg=40)
        truth = []
        for colour, shape, dx, dy in objs:
            p = (centre[0] + dx, centre[1] + dy, d.floor_z + fx.shapes[shape].height / 2)
            u, v = kinematics.project(c.chain, c.camera, q, p)
            truth.append({"id": f"{colour}_{shape}", "color": colour, "shape": shape, "center_px": [u, v],
                          "bbox_px": [u - 15, v - 15, u + 15, v + 15], "in_image": True})
        name = f"drawer_{i}.jpg"
        (tmp_path / name).write_bytes(b"\xff\xd8" + bytes([i]) + b"\xff\xd9")
        frames.append({"file": name, "view": "tilted", "drawer": did, "angles_deg": q, "objects": truth})
    (tmp_path / "cab.jpg").write_bytes(b"\xff\xd8" + bytes([9]) + b"\xff\xd9")
    frames.append({"file": "cab.jpg", "view": "cabinet", "angles_deg": fx.cabinet_look_pose_deg,
                   "drawers": {d: {"open_m": fx.open_m if d == "r1c2" else 0.0} for d in fx.drawers}})
    (tmp_path / "truth.json").write_text(json.dumps({"frames": frames}))
    return tmp_path, frames


def test_scores_colour_shape_the_cone_and_open_drawers(tmp_path, capsys, monkeypatch):
    g, frames = ground(tmp_path)

    def fake_chat(request, base_url):
        url = request["messages"][1]["content"][0]["image_url"]["url"]
        idx = base64.b64decode(url.split(",")[1])[2]  # the frame's number, written into its fake JPEG
        f = frames[idx] if idx < 9 else frames[-1]
        if f["view"] == "cabinet":
            return json.dumps({"seen": "cabinet", "open_drawers": ["r1c2", "r0c0"]})  # one false open
        objs = []
        for o in f["objects"]:
            u, v = o["center_px"]
            box = [(u - 12) * 1000 / 640, (v - 12) * 1000 / 480, (u + 12) * 1000 / 640, (v + 12) * 1000 / 480]
            shape = "cylinder" if f["drawer"] == "r2c1" else o["shape"]  # the green cone read as a cylinder
            objs.append({"colour": o["color"], "shape": shape, "bbox_2d": box})
        target = next((i for i, o in enumerate(objs) if (o["colour"], o["shape"]) == ("yellow", "cone")), -1)
        if f["drawer"] == "r0c0":
            target = 1  # says the yellow cylinder is the cone: a false positive
        return json.dumps({"seen": "x", "open": True, "objects": objs, "target": target})

    monkeypatch.setattr(d2.planner, "chat", fake_chat)
    assert d2.main(["--ground", str(g), "--calib", str(cs.write(tmp_path / "c.json")), "--drawer-variants", "full",
                    "--cabinet-variants", "handles", "--out", str(tmp_path / "out")]) == 0
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    t = s["tilted"]["full"]
    assert (t["objects"], t["matched"], t["recall"], t["precision"]) == (4, 4, 1.0, 1.0)
    assert t["colour_acc"] == 1.0 and t["shape_acc"] == 0.75 and t["both_acc"] == 0.75
    assert (t["cone_tp"], t["cone_fp"], t["cone_fn"], t["cone_tn"]) == (1, 1, 0, 1)
    assert t["cone_precision"] == 0.5 and t["cone_recall"] == 1.0
    assert t["confusions"] == {"green cone -> green cylinder": 1}
    cab = s["cabinet"]["handles"]
    assert cab["accuracy"] == pytest.approx(8 / 9, abs=1e-3) and cab["open_precision"] == 0.5
    assert cab["open_recall"] == 1.0 and cab["frames_exact"] == 0.0
    assert "tilted" in capsys.readouterr().out


def test_match_pairs_nearest_first_and_ignores_far_detections():
    truth = [{"pixel": [100, 100], "bbox": None}, {"pixel": [200, 100], "bbox": [180, 80, 220, 120]}]
    dets = [{"pixel": [205, 104]}, {"pixel": [104, 98]}, {"pixel": [400, 400]}]
    assert sorted(d2.match(dets, truth)) == [(0, 1), (1, 0)]
