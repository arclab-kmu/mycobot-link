"""Console pieces that need no camera, VLM, arm or GPU."""

from __future__ import annotations

import json

import pytest

from mycobot_brain.console import vlm
from mycobot_brain.console.feeds import split_jpegs
from mycobot_brain.planner import PlannerError


def test_split_jpegs_cuts_complete_frames_and_keeps_the_tail():
    a = b"\xff\xd8AAA\xff\xd9"
    b = b"\xff\xd8BB\xff\xd9"
    buf = bytearray(b"--frame\r\nheaders\r\n\r\n" + a + b"\r\n--frame\r\n" + b + b"\r\n" + b"\xff\xd8CC")
    assert split_jpegs(buf) == [a, b]
    assert bytes(buf) == b"\xff\xd8CC"  # incomplete frame waits for more bytes
    buf += b"C\xff\xd9"
    assert split_jpegs(buf) == [b"\xff\xd8CCC\xff\xd9"]


def test_split_jpegs_keeps_a_split_start_marker():
    buf = bytearray(b"junk\xff")
    assert split_jpegs(buf) == []
    buf += b"\xd8X\xff\xd9"
    assert split_jpegs(buf) == [b"\xff\xd8X\xff\xd9"]


def _step(**kw):
    s = {"op": "point", "target": "cup", "x": 500, "y": 500, "height_mm": 50, "dx_mm": 0, "dy_mm": 0, "dz_mm": 0,
         "grip": 0, "seconds": 0}
    s.update(kw)
    return s


def test_parse_chunk_clamps_and_keeps_order():
    raw = {"seen": "a cup", "say": "가요", "done": False,
           "steps": [_step(height_mm=999), _step(op="move", dz_mm=-500), _step(op="grip", grip=150)]}
    ch = vlm.parse_chunk(json.dumps(raw))
    assert [s["op"] for s in ch["steps"]] == ["point", "move", "grip"]
    assert ch["steps"][0]["height_mm"] == vlm.MAX_HEIGHT_MM
    assert ch["steps"][1]["dz_mm"] == -vlm.MAX_NUDGE_MM
    assert ch["steps"][2]["grip"] == 100


def test_parse_chunk_accepts_a_fenced_reply_and_caps_length():
    raw = {"seen": "", "say": "", "done": True, "steps": [_step()] * 20}
    ch = vlm.parse_chunk("```json\n" + json.dumps(raw) + "\n```")
    assert len(ch["steps"]) == vlm.MAX_STEPS and ch["done"] is True


@pytest.mark.parametrize("bad", ["not json", json.dumps({"steps": [{"op": "fly"}]}), json.dumps([1, 2])])
def test_parse_chunk_refuses_garbage(bad):
    with pytest.raises(PlannerError):
        vlm.parse_chunk(bad)


def test_chunk_request_carries_images_schema_and_no_thinking():
    req = vlm.build_chunk_request("컵 위로", {"top": b"\xff\xd8x\xff\xd9", "side": b"\xff\xd8y\xff\xd9"}, "tip at (0,0,300)",
                                  "+z is up")
    content = req["messages"][1]["content"]
    assert [p["type"] for p in content] == ["text", "image_url", "text", "image_url", "text"]
    assert req["response_format"]["json_schema"]["schema"] is vlm.CHUNK_SCHEMA
    assert req["chat_template_kwargs"] == {"enable_thinking": False}


np = pytest.importorskip("numpy")


def _workspace(tmp_path):
    from mycobot_brain.console import workspace

    cfg = {
        "table_z_mm": 197.0,
        "box": {"lo_mm": [-100, -300, 210], "hi_mm": [300, 300, 400]},
        "cameras": {"top": {
            "color_mjpg": "http://x/c.mjpg", "depth_mjpg": "http://x/d.mjpg", "color_jpg": "http://x/c.jpg",
            "K": {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0},
            # camera 700 mm above the world origin, looking straight down, image x = world +x, image y = world -y
            "R": [[1, 0, 0], [0, -1, 0], [0, 0, -1]], "t": [0, 0, 700]}},
        "arms": {"a": {"port": 1, "home_deg": None, "origin_mm": [100, 0, 0], "yaw_deg": 90}},
        "obstacles": [{"name": "slab", "center_mm": [200, 0, 100], "dims_mm": [10, 20, 30], "yaw_deg": 90},
                      {"name": "self", "center_mm": [0, 0, 0], "dims_mm": [1, 1, 1], "skip_for": "a"}],
    }
    p = tmp_path / "ws.json"
    p.write_text(json.dumps(cfg))
    return workspace.load(p)


def test_pixel_ray_and_projection_round_trip(tmp_path):
    ws = _workspace(tmp_path)
    cam = ws.cameras["top"]
    p = np.array([60.0, -30.0, 197.0])
    u, v = cam.world_to_pixel(p)
    c, d = cam.pixel_ray(u, v)
    hit = c + d * (197.0 - c[2]) / d[2]
    assert np.allclose(hit, p, atol=1e-6)
    assert "image right" in ws.world_axes_text() and "image up" in ws.world_axes_text()


def test_pixel_to_world_uses_the_depth_window(tmp_path):
    ws = _workspace(tmp_path)
    cam = ws.cameras["top"]
    depth = np.zeros((480, 640))
    depth[230:251, 310:331] = 503.0  # 503 mm below the camera = world z 197
    p, frac = cam.pixel_to_world(320, 240, depth)
    assert frac == 1.0 and np.allclose(p, [0, 0, 197.0])
    assert cam.pixel_to_world(10, 10, depth) == (None, 0.0)


def test_arm_frames_and_obstacles(tmp_path):
    ws = _workspace(tmp_path)
    arm = ws.arms["a"]
    w = np.array([100.0, 50.0, 10.0])  # 50 mm along world +y = arm +x (arm yawed +90)
    assert np.allclose(arm.to_base(w), [50, 0, 10])
    assert np.allclose(arm.to_world(arm.to_base(w)), w)
    obs = ws.obstacles_for(arm)
    assert [o["name"] for o in obs] == ["slab"]
    assert np.allclose(obs[0]["pose"][:3], [0, -100, 100])
    assert np.allclose(obs[0]["pose"][3:], [1, 0, 0, 0], atol=1e-9)  # world yaw 90 - arm yaw 90 = 0


def test_box_clamps(tmp_path):
    ws = _workspace(tmp_path)
    q, clipped = ws.box.clamp(np.array([0.0, 0.0, 100.0]))
    assert clipped and q[2] == 210
