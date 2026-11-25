import copy
import json
import math
from pathlib import Path

import pytest
import scene

from mycobot_brain import calib, kinematics


def test_the_scene_file_loads(tmp_path):
    c = calib.load(scene.write(tmp_path / "calib.json"))
    assert c.camera == kinematics.Intrinsics(500, 500, 320, 240, 640, 480)
    assert c.block_plane_z == pytest.approx(0.0125)
    assert c.box_rim_z == pytest.approx(0.05)
    assert (c.gripper_open, c.gripper_closed) == (80, 15)
    assert c.chain.pose([0] * 6) == kinematics.MYCOBOT_280_JN.pose([0] * 6)


def test_rpy_quaternion_and_matrix_transforms_agree():
    rpy = calib._tf({"xyz": [0.01, 0.02, 0.03], "rpy": [0.3, -0.2, 1.1]}, "t")
    r = kinematics.rpy_matrix(0.3, -0.2, 1.1)
    # quaternion of r (r is proper, trace > -1 here)
    w = math.sqrt(1 + r[0][0] + r[1][1] + r[2][2]) / 2
    x, y, z = (r[2][1] - r[1][2]) / (4 * w), (r[0][2] - r[2][0]) / (4 * w), (r[1][0] - r[0][1]) / (4 * w)
    for spec in ({"xyz": [0.01, 0.02, 0.03], "quat_xyzw": [x, y, z, w]},
                 {"xyz": [0.01, 0.02, 0.03], "quat_wxyz": [w, x, y, z]},
                 rpy):
        t = calib._tf(spec, "t")
        assert [v for row in t for v in row] == pytest.approx([v for row in rpy for v in row], abs=1e-12)


def test_rim_can_be_given_absolute_and_K_as_fields():
    d = copy.deepcopy(scene.SCENE)
    d["table_z_m"] = 0.4
    d["box"] = {"size_m": [0.1, 0.1, 0.06], "rim_z_m": 0.47}
    d["camera"] = {"fx": 600, "fy": 601, "cx": 319.5, "cy": 239.5}
    c = calib.from_dict(d)
    assert c.box_rim_z == 0.47 and c.camera.fy == 601 and c.block_plane_z == pytest.approx(0.4125)


def test_a_missing_field_is_named(tmp_path):
    d = copy.deepcopy(scene.SCENE)
    del d["look_pose_deg"]
    (tmp_path / "c.json").write_text(json.dumps(d))
    with pytest.raises(calib.CalibError, match="look_pose_deg"):
        calib.load(tmp_path / "c.json")


SIM = Path(__file__).parent / "data" / "sim_calib.json"


def test_the_sim_calib_loads_and_fk_matches_the_sims_own():
    """data/sim_calib.json is the Isaac Lab scene's file as sim-scene wrote it
    (2026-11-01); fk_check holds the sim's TCP and camera poses at five angle sets."""
    c = calib.load(SIM)
    assert (c.block_edge, c.box_rim_z, c.gripper_open, c.gripper_closed) == (0.025, 0.045, 100, 0)
    assert c.pad_below_tcp == pytest.approx(0.0114) and c.camera.fx == pytest.approx(548.3)
    for check in json.loads(SIM.read_text())["fk_check"]:
        q = check["angles_deg"]
        assert kinematics.position(c.chain.pose(q, "tcp")) == pytest.approx(check["tcp_xyz_m"], abs=1e-5)
        cam = c.chain.pose(q, "camera")
        assert kinematics.position(cam) == pytest.approx(check["camera_xyz_m"], abs=1e-5)
        assert [cam[i][j] for i in range(3) for j in range(3)] == pytest.approx(
            [v for row in check["camera_R"] for v in row], abs=1e-5)


def test_firmware_offsets_follow_the_sim_convention():
    d = json.loads(SIM.read_text())
    d["arm_joints"][0]["offset_deg"] = 10.0
    c = calib.from_dict(d)
    assert c.chain.pose([10, 0, 0, 0, 0, 0], "tcp") == calib.load(SIM).chain.pose([0] * 6, "tcp")
