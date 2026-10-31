"""The scene's fixed geometry: kinematic chain, camera, table, block, box, look pose.

Read from one JSON file (``sim_calib.json`` from the Isaac Lab scene, or one
measured on the real arm). It holds no object poses: where the block and the
box are is for the VLM to say, from the camera image, every run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from mycobot_brain.kinematics import Chain, Intrinsics, Joint, Mat, quat_matrix, rpy_matrix, transform


class CalibError(ValueError):
    """The calib file is missing a field or has one in a shape this loader does not know."""


@dataclass
class Calib:
    chain: Chain
    camera: Intrinsics
    table_z: float
    block_edge: float
    box_size: Sequence[float]  # outer x, y, z (m)
    box_rim_z: float
    look_pose_deg: List[float]
    gripper_open: int
    gripper_closed: int
    source: str = ""

    @property
    def block_plane_z(self) -> float:
        """Height of the block's centre on the table: the plane a block pixel is put on."""
        return self.table_z + self.block_edge / 2


def _tf(spec: Any, what: str) -> Mat:
    """``{"xyz", "rpy"}``, ``{"xyz", "quat_xyzw"}``, ``{"xyz", "quat_wxyz"}`` or a 4x4 list."""
    if isinstance(spec, list) and len(spec) == 4 and all(isinstance(r, list) and len(r) == 4 for r in spec):
        return [[float(x) for x in row] for row in spec]
    if isinstance(spec, dict):
        xyz = spec.get("xyz", spec.get("pos", spec.get("translation", (0.0, 0.0, 0.0))))
        if "rpy" in spec:
            return transform(rpy_matrix(*spec["rpy"]), xyz)
        if "rpy_deg" in spec:
            return transform(rpy_matrix(*(math.radians(a) for a in spec["rpy_deg"])), xyz)
        if "quat_xyzw" in spec:
            return transform(quat_matrix(*spec["quat_xyzw"]), xyz)
        if "quat_wxyz" in spec:
            w, x, y, z = spec["quat_wxyz"]
            return transform(quat_matrix(x, y, z, w), xyz)
        if "matrix" in spec:
            return _tf(spec["matrix"], what)
        return transform(None, xyz)
    raise CalibError(f"{what}: cannot read a transform from {str(spec)[:120]}")


def _need(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    raise CalibError(f"missing {' / '.join(keys)}")


def chain_from(spec: Dict[str, Any]) -> Chain:
    joints = []
    for i, j in enumerate(_need(spec, "joints")):
        joints.append(Joint(
            origin=_tf(j.get("origin", j), f"joint {i + 1}"),
            axis=tuple(float(a) for a in j.get("axis", (0, 0, 1))),  # type: ignore[arg-type]
            sign=float(j.get("sign", 1.0)),
            offset_deg=float(j.get("offset_deg", 0.0)),
        ))
    if len(joints) != 6:
        raise CalibError(f"need 6 joints, got {len(joints)}")
    tools = {"camera": _tf(_need(spec, "flange_to_camera", "camera"), "flange_to_camera"),
             "tcp": _tf(_need(spec, "flange_to_tcp", "tcp"), "flange_to_tcp")}
    return Chain(joints=joints, base=_tf(spec.get("base", {}), "base"),
                 flange=_tf(spec.get("flange", spec.get("joint6_to_flange", {})), "flange"), tools=tools)


def intrinsics_from(spec: Dict[str, Any]) -> Intrinsics:
    k = spec.get("K", spec)
    if isinstance(k, list):
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    else:
        fx, fy, cx, cy = (_need(k, n) for n in ("fx", "fy", "cx", "cy"))
    return Intrinsics(float(fx), float(fy), float(cx), float(cy),
                      int(spec.get("width", 640)), int(spec.get("height", 480)))


def from_dict(d: Dict[str, Any], source: str = "") -> Calib:
    try:
        box = d.get("box", {})
        gripper = d.get("gripper", {})
        table_z = float(_need(d, "table_z_m", "table_z"))
        calib = Calib(
            chain=chain_from(_need(d, "chain", "kinematics")),
            camera=intrinsics_from(_need(d, "camera", "intrinsics")),
            table_z=table_z,
            block_edge=float(_need(d, "block_edge_m", "block_edge")),
            box_size=[float(x) for x in _need(box, "size_m", "outer_m", "size")],
            box_rim_z=float(box["rim_z_m"]) if "rim_z_m" in box else table_z + float(_need(box, "rim_height_m", "height_m")),
            look_pose_deg=[float(a) for a in _need(d, "look_pose_deg")],
            gripper_open=int(_need(gripper, "open")),
            gripper_closed=int(_need(gripper, "closed")),
            source=source,
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise CalibError(f"{source or 'calib'}: {exc!r}") from exc
    if len(calib.look_pose_deg) != 6:
        raise CalibError(f"look_pose_deg needs 6 angles, got {calib.look_pose_deg}")
    return calib


def load(path: Union[str, Path]) -> Calib:
    try:
        return from_dict(json.loads(Path(path).read_text()), str(path))
    except CalibError as exc:
        raise CalibError(f"{path}: {exc}") from exc
