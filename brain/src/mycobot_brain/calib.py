"""The scene's fixed geometry: kinematic chain, camera, table, block, box, look pose,
and for the drawer scene the cabinet (:mod:`mycobot_brain.fixture`).

Read from one JSON file (``sim_calib.json`` from the Isaac Lab scene, or one
measured on the real arm). It holds no object poses: where the block and the
box are, and what is in which drawer, is for the VLM to say, from the camera
image, every run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from mycobot_brain import collision
from mycobot_brain import fixture as fixture_mod
from mycobot_brain.kinematics import Chain, Intrinsics, Joint, Mat, quat_matrix, rpy_matrix, transform


class CalibError(ValueError):
    """The calib file is missing a field or has one in a shape this loader does not know."""


@dataclass
class Calib:
    chain: Chain
    camera: Intrinsics
    table_z: float
    block_edge: Optional[float]  # None in a scene without the block (the drawers)
    box_size: Sequence[float]  # outer x, y, z (m); empty without the box
    box_rim_z: Optional[float]
    look_pose_deg: List[float]
    gripper_open: int
    gripper_closed: int
    #: how far the finger pads reach past the TCP along the approach axis.
    pad_below_tcp: float = 0.0
    #: the jaw's gap shut and fully open (m); with it the pads holding an object sit at its width
    jaw_gap: Optional[Sequence[float]] = None
    source: str = ""
    fixture: Optional[fixture_mod.Fixture] = None
    #: the arm's links as capsules (calib v3 ``robot_collision``); None: only the TCP and camera are checked.
    robot: Optional[collision.RobotBody] = None

    @property
    def block_plane_z(self) -> float:
        """Height of the block's centre on the table: the plane a block pixel is put on."""
        if self.block_edge is None:
            raise CalibError(f"{self.source or 'calib'} has no block")
        return self.table_z + self.block_edge / 2


def _tf(spec: Any, what: str) -> Mat:
    """``{"xyz", "rpy"}``, ``{"xyz", "quat_xyzw"}``, ``{"xyz", "quat_wxyz"}`` or a 4x4 list."""
    if isinstance(spec, list) and len(spec) == 4 and all(isinstance(r, list) and len(r) == 4 for r in spec):
        return [[float(x) for x in row] for row in spec]
    if isinstance(spec, dict):
        if "matrix" in spec:
            return _tf(spec["matrix"], what)
        xyz = spec.get("xyz", spec.get("xyz_m", spec.get("origin_xyz_m", spec.get("pos", (0.0, 0.0, 0.0)))))
        if "rpy" in spec or "origin_rpy_rad" in spec:
            return transform(rpy_matrix(*spec.get("rpy", spec.get("origin_rpy_rad"))), xyz)
        if "rpy_deg" in spec:
            return transform(rpy_matrix(*(math.radians(a) for a in spec["rpy_deg"])), xyz)
        if "quat_xyzw" in spec:
            return transform(quat_matrix(*spec["quat_xyzw"]), xyz)
        if "quat_wxyz" in spec:
            w, x, y, z = spec["quat_wxyz"]
            return transform(quat_matrix(x, y, z, w), xyz)
        return transform(None, xyz)
    raise CalibError(f"{what}: cannot read a transform from {str(spec)[:120]}")


def _need(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    raise CalibError(f"missing {' / '.join(keys)}")


def chain_from(spec: Dict[str, Any]) -> Chain:
    """``{"joints", "flange_to_camera", "flange_to_tcp"}`` or the sim calib's top level
    (``arm_joints``, ``flange_T_camera_optical``, ``flange_T_tcp``)."""
    joints = []
    rows = sorted(_need(spec, "joints", "arm_joints"), key=lambda j: j.get("firmware_index", 0))
    for i, j in enumerate(rows):
        joints.append(Joint(
            origin=_tf(j.get("origin", j), f"joint {i + 1}"),
            axis=tuple(float(a) for a in j.get("axis", (0, 0, 1))),  # type: ignore[arg-type]
            sign=float(j.get("sign", 1.0)),
            offset_deg=float(j.get("offset_deg", 0.0)),
        ))
    if len(joints) != 6:
        raise CalibError(f"need 6 joints, got {len(joints)}")
    links = [rows[0]["parent"]] + [j["child"] for j in rows] if all("child" in j for j in rows) and "parent" in rows[0] \
        else None
    # The fixed joints from the flange to the camera and TCP name the frames a collision capsule may sit in.
    fixed: Dict[str, Any] = {}
    for j in [*spec.get("flange_to_camera_joints", ()), *spec.get("flange_to_tcp_joints", ())]:
        if j.get("type", "fixed") == "fixed" and j["child"] not in fixed:
            fixed[j["child"]] = (j["parent"], _tf(j, j["child"]))
    tools = {"camera": _tf(_need(spec, "flange_to_camera", "flange_T_camera_optical"), "flange_to_camera"),
             "tcp": _tf(_need(spec, "flange_to_tcp", "flange_T_tcp"), "flange_to_tcp")}
    chain = Chain(joints=joints, base=_tf(spec.get("base", {}), "base"),
                  flange=_tf(spec.get("flange", spec.get("joint6_to_flange", {})), "flange"), tools=tools)
    if links:
        chain.links = links
    known = set(chain.frame_names())
    for name, (parent, tf) in fixed.items():
        if parent in known or parent in fixed:
            chain.fixed[name] = (parent, tf)
            known.add(name)
    return chain


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
        box = d.get("box") or {}
        gripper = d.get("gripper", {})
        table_z = float(_need(d, "table_z_m", "table_z"))
        has_cabinet = "cabinet" in d
        if "block" in d:
            block_edge: Optional[float] = float(d["block"]["edge_m"])
        elif has_cabinet and not any(k in d for k in ("block_edge_m", "block_edge")):
            block_edge = None
        else:
            block_edge = float(_need(d, "block_edge_m", "block_edge"))
        if box:
            box_size = [float(x) for x in _need(box, "size_m", "outer_m", "size")]
            rim: Optional[float] = (float(box["rim_z_m"]) if "rim_z_m" in box
                                    else table_z + float(_need(box, "rim_height_m", "height_m")))
        elif has_cabinet:
            box_size, rim = [], None
        else:
            raise CalibError("missing box")
        fix = fixture_mod.from_dict(d) if has_cabinet else None
        look = d.get("look_pose_deg") or (fix.cabinet_look_pose_deg if fix else None)
        if look is None:
            raise CalibError("missing look_pose_deg")
        pads = (d.get("tcp") or {}).get("pad_extent_along_tool_z_m")
        chain = chain_from(d["chain"] if "chain" in d else d)
        robot = collision.from_dict(d["robot_collision"], chain) if d.get("robot_collision") else None
        calib = Calib(
            chain=chain,
            camera=intrinsics_from(_need(d, "camera", "intrinsics")),
            table_z=table_z,
            block_edge=block_edge,
            box_size=box_size,
            box_rim_z=rim,
            look_pose_deg=[float(a) for a in look],
            gripper_open=int(_need(gripper, "open", "open_value")),
            gripper_closed=int(_need(gripper, "closed", "closed_value")),
            pad_below_tcp=float(max(pads)) if pads else 0.0,
            jaw_gap=(lambda g: None if not g else (float(g["closed"]), float(g["open"])))(
                (d.get("tcp") or {}).get("jaw_gap_m")),
            source=source,
            fixture=fix,
            robot=robot,
        )
    except (fixture_mod.FixtureError, collision.CollisionSpecError) as exc:
        raise CalibError(f"{source or 'calib'}: {exc}") from exc
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
