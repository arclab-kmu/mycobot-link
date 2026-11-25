#!/usr/bin/env python3
"""Write the myCobot 280 JN URDF this directory plans with.

The vendor file in ``~/manipulation/mycobot_ros2`` is malformed XML and has
zero velocity limits; ``~/manipulation/arc-mycobot`` already repairs it
(``assets/robots/mycobot_urdf.py``, see its docstring for the four defects).
That module is loaded here **by file path**, not imported as a package, so
nothing from Isaac Lab comes along - it only uses the standard library. Its
``"welded"`` mode fixes the gripper open, which leaves the six arm joints: the
same DOFs pymycobot's ``send_angles`` commands.

Two things are changed. Every mesh is converted to STL in metres
(``convert_meshes.py`` explains why the ``.dae`` files cannot be used as they
are) and the URDF points at those. And one thing is added: a virtual ``tcp``
link on ``joint6_flange``, 79 mm along the flange +z, with the flange's orientation.
TODO(unverified): 79 mm is an estimate from the URDF, not a measurement - the
gripper base sits 34 mm out (``joint6output_to_gripper_base``), its fingertip
origins 21 mm further, and arc-mycobot puts the pad centre 24 mm beyond those
(``_PAD_FORWARD``). The approach axis is gripper_base +y, which the 1.579 rad
roll of that fixed joint turns into flange +z (off by 0.5 deg).

``--camera-flange``: camera flange between J6 and the gripper
------------------------------------------------------------------
Elephant Robotics ships no URDF with *both* the camera flange and a gripper,
and none with the camera flange for the JN at all. What they do ship, for the
280 Pi / M5 / RISCV (one identical ``camera_flange.dae``, same md5), is the
flange alone and the flange with the suction pump. Measured in the
``joint6_flange`` frame from those files:

* the J6 flange face is at z = 1.7 mm, and without a camera flange both the
  pump (``*_with_pump``) and the adaptive gripper (``*_adaptive_gripper``, JN
  and Pi alike) start right there, at z = 1.6-1.9 mm;
* the camera flange plate occupies z = 1.9 .. 18.1 mm (16.2 mm thick), its pin
  pattern centred on the flange axis and its lens on the tool side at
  (41.5, 0, 18.1) mm, looking along flange +z;
* with the camera flange the pump starts at z = 18.9 mm: the vendor stacks the
  tool **17.0 mm further out along flange +z** and changes nothing else.

The same rule is applied here to the adaptive gripper: the camera flange is
placed exactly as in the vendor Pi file, ``joint6output_to_gripper_base`` and
the ``tcp`` move out by :data:`CAMERA_FLANGE_STACK_M`, and a ``camera_link``
frame is added at the lens.

TODO(unverified), all three to check against the assembled arm:
* the 17.0 mm stack height (a calliper across the camera flange settles it);
* ``--camera-yaw``: which way the camera sticks out relative to the jaw. The
  vendor puts it along flange +x (yaw 0), which is also the axis the jaw opens
  along (right finger at +x), so the finger would sit over the lens. This lab's
  default is **90 deg: camera toward flange +y**, clear of both fingers (set
  2026-09-23 at the user's request). The pin pattern allows 90 deg steps;
* ``camera_link`` is the lens centre from the mesh with z = viewing direction.
  Its x/y and the true optical centre need a hand-eye calibration anyway.

At q = [0, 0, 0, 0, 0, 0], Hojin observed the installed camera and gripper
rotated 45 deg counterclockwise when viewed from the front. The camera-flange
variant therefore adds a shared ``tool_mount`` frame rotated +45 deg about the
J6 flange +z axis. ``--assembly-yaw`` changes that shared mounting angle;
``--camera-yaw`` still sets the camera's direction relative to the gripper.
"""

from __future__ import annotations

import argparse
import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ARC_MODULE = Path.home() / "manipulation/arc-mycobot/src/arc_mycobot/assets/robots/mycobot_urdf.py"
TCP_OFFSET_M = (0.0, 0.0, 0.079)
FLANGE = "joint6_flange"

DESC = Path.home() / "manipulation/mycobot_ros2/mycobot_description/urdf"
CAMERA_FLANGE_DAE = DESC / "mycobot_280_pi/camera_flange.dae"
CAMERA_FLANGE_STACK_M = 0.017
#: Rotation of the camera flange about the flange axis. 0 is the vendor file's
#: placement (camera toward +x); 90 turns it to +y, which is how this lab mounts it.
DEFAULT_CAMERA_YAW_DEG = 90.0
# User-observed rotation of the complete camera/gripper assembly at joint zero.
DEFAULT_ASSEMBLY_YAW_DEG = 45.0
#: Lens centre on the camera flange's tool-side face, in the (un-yawed) flange frame.
CAMERA_LENS_M = (0.0415, 0.0, 0.0181)


def load_repair_module(path: Path):
    spec = importlib.util.spec_from_file_location("mycobot_urdf", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixed(root: ET.Element, name: str, parent: str, child: str, xyz, rpy=(0.0, 0.0, 0.0)) -> None:
    joint = ET.SubElement(root, "joint", name=name, type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=" ".join(f"{v:.6g}" for v in xyz), rpy=" ".join(f"{v:.6g}" for v in rpy))


def add_tcp(root: ET.Element, extra_z: float = 0.0, parent: str = FLANGE) -> None:
    if root.find("link[@name='tcp']") is not None:
        return
    ET.SubElement(root, "link", name="tcp")
    x, y, z = TCP_OFFSET_M
    _fixed(root, "joint6_flange_to_tcp", parent, "tcp", (x, y, z + extra_z))


def add_camera_flange(root: ET.Element, yaw_deg: float, assembly_yaw_deg: float) -> None:
    """Camera flange as in the vendor Pi URDF, the gripper pushed out by the stack height."""
    import math

    ET.SubElement(root, "link", name="tool_mount")
    _fixed(root, "joint6_flange_to_tool_mount", FLANGE, "tool_mount", (0, 0, 0),
           (0, 0, math.radians(assembly_yaw_deg)))
    ET.SubElement(root, "link", name="camera_mount")  # plate yaw relative to the gripper
    _fixed(root, "joint6_flange_to_camera_mount", "tool_mount", "camera_mount", (0, 0, 0),
           (0, 0, math.radians(yaw_deg)))
    link = ET.SubElement(root, "link", name="camera_flange")
    visual = ET.SubElement(link, "visual")
    # Vendor mycobot_280_pi_with_camera_flange.urdf, verbatim: joint xyz (0, 0, 0.01)
    # rpy (1.579, 0, 0), visual origin xyz (0.024, -0.008, 0) rpy (0, pi, 0). That
    # joint also carries a commented-out origin (rpy -1.5708) - a text search finds
    # that one first; it is not the live value.
    ET.SubElement(visual, "origin", xyz="0.024 -0.008 0.0", rpy="0 3.14159 0")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=str(CAMERA_FLANGE_DAE))
    _fixed(root, "camera_mount_to_camera_flange", "camera_mount", "camera_flange", (0, 0, 0.01), (1.579, 0, 0))
    ET.SubElement(root, "link", name="camera_link")
    _fixed(root, "camera_mount_to_camera_link", "camera_mount", "camera_link", CAMERA_LENS_M)

    joint = root.find("joint[@name='joint6output_to_gripper_base']")
    if joint is None:
        raise ValueError("joint6output_to_gripper_base not found; the repaired URDF changed")
    joint.find("parent").set("link", "tool_mount")
    origin = joint.find("origin")
    xyz = [float(v) for v in origin.get("xyz").split()]
    xyz[2] += CAMERA_FLANGE_STACK_M
    origin.set("xyz", " ".join(f"{v:.6g}" for v in xyz))


def use_stl_meshes(root: ET.Element, mesh_dir: Path) -> None:
    """Visuals -> metre STL; every link's collision -> the convex hull of its visual."""
    from convert_meshes import convert, convex_hull  # sibling script

    for link in root.findall("link"):
        visual = link.find("visual")
        if visual is None or visual.find("geometry/mesh") is None:
            continue
        mesh = visual.find("geometry/mesh")
        stl = convert(Path(mesh.get("filename")), mesh_dir)
        mesh.set("filename", str(stl))
        mesh.attrib.pop("scale", None)
        for old in link.findall("collision"):
            link.remove(old)
        collision = ET.SubElement(link, "collision")
        origin = visual.find("origin")
        if origin is not None:
            ET.SubElement(collision, "origin", dict(origin.attrib))
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", filename=str(convex_hull(stl)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None,
                        help="default urdf/mycobot_280_jn.urdf, or urdf/mycobot_280_jn_camera_gripper.urdf")
    parser.add_argument("--camera-flange", action="store_true", help="camera flange between J6 and the gripper")
    parser.add_argument("--camera-yaw", type=float, default=DEFAULT_CAMERA_YAW_DEG,
                        help="camera flange rotation about the flange axis, deg "
                             f"(default {DEFAULT_CAMERA_YAW_DEG:g} = camera toward +y; vendor file: 0 = +x)")
    parser.add_argument("--assembly-yaw", type=float, default=DEFAULT_ASSEMBLY_YAW_DEG,
                        help="camera and gripper mount rotation about J6 flange +z, deg "
                             f"(default {DEFAULT_ASSEMBLY_YAW_DEG:g} = observed CCW at joint zero)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repaired = load_repair_module(ARC_MODULE).build_mycobot_urdf("welded", force=args.force)
    tree = ET.parse(repaired)
    root = tree.getroot()
    if args.camera_flange:
        add_camera_flange(root, args.camera_yaw, args.assembly_yaw)
        root.set("name", root.get("name", "mycobot") + "_camera_gripper")
    use_stl_meshes(root, HERE / "assets/meshes")
    add_tcp(root, CAMERA_FLANGE_STACK_M if args.camera_flange else 0.0,
            "tool_mount" if args.camera_flange else FLANGE)
    default = "urdf/mycobot_280_jn_camera_gripper.urdf" if args.camera_flange else "urdf/mycobot_280_jn.urdf"
    out = Path(args.output or HERE / default)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"source : {repaired}\nwrote  : {out}")


if __name__ == "__main__":
    main()
