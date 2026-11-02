"""A 3x3 drawer cabinet on the sim arm's chain, a world that follows the arm,
and a fake VLM that answers from that world.

The world replays the run's ``move`` / ``gripper`` events: a gripper that
closes on a handle drags the drawer along the pull axis as the TCP moves; one
that closes on an object carries it; opening drops it on whatever is under it
(an open drawer's floor, the handover zone, the table). So a test asserts what
happened to the drawers and objects, not only what was sent. With the calib's
hulls it also lists ``faults``: an object let go onto another's footprint, and a
finger hull touching the table, a tray, or an object standing (leader 18:5x:
the pads are never excused from contact; D0 must not plan what the sim tips).
"""

import functools
import json
import math
import re
from pathlib import Path

from mycobot_brain import calib, collision, drawer_vlm, kinematics

SIM = json.loads((Path(__file__).parent / "data" / "sim_calib.json").read_text())

FRONT_X = 0.26
FLOORS = (0.005, 0.04, 0.075)  # rows 0..2
COLS_Y = (0.072, 0.0, -0.072)  # cols 0..2: robot's left is +y
INNER = (0.095, 0.066, 0.028)  # x, y, z
OPEN_M = 0.085
HANDLE_X = FRONT_X - 0.012
HANDOVER = (0.13, -0.17)

SHAPES = {
    "cube": {"height_m": 0.024, "width_m": 0.024, "grasp_height_m": 0.016},
    "cylinder": {"height_m": 0.026, "width_m": 0.028, "grasp_height_m": 0.016},
    "cone": {"height_m": 0.026, "width_m": 0.032, "grasp_height_m": 0.016},
    "sphere": {"height_m": 0.026, "width_m": 0.026, "grasp_height_m": 0.016},
}
COLOURS = ["red", "yellow", "blue", "green", "orange", "purple"]


def _drawers():
    out = []
    for r, fz in enumerate(FLOORS):
        for c, y in enumerate(COLS_Y):
            out.append({
                "id": f"r{r}c{c}", "row": r, "col": c,
                "handle_closed_m": [HANDLE_X, y, fz + 0.015],
                "floor_z_m": fz,
                "interior_closed_m": {"center": [FRONT_X + 0.006 + INNER[0] / 2, y], "size": list(INNER)},
            })
    return out


@functools.lru_cache(maxsize=1)
def _cabinet_look():
    c = calib.from_dict(SIM)
    # From the table look pose, camera tipped forward at the cabinet face; roll as it comes.
    eye, target = (0.12, 0.0, 0.29), (FRONT_X, 0.0, 0.06)
    q = kinematics.solve_ik(c.chain, c.look_pose_deg, eye, tool="camera",
                            approach=[t - e for t, e in zip(target, eye)])
    return [round(a, 3) for a in q]


def scene_dict():
    d = {k: v for k, v in SIM.items() if k not in ("block", "box", "workspace_m", "fk_check")}
    d["cabinet_look_pose_deg"] = _cabinet_look()
    d["cabinet"] = {
        "box_m": {"min": [FRONT_X, -0.111, 0.0], "max": [FRONT_X + 0.105, 0.111, 0.108]},
        "pull_axis": [-1.0, 0.0, 0.0],
        "open_m": OPEN_M,
        "handle_grasp": {"approach": [0, 0, -1], "jaw_axis": [1, 0, 0], "pregrasp_m": 0.03, "close_value": 0},
        "drawers": _drawers(),
    }
    d["catalogue"] = {"colours": COLOURS, "shapes": SHAPES}
    d["handover"] = {"center_m": list(HANDOVER), "size_m": [0.08, 0.08]}
    return d


def write(path):
    path.write_text(json.dumps(scene_dict()))
    return path


def load():
    return calib.from_dict(scene_dict(), "cabinet_scene.py")


class World:
    """Drawers and objects, moved by the arm's own commands as the run logged them.

    ``objects``: id -> {colour, shape, drawer (id or None), xy (closed-drawer
    frame inside a drawer, base frame outside), z (centre), held}.
    """

    def __init__(self, scene, objects, slip=None):
        self.c = scene
        self.slip = dict(slip or {})  # object id -> grasps that close on air first
        self.fx = scene.fixture
        self.opening = {d: 0.0 for d in self.fx.drawers}
        self.objects = {k: dict(v, held=False) for k, v in objects.items()}
        self.start = json.loads(json.dumps(self.objects))
        self.grip = None  # ("handle", drawer id, the calib's HandleStage) or ("object", id, offset)
        self.angles = list(scene.look_pose_deg)
        self._seen = 0
        self.faults = []  # what the sim would have tipped or touched: see the module's doc
        self.jaw = 1.0  # how far open the pads are (0..1), as the gripper values say

    # -- where things are ------------------------------------------------------

    def pos(self, oid):
        o = self.objects[oid]
        if o["drawer"]:
            a = self.fx.pull_axis
            s = self.opening[o["drawer"]]
            return (o["xy"][0] + s * a[0], o["xy"][1] + s * a[1], o["z"])
        return (o["xy"][0], o["xy"][1], o["z"])

    def tcp(self, q=None):
        return kinematics.position(self.c.chain.pose(q or self.angles, "tcp"))

    # -- following the run -------------------------------------------------------

    def sync(self, run_dir):
        lines = (Path(run_dir) / "events.jsonl").read_text().splitlines()
        for line in lines[self._seen:]:
            e = json.loads(line)
            if e["event"] == "move":
                self.move(e["angles_deg"])
            elif e["event"] == "gripper":
                self.gripper(e["value"])
        self._seen = len(lines)

    def move(self, q):
        for step in collision.densify([self.angles, list(q)])[1:]:
            self._touch(step)
        self.angles = list(q)
        p = self.tcp()
        if self.grip and self.grip[0] == "handle":
            d, stage = self.fx.drawers[self.grip[1]], self.grip[2]  # the drawer follows the TCP along the pull
            s = stage.from_m + kinematics.dot([p[i] - stage.tcp_from[i] for i in range(3)], self.fx.pull_axis)
            self.opening[d.id] = min(max(s, 0.0), self.fx.open_m)
        elif self.grip and self.grip[0] == "object":
            o = self.objects[self.grip[1]]
            o["xy"] = [p[0] + self.grip[2][0], p[1] + self.grip[2][1]]
            o["z"] = p[2] + self.grip[2][2]

    def gripper(self, value):
        o, c = self.c.gripper_open, self.c.gripper_closed
        self.jaw = min(max((value - c) / (o - c), 0.0), 1.0) if o != c else 1.0
        p = self.tcp()
        if value < 50 and self.grip is None:
            for d in self.fx.drawers.values():
                for stage in self.fx.stages(d):  # any of the calib's grips on it
                    if math.dist(p, self.fx.stage_tcp(stage, self.opening[d.id])) < 0.006:
                        self.grip = ("handle", d.id, stage)
                        return
            for oid, o in self.objects.items():
                shape = self.fx.shapes[o["shape"]]
                q = self.pos(oid)
                bottom = q[2] - shape.height / 2
                if math.hypot(p[0] - q[0], p[1] - q[1]) < 0.008 and bottom < p[2] < bottom + shape.height:
                    if self.slip.get(oid):
                        self.slip[oid] -= 1
                        return
                    if o["drawer"]:
                        o["xy"], o["drawer"] = [q[0], q[1]], None
                    o["held"] = True
                    self.grip = ("object", oid, (q[0] - p[0], q[1] - p[1], q[2] - p[2]))
                    g = self.c.jaw_gap
                    if g and g[1] > g[0]:  # the pads stop on it
                        self.jaw = min(max((shape.width - g[0]) / (g[1] - g[0]), 0.0), 1.0)
                    return
        elif value >= 50 and self.grip is not None:
            if self.grip[0] == "object":
                self._drop(self.grip[1])
            self.grip = None

    def _drop(self, oid):
        o = self.objects[oid]
        o["held"] = False
        shape = self.fx.shapes[o["shape"]]
        x, y = o["xy"]
        bottom = o["z"] - shape.height / 2
        for d in self.fx.drawers.values():
            if self.opening[d.id] > 0.01 and self.fx.inside(d, (x, y), self.opening[d.id]) \
                    and d.floor_z - 0.002 <= bottom:
                a, s = self.fx.pull_axis, self.opening[d.id]
                o.update(drawer=d.id, xy=[x - s * a[0], y - s * a[1]], z=d.floor_z + shape.height / 2, fell=bottom - d.floor_z)
                self._overlap(oid)
                return
        o.update(z=self.c.table_z + shape.height / 2, fell=bottom - self.c.table_z)
        self._overlap(oid)

    def _overlap(self, oid):
        """A fault if ``oid``, just let go, stands on another object's footprint (catalogue widths)."""
        o, p = self.objects[oid], self.pos(oid)
        for k, v in self.objects.items():
            if k == oid or v["held"] or v["drawer"] != o["drawer"]:
                continue
            need = (self.fx.shapes[o["shape"]].width + self.fx.shapes[v["shape"]].width) / 2
            gap = math.hypot(p[0] - self.pos(k)[0], p[1] - self.pos(k)[1]) - need
            if gap < 0:
                self.faults.append(f"{oid} let go {-gap * 1000:.1f} mm into {k}'s footprint")

    def _touch(self, q):
        """A fault if at ``q`` a finger hull touches the table, an open tray, or an object standing (the
        one held, or centred between the pads, excepted). Handles are the taught motions' and sim-scene's D1:
        not checked here."""
        robot = self.c.robot
        if robot is None or not robot.hulls:
            return
        boxes = [("table", (-9.0, -9.0, -9.0), (9.0, 9.0, self.c.table_z))]
        tcp = kinematics.position(self.c.chain.pose(q, "tcp"))
        for d_id, s in self.opening.items():
            if s > 0.001:
                d = self.fx.drawers[d_id]
                # At its handle the pads grip the knob just in front of the panel; this box is coarser than
                # the sim's panel, and sim-scene's audit holds the taught grips (D1: 0 contacts).
                at_handle = math.dist(tcp, self.fx.knob(d, s)) < 0.04
                boxes += [b for b in self.fx.tray_parts(d, s) if not (at_handle and b[0].endswith(" front"))]
        held = self.grip[1] if self.grip and self.grip[0] == "object" else None
        for k, v in self.objects.items():
            # Not the one held, nor the one centred between the pads (being closed on, or just let go: the
            # brain's own guard keeps the retreat off it; this cylinder is wider than a cube's faces).
            if k == held or math.hypot(self.pos(k)[0] - tcp[0], self.pos(k)[1] - tcp[1]) < 0.008:
                continue
            p, sh = self.pos(k), self.fx.shapes[v["shape"]]  # an upright cylinder of the catalogue radius
            boxes.append((k, (p[0], p[1], p[2] - sh.height / 2), (p[0], p[1], p[2] + sh.height / 2), sh.width / 2))
        hit = robot.hull_gaps(self.c.chain, q, boxes, 1e-4, lambda name, box: name not in robot.fingers, jaw=self.jaw)
        if hit is not None and (not self.faults or not self.faults[-1].startswith(f"{hit[0]} on {hit[1]}")):
            self.faults.append(f"{hit[0]} on {hit[1]} at {[round(a, 1) for a in q]}")

    # -- what a test asserts --------------------------------------------------------

    def in_handover(self, oid):
        o = self.objects[oid]
        return not o["held"] and o["drawer"] is None and self.fx.in_handover(self.pos(oid))

    def moved(self, oid):
        a, b = self.objects[oid], self.start[oid]
        return a["drawer"] != b["drawer"] or math.dist(a["xy"], b["xy"]) > 0.003

    def all_shut(self):
        return all(s < 0.002 for s in self.opening.values())


class CabinetVLM:
    """Answers the drawer task's requests from the world, as of the run's events so far."""

    def __init__(self, world, run_dir, *, commands, order=None, open_ok=None, overrides=None):
        self.w, self.run_dir = world, Path(run_dir)
        self.commands = commands  # instruction -> parse fields
        self.order = order or sorted(world.fx.drawers)
        self.open_ok = open_ok  # drawer id -> list of answers to "is it open", for a stuck drawer
        self.overrides = overrides or {}  # schema name -> list of raw replies to send first
        self.requests = []

    def _box(self, p, half=14):
        k = self.w.c.camera
        uv = kinematics.project(self.w.c.chain, k, self.w.angles, p)
        if uv is None or not (0 <= uv[0] <= k.width and 0 <= uv[1] <= k.height):
            return None
        sx, sy = 1000 / k.width, 1000 / k.height
        return [(uv[0] - half) * sx, (uv[1] - half) * sy, (uv[0] + half) * sx, (uv[1] + half) * sy]

    def __call__(self, request):
        self.requests.append(request)
        self.w.sync(self.run_dir)
        fmt = request["response_format"]["json_schema"]
        name, props = fmt["name"], fmt["schema"]["properties"]
        if self.overrides.get(name):
            return self.overrides[name].pop(0)
        system = request["messages"][0]["content"]
        text = request["messages"][1]["content"][-1]["text"]
        if name == "parse":
            return json.dumps(self.commands[text.removeprefix("Command: ")])
        if name == "cabinet":
            opened = [d for d, s in self.w.opening.items() if s > 0.01]
            reply = {"seen": "a cabinet of drawers"}
            if "open_trays" in props:  # box each pulled-out tray where it is
                k, chain = self.w.c.camera, self.w.c.chain
                reply["open_trays"] = [{"bbox_2d": drawer_vlm.tray_boxes(self.w.fx, chain, k, self.w.angles,
                                                                         self.w.opening[d])[d]} for d in opened]
            else:
                reply["open_drawers"] = opened
            if "next" in props:
                enum = props["next"]["enum"]
                reply["next"] = next((d for d in self.order if d in enum), "none")
            return json.dumps(reply)
        if name == "drawer":
            d = re.search(r"into drawer (r\dc\d)", system).group(1)
            is_open = self.w.opening[d] > 0.5 * self.w.fx.open_m
            if self.open_ok and self.open_ok.get(d):
                is_open = self.open_ok[d].pop(0)
            objects = []
            for oid, o in self.w.objects.items():
                if o["drawer"] == d and is_open:
                    box = self._box(self.w.pos(oid))
                    if box:
                        objects.append({"colour": o["colour"], "shape": o["shape"], "bbox_2d": box})
            reply = {"seen": f"{len(objects)} objects", "open": is_open, "objects": objects}
            if "target" in props:
                want = text.removeprefix("Looking for: ")
                reply["target"] = next((i for i, o in enumerate(objects)
                                        if want == "any object" or want in (f"{o['colour']} {o['shape']}",
                                                                            o["colour"] + " object", o["shape"])), -1)
            return json.dumps(reply)
        if name == "locate":
            want = re.search(r"pick = the ([a-z ]+?)\.", system).group(1)
            for oid, o in self.w.objects.items():
                if want in (f"{o['colour']} {o['shape']}",) and not o["held"]:
                    box = self._box(self.w.pos(oid))
                    if box:
                        return json.dumps({"seen": "it", "pick": {"label": want, "visible": True, "bbox_2d": box}})
            return json.dumps({"seen": "not here", "pick": {"label": want, "visible": False, "bbox_2d": [0, 0, 0, 0]}})
        if name == "check_held":
            return json.dumps({"seen": "gripper", "ok": bool(self.w.grip and self.w.grip[0] == "object")})
        if name == "check_handover":
            ok = any(self.w.in_handover(oid) for oid in self.w.objects)
            return json.dumps({"seen": "handover zone", "ok": ok})
        if name == "check_put":
            d = re.search(r"drawer (r\dc\d)", text).group(1)
            ok = any(o["drawer"] == d and self.w.moved(oid) for oid, o in self.w.objects.items())
            return json.dumps({"seen": "drawer", "ok": ok})
        raise AssertionError(f"unexpected request {name}")


# -- sim-scene's cabinet as /home/hkim/mycobot-sim/sim_calib_drawers.json has it (v2, 2026-11-01 19:07),
#    in its own key layout: knob pinched top and bottom at a heading per column (no cabinet-wide one),
#    the TCP target 3 mm short of the front, a 100 mm pull, a full-height front over 16 mm tray walls,
#    a preshape value instead of fully open, the pick height and jaw from its "pick" section.
SIM_FRONT_X = 0.245
SIM_COL_Y = (0.081, 0.0, -0.081)
SIM_BAY_Z = (0.018, 0.058, 0.098)  # bay bottoms
SIM_YAW = {0: -20.0, 1: 40.0, 2: 20.0}  # handle heading per column
SIM_TCP_X = {0: 0.2226, 1: 0.222, 2: 0.2226}
SIM_OPEN_M = 0.1
SIM_CABINET_LOOK = [-7.9, 19.8, -5.0, -66.0, 25.9, -132.1]
SIM_TABLE_LOOK = [81.1, -38.0, -42.9, -1.3, 1.3, -53.8]  # straight down over the handover mat
SIM_LOOKS = {  # each drawer's tilted look, opened; r1c2 and r2c2 hold the TCP lower than wall top + 2 cm
    "r0c0": [95.5, -56.2, -22.8, -51.1, -41.9, 12.2], "r0c1": [-69.1, -12.5, -117.6, 25.3, 46.3, -8.5],
    "r0c2": [-79.3, -40.9, -59.4, -22.2, 37.7, 32.2], "r1c0": [117.9, -23.3, -43.7, -64.9, -40.4, 14.8],
    "r1c1": [-71.9, 2.5, -104.2, -10.2, 42.7, 14.4], "r1c2": [-73.6, -27.9, -69.8, -20.9, 40.6, 38.6],
    "r2c0": [112.4, -20.5, -35.0, -72.6, -43.9, 9.1], "r2c1": [-71.9, 2.4, -80.1, -34.2, 42.7, 14.4],
    "r2c2": [-73.6, -29.6, -43.1, -45.5, 40.7, 38.8],
}
SIM_TOPDOWN = {  # camera straight down over each opened drawer's floor
    "r0c0": [32.3, -30.5, -18.7, -41.2, -8.1, 47.3], "r0c1": [10.0, -7.0, -71.1, -13.3, -7.9, 55.1],
    "r0c2": [-13.9, -22.3, -72.1, 6.3, -7.8, 31.0], "r1c0": [33.8, -11.3, -57.5, -25.7, -6.7, 79.2],
    "r1c1": [10.0, -8.5, -63.2, -19.8, -7.9, 55.1], "r1c2": [-13.9, -26.2, -41.8, -20.2, -7.8, 31.0],
    "r2c0": [43.8, -25.3, -0.0, -72.3, 1.9, 148.7], "r2c1": [12.5, -19.1, -14.8, -61.4, -5.9, 87.8],
    "r2c2": [-0.1, -21.4, -8.6, -66.9, 4.0, 164.7],
}
SIM_SHAPES = {
    "cube": {"height_m": 0.025, "width_m": 0.025, "grasp_height_m": 0.0205, "diagonal_m": 0.0354},
    "cylinder": {"height_m": 0.025, "width_m": 0.026, "grasp_height_m": 0.0205},
    "sphere": {"height_m": 0.026, "width_m": 0.026, "grasp_height_m": 0.0205},
    "cone": {"height_m": 0.026, "width_m": 0.03, "grasp_height_m": 0.019},
}


def sim_like_dict():
    d = scene_dict()
    drawers = []
    for r, z0 in enumerate(SIM_BAY_Z):
        for c, y in enumerate(SIM_COL_Y):
            a = math.radians(SIM_YAW[c])
            approach = [round(math.cos(a), 6), round(math.sin(a), 6), 0.0]
            drawers.append({
                "id": f"r{r}c{c}", "row": r, "col": c,
                "front_panel_z_m": [z0 + 0.0005, z0 + 0.0355],
                "handle_center_closed_m": [0.236, y, z0 + 0.018],
                "handle_grasp": {"tool_yaw_deg": SIM_YAW[c], "tcp_closed_m": [SIM_TCP_X[c], y, z0 + 0.018]},
                "handle_closed_m": [SIM_TCP_X[c], y, z0 + 0.018],
                "approach": approach, "jaw_axis": [0.0, 0.0, 1.0],
                "floor_z_m": z0 + 0.005,
                "interior_closed_m": {"center": [0.289, y], "size": [0.072, 0.072, 0.031], "wall_height_m": 0.016},
                "look_pose_deg": SIM_LOOKS[f"r{r}c{c}"],
                "topdown_pose_deg": SIM_TOPDOWN[f"r{r}c{c}"],
            })
    d["cabinet"] = {
        "front_x_m": SIM_FRONT_X,
        "box_m": {"min": [SIM_FRONT_X, -0.1245, 0.0], "max": [0.335, 0.1245, 0.14]},
        "pull_axis": [-1.0, 0.0, 0.0],
        "open_m": SIM_OPEN_M,
        "interior_m": {"width": 0.072, "depth": 0.072, "wall_height": 0.016},
        "handle_grasp": {"approach": None, "jaw_axis": [0.0, 0.0, 1.0], "pregrasp_m": 0.04, "far_m": 0.08,
                         "preshape_value": 55, "close_value": 0, "release_value": 55},
        "handle_m": {"protrusion": 0.018, "width": 0.032, "height": 0.024},
        "drawer_outer_m": {"width": 0.078, "depth": 0.084, "front_height": 0.035},
        "drawers": drawers,
    }
    d["pick"] = {"jaw_yaw_deg": 180.0, "approach_tcp_height_above_support_m": 0.057}
    d["catalogue"] = {"colours": COLOURS, "shapes": SIM_SHAPES}
    d["handover"] = {"center_m": [0.10, 0.19], "size_m": [0.09, 0.09]}
    d["tcp"] = dict(d["tcp"], pad_extent_along_tool_z_m=[-0.0146, 0.0165])
    d["cabinet_look_pose_deg"] = SIM_CABINET_LOOK
    d["look_pose_deg"] = SIM_TABLE_LOOK
    return d


def load_sim_like():
    return calib.from_dict(sim_like_dict(), "cabinet_scene.py (sim-scene numbers)")


#: Made-up capsules in calib v3's ``robot_collision`` format (BOARD 2026-11-02), shaped like the URDF
#: hulls' boxes: the base, the upper arm and forearm, the gripper body, the camera, the two pads. Not
#: sim-scene's numbers; those replace these when v3 lands.
SYNTH_CAPSULES = {
    "capsules": [
        {"name": "joint1", "frame": "joint1", "a_m": [0, 0, 0.02], "b_m": [0, 0, 0.08], "radius_m": 0.06},
        {"name": "joint3", "frame": "joint3", "a_m": [-0.11, 0, 0.058], "b_m": [0, 0, 0.058], "radius_m": 0.035},
        {"name": "joint4", "frame": "joint4", "a_m": [-0.094, 0, 0.006], "b_m": [0, 0, 0.006], "radius_m": 0.033},
        {"name": "gripper_base", "frame": "joint6_flange", "a_m": [0, 0, 0.03], "b_m": [0, 0, 0.045],
         "radius_m": 0.035},
        {"name": "camera_flange", "frame": "joint6_flange", "a_m": [-0.035, 0.035, 0.01], "b_m": [0, 0, 0.01],
         "radius_m": 0.02},
        {"name": "gripper_left3", "frame": "joint6_flange", "a_m": [-0.013, -0.013, 0.088],
         "b_m": [-0.013, -0.013, 0.11], "radius_m": 0.01},
        {"name": "gripper_right3", "frame": "joint6_flange", "a_m": [0.013, 0.013, 0.088],
         "b_m": [0.013, 0.013, 0.11], "radius_m": 0.01},
    ],
    "pairs": [["joint1", "joint4"], ["joint1", "gripper_base"], ["joint1", "camera_flange"],
              ["joint1", "gripper_left3"], ["joint1", "gripper_right3"], ["joint3", "gripper_base"],
              ["joint3", "camera_flange"]],
}


def load_sim_like_with_capsules(capsules=None):
    d = sim_like_dict()
    d["robot_collision"] = capsules or SYNTH_CAPSULES
    return calib.from_dict(d, "cabinet_scene.py (sim-scene numbers, made-up capsules)")


def with_two_stage_handles(d, ids=None):
    """sim-scene's v3 handle recipe (BOARD 2026-11-02 11:32), in the shape the brain reads: a side pinch
    from shut to 50 mm, then a top-down pinch of the knob's left/right faces to fully open. The numbers
    are made up on the v2 geometry; v3's own replace them."""
    for row in d["cabinet"]["drawers"]:
        if ids is not None and row["id"] not in ids:
            continue
        knob = row["handle_center_closed_m"]
        row["handle_stages"] = [
            {"from_m": 0.0, "to_m": 0.05, "tcp_closed_m": row["handle_closed_m"], "approach": row["approach"],
             "jaw_axis": [0.0, 0.0, 1.0], "standoff_m": 0.05},
            {"from_m": 0.05, "to_m": d["cabinet"]["open_m"], "tcp_closed_m": [knob[0], knob[1], knob[2] + 0.004],
             "approach": [0.0, 0.0, -1.0], "jaw_axis": [0.0, 1.0, 0.0], "standoff_m": 0.03},
        ]
    return d


def load_sim_like_two_stage():
    return calib.from_dict(with_two_stage_handles(sim_like_dict()), "cabinet_scene.py (two-stage handles)")


def with_side_top_handles(d, tilt_deg=10.0):
    """The same recipe as sim-scene's calib writer lays it out (write_sim_calib_drawers.py): per drawer
    handle_grasp {stage_m, side, top}, the top grip leaning ``tilt_deg`` towards the robot at full open
    (about the jaw axis, y). Made-up numbers on the v2 geometry."""
    t = math.radians(tilt_deg)
    open_m = d["cabinet"]["open_m"]
    for row in d["cabinet"]["drawers"]:
        knob = row["handle_center_closed_m"]
        top = [knob[0], knob[1], knob[2] + 0.004]
        row["handle_grasp"] = {
            "stage_m": 0.05,
            "side": {"approach": row["approach"], "jaw_axis": [0.0, 0.0, 1.0], "tcp_closed_m": row["handle_closed_m"],
                     "tcp_stage_m": [row["handle_closed_m"][0] - 0.05, *row["handle_closed_m"][1:]]},
            "top": {"jaw_axis": [0.0, 1.0, 0.0], "approach_at_stage": [0.0, 0.0, -1.0],
                    "approach_at_open": [-math.sin(t), 0.0, -math.cos(t)],
                    "tcp_stage_m": [top[0] - 0.05, top[1], top[2]], "tcp_open_m": [top[0] - open_m, top[1], top[2]]},
        }
    d["cabinet"]["handle_grasp"].update(
        side={"preshape_value": 55, "close_value": 0, "release_value": 55, "far_closed_m": 0.05, "backoff_m": 0.03},
        top={"open_value": 100, "close_value": 0, "release_value": 100, "lift_m": 0.03})
    return d


def load_sim_like_side_top():
    return calib.from_dict(with_side_top_handles(sim_like_dict()), "cabinet_scene.py (sim-scene's side/top layout)")


def _corners(lo, hi):
    return [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


#: Fake hulls in calib v3.1's ``robot_collision.hulls`` format: each collider's URDF hull bounding box as
#: its 8 corners (frames as the capsules). sim-scene's own hulls replace these when v3.1 lands.
SYNTH_HULLS = [
    {"name": "joint1", "frame": "joint1", "vertices_m": _corners((-0.055, -0.075, 0.0), (0.055, 0.075, 0.1097))},
    {"name": "joint3", "frame": "joint3", "vertices_m": _corners((-0.132, -0.024, 0.0298), (0.024, 0.024, 0.0853))},
    {"name": "joint4", "frame": "joint4", "vertices_m": _corners((-0.1152, -0.0216, -0.0224), (0.0216, 0.0216, 0.0354))},
    {"name": "gripper_base", "frame": "joint6_flange",
     "vertices_m": _corners((-0.0279, -0.0443, 0.0186), (0.0334, 0.0279, 0.0809))},
    {"name": "camera_flange", "frame": "joint6_flange",
     "vertices_m": _corners((-0.0511, -0.0192, 0.0019), (0.0192, 0.0511, 0.0181))},
    {"name": "gripper_left3", "frame": "joint6_flange",
     "vertices_m": _corners((-0.0279, -0.0285, 0.0863), (-0.008, -0.0085, 0.1125))},
    {"name": "gripper_right3", "frame": "joint6_flange",
     "vertices_m": _corners((0.008, 0.0085, 0.0863), (0.0279, 0.0285, 0.1125))},
]


def load_sim_like_with_hulls():
    return load_sim_like_with_capsules(dict(SYNTH_CAPSULES, hulls=SYNTH_HULLS))
