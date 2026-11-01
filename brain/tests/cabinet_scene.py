"""A 3x3 drawer cabinet on the sim arm's chain, a world that follows the arm,
and a fake VLM that answers from that world.

The world replays the run's ``move`` / ``gripper`` events: a gripper that
closes on a handle drags the drawer along the pull axis as the TCP moves; one
that closes on an object carries it; opening drops it on whatever is under it
(an open drawer's floor, the handover zone, the table). So a test asserts what
happened to the drawers and objects, not only what was sent.
"""

import functools
import json
import math
import re
from pathlib import Path

from mycobot_brain import calib, drawer_vlm, kinematics

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
        self.grip = None  # ("handle", drawer id, tcp - handle) or ("object", id, offset)
        self.angles = list(scene.look_pose_deg)
        self._seen = 0

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
        self.angles = list(q)
        p = self.tcp()
        if self.grip and self.grip[0] == "handle":
            d = self.fx.drawers[self.grip[1]]
            s = kinematics.dot([p[i] - d.handle_closed[i] for i in range(3)], self.fx.pull_axis)
            self.opening[d.id] = min(max(s, 0.0), self.fx.open_m)
        elif self.grip and self.grip[0] == "object":
            o = self.objects[self.grip[1]]
            o["xy"] = [p[0] + self.grip[2][0], p[1] + self.grip[2][1]]
            o["z"] = p[2] + self.grip[2][2]

    def gripper(self, value):
        p = self.tcp()
        if value < 50 and self.grip is None:
            for d in self.fx.drawers.values():
                if math.dist(p, self.fx.handle(d, self.opening[d.id])) < 0.006:
                    self.grip = ("handle", d.id)
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
                return
        o.update(z=self.c.table_z + shape.height / 2, fell=bottom - self.c.table_z)

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
            })
    d["cabinet"] = {
        "front_x_m": SIM_FRONT_X,
        "box_m": {"min": [SIM_FRONT_X, -0.1245, 0.0], "max": [0.335, 0.1245, 0.14]},
        "pull_axis": [-1.0, 0.0, 0.0],
        "open_m": SIM_OPEN_M,
        "interior_m": {"width": 0.072, "depth": 0.072, "wall_height": 0.016},
        "handle_grasp": {"approach": None, "jaw_axis": [0.0, 0.0, 1.0], "pregrasp_m": 0.04, "far_m": 0.08,
                         "preshape_value": 55, "close_value": 0, "release_value": 55},
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
