"""A pick-and-place scene on the default 280 JN chain, and a fake VLM that sees it.

The fake projects the true block and box centres through the camera pose of
the angles it is handed, so a run against it lands exactly where the
geometry says, and any error is the code's.
"""

import json
import math

from mycobot_brain import calib, kinematics

# Elephant's mycobot_280_jn URDF joint origins, as kinematics.MYCOBOT_280_JN.
JOINTS = [
    {"xyz": [0, 0, 0.13156], "rpy": [0, 0, 0]},
    {"xyz": [0, 0, 0], "rpy": [0, math.pi / 2, -math.pi / 2]},
    {"xyz": [-0.1104, 0, 0], "rpy": [0, 0, 0]},
    {"xyz": [-0.096, 0, 0.06462], "rpy": [0, 0, -math.pi / 2]},
    {"xyz": [0, -0.07318, 0], "rpy": [math.pi / 2, -math.pi / 2, 0]},
    {"xyz": [0, 0.0456, 0], "rpy": [-math.pi / 2, 0, 0]},
]

SCENE = {
    "chain": {
        "joints": JOINTS,
        "flange_to_camera": {"xyz": [0, 0.035, 0.03], "rpy": [0, 0, 0]},
        "flange_to_tcp": {"xyz": [0, 0, 0.10], "rpy": [0, 0, 0]},
    },
    "camera": {"K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]], "width": 640, "height": 480},
    "table_z_m": 0.0,
    "block_edge_m": 0.025,
    "box": {"size_m": [0.08, 0.08, 0.05], "rim_height_m": 0.05},
    # Camera 21 cm over the table at (0.17, -0.03), looking down: both objects in view.
    "look_pose_deg": [12.0, -21.3, -36.7, -32.0, 0.0, 0.0],
    "gripper": {"open": 80, "closed": 15},
}
BLOCK = (0.21, -0.10, 0.0125)  # centre; table + half the edge
BOX = (0.15, 0.03, 0.05)  # opening centre at rim height


def write(path):
    path.write_text(json.dumps(SCENE))
    return path


def load():
    return calib.from_dict(SCENE, "scene.py")


class SceneVLM:
    """Plays vLLM for a pick-place run: answers every locate and check request.

    ``angles()`` says where the arm is when the request is made. ``bias``
    shifts the first look's answers by (du, dv) pixels, as a VLM that is a
    little off from far away. ``held`` / ``final`` are the check answers in
    order (the last one repeats).
    """

    def __init__(self, scene, angles, *, coords="norm1000", shape="bbox", bias=(0, 0), next="pick",
                 held=(True,), final=(True,), block=BLOCK, box=BOX):
        self.scene, self.angles, self.coords, self.shape = scene, angles, coords, shape
        self.bias, self.next = bias, next
        self.held, self.final = list(held), list(final)
        self.truth = {"pick": block, "place": box}
        self.requests = []
        self.locates = 0

    def _where(self, role, q, shift):
        k = self.scene.camera
        u, v = kinematics.project(self.scene.chain, k, q, self.truth[role])
        u, v = u + shift[0], v + shift[1]
        if not (0 <= u <= k.width and 0 <= v <= k.height):
            return {"label": role, "visible": False, "bbox_2d" if self.shape == "bbox" else "point_2d": [0] * 4}
        sx, sy = (1000 / k.width, 1000 / k.height) if self.coords == "norm1000" else (1, 1)
        if self.shape == "bbox":
            return {"label": {"pick": "red block", "place": "brown box"}[role], "visible": True,
                    "bbox_2d": [(u - 12) * sx, (v - 12) * sy, (u + 12) * sx, (v + 12) * sy]}
        return {"label": {"pick": "red block", "place": "brown box"}[role], "visible": True,
                "point_2d": [u * sx, v * sy]}

    def __call__(self, request):
        self.requests.append(request)
        name = request["response_format"]["json_schema"]["name"]
        props = request["response_format"]["json_schema"]["schema"]["properties"]
        q = self.angles()
        if name == "locate":
            self.locates += 1
            shift = self.bias if self.locates == 1 else (0, 0)
            reply = {"seen": "a red block and a brown box on a table"}
            for role in ("pick", "place"):
                if role in props:
                    reply[role] = self._where(role, q, shift)
            if "next" in props:
                reply["next"] = self.next
            return json.dumps(reply)
        question = request["messages"][1]["content"][-1]["text"]
        answers = self.held if "held" in question else self.final
        ok = answers.pop(0) if len(answers) > 1 else answers[0]
        return json.dumps({"seen": "the gripper camera view", "ok": ok})
