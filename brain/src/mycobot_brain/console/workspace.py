"""Lab geometry for the console: cameras, arm bases, tools, and the safe box.

Everything is in one *world* frame (mm): the .77 arm's base frame, because the
top camera was calibrated against that arm (Kabsch on the gripper-plate hole
pattern, 1-3 mm). The .70 arm's base pose in that frame comes from a J1 sweep
seen by the D405 (about +-15 mm, z rough) - TODO(unverified) beyond that.

Calibration numbers live in a JSON file (``--workspace``), not in code: the
cameras get bumped and re-calibrated, and the numbers change with them.
numpy and Pillow are needed here (the ``console`` extra); the rest of the brain
stays stdlib-only.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Camera:
    name: str
    label: str
    color_mjpg: str
    depth_mjpg: str
    color_jpg: str
    depth_png: str = ""
    K: dict | None = None  # colour intrinsics fx fy ppx ppy
    R: np.ndarray | None = None  # world -> colour camera
    t: np.ndarray | None = None
    depth_K: dict | None = None
    R_dc: np.ndarray | None = None  # depth -> colour (mm)
    t_dc: np.ndarray | None = None
    size: tuple[int, int] = (640, 480)
    depth_set: str = ""  # viewer URL that sets the depth colour range

    @property
    def calibrated(self) -> bool:
        return self.K is not None and self.R is not None and self.t is not None

    def pixel_ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """World-frame camera centre and unit ray through colour pixel (u, v)."""
        K = self.K
        d_cam = np.array([(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0])
        centre = -self.R.T @ self.t
        d = self.R.T @ d_cam
        return centre, d / np.linalg.norm(d)

    def world_to_pixel(self, p: np.ndarray) -> np.ndarray:
        c = self.R @ np.asarray(p, float) + self.t
        return np.array([self.K["fx"] * c[0] / c[2] + self.K["ppx"], self.K["fy"] * c[1] / c[2] + self.K["ppy"]])

    def aligned_depth(self, depth_png: bytes) -> np.ndarray:
        """Z16 depth (depth sensor pixels) -> colour-pixel depth in mm (0 = none).

        The viewer's depth is not registered to colour (wider field of view, 15 mm
        baseline), so it is re-projected with the factory intrinsics/extrinsics.
        """
        from PIL import Image

        d = np.asarray(Image.open(io.BytesIO(depth_png))).astype(np.float64)
        v, u = np.nonzero(d > 0)
        z = d[v, u]
        D = self.depth_K
        X = np.stack([(u - D["ppx"]) / D["fx"] * z, (v - D["ppy"]) / D["fy"] * z, z], 1)
        Xc = X @ self.R_dc.T + self.t_dc
        K = self.K
        uc = np.round(Xc[:, 0] / Xc[:, 2] * K["fx"] + K["ppx"]).astype(int)
        vc = np.round(Xc[:, 1] / Xc[:, 2] * K["fy"] + K["ppy"]).astype(int)
        w, h = self.size
        ok = (uc >= 0) & (uc < w) & (vc >= 0) & (vc < h)
        out = np.full((h, w), np.inf)
        np.minimum.at(out, (vc[ok], uc[ok]), Xc[ok, 2])
        out[np.isinf(out)] = 0
        return out

    def pixel_to_world(self, u: float, v: float, depth: np.ndarray | None, r: int = 4) -> tuple[np.ndarray | None, float]:
        """3-D point under colour pixel (u, v) from aligned depth (median of a
        (2r+1)^2 window). Returns (point or None, fraction of valid depth)."""
        if depth is None:
            return None, 0.0
        h, w = depth.shape
        ui, vi = int(round(u)), int(round(v))
        win = depth[max(0, vi - r): min(h, vi + r + 1), max(0, ui - r): min(w, ui + r + 1)].ravel()
        good = win[win > 0]
        if good.size == 0:
            return None, 0.0
        z = float(np.median(good))
        K = self.K
        p_cam = np.array([(u - K["ppx"]) / K["fx"] * z, (v - K["ppy"]) / K["fy"] * z, z])
        return self.R.T @ (p_cam - self.t), good.size / max(1, win.size)


@dataclass
class Arm:
    name: str
    port: int
    R: np.ndarray  # arm base -> world rotation
    t: np.ndarray  # arm base origin in world (mm)
    tool: np.ndarray  # tool point in the flange frame (mm)
    home: list[float] | None  # None: return to the pose the console first saw
    label: str = ""

    def to_base(self, p_world: np.ndarray) -> np.ndarray:
        return self.R.T @ (np.asarray(p_world, float) - self.t)

    def dir_to_base(self, d_world: np.ndarray) -> np.ndarray:
        return self.R.T @ np.asarray(d_world, float)

    def to_world(self, p_base: np.ndarray) -> np.ndarray:
        return self.R @ np.asarray(p_base, float) + self.t


@dataclass
class Box:
    """Where the tool point may go, in world mm. The arm server still checks joints."""

    lo: np.ndarray
    hi: np.ndarray
    reach_mm: float  # from each arm's J2 axis

    def clamp(self, p: np.ndarray) -> tuple[np.ndarray, bool]:
        q = np.minimum(np.maximum(p, self.lo), self.hi)
        return q, bool(np.any(np.abs(q - p) > 1e-9))


@dataclass
class Workspace:
    cameras: dict[str, Camera]
    arms: dict[str, Arm]
    box: Box
    table_z: float
    vlm_camera: str
    curobo_api: str
    curobo_view: str
    obstacles: list[dict]

    def obstacles_for(self, arm: "Arm", extra: list[dict] | None = None) -> list[dict]:
        """Static cuboids (plus ``extra`` world-frame ones) in ``arm``'s base frame, cuRobo format."""
        out = []
        for o in list(self.obstacles) + list(extra or []):
            if o.get("skip_for") == arm.name:
                continue
            c = arm.to_base(np.array(o["center_mm"], float))
            yaw = math.radians(o.get("yaw_deg", 0.0)) - math.atan2(arm.R[1, 0], arm.R[0, 0])
            out.append({"name": o["name"], "dims_mm": list(map(float, o["dims_mm"])),
                        "pose": [*map(float, c), math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]})
        return out

    def world_axes_text(self) -> str:
        """How the world axes look in the VLM camera, so the model can say "left"."""
        cam = self.cameras.get(self.vlm_camera)
        if cam is None or not cam.calibrated:
            return "World axes: +z is up."
        names = []
        for axis, label in ((0, "+x"), (1, "+y")):
            d = cam.R[:, axis]  # image-plane direction of the world axis
            ang = math.degrees(math.atan2(d[1], d[0]))  # image y is down
            names.append(f"{label} points toward image {_compass(ang)}")
        return "World axes in the TOP image: " + ", ".join(names) + ", +z is up (away from the table)."


def _compass(ang: float) -> str:
    dirs = ["right", "lower-right", "down", "lower-left", "left", "upper-left", "up", "upper-right"]
    return dirs[int(((ang % 360) + 22.5) // 45) % 8]


def _mat(v: Any) -> np.ndarray | None:
    return None if v is None else np.array(v, float)


def load(path: str | Path) -> Workspace:
    cfg = json.loads(Path(path).read_text())
    cams = {}
    for name, c in cfg["cameras"].items():
        cams[name] = Camera(
            name=name,
            label=c.get("label", name),
            color_mjpg=c["color_mjpg"],
            depth_mjpg=c["depth_mjpg"],
            color_jpg=c["color_jpg"],
            depth_png=c.get("depth_png", ""),
            K=c.get("K"),
            R=_mat(c.get("R")),
            t=_mat(c.get("t")),
            depth_K=c.get("depth_K"),
            R_dc=_mat(c.get("R_dc")),
            t_dc=_mat(c.get("t_dc")),
            depth_set=c.get("depth_set", ""),
        )
    arms = {}
    for name, a in cfg["arms"].items():
        yaw = math.radians(a.get("yaw_deg", 0.0))
        R = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        arms[name] = Arm(
            name=name,
            port=int(a["port"]),
            R=R,
            t=np.array(a.get("origin_mm", [0, 0, 0]), float),
            tool=np.array(a.get("tool_mm", [0, 0, 0]), float),
            home=[float(x) for x in a["home_deg"]] if a.get("home_deg") else None,
            label=a.get("label", name),
        )
    b = cfg["box"]
    return Workspace(
        cameras=cams,
        arms=arms,
        box=Box(np.array(b["lo_mm"], float), np.array(b["hi_mm"], float), float(b.get("reach_mm", 270.0))),
        table_z=float(cfg["table_z_mm"]),
        vlm_camera=cfg.get("vlm_camera", "top"),
        curobo_api=cfg.get("curobo_api", "http://127.0.0.1:8096"),
        curobo_view=cfg.get("curobo_view", "http://127.0.0.1:8080"),
        obstacles=list(cfg.get("obstacles", [])),
    )


def heightmap_boxes(cam: Camera, depth: np.ndarray, table_z: float, box: Box, avoid: list[np.ndarray],
                    avoid_r: float | list[float] = 100.0, cell: float = 20.0, tile: float = 60.0,
                    reach_from: np.ndarray | None = None, reach_mm: float = 340.0, max_boxes: int = 12,
                    stride: int = 3, spheres: np.ndarray | None = None, sphere_margin: float = 25.0) -> list[dict]:
    """Things standing on the table, as world-frame cuboids for cuRobo.

    Back-projects the top camera's aligned depth and keeps points 12-250 mm above
    the table, within ``reach_mm`` (xy) of ``reach_from`` (the arm's base). Points
    within ``avoid_r`` of the polyline ``avoid`` (the arm's own DH chain; one radius
    per segment allowed) are the arm itself and are dropped. The rest is binned on a
    ``cell`` mm grid; connected blobs become boxes, and a blob wider than ``tile``
    is cut into ``tile``-sized boxes so one long blob cannot wall off free table
    (2026-11-24: a cup, a lamp stand and clamps merged into one 190 mm tall box
    across the board). Each box is its cells' extent + 10 mm, from the table to
    the highest point + 10 mm.
    TODO(unverified): transparent things (the vials' plastic) give little depth;
    a blob may then be shorter than the object, or missing.
    """
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[vv, uu]
    ok = z > 0
    u, v, z = uu[ok].astype(float), vv[ok].astype(float), z[ok]
    K = cam.K
    pc = np.stack([(u - K["ppx"]) / K["fx"] * z, (v - K["ppy"]) / K["fy"] * z, z], 1)
    pw = (pc - cam.t) @ cam.R  # R^T (p - t) row-wise
    keep = (pw[:, 2] > table_z + 12) & (pw[:, 2] < table_z + 250)
    if reach_from is not None:
        keep &= np.hypot(pw[:, 0] - reach_from[0], pw[:, 1] - reach_from[1]) < reach_mm
    pw = pw[keep]
    if spheres is not None and len(spheres) and len(pw):
        # the arm's own collision spheres (world mm, x y z r): the precise way to cut the arm out
        S = np.asarray(spheres, float)
        near = np.zeros(len(pw), bool)
        for c0 in range(0, len(S), 32):
            blk = S[c0:c0 + 32]
            d = np.linalg.norm(pw[:, None, :] - blk[None, :, :3], axis=2) - blk[None, :, 3]
            near |= (d < sphere_margin).any(1)
        pw = pw[~near]
    elif len(avoid) > 1 and len(pw):
        A = np.asarray(avoid, float)
        radii = np.broadcast_to(np.asarray(avoid_r, float), (len(A) - 1,))
        far = np.ones(len(pw), bool)
        for (a, b), r in zip(zip(A[:-1], A[1:]), radii):
            ab = b - a
            t = np.clip(((pw - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
            far &= np.linalg.norm(pw - (a + t[:, None] * ab), axis=1) > r
        pw = pw[far]
    if not len(pw):
        return []
    ij = np.floor(pw[:, :2] / cell).astype(int)
    top: dict[tuple[int, int], float] = {}
    count: dict[tuple[int, int], int] = {}
    for k, zz in zip(map(tuple, ij), pw[:, 2]):
        top[k] = max(top.get(k, -1e9), float(zz))
        count[k] = count.get(k, 0) + 1
    occ = {k for k, n in count.items() if n >= 2}  # one stray pixel is not an object
    groups: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    per_tile = max(1, int(round(tile / cell)))
    for k in occ:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.append(c)
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if n in occ and n not in seen:
                    seen.add(n)
                    stack.append(n)
        if len(comp) < 2:
            continue
        c = np.array(comp)
        if np.ptp(c[:, 0]) < per_tile and np.ptp(c[:, 1]) < per_tile:
            groups.append(comp)
            continue
        tiles: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cc in comp:
            tiles.setdefault((cc[0] // per_tile, cc[1] // per_tile), []).append(cc)
        groups += list(tiles.values())
    out = []
    ref = np.zeros(2) if reach_from is None else np.asarray(reach_from[:2], float)
    def dist(g):
        c = np.array(g, float).mean(0) * cell
        return float(np.hypot(*(c - ref)))
    for g in sorted(groups, key=dist)[:max_boxes]:
        c = np.array(g, float)
        lo = c.min(0) * cell - 10
        hi = (c.max(0) + 1) * cell + 10
        zt = max(top[k] for k in g) + 10
        out.append({"name": f"obj{len(out)}", "center_mm": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2),
                                                              float((table_z + zt) / 2)],
                    "dims_mm": [float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(zt - table_z)], "yaw_deg": 0.0})
    return out
