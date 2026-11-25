"""COLLADA -> STL in metres, for cuRobo.

The vendor ``.dae`` files disagree on units: ``joint2``-``joint4``, ``joint6``,
``joint7`` declare ``<unit meter="1"/>``, while ``joint1_jet``, ``joint5`` and
every gripper mesh declare ``meter="0.001"``. ROS tools honour that tag;
trimesh (which cuRobo uses) ignores it, so the base and gripper come out 1000x
too large. ``joint5.dae`` also uses ``<polygons>``, which trimesh reads as an
empty mesh, and its ``<bind_material>`` names a material that does not exist,
which makes pycollada drop the scene node - and with it the node's rotation.
So ``<bind_material>`` is stripped (colour only), every mesh is read with
pycollada *through its scene graph* (node transforms applied), triangulated,
scaled by its own ``<unit>`` and written as STL.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import collada
import numpy as np
import trimesh


def dae_to_trimesh(path: Path) -> trimesh.Trimesh:
    text = re.sub(r"<bind_material>.*?</bind_material>", "", path.read_text(), flags=re.S)
    doc = collada.Collada(io.BytesIO(text.encode()), ignore=[collada.common.DaeUnsupportedError])
    scale = float(doc.assetInfo.unitmeter or 1.0)
    parts = []
    for geom in doc.scene.objects("geometry"):
        for prim in geom.primitives():
            if isinstance(prim, (collada.polylist.BoundPolylist, collada.polygons.BoundPolygons)):
                prim = prim.triangleset()
            if not isinstance(prim, collada.triangleset.BoundTriangleSet) or len(prim.vertex_index) == 0:
                continue
            parts.append(trimesh.Trimesh(vertices=np.asarray(prim.vertex) * scale,
                                         faces=np.asarray(prim.vertex_index), process=False))
    if not parts:
        raise ValueError(f"{path}: no triangles")
    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    return mesh


def convert(src: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (src.stem + ".stl")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        dae_to_trimesh(src).export(dst)
    return dst


def convex_hull(stl: Path) -> Path:
    """Watertight stand-in for sphere fitting.

    Most vendor meshes are open shells (``is_watertight`` False), and MorphIt's
    interior sampling collapses on those: ``gripper_base`` came out at 0.6 %
    coverage. The arm links are close to convex, so a hull costs a few mm of
    over-approximation and buys a fit that actually covers the part.
    """
    dst = stl.with_name(stl.stem + "_hull.stl")
    if not dst.exists() or dst.stat().st_mtime < stl.stat().st_mtime:
        trimesh.load(stl, force="mesh").convex_hull.export(dst)
    return dst
