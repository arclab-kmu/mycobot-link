"""The .77's kinematics in the table-referenced planner frame.

Everything the .77 planner knows about heights was taught with the table top 197 mm above the .77 base: the
templates, TABLE, the cap / rim heights, the hover and pour heights, the camera world (locate). On 2026-11-26 the
exhibit setup moved rooms and the .77 may stand at another height under the table. Instead of editing every height,
the planner keeps its frame glued to the table: this module's fk() reports z - BASE_DZ, where
BASE_DZ = (table top in the real .77 base frame) - 197, from MYCOBOT_TABLE_Z or $MYCOBOT_LAB/workspace.json
"table_z_mm". A point 40 mm above the table is then z 237 here, whatever the room. Joint angles are frame-free;
only cuRobo obstacles go back to the real base frame (plan77.box adds BASE_DZ) and so do CUDA IK targets.

The .70 planner (s70/) keeps using kin directly: its base pose is its own.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import kin
from kin import DH, HI, LO, _A, rpy_xyz  # noqa: F401 - same model, re-exported

TAUGHT_TABLE_Z = 197.0


def _base_dz() -> float:
    v = os.environ.get("MYCOBOT_TABLE_Z")
    if v:
        return float(v) - TAUGHT_TABLE_Z
    p = Path(os.environ.get("MYCOBOT_LAB", "~/Documents/lab-jetson/console")).expanduser() / "workspace.json"
    try:
        return float(json.loads(p.read_text()).get("table_z_mm", TAUGHT_TABLE_Z)) - TAUGHT_TABLE_Z
    except (OSError, ValueError):
        return 0.0


BASE_DZ = _base_dz()


def fk(q_deg, tool=np.eye(4)):
    T = kin.fk(q_deg, tool)
    if BASE_DZ:
        T = T.copy()
        T[2, 3] -= BASE_DZ
    return T


def fk_taught(q_deg, tool=np.eye(4)):
    """FK of a TAUGHT joint configuration (templates in data/, recorded in the 197 mm room), in the planner frame.
    Those joint angles mean "this pose relative to the table as it stood then", i.e. the plain FK. Using fk() on
    them would drop every template by BASE_DZ relative to the table (2026-11-26: table 222 -> no grasp anywhere,
    gripper 22 mm into the table)."""
    return kin.fk(q_deg, tool)


def to_base(T):
    """A planner-frame pose -> the real .77 base frame."""
    T = np.array(T, float)
    T[..., 2, 3] += BASE_DZ
    return T


def ik(target_T, q0, **kw):
    return kin.ik(to_base(target_T), q0, **kw)
