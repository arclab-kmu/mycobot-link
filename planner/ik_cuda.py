"""ctypes front end of planner/cuda/libikchain.so (planner/cuda/ik_chain.cu): every candidate's IK chain in one
GPU launch. `available()` is False when the library is missing (not built on this machine) or fails to load;
the planner then keeps its numpy loop.

Set MYCOBOT_IK_CUDA=0 to force the numpy path (e.g. to compare the two).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

import kin

LIB_PATH = Path(__file__).resolve().parent / "cuda" / "libikchain.so"
_lib = None
_tried = False


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    if os.environ.get("MYCOBOT_IK_CUDA", "1") == "0" or not LIB_PATH.exists():
        return None
    try:
        lib = ctypes.CDLL(str(LIB_PATH))
    except OSError:
        return None
    D = ctypes.POINTER(ctypes.c_double)
    I = ctypes.POINTER(ctypes.c_int)
    lib.ik_chain.argtypes = [D, D, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                             ctypes.c_double, D, D, D, I]
    lib.ik_chain.restype = ctypes.c_int
    _lib = lib
    return _lib


def available() -> bool:
    return _load() is not None


_DH = np.ascontiguousarray(np.array(kin.DH, float).reshape(-1))


def chain(T: np.ndarray, seed: np.ndarray, step0: float = 12.0, step_fb: float = 12.0, fb_from_prev: bool = False,
          fail_e: float = 0.3):
    """T [C,K,4,4] targets, seed [C,K,6] deg. Returns (q [C,K,6], err [C,K], bad [C]; bad = first failing k or -1).
    Entries after a candidate's failing pose are left zero."""
    lib = _load()
    if lib is None:
        raise RuntimeError("libikchain.so is not available")
    C, K = T.shape[:2]
    T = np.ascontiguousarray(T, dtype=np.float64)
    seed = np.ascontiguousarray(seed, dtype=np.float64)
    q = np.zeros((C, K, 6))
    e = np.zeros((C, K))
    bad = np.zeros(C, dtype=np.int32)
    P = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))  # noqa: E731
    rc = lib.ik_chain(P(T), P(seed), C, K, float(step0), float(step_fb), int(bool(fb_from_prev)), float(fail_e), P(_DH),
                      P(q), P(e), bad.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    if rc != 0:
        raise RuntimeError(f"ik_chain CUDA error {rc}")
    return q, e, bad
