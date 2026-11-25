"""Local copy of the cuRobo collision spheres: each sphere is fixed in one DH link frame, so after one fit
(a few /spheres calls at random poses) the spheres for any q come from numpy FK in microseconds."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

import kin

CACHE = Path(__file__).resolve().parent / "data" / "spheres_dh.json"


def frames(q):
    T = np.eye(4)
    out = [T.copy()]
    for qi, (off, d, a, al) in zip(np.radians(q), kin.DH):
        T = T @ kin._A(qi + off, d, a, al)
        out.append(T.copy())
    return out


def fit(api, n=5, seed=0):
    rng = np.random.default_rng(seed)
    qs = [rng.uniform(kin.LO * 0.6, kin.HI * 0.6) for _ in range(n)]
    S = []
    for q in qs:
        req = urllib.request.Request(api.rstrip("/") + "/spheres", data=json.dumps({"q": [float(x) for x in q]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            S.append(np.array(json.load(r)["spheres"], float))
    m = len(S[0])
    model = []
    for i in range(m):
        best = None
        for k in range(7):
            loc = [np.linalg.solve(frames(q)[k], np.r_[s[i, :3], 1.0])[:3] for q, s in zip(qs, S)]
            spread = float(np.max(np.std(loc, axis=0)))
            if best is None or spread < best[0]:
                best = (spread, k, np.mean(loc, axis=0))
        model.append({"link": best[1], "p": best[2].round(3).tolist(), "r": float(S[0][i, 3]), "spread": round(best[0], 3)})
    CACHE.write_text(json.dumps(model))
    return model


_M = None


def spheres(q):
    global _M
    if _M is None:
        _M = json.loads(CACHE.read_text())
    F = frames(q)
    return np.array([np.r_[(F[s["link"]] @ np.r_[s["p"], 1.0])[:3], s["r"]] for s in _M])
