"""Align a D435i Z16 depth frame to the color frame with factory intrinsics/extrinsics (read 2026-11-22)."""
import numpy as np
from PIL import Image
D = dict(fx=380.8627014160156, fy=380.8627014160156, ppx=317.6974182128906, ppy=237.435791015625)
C = dict(fx=606.38916015625, fy=606.2594604492188, ppx=320.3999328613281, ppy=243.50770568847656)
R_DC = np.array([0.9999464154243469, -0.009539746679365635, -0.004017072729766369, 0.00952951330691576, 0.9999513030052185,
                 -0.00255896826274693, 0.004041289445012808, 0.002520550275221467, 0.9999886751174927]).reshape(3, 3).T  # rs stores column-major
T_DC = np.array([0.014641907997429371, 7.076869223965332e-06, 0.0002481822739355266]) * 1000.0  # mm
def aligned_depth(depth_png):
    d = np.array(Image.open(depth_png)).astype(np.float64)
    v, u = np.nonzero(d > 0); z = d[v, u]
    X = np.stack([(u - D["ppx"]) / D["fx"] * z, (v - D["ppy"]) / D["fy"] * z, z], 1)
    Xc = X @ R_DC.T + T_DC
    uc = np.round(Xc[:, 0] / Xc[:, 2] * C["fx"] + C["ppx"]).astype(int); vc = np.round(Xc[:, 1] / Xc[:, 2] * C["fy"] + C["ppy"]).astype(int)
    ok = (uc >= 0) & (uc < 640) & (vc >= 0) & (vc < 480)
    out = np.full((480, 640), np.inf)
    np.minimum.at(out, (vc[ok], uc[ok]), Xc[ok, 2])
    out[np.isinf(out)] = 0
    return out
def deproject_color(u, v, z):
    return np.array([(u - C["ppx"]) / C["fx"] * z, (v - C["ppy"]) / C["fy"] * z, z])
def point(ad, u, v, r=3):
    w = ad[int(v) - r:int(v) + r + 1, int(u) - r:int(u) + r + 1].ravel(); w = w[w > 0]
    if w.size == 0: return None
    z = float(np.median(w)); return deproject_color(u, v, z)
