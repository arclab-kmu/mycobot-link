"""px2w.py u v z : top-camera pixel -> world (mm) on plane z using calib_top3 (p_cam = R p + t)."""
import os as _os; _P = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, numpy as np, rsalign
S = _P
_c = np.load(f"{S}/data/calib_top3.npz"); R, t = _c["R"], _c["t"]
def px2w(u, v, z):
    C = rsalign.C; dc = np.array([(u - C["ppx"]) / C["fx"], (v - C["ppy"]) / C["fy"], 1.0]); cam = -R.T @ t; d = R.T @ dc; s = (z - cam[2]) / d[2]; return cam + s * d
def w2px(p):
    C = rsalign.C; c = R @ np.asarray(p, float) + t; return np.array([C["fx"] * c[0] / c[2] + C["ppx"], C["fy"] * c[1] / c[2] + C["ppy"]])
if __name__ == "__main__":
    print(np.round(px2w(float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])), 1))
