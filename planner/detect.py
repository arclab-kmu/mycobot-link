"""Find the paper cup and the two powder vials in the top camera's colour image (640 x 480).

- Cup: the strongest gradient ring of radius 38-62 px whose inside is bright and grey-white
  (FFT correlation of the gradient magnitude with thin ring kernels). ``cup_ok(u, v)`` (from locate)
  limits it to where a cup can be used; without that the white elbow of the parked .77 arm won over
  a cup at the frame edge (2026-11-26).
- Vials: white caps, i.e. bright disks of radius ~7-10 px that are brighter than the ring around them,
  outside the cup. The matcha vial is the one whose cap and body are the most green
  (g - (r + b) / 2); the other is the brown one.

Returns pixel centres; ``locate.py`` turns them into arm coordinates.
"""
from __future__ import annotations

import io
import urllib.request

import numpy as np
from PIL import Image


CUP_ARC = (-100.0, 150.0)
CAP_R = 13


def grab(url="http://127.0.0.1:8090/color.jpg", timeout=3.0) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB")).astype(float)


def _grad(gray):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1] = gray[2:] - gray[:-2]
    return np.hypot(gx, gy)


def _fft_corr(img, ker):
    H, W = img.shape
    kh, kw = ker.shape
    F = np.fft.rfft2(img, s=(H + kh, W + kw))
    K = np.fft.rfft2(ker[::-1, ::-1], s=(H + kh, W + kw))
    out = np.fft.irfft2(F * K, s=(H + kh, W + kw))
    return out[kh // 2: kh // 2 + H, kw // 2: kw // 2 + W]


def _ring(r, w=1.5, arc=None):
    """Thin ring kernel; ``arc`` = (a0, a1) deg (image angles, v down) keeps only that part of the ring."""
    n = int(r + w + 2)
    v, u = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(u, v)
    k = (np.abs(d - r) <= w).astype(float)
    if arc is not None:
        ang = np.degrees(np.arctan2(v, u))
        k *= ((ang >= arc[0]) & (ang <= arc[1])).astype(float)
    return k / k.sum()


def _disk(r):
    n = int(r + 1)
    v, u = np.mgrid[-n:n + 1, -n:n + 1]
    k = (np.hypot(u, v) <= r).astype(float)
    return k / k.sum()


def _edge_fit(G, u, v, r_guess, r_fix, shape):
    """Strongest edge on each ray near the expected radius, then a circle fit (radius fixed by r_fix if given)."""
    angs = np.radians(np.linspace(CUP_ARC[0], CUP_ARC[1], 90))
    rr0 = r_fix(u, v) if r_fix is not None else r_guess
    lo, hi = (rr0 - 2.5, rr0 + 3.0) if r_fix is not None else (rr0 - 9, rr0 + 9)
    pts = []
    for a in angs:
        rr = np.arange(lo, hi + 0.01, 0.5)
        uu, vv = u + rr * np.cos(a), v + rr * np.sin(a)
        ok = (uu >= 1) & (uu < shape[1] - 1) & (vv >= 1) & (vv < shape[0] - 1)
        if ok.sum() < 5:
            continue
        g = G[np.round(vv[ok]).astype(int), np.round(uu[ok]).astype(int)]
        i = int(np.argmax(g))
        pts.append((uu[ok][i], vv[ok][i]))
    P = np.array(pts)
    cx, cy, rad = u, v, rr0
    keep = np.ones(len(P), bool)
    res = np.zeros(len(P))
    for _ in range(4):
        if r_fix is None:
            A = np.column_stack([2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))])
            cx, cy, c0 = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
            rad = np.sqrt(c0 + cx * cx + cy * cy)
        else:  # centre only (Gauss-Newton), radius from the camera model
            rad = float(r_fix(cx, cy))
            for _ in range(10):
                d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
                Jm = np.column_stack([(cx - P[:, 0]) / d, (cy - P[:, 1]) / d])
                step = np.linalg.lstsq(Jm, -(d - rad), rcond=None)[0]
                cx, cy = cx + step[0], cy + step[1]
        res = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - rad)
        keep = res < max(1.5, 2.5 * np.median(res))
        if keep.all():
            break
        P, res = P[keep], res[keep]
    return dict(u=float(cx), v=float(cy), r_px=float(rad), n_edge=int(len(P)), fit_rms=float(np.sqrt(np.mean(res ** 2))))


def _peaks(score, n, sep):
    """Up to n local maxima of score, at least sep px apart, strongest first."""
    s = score.copy()
    out = []
    for _ in range(n):
        j, i = np.unravel_index(int(np.argmax(s)), s.shape)
        if s[j, i] <= 0:
            break
        out.append((float(s[j, i]), float(i), float(j)))
        s[max(j - sep, 0): j + sep + 1, max(i - sep, 0): i + sep + 1] = 0.0
    return out


def find_cup(rgb, radii=range(38, 64, 2), r_fix=None, cup_ok=None):
    """Coarse: the best partial-arc gradient ring (inside bright) for each radius. Fine: an edge circle fit
    from each coarse centre; the fit with the smallest rms (and enough edges) wins. ``r_fix(u, v) -> px``
    pins the radius to the rim's outer edge: the rolled rim shows an inner and an outer edge ~4 px apart.
    ``cup_ok(u, v) -> bool`` drops centres where no usable cup can stand (checked on coarse and fitted centres)."""
    gray = rgb.mean(2)
    G = _grad(gray)
    white = np.clip((gray - 150) / 60, 0, 1) * np.clip(1 - np.abs(rgb[..., 1] - rgb[..., 0]) / 40, 0, 1)
    coarse = []
    for r in radii:
        ring = _fft_corr(G, _ring(r, arc=CUP_ARC))
        inside = _fft_corr(white, _disk(0.7 * r))
        score = ring * (0.3 + inside)
        for s_, i, j in _peaks(score, 1 if cup_ok is None else 8, int(r)):
            if cup_ok is None or cup_ok(i, j):
                coarse.append((s_, i, j, r, float(inside[int(j), int(i)])))
                break
    if not coarse:
        return dict(u=0.0, v=0.0, r_px=0.0, r0=0, inside=0.0, score=0.0, n_edge=0, fit_rms=99.0)
    coarse.sort(reverse=True)
    top = [c for c in coarse if c[0] > 0.6 * coarse[0][0]]
    fits = []
    for s_, u, v, r, ins in top:
        f = _edge_fit(G, u, v, r, r_fix, rgb.shape)
        f.update(r0=r, inside=ins, score=s_)
        if f["n_edge"] >= 40 and (cup_ok is None or cup_ok(f["u"], f["v"])):
            fits.append(f)
    if not fits:
        s_, u, v, r, ins = coarse[0]
        return dict(u=u, v=v, r_px=float(r), r0=r, inside=ins, score=s_, n_edge=0, fit_rms=99.0)
    return min(fits, key=lambda f: f["fit_rms"])


def wood_mask(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    m = (r - b > 18) & (r > g) & (rgb.mean(2) > 90) & (rgb.mean(2) < 215)
    return _fft_corr(m.astype(float), _disk(6)) > 0.5


def find_caps(rgb, cup=None, n=6, gray_min=150.0, wood_min=0.6):
    gray = rgb.mean(2)
    cen = _fft_corr(gray, _disk(CAP_R - 3))
    ann = _fft_corr(gray, _ring(CAP_R + 6, 2.5))
    wood_ring = _fft_corr(wood_mask(rgb).astype(float), _ring(CAP_R + 6, 2.5))
    score = cen - ann
    score[(gray < gray_min) | (wood_ring < wood_min)] = -1e9
    B = CAP_R + 10
    score[:B], score[-B:], score[:, :B], score[:, -B:] = -1e9, -1e9, -1e9, -1e9
    if cup is not None:
        vv, uu = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
        score[np.hypot(uu - cup["u"], vv - cup["v"]) < cup["r_px"] + 14] = -1e9
    out = []
    s = score.copy()
    for _ in range(n):
        j, i = np.unravel_index(int(np.argmax(s)), s.shape)
        if s[j, i] < 12:
            break
        # centroid of the bright cap pixels around the peak
        y0, y1, x0, x1 = max(j - CAP_R, 0), j + CAP_R + 1, max(i - CAP_R, 0), i + CAP_R + 1
        patch = gray[y0:y1, x0:x1]
        m = patch > np.percentile(patch, 70)
        yy, xx = np.nonzero(m)
        if len(xx) == 0:  # a flat patch (possible with loose thresholds): no cap here
            s[max(j - 18, 0):j + 19, max(i - 18, 0):i + 19] = -1e9
            continue
        u, v = x0 + xx.mean(), y0 + yy.mean()
        body = rgb[max(j - CAP_R + 3, 0):j + CAP_R - 2, max(i - CAP_R + 3, 0):i + CAP_R - 2].reshape(-1, 3)
        green = float(np.mean(body[:, 1] - (body[:, 0] + body[:, 2]) / 2))
        out.append(dict(u=float(u), v=float(v), score=float(s[j, i]), green=green))
        s[max(j - 18, 0):j + 19, max(i - 18, 0):i + 19] = -1e9
    return out


def detect(rgb, height_of=None, cap_z=(245.0, 282.0), r_fix=None, cup_ok=None):
    """``height_of(list of (u, v)) -> list of z`` (depth) keeps only caps standing at vial height."""
    cup = find_cup(rgb, r_fix=r_fix, cup_ok=cup_ok)
    caps = find_caps(rgb, cup, n=8)
    if height_of is not None and caps:
        hs = height_of([(c["u"], c["v"]) for c in caps])
        for c, h in zip(caps, hs):
            c["z"] = h
        caps = [c for c in caps if c["z"] is not None and cap_z[0] <= c["z"] <= cap_z[1]]
    caps = sorted(caps, key=lambda c: -c["green"])
    matcha = caps[0] if caps and caps[0]["green"] > 3.0 else None
    others = [c for c in caps if c is not matcha]
    return dict(cup=cup, matcha=matcha, others=others)


def _label(mask):
    """4-connected components of a boolean image (numpy + a stack; masks here hold a few thousand pixels)."""
    lab = np.zeros(mask.shape, int)
    n = 0
    for v0, u0 in zip(*np.nonzero(mask)):
        if lab[v0, u0]:
            continue
        n += 1
        st = [(v0, u0)]
        lab[v0, u0] = n
        while st:
            v, u = st.pop()
            for dv, du in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = v + dv, u + du
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not lab[a, b]:
                    lab[a, b] = n
                    st.append((a, b))
    return lab, n


def caps_by_depth(depth_png, cam, table_z, cup=None, h_range=(50.0, 85.0), dia_mm=(25.0, 60.0)):
    """Vial caps from the top depth alone, for caps the colour finder cannot see (2026-11-27: silver caps at the
    exhibit site; AGX session's vials_by_depth.py). Blobs whose top stands h_range above the table and spans dia_mm
    (from the pixel size at the blob's distance, so it holds at any camera height), away from the cup. Returns [dict(u, v, z, dia_px, green=0)] in colour pixels, z = planner z."""
    import io
    from PIL import Image
    D = np.asarray(Image.open(io.BytesIO(depth_png))).astype(float)
    h, w = D.shape
    vv, uu = np.mgrid[0:h, 0:w]
    K = cam.depth_K
    ok = D > 0
    X = np.stack([(uu - K["ppx"]) / K["fx"] * D, (vv - K["ppy"]) / K["fy"] * D, D], -1)[ok]
    Xc = X @ np.asarray(cam.R_dc).T + np.asarray(cam.t_dc)
    Hz = ((Xc - np.asarray(cam.t)) @ np.asarray(cam.R))[:, 2] - table_z
    Kc = cam.K
    uc = Xc[:, 0] / Xc[:, 2] * Kc["fx"] + Kc["ppx"]
    vc = Xc[:, 1] / Xc[:, 2] * Kc["fy"] + Kc["ppy"]
    W_, H_ = cam.size
    sel = (Hz > h_range[0]) & (Hz < h_range[1]) & (uc >= 0) & (uc < W_) & (vc >= 0) & (vc < H_)
    m = np.zeros((H_, W_), bool)
    m[vc[sel].astype(int), uc[sel].astype(int)] = True
    for _ in range(2):  # closing: fill the stereo holes in a cap
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    for _ in range(2):
        m = m & np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    lab, n = _label(m)
    out = []
    us, vs, hs, zs = uc[sel], vc[sel], Hz[sel], Xc[sel, 2]
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) < 40:
            continue
        dia = 2.0 * np.sqrt(len(xs) / np.pi)
        u, v = float(xs.mean()), float(ys.mean())
        inb = (np.abs(us - u) < dia / 2) & (np.abs(vs - v) < dia / 2)
        if not inb.any():
            continue
        dmm = dia * float(np.median(zs[inb])) / Kc["fx"]
        if not dia_mm[0] <= dmm <= dia_mm[1]:
            continue
        if cup is not None and cup.get("n_edge", 0) >= 40 and np.hypot(u - cup["u"], v - cup["v"]) < cup["r_px"] + 20:
            continue
        out.append(dict(u=u, v=v, z=float(table_z + np.percentile(hs[inb], 90)), dia_px=float(dia), dia_mm=float(dmm),
                        green=0.0, by="depth"))
    return out
