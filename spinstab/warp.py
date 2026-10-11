"""Warp real frames into virtual turntable cameras with plane-induced homographies.

For a source camera (R_s, t_s) and a virtual camera (R_v, t_v), points on a world plane n.X = c map between the
two images by the homography

    H = K_v (R_rel + t_rel n_s^T / d_s) K_s^-1,   R_rel = R_v R_s^T,  t_rel = t_v - R_rel t_s,
                                                  n_s = R_s n,        d_s = c + n_s . t_s

The plane goes through the subject centre and faces the virtual camera, so the subject is registered exactly
and depth away from that plane only causes small parallax (the nearest real frame is always used, so the
baseline between source and virtual camera stays a few centimetres).
"""
from __future__ import annotations

import numpy as np


def plane_homography(K_s, R_s, t_s, K_v, R_v, t_v, normal, point) -> np.ndarray:
    R_rel = R_v @ R_s.T
    t_rel = t_v - R_rel @ t_s
    n_s = R_s @ normal
    d_s = float(normal @ point) + float(n_s @ t_s)
    H = K_v @ (R_rel + np.outer(t_rel, n_s) / d_s) @ np.linalg.inv(K_s)
    return H / H[2, 2]


def corners_inside(H: np.ndarray, out_wh: tuple[int, int], src_wh: tuple[int, int]) -> bool:
    """True if every output corner maps (through H^-1) inside the source image, i.e. no black border."""
    w, h = out_wh
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=float).T
    src = np.linalg.inv(H) @ pts
    src = src[:2] / src[2]
    return bool((src[0] >= 0).all() and (src[0] <= src_wh[0]).all() and (src[1] >= 0).all() and (src[1] <= src_wh[1]).all())


def zoomed_K(K: np.ndarray, zoom: float, out_wh: tuple[int, int]) -> np.ndarray:
    Kz = K.copy()
    Kz[0, 0] *= zoom
    Kz[1, 1] *= zoom
    Kz[0, 2], Kz[1, 2] = out_wh[0] / 2, out_wh[1] / 2
    return Kz


def auto_zoom(make_H, out_wh, src_wh, coverage: float = 0.95, lo: float = 1.0, hi: float = 2.5) -> float:
    """Smallest zoom at which at least `coverage` of the frames have no black border (binary search).

    make_H(zoom) -> list of homographies for every output frame at that zoom.
    """
    def ok(z):
        hs = make_H(z)
        return np.mean([corners_inside(H, out_wh, src_wh) for H in hs]) >= coverage
    if not ok(hi):
        return hi
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi
