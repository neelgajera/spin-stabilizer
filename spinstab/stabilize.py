"""Stabilize a walk-around into a turntable spin.

Inputs: a COLMAP model (sparse/0) and the images it was built from. Outputs: N level frames at uniform angles,
the matching hand-held frames for comparison, and jitter metrics measured on the sparse SfM points.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from spinstab import colmap_io as cio
from spinstab import orbit as O
from spinstab import warp as W


@dataclass
class Plan:
    orbit: O.Orbit
    sources: list            # cio.Image used for each output frame
    virtual: list            # (angle, R, t) for each output frame
    K_src: np.ndarray
    src_wh: tuple
    out_wh: tuple


def plan(model: Path, n_out: int = 72, path: str = "smooth", sigma_deg: float = 30.0) -> tuple[Plan, np.ndarray]:
    cams, imgs, xyz, _ = cio.read_model(model)
    if len(cams) != 1:
        print(f"note: {len(cams)} cameras in the model; using the first one's intrinsics for all frames")
    cam = next(iter(cams.values()))
    if cam.model not in ("PINHOLE", "SIMPLE_PINHOLE"):
        raise ValueError(f"expected undistorted PINHOLE images, got {cam.model}; run COLMAP image_undistorter first")
    ims = sorted(imgs.values(), key=lambda im: im.name)
    centers = np.array([im.center for im in ims])
    rots = np.array([im.R for im in ims])
    orb = O.fit_orbit(centers, rots)

    angles = np.unwrap(orb.angle_of(centers))
    direction = 1 if angles[-1] > angles[0] else -1
    profile = O.smooth_profile(orb, centers, sigma_deg) if path == "smooth" else None
    virtual = O.virtual_cameras(orb, n_out, angles[0], direction, profile)
    src_angles = orb.angle_of(centers)
    sources = []
    for a, _, _ in virtual:
        diff = np.abs(np.angle(np.exp(1j * (src_angles - a))))   # circular distance
        sources.append(ims[int(np.argmin(diff))])
    wh = (int(cam.width), int(cam.height))
    return Plan(orb, sources, virtual, cam.K(), wh, wh), xyz


def homographies(p: Plan, zoom: float) -> list[np.ndarray]:
    K_v = W.zoomed_K(p.K_src, zoom, p.out_wh)
    hs = []
    for (_, Rv, tv), src in zip(p.virtual, p.sources):
        cv_ = -Rv.T @ tv
        n = (cv_ - p.orbit.target) / np.linalg.norm(cv_ - p.orbit.target)
        hs.append(W.plane_homography(p.K_src, src.R, src.t, K_v, Rv, tv, n, p.orbit.target))
    return hs


def subject_points(xyz: np.ndarray, orb: O.Orbit, frac: float = 0.6) -> np.ndarray:
    """Sparse points within `frac` x orbit radius of the subject centre (drops far background)."""
    keep = np.linalg.norm(xyz - orb.target, axis=1) < frac * orb.radius
    return xyz[keep] if keep.sum() >= 20 else xyz


def jitter(tracks: np.ndarray) -> float:
    """RMS second difference of 2D tracks (K frames, P points, 2): 0 for perfectly smooth motion."""
    acc = tracks[2:] - 2 * tracks[1:-1] + tracks[:-2]
    return float(np.sqrt(np.nanmean(np.sum(acc ** 2, axis=2))))


def metrics(p: Plan, hs: list[np.ndarray], pts: np.ndarray, zoom: float) -> dict:
    """Jitter of the subject's sparse points along the spin, in source pixels.

    raw:        the nearest hand-held frames, as shot
    stabilized: the same frames after warping
    ideal:      a perfect turntable (virtual cameras projecting the points directly); points still move on
                curves, so this is the floor a stabilizer can reach, not zero.
    """
    def project(K, R, t, X):
        x = K @ (R @ X.T + t[:, None])
        return (x[:2] / x[2]).T

    raw = np.stack([project(p.K_src, s.R, s.t, pts) for s in p.sources])
    stab = []
    for H, r in zip(hs, raw):
        x = H @ np.c_[r, np.ones(len(r))].T
        stab.append((x[:2] / x[2]).T)
    stab = np.stack(stab)
    stab /= zoom   # output pixels are zoomed; measure in source pixels so the comparison is fair
    K_v1 = W.zoomed_K(p.K_src, 1.0, p.out_wh)
    ideal = np.stack([project(K_v1, Rv, tv, pts) for _, Rv, tv in p.virtual])   # perfect turntable, same frames

    def roll_deg(K, R, t):   # tilt of the projected up axis through the subject centre
        a = project(K, R, t, p.orbit.target[None])[0]
        b = project(K, R, t, (p.orbit.target + 0.1 * p.orbit.radius * p.orbit.up)[None])[0]
        return np.degrees(np.arctan2(b[0] - a[0], -(b[1] - a[1])))
    rolls = np.array([roll_deg(p.K_src, s.R, s.t) for s in p.sources])

    centre_raw = np.stack([project(p.K_src, s.R, s.t, p.orbit.target[None])[0] for s in p.sources])
    return {
        "frames": len(hs),
        "jitter_px_raw": round(jitter(raw), 2),
        "jitter_px_stabilized": round(jitter(stab), 2),
        "jitter_px_ideal_turntable": round(jitter(ideal), 2),
        "subject_centre_drift_px_raw": round(float(np.linalg.norm(centre_raw - centre_raw.mean(0), axis=1).std()), 2),
        "roll_std_deg_raw": round(float(rolls.std()), 2),
        "roll_std_deg_stabilized": 0.0,     # virtual cameras are level by construction
        "orbit_rms_radial": round(p.orbit.rms_radial / p.orbit.radius * 100, 2),   # % of radius
        "orbit_rms_height": round(p.orbit.rms_height / p.orbit.radius * 100, 2),
    }


def render(p: Plan, hs: list[np.ndarray], images: Path, out: Path, fps: int = 24) -> None:
    (out / "frames").mkdir(parents=True, exist_ok=True)
    w, h = p.out_wh
    four = cv2.VideoWriter_fourcc(*"mp4v")
    spin = cv2.VideoWriter(str(out / "spin.mp4"), four, fps, (w, h))
    side = cv2.VideoWriter(str(out / "before_after.mp4"), four, fps, (w, h // 2))
    for k, (H, src) in enumerate(zip(hs, p.sources)):
        img = cv2.imread(str(images / src.name))
        if img is None:
            raise FileNotFoundError(images / src.name)
        warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(out / "frames" / f"{k:03d}.jpg"), warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
        spin.write(warped)
        pair = np.hstack([cv2.resize(img, (w // 2, h // 2)), cv2.resize(warped, (w // 2, h // 2))])
        for text, x in (("hand-held", 12), ("stabilized", w // 2 + 12)):
            cv2.putText(pair, text, (x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        side.write(pair)
    spin.release()
    side.release()
