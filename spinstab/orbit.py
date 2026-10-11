"""Turn the hand-held cameras of a walk-around into an ideal turntable orbit.

Given SfM camera poses (from COLMAP, VGGT, ...):
  * subject centre: the 3D point closest to all optical axes (least squares), i.e. what the operator aimed at;
  * orbit plane: PCA plane of the camera centres; its normal is "up", oriented with the cameras' up vectors;
  * orbit circle: algebraic circle fit of the centres projected into that plane.

Every real camera then gets an angle on the orbit, and virtual cameras are placed at uniform angles on the
fitted circle, level (zero roll) and aimed at the subject centre. A turntable shot is exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Orbit:
    target: np.ndarray      # subject centre (3,)
    up: np.ndarray          # unit normal of the orbit plane (3,)
    center: np.ndarray      # circle centre, in the orbit plane (3,)
    radius: float
    e1: np.ndarray          # in-plane basis: angle 0 direction (3,)
    e2: np.ndarray          # in-plane basis: angle 90 direction (3,)
    rms_radial: float       # RMS deviation of real cameras from the circle (same units as poses)
    rms_height: float       # RMS deviation of real cameras from the plane

    def angle_of(self, p: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(p) - self.center
        return np.arctan2(d @ self.e2, d @ self.e1)

    def position(self, angle: float, height: float = 0.0) -> np.ndarray:
        return self.center + self.radius * (np.cos(angle) * self.e1 + np.sin(angle) * self.e2) + height * self.up


def closest_point_to_rays(origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Least-squares point minimising the summed squared distance to lines (o_i + t d_i)."""
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)      # projector onto the plane orthogonal to the ray
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


def fit_circle_2d(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Kasa algebraic fit: x^2 + y^2 + Dx + Ey + F = 0."""
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    rhs = -(xy ** 2).sum(axis=1)
    D, E, F = np.linalg.lstsq(A, rhs, rcond=None)[0]
    c = np.array([-D / 2, -E / 2])
    return c, float(np.sqrt(max(c @ c - F, 0.0)))


def fit_orbit(centers: np.ndarray, rotations: np.ndarray) -> Orbit:
    """centers: (N,3) camera centres; rotations: (N,3,3) world->camera (OpenCV: x right, y down, z forward)."""
    forward = rotations[:, 2, :]            # camera z axis in world coordinates = row 2 of R
    cam_up = -rotations[:, 1, :]            # OpenCV y points down
    target = closest_point_to_rays(centers, forward)

    mean = centers.mean(axis=0)
    _, _, vt = np.linalg.svd(centers - mean)
    up = vt[2] if np.dot(vt[2], cam_up.mean(axis=0)) > 0 else -vt[2]
    e1 = vt[0] - np.dot(vt[0], up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    rel = centers - mean
    xy = np.column_stack([rel @ e1, rel @ e2])
    c2, radius = fit_circle_2d(xy)
    center = mean + c2[0] * e1 + c2[1] * e2
    heights = rel @ up
    radial = np.linalg.norm(xy - c2, axis=1) - radius
    return Orbit(target, up, center, radius, e1, e2,
                 float(np.sqrt(np.mean(radial ** 2))), float(np.sqrt(np.mean(heights ** 2))))


def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) in OpenCV convention, level with respect to `up` (no roll)."""
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(z, up)                     # right = forward x up  (OpenCV: y down)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ position


def smooth_profile(orbit: Orbit, centers: np.ndarray, sigma_deg: float):
    """Radius and height of the real cameras as smooth functions of orbit angle (circular Gaussian kernel).

    Following the walker's slow distance/height changes keeps each virtual camera close to its source frame,
    which shrinks parallax for scene points off the subject plane. Fast wobble is still removed.
    """
    rel = centers - orbit.center
    ang = orbit.angle_of(centers)
    radius = np.linalg.norm(rel - np.outer(rel @ orbit.up, orbit.up), axis=1)
    height = rel @ orbit.up
    s = np.deg2rad(sigma_deg)

    def at(a: float) -> tuple[float, float]:
        d = np.angle(np.exp(1j * (ang - a)))
        w = np.exp(-0.5 * (d / s) ** 2)
        w /= w.sum()
        return float(w @ radius), float(w @ height)
    return at


def virtual_cameras(orbit: Orbit, n: int, start_angle: float, direction: int,
                    profile=None) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """n level cameras at uniform angles, aimed at the subject. Returns (angle, R, t) tuples.

    profile: None for an exact circle (constant radius and height), or smooth_profile(...) to follow the
    capture's slowly varying distance and height.
    """
    out = []
    for k in range(n):
        a = start_angle + direction * 2 * np.pi * k / n
        if profile is None:
            pos = orbit.position(a)
        else:
            r, h = profile(a)
            pos = orbit.center + r * (np.cos(a) * orbit.e1 + np.sin(a) * orbit.e2) + h * orbit.up
        R, t = look_at(pos, orbit.target, orbit.up)
        out.append((a, R, t))
    return out
