"""Fast geometry checks (no images): python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spinstab import colmap_io as cio  # noqa: E402
from spinstab import orbit as O  # noqa: E402
from spinstab import warp as W  # noqa: E402


def jittered_orbit(rng, n=120, tilt_deg=25.0):
    T = np.array([0.3, -0.2, 1.0])
    a = np.deg2rad(tilt_deg)
    up = np.array([0, np.sin(a), np.cos(a)])
    e1 = np.cross(up, [1, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    Rs, cs = [], []
    for ang in np.linspace(0, 2 * np.pi, n, endpoint=False):
        c = T + (4 + rng.normal(0, 0.15)) * (np.cos(ang) * e1 + np.sin(ang) * e2) + (0.6 + rng.normal(0, 0.08)) * up
        R, _ = O.look_at(c, T + rng.normal(0, 0.15, 3), up)
        r = np.deg2rad(rng.normal(0, 3))
        Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        Rs.append(Rz @ R); cs.append(c)
    return T, up, np.array(cs), np.array(Rs)


def test_fit_orbit_recovers_subject_and_up():
    T, up, cs, Rs = jittered_orbit(np.random.default_rng(0))
    orb = O.fit_orbit(cs, Rs)
    assert np.linalg.norm(orb.target - T) < 0.06
    assert np.degrees(np.arccos(orb.up @ up)) < 0.5
    assert abs(orb.radius - 4.0) < 0.05


def test_virtual_cameras_are_level_and_aimed():
    T, up, cs, Rs = jittered_orbit(np.random.default_rng(1))
    orb = O.fit_orbit(cs, Rs)
    for _, R, t in O.virtual_cameras(orb, 12, 0.0, 1):
        assert abs(R[0] @ orb.up) < 1e-9                    # camera x axis horizontal: no roll
        x = R @ orb.target + t
        assert abs(x[0]) < 1e-9 and abs(x[1]) < 1e-9          # subject on the optical axis


def test_plane_homography_is_exact_on_plane():
    rng = np.random.default_rng(2)
    T, up, cs, Rs = jittered_orbit(rng)
    orb = O.fit_orbit(cs, Rs)
    K = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1.0]])
    _, Rv, tv = O.virtual_cameras(orb, 1, orb.angle_of(cs[0])[0], 1)[0]
    Rs0, ts0 = Rs[0], -Rs[0] @ cs[0]
    cv = -Rv.T @ tv
    n = (cv - orb.target) / np.linalg.norm(cv - orb.target)
    H = W.plane_homography(K, Rs0, ts0, K, Rv, tv, n, orb.target)
    u = np.cross(n, [1, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    P = orb.target + rng.uniform(-1, 1, (50, 1)) * u + rng.uniform(-1, 1, (50, 1)) * v

    def proj(R, t, X):
        x = K @ (R @ X.T + t[:, None])
        return (x[:2] / x[2]).T
    xs = proj(Rs0, ts0, P)
    xh = H @ np.c_[xs, np.ones(len(xs))].T
    assert np.abs((xh[:2] / xh[2]).T - proj(Rv, tv, P)).max() < 1e-6


def test_colmap_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    cams = [cio.Camera(1, "PINHOLE", 1600, 900, np.array([1200.0, 1210.0, 800.0, 450.0]))]
    ims = [cio.Image(i, f"{i:04d}.jpg", 1, cio.qvec_to_R(rng.normal(size=4)), rng.normal(size=3)) for i in range(1, 6)]
    xyz = rng.normal(size=(50, 3))
    rgb = rng.integers(0, 255, (50, 3), dtype=np.uint8)
    cio.write_model(tmp_path, cams, ims, xyz, rgb)
    c2, i2, x2, r2 = cio.read_model(tmp_path)
    assert np.allclose(c2[1].params, cams[0].params)
    assert all(np.allclose(i2[k].R, ims[k - 1].R) and np.allclose(i2[k].t, ims[k - 1].t) for k in i2)
    assert np.allclose(x2, xyz) and (r2 == rgb).all()
