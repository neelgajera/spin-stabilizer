"""Synthetic walk-around: a textured box on a checkerboard floor, filmed by a wobbly hand-held orbit.

Writes <out>/images/*.jpg and <out>/sparse/0 (exact poses), so the stabilizer can be tested and demoed
without any real capture:
    python tests/make_synthetic.py data/synthetic && python stabilize.py data/synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spinstab import colmap_io as cio   # noqa: E402
from spinstab import orbit as O          # noqa: E402

W_, H_ = 960, 540
K = np.array([[700.0, 0, W_ / 2], [0, 700.0, H_ / 2], [0, 0, 1]])


def scene(rng) -> tuple[np.ndarray, np.ndarray]:
    """Dense colored points: a 4 x 1.8 x 1.4 box (the 'car') and a 12 x 12 checkerboard floor (z up)."""
    pts, cols = [], []
    n = 700_000
    # box surface
    u = rng.uniform(-1, 1, (n, 3)) * [2.0, 0.9, 0.7]
    face = rng.integers(0, 3, n)
    u[np.arange(n), face] = np.sign(rng.uniform(-1, 1, n)) * np.array([2.0, 0.9, 0.7])[face]
    u[:, 2] += 0.7
    stripes = ((np.floor(u[:, 0] * 3) + np.floor(u[:, 2] * 5)) % 2)[:, None]
    base = np.array([[40, 40, 200]]) * (0.6 + 0.4 * stripes) + rng.normal(0, 18, (n, 3))
    pts.append(u); cols.append(base)
    # floor
    f = np.c_[rng.uniform(-6, 6, (n, 2)), np.zeros(n)]
    chk = ((np.floor(f[:, 0]) + np.floor(f[:, 1])) % 2)[:, None]
    pts.append(f); cols.append(np.where(chk > 0, 190, 70) + rng.normal(0, 10, (n, 3)))
    return np.vstack(pts), np.clip(np.vstack(cols), 0, 255).astype(np.uint8)


def render(R, t, X, C) -> np.ndarray:
    cam = (R @ X.T + t[:, None]).T
    ok = cam[:, 2] > 0.1
    cam, C = cam[ok], C[ok]
    uv = (K @ cam.T).T
    uv = uv[:, :2] / uv[:, 2:]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W_ - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H_ - 1)
    uv, z, C = uv[inside].astype(int), cam[inside, 2], C[inside]
    # 3x3 splats; all splat pixels drawn far to near (painter's algorithm) so near surfaces occlude.
    off = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    px = np.clip((uv[:, None, :] + off[None]).reshape(-1, 2), 0, [W_ - 1, H_ - 1])
    zz, cc = np.repeat(z, len(off)), np.repeat(C, len(off), axis=0)
    order = np.argsort(-zz)
    img = np.full((H_, W_, 3), (235, 225, 210), np.uint8)
    img[px[order, 1], px[order, 0]] = cc[order]
    return cv2.GaussianBlur(img, (3, 3), 0)


def main(out: Path, n: int = 120, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    X, C = scene(rng)
    up = np.array([0, 0, 1.0])
    target = np.array([0, 0, 0.7])
    (out / "images").mkdir(parents=True, exist_ok=True)
    ims = []
    for i, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False)):
        r = 5.5 + 0.25 * np.sin(3 * a) + rng.normal(0, 0.08)         # uneven walking distance
        h = 1.4 + rng.normal(0, 0.1)                                   # hand height
        c = np.array([r * np.cos(a), r * np.sin(a), h])
        R, t = O.look_at(c, target + rng.normal(0, 0.18, 3), up)       # aim jitter
        roll = np.deg2rad(rng.normal(0, 3.0))                          # hand roll
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R, t = Rz @ R, Rz @ t
        name = f"{i:04d}.jpg"
        cv2.imwrite(str(out / "images" / name), render(R, t, X, C), [cv2.IMWRITE_JPEG_QUALITY, 92])
        ims.append(cio.Image(i + 1, name, 1, R, t))
    sub = rng.choice(len(X), 20_000, replace=False)
    cio.write_model(out / "sparse" / "0", [cio.Camera(1, "PINHOLE", W_, H_, np.array([700, 700, W_ / 2, H_ / 2]))],
                    ims, X[sub], C[sub])
    print(f"wrote {n} frames to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/synthetic"))
