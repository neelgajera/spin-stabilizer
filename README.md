# spin-stabilizer

Turn a hand-held 360° walk-around into a **level, centred, evenly spaced turntable spin**, using nothing but
the camera poses and sparse points from Structure-from-Motion.

![Hand-held vs stabilized](docs/before_after.gif)

*Synthetic walk-around with ±3° hand roll, aim jitter and uneven walking distance (left) and the stabilized
spin (right). Reproduce it with `python tests/make_synthetic.py data/synthetic && python stabilize.py data/synthetic`.*

| Synthetic benchmark (72 output frames) | Hand-held | Stabilized | Perfect turntable |
|---|---|---|---|
| Jitter of subject points (RMS 2nd difference, px) | 81.4 | **10.6** | 1.7 |
| Roll std (deg) | 3.2 | **0** | 0 |
| Runtime (CPU) | | **~3 s** | |

## Why

Dealers, marketplaces and e-commerce sites show products as interactive 360° spins. A turntable produces
perfect ones, but most items (cars, furniture) are filmed by someone walking around them with a phone. Those
frames tilt, drift off-centre, change scale and are unevenly spaced, so the spin wobbles. Classic 2D video
stabilizers smooth the shake but don't know the camera is orbiting a subject, so they can't produce evenly
spaced, level views.

## How it works

1. **Poses.** Any undistorted COLMAP model works: run COLMAP, or use
   [video-to-splat](../video-to-splat) with `--poses vggt` to get poses from a single
   [VGGT](https://github.com/facebookresearch/vggt) forward pass in seconds.
2. **Orbit fit** ([`spinstab/orbit.py`](spinstab/orbit.py)).
   - *Subject centre*: the least-squares point closest to all optical axes, i.e. what the operator aimed at.
   - *Up axis*: normal of the PCA plane through the camera centres, oriented with the cameras' up vectors.
   - *Orbit*: algebraic (Kasa) circle fit in that plane. With `--path smooth` (default), radius and height
     follow the capture's slow changes (circular Gaussian, σ = 30°), which keeps virtual cameras close to real ones.
3. **Virtual cameras.** N cameras at uniform angles on the orbit, aimed at the subject centre with zero roll.
4. **Warp** ([`spinstab/warp.py`](spinstab/warp.py)). Each output frame comes from the nearest real frame
   through the homography induced by the plane through the subject centre facing the virtual camera:

   $$H = K_v\left(R_{rel} + \frac{t_{rel}\, n_s^\top}{d_s}\right)K_s^{-1}$$

   This corrects rotation, roll, height and distance in one warp, and the subject plane is registered exactly.
5. **Auto-zoom.** A binary search picks the smallest zoom with no black borders on 95% of frames.
6. **Metrics.** Sparse SfM points near the subject are tracked through the raw and warped frames, and the RMS
   of their second difference (acceleration) measures jitter. The same points through ideal turntable cameras
   give the floor.

## Usage

```bash
pip install -r requirements.txt
python stabilize.py path/to/dataset                 # dataset = images/ + sparse/0/ (undistorted PINHOLE)
python stabilize.py path/to/dataset --frames 36 --path circle --out out/
```

Outputs: `spin.mp4`, `before_after.mp4`, `frames/NNN.jpg` (for a web 360 viewer) and `metrics.json`.

From a phone video:

```bash
python ../video-to-splat/run.py walkaround.mp4 --until poses              # COLMAP, minutes
python ../video-to-splat/run.py walkaround.mp4 --until poses --poses vggt # VGGT, seconds
python stabilize.py ../video-to-splat/runs/walkaround/dataset
```

## Limitations and next steps

- A single homography is exact only on the subject plane. Points far in front of or behind it keep some
  parallax, which is the gap between 10.6 px and the 1.7 px floor above. Next step: a content-preserving
  mesh warp (Liu et al., *Content-Preserving Warps for 3D Video Stabilization*, SIGGRAPH 2009) driven by
  the same sparse points.
- Assumes one roughly closed orbit around one subject.
- Output frames come from real frames, so angles the capture skipped can't be filled.
  [video-to-splat](../video-to-splat) solves that by rendering any angle from a 3D model.

## Background

I led the production version of this idea, a 360° car cataloging product for dealerships, at Spyne
([spyne.ai/360-car-photography](https://www.spyne.ai/360-car-photography)). This repository is an
independent re-implementation written from scratch on synthetic and self-captured data. It contains no
Spyne code, models or data.

## License

MIT, see [LICENSE](LICENSE).
