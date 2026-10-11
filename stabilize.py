"""Turn a hand-held 360 walk-around into a level, evenly spaced turntable spin.

    python stabilize.py runs/my-car/dataset                  # images/ + sparse/0 from COLMAP or VGGT
    python stabilize.py runs/my-car/dataset --frames 36 --out out/my-car

The dataset is a standard undistorted COLMAP layout. video-to-splat produces one from a phone video:
    python ../video-to-splat/run.py my_car.mp4 --until poses [--poses vggt]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spinstab import stabilize as S
from spinstab import warp as W


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="folder with images/ and sparse/0/")
    ap.add_argument("--frames", type=int, default=72, help="output frames per revolution (default 72 = 5 deg)")
    ap.add_argument("--zoom", type=float, help="fixed zoom; default: smallest zoom with no borders on 95%% of frames")
    ap.add_argument("--path", choices=["smooth", "circle"], default="smooth",
                    help="smooth: follow the capture's slow distance/height changes (less parallax); "
                         "circle: exact turntable")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", type=Path, help="output folder (default: <dataset>/../spin)")
    args = ap.parse_args()

    out = args.out or args.dataset.parent / "spin"
    t0 = time.time()
    plan, xyz = S.plan(args.dataset / "sparse" / "0", args.frames, args.path)
    zoom = args.zoom or W.auto_zoom(lambda z: S.homographies(plan, z), plan.out_wh, plan.src_wh)
    hs = S.homographies(plan, zoom)
    m = S.metrics(plan, hs, S.subject_points(xyz, plan.orbit), zoom)
    S.render(plan, hs, args.dataset / "images", out, args.fps)
    m.update({"zoom": round(zoom, 3), "seconds": round(time.time() - t0, 1),
              "unique_source_frames": len({s.name for s in plan.sources})})
    (out / "metrics.json").write_text(json.dumps(m, indent=2))

    print(f"wrote {out / 'spin.mp4'} and {out / 'before_after.mp4'}")
    print(f"jitter of subject points: {m['jitter_px_raw']} px -> {m['jitter_px_stabilized']} px "
          f"(perfect turntable: {m['jitter_px_ideal_turntable']} px); "
          f"roll std {m['roll_std_deg_raw']} deg -> 0; zoom {m['zoom']}x")


if __name__ == "__main__":
    main()
