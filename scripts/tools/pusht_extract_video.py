#!/usr/bin/env python3
"""Extract MP4 videos from a Push-T bimanual HDF5 dataset.

Each saved demo becomes one MP4 file.  Pass --obs_key to select which image
observation to extract (see available keys below).

Available --obs_key values (bimanual Push-T)
--------------------------------------------
    obs/fixed_cam   128×128 RGB top-down camera image  [default]
    obs/edge_cam    128×128 synthetic edge image: white T-block outline, green
                    goal-T outline, gray table boundary, cyan left-arm contour
                    and orange right-arm contour (from instance segmentation).

HDF5 path layout
----------------
    data/<demo>/obs/fixed_cam        shape (T, 128, 128, 3)  uint8
    data/<demo>/obs/edge_cam         shape (T, 128, 128, 3)  uint8
    data/<demo>/obs/state            shape (T, 20)            float32
    data/<demo>/obs/actions          shape (T, 6)             float32
    data/<demo>/actions              shape (T, 6)             float32  (raw)
    data/<demo>/states/...           nested articulation/rigid-object states

Usage
-----
    python scripts/tools/pusht_extract_video.py datasets/pusht_bimanual.hdf5
    python scripts/tools/pusht_extract_video.py datasets/pusht_bimanual.hdf5 --obs_key obs/edge_cam
    python scripts/tools/pusht_extract_video.py datasets/pusht_bimanual.hdf5 --fps 30 --out_dir videos/
    python scripts/tools/pusht_extract_video.py datasets/pusht_bimanual.hdf5 --demo demo_0
"""

import argparse
import os

import h5py
import imageio.v3 as iio
import numpy as np


# Default key — rendered RGB top-down camera.
# Switch to "obs/edge_cam" for the synthetic edge image.
OBS_KEY = "obs/fixed_cam"


def main():
    parser = argparse.ArgumentParser(description="Extract MP4 from Push-T HDF5 dataset.")
    parser.add_argument("dataset", help="Path to the HDF5 file.")
    parser.add_argument("--demo", default=None, help="Single demo to extract (e.g. demo_0). Omit for all.")
    parser.add_argument(
        "--obs_key",
        default=OBS_KEY,
        help=(
            f"HDF5 key for image frames (default: {OBS_KEY})."
            " Available: obs/fixed_cam (RGB), obs/edge_cam (synthetic edge w/ arm contours)."
        ),
    )
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate (default: 30).")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: same folder as dataset).")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.dataset))
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(args.dataset, "r") as f:
        demos = [args.demo] if args.demo else sorted(f["data"].keys())

        for name in demos:
            key = f"data/{name}/{args.obs_key}"
            if key not in f:
                print(f"  [skip] {key} not found.")
                continue

            frames = np.asarray(f[key], dtype=np.uint8)   # (T, H, W, 3)
            out_path = os.path.join(out_dir, f"{name}.mp4")
            iio.imwrite(out_path, frames, fps=args.fps, codec="libx264", pixelformat="yuv420p")
            print(f"  {name}: {len(frames)} frames → {out_path}")


if __name__ == "__main__":
    main()
