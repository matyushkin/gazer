#!/usr/bin/env python3
"""
Preprocess MPIIGaze dataset: convert .mat files to PNG images + TSV labels.

MPIIGaze Normalized data (Data/Normalized/p*/day*.mat) contains:
  - data.left.image   (N, 36, 60)  uint8 left eye patches
  - data.right.image  (N, 36, 60)  uint8 right eye patches
  - data.pose         (N, 3)       head pose
  - data.gaze         (N, 2)       gaze angles [theta, phi] in radians
    theta = pitch = asin(-y)  (positive = up)
    phi   = yaw   = atan2(-x, -z)  (positive = left)

Usage:
  python3 tools/preprocess_mpii.py ./MPIIGaze ./MPIIGaze_processed

Output:
  MPIIGaze_processed/
    p00/
      0000000_left.png   (36x60 grayscale eye patch)
      0000000_right.png
      labels.tsv         (idx yaw pitch head0 head1 head2)
    ...
"""

import sys
import os
import struct
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, using fallback HDF5 reader")
    import h5py


def load_mat(path):
    """Load MPIIGaze .mat file, returning a dict with arrays."""
    if HAS_SCIPY:
        try:
            mat = sio.loadmat(path)
            return mat
        except Exception:
            pass
    # Try HDF5 (MATLAB v7.3)
    import h5py
    with h5py.File(path, 'r') as f:
        return _hdf5_to_dict(f)


def _hdf5_to_dict(h5group):
    result = {}
    for key in h5group:
        val = h5group[key]
        if hasattr(val, 'keys'):
            result[key] = _hdf5_to_dict(val)
        else:
            result[key] = val[()]
    return result


def extract_data(mat):
    """
    Extract (left_imgs, right_imgs, gaze, pose) from the loaded mat dict.
    Handles both MATLAB v5 (scipy) and v7.3 (HDF5/h5py) layouts.

    MPIIGaze mat structure:
      data.left.image   shape (N,36,60) or (60,36,N) depending on format
      data.right.image
      data.pose         shape (N,3) or (3,N)
      data.gaze         shape (N,2) or (2,N)
    """
    d = mat.get('data', mat)

    # Handle scipy nested structured arrays
    if hasattr(d, 'dtype') and d.dtype.names:
        left  = d['left'][0,0]['image'][0,0]   # (36,60,N) or (N,36,60)
        right = d['right'][0,0]['image'][0,0]
        gaze  = d['gaze'][0,0]                  # (N,2)
        pose  = d['pose'][0,0]                  # (N,3)
    else:
        left  = d['left']['image']
        right = d['right']['image']
        gaze  = d['gaze']
        pose  = d['pose']

    # Normalise to (N, H, W)
    left  = np.array(left,  dtype=np.uint8)
    right = np.array(right, dtype=np.uint8)
    gaze  = np.array(gaze,  dtype=np.float32)
    pose  = np.array(pose,  dtype=np.float32)

    # MATLAB column-major → (W,H,N) or (H,W,N); transpose to (N,H,W)
    if left.ndim == 3 and left.shape[2] > 10:
        # shape is (H,W,N) → (N,H,W)
        left  = left.transpose(2,0,1)
        right = right.transpose(2,0,1)
    if left.ndim == 3 and left.shape[0] == 60:
        # shape is (W,H,N) → (N,H,W)
        left  = left.transpose(2,1,0)
        right = right.transpose(2,1,0)

    # Gaze/pose: (N,2) or (2,N)
    if gaze.ndim == 2 and gaze.shape[0] == 2 and gaze.shape[1] != 2:
        gaze = gaze.T
    if pose.ndim == 2 and pose.shape[0] == 3 and pose.shape[1] != 3:
        pose = pose.T

    N = gaze.shape[0]
    assert left.shape[0] == N, f"left shape {left.shape} vs N={N}"
    return left, right, gaze, pose


def process_subject(subject_dir: Path, out_dir: Path):
    """Process all days for one subject."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / "labels.tsv"

    total = 0
    with open(label_path, 'w') as lf:
        lf.write("idx\tyaw_rad\tpitch_rad\thead0\thead1\thead2\n")

        for mat_path in sorted(subject_dir.glob("day*.mat")):
            try:
                mat = load_mat(str(mat_path))
                left_imgs, right_imgs, gaze, pose = extract_data(mat)
            except Exception as e:
                print(f"  WARNING: skip {mat_path.name}: {e}", file=sys.stderr)
                continue

            N = gaze.shape[0]
            for i in range(N):
                idx = total + i

                # gaze: [theta, phi] where theta=pitch, phi=yaw (MPIIGaze convention)
                # theta = asin(-y), phi = atan2(-x, -z)
                # We want yaw (phi with sign flip) and pitch (theta with sign flip)
                # to match our convention: positive yaw = right, positive pitch = up
                theta = float(gaze[i, 0])   # pitch in MPIIGaze convention
                phi   = float(gaze[i, 1])   # yaw in MPIIGaze convention
                # MPIIGaze: theta=asin(-gy), phi=atan2(-gx,-gz)
                # Our convention: pitch=asin(-gy), yaw=atan2(gx,gz) = -phi
                yaw_rad   = -phi
                pitch_rad = theta  # same sign

                # Save eye patches
                left_png  = out_dir / f"{idx:07d}_left.png"
                right_png = out_dir / f"{idx:07d}_right.png"
                Image.fromarray(left_imgs[i]).save(str(left_png))
                Image.fromarray(right_imgs[i]).save(str(right_png))

                p0, p1, p2 = float(pose[i,0]), float(pose[i,1]), float(pose[i,2])
                lf.write(f"{idx}\t{yaw_rad:.6f}\t{pitch_rad:.6f}\t{p0:.4f}\t{p1:.4f}\t{p2:.4f}\n")

            total += N
            print(f"  {mat_path.name}: {N} samples (total {total})")

    return total


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <MPIIGaze_root> <output_dir>")
        sys.exit(1)

    root = Path(sys.argv[1])
    out  = Path(sys.argv[2])
    norm_dir = root / "Data" / "Normalized"

    if not norm_dir.exists():
        print(f"ERROR: {norm_dir} not found. Is this the correct MPIIGaze root?")
        sys.exit(1)

    print(f"MPIIGaze preprocessor: {norm_dir} → {out}")

    grand_total = 0
    for subject_dir in sorted(norm_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        out_subject = out / subject
        print(f"\n{subject}:")
        n = process_subject(subject_dir, out_subject)
        grand_total += n

    print(f"\nDone. Total samples: {grand_total}")
    print(f"Output: {out}")
    print(f"\nNow run: cargo run --release --example mpii_bench -- {out}")


if __name__ == "__main__":
    main()
