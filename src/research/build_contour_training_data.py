"""Attempt 7: builds (crop, init_radii, true_radii) tuples for the Deep-Snake-style contour
deformation network, from the same synthetic bubble pool build_bubble_instance_crops.py already
uses (b2_bubbles_2k_prestage), reusing _find_bubble_interior_holes (ml_cleaner.py) for instance
location/contour -- exact ground truth is available procedurally, no new generation needed.

CROP_SIZE=320 (up from the dense instance-net's 224) per Part 0's measured finding that model
2.0's real training crops carry substantially more background-margin context (median 37.6%
background fraction, background-band at top/bottom in >85% of crops) than a tight per-object
crop provides -- informs the crop scale, not a copy of any real content.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_cleaner import GuidanceParams, build_input_tensor, _find_bubble_interior_holes  # noqa: E402
from contour_common import build_angles, ray_radius_mask, fit_init_ellipse_radii  # noqa: E402

DATASET_ROOT = ROOT / ".tmp/datasets/b2_bubbles_2k_prestage"
OUT_DIR = ROOT / ".tmp/checkpoints/contour_deform_smoke"
CROP_SIZE = 512
N_VERTICES = 64
FRAME_DARKNESS = 40
MIN_BUBBLE_AREA = 2000
MAX_R = CROP_SIZE // 2 - 5
MAX_BBOX_DIAG_FRAC = 0.75  # skip instances whose bbox diagonal is too large relative to CROP_SIZE
# 2026-08-04 scale-bug fix: the original CROP_SIZE=320/MAX_BBOX_DIAG_FRAC=0.55 (max accepted
# diag=176px) excluded 40% of ALL bubbles in the synthetic pool -- specifically the larger half
# (measured full-pool bbox diagonal: mean=201px, median=165.5px, p90=299px, vs the 176px cutoff).
# A real instance visual check (inst1) showed the trained network's deformed contour staying far
# inside the true (much larger) bubble boundary -- root-caused to this training-population bias,
# not a mechanism failure. CROP_SIZE=512/MAX_BBOX_DIAG_FRAC=0.75 (max accepted diag=384px) covers
# up to roughly the real p90-p95 range instead of truncating at the low end of the distribution.


def crop_and_pad(arr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    half = size // 2
    x0, y0 = cx - half, cy - half
    out = np.zeros((size, size) + arr.shape[2:], dtype=arr.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return out


def build_examples(split_dir: Path, limit_pages: int, limit_examples: int) -> dict:
    bubbles_dir = split_dir / "ep1/bubbles"
    paths = sorted(bubbles_dir.iterdir())[:limit_pages]
    angles = build_angles(N_VERTICES)

    crops, init_radii_list, true_radii_list = [], [], []
    gp = GuidanceParams()

    for p in paths:
        if len(crops) >= limit_examples:
            break
        rgb = np.asarray(Image.open(p).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        holes = _find_bubble_interior_holes(gray, FRAME_DARKNESS, MIN_BUBBLE_AREA)
        if not holes:
            continue
        input_tensor = build_input_tensor(rgb, gp)
        h, w = gray.shape

        for hole in holes:
            if len(crops) >= limit_examples:
                break
            x, y, bw, bh = hole["bbox"]
            diag = float(np.hypot(bw, bh))
            if diag > MAX_BBOX_DIAG_FRAC * CROP_SIZE:
                continue

            # exact instance mask from the true contour (procedural ground truth, not detection)
            inst_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(inst_mask, [hole["contour"]], -1, 1, thickness=-1)
            inst_mask_bool = inst_mask.astype(bool)

            m = cv2.moments(hole["contour"])
            if m["m00"] == 0:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]

            true_radii = np.array([ray_radius_mask(inst_mask_bool, cx, cy, a, MAX_R) for a in angles])
            if (true_radii <= 1.0).any():
                continue  # degenerate instance (centroid too close to an edge), skip

            init_radii = fit_init_ellipse_radii(angles, bw, bh)

            crop = crop_and_pad(input_tensor, int(round(cx)), int(round(cy)), CROP_SIZE)
            crops.append(crop)
            init_radii_list.append(init_radii)
            true_radii_list.append(true_radii)

    return {
        "crops": np.stack(crops) if crops else np.zeros((0, CROP_SIZE, CROP_SIZE, 7), np.float32),
        "init_radii": np.stack(init_radii_list) if init_radii_list else np.zeros((0, N_VERTICES), np.float32),
        "true_radii": np.stack(true_radii_list) if true_radii_list else np.zeros((0, N_VERTICES), np.float32),
        "angles": angles,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-examples", type=int, default=250,
                     help="cheapest-check default: 250, per the plan's Part 1 step 3")
    ap.add_argument("--limit-pages", type=int, default=200)
    ap.add_argument("--val-examples", type=int, default=60)
    ap.add_argument("--val-pages", type=int, default=60)
    ap.add_argument("--tag", type=str, default="smoke")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = build_examples(DATASET_ROOT / "train_root", args.limit_pages, args.limit_examples)
    val = build_examples(DATASET_ROOT / "val_root", args.val_pages, args.val_examples)

    print(f"[{args.tag}] train examples: {train['crops'].shape[0]} (target {args.limit_examples})")
    print(f"[{args.tag}] val examples:   {val['crops'].shape[0]} (target {args.val_examples})")
    resid = train["true_radii"] - train["init_radii"]
    print(f"[{args.tag}] train radial residual (true-init): mean={resid.mean():.2f}px "
          f"std={resid.std():.2f}px min={resid.min():.2f} max={resid.max():.2f}")

    np.savez_compressed(OUT_DIR / f"train_{args.tag}.npz", **train)
    np.savez_compressed(OUT_DIR / f"val_{args.tag}.npz", **val)
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
