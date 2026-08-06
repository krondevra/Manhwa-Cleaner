"""Part 1 step 1 of the instance-aware architecture pivot (see
.claude/plans/snazzy-cuddling-creek.md -- not in this repo, referenced for context only;
see notes/synthetic_curriculum_plan.md for the project-visible record).

Extracts (local crop, local target mask) pairs for individual SFX instances from the
already-generated `stage3_sfx_2k` synthetic pool (`.tmp/datasets/stage3_sfx_2k/`), reusing
that dataset rather than regenerating anything, and touching no PepperNCarrotDataset code.

No per-instance metadata sidecar was saved at generation time (confirmed: no .json next to
the PNGs), and SFX-on-blank-interior instances are NOT separable from the surrounding "keep"
region via the cleaned alpha mask alone (both are alpha=255 -- that lack of a mask-level
boundary is exactly the mechanism behind the ch1_sfx_text defect this experiment targets).
Instance location instead comes from the RGB's own dark ink strokes (same family of
technique as ml_cleaner.py::repair_frame_interiors / style_analysis.py::extract_enclosed_holes),
with panel/frame border strokes excluded by a bounding-box-fraction heuristic (a stroke
spanning >=40% of the page in either dimension is a frame line, not a glyph) and per-letter
strokes merged into one instance via a wide dilation before the final connected-components
pass -- a heuristic proposal generator good enough for this proof-of-mechanism (we authored
this synthetic data ourselves, so approximate self-supervised localization is legitimate,
unlike on real pages where recall would need to be validated against something else).

Crop margin (96px each side) is chosen from the occlusion-probe finding in
docs/ml_strategy_history.md (2026-07-31 deep diagnosis): the whole-page model is still
CORRECT at context radius 64px around an SFX glyph and only flips wrong once radius >=128px
admits the surrounding blank page. A 96px margin sits inside the "still correct" band, so a
locally-cropped model structurally cannot see the R>=128 trigger.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ml_cleaner import GuidanceParams, build_input_tensor  # noqa: E402

DATASET_ROOT = ROOT / ".tmp/datasets/stage3_sfx_2k"
OUT_DIR = ROOT / ".tmp/checkpoints/instance_sfx_smoke"
CROP_SIZE = 224  # fixed output size crops are padded/cropped to, for simple batching
MARGIN = 96
FRAME_DARKNESS = 40
FRAME_BBOX_FRAC = 0.40  # stroke components spanning this much of the page = frame/panel lines
MERGE_DILATE_PX = 31  # merges per-letter/word ink strokes into one instance blob
MIN_INK_AREA = 20
MAX_INSTANCE_FRAC = 0.15  # reject accidentally-merged giant blobs (page fraction)


def find_sfx_instances(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Returns a list of (x, y, w, h) tight ink-bounding-boxes, one per detected SFX instance."""
    h, w = rgb.shape[:2]
    page_area = h * w
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    stroke = (gray <= FRAME_DARKNESS).astype(np.uint8)

    # Vectorized label filtering (NOT a per-label python loop over the full image -- on a
    # real ~50k-tall page with thousands of small dark components, "labels == lbl" scanning
    # the whole array per label was catastrophically slow, confirmed: >3.5min and still not
    # done on a single 690x53589px chapter). np.isin against the small valid-label set is a
    # single vectorized pass.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(stroke, connectivity=8)
    ws, hs, areas = stats[:, 2], stats[:, 3], stats[:, 4]
    valid = (ws < FRAME_BBOX_FRAC * w) & (hs < FRAME_BBOX_FRAC * h) & (areas >= 3)
    valid[0] = False  # label 0 is background
    valid_labels = np.nonzero(valid)[0]
    if valid_labels.size == 0:
        return []
    candidate = np.isin(labels, valid_labels).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MERGE_DILATE_PX, MERGE_DILATE_PX))
    merged = cv2.dilate(candidate, k, iterations=1)
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)

    boxes = []
    for lbl in range(1, n2):
        # Per-component work inside each component's own bounding-box crop (same pattern as
        # ml_cleaner.py::repair_frame_interiors), not the full page -- bounded cost per
        # instance regardless of page height.
        bx, by, bw, bh, _ = stats2[lbl]
        sub_candidate = candidate[by : by + bh, bx : bx + bw]
        sub_labels2 = labels2[by : by + bh, bx : bx + bw]
        # tight bbox from the ORIGINAL (undilated) ink within this merged blob, not the
        # dilated blob itself, so the crop center/size reflects the real glyph extent.
        ink_here = sub_candidate & (sub_labels2 == lbl)
        area = int(ink_here.sum())
        if area < MIN_INK_AREA or area > MAX_INSTANCE_FRAC * page_area:
            continue
        ys, xs = np.where(ink_here)
        x0, x1, y0, y1 = xs.min() + bx, xs.max() + bx, ys.min() + by, ys.max() + by
        boxes.append((int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)))
    return boxes


def crop_and_pad(arr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    """Crop a size x size window centered at (cx, cy), zero-padding past the page edge."""
    h, w = arr.shape[:2]
    half = size // 2
    x0, y0 = cx - half, cy - half
    out_shape = (size, size) + arr.shape[2:]
    out = np.zeros(out_shape, dtype=arr.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return out


def sample_background_crops(
    rng: np.random.Generator, delete_mask: np.ndarray, input_tensor: np.ndarray, n: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pure/mostly-background negative crops, sampled away from any detected ink instance.
    Added after the first smoke check showed the ink-glyph-centered crop distribution
    under-represents plain background (0.20 mean delete-prob on a synthetic all-white page,
    vs. the >0.5 that should hold for confidently-delete content) -- 46% of glyph-centered
    crops had zero delete pixels at all, so the model saw few genuinely background-dominant
    examples. This is a calibration refinement of the same crop-based approach, not a new
    mechanism."""
    h, w = delete_mask.shape
    out = []
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        if not delete_mask[cy, cx]:
            continue
        tgt_crop = crop_and_pad(delete_mask.astype(np.float32), cx, cy, CROP_SIZE)
        if tgt_crop.mean() < 0.6:  # want genuinely background-dominant crops
            continue
        inp_crop = crop_and_pad(input_tensor, cx, cy, CROP_SIZE)
        out.append((inp_crop, tgt_crop))
    return out


def build_split(
    split_dir: Path, limit: int | None = None, bg_ratio: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """bg_ratio: probability per page of adding 1 background-only negative crop (0.0-1.0).
    1.0 matches the original with_bg variant (every page); a fractional value tests whether a
    lighter mixture can fix blank-crop calibration without overcorrecting real-instance recall
    the way the full 1.0 ratio did."""
    sfx_dir = split_dir / "ep1/sfx"
    cleaned_dir = split_dir / "ep1/sfx_cleaned"
    paths = sorted(sfx_dir.iterdir())
    if limit is not None:
        paths = paths[:limit]

    gp = GuidanceParams()
    rng = np.random.default_rng(0)
    inputs, targets, is_bg = [], [], []
    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        alpha = np.asarray(Image.open(cleaned_dir / p.name))[..., -1]
        delete_mask = (alpha < 128)

        boxes = find_sfx_instances(rgb)
        input_tensor = build_input_tensor(rgb, gp)  # (H, W, 7)

        for (x, y, bw, bh) in boxes:
            cx, cy = x + bw // 2, y + bh // 2
            # only keep instances whose padded crop fits the "still correct" occlusion band --
            # reject glyphs so large the 96px margin wouldn't leave room to matter.
            if max(bw, bh) + 2 * MARGIN > 4 * CROP_SIZE:
                continue
            inp_crop = crop_and_pad(input_tensor, cx, cy, CROP_SIZE)
            tgt_crop = crop_and_pad(delete_mask.astype(np.float32), cx, cy, CROP_SIZE)
            inputs.append(inp_crop)
            targets.append(tgt_crop)
            is_bg.append(False)

        n_bg = 1 if rng.random() < bg_ratio else 0
        for inp_crop, tgt_crop in sample_background_crops(rng, delete_mask, input_tensor, n_bg):
            inputs.append(inp_crop)
            targets.append(tgt_crop)
            is_bg.append(True)

    if not inputs:
        return (np.zeros((0, CROP_SIZE, CROP_SIZE, 7), np.float32), np.zeros((0, CROP_SIZE, CROP_SIZE), np.float32),
                np.zeros((0,), bool))
    return np.stack(inputs), np.stack(targets), np.array(is_bg, dtype=bool)


VARIANT_BG_RATIO = {"glyph_only": 0.0, "with_bg": 1.0, "with_bg_light": 0.25, "with_bg_weighted": 1.0}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_BG_RATIO),
                    help="glyph_only: attempt-1 crops (glyph-centered only). "
                         "with_bg: attempt-1-refinement, adds a background crop on every page. "
                         "with_bg_light: lighter mixture (quarter of pages). "
                         "with_bg_weighted: full background crops (same data as with_bg), but "
                         "the training script downweights their loss contribution instead of "
                         "their data quantity -- ratio-tuning and post-hoc recalibration both "
                         "failed (see notes/instance_aware_pivot_2026-08-03.md), this tests a "
                         "loss-side fix instead.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bg_ratio = VARIANT_BG_RATIO[args.variant]

    train_x, train_y, train_bg = build_split(DATASET_ROOT / "train_root", limit=300, bg_ratio=bg_ratio)
    val_x, val_y, val_bg = build_split(DATASET_ROOT / "val_root", limit=80, bg_ratio=bg_ratio)

    print(f"[{args.variant}] train crops: {train_x.shape[0]} (from up to 300 source pages)")
    print(f"[{args.variant}] val crops:   {val_x.shape[0]} (from up to 80 source pages)")
    print(f"[{args.variant}] train target keep-fraction (mean): {1.0 - train_y.mean():.4f}")

    np.savez_compressed(OUT_DIR / f"crops_train_{args.variant}.npz", x=train_x, y=train_y, is_bg=train_bg)
    np.savez_compressed(OUT_DIR / f"crops_val_{args.variant}.npz", x=val_x, y=val_y, is_bg=val_bg)
    print(f"saved to {OUT_DIR} (variant={args.variant})")


if __name__ == "__main__":
    main()
