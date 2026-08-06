"""Part 2 of the class-generalization test (.claude/plans/snazzy-cuddling-creek.md): does the
instance-scoping + loss-weighting mechanism that worked for SFX (notes/instance_aware_pivot_
2026-08-03.md) also help the original motivating defect, bubble/cloud halo?

Mirrors build_sfx_instance_crops.py's structure, but reuses the EXISTING, already-validated
bubble/cloud contour detector (_find_bubble_interior_holes, ml_cleaner.py -- the same
flood-fill-from-corner technique close_bubble_halo/apply_halo_refine use) instead of a new
ink-stroke heuristic -- bubbles already have a real detector, unlike SFX last night.

Background-crop loss-weighting reuses last night's PROVEN with_bg_weighted approach directly
(0.2x loss weight) rather than re-running the 2 already-failed data-ratio attempts -- per the
plan's explicit instruction. instance_sfx_net.py's TinyInstanceNet/dice_bce are reused as-is
(architecture-agnostic to content type), just retrained on bubble crops.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_cleaner import GuidanceParams, build_input_tensor, _find_bubble_interior_holes  # noqa: E402
import cv2  # noqa: E402
from build_sfx_instance_crops import crop_and_pad, sample_background_crops  # noqa: E402

DATASET_ROOT = ROOT / ".tmp/datasets/b2_bubbles_2k_prestage"
OUT_DIR = ROOT / ".tmp/checkpoints/instance_bubble_smoke"
CROP_SIZE = 224
MARGIN = 96
FRAME_DARKNESS = 40
MIN_BUBBLE_AREA = 2000  # matches apply_halo_refine's own default
MAX_INSTANCE_FRAC = 0.5  # bubbles are legitimately much bigger than SFX glyphs; loosen vs SFX's 0.15


def find_bubble_instances(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Thin wrapper around the existing, already-validated bubble/cloud detector -- returns
    (x, y, w, h) boxes, matching find_sfx_instances' return shape for reuse downstream."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    page_area = rgb.shape[0] * rgb.shape[1]
    holes = _find_bubble_interior_holes(gray, FRAME_DARKNESS, MIN_BUBBLE_AREA)
    boxes = []
    for hole in holes:
        x, y, w, h = hole["bbox"]
        if w * h > MAX_INSTANCE_FRAC * page_area:
            continue
        boxes.append((x, y, w, h))
    return boxes


def build_split(
    split_dir: Path, limit: int | None = None, bg_loss_weight: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bubbles_dir = split_dir / "ep1/bubbles"
    cleaned_dir = split_dir / "ep1/bubbles_cleaned"
    paths = sorted(bubbles_dir.iterdir())
    if limit is not None:
        paths = paths[:limit]

    gp = GuidanceParams()
    rng = np.random.default_rng(0)
    inputs, targets, is_bg = [], [], []
    for p in paths:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        alpha = np.asarray(Image.open(cleaned_dir / p.name))[..., -1]
        delete_mask = (alpha < 128)

        boxes = find_bubble_instances(rgb)
        input_tensor = build_input_tensor(rgb, gp)

        for (x, y, bw, bh) in boxes:
            cx, cy = x + bw // 2, y + bh // 2
            if max(bw, bh) + 2 * MARGIN > 4 * CROP_SIZE:
                continue
            inp_crop = crop_and_pad(input_tensor, cx, cy, CROP_SIZE)
            tgt_crop = crop_and_pad(delete_mask.astype(np.float32), cx, cy, CROP_SIZE)
            inputs.append(inp_crop)
            targets.append(tgt_crop)
            is_bg.append(False)

        for inp_crop, tgt_crop in sample_background_crops(rng, delete_mask, input_tensor, 1):
            inputs.append(inp_crop)
            targets.append(tgt_crop)
            is_bg.append(True)

    if not inputs:
        return (np.zeros((0, CROP_SIZE, CROP_SIZE, 7), np.float32), np.zeros((0, CROP_SIZE, CROP_SIZE), np.float32),
                np.zeros((0,), bool))
    return np.stack(inputs), np.stack(targets), np.array(is_bg, dtype=bool)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Bubble pages are much taller (695x6000, multi-panel strips) than SFX pages (910x1619) --
    # each page likely contains many bubble instances, so fewer source pages are needed for a
    # comparable crop count. Start with a modest page limit and report actual crop counts.
    train_x, train_y, train_bg = build_split(DATASET_ROOT / "train_root", limit=500)
    val_x, val_y, val_bg = build_split(DATASET_ROOT / "val_root", limit=120)

    print(f"train crops: {train_x.shape[0]} (from up to 500 source pages)")
    print(f"val crops:   {val_x.shape[0]} (from up to 120 source pages)")
    print(f"train target keep-fraction (mean): {1.0 - train_y.mean():.4f}")
    print(f"train bg-crop fraction: {train_bg.mean():.4f}")

    np.savez_compressed(OUT_DIR / "crops_train_bubble.npz", x=train_x, y=train_y, is_bg=train_bg)
    np.savez_compressed(OUT_DIR / "crops_val_bubble.npz", x=val_x, y=val_y, is_bg=val_bg)
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
