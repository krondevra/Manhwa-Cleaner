#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# ==================================================
# DEFAULT SETTINGS
# ==================================================

ALPHA_THRESHOLD = 10

# How many visible/content pixels must exist in a row
# to treat this row as part of useful content.
MIN_ROW_PIXELS = 5

# Merge vertical regions if the empty gap between them is small.
# Lower value = smaller cuts.
MERGE_GAP = 60

# Extra context above and below each detected region.
PADDING_TOP = 120
PADDING_BOTTOM = 120

# Ignore accidental tiny fragments.
MIN_SEGMENT_HEIGHT = 40

# If padded crops overlap, merge them.
MERGE_PADDED_OVERLAPS = True

# Optional safety split.
# 0 = disabled.
# If enabled, very tall crops are split into smaller overlapping chunks.
MAX_CROP_HEIGHT = 0
MAX_CROP_OVERLAP = 160

# ==================================================


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def get_content_mask(cleaned: Image.Image) -> np.ndarray:
    """
    Detect non-background pixels from cleaned image.

    Main case:
    - cleaned PNG has transparency
    - alpha > threshold means useful content

    Fallback:
    - if no real alpha exists, detect non-white pixels
    """
    arr = np.asarray(cleaned, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    has_transparency = alpha.min() < 250

    if has_transparency:
        return alpha > ALPHA_THRESHOLD

    # Fallback for non-alpha images.
    not_white = np.any(rgb < 245, axis=2)

    max_c = rgb.max(axis=2).astype(np.int16)
    min_c = rgb.min(axis=2).astype(np.int16)
    saturation_like = (max_c - min_c) > 12

    return not_white | saturation_like


def rows_to_runs(active_rows: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert active row boolean array into inclusive-exclusive runs.
    """
    ys = np.where(active_rows)[0]

    if len(ys) == 0:
        return []

    runs = []
    start = ys[0]
    prev = ys[0]

    for y in ys[1:]:
        if y == prev + 1:
            prev = y
        else:
            runs.append((start, prev + 1))
            start = y
            prev = y

    runs.append((start, prev + 1))
    return runs


def merge_close_runs(
    runs: list[tuple[int, int]],
    max_gap: int,
) -> list[tuple[int, int]]:
    """
    Merge runs if vertical gap between them is <= max_gap.
    """
    if not runs:
        return []

    merged = [runs[0]]

    for y0, y1 in runs[1:]:
        last_y0, last_y1 = merged[-1]
        gap = y0 - last_y1

        if gap <= max_gap:
            merged[-1] = (last_y0, y1)
        else:
            merged.append((y0, y1))

    return merged


def apply_padding(
    segments: list[tuple[int, int]],
    image_height: int,
    pad_top: int,
    pad_bottom: int,
) -> list[tuple[int, int]]:
    padded = []

    for y0, y1 in segments:
        if y1 - y0 < MIN_SEGMENT_HEIGHT:
            continue

        py0 = max(0, y0 - pad_top)
        py1 = min(image_height, y1 + pad_bottom)

        padded.append((py0, py1))

    return padded


def merge_overlapping_segments(
    segments: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not segments:
        return []

    segments = sorted(segments)
    merged = [segments[0]]

    for y0, y1 in segments[1:]:
        last_y0, last_y1 = merged[-1]

        if y0 <= last_y1:
            merged[-1] = (last_y0, max(last_y1, y1))
        else:
            merged.append((y0, y1))

    return merged


def split_tall_segments(
    segments: list[tuple[int, int]],
    max_height: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """
    Optional split for very tall segments.
    Disabled when max_height <= 0.
    """
    if max_height <= 0:
        return segments

    result = []

    for y0, y1 in segments:
        height = y1 - y0

        if height <= max_height:
            result.append((y0, y1))
            continue

        step = max(1, max_height - overlap)
        start = y0

        while start < y1:
            end = min(start + max_height, y1)

            if end - start >= MIN_SEGMENT_HEIGHT:
                result.append((start, end))

            if end >= y1:
                break

            start += step

    return result


def save_crop_pair(
    original: Image.Image,
    cleaned: Image.Image,
    y0: int,
    y1: int,
    out_dir: Path,
    base_name: str,
    index: int,
) -> None:
    width, _ = original.size

    original_crop = original.crop((0, y0, width, y1))
    cleaned_crop = cleaned.crop((0, y0, width, y1))

    name = f"{base_name}-{index:03d}"

    original_crop.save(out_dir / f"{name}.png")
    cleaned_crop.save(out_dir / f"{name}_cleaned.png")


def split_pair(
    base_name: str,
    padding_top: int,
    padding_bottom: int,
    merge_gap: int,
    min_row_pixels: int,
    max_crop_height: int,
    max_crop_overlap: int,
    clear_output: bool,
) -> None:
    original_path = Path(f"{base_name}.png")
    cleaned_path = Path(f"{base_name}_cleaned.png")

    if not original_path.exists():
        raise FileNotFoundError(f"Original not found: {original_path}")

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned not found: {cleaned_path}")

    original = load_rgba(original_path)
    cleaned = load_rgba(cleaned_path)

    if original.size != cleaned.size:
        raise ValueError(
            f"Size mismatch: {original_path.name} {original.size} "
            f"vs {cleaned_path.name} {cleaned.size}"
        )

    width, height = original.size

    print(f"Original: {original_path}")
    print(f"Cleaned:  {cleaned_path}")
    print(f"Size:     {width}x{height}")

    content_mask = get_content_mask(cleaned)

    row_counts = content_mask.sum(axis=1)
    active_rows = row_counts >= min_row_pixels

    runs = rows_to_runs(active_rows)
    segments = merge_close_runs(runs, merge_gap)

    segments = apply_padding(
        segments,
        height,
        padding_top,
        padding_bottom,
    )

    if MERGE_PADDED_OVERLAPS:
        segments = merge_overlapping_segments(segments)

    segments = split_tall_segments(
        segments,
        max_crop_height,
        max_crop_overlap,
    )

    out_dir = Path(f"{base_name}_frames")

    if clear_output and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output:   {out_dir}")
    print(f"Segments: {len(segments)}")
    print(
        f"Settings: pad_top={padding_top}, pad_bottom={padding_bottom}, "
        f"merge_gap={merge_gap}, min_row_pixels={min_row_pixels}, "
        f"max_crop_height={max_crop_height}"
    )

    for i, (y0, y1) in enumerate(segments):
        save_crop_pair(
            original,
            cleaned,
            y0,
            y1,
            out_dir,
            base_name,
            i,
        )

        print(f"[{i:03d}] y={y0}-{y1}, h={y1 - y0}")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split original + cleaned long manhwa image into contextual sample crops."
    )

    parser.add_argument(
        "base_name",
        help="Base name without extension. Example: 012-1",
    )

    parser.add_argument(
        "--pad-top",
        type=int,
        default=PADDING_TOP,
        help=f"Extra pixels above each crop. Default: {PADDING_TOP}",
    )

    parser.add_argument(
        "--pad-bottom",
        type=int,
        default=PADDING_BOTTOM,
        help=f"Extra pixels below each crop. Default: {PADDING_BOTTOM}",
    )

    parser.add_argument(
        "--merge-gap",
        type=int,
        default=MERGE_GAP,
        help=f"Merge regions separated by this vertical gap. Default: {MERGE_GAP}",
    )

    parser.add_argument(
        "--min-row-pixels",
        type=int,
        default=MIN_ROW_PIXELS,
        help=f"Minimum content pixels in a row. Default: {MIN_ROW_PIXELS}",
    )

    parser.add_argument(
        "--max-height",
        type=int,
        default=MAX_CROP_HEIGHT,
        help="Optional max crop height. 0 disables forced splitting.",
    )

    parser.add_argument(
        "--max-overlap",
        type=int,
        default=MAX_CROP_OVERLAP,
        help=f"Overlap when --max-height is used. Default: {MAX_CROP_OVERLAP}",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete output folder before writing new crops.",
    )

    args = parser.parse_args()

    split_pair(
        base_name=args.base_name,
        padding_top=args.pad_top,
        padding_bottom=args.pad_bottom,
        merge_gap=args.merge_gap,
        min_row_pixels=args.min_row_pixels,
        max_crop_height=args.max_height,
        max_crop_overlap=args.max_overlap,
        clear_output=args.clear,
    )


if __name__ == "__main__":
    main()
