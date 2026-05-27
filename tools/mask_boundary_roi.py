#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


# ==================================================
# Data
# ==================================================

BLACK_LABEL = "BLACK"
WHITE_LABEL = "WHITE"


@dataclass
class ROI:
    x0: int
    y0: int
    x1: int
    y1: int
    label: str
    score: float


# ==================================================
# Loading
# ==================================================

def resolve_paths(input_arg: str, cleaned_arg: str | None = None) -> tuple[Path, Path, str, Path]:
    """
    Resolve original + cleaned paths for the current project layout.

    Examples:
        python tools/mask_boundary_roi.py 033
        python tools/mask_boundary_roi.py 033.png
        python tools/mask_boundary_roi.py data/chapters-long/033.png

    Defaults:
        original: data/chapters-long/<CH>.png
        cleaned:  data/temp/<CH>/<CH>_cleaned.png
        temp dir: data/temp/<CH>
    """
    p = Path(input_arg)

    if p.exists():
        original_path = p
        stem = p.stem
    elif p.suffix.lower() == ".png":
        stem = p.stem
        original_path = Path("data") / "chapters-long" / p.name
    else:
        stem = input_arg
        original_path = Path("data") / "chapters-long" / f"{stem}.png"

    temp_dir = Path("data") / "temp" / stem

    if cleaned_arg:
        cleaned_path = Path(cleaned_arg)
    else:
        cleaned_path = temp_dir / f"{stem}_cleaned.png"

    if not original_path.exists():
        raise FileNotFoundError(
            f"Original image not found: {original_path}\n"
            f"Run from project root or pass an explicit original path."
        )

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned image not found: {cleaned_path}\n"
            f"Expected cleaned chapter at: data/temp/{stem}/{stem}_cleaned.png"
        )

    return original_path, cleaned_path, stem, temp_dir


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def crop_to_same_size(
    original_rgb: np.ndarray,
    cleaned_rgba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h = min(original_rgb.shape[0], cleaned_rgba.shape[0])
    w = min(original_rgb.shape[1], cleaned_rgba.shape[1])

    return original_rgb[:h, :w], cleaned_rgba[:h, :w]


# ==================================================
# Deleted / kept mask detection
# ==================================================

def infer_deleted_mask(
    original_rgb: np.ndarray,
    cleaned_rgba: np.ndarray,
    alpha_threshold: int,
    fallback_mode: str,
) -> np.ndarray:
    """
    Best case:
    - cleaned has real alpha
    - alpha < threshold means deleted background

    Fallback modes are for RGB-only cleaned images.
    """

    alpha = cleaned_rgba[:, :, 3]

    if alpha.min() < 250:
        return alpha < alpha_threshold

    cleaned_rgb = cleaned_rgba[:, :, :3]

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    cleaned_gray = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2GRAY)

    if fallback_mode == "white":
        return cleaned_gray > 245

    if fallback_mode == "black":
        return cleaned_gray < 10

    if fallback_mode == "red":
        r = cleaned_rgb[:, :, 0]
        g = cleaned_rgb[:, :, 1]
        b = cleaned_rgb[:, :, 2]
        return (r > 220) & (g < 40) & (b < 40)

    if fallback_mode == "diff":
        diff = np.abs(original_rgb.astype(np.int16) - cleaned_rgb.astype(np.int16)).mean(axis=2)
        return diff > 20

    # auto fallback:
    # Prefer strong difference if no alpha exists.
    diff = np.abs(original_rgb.astype(np.int16) - cleaned_rgb.astype(np.int16)).mean(axis=2)

    red = (
        (cleaned_rgb[:, :, 0] > 220)
        & (cleaned_rgb[:, :, 1] < 40)
        & (cleaned_rgb[:, :, 2] < 40)
    )

    white = cleaned_gray > 248
    black = cleaned_gray < 5

    if red.mean() > 0.01:
        return red

    if diff.mean() > 2.0:
        return diff > 20

    # Last fallback. Not ideal.
    # If cleaned has no alpha and no visible difference, the script cannot know
    # what was deleted.
    return white | black


# ==================================================
# Integral image helpers
# ==================================================

def integral(mask: np.ndarray) -> np.ndarray:
    return cv2.integral(mask.astype(np.uint8))


def rect_sum(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> int:
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])


def rect_mean(ii: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    area = max(1, (x1 - x0) * (y1 - y0))
    return rect_sum(ii, x0, y0, x1, y1) / area


# ==================================================
# Boundary ROI generation
# ==================================================

def build_boundary_bands(
    deleted_mask: np.ndarray,
    band_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    deleted_mask:
        True  = removed external background
        False = kept content/frame

    black_band:
        deleted pixels close to kept content

    white_band:
        kept pixels close to deleted background

    boundary:
        union of both, useful for preview/debug
    """

    kept_mask = ~deleted_mask

    k = band_radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    dilated_deleted = cv2.dilate(deleted_mask.astype(np.uint8), kernel, iterations=1) > 0
    dilated_kept = cv2.dilate(kept_mask.astype(np.uint8), kernel, iterations=1) > 0

    black_band = deleted_mask & dilated_kept
    white_band = kept_mask & dilated_deleted

    boundary = black_band | white_band

    return black_band, white_band, boundary


def has_enough_texture_or_edge(
    gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_std: float,
) -> bool:
    patch = gray[y0:y1, x0:x1]

    if patch.size == 0:
        return False

    return float(patch.std()) >= min_std


def generate_candidates_for_label(
    label_mask: np.ndarray,
    boundary_band: np.ndarray,
    gray: np.ndarray,
    label: str,
    roi_w: int,
    roi_h: int,
    scan_step: int,
    purity: float,
    min_band_ratio: float,
    min_std: float,
) -> list[ROI]:
    h, w = label_mask.shape[:2]

    label_ii = integral(label_mask)
    band_ii = integral(boundary_band)

    candidates: list[ROI] = []

    half_w = roi_w // 2
    half_h = roi_h // 2

    ys = range(0, max(1, h - roi_h), scan_step)
    xs = range(0, max(1, w - roi_w), max(2, scan_step // 2))

    for y0 in ys:
        y1 = min(h, y0 + roi_h)

        if y1 - y0 < max(4, roi_h // 2):
            continue

        for x0 in xs:
            x1 = min(w, x0 + roi_w)

            if x1 - x0 < max(4, roi_w // 2):
                continue

            label_ratio = rect_mean(label_ii, x0, y0, x1, y1)

            if label_ratio < purity:
                continue

            band_ratio = rect_mean(band_ii, x0, y0, x1, y1)

            if band_ratio < min_band_ratio:
                continue

            # For WHITE/keep regions, avoid huge flat speech-bubble interiors.
            # We mainly want boundary-critical content, frame edges, SFX, dark content.
            if label == WHITE_LABEL and min_std > 0:
                if not has_enough_texture_or_edge(gray, x0, y0, x1, y1, min_std):
                    continue

            score = band_ratio * 2.0 + label_ratio

            candidates.append(
                ROI(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    label=label,
                    score=score,
                )
            )

    return candidates


def roi_overlap(a: ROI, b: ROI) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)

    inter = max(0, x1 - x0) * max(0, y1 - y0)

    if inter <= 0:
        return 0.0

    area_a = max(1, (a.x1 - a.x0) * (a.y1 - a.y0))
    area_b = max(1, (b.x1 - b.x0) * (b.y1 - b.y0))

    return inter / min(area_a, area_b)


def select_distributed_rois(
    candidates: list[ROI],
    max_rois: int,
    y_bin_size: int,
    max_per_bin: int,
    max_overlap: float,
) -> list[ROI]:
    """
    Select strong ROIs, but distribute them vertically so that all frames/areas
    get represented, not only one difficult region.
    """

    candidates = sorted(candidates, key=lambda r: r.score, reverse=True)

    selected: list[ROI] = []
    per_bin: dict[int, int] = {}

    for roi in candidates:
        y_mid = (roi.y0 + roi.y1) // 2
        bin_id = y_mid // y_bin_size

        if per_bin.get(bin_id, 0) >= max_per_bin:
            continue

        too_close = False

        for existing in selected:
            if roi_overlap(roi, existing) > max_overlap:
                too_close = True
                break

        if too_close:
            continue

        selected.append(roi)
        per_bin[bin_id] = per_bin.get(bin_id, 0) + 1

        if len(selected) >= max_rois:
            break

    selected.sort(key=lambda r: (r.y0, r.x0, r.label))

    return selected


def generate_boundary_rois(
    original_rgb: np.ndarray,
    deleted_mask: np.ndarray,
    roi_w: int,
    roi_h: int,
    scan_step: int,
    band_radius: int,
    purity: float,
    min_band_ratio: float,
    max_rois_per_label: int,
    y_bin_size: int,
    max_per_bin: int,
    max_overlap: float,
    white_min_std: float,
) -> tuple[list[ROI], dict[str, np.ndarray]]:
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    black_band, white_band, boundary = build_boundary_bands(
        deleted_mask=deleted_mask,
        band_radius=band_radius,
    )

    black_candidates = generate_candidates_for_label(
        label_mask=deleted_mask,
        boundary_band=black_band,
        gray=gray,
        label=BLACK_LABEL,
        roi_w=roi_w,
        roi_h=roi_h,
        scan_step=scan_step,
        purity=purity,
        min_band_ratio=min_band_ratio,
        min_std=0.0,
    )

    white_candidates = generate_candidates_for_label(
        label_mask=~deleted_mask,
        boundary_band=white_band,
        gray=gray,
        label=WHITE_LABEL,
        roi_w=roi_w,
        roi_h=roi_h,
        scan_step=scan_step,
        purity=purity,
        min_band_ratio=min_band_ratio,
        min_std=white_min_std,
    )

    black_rois = select_distributed_rois(
        candidates=black_candidates,
        max_rois=max_rois_per_label,
        y_bin_size=y_bin_size,
        max_per_bin=max_per_bin,
        max_overlap=max_overlap,
    )

    white_rois = select_distributed_rois(
        candidates=white_candidates,
        max_rois=max_rois_per_label,
        y_bin_size=y_bin_size,
        max_per_bin=max_per_bin,
        max_overlap=max_overlap,
    )

    all_rois = sorted(black_rois + white_rois, key=lambda r: (r.y0, r.x0, r.label))

    debug_masks = {
        "black_band": black_band,
        "white_band": white_band,
        "boundary": boundary,
    }

    return all_rois, debug_masks


# ==================================================
# Save files
# ==================================================

def save_rois_txt(path: Path, rois: list[ROI]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, roi in enumerate(rois):
            f.write(
                f"{i:03d} | {roi.label} | "
                f"x {roi.x0}-{roi.x1} | y {roi.y0}-{roi.y1} | "
                f"score {roi.score:.4f}\n"
            )


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]

    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass

    return ImageFont.load_default()


def save_preview(
    path: Path,
    original_rgb: np.ndarray,
    deleted_mask: np.ndarray,
    rois: list[ROI],
    max_preview_height: int,
) -> None:
    h, w = original_rgb.shape[:2]

    scale = min(1.0, max_preview_height / h)
    preview_w = max(1, int(w * scale))
    preview_h = max(1, int(h * scale))

    img = Image.fromarray(original_rgb).resize(
        (preview_w, preview_h),
        Image.Resampling.LANCZOS,
    ).convert("RGB")

    # Background deletion overlay
    deleted_small = cv2.resize(
        deleted_mask.astype(np.uint8),
        (preview_w, preview_h),
        interpolation=cv2.INTER_NEAREST,
    ) > 0

    arr = np.asarray(img).copy()

    red = np.zeros_like(arr)
    red[:, :, 0] = 255

    arr[deleted_small] = (
        arr[deleted_small].astype(np.float32) * 0.60
        + red[deleted_small].astype(np.float32) * 0.40
    ).astype(np.uint8)

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    font = load_font(14)

    for roi in rois:
        x0 = int(roi.x0 * scale)
        y0 = int(roi.y0 * scale)
        x1 = int(roi.x1 * scale)
        y1 = int(roi.y1 * scale)

        if roi.label == BLACK_LABEL:
            color = (255, 255, 0)
        else:
            color = (0, 255, 255)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    legend = (
        "Red overlay = deleted background from cleaned image\n"
        "Yellow ROI = BLACK/remove for threshold finder\n"
        "Cyan ROI = WHITE/keep/protect for threshold finder"
    )

    draw.rectangle([8, 8, 650, 70], fill=(0, 0, 0))
    draw.text((16, 14), legend, font=font, fill=(255, 255, 255))

    img.save(path, quality=95)


def save_boundary_debug(
    path: Path,
    debug_masks: dict[str, np.ndarray],
    max_preview_height: int,
) -> None:
    boundary = debug_masks["boundary"]
    black_band = debug_masks["black_band"]
    white_band = debug_masks["white_band"]

    h, w = boundary.shape[:2]

    scale = min(1.0, max_preview_height / h)
    preview_w = max(1, int(w * scale))
    preview_h = max(1, int(h * scale))

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[black_band] = (255, 255, 0)
    rgb[white_band] = (0, 255, 255)

    img = Image.fromarray(rgb).resize(
        (preview_w, preview_h),
        Image.Resampling.NEAREST,
    )

    img.save(path, quality=95)


def save_stats(
    path: Path,
    original_path: Path,
    cleaned_path: Path,
    deleted_mask: np.ndarray,
    rois: list[ROI],
    args: argparse.Namespace,
) -> None:
    h, w = deleted_mask.shape[:2]

    black_count = sum(1 for r in rois if r.label == BLACK_LABEL)
    white_count = sum(1 for r in rois if r.label == WHITE_LABEL)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"original: {original_path}\n")
        f.write(f"cleaned: {cleaned_path}\n")
        f.write(f"size: {w}x{h}\n")
        f.write(f"deleted_ratio: {deleted_mask.mean():.6f}\n")
        f.write(f"rois_total: {len(rois)}\n")
        f.write(f"black_rois: {black_count}\n")
        f.write(f"white_rois: {white_count}\n")
        f.write("\n")
        f.write("settings:\n")
        f.write(f"  roi_w: {args.roi_w}\n")
        f.write(f"  roi_h: {args.roi_h}\n")
        f.write(f"  scan_step: {args.scan_step}\n")
        f.write(f"  band_radius: {args.band_radius}\n")
        f.write(f"  purity: {args.purity}\n")
        f.write(f"  min_band_ratio: {args.min_band_ratio}\n")
        f.write(f"  max_rois_per_label: {args.max_rois_per_label}\n")
        f.write(f"  y_bin_size: {args.y_bin_size}\n")
        f.write(f"  max_per_bin: {args.max_per_bin}\n")
        f.write(f"  white_min_std: {args.white_min_std}\n")


# ==================================================
# Main
# ==================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BLACK/WHITE boundary ROIs from original + cleaned image."
    )

    parser.add_argument(
        "image",
        help="Chapter id or original image path. Example: 033, 033.png, data/chapters-long/033.png",
    )

    parser.add_argument(
        "--cleaned",
        default=None,
        help="Optional cleaned image path. Default: data/temp/<CH>/<CH>_cleaned.png",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output ROI file. Default: data/temp/<CH>/used_rois_<CH>_boundary.txt",
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=16,
        help="Alpha below this value is treated as deleted. Default: 16",
    )

    parser.add_argument(
        "--fallback-mode",
        choices=["auto", "white", "black", "red", "diff"],
        default="auto",
        help="Fallback mode if cleaned image has no alpha. Default: auto",
    )

    parser.add_argument(
        "--roi-w",
        type=int,
        default=18,
        help="ROI width. Default: 18",
    )

    parser.add_argument(
        "--roi-h",
        type=int,
        default=10,
        help="ROI height. Default: 10",
    )

    parser.add_argument(
        "--scan-step",
        type=int,
        default=12,
        help="Grid scan step. Default: 12",
    )

    parser.add_argument(
        "--band-radius",
        type=int,
        default=6,
        help="How far from boundary ROI candidates may be. Default: 6",
    )

    parser.add_argument(
        "--purity",
        type=float,
        default=0.98,
        help="Required ratio of ROI belonging to its class. Default: 0.98",
    )

    parser.add_argument(
        "--min-band-ratio",
        type=float,
        default=0.35,
        help="Required ratio of ROI being near boundary. Default: 0.35",
    )

    parser.add_argument(
        "--max-rois-per-label",
        type=int,
        default=180,
        help="Maximum ROIs for BLACK and WHITE separately. Default: 180",
    )

    parser.add_argument(
        "--y-bin-size",
        type=int,
        default=900,
        help="Vertical distribution bin size. Default: 900",
    )

    parser.add_argument(
        "--max-per-bin",
        type=int,
        default=4,
        help="Maximum selected ROIs per vertical bin per label. Default: 4",
    )

    parser.add_argument(
        "--max-overlap",
        type=float,
        default=0.30,
        help="Maximum overlap between selected ROIs. Default: 0.30",
    )

    parser.add_argument(
        "--white-min-std",
        type=float,
        default=4.0,
        help="Minimum grayscale std for WHITE ROI. Avoids flat bubble interiors. Default: 4.0",
    )

    parser.add_argument(
        "--preview-height",
        type=int,
        default=6000,
        help="Max preview height. Default: 6000",
    )

    args = parser.parse_args()

    original_path, cleaned_path, stem, temp_dir = resolve_paths(args.image, args.cleaned)

    output_path = Path(args.output) if args.output else temp_dir / f"used_rois_{stem}_boundary.txt"

    print(f"Original: {original_path}")
    print(f"Cleaned:  {cleaned_path}")

    original_rgb = load_rgb(original_path)
    cleaned_rgba = load_rgba(cleaned_path)

    original_rgb, cleaned_rgba = crop_to_same_size(original_rgb, cleaned_rgba)

    print(f"Size: {original_rgb.shape[1]}x{original_rgb.shape[0]}")

    deleted_mask = infer_deleted_mask(
        original_rgb=original_rgb,
        cleaned_rgba=cleaned_rgba,
        alpha_threshold=args.alpha_threshold,
        fallback_mode=args.fallback_mode,
    )

    print(f"Deleted ratio: {deleted_mask.mean():.4f}")

    rois, debug_masks = generate_boundary_rois(
        original_rgb=original_rgb,
        deleted_mask=deleted_mask,
        roi_w=args.roi_w,
        roi_h=args.roi_h,
        scan_step=args.scan_step,
        band_radius=args.band_radius,
        purity=args.purity,
        min_band_ratio=args.min_band_ratio,
        max_rois_per_label=args.max_rois_per_label,
        y_bin_size=args.y_bin_size,
        max_per_bin=args.max_per_bin,
        max_overlap=args.max_overlap,
        white_min_std=args.white_min_std,
    )

    black_count = sum(1 for r in rois if r.label == BLACK_LABEL)
    white_count = sum(1 for r in rois if r.label == WHITE_LABEL)

    print(f"Generated ROIs: {len(rois)}")
    print(f"BLACK/remove:  {black_count}")
    print(f"WHITE/keep:    {white_count}")

    save_rois_txt(output_path, rois)

    preview_path = output_path.parent / f"{stem}_boundary_rois_preview.jpg"
    boundary_path = output_path.parent / f"{stem}_boundary_debug.jpg"
    stats_path = output_path.parent / f"{stem}_boundary_stats.txt"

    save_preview(
        path=preview_path,
        original_rgb=original_rgb,
        deleted_mask=deleted_mask,
        rois=rois,
        max_preview_height=args.preview_height,
    )

    save_boundary_debug(
        path=boundary_path,
        debug_masks=debug_masks,
        max_preview_height=args.preview_height,
    )

    save_stats(
        path=stats_path,
        original_path=original_path,
        cleaned_path=cleaned_path,
        deleted_mask=deleted_mask,
        rois=rois,
        args=args,
    )

    print()
    print(f"Saved ROI file: {output_path}")
    print(f"Saved preview:  {preview_path}")
    print(f"Saved boundary: {boundary_path}")
    print(f"Saved stats:    {stats_path}")
    print()
    print("Next hard/black-bg search example:")
    print(
        f"python tools/mask_parameter_search.py {stem} "
        f"--priority black-bg "
        f"--morph-order both "
        f"--top-tonal 60"
    )
    print()
    print("Next soft search example:")
    print(
        f"python tools/mask_parameter_search.py {stem} "
        f"--priority soft "
        f"--morph-order both "
        f"--top-tonal 60"
    )


if __name__ == "__main__":
    main()
