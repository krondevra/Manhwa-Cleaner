#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ==================================================
# LABELS
# ==================================================

BLACK_LABEL = "black"
WHITE_LABEL = "white"

ROI_PAD = 64
TOP_N = 30


# ==================================================
# DATA CLASSES
# ==================================================

@dataclass
class ROI:
    x0: int
    y0: int
    x1: int
    y1: int
    label: str


@dataclass
class TonalParams:
    channel: str
    black: int
    white: int
    gamma_x100: int
    threshold: int
    score: float
    black_error: float
    white_error: float


@dataclass
class FinalParams:
    channel: str
    black: int
    white: int
    gamma_x100: int
    threshold: int
    min_radius: int
    max_radius: int
    score: float
    black_error: float
    white_error: float


@dataclass
class SearchProfile:
    priority: str
    black_weight: float
    white_weight: float
    channels: list[str]
    level_black_values: list[int]
    level_white_values: list[int]
    gamma_x100_values: list[int]
    threshold_values: list[int]
    min_radius_values: list[int]
    max_radius_values: list[int]


# ==================================================
# SEARCH PROFILES
# ==================================================

def get_search_profile(priority: str, wide: bool) -> SearchProfile:
    """
    hard:
        Optimized for creating a strong barrier.
        BLACK is more important.
        Useful for preventing flood-fill leakage.

    soft:
        Optimized for cleaning white/gray background artifacts.
        WHITE is more important.
        Useful for soft cleanup pass.
    """

    priority = priority.lower().strip()

    if priority not in {"hard", "soft"}:
        raise ValueError("priority must be 'hard' or 'soft'")

    if priority == "hard":
        if wide:
            return SearchProfile(
                priority="hard",
                black_weight=4.0,
                white_weight=1.5,
                channels=["grayscale", "min RGB", "max RGB"],
                level_black_values=list(range(8, 51, 2)),
                level_white_values=list(range(45, 171, 5)),
                gamma_x100_values=[25, 30, 35, 40, 45, 48, 55, 65, 80, 100, 120],
                threshold_values=[8, 12, 16, 20, 24, 28, 29, 32, 36, 40, 48, 56, 64],
                min_radius_values=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                max_radius_values=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
            )

        return SearchProfile(
            priority="hard",
            black_weight=4.0,
            white_weight=1.5,
            channels=["grayscale", "min RGB"],
            level_black_values=[18, 20, 22, 24, 26, 28, 30, 34, 38, 42],
            level_white_values=[70, 80, 90, 100, 110, 120, 130, 150],
            gamma_x100_values=[30, 35, 40, 45, 48, 55, 65, 80],
            threshold_values=[20, 24, 28, 29, 32, 36, 40, 48, 56, 64],
            min_radius_values=[10, 12, 14, 16, 18, 20, 22],
            max_radius_values=[10, 12, 14, 16, 18, 20, 22],
        )

    # SOFT
    if wide:
        return SearchProfile(
            priority="soft",
            black_weight=1.0,
            white_weight=4.0,
            channels=["grayscale", "max RGB", "min RGB"],
            level_black_values=list(range(0, 61, 2)),
            level_white_values=list(range(80, 256, 5)),
            gamma_x100_values=[40, 50, 60, 70, 80, 100, 120, 150, 180, 220],
            threshold_values=[32, 48, 56, 64, 80, 96, 112, 128, 141, 160, 180, 200],
            min_radius_values=[0, 1, 2, 3, 4, 6, 8, 10, 12],
            max_radius_values=[0, 1, 2, 3, 4, 6, 8, 10, 12],
        )

    return SearchProfile(
        priority="soft",
        black_weight=1.0,
        white_weight=4.0,
        channels=["grayscale", "max RGB"],
        level_black_values=[10, 14, 18, 22, 26, 30, 34, 38, 42],
        level_white_values=[90, 110, 130, 150, 170, 190, 210, 230],
        gamma_x100_values=[60, 80, 100, 120, 150],
        threshold_values=[48, 64, 80, 96, 112, 128, 141, 160],
        min_radius_values=[0, 2, 4, 6, 8, 10],
        max_radius_values=[0, 2, 4, 6, 8, 10],
    )


# ==================================================
# BASIC HELPERS
# ==================================================

def clamp(value, low, high):
    return max(low, min(value, high))


def normalize_label(label: str) -> str:
    label = label.strip().lower()

    if label == BLACK_LABEL:
        return BLACK_LABEL

    if label == WHITE_LABEL:
        return WHITE_LABEL

    raise ValueError(f"Unknown label: {label}")


def normalize_roi(x0: int, y0: int, x1: int, y1: int, label: str) -> ROI:
    rx0, rx1 = sorted([int(x0), int(x1)])
    ry0, ry1 = sorted([int(y0), int(y1)])

    return ROI(
        x0=rx0,
        y0=ry0,
        x1=rx1,
        y1=ry1,
        label=normalize_label(label),
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


# ==================================================
# ROI LOADING
# ==================================================

def parse_rois_from_text(text: str) -> list[ROI]:
    """
    Supported format:
    000 | BLACK | x 191-294 | y 1600-1608
    WHITE x 191-306 y 1594-1600
    """
    rois: list[ROI] = []

    pattern = re.compile(
        r"(?P<label>BLACK|WHITE|black|white).*?"
        r"x\s*(?P<x0>\d+)\s*-\s*(?P<x1>\d+).*?"
        r"y\s*(?P<y0>\d+)\s*-\s*(?P<y1>\d+)"
    )

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        match = pattern.search(line)

        if not match:
            continue

        roi = normalize_roi(
            x0=int(match.group("x0")),
            y0=int(match.group("y0")),
            x1=int(match.group("x1")),
            y1=int(match.group("y1")),
            label=match.group("label"),
        )

        if roi.x1 > roi.x0 and roi.y1 > roi.y0:
            rois.append(roi)

    return rois


def load_rois(path: Path) -> list[ROI]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rois = parse_rois_from_text(text)

    if not rois:
        raise ValueError(f"No ROIs found in: {path}")

    return rois


def validate_rois(rois: list[ROI]) -> None:
    black_count = sum(1 for roi in rois if roi.label == BLACK_LABEL)
    white_count = sum(1 for roi in rois if roi.label == WHITE_LABEL)

    if black_count == 0:
        raise ValueError("Need at least one BLACK ROI.")

    if white_count == 0:
        raise ValueError("Need at least one WHITE ROI.")


def save_used_rois(out_dir: Path, rois: list[ROI]) -> None:
    path = out_dir / "used_rois.txt"

    with path.open("w", encoding="utf-8") as f:
        for i, roi in enumerate(rois):
            f.write(
                f"{i:03d} | {roi.label.upper()} | "
                f"x {roi.x0}-{roi.x1} | y {roi.y0}-{roi.y1}\n"
            )


# ==================================================
# IMAGE PROCESSING
# ==================================================

def apply_channel(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "grayscale":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    if mode == "min RGB":
        return np.minimum(np.minimum(r, g), b)

    if mode == "max RGB":
        return np.maximum(np.maximum(r, g), b)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def apply_levels(channel: np.ndarray, black: int, white: int, gamma_x100: int) -> np.ndarray:
    black = int(clamp(black, 0, 254))
    white = int(clamp(white, black + 1, 255))

    gamma = max(0.05, gamma_x100 / 100.0)

    x = channel.astype(np.float32)
    x = (x - black) / (white - black)
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, 1.0 / gamma)
    x = x * 255.0

    return x.astype(np.uint8)


def threshold_mask(levels_channel: np.ndarray, threshold_value: int) -> np.ndarray:
    return np.where(levels_channel < threshold_value, 0, 255).astype(np.uint8)


def apply_minimum_maximum(mask: np.ndarray, min_radius: int, max_radius: int) -> np.ndarray:
    out = mask

    if min_radius > 0:
        k = min_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.erode(out, kernel, iterations=1)

    if max_radius > 0:
        k = max_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel, iterations=1)

    return out


def build_tonal_mask(
    rgb: np.ndarray,
    channel_mode: str,
    black: int,
    white: int,
    gamma_x100: int,
    threshold_value: int,
) -> np.ndarray:
    channel = apply_channel(rgb, channel_mode)
    levels = apply_levels(channel, black, white, gamma_x100)
    return threshold_mask(levels, threshold_value)


def build_final_mask(
    rgb: np.ndarray,
    channel_mode: str,
    black: int,
    white: int,
    gamma_x100: int,
    threshold_value: int,
    min_radius: int,
    max_radius: int,
) -> np.ndarray:
    mask = build_tonal_mask(
        rgb=rgb,
        channel_mode=channel_mode,
        black=black,
        white=white,
        gamma_x100=gamma_x100,
        threshold_value=threshold_value,
    )

    return apply_minimum_maximum(mask, min_radius, max_radius)


def crop_for_roi(rgb: np.ndarray, roi: ROI, pad: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = rgb.shape[:2]

    x0 = max(0, roi.x0 - pad)
    y0 = max(0, roi.y0 - pad)
    x1 = min(w, roi.x1 + pad)
    y1 = min(h, roi.y1 + pad)

    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI outside image: {roi}")

    crop = rgb[y0:y1, x0:x1].copy()

    local = (
        roi.x0 - x0,
        roi.y0 - y0,
        roi.x1 - x0,
        roi.y1 - y0,
    )

    return crop, local


def make_preview_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.70) -> np.ndarray:
    out = rgb.copy()
    black_pixels = mask < 128

    red = np.zeros_like(out)
    red[:, :, 0] = 255

    out[black_pixels] = (
        out[black_pixels].astype(np.float32) * (1.0 - alpha)
        + red[black_pixels].astype(np.float32) * alpha
    ).astype(np.uint8)

    return out


# ==================================================
# SCORING
# ==================================================

def score_mask_patch(mask_patch: np.ndarray, label: str) -> float:
    if mask_patch.size == 0:
        return 1.0

    if label == BLACK_LABEL:
        return float((mask_patch > 127).mean())

    if label == WHITE_LABEL:
        return float((mask_patch < 128).mean())

    return 1.0


def calculate_weighted_score(
    black_errors: list[float],
    white_errors: list[float],
    profile: SearchProfile,
) -> tuple[float, float, float]:
    black_error = float(np.mean(black_errors)) if black_errors else 0.0
    white_error = float(np.mean(white_errors)) if white_errors else 0.0

    score = black_error * profile.black_weight + white_error * profile.white_weight

    return score, black_error, white_error


def evaluate_tonal_params(
    rgb: np.ndarray,
    rois: list[ROI],
    profile: SearchProfile,
    channel: str,
    black: int,
    white: int,
    gamma_x100: int,
    threshold_value: int,
) -> TonalParams:
    black_errors = []
    white_errors = []

    for roi in rois:
        crop, local = crop_for_roi(rgb, roi, pad=0)
        lx0, ly0, lx1, ly1 = local

        mask = build_tonal_mask(
            rgb=crop,
            channel_mode=channel,
            black=black,
            white=white,
            gamma_x100=gamma_x100,
            threshold_value=threshold_value,
        )

        patch = mask[ly0:ly1, lx0:lx1]
        err = score_mask_patch(patch, roi.label)

        if roi.label == BLACK_LABEL:
            black_errors.append(err)
        else:
            white_errors.append(err)

    score, black_error, white_error = calculate_weighted_score(
        black_errors,
        white_errors,
        profile,
    )

    return TonalParams(
        channel=channel,
        black=black,
        white=white,
        gamma_x100=gamma_x100,
        threshold=threshold_value,
        score=score,
        black_error=black_error,
        white_error=white_error,
    )


def evaluate_final_params(
    rgb: np.ndarray,
    rois: list[ROI],
    profile: SearchProfile,
    tonal: TonalParams,
    min_radius: int,
    max_radius: int,
) -> FinalParams:
    black_errors = []
    white_errors = []

    for roi in rois:
        crop, local = crop_for_roi(rgb, roi, pad=ROI_PAD)
        lx0, ly0, lx1, ly1 = local

        mask = build_final_mask(
            rgb=crop,
            channel_mode=tonal.channel,
            black=tonal.black,
            white=tonal.white,
            gamma_x100=tonal.gamma_x100,
            threshold_value=tonal.threshold,
            min_radius=min_radius,
            max_radius=max_radius,
        )

        patch = mask[ly0:ly1, lx0:lx1]
        err = score_mask_patch(patch, roi.label)

        if roi.label == BLACK_LABEL:
            black_errors.append(err)
        else:
            white_errors.append(err)

    score, black_error, white_error = calculate_weighted_score(
        black_errors,
        white_errors,
        profile,
    )

    return FinalParams(
        channel=tonal.channel,
        black=tonal.black,
        white=tonal.white,
        gamma_x100=tonal.gamma_x100,
        threshold=tonal.threshold,
        min_radius=min_radius,
        max_radius=max_radius,
        score=score,
        black_error=black_error,
        white_error=white_error,
    )


# ==================================================
# SEARCH
# ==================================================

def search_stage1_tonal(
    rgb: np.ndarray,
    rois: list[ROI],
    profile: SearchProfile,
) -> list[TonalParams]:
    combinations = []

    for channel in profile.channels:
        for black in profile.level_black_values:
            for white in profile.level_white_values:
                if white <= black + 1:
                    continue

                for gamma_x100 in profile.gamma_x100_values:
                    for threshold_value in profile.threshold_values:
                        combinations.append(
                            (
                                channel,
                                black,
                                white,
                                gamma_x100,
                                threshold_value,
                            )
                        )

    total = len(combinations)
    results: list[TonalParams] = []

    print()
    print(f"Stage 1: Levels + Threshold only [{profile.priority}]")
    print(f"Combinations: {total}")
    print(f"Weights: BLACK={profile.black_weight}, WHITE={profile.white_weight}")

    start = time.time()

    for i, (channel, black, white, gamma_x100, threshold_value) in enumerate(combinations, start=1):
        params = evaluate_tonal_params(
            rgb=rgb,
            rois=rois,
            profile=profile,
            channel=channel,
            black=black,
            white=white,
            gamma_x100=gamma_x100,
            threshold_value=threshold_value,
        )

        results.append(params)

        if i % 250 == 0 or i == total:
            best = min(results, key=lambda p: p.score)
            elapsed = time.time() - start

            print(
                f"[stage1 {i}/{total}] {i / total * 100:.1f}% "
                f"elapsed={elapsed:.1f}s "
                f"best_score={best.score:.6f} "
                f"black_err={best.black_error:.4f} "
                f"white_err={best.white_error:.4f} "
                f"best={best.channel} "
                f"L {best.black}/{best.gamma_x100 / 100:.2f}/{best.white} "
                f"T{best.threshold}",
                flush=True,
            )

    results.sort(key=lambda p: p.score)

    return results


def search_stage2_morphology(
    rgb: np.ndarray,
    rois: list[ROI],
    profile: SearchProfile,
    tonal_results: list[TonalParams],
    top_tonal: int,
) -> list[FinalParams]:
    tonal_candidates = tonal_results[:top_tonal]

    combinations = []

    for tonal in tonal_candidates:
        for min_radius in profile.min_radius_values:
            for max_radius in profile.max_radius_values:
                combinations.append((tonal, min_radius, max_radius))

    total = len(combinations)
    results: list[FinalParams] = []

    print()
    print(f"Stage 2: Minimum + Maximum [{profile.priority}]")
    print(f"Tonal candidates: {len(tonal_candidates)}")
    print(f"Combinations: {total}")

    start = time.time()

    for i, (tonal, min_radius, max_radius) in enumerate(combinations, start=1):
        params = evaluate_final_params(
            rgb=rgb,
            rois=rois,
            profile=profile,
            tonal=tonal,
            min_radius=min_radius,
            max_radius=max_radius,
        )

        results.append(params)

        if i % 100 == 0 or i == total:
            best = min(results, key=lambda p: p.score)
            elapsed = time.time() - start

            print(
                f"[stage2 {i}/{total}] {i / total * 100:.1f}% "
                f"elapsed={elapsed:.1f}s "
                f"best_score={best.score:.6f} "
                f"black_err={best.black_error:.4f} "
                f"white_err={best.white_error:.4f} "
                f"best={best.channel} "
                f"L {best.black}/{best.gamma_x100 / 100:.2f}/{best.white} "
                f"T{best.threshold} "
                f"Min{best.min_radius} Max{best.max_radius}",
                flush=True,
            )

    results.sort(key=lambda p: p.score)

    return results


# ==================================================
# SAVE OUTPUT
# ==================================================

def save_stage1_csv(out_dir: Path, results: list[TonalParams]) -> None:
    path = out_dir / "stage1_tonal_top.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "score",
                "black_error",
                "white_error",
                "channel",
                "levels_black",
                "levels_white",
                "gamma",
                "threshold",
            ]
        )

        for rank, p in enumerate(results[:TOP_N], start=1):
            writer.writerow(
                [
                    rank,
                    f"{p.score:.8f}",
                    f"{p.black_error:.8f}",
                    f"{p.white_error:.8f}",
                    p.channel,
                    p.black,
                    p.white,
                    f"{p.gamma_x100 / 100:.2f}",
                    p.threshold,
                ]
            )


def save_final_csv(out_dir: Path, results: list[FinalParams]) -> None:
    path = out_dir / "top_results.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "score",
                "black_error",
                "white_error",
                "channel",
                "levels_black",
                "levels_white",
                "gamma",
                "threshold",
                "minimum_radius",
                "maximum_radius",
            ]
        )

        for rank, p in enumerate(results[:TOP_N], start=1):
            writer.writerow(
                [
                    rank,
                    f"{p.score:.8f}",
                    f"{p.black_error:.8f}",
                    f"{p.white_error:.8f}",
                    p.channel,
                    p.black,
                    p.white,
                    f"{p.gamma_x100 / 100:.2f}",
                    p.threshold,
                    p.min_radius,
                    p.max_radius,
                ]
            )


def save_best_settings(out_dir: Path, best: FinalParams, profile: SearchProfile) -> None:
    path = out_dir / "best_settings.txt"

    with path.open("w", encoding="utf-8") as f:
        f.write(f"priority: {profile.priority}\n")
        f.write(f"black_weight: {profile.black_weight}\n")
        f.write(f"white_weight: {profile.white_weight}\n")
        f.write(f"channel: {best.channel}\n")
        f.write(f"levels_black: {best.black}\n")
        f.write(f"levels_white: {best.white}\n")
        f.write(f"gamma: {best.gamma_x100 / 100:.2f}\n")
        f.write(f"threshold: {best.threshold}\n")
        f.write(f"minimum_radius: {best.min_radius}\n")
        f.write(f"maximum_radius: {best.max_radius}\n")
        f.write(f"score: {best.score:.8f}\n")
        f.write(f"black_error: {best.black_error:.8f}\n")
        f.write(f"white_error: {best.white_error:.8f}\n")


def save_contact_sheet(
    out_dir: Path,
    rgb: np.ndarray,
    rois: list[ROI],
    final_results: list[FinalParams],
) -> None:
    if not final_results or not rois:
        return

    params_list = final_results[:10]

    cell_w = 280
    cell_h = 210
    label_h = 74
    pad = 12

    cols = min(4, len(rois))
    rows = len(params_list) * math.ceil(len(rois) / cols)

    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows * (cell_h + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (25, 25, 25))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            16,
        )
    except Exception:
        font = ImageFont.load_default()

    row_idx = 0

    for rank, p in enumerate(params_list, start=1):
        for start_idx in range(0, len(rois), cols):
            group = rois[start_idx:start_idx + cols]

            for col_idx, roi in enumerate(group):
                crop, _local = crop_for_roi(rgb, roi, pad=ROI_PAD)

                mask = build_final_mask(
                    rgb=crop,
                    channel_mode=p.channel,
                    black=p.black,
                    white=p.white,
                    gamma_x100=p.gamma_x100,
                    threshold_value=p.threshold,
                    min_radius=p.min_radius,
                    max_radius=p.max_radius,
                )

                preview = make_preview_overlay(crop, mask, alpha=0.70)

                img = Image.fromarray(preview)
                img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

                cell = Image.new("RGB", (cell_w, cell_h + label_h), (0, 0, 0))
                draw = ImageDraw.Draw(cell)

                label_text = (
                    f"#{rank} score={p.score:.4f}\n"
                    f"black={p.black_error:.3f} white={p.white_error:.3f}\n"
                    f"{p.channel} L {p.black}/{p.gamma_x100 / 100:.2f}/{p.white} "
                    f"T{p.threshold} Min{p.min_radius} Max{p.max_radius}"
                )

                draw.text((6, 5), label_text, font=font, fill=(255, 255, 255))
                cell.paste(img, ((cell_w - img.width) // 2, label_h))

                x = pad + col_idx * (cell_w + pad)
                y = pad + row_idx * (cell_h + label_h + pad)

                sheet.paste(cell, (x, y))

            row_idx += 1

    sheet.save(out_dir / "top10_contact_sheet.jpg", quality=95)


def save_all_results(
    image_path: Path,
    rgb: np.ndarray,
    rois: list[ROI],
    stage1_results: list[TonalParams],
    final_results: list[FinalParams],
    profile: SearchProfile,
) -> None:
    out_dir = Path(f"{image_path.stem}_{profile.priority}_params")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_used_rois(out_dir, rois)
    save_stage1_csv(out_dir, stage1_results)
    save_final_csv(out_dir, final_results)
    save_best_settings(out_dir, final_results[0], profile)
    save_contact_sheet(out_dir, rgb, rois, final_results)

    best = final_results[0]

    print()
    print("Best final settings:")
    print(f"  priority: {profile.priority}")
    print(f"  weights: BLACK={profile.black_weight}, WHITE={profile.white_weight}")
    print(f"  channel: {best.channel}")
    print(f"  levels: {best.black} / {best.gamma_x100 / 100:.2f} / {best.white}")
    print(f"  threshold: {best.threshold}")
    print(f"  minimum_radius: {best.min_radius}")
    print(f"  maximum_radius: {best.max_radius}")
    print(f"  score: {best.score:.8f}")
    print(f"  black_error: {best.black_error:.8f}")
    print(f"  white_error: {best.white_error:.8f}")
    print()
    print(f"Saved to: {out_dir}")
    print(f"  {out_dir / 'best_settings.txt'}")
    print(f"  {out_dir / 'stage1_tonal_top.csv'}")
    print(f"  {out_dir / 'top_results.csv'}")
    print(f"  {out_dir / 'top10_contact_sheet.jpg'}")


# ==================================================
# MAIN
# ==================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto finder for Levels + Threshold + Minimum/Maximum with hard/soft priority.",
    )

    parser.add_argument(
        "image",
        help="Input image, e.g. 012-1.png",
    )

    parser.add_argument(
        "--rois",
        required=True,
        help="ROI text file, e.g. used_rois.txt",
    )

    parser.add_argument(
        "--priority",
        choices=["hard", "soft"],
        default="hard",
        help="Search priority: hard = barrier mask, soft = cleanup mask. Default: hard",
    )

    parser.add_argument(
        "--wide",
        action="store_true",
        help="Use wider and slower search space.",
    )

    parser.add_argument(
        "--top-tonal",
        type=int,
        default=30,
        help="How many best tonal candidates from stage 1 to use in stage 2. Default: 30",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    roi_path = Path(args.rois)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI file not found: {roi_path}")

    rgb = load_rgb(image_path)
    rois = load_rois(roi_path)
    validate_rois(rois)

    profile = get_search_profile(args.priority, args.wide)

    print(f"Image: {image_path}")
    print(f"Size: {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"ROIs: {len(rois)}")
    print(f"BLACK: {sum(1 for r in rois if r.label == BLACK_LABEL)}")
    print(f"WHITE: {sum(1 for r in rois if r.label == WHITE_LABEL)}")
    print(f"Priority: {profile.priority}")
    print(f"Weights: BLACK={profile.black_weight}, WHITE={profile.white_weight}")
    print(f"Mode: {'WIDE' if args.wide else 'FAST'}")
    print(f"Top tonal candidates for stage 2: {args.top_tonal}")

    stage1_results = search_stage1_tonal(
        rgb=rgb,
        rois=rois,
        profile=profile,
    )

    stage2_results = search_stage2_morphology(
        rgb=rgb,
        rois=rois,
        profile=profile,
        tonal_results=stage1_results,
        top_tonal=args.top_tonal,
    )

    save_all_results(
        image_path=image_path,
        rgb=rgb,
        rois=rois,
        stage1_results=stage1_results,
        final_results=stage2_results,
        profile=profile,
    )


if __name__ == "__main__":
    main()
