#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


@dataclass
class ChapterScore:
    chapter: str
    width: int
    height: int
    total_pixels: int

    deleted_pixels: int
    damaged_pixels: int
    leftover_grayline_pixels: int

    delete_percent_total: float
    damage_percent_total: float
    damage_percent_deleted: float
    leftover_grayline_percent_total: float

    estimated_success_percent: float
    priority_score: float

    original_path: str
    result_path: str


def load_original_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_result_rgba(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    img = Image.open(path)

    if img.mode == "RGBA":
        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
        return rgb, alpha

    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return rgb, None


def crop_to_same_size(
    original_rgb: np.ndarray,
    result_rgb: np.ndarray,
    alpha: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    h = min(original_rgb.shape[0], result_rgb.shape[0])
    w = min(original_rgb.shape[1], result_rgb.shape[1])

    original_rgb = original_rgb[:h, :w]
    result_rgb = result_rgb[:h, :w]

    if alpha is not None:
        alpha = alpha[:h, :w]

    return original_rgb, result_rgb, alpha


def discover_pairs(
    original_dir: Path,
    result_dir: Path,
    result_suffix: str,
) -> list[tuple[str, Path, Path]]:
    pairs = []

    for result_path in sorted(result_dir.glob(f"*{result_suffix}.png")):
        if result_path.name.endswith("_red_preview.png"):
            continue

        stem = result_path.stem

        if not stem.endswith(result_suffix):
            continue

        chapter = stem[: -len(result_suffix)]
        original_path = original_dir / f"{chapter}.png"

        if not original_path.exists():
            print(f"WARNING: original not found: {original_path}")
            continue

        pairs.append((chapter, original_path, result_path))

    return pairs


def infer_deleted_mask(
    original_rgb: np.ndarray,
    result_rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    alpha_threshold: int,
    rgb_white_threshold: int,
) -> np.ndarray:
    """
    Best case:
    - result has alpha channel
    - transparent pixels = deleted

    Fallback:
    - result became near-white while original was not plain white
    """
    if alpha is not None:
        return alpha < alpha_threshold

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    result_gray = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY)

    original_hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    original_sat = original_hsv[:, :, 1]

    result_near_white = result_gray >= rgb_white_threshold
    original_not_plain_white = (original_gray < rgb_white_threshold) | (original_sat > 20)

    return result_near_white & original_not_plain_white


def calculate_chapter_score(
    chapter: str,
    original_path: Path,
    result_path: Path,
    alpha_threshold: int,
    rgb_white_threshold: int,
) -> ChapterScore:
    original_rgb = load_original_rgb(original_path)
    result_rgb, alpha = load_result_rgba(result_path)

    original_rgb, result_rgb, alpha = crop_to_same_size(
        original_rgb,
        result_rgb,
        alpha,
    )

    h, w = original_rgb.shape[:2]
    total_pixels = h * w

    deleted_mask = infer_deleted_mask(
        original_rgb=original_rgb,
        result_rgb=result_rgb,
        alpha=alpha,
        alpha_threshold=alpha_threshold,
        rgb_white_threshold=rgb_white_threshold,
    )

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    edges = cv2.Canny(gray, 50, 150) > 0
    kept_mask = ~deleted_mask

    deleted_dark = deleted_mask & (gray < 180)
    deleted_edges = deleted_mask & edges
    deleted_color = deleted_mask & (sat > 45)

    damaged_mask = deleted_dark | deleted_edges | deleted_color

    leftover_grayline_mask = (
        kept_mask
        & edges
        & (gray >= 120)
        & (gray <= 245)
        & (sat <= 35)
    )

    deleted_pixels = int(deleted_mask.sum())
    damaged_pixels = int(damaged_mask.sum())
    leftover_grayline_pixels = int(leftover_grayline_mask.sum())

    delete_percent_total = deleted_pixels / total_pixels * 100.0
    damage_percent_total = damaged_pixels / total_pixels * 100.0

    if deleted_pixels > 0:
        damage_percent_deleted = damaged_pixels / deleted_pixels * 100.0
    else:
        damage_percent_deleted = 0.0

    leftover_grayline_percent_total = leftover_grayline_pixels / total_pixels * 100.0

    # Heuristic:
    # - damage_percent_deleted is the main problem indicator
    # - leftover gray lines are secondary but still important
    #
    # This is not a real accuracy metric. It is a prioritization metric.
    penalty = damage_percent_deleted + leftover_grayline_percent_total * 5.0
    estimated_success_percent = max(0.0, min(100.0, 100.0 - penalty))

    priority_score = (
        damage_percent_deleted * 1.0
        + damage_percent_total * 0.5
        + leftover_grayline_percent_total * 5.0
    )

    return ChapterScore(
        chapter=chapter,
        width=w,
        height=h,
        total_pixels=total_pixels,
        deleted_pixels=deleted_pixels,
        damaged_pixels=damaged_pixels,
        leftover_grayline_pixels=leftover_grayline_pixels,
        delete_percent_total=delete_percent_total,
        damage_percent_total=damage_percent_total,
        damage_percent_deleted=damage_percent_deleted,
        leftover_grayline_percent_total=leftover_grayline_percent_total,
        estimated_success_percent=estimated_success_percent,
        priority_score=priority_score,
        original_path=str(original_path),
        result_path=str(result_path),
    )


def save_summary(scores: list[ChapterScore], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "chapter",
                "estimated_success_percent",
                "priority_score",
                "damage_percent_deleted",
                "damage_percent_total",
                "delete_percent_total",
                "leftover_grayline_percent_total",
                "total_pixels",
                "deleted_pixels",
                "damaged_pixels",
                "leftover_grayline_pixels",
                "width",
                "height",
                "original_path",
                "result_path",
            ]
        )

        for rank, score in enumerate(scores, start=1):
            writer.writerow(
                [
                    rank,
                    score.chapter,
                    f"{score.estimated_success_percent:.3f}",
                    f"{score.priority_score:.3f}",
                    f"{score.damage_percent_deleted:.3f}",
                    f"{score.damage_percent_total:.3f}",
                    f"{score.delete_percent_total:.3f}",
                    f"{score.leftover_grayline_percent_total:.5f}",
                    score.total_pixels,
                    score.deleted_pixels,
                    score.damaged_pixels,
                    score.leftover_grayline_pixels,
                    score.width,
                    score.height,
                    score.original_path,
                    score.result_path,
                ]
            )


def print_summary(scores: list[ChapterScore], limit: int) -> None:
    print()
    print("Most problematic chapters first:")
    print()

    header = (
        f"{'rank':>4}  "
        f"{'chapter':>7}  "
        f"{'success%':>9}  "
        f"{'damage%del':>10}  "
        f"{'damage%all':>10}  "
        f"{'deleted%':>9}  "
        f"{'grayline%':>9}  "
        f"{'priority':>9}"
    )

    print(header)
    print("-" * len(header))

    for rank, score in enumerate(scores[:limit], start=1):
        print(
            f"{rank:>4}  "
            f"{score.chapter:>7}  "
            f"{score.estimated_success_percent:>9.2f}  "
            f"{score.damage_percent_deleted:>10.2f}  "
            f"{score.damage_percent_total:>10.2f}  "
            f"{score.delete_percent_total:>9.2f}  "
            f"{score.leftover_grayline_percent_total:>9.4f}  "
            f"{score.priority_score:>9.2f}"
        )

    print()
    print("Columns:")
    print("  success%    = estimated success, higher is better")
    print("  damage%del  = suspicious damaged pixels among deleted pixels")
    print("  damage%all  = suspicious damaged pixels among all pixels")
    print("  deleted%    = how much of chapter was removed")
    print("  grayline%   = likely leftover gray-line artifacts")
    print("  priority    = larger means check this chapter first")
    print()
    print("Important:")
    print("  This is a heuristic. It finds suspicious chapters, not perfect ground truth.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate chapter-level ML cleanup quality from chapters-long and chapters-results."
    )

    parser.add_argument(
        "--original-dir",
        default="chapters-long",
        help="Original chapters directory. Default: chapters-long",
    )

    parser.add_argument(
        "--result-dir",
        default="chapters-results",
        help="Processed chapters directory. Default: chapters-results",
    )

    parser.add_argument(
        "--result-suffix",
        default="_result",
        help="Result suffix. Default: _result",
    )

    parser.add_argument(
        "--output",
        default="problem_candidates/summary.csv",
        help="Output CSV path. Default: problem_candidates/summary.csv",
    )

    parser.add_argument(
        "--print-limit",
        type=int,
        default=30,
        help="How many chapters to print. Default: 30",
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=16,
        help="Alpha below this value is treated as deleted. Default: 16",
    )

    parser.add_argument(
        "--rgb-white-threshold",
        type=int,
        default=245,
        help="Fallback threshold if result has no alpha. Default: 245",
    )

    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    result_dir = Path(args.result_dir)

    if not original_dir.exists():
        raise FileNotFoundError(f"Original directory not found: {original_dir}")

    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    pairs = discover_pairs(
        original_dir=original_dir,
        result_dir=result_dir,
        result_suffix=args.result_suffix,
    )

    if not pairs:
        print("No pairs found.")
        print("Expected:")
        print(f"  {original_dir}/003.png")
        print(f"  {result_dir}/003_result.png")
        return

    print(f"Found pairs: {len(pairs)}")

    scores: list[ChapterScore] = []

    for chapter, original_path, result_path in pairs:
        print(f"Evaluating {chapter}")
        score = calculate_chapter_score(
            chapter=chapter,
            original_path=original_path,
            result_path=result_path,
            alpha_threshold=args.alpha_threshold,
            rgb_white_threshold=args.rgb_white_threshold,
        )
        scores.append(score)

    scores.sort(key=lambda s: s.priority_score, reverse=True)

    output_path = Path(args.output)
    save_summary(scores, output_path)
    print_summary(scores, args.print_limit)

    print(f"Saved summary: {output_path}")


if __name__ == "__main__":
    main()
