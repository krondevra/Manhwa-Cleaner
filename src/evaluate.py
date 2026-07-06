#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
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


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def normalize_chapter(value: str) -> str:
    """
    Accept:
        33
        033
        033.png
        data/chapters-long/033.png

    Return the chapter name exactly as written in the filename/input,
    without zero-padding.

    Examples:
        2   -> 2
        24  -> 24
        024 -> 024
        003 -> 003
    """
    stem = Path(str(value)).stem

    if stem.endswith("_result"):
        stem = stem[:-len("_result")]

    return stem



def parse_chapter_list(value: str) -> list[str]:
    """
    Parse a comma/space separated chapter list.

    Supported examples:
        2,34,24
        002,034,024
        2 34 24
        003-005,012

    No zero-padding is added automatically.

    Examples:
        2       -> 2
        34      -> 34
        024     -> 024
        003-005 -> 003,004,005
        3-5     -> 3,4,5
    """
    if not value:
        return []

    raw_items = re.split(r"[,\s]+", value.strip())
    chapters: list[str] = []

    for item in raw_items:
        if not item:
            continue

        # Optional small range support:
        # 003-005 -> 003,004,005
        # 3-5     -> 3,4,5
        if re.fullmatch(r"\d+\s*-\s*\d+", item):
            left, right = re.split(r"\s*-\s*", item)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1

            # Preserve padding only when the user wrote padding.
            width = max(len(left), len(right))

            for number in range(start, end + step, step):
                if width > 1:
                    chapters.append(f"{number:0{width}d}")
                else:
                    chapters.append(str(number))

            continue

        chapters.append(normalize_chapter(item))

    # Preserve user order, remove duplicates.
    seen: set[str] = set()
    unique: list[str] = []

    for chapter in chapters:
        if chapter in seen:
            continue
        seen.add(chapter)
        unique.append(chapter)

    return unique


def chapter_sort_key(chapter: str) -> tuple[int, str]:
    if chapter.isdigit():
        return (0, f"{int(chapter):09d}")

    numbers = re.findall(r"\d+", chapter)
    if numbers:
        return (1, f"{int(numbers[-1]):09d}_{chapter}")

    return (2, chapter)


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


def result_path_for_chapter(
    chapter: str,
    result_dir: Path,
    result_suffix: str,
) -> Path:
    return result_dir / f"{chapter}{result_suffix}.png"


def original_path_for_chapter(
    chapter: str,
    original_dir: Path,
) -> Path:
    return original_dir / f"{chapter}.png"


def discover_pairs(
    original_dir: Path,
    result_dir: Path,
    result_suffix: str,
    chapter: str | None = None,
    chapters: str | None = None,
    chapter_from: str | None = None,
    chapter_to: str | None = None,
    allow_missing: bool = False,
) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []

    if chapter is not None:
        ch = normalize_chapter(chapter)
        original_path = original_path_for_chapter(ch, original_dir)
        result_path = result_path_for_chapter(ch, result_dir, result_suffix)

        missing: list[str] = []

        if not original_path.exists():
            missing.append(f"original not found: {original_path}")

        if not result_path.exists():
            missing.append(f"result not found: {result_path}")

        if missing and not allow_missing:
            raise FileNotFoundError(
                "Requested chapter cannot be evaluated because required file(s) are missing:\n"
                + "\n".join(f"  - {item}" for item in missing)
            )

        if missing:
            for item in missing:
                print(f"WARNING: {item}")
            return []

        return [(ch, original_path, result_path)]

    if chapters is not None:
        selected_chapters = parse_chapter_list(chapters)
        missing: list[str] = []

        for ch in selected_chapters:
            original_path = original_path_for_chapter(ch, original_dir)
            result_path = result_path_for_chapter(ch, result_dir, result_suffix)

            chapter_missing: list[str] = []

            if not original_path.exists():
                chapter_missing.append(f"original not found: {original_path}")

            if not result_path.exists():
                chapter_missing.append(f"result not found: {result_path}")

            if chapter_missing:
                for item in chapter_missing:
                    missing.append(f"{ch}: {item}")

                if allow_missing:
                    for item in chapter_missing:
                        print(f"WARNING: {ch}: {item}")
                    continue

                continue

            pairs.append((ch, original_path, result_path))

        if missing and not allow_missing:
            raise FileNotFoundError(
                "Requested chapter list cannot be evaluated completely. "
                "Missing file(s):\n"
                + "\n".join(f"  - {item}" for item in missing)
            )

        if len(pairs) != len(selected_chapters) and not allow_missing:
            raise RuntimeError(
                f"Requested {len(selected_chapters)} chapter(s), but found {len(pairs)} complete pair(s)."
            )

        return pairs

    chapter_from_norm = normalize_chapter(chapter_from) if chapter_from else None
    chapter_to_norm = normalize_chapter(chapter_to) if chapter_to else None

    for result_path in sorted(result_dir.glob(f"*{result_suffix}.png")):
        if result_path.name.endswith("_red_preview.png"):
            continue

        stem = result_path.stem

        if not stem.endswith(result_suffix):
            continue

        ch = stem[: -len(result_suffix)]
        ch = normalize_chapter(ch)

        if chapter_from_norm and chapter_sort_key(ch) < chapter_sort_key(chapter_from_norm):
            continue

        if chapter_to_norm and chapter_sort_key(ch) > chapter_sort_key(chapter_to_norm):
            continue

        original_path = original_path_for_chapter(ch, original_dir)

        if not original_path.exists():
            print(f"WARNING: original not found: {original_path}")
            continue

        pairs.append((ch, original_path, result_path))

    pairs.sort(key=lambda x: chapter_sort_key(x[0]))
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



def build_summary_filename(
    chapter: str | None,
    chapters: str | None,
    chapter_from: str | None,
    chapter_to: str | None,
) -> str:
    """
    Generate CSV filename from CLI arguments.

    Examples:
        --chapter 033
        -> 033_summary.csv

        --chapters 2,34,24
        -> 2-34-24_summary.csv

        --from-chapter 003 --to-chapter 175
        -> 003-175_summary.csv

        no range arguments
        -> all_summary.csv
    """
    if chapter:
        ch = normalize_chapter(chapter)
        return f"{ch}_summary.csv"

    if chapters:
        selected_chapters = parse_chapter_list(chapters)

        if not selected_chapters:
            return "selected_empty_summary.csv"

        if len(selected_chapters) <= 10:
            joined = "-".join(selected_chapters)
            return f"{joined}_summary.csv"

        first = selected_chapters[0]
        last = selected_chapters[-1]
        return f"selected_{len(selected_chapters)}_chapters_{first}-{last}_summary.csv"

    if chapter_from and chapter_to:
        ch_from = normalize_chapter(chapter_from)
        ch_to = normalize_chapter(chapter_to)
        return f"{ch_from}-{ch_to}_summary.csv"

    if chapter_from:
        ch_from = normalize_chapter(chapter_from)
        return f"{ch_from}-end_summary.csv"

    if chapter_to:
        ch_to = normalize_chapter(chapter_to)
        return f"start-{ch_to}_summary.csv"

    return "all_summary.csv"


def resolve_output_path(
    output: str | None,
    output_dir: str | Path,
    chapter: str | None,
    chapters: str | None,
    chapter_from: str | None,
    chapter_to: str | None,
) -> Path:
    auto_filename = build_summary_filename(
        chapter=chapter,
        chapters=chapters,
        chapter_from=chapter_from,
        chapter_to=chapter_to,
    )

    if output is None:
        return expand_path(output_dir) / auto_filename

    output_path = expand_path(output)

    # If --output points to a CSV file, keep exact backward-compatible behavior.
    if output_path.suffix.lower() == ".csv":
        return output_path

    # If --output is a directory/path without .csv, put auto filename inside it.
    return output_path / auto_filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate chapter-level ML cleanup quality. "
            "Default project layout: original=data/chapters-long, result=data/chapters-results."
        )
    )

    parser.add_argument(
        "--original-dir",
        default="data/chapters-long",
        help="Original chapters directory. Default: data/chapters-long",
    )

    parser.add_argument(
        "--result-dir",
        default="data/chapters-results",
        help="Processed chapters directory. Default: data/chapters-results",
    )

    parser.add_argument(
        "--result-suffix",
        default="_result",
        help="Result suffix. Default: _result",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path or output directory. "
            "If omitted, filename is generated from --chapter / --from-chapter / --to-chapter."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="reports/evaluation",
        help="Default output directory when --output is omitted. Default: reports/evaluation",
    )

    parser.add_argument(
        "--chapter",
        default=None,
        help="Evaluate only one chapter. Example: --chapter 033",
    )

    parser.add_argument(
        "--chapters",
        default=None,
        help=(
            "Evaluate selected chapters. "
            "Examples: --chapters 2,34,24 or --chapters 003-005,012"
        ),
    )

    parser.add_argument(
        "--from-chapter",
        default=None,
        help="Evaluate from this chapter. Example: --from-chapter 003",
    )

    parser.add_argument(
        "--to-chapter",
        default=None,
        help="Evaluate up to this chapter. Example: --to-chapter 175",
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

    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Do not fail when requested chapters are missing. "
            "Missing chapters will be skipped with warnings."
        ),
    )

    args = parser.parse_args()

    selection_modes = [
        bool(args.chapter),
        bool(args.chapters),
        bool(args.from_chapter or args.to_chapter),
    ]

    if sum(selection_modes) > 1:
        raise ValueError(
            "Use only one selection mode: --chapter, --chapters, or --from-chapter/--to-chapter."
        )

    original_dir = expand_path(args.original_dir)
    result_dir = expand_path(args.result_dir)

    if not original_dir.exists():
        raise FileNotFoundError(f"Original directory not found: {original_dir}")

    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    pairs = discover_pairs(
        original_dir=original_dir,
        result_dir=result_dir,
        result_suffix=args.result_suffix,
        chapter=args.chapter,
        chapters=args.chapters,
        chapter_from=args.from_chapter,
        chapter_to=args.to_chapter,
        allow_missing=args.allow_missing,
    )

    if not pairs:
        print("No pairs found.")
        print("Expected:")
        print(f"  {original_dir}/003.png")
        print(f"  {result_dir}/003_result.png")
        return

    print(f"Original dir: {original_dir}")
    print(f"Result dir:   {result_dir}")
    output_path = resolve_output_path(
        output=args.output,
        output_dir=args.output_dir,
        chapter=args.chapter,
        chapters=args.chapters,
        chapter_from=args.from_chapter,
        chapter_to=args.to_chapter,
    )
    print(f"Output CSV:   {output_path}")
    print(f"Found pairs:  {len(pairs)}")

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

    save_summary(scores, output_path)
    print_summary(scores, args.print_limit)

    print(f"Saved summary: {output_path}")


if __name__ == "__main__":
    main()
