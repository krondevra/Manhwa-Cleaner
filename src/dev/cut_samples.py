#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ==================================================
# DEFAULT SETTINGS
# Safe cutting: smaller samples, no cuts through content,
# and padding is not allowed to leak into neighboring content.
# ==================================================

ALPHA_THRESHOLD = 10
MIN_ROW_PIXELS = 5

# Rows separated by <= this gap are treated as one content block.
MERGE_GAP = 20

# Context around each crop.
PADDING_TOP = 80
PADDING_BOTTOM = 80

# Prevent crop padding from touching the next/previous content block.
BOUNDARY_MARGIN = 2

MIN_SEGMENT_HEIGHT = 40

# Preferred sample height.
# Important: this is NOT a hard cut through content.
TARGET_CROP_HEIGHT = 900

# Post-processing:
# Merge nearby/overlapping small samples only when the combined crop stays reasonable.
POST_MERGE_GAP = 120
POST_MERGE_MAX_HEIGHT = 1800
MIN_STANDALONE_HEIGHT = 650

# Optional unsafe mode. Disabled by default.
UNSAFE_SPLIT_OVERLAP = 160


@dataclass
class Segment:
    y0: int
    y1: int
    reason: str = "safe"

    @property
    def height(self) -> int:
        return self.y1 - self.y0


@dataclass
class Group:
    start_idx: int
    end_idx: int


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def resolve_paths(
    chapter: str,
    original: str | None,
    cleaned: str | None,
    output_dir: str | None,
) -> tuple[str, Path, Path, Path]:
    chapter = Path(chapter).stem

    original_path = Path(original) if original else Path("data") / "chapters-long" / f"{chapter}.png"
    cleaned_path = Path(cleaned) if cleaned else Path("data") / "temp" / chapter / f"{chapter}_cleaned.png"
    out_dir = Path(output_dir) if output_dir else Path("data") / "temp" / chapter / "cut-samples"

    return chapter, original_path, cleaned_path, out_dir


def get_content_mask(cleaned: Image.Image, alpha_threshold: int) -> np.ndarray:
    arr = np.asarray(cleaned, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    has_transparency = alpha.min() < 250

    if has_transparency:
        return alpha > alpha_threshold

    # Fallback for non-alpha images.
    # This is less reliable for black-background pages, so alpha PNG is preferred.
    not_white = np.any(rgb < 245, axis=2)

    max_c = rgb.max(axis=2).astype(np.int16)
    min_c = rgb.min(axis=2).astype(np.int16)
    saturation_like = (max_c - min_c) > 12

    return not_white | saturation_like


def rows_to_runs(active_rows: np.ndarray) -> list[tuple[int, int]]:
    ys = np.where(active_rows)[0]

    if len(ys) == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(ys[0])
    prev = int(ys[0])

    for y_raw in ys[1:]:
        y = int(y_raw)

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


def crop_bounds_for_group(
    runs: list[tuple[int, int]],
    group: Group,
    image_height: int,
    pad_top: int,
    pad_bottom: int,
    boundary_margin: int,
) -> tuple[int, int]:
    """
    Build crop bounds for a group of content runs.

    Key rule:
    padding may expand into empty background, but it must not cross into the
    previous or next content run. This prevents samples from catching a piece of
    the next frame/bubble by accident.
    """
    raw_y0 = runs[group.start_idx][0]
    raw_y1 = runs[group.end_idx][1]

    if group.start_idx > 0:
        prev_end = runs[group.start_idx - 1][1]
        top_limit = min(image_height, prev_end + boundary_margin)
    else:
        top_limit = 0

    if group.end_idx < len(runs) - 1:
        next_start = runs[group.end_idx + 1][0]
        bottom_limit = max(0, next_start - boundary_margin)
    else:
        bottom_limit = image_height

    y0 = max(top_limit, raw_y0 - pad_top)
    y1 = min(bottom_limit, raw_y1 + pad_bottom)

    # Safety: never crop away actual content belonging to this group.
    y0 = min(y0, raw_y0)
    y1 = max(y1, raw_y1)

    y0 = max(0, min(y0, image_height))
    y1 = max(0, min(y1, image_height))

    if y1 <= y0:
        return raw_y0, raw_y1

    return y0, y1


def group_runs_safely(
    runs: list[tuple[int, int]],
    image_height: int,
    target_height: int,
    pad_top: int,
    pad_bottom: int,
    boundary_margin: int,
    min_segment_height: int,
) -> list[Segment]:
    """
    Group content runs into samples without cutting through a content run.

    target_height is only a preferred size. If a continuous content block is
    taller than target_height, it is kept intact.
    """
    useful_runs = [(y0, y1) for y0, y1 in runs if (y1 - y0) >= min_segment_height]

    if not useful_runs:
        return []

    groups: list[Group] = []

    current_start = 0
    current_end = 0

    for idx in range(1, len(useful_runs)):
        candidate = Group(current_start, idx)
        cand_y0, cand_y1 = crop_bounds_for_group(
            runs=useful_runs,
            group=candidate,
            image_height=image_height,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            boundary_margin=boundary_margin,
        )

        if target_height > 0 and (cand_y1 - cand_y0) > target_height:
            groups.append(Group(current_start, current_end))
            current_start = idx
            current_end = idx
        else:
            current_end = idx

    groups.append(Group(current_start, current_end))

    segments: list[Segment] = []

    for group in groups:
        y0, y1 = crop_bounds_for_group(
            runs=useful_runs,
            group=group,
            image_height=image_height,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            boundary_margin=boundary_margin,
        )

        reason = "safe"

        if target_height > 0 and (y1 - y0) > target_height:
            reason = "oversized_content_block_kept_intact"

        segments.append(Segment(y0, y1, reason))

    return segments


def post_merge_small_or_nearby_segments(
    segments: list[Segment],
    post_merge_gap: int,
    post_merge_max_height: int,
    min_standalone_height: int,
) -> list[Segment]:
    """
    Merge nearby/overlapping output samples when the combined crop stays reasonable.

    This keeps cuts safe because it only merges; it never cuts through content.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: (s.y0, s.y1))
    result: list[Segment] = []

    for segment in segments:
        if not result:
            result.append(segment)
            continue

        last = result[-1]

        gap = segment.y0 - last.y1
        overlap_or_near = gap <= post_merge_gap
        combined_y0 = min(last.y0, segment.y0)
        combined_y1 = max(last.y1, segment.y1)
        combined_height = combined_y1 - combined_y0

        one_is_small = (
            last.height < min_standalone_height
            or segment.height < min_standalone_height
        )

        can_merge = (
            overlap_or_near
            and combined_height <= post_merge_max_height
            and (one_is_small or gap <= 0)
        )

        if can_merge:
            reasons = {last.reason, segment.reason}
            if "oversized_content_block_kept_intact" in reasons:
                reason = "merged_with_oversized_neighbor"
            else:
                reason = "merged_small_or_nearby"

            result[-1] = Segment(combined_y0, combined_y1, reason)
        else:
            result.append(segment)

    return result


def split_unsafe_by_height(
    segments: list[Segment],
    max_height: int,
    overlap: int,
    min_segment_height: int,
) -> list[Segment]:
    """
    Optional old behavior: split by fixed height even if content is continuous.
    Disabled unless --unsafe-split is passed.
    """
    if max_height <= 0:
        return segments

    result: list[Segment] = []

    for segment in segments:
        y0, y1 = segment.y0, segment.y1
        height = y1 - y0

        if height <= max_height:
            result.append(segment)
            continue

        step = max(1, max_height - overlap)
        start = y0

        while start < y1:
            end = min(start + max_height, y1)

            if end - start >= min_segment_height:
                result.append(Segment(start, end, "unsafe_forced_split"))

            if end >= y1:
                break

            start += step

    return result


def remove_near_duplicate_segments(
    segments: list[Segment],
    min_unique_shift: int,
) -> list[Segment]:
    if min_unique_shift <= 0 or not segments:
        return segments

    result: list[Segment] = []

    for segment in sorted(segments, key=lambda s: (s.y0, s.y1)):
        if not result:
            result.append(segment)
            continue

        last = result[-1]

        if (
            abs(segment.y0 - last.y0) < min_unique_shift
            and abs(segment.y1 - last.y1) < min_unique_shift
        ):
            continue

        result.append(segment)

    return result


def detect_segments(
    cleaned: Image.Image,
    alpha_threshold: int,
    padding_top: int,
    padding_bottom: int,
    boundary_margin: int,
    merge_gap: int,
    min_row_pixels: int,
    min_segment_height: int,
    target_height: int,
    post_merge_gap: int,
    post_merge_max_height: int,
    min_standalone_height: int,
    unsafe_split: bool,
    unsafe_max_height: int,
    unsafe_overlap: int,
    min_unique_shift: int,
) -> list[Segment]:
    _width, height = cleaned.size

    content_mask = get_content_mask(cleaned, alpha_threshold=alpha_threshold)

    row_counts = content_mask.sum(axis=1)
    active_rows = row_counts >= min_row_pixels

    raw_runs = rows_to_runs(active_rows)
    content_runs = merge_close_runs(raw_runs, merge_gap)

    segments = group_runs_safely(
        runs=content_runs,
        image_height=height,
        target_height=target_height,
        pad_top=padding_top,
        pad_bottom=padding_bottom,
        boundary_margin=boundary_margin,
        min_segment_height=min_segment_height,
    )

    segments = post_merge_small_or_nearby_segments(
        segments=segments,
        post_merge_gap=post_merge_gap,
        post_merge_max_height=post_merge_max_height,
        min_standalone_height=min_standalone_height,
    )

    if unsafe_split:
        segments = split_unsafe_by_height(
            segments=segments,
            max_height=unsafe_max_height,
            overlap=unsafe_overlap,
            min_segment_height=min_segment_height,
        )

    segments = remove_near_duplicate_segments(
        segments=segments,
        min_unique_shift=min_unique_shift,
    )

    return segments


def save_crop_pair(
    original: Image.Image,
    cleaned: Image.Image,
    y0: int,
    y1: int,
    out_dir: Path,
    chapter: str,
    index: int,
) -> None:
    width, _height = original.size

    original_crop = original.crop((0, y0, width, y1))
    cleaned_crop = cleaned.crop((0, y0, width, y1))

    name = f"{chapter}-{index:03d}"

    original_crop.save(out_dir / f"{name}.png")
    cleaned_crop.save(out_dir / f"{name}_cleaned.png")


def save_segments_csv(
    out_dir: Path,
    chapter: str,
    segments: list[Segment],
) -> None:
    csv_path = out_dir / f"{chapter}_segments.csv"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("index,y0,y1,height,reason,original,cleaned\n")

        for index, segment in enumerate(segments):
            name = f"{chapter}-{index:03d}"
            f.write(
                f"{index},{segment.y0},{segment.y1},{segment.y1 - segment.y0},"
                f"{segment.reason},{name}.png,{name}_cleaned.png\n"
            )


def split_pair(
    chapter: str,
    original_path: Path,
    cleaned_path: Path,
    out_dir: Path,
    padding_top: int,
    padding_bottom: int,
    boundary_margin: int,
    merge_gap: int,
    min_row_pixels: int,
    min_segment_height: int,
    target_height: int,
    post_merge_gap: int,
    post_merge_max_height: int,
    min_standalone_height: int,
    unsafe_split: bool,
    unsafe_max_height: int,
    unsafe_overlap: int,
    alpha_threshold: int,
    min_unique_shift: int,
    clear_output: bool,
) -> None:
    if not original_path.exists():
        raise FileNotFoundError(f"Original not found: {original_path}")

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned not found: {cleaned_path}")

    original = load_rgba(original_path)
    cleaned = load_rgba(cleaned_path)

    if original.size != cleaned.size:
        raise ValueError(
            f"Size mismatch: {original_path} {original.size} "
            f"vs {cleaned_path} {cleaned.size}"
        )

    width, height = original.size

    print(f"Chapter:  {chapter}")
    print(f"Original: {original_path}")
    print(f"Cleaned:  {cleaned_path}")
    print(f"Output:   {out_dir}")
    print(f"Size:     {width}x{height}")

    segments = detect_segments(
        cleaned=cleaned,
        alpha_threshold=alpha_threshold,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
        boundary_margin=boundary_margin,
        merge_gap=merge_gap,
        min_row_pixels=min_row_pixels,
        min_segment_height=min_segment_height,
        target_height=target_height,
        post_merge_gap=post_merge_gap,
        post_merge_max_height=post_merge_max_height,
        min_standalone_height=min_standalone_height,
        unsafe_split=unsafe_split,
        unsafe_max_height=unsafe_max_height,
        unsafe_overlap=unsafe_overlap,
        min_unique_shift=min_unique_shift,
    )

    if clear_output and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Segments: {len(segments)}")
    print(
        f"Settings: alpha_threshold={alpha_threshold}, "
        f"pad_top={padding_top}, pad_bottom={padding_bottom}, "
        f"boundary_margin={boundary_margin}, "
        f"merge_gap={merge_gap}, min_row_pixels={min_row_pixels}, "
        f"min_segment_height={min_segment_height}, "
        f"target_height={target_height}, "
        f"post_merge_gap={post_merge_gap}, "
        f"post_merge_max_height={post_merge_max_height}, "
        f"min_standalone_height={min_standalone_height}, "
        f"unsafe_split={unsafe_split}, "
        f"unsafe_max_height={unsafe_max_height}, unsafe_overlap={unsafe_overlap}, "
        f"min_unique_shift={min_unique_shift}"
    )

    oversized = 0
    merged = 0

    for index, segment in enumerate(segments):
        save_crop_pair(
            original=original,
            cleaned=cleaned,
            y0=segment.y0,
            y1=segment.y1,
            out_dir=out_dir,
            chapter=chapter,
            index=index,
        )

        if "oversized" in segment.reason:
            oversized += 1

        if "merged" in segment.reason:
            merged += 1

        print(
            f"[{index:03d}] y={segment.y0}-{segment.y1}, "
            f"h={segment.y1 - segment.y0}, {segment.reason}"
        )

    save_segments_csv(out_dir=out_dir, chapter=chapter, segments=segments)

    if oversized:
        print()
        print(
            f"Warning: {oversized} segment(s) are taller than target height, "
            "but they were kept intact to avoid cutting through content."
        )

    if merged:
        print(f"Post-merged segments: {merged}")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split original + cleaned long manhwa chapter into boundary-safe sample crops. "
            "Default layout: original=data/chapters-long/<CH>.png, "
            "cleaned=data/temp/<CH>/<CH>_cleaned.png, "
            "output=data/temp/<CH>/cut-samples."
        )
    )

    parser.add_argument("chapter", help="Chapter id or base name. Example: 033")

    parser.add_argument("--original", default=None)
    parser.add_argument("--cleaned", default=None)
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--pad-top", type=int, default=PADDING_TOP)
    parser.add_argument("--pad-bottom", type=int, default=PADDING_BOTTOM)
    parser.add_argument("--boundary-margin", type=int, default=BOUNDARY_MARGIN)
    parser.add_argument("--merge-gap", type=int, default=MERGE_GAP)
    parser.add_argument("--min-row-pixels", type=int, default=MIN_ROW_PIXELS)
    parser.add_argument("--min-segment-height", type=int, default=MIN_SEGMENT_HEIGHT)

    parser.add_argument(
        "--target-height",
        type=int,
        default=TARGET_CROP_HEIGHT,
        help=(
            "Preferred crop height. This is not a hard cut. "
            "Continuous content blocks taller than this are kept intact."
        ),
    )

    parser.add_argument(
        "--post-merge-gap",
        type=int,
        default=POST_MERGE_GAP,
        help="Merge nearby output samples if the gap is below this value.",
    )

    parser.add_argument(
        "--post-merge-max-height",
        type=int,
        default=POST_MERGE_MAX_HEIGHT,
        help="Maximum height allowed when post-merging small/nearby samples.",
    )

    parser.add_argument(
        "--min-standalone-height",
        type=int,
        default=MIN_STANDALONE_HEIGHT,
        help="Samples below this height are merged with a nearby neighbor when possible.",
    )

    parser.add_argument(
        "--unsafe-split",
        action="store_true",
        help="Allow fixed-height splitting through continuous content. Disabled by default.",
    )

    parser.add_argument(
        "--unsafe-max-height",
        type=int,
        default=TARGET_CROP_HEIGHT,
        help="Height used only with --unsafe-split.",
    )

    parser.add_argument(
        "--unsafe-overlap",
        type=int,
        default=UNSAFE_SPLIT_OVERLAP,
        help="Overlap used only with --unsafe-split.",
    )

    parser.add_argument("--alpha-threshold", type=int, default=ALPHA_THRESHOLD)

    parser.add_argument(
        "--min-unique-shift",
        type=int,
        default=16,
        help="Drop almost duplicate segments whose borders shift less than this value.",
    )

    parser.add_argument("--clear", action="store_true")

    args = parser.parse_args()

    chapter, original_path, cleaned_path, out_dir = resolve_paths(
        chapter=args.chapter,
        original=args.original,
        cleaned=args.cleaned,
        output_dir=args.output_dir,
    )

    split_pair(
        chapter=chapter,
        original_path=original_path,
        cleaned_path=cleaned_path,
        out_dir=out_dir,
        padding_top=args.pad_top,
        padding_bottom=args.pad_bottom,
        boundary_margin=args.boundary_margin,
        merge_gap=args.merge_gap,
        min_row_pixels=args.min_row_pixels,
        min_segment_height=args.min_segment_height,
        target_height=args.target_height,
        post_merge_gap=args.post_merge_gap,
        post_merge_max_height=args.post_merge_max_height,
        min_standalone_height=args.min_standalone_height,
        unsafe_split=args.unsafe_split,
        unsafe_max_height=args.unsafe_max_height,
        unsafe_overlap=args.unsafe_overlap,
        alpha_threshold=args.alpha_threshold,
        min_unique_shift=args.min_unique_shift,
        clear_output=args.clear,
    )


if __name__ == "__main__":
    main()
