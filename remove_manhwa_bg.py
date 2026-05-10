#!/usr/bin/env python3
"""
v0.2.0 — rule-based prototype: flood fill + panel detection.

Strategy:
  1. Build near-white mask (HSV: high V, low S).
  2. Detect full-width horizontal panel borders (long dark runs touching both edges).
  3. Classify borders as top/bottom by checking ink density above/below.
  4. Mark everything between a top/bottom pair as protected panel area.
  5. Flood-fill from image edges through white pixels that are NOT inside panels.
  6. Set flood-filled pixels to alpha=0.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgba(path: str, rgba: np.ndarray) -> None:
    Image.fromarray(rgba, mode="RGBA").save(path)


def longest_dark_runs(dark: np.ndarray) -> np.ndarray:
    """Return the longest continuous dark run for every image row."""
    h, _ = dark.shape
    runs = np.zeros(h, dtype=np.int32)

    for y in range(h):
        row = dark[y]
        padded = np.concatenate(([False], row, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if len(starts) > 0:
            runs[y] = np.max(ends - starts)

    return runs


def merge_close_rows(rows: np.ndarray, max_gap: int = 3) -> list[int]:
    """Merge nearby horizontal line rows into single y coordinates."""
    if len(rows) == 0:
        return []

    groups = []
    start = prev = int(rows[0])

    for r in rows[1:]:
        r = int(r)
        if r - prev <= max_gap:
            prev = r
        else:
            groups.append((start, prev))
            start = prev = r

    groups.append((start, prev))
    return [(a + b) // 2 for a, b in groups]


def detect_panel_intervals(
    rgb: np.ndarray,
    white_v_thr: int = 240,
    white_s_thr: int = 10,
    black_thr: int = 45,
    edge_margin: int = 5,
    line_run_ratio: float = 0.15,
    line_pixels_ratio: float = 0.15,
    inspect_window: int = 30,
    min_panel_height: int = 80,
) -> list[tuple[int, int]]:
    """
    Detect full-width panel intervals using horizontal black borders.

    A top border has more ink below; a bottom border has more ink above.
    """
    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = gray < black_thr

    dark_counts = dark.sum(axis=1)
    dark_runs = longest_dark_runs(dark)

    touches_left = dark[:, :edge_margin].any(axis=1)
    touches_right = dark[:, -edge_margin:].any(axis=1)

    candidate_rows = np.where(
        touches_left
        & touches_right
        & (dark_counts >= line_pixels_ratio * w)
        & (dark_runs >= line_run_ratio * w)
    )[0]

    lines = merge_close_rows(candidate_rows)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    erasable_white = (hsv[:, :, 2] >= white_v_thr) & (hsv[:, :, 1] <= white_s_thr)
    ink = ~erasable_white

    typed_lines = []

    for y in lines:
        above_1 = max(0, y - inspect_window - 3)
        above_2 = max(0, y - 3)
        below_1 = min(h, y + 3)
        below_2 = min(h, y + inspect_window + 3)

        above_ink = ink[above_1:above_2, :].mean() if above_2 > above_1 else 0.0
        below_ink = ink[below_1:below_2, :].mean() if below_2 > below_1 else 0.0

        if below_ink > above_ink + 0.05:
            typed_lines.append((y, "top"))
        elif above_ink > below_ink + 0.05:
            typed_lines.append((y, "bottom"))
        else:
            typed_lines.append((y, "unknown"))

    intervals = []
    open_top = None

    for y, line_type in typed_lines:
        if line_type == "top":
            if open_top is None:
                open_top = y
        elif line_type == "bottom":
            if open_top is not None and y - open_top >= min_panel_height:
                intervals.append((open_top, y))
                open_top = None

    return intervals


def remove_white_background(
    rgb: np.ndarray,
    white_v_thr: int = 240,
    white_s_thr: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    erasable_white = (hsv[:, :, 2] >= white_v_thr) & (hsv[:, :, 1] <= white_s_thr)

    panel_intervals = detect_panel_intervals(rgb, white_v_thr=white_v_thr, white_s_thr=white_s_thr)

    panel_mask = np.zeros((h, w), dtype=bool)
    for y1, y2 in panel_intervals:
        panel_mask[y1:y2 + 1, :] = True

    # White pixels inside panels are protected from flood fill.
    traversable_background = erasable_white & ~panel_mask

    work = traversable_background.astype(np.uint8)
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for x in range(w):
        if work[0, x] == 1:
            cv2.floodFill(work, flood_mask, (x, 0), 2)
        if work[h - 1, x] == 1:
            cv2.floodFill(work, flood_mask, (x, h - 1), 2)

    for y in range(h):
        if work[y, 0] == 1:
            cv2.floodFill(work, flood_mask, (0, y), 2)
        if work[y, w - 1] == 1:
            cv2.floodFill(work, flood_mask, (w - 1, y), 2)

    remove_mask = work == 2

    alpha = np.full((h, w), 255, dtype=np.uint8)
    alpha[remove_mask] = 0

    rgba = np.dstack([rgb, alpha])

    return rgba, remove_mask, panel_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input manhwa image")
    parser.add_argument("output", help="Output PNG with transparent background")
    parser.add_argument("--white-v", type=int, default=240)
    parser.add_argument("--white-s", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    rgb = load_rgb(args.input)
    rgba, remove_mask, panel_mask = remove_white_background(rgb, white_v_thr=args.white_v, white_s_thr=args.white_s)
    save_rgba(args.output, rgba)

    if args.debug:
        out = Path(args.output)
        Image.fromarray((remove_mask * 255).astype(np.uint8)).save(out.with_name(out.stem + "_remove_mask.png"))
        Image.fromarray((panel_mask * 255).astype(np.uint8)).save(out.with_name(out.stem + "_panel_mask.png"))


if __name__ == "__main__":
    main()
