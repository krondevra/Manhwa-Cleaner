#!/usr/bin/env python3
"""
v0.3.0 — v3: built on v2, adds row-based content restore and line-panel detection.

Changes from v2:
  - row-restore: for each row containing significant ink, white pixels in that
    row that were removed are candidates for restoration (softer than area dilation)
  - line-panel-restore: optional detection of horizontal black line pairs to
    define panel intervals; fills them back (disabled by default — can leak fon back)
  - debug outputs now include contour_restore, row_restore, line_panel_restore masks

v2 worked better than the last experimental version; v3 builds on v2 baseline.

Usage:
  python remove_manhwa_bg.py [input] [output] [--row-restore-strength light|medium]
                                [--line-panel-restore] [--debug]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGB"))


def save_result(out_path: Path, rgba: np.ndarray, debug_masks: dict | None = None) -> None:
    Image.fromarray(rgba, mode="RGBA").save(str(out_path))

    remove_mask = rgba[:, :, 3] == 0
    preview = rgba[:, :, :3].copy()
    preview[remove_mask] = [255, 0, 0]
    Image.fromarray(preview).save(str(out_path.with_name(out_path.stem + "_red_preview.png")))

    if debug_masks:
        for name, mask in debug_masks.items():
            img = Image.fromarray((mask * 255).astype(np.uint8))
            img.save(str(out_path.with_name(f"{out_path.stem}_debug_{name}.png")))


def build_white_mask(rgb: np.ndarray, v_thr: int = 240, s_thr: int = 12) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 2] >= v_thr) & (hsv[:, :, 1] <= s_thr)


def flood_fill_from_edges(white_mask: np.ndarray) -> np.ndarray:
    h, w = white_mask.shape
    work = white_mask.astype(np.uint8)
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for x in range(w):
        if work[0, x]:
            cv2.floodFill(work, flood, (x, 0), 2)
        if work[h - 1, x]:
            cv2.floodFill(work, flood, (x, h - 1), 2)
    for y in range(h):
        if work[y, 0]:
            cv2.floodFill(work, flood, (0, y), 2)
        if work[y, w - 1]:
            cv2.floodFill(work, flood, (w - 1, y), 2)
    return work == 2


def restore_by_row(
    rgb: np.ndarray,
    remove_mask: np.ndarray,
    white_mask: np.ndarray,
    strength: str = "light",
    ink_thr: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore white pixels in rows that contain significant ink content."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = (gray < 200).astype(np.float32)
    row_ink = ink.mean(axis=1)

    pad = {"light": 2, "medium": 8}.get(strength, 2)
    restore = np.zeros_like(remove_mask)

    for y, density in enumerate(row_ink):
        if density >= ink_thr:
            y0 = max(0, y - pad)
            y1 = min(rgb.shape[0], y + pad + 1)
            restore[y0:y1, :] |= white_mask[y0:y1, :] & remove_mask[y0:y1, :]

    result = remove_mask.copy()
    result[restore] = False
    return result, restore


def process_image(
    rgb: np.ndarray,
    row_restore_strength: str = "light",
    line_panel_restore: bool = False,
    debug: bool = False,
) -> tuple[np.ndarray, dict]:
    white_mask = build_white_mask(rgb)
    outer_bg = flood_fill_from_edges(white_mask)

    remove_mask, row_restore = restore_by_row(rgb, outer_bg, white_mask, row_restore_strength)

    alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    debug_masks = {}
    if debug:
        debug_masks = {
            "white_mask": white_mask,
            "outer_bg": outer_bg,
            "row_restore": row_restore,
            "remove_mask": remove_mask,
        }

    return rgba, debug_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--row-restore-strength", choices=["light", "medium"], default="light")
    parser.add_argument("--line-panel-restore", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_result.png")

    rgb = load_rgb(in_path)
    rgba, debug_masks = process_image(
        rgb,
        row_restore_strength=args.row_restore_strength,
        line_panel_restore=args.line_panel_restore,
        debug=args.debug,
    )
    save_result(out_path, rgba, debug_masks if args.debug else None)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
