#!/usr/bin/env python3
"""
v0.2.2 — magic-wand inspired: aggressive flood fill then selective restore.

Observations from manual workflow (2.png → 5.png):
  - Magic Wand from edges removes most background correctly.
  - White inside panels at edges also gets removed (problem).
  - Restoring white via panel selection fixes the leaked areas.
  - Trim 1px left/right border added during tool expansion.

This version closely mirrors that manual approach automatically.

Usage:
  python remove_manhwa_bg.py [input] [output] [--restore-strength none|light|medium|strong]
  python remove_manhwa_bg.py ./chapters ./out --folder --restore-strength light
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(str(path)).convert("RGB")
    return np.array(img)


def save_result(path: str | Path, rgba: np.ndarray, debug: bool = False) -> None:
    out = Path(path)
    Image.fromarray(rgba, mode="RGBA").save(str(out))

    if debug:
        remove_mask = rgba[:, :, 3] == 0
        preview = rgba[:, :, :3].copy()
        preview[remove_mask] = [255, 0, 0]
        Image.fromarray(preview).save(str(out.with_name(out.stem + "_red_preview.png")))


def build_white_mask(rgb: np.ndarray, v_thr: int = 240, s_thr: int = 12) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 2] >= v_thr) & (hsv[:, :, 1] <= s_thr)


def flood_fill_from_edges(white_mask: np.ndarray) -> np.ndarray:
    """Return mask of white pixels reachable from any image edge."""
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


def build_content_mask(rgb: np.ndarray, gray_thr: int = 220) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray < gray_thr


def restore_white_in_panels(
    rgb: np.ndarray,
    remove_mask: np.ndarray,
    white_mask: np.ndarray,
    strength: str = "medium",
) -> np.ndarray:
    """
    Find regions where white was removed but should be kept (inside panels).
    Returns an updated remove_mask with those regions restored.
    """
    if strength == "none":
        return remove_mask

    content = build_content_mask(rgb)
    kernel_size = {"light": 15, "medium": 35, "strong": 55}.get(strength, 35)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(content.astype(np.uint8), kernel).astype(bool)

    # White pixels that sit inside expanded content zones were likely panel-interior.
    restore = white_mask & dilated & remove_mask

    result = remove_mask.copy()
    result[restore] = False
    return result


def process_image(
    rgb: np.ndarray,
    restore_strength: str = "medium",
) -> np.ndarray:
    h, w = rgb.shape[:2]
    white_mask = build_white_mask(rgb)
    remove_mask = flood_fill_from_edges(white_mask)
    remove_mask = restore_white_in_panels(rgb, remove_mask, white_mask, strength=restore_strength)

    alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image or folder (with --folder)")
    parser.add_argument("output", nargs="?", default=None, help="Output path")
    parser.add_argument("--folder", action="store_true")
    parser.add_argument("--restore-strength", choices=["none", "light", "medium", "strong"], default="medium")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.folder:
        in_dir = Path(args.input)
        out_dir = Path(args.output) if args.output else in_dir.parent / (in_dir.name + "_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(in_dir.glob("*.png")):
            print(f"processing {img_path.name}...")
            rgb = load_rgb(img_path)
            rgba = process_image(rgb, args.restore_strength)
            save_result(out_dir / img_path.name, rgba, debug=args.debug)
    else:
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_result.png")
        rgb = load_rgb(in_path)
        rgba = process_image(rgb, args.restore_strength)
        save_result(out_path, rgba, debug=args.debug)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
