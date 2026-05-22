#!/usr/bin/env python3
"""
longify.py — join a sequence of per-page PNG exports into a single long-strip PNG.

Webtoon chapters exported from Photoshop as individual page images (e.g.
001_page01.png, 001_page02.png, ...) must be stacked vertically before the
ML cleaner can process them, because the model relies on context from adjacent
panels.

Usage:
  python longify.py input_dir/ output.png
  python longify.py input_dir/ output.png --pattern "*.png"

The images are sorted by filename (natural alphabetical order).
Width must match across all pages; mismatched widths raise an error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def longify(pages_dir: Path, output_path: Path, pattern: str = "*.png") -> None:
    files = sorted(pages_dir.glob(pattern))
    if not files:
        sys.exit(f"No files matching {pattern!r} in {pages_dir}")

    print(f"Found {len(files)} pages; loading...")
    arrays = []
    width: int | None = None

    for f in files:
        img = Image.open(f).convert("RGBA")
        arr = np.array(img)
        w = arr.shape[1]
        if width is None:
            width = w
        elif w != width:
            sys.exit(f"Width mismatch: {f.name} is {w}px wide, expected {width}px")
        arrays.append(arr)
        print(f"  {f.name}: {arr.shape[1]}x{arr.shape[0]}")

    combined = np.concatenate(arrays, axis=0)
    total_h = combined.shape[0]
    print(f"Combined: {width}x{total_h}")
    Image.fromarray(combined, mode="RGBA").save(str(output_path))
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output")
    parser.add_argument("--pattern", default="*.png")
    args = parser.parse_args()
    longify(Path(args.input_dir), Path(args.output), args.pattern)


if __name__ == "__main__":
    main()
