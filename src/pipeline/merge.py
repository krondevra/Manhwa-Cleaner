#!/usr/bin/env python3
"""
merge.py

Merge cleaned vertical image parts into one long cleaned image.

Behavior:
    ./merge.sh 033
finds:
    033-*_cleaned.png
and saves:
    033_cleaned.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def natural_sort_key(path: Path) -> list[object]:
    """Sort paths naturally: 033-2_cleaned.png before 033-10_cleaned.png."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def find_cleaned_parts(prefix: str, directory: Path) -> list[Path]:
    return sorted(directory.glob(f"{prefix}-*_cleaned.png"), key=natural_sort_key)


def detect_output_mode(files: Iterable[Path]) -> str:
    """Use RGBA if at least one input has alpha. Otherwise use RGB."""
    for path in files:
        with Image.open(path) as img:
            if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
                return "RGBA"
    return "RGB"


def merge_images(
    files: list[Path],
    output_path: Path,
    allow_different_widths: bool = False,
) -> None:
    if not files:
        raise ValueError("No input files were provided.")

    mode = detect_output_mode(files)

    sizes: list[tuple[int, int]] = []
    for path in files:
        with Image.open(path) as img:
            sizes.append(img.size)

    widths = [w for w, _h in sizes]
    heights = [h for _w, h in sizes]

    if not allow_different_widths and len(set(widths)) != 1:
        details = "\n".join(
            f"  {path.name}: {size[0]}x{size[1]}"
            for path, size in zip(files, sizes)
        )
        raise ValueError(
            "Input images have different widths. "
            "Use --allow-different-widths to pad smaller images.\n"
            + details
        )

    output_width = max(widths)
    output_height = sum(heights)

    if mode == "RGBA":
        canvas = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", (output_width, output_height), (0, 0, 0))

    y = 0

    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] loading: {path}")

        with Image.open(path) as img:
            img = img.convert(mode)
            x = (output_width - img.width) // 2
            canvas.paste(img, (x, y))
            y += img.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    print(f"Saved: {output_path}")
    print(f"Size: {output_width}x{output_height}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge <prefix>-*_cleaned.png parts into <prefix>_cleaned.png."
    )

    parser.add_argument("prefix", help="Prefix of split cleaned parts. Example: 033")
    parser.add_argument("--dir", default=".", help="Directory containing parts.")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument(
        "--allow-different-widths",
        action="store_true",
        help="Pad images if widths differ instead of failing.",
    )

    args = parser.parse_args()

    directory = Path(args.dir)
    prefix = Path(args.prefix).stem

    files = find_cleaned_parts(prefix, directory)

    if not files:
        print(f"No files found: {directory / (prefix + '-*_cleaned.png')}")
        raise SystemExit(1)

    output_path = Path(args.output) if args.output else directory / f"{prefix}_cleaned.png"

    print(f"Prefix: {prefix}")
    print(f"Parts found: {len(files)}")
    for path in files:
        print(f"  {path}")

    merge_images(
        files=files,
        output_path=output_path,
        allow_different_widths=args.allow_different_widths,
    )


if __name__ == "__main__":
    main()
