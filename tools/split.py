#!/usr/bin/env python3
"""
split.py

Split a long vertical image into smaller vertical chunks.

Behavior:
    ./split.sh 033.png
creates:
    033-1.png
    033-2.png
    033-3.png
    ...

Default chunk height: 50000 px.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def split_image(
    input_path: Path,
    chunk_height: int,
    output_dir: Path | None = None,
    start_index: int = 1,
    overwrite: bool = True,
) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    if chunk_height <= 0:
        raise ValueError("chunk_height must be greater than 0.")

    output_dir = output_dir if output_dir is not None else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []

    with Image.open(input_path) as img:
        width, height = img.size
        stem = input_path.stem
        suffix = input_path.suffix or ".png"

        print(f"Input: {input_path}")
        print(f"Size: {width}x{height}")
        print(f"Chunk height: {chunk_height}")

        offset = 0
        index = start_index

        while offset < height:
            current_height = min(chunk_height, height - offset)

            box = (0, offset, width, offset + current_height)
            part = img.crop(box)

            output_path = output_dir / f"{stem}-{index}{suffix}"

            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Output already exists: {output_path}")

            part.save(output_path)

            print(f"Saved: {output_path} ({width}x{current_height})")

            output_files.append(output_path)

            offset += current_height
            index += 1

    print(f"Done. Parts created: {len(output_files)}")
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a long vertical image into chunks."
    )

    parser.add_argument("input", help="Input image path. Example: 033.png")
    parser.add_argument(
        "--chunk",
        type=int,
        default=50000,
        help="Chunk height in pixels. Default: 50000.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output parts. Default: same directory as input.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First part index. Default: 1.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if output files already exist.",
    )

    args = parser.parse_args()

    split_image(
        input_path=Path(args.input),
        chunk_height=args.chunk,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        start_index=args.start_index,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
