#!/usr/bin/env python3
"""
longify.py

Create long vertical chapter images from folders that contain page images.

Old behavior replacement:
    python src/longify.py data/chapters-initial --output-dir data/chapters-long

Expected input:
    data/chapters-initial/
      001/
        001.jpg
        002.jpg
        ...
      002/
        001.jpg
        002.jpg
        ...

Default output with --name-mode folder:
    data/chapters-long/001.png
    data/chapters-long/002.png

Useful commands:
    python src/longify.py data/chapters-initial --output-dir data/chapters-long
    python src/longify.py data/chapters-initial --output-dir data/chapters-long --chapter 033
    python src/longify.py data/chapters-initial --output-dir data/chapters-long --name-mode index --start-index 0
    python src/longify.py data/chapters-initial --output-dir data/chapters-long --resize-mode most-common
    python src/longify.py data/chapters-initial --output-dir data/chapters-long --no-overwrite
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def natural_key(path: Path) -> list[object]:
    """Natural sort: 2.png comes before 10.png."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def find_images(chapter_dir: Path) -> list[Path]:
    return sorted(
        [path for path in chapter_dir.iterdir() if is_image(path)],
        key=natural_key,
    )


def find_chapter_dirs(input_dir: Path, output_dir: Path | None = None) -> list[Path]:
    chapter_dirs: list[Path] = []

    for path in input_dir.iterdir():
        if not path.is_dir():
            continue

        if output_dir is not None:
            try:
                if path.resolve() == output_dir.resolve():
                    continue
            except FileNotFoundError:
                pass

        if find_images(path):
            chapter_dirs.append(path)

    return sorted(chapter_dirs, key=natural_key)


def get_image_sizes(image_paths: list[Path]) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []

    for path in image_paths:
        with Image.open(path) as img:
            sizes.append(img.size)

    return sizes


def choose_target_width(
    sizes: list[tuple[int, int]],
    resize_mode: str,
    fixed_width: int | None,
) -> int:
    widths = [width for width, _height in sizes]

    if fixed_width is not None:
        if fixed_width <= 0:
            raise ValueError("--width must be greater than 0.")
        return fixed_width

    if resize_mode == "none":
        if len(set(widths)) != 1:
            raise ValueError(
                "Images have different widths. "
                "Use --resize-mode most-common, --resize-mode max-width, "
                "--resize-mode min-width, or pass --width."
            )
        return widths[0]

    if resize_mode == "most-common":
        return Counter(widths).most_common(1)[0][0]

    if resize_mode == "max-width":
        return max(widths)

    if resize_mode == "min-width":
        return min(widths)

    raise ValueError(f"Unknown resize mode: {resize_mode}")


def scaled_size(width: int, height: int, target_width: int) -> tuple[int, int]:
    if width == target_width:
        return width, height

    ratio = target_width / width
    target_height = max(1, round(height * ratio))

    return target_width, target_height


def output_name_for_chapter(
    chapter_dir: Path,
    chapter_index: int,
    name_mode: str,
    start_index: int,
) -> str:
    if name_mode == "folder":
        return f"{chapter_dir.name}.png"

    if name_mode == "index":
        return f"{start_index + chapter_index:03}.png"

    raise ValueError(f"Unknown name mode: {name_mode}")


def merge_chapter(
    chapter_dir: Path,
    output_path: Path,
    resize_mode: str,
    fixed_width: int | None,
    background: str,
    overwrite: bool,
) -> None:
    image_paths = find_images(chapter_dir)

    if not image_paths:
        print(f"Skipped empty folder: {chapter_dir}")
        return

    if output_path.exists() and not overwrite:
        print(f"Skipped existing output: {output_path}")
        return

    sizes = get_image_sizes(image_paths)

    target_width = choose_target_width(
        sizes=sizes,
        resize_mode=resize_mode,
        fixed_width=fixed_width,
    )

    scaled_sizes = [
        scaled_size(width, height, target_width)
        for width, height in sizes
    ]

    total_height = sum(height for _width, height in scaled_sizes)

    mode = "RGBA" if background == "transparent" else "RGB"

    if background == "transparent":
        fill = (0, 0, 0, 0)
    elif background == "black":
        fill = (0, 0, 0)
    else:
        fill = (255, 255, 255)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Chapter: {chapter_dir.name}")
    print(f"  pages: {len(image_paths)}")
    print(f"  output: {output_path}")
    print(f"  size: {target_width}x{total_height}")

    result = Image.new(mode, (target_width, total_height), fill)

    y = 0

    for index, (image_path, target_size) in enumerate(zip(image_paths, scaled_sizes), start=1):
        with Image.open(image_path) as img:
            img = img.convert(mode)

            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            result.paste(img, (0, y))
            y += img.height

        print(f"    [{index}/{len(image_paths)}] {image_path.name}")

    result.save(output_path, "PNG")
    print(f"  saved: {output_path}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge page images from chapter folders into long vertical PNG chapters."
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Directory containing chapter folders. Default: current directory.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <input_dir>/merged.",
    )

    parser.add_argument(
        "--name-mode",
        choices=["index", "folder"],
        default="folder",
        help="Output naming mode. folder -> 033.png, index -> 000.png. Default: folder.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First index for --name-mode index. Default: 0.",
    )

    parser.add_argument(
        "--resize-mode",
        choices=["none", "most-common", "max-width", "min-width"],
        default="most-common",
        help="How to handle pages with different widths. Default: most-common.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Force output width in pixels. Overrides --resize-mode.",
    )

    parser.add_argument(
        "--background",
        choices=["white", "black", "transparent"],
        default="white",
        help="Canvas background. Default: white.",
    )

    parser.add_argument(
        "--chapter",
        default=None,
        help="Process only one chapter folder by name. Example: --chapter 033",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing output files.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "merged"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if args.chapter:
        chapter_dir = input_dir / args.chapter

        if not chapter_dir.exists():
            raise FileNotFoundError(f"Chapter folder not found: {chapter_dir}")

        chapter_dirs = [chapter_dir]
    else:
        chapter_dirs = find_chapter_dirs(input_dir=input_dir, output_dir=output_dir)

    if not chapter_dirs:
        print(f"No chapter folders with images found in: {input_dir}")
        raise SystemExit(1)

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Chapters found: {len(chapter_dirs)}")
    print()

    for chapter_index, chapter_dir in enumerate(chapter_dirs):
        output_name = output_name_for_chapter(
            chapter_dir=chapter_dir,
            chapter_index=chapter_index,
            name_mode=args.name_mode,
            start_index=args.start_index,
        )

        output_path = output_dir / output_name

        merge_chapter(
            chapter_dir=chapter_dir,
            output_path=output_path,
            resize_mode=args.resize_mode,
            fixed_width=args.width,
            background=args.background,
            overwrite=not args.no_overwrite,
        )

    print("Done.")


if __name__ == "__main__":
    main()
