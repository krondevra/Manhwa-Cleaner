from PIL import Image
from pathlib import Path
from collections import Counter
import re

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "merged"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def get_target_width(images):
    widths = [img.width for img in images]
    return Counter(widths).most_common(1)[0][0]


def resize_to_width(img, target_width):
    if img.width == target_width:
        return img

    ratio = target_width / img.width
    new_height = int(img.height * ratio)

    return img.resize((target_width, new_height), Image.Resampling.LANCZOS)


def merge_chapter(chapter_dir: Path, output_path: Path):
    image_paths = sorted(
        [p for p in chapter_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key
    )

    if not image_paths:
        print(f"Skipped empty folder: {chapter_dir.name}")
        return

    images = [Image.open(p).convert("RGB") for p in image_paths]

    target_width = get_target_width(images)
    images = [resize_to_width(img, target_width) for img in images]

    total_height = sum(img.height for img in images)

    result = Image.new("RGB", (target_width, total_height), (255, 255, 255))

    y = 0
    for img in images:
        result.paste(img, (0, y))
        y += img.height

    result.save(output_path, "PNG")
    print(f"Saved: {output_path.name} <- {chapter_dir.name}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    chapter_dirs = sorted(
        [p for p in BASE_DIR.iterdir() if p.is_dir() and p.name != OUTPUT_DIR.name],
        key=natural_key
    )

    for index, chapter_dir in enumerate(chapter_dirs):
        output_path = OUTPUT_DIR / f"{index:03}.png"
        merge_chapter(chapter_dir, output_path)


if __name__ == "__main__":
    main()
