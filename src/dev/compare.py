#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

DEFAULT_CHAPTERS_LONG_DIR = Path("data/chapters-long")
DEFAULT_COMPARE_DIR = Path("data/compare")
DEFAULT_RESULTS_DIR = Path("data/chapters-results")


@dataclass
class CompareImage:
    path: Path
    title: str
    subtitle: str


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def parse_csv_ints(value: str | None) -> list[int]:
    if not value:
        return []

    result: list[int] = []

    for item in re.split(r"[,\s]+", value.strip()):
        if not item:
            continue
        result.append(int(item))

    return result


def parse_csv_strings(value: str | None) -> list[str]:
    if not value:
        return []

    return [item.strip() for item in value.split(",") if item.strip()]


def load_image_as_rgb(path: str | Path, alpha_red_overlay: bool = True) -> np.ndarray:
    path = expand_path(path)
    img = Image.open(path)

    if img.mode == "RGBA" and alpha_red_overlay:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3]
        rgb[alpha < 128] = (255, 0, 0)
        return rgb

    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def chapter_sort_key(path_or_stem: str | Path) -> tuple[int, str]:
    stem = Path(str(path_or_stem)).stem

    if stem.endswith("_result_red_preview"):
        stem = stem[: -len("_result_red_preview")]
    elif stem.endswith("_red_preview"):
        stem = stem[: -len("_red_preview")]
    elif stem.endswith("_result"):
        stem = stem[: -len("_result")]

    numbers = re.findall(r"\d+", stem)

    if numbers:
        return (0, f"{int(numbers[-1]):09d}_{stem}")

    return (1, stem)


def resolve_chapter_path(chapter: str, chapters_dir: Path) -> Path:
    chapter_stem = Path(chapter).stem
    return chapters_dir / f"{chapter_stem}.png"


def default_output_path(chapter: str, output_dir: Path) -> Path:
    chapter_stem = Path(chapter).stem
    return output_dir / f"{chapter_stem}_compare_static.mp4"


def clean_label_from_path(path: Path, chapter: str) -> str:
    chapter_stem = Path(chapter).stem
    label = path.stem

    prefixes = [
        f"{chapter_stem}_",
        f"{chapter_stem}-",
    ]

    for prefix in prefixes:
        if label.startswith(prefix):
            label = label[len(prefix):]
            break

    suffixes = [
        "_result_red_preview",
        "_red_preview",
        "_result",
        "_cleaned",
    ]

    for suffix in suffixes:
        if label.endswith(suffix):
            label = label[: -len(suffix)]

    label = label.replace("_", " ").replace("-", " ").strip()

    if not label:
        return path.stem

    return label.upper()


def discover_compare_images(chapter: str, compare_dir: Path, results_dir: Path) -> list[Path]:
    chapter_stem = Path(chapter).stem
    candidates: list[Path] = []

    patterns = [
        compare_dir / f"{chapter_stem}_*_red_preview.png",
        compare_dir / f"{chapter_stem}_*.png",
        results_dir / f"{chapter_stem}_result_red_preview.png",
        results_dir / f"{chapter_stem}_result.png",
    ]

    seen: set[Path] = set()

    for pattern_path in patterns:
        parent = pattern_path.parent
        pattern = pattern_path.name

        if not parent.exists():
            continue

        for path in sorted(parent.glob(pattern), key=chapter_sort_key):
            if not path.is_file():
                continue

            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            if path in seen:
                continue

            seen.add(path)
            candidates.append(path)

    return candidates


def build_compare_images(
    chapter: str,
    result_paths: list[Path],
    labels: list[str],
) -> list[CompareImage]:
    items: list[CompareImage] = []

    for index, path in enumerate(result_paths, start=1):
        if index <= len(labels):
            label = labels[index - 1]
        else:
            label = clean_label_from_path(path, chapter)

        title = label if label else f"RESULT {index}"
        subtitle = path.name

        items.append(
            CompareImage(
                path=path,
                title=title,
                subtitle=subtitle,
            )
        )

    return items


def put_centered_text(
    img: np.ndarray,
    text: str,
    y: int,
    scale: float,
    thickness: int,
    color: tuple[int, int, int],
    video_w: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (video_w - text_w) // 2)

    cv2.putText(
        img,
        text,
        (x + 2, y + 2),
        font,
        scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_bottom_panel(
    frame_bgr: np.ndarray,
    title: str,
    subtitle: str,
    info_text: str,
    video_w: int,
    video_h: int,
    bottom_bar_h: int,
    title_scale: float,
    subtitle_scale: float,
    info_scale: float,
    title_thickness: int,
    subtitle_thickness: int,
    info_thickness: int,
) -> np.ndarray:
    frame = frame_bgr.copy()

    cv2.rectangle(
        frame,
        (0, video_h - bottom_bar_h),
        (video_w, video_h),
        (0, 0, 0),
        -1,
    )

    put_centered_text(
        frame,
        title,
        video_h - 62,
        title_scale,
        title_thickness,
        (255, 255, 255),
        video_w,
    )

    put_centered_text(
        frame,
        subtitle,
        video_h - 34,
        subtitle_scale,
        subtitle_thickness,
        (220, 220, 220),
        video_w,
    )

    put_centered_text(
        frame,
        info_text,
        video_h - 10,
        info_scale,
        info_thickness,
        (220, 220, 220),
        video_w,
    )

    return frame


def crop_view(
    img_rgb: np.ndarray,
    y_top: int,
    video_w: int,
    video_h: int,
    view_h: int,
    top_bar_h: int,
    zoom: float,
    bg_color: tuple[int, int, int],
) -> np.ndarray:
    h, _w, _ = img_rgb.shape

    source_view_h = int(round(view_h / zoom))
    source_view_w = int(round(video_w / zoom))

    if h <= source_view_h:
        y_top = 0
    else:
        y_top = max(0, min(y_top, h - source_view_h))

    crop = img_rgb[y_top:y_top + min(source_view_h, h), :, :]

    _, crop_w = crop.shape[:2]

    if crop_w > source_view_w:
        x0 = (crop_w - source_view_w) // 2
        crop = crop[:, x0:x0 + source_view_w, :]

    new_w = max(1, int(round(crop.shape[1] * zoom)))
    new_h = max(1, int(round(crop.shape[0] * zoom)))

    crop_resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full((video_h, video_w, 3), bg_color, dtype=np.uint8)

    x = (video_w - new_w) // 2
    y = top_bar_h

    crop_resized = crop_resized[:view_h, :, :]
    new_h, new_w = crop_resized.shape[:2]

    canvas[y:y + new_h, x:x + new_w] = crop_resized

    return rgb_to_bgr(canvas)


def make_still_frame(
    img_rgb: np.ndarray,
    y_top: int,
    title: str,
    subtitle: str,
    info_text: str,
    video_w: int,
    video_h: int,
    view_h: int,
    top_bar_h: int,
    bottom_bar_h: int,
    zoom: float,
    bg_color: tuple[int, int, int],
    title_scale: float,
    subtitle_scale: float,
    info_scale: float,
    title_thickness: int,
    subtitle_thickness: int,
    info_thickness: int,
) -> np.ndarray:
    frame = crop_view(
        img_rgb=img_rgb,
        y_top=y_top,
        video_w=video_w,
        video_h=video_h,
        view_h=view_h,
        top_bar_h=top_bar_h,
        zoom=zoom,
        bg_color=bg_color,
    )

    frame = draw_bottom_panel(
        frame_bgr=frame,
        title=title,
        subtitle=subtitle,
        info_text=info_text,
        video_w=video_w,
        video_h=video_h,
        bottom_bar_h=bottom_bar_h,
        title_scale=title_scale,
        subtitle_scale=subtitle_scale,
        info_scale=info_scale,
        title_thickness=title_thickness,
        subtitle_thickness=subtitle_thickness,
        info_thickness=info_thickness,
    )

    return frame


def crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0 - t, b, t, 0.0)


def repeat_frame(writer: cv2.VideoWriter, frame: np.ndarray, sec: float, fps: int) -> None:
    count = int(round(sec * fps))

    for _ in range(count):
        writer.write(frame)


def transition(writer: cv2.VideoWriter, a: np.ndarray, b: np.ndarray, sec: float, fps: int) -> None:
    count = int(round(sec * fps))

    for i in range(count):
        t = (i + 1) / count
        writer.write(crossfade(a, b, t))


def make_hotspots_from_centers(
    image_height: int,
    centers: list[int],
    view_h: int,
    zoom: float,
) -> list[int]:
    source_view_h = int(round(view_h / zoom))
    max_y = max(0, image_height - source_view_h)

    hotspots: list[int] = []

    for center_y in centers:
        y_top = int(center_y - source_view_h / 2)
        y_top = max(0, min(y_top, max_y))
        hotspots.append(y_top)

    return hotspots


def make_auto_centers(image_height: int, count: int) -> list[int]:
    if count <= 1:
        return [image_height // 2]

    margin = max(0, image_height // 12)

    start = margin
    end = max(start, image_height - margin)

    return [int(round(v)) for v in np.linspace(start, end, count)]


def validate_same_size(name_to_image: list[tuple[str, np.ndarray]]) -> None:
    if not name_to_image:
        return

    first_name, first_img = name_to_image[0]
    first_shape = first_img.shape

    for name, img in name_to_image[1:]:
        if img.shape != first_shape:
            raise ValueError(
                "All images must have the same dimensions.\n"
                f"  {first_name}: {first_shape[1]}x{first_shape[0]}\n"
                f"  {name}: {img.shape[1]}x{img.shape[0]}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a static comparison video for one long manhwa chapter. "
            "Default layout uses data/chapters-long, data/compare and data/chapters-results."
        )
    )

    parser.add_argument(
        "chapter",
        help="Chapter id or original image path. Example: 009 or data/chapters-long/009.png",
    )

    parser.add_argument(
        "--before",
        default=None,
        help="Original image path. Default: data/chapters-long/<chapter>.png",
    )

    parser.add_argument(
        "--results",
        nargs="+",
        default=None,
        help=(
            "Result images to compare. "
            "Can be *_red_preview.png or RGBA *_result.png. "
            "If omitted, script auto-discovers files in data/compare and data/chapters-results."
        ),
    )

    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated labels for results. Example: 'MODEL 1,MODEL 2'",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output video path. Default: data/compare/<chapter>_compare_static.mp4",
    )

    parser.add_argument("--chapters-dir", default=str(DEFAULT_CHAPTERS_LONG_DIR))
    parser.add_argument("--compare-dir", default=str(DEFAULT_COMPARE_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))

    parser.add_argument(
        "--centers",
        default=None,
        help="Comma/space separated vertical center Y values. Example: 18000,45950,59100",
    )

    parser.add_argument(
        "--auto-count",
        type=int,
        default=5,
        help="Number of automatic evenly spaced fragments when --centers is omitted. Default: 5",
    )

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-w", type=int, default=720)
    parser.add_argument("--video-h", type=int, default=1280)
    parser.add_argument("--zoom", type=float, default=0.80)

    parser.add_argument("--top-bar-h", type=int, default=0)
    parser.add_argument("--bottom-bar-h", type=int, default=95)

    parser.add_argument("--hold-sec", type=float, default=2.0)
    parser.add_argument("--transition-sec", type=float, default=0.60)
    parser.add_argument("--group-pause-sec", type=float, default=0.35)

    parser.add_argument(
        "--no-alpha-red-overlay",
        action="store_true",
        help="Do not convert transparent pixels in RGBA result images to red.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    chapters_dir = expand_path(args.chapters_dir)
    compare_dir = expand_path(args.compare_dir)
    results_dir = expand_path(args.results_dir)

    chapter_stem = Path(args.chapter).stem

    if args.before:
        before_path = expand_path(args.before)
    elif Path(args.chapter).suffix:
        before_path = expand_path(args.chapter)
        chapter_stem = before_path.stem
    else:
        before_path = resolve_chapter_path(args.chapter, chapters_dir)

    if not before_path.exists():
        raise FileNotFoundError(f"Original image not found: {before_path}")

    if args.results:
        result_paths = [expand_path(path) for path in args.results]
    else:
        result_paths = discover_compare_images(
            chapter=chapter_stem,
            compare_dir=compare_dir,
            results_dir=results_dir,
        )

    if not result_paths:
        raise FileNotFoundError(
            "No result images found.\n"
            "Use --results or place files like:\n"
            f"  {compare_dir}/{chapter_stem}_*_red_preview.png\n"
            f"  {results_dir}/{chapter_stem}_result.png"
        )

    missing = [path for path in result_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Result image(s) not found:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    labels = parse_csv_strings(args.labels)
    compare_items = build_compare_images(
        chapter=chapter_stem,
        result_paths=result_paths,
        labels=labels,
    )

    output_path = expand_path(args.output) if args.output else default_output_path(chapter_stem, compare_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    view_h = args.video_h - args.top_bar_h - args.bottom_bar_h

    if view_h <= 0:
        raise ValueError("Invalid layout: video_h must be larger than top_bar_h + bottom_bar_h")

    alpha_overlay = not args.no_alpha_red_overlay

    before = load_image_as_rgb(before_path, alpha_red_overlay=False)
    images: list[tuple[str, np.ndarray]] = [("BEFORE", before)]

    for item in compare_items:
        images.append((item.title, load_image_as_rgb(item.path, alpha_red_overlay=alpha_overlay)))

    validate_same_size(images)

    image_height = before.shape[0]

    centers = parse_csv_ints(args.centers)

    if not centers:
        centers = make_auto_centers(image_height, args.auto_count)

    hotspots = make_hotspots_from_centers(
        image_height=image_height,
        centers=centers,
        view_h=view_h,
        zoom=args.zoom,
    )

    print(f"Chapter: {chapter_stem}")
    print(f"Original: {before_path}")
    print("Results:")
    for item in compare_items:
        print(f"  - {item.title}: {item.path}")
    print(f"Output: {output_path}")
    print(f"Image height: {image_height}")
    print(f"Source viewport height: {int(round(view_h / args.zoom))}")
    print(f"Hotspot center Y values: {centers}")
    print(f"Hotspot top Y values: {hotspots}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        args.fps,
        (args.video_w, args.video_h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for: {output_path}")

    bg_color = (10, 14, 20)

    try:
        for hotspot_index, y_top in enumerate(hotspots):
            info_text = (
                f"CHAPTER {chapter_stem}  |  FRAGMENT {hotspot_index + 1}  "
                f"|  ZOOM {int(args.zoom * 100)}%"
            )

            frame_before = make_still_frame(
                img_rgb=before,
                y_top=y_top,
                title="BEFORE",
                subtitle=before_path.name,
                info_text=info_text,
                video_w=args.video_w,
                video_h=args.video_h,
                view_h=view_h,
                top_bar_h=args.top_bar_h,
                bottom_bar_h=args.bottom_bar_h,
                zoom=args.zoom,
                bg_color=bg_color,
                title_scale=1.00,
                subtitle_scale=0.85,
                info_scale=0.75,
                title_thickness=2,
                subtitle_thickness=2,
                info_thickness=2,
            )

            repeat_frame(writer, frame_before, args.hold_sec, args.fps)
            previous_frame = frame_before

            for item in compare_items:
                result_rgb = load_image_as_rgb(item.path, alpha_red_overlay=alpha_overlay)

                result_frame = make_still_frame(
                    img_rgb=result_rgb,
                    y_top=y_top,
                    title=item.title,
                    subtitle=item.subtitle,
                    info_text=info_text,
                    video_w=args.video_w,
                    video_h=args.video_h,
                    view_h=view_h,
                    top_bar_h=args.top_bar_h,
                    bottom_bar_h=args.bottom_bar_h,
                    zoom=args.zoom,
                    bg_color=bg_color,
                    title_scale=1.00,
                    subtitle_scale=0.85,
                    info_scale=0.75,
                    title_thickness=2,
                    subtitle_thickness=2,
                    info_thickness=2,
                )

                transition(writer, previous_frame, result_frame, args.transition_sec, args.fps)
                repeat_frame(writer, result_frame, args.hold_sec, args.fps)

                previous_frame = result_frame

            repeat_frame(writer, previous_frame, args.group_pause_sec, args.fps)

    finally:
        writer.release()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
