#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Pair:
    name: str
    original_path: Path
    result_path: Path


@dataclass
class Candidate:
    chapter: str
    y0: int
    y1: int
    score: float
    delete_ratio: float
    damage_dark_ratio: float
    damage_edge_ratio: float
    damage_color_ratio: float
    leftover_grayline_ratio: float
    original_crop_path: str = ""
    result_crop_path: str = ""
    overlay_crop_path: str = ""


def load_original_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_result_rgba(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    img = Image.open(path)

    if img.mode == "RGBA":
        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        return rgba[:, :, :3], rgba[:, :, 3]

    return np.asarray(img.convert("RGB"), dtype=np.uint8), None


def crop_to_same_size(
    original: np.ndarray,
    result_rgb: np.ndarray,
    alpha: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    h = min(original.shape[0], result_rgb.shape[0])
    w = min(original.shape[1], result_rgb.shape[1])

    original = original[:h, :w]
    result_rgb = result_rgb[:h, :w]

    if alpha is not None:
        alpha = alpha[:h, :w]

    return original, result_rgb, alpha


def discover_pairs(original_dir: Path, result_dir: Path, result_suffix: str) -> list[Pair]:
    pairs: list[Pair] = []

    for result_path in sorted(result_dir.glob(f"*{result_suffix}.png")):
        if result_path.name.endswith("_red_preview.png"):
            continue

        stem = result_path.stem

        if not stem.endswith(result_suffix):
            continue

        chapter_name = stem[: -len(result_suffix)]
        original_path = original_dir / f"{chapter_name}.png"

        if not original_path.exists():
            print(f"WARNING: original not found: {original_path}")
            continue

        pairs.append(
            Pair(
                name=chapter_name,
                original_path=original_path,
                result_path=result_path,
            )
        )

    return pairs


def infer_deleted_mask(
    original_rgb: np.ndarray,
    result_rgb: np.ndarray,
    alpha: Optional[np.ndarray],
    alpha_threshold: int,
    rgb_white_threshold: int,
) -> np.ndarray:
    if alpha is not None:
        return alpha < alpha_threshold

    original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    result_gray = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY)

    original_hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    original_sat = original_hsv[:, :, 1]

    result_near_white = result_gray >= rgb_white_threshold
    original_not_plain_white = (original_gray < rgb_white_threshold) | (original_sat > 20)

    return result_near_white & original_not_plain_white


def build_suspicion_maps(
    original_rgb: np.ndarray,
    deleted_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    edges = cv2.Canny(gray, 50, 150) > 0
    kept_mask = ~deleted_mask

    deleted_dark = deleted_mask & (gray < 180)
    deleted_edges = deleted_mask & edges
    deleted_color = deleted_mask & (sat > 45)

    leftover_grayline = (
        kept_mask
        & edges
        & (gray >= 120)
        & (gray <= 245)
        & (sat <= 35)
    )

    return {
        "deleted_dark": deleted_dark,
        "deleted_edges": deleted_edges,
        "deleted_color": deleted_color,
        "leftover_grayline": leftover_grayline,
    }


def score_window(
    deleted_mask: np.ndarray,
    maps: dict[str, np.ndarray],
    y0: int,
    y1: int,
) -> Candidate:
    area = max(1, (y1 - y0) * deleted_mask.shape[1])

    delete_ratio = float(deleted_mask[y0:y1].mean())

    damage_dark_ratio = float(maps["deleted_dark"][y0:y1].sum() / area)
    damage_edge_ratio = float(maps["deleted_edges"][y0:y1].sum() / area)
    damage_color_ratio = float(maps["deleted_color"][y0:y1].sum() / area)
    leftover_grayline_ratio = float(maps["leftover_grayline"][y0:y1].sum() / area)

    score = (
        damage_dark_ratio * 18.0
        + damage_edge_ratio * 14.0
        + damage_color_ratio * 10.0
        + leftover_grayline_ratio * 8.0
        + abs(delete_ratio - 0.35) * 0.15
    )

    return Candidate(
        chapter="",
        y0=y0,
        y1=y1,
        score=score,
        delete_ratio=delete_ratio,
        damage_dark_ratio=damage_dark_ratio,
        damage_edge_ratio=damage_edge_ratio,
        damage_color_ratio=damage_color_ratio,
        leftover_grayline_ratio=leftover_grayline_ratio,
    )


def find_candidates_for_pair(
    pair: Pair,
    window_height: int,
    stride: int,
    alpha_threshold: int,
    rgb_white_threshold: int,
) -> list[Candidate]:
    print(f"Processing {pair.name}")

    original_rgb = load_original_rgb(pair.original_path)
    result_rgb, alpha = load_result_rgba(pair.result_path)
    original_rgb, result_rgb, alpha = crop_to_same_size(original_rgb, result_rgb, alpha)

    deleted_mask = infer_deleted_mask(
        original_rgb=original_rgb,
        result_rgb=result_rgb,
        alpha=alpha,
        alpha_threshold=alpha_threshold,
        rgb_white_threshold=rgb_white_threshold,
    )

    maps = build_suspicion_maps(original_rgb, deleted_mask)

    h = original_rgb.shape[0]
    candidates: list[Candidate] = []

    if h <= window_height:
        cand = score_window(deleted_mask, maps, 0, h)
        cand.chapter = pair.name
        return [cand]

    for y0 in range(0, h - window_height + 1, stride):
        y1 = y0 + window_height
        cand = score_window(deleted_mask, maps, y0, y1)
        cand.chapter = pair.name
        candidates.append(cand)

    last_y0 = max(0, h - window_height)

    if candidates and candidates[-1].y0 != last_y0:
        cand = score_window(deleted_mask, maps, last_y0, h)
        cand.chapter = pair.name
        candidates.append(cand)

    return candidates


def interval_overlap_ratio(a0: int, a1: int, b0: int, b1: int) -> float:
    overlap = max(0, min(a1, b1) - max(a0, b0))
    shortest = max(1, min(a1 - a0, b1 - b0))
    return overlap / shortest


def select_top_candidates(
    candidates: list[Candidate],
    top: int,
    max_overlap: float,
) -> list[Candidate]:
    selected: list[Candidate] = []

    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        conflict = False

        for chosen in selected:
            if cand.chapter != chosen.chapter:
                continue

            overlap = interval_overlap_ratio(cand.y0, cand.y1, chosen.y0, chosen.y1)

            if overlap > max_overlap:
                conflict = True
                break

        if not conflict:
            selected.append(cand)

        if len(selected) >= top:
            break

    return selected


def composite_result_on_white(result_rgb: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
    if alpha is None:
        return result_rgb.copy()

    a = alpha.astype(np.float32) / 255.0
    a = a[:, :, None]

    white = np.full_like(result_rgb, 255)
    out = result_rgb.astype(np.float32) * a + white.astype(np.float32) * (1.0 - a)

    return out.astype(np.uint8)


def apply_overlay(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = base_rgb.copy()

    color_img = np.zeros_like(out)
    color_img[:, :, 0] = color[0]
    color_img[:, :, 1] = color[1]
    color_img[:, :, 2] = color[2]

    out[mask] = (
        out[mask].astype(np.float32) * (1.0 - alpha)
        + color_img[mask].astype(np.float32) * alpha
    ).astype(np.uint8)

    return out


def make_overlay_crop(
    original_crop: np.ndarray,
    deleted_mask_crop: np.ndarray,
    maps_crop: dict[str, np.ndarray],
) -> np.ndarray:
    overlay = original_crop.copy()

    overlay = apply_overlay(overlay, deleted_mask_crop, (255, 0, 0), 0.25)

    damage = (
        maps_crop["deleted_dark"]
        | maps_crop["deleted_edges"]
        | maps_crop["deleted_color"]
    )

    overlay = apply_overlay(overlay, damage, (255, 0, 0), 0.75)
    overlay = apply_overlay(overlay, maps_crop["leftover_grayline"], (0, 255, 255), 0.85)

    return overlay


def save_candidate_crops(
    pair: Pair,
    candidates: list[Candidate],
    output_dir: Path,
    alpha_threshold: int,
    rgb_white_threshold: int,
) -> list[Candidate]:
    original_rgb = load_original_rgb(pair.original_path)
    result_rgb, alpha = load_result_rgba(pair.result_path)
    original_rgb, result_rgb, alpha = crop_to_same_size(original_rgb, result_rgb, alpha)

    deleted_mask = infer_deleted_mask(
        original_rgb=original_rgb,
        result_rgb=result_rgb,
        alpha=alpha,
        alpha_threshold=alpha_threshold,
        rgb_white_threshold=rgb_white_threshold,
    )

    maps = build_suspicion_maps(original_rgb, deleted_mask)
    result_on_white = composite_result_on_white(result_rgb, alpha)

    chapter_dir = output_dir / pair.name
    chapter_dir.mkdir(parents=True, exist_ok=True)

    updated: list[Candidate] = []

    for cand in candidates:
        y0 = cand.y0
        y1 = cand.y1

        original_crop = original_rgb[y0:y1]
        result_crop = result_on_white[y0:y1]

        maps_crop = {
            key: value[y0:y1]
            for key, value in maps.items()
        }

        deleted_crop = deleted_mask[y0:y1]

        overlay_crop = make_overlay_crop(
            original_crop=original_crop,
            deleted_mask_crop=deleted_crop,
            maps_crop=maps_crop,
        )

        prefix = f"{cand.chapter}_y{y0:06d}_{y1:06d}_score{cand.score:.4f}"

        original_path = chapter_dir / f"{prefix}_original.png"
        result_path = chapter_dir / f"{prefix}_result.png"
        overlay_path = chapter_dir / f"{prefix}_overlay.png"

        Image.fromarray(original_crop).save(original_path)
        Image.fromarray(result_crop).save(result_path)
        Image.fromarray(overlay_crop).save(overlay_path)

        cand.original_crop_path = str(original_path)
        cand.result_crop_path = str(result_path)
        cand.overlay_crop_path = str(overlay_path)

        updated.append(cand)

    return updated


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass

    return ImageFont.load_default()


def make_contact_sheet(candidates: list[Candidate], output_path: Path) -> None:
    if not candidates:
        return

    font = load_font(16)

    cell_w = 280
    cell_h = 260
    label_h = 80
    pad = 12

    cols = 3
    rows = len(candidates)

    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = rows * (cell_h + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)

    headers = ["original", "result", "overlay"]

    for col, header in enumerate(headers):
        x = pad + col * (cell_w + pad)
        draw.text((x, 4), header, font=font, fill=(255, 255, 255))

    for row, cand in enumerate(candidates):
        label = (
            f"{row + 1:03d} | {cand.chapter} | y {cand.y0}-{cand.y1}\n"
            f"score={cand.score:.4f} del={cand.delete_ratio:.3f}\n"
            f"dmg_dark={cand.damage_dark_ratio:.4f} "
            f"dmg_edge={cand.damage_edge_ratio:.4f} "
            f"gray={cand.leftover_grayline_ratio:.4f}"
        )

        y_base = pad + row * (cell_h + label_h + pad)

        images = [
            cand.original_crop_path,
            cand.result_crop_path,
            cand.overlay_crop_path,
        ]

        for col, img_path in enumerate(images):
            x = pad + col * (cell_w + pad)

            cell = Image.new("RGB", (cell_w, cell_h + label_h), (0, 0, 0))
            cell_draw = ImageDraw.Draw(cell)
            cell_draw.text((6, 5), label if col == 0 else "", font=font, fill=(255, 255, 255))

            img = Image.open(img_path).convert("RGB")
            img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

            cell.paste(img, ((cell_w - img.width) // 2, label_h))
            sheet.paste(cell, (x, y_base))

    sheet.save(output_path, quality=95)


def save_csv(candidates: list[Candidate], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "chapter",
                "y0",
                "y1",
                "score",
                "delete_ratio",
                "damage_dark_ratio",
                "damage_edge_ratio",
                "damage_color_ratio",
                "leftover_grayline_ratio",
                "original_crop_path",
                "result_crop_path",
                "overlay_crop_path",
            ]
        )

        for rank, cand in enumerate(candidates, start=1):
            writer.writerow(
                [
                    rank,
                    cand.chapter,
                    cand.y0,
                    cand.y1,
                    f"{cand.score:.8f}",
                    f"{cand.delete_ratio:.8f}",
                    f"{cand.damage_dark_ratio:.8f}",
                    f"{cand.damage_edge_ratio:.8f}",
                    f"{cand.damage_color_ratio:.8f}",
                    f"{cand.leftover_grayline_ratio:.8f}",
                    cand.original_crop_path,
                    cand.result_crop_path,
                    cand.overlay_crop_path,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find suspicious/problematic regions by comparing chapters-long originals and chapters-results ML outputs."
    )

    parser.add_argument(
        "--original-dir",
        default="chapters-long",
        help="Directory with original chapters. Default: chapters-long",
    )

    parser.add_argument(
        "--result-dir",
        default="chapters-results",
        help="Directory with processed results. Default: chapters-results",
    )

    parser.add_argument(
        "--result-suffix",
        default="_result",
        help="Result suffix. Default: _result",
    )

    parser.add_argument(
        "--output",
        default="problem_candidates",
        help="Output folder. Default: problem_candidates",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="How many top candidates to save. Default: 50",
    )

    parser.add_argument(
        "--window-height",
        type=int,
        default=690,
        help="Vertical window height. Default: 690",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=345,
        help="Vertical stride. Default: 345",
    )

    parser.add_argument(
        "--max-overlap",
        type=float,
        default=0.50,
        help="Maximum overlap between windows from same chapter. Default: 0.50",
    )

    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=16,
        help="Alpha below this value is treated as deleted. Default: 16",
    )

    parser.add_argument(
        "--rgb-white-threshold",
        type=int,
        default=245,
        help="Fallback threshold when result has no alpha. Default: 245",
    )

    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not original_dir.exists():
        raise FileNotFoundError(f"Original directory not found: {original_dir}")

    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    pairs = discover_pairs(
        original_dir=original_dir,
        result_dir=result_dir,
        result_suffix=args.result_suffix,
    )

    if not pairs:
        print("No pairs found.")
        print("Expected files like:")
        print(f"  {original_dir}/009.png")
        print(f"  {result_dir}/009_result.png")
        return

    print(f"Found pairs: {len(pairs)}")
    print(f"Original dir: {original_dir}")
    print(f"Result dir:   {result_dir}")

    all_candidates: list[Candidate] = []

    for pair in pairs:
        candidates = find_candidates_for_pair(
            pair=pair,
            window_height=args.window_height,
            stride=args.stride,
            alpha_threshold=args.alpha_threshold,
            rgb_white_threshold=args.rgb_white_threshold,
        )

        all_candidates.extend(candidates)

    selected = select_top_candidates(
        candidates=all_candidates,
        top=args.top,
        max_overlap=args.max_overlap,
    )

    final_candidates: list[Candidate] = []
    pairs_by_name = {pair.name: pair for pair in pairs}

    for chapter in sorted(set(c.chapter for c in selected)):
        pair = pairs_by_name[chapter]
        chapter_candidates = [c for c in selected if c.chapter == chapter]

        saved = save_candidate_crops(
            pair=pair,
            candidates=chapter_candidates,
            output_dir=output_dir,
            alpha_threshold=args.alpha_threshold,
            rgb_white_threshold=args.rgb_white_threshold,
        )

        final_candidates.extend(saved)

    final_candidates = sorted(final_candidates, key=lambda c: c.score, reverse=True)

    save_csv(final_candidates, output_dir / "candidates.csv")
    make_contact_sheet(final_candidates, output_dir / "contact_sheet.jpg")

    print()
    print(f"Saved candidates: {len(final_candidates)}")
    print(f"Output folder: {output_dir}")
    print(f"CSV: {output_dir / 'candidates.csv'}")
    print(f"Contact sheet: {output_dir / 'contact_sheet.jpg'}")
    print()
    print("Overlay colors:")
    print("  faint red = deleted by model")
    print("  strong red = likely damaged deleted content")
    print("  cyan = likely leftover gray-line artifact")


if __name__ == "__main__":
    main()
