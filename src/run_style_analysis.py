"""Style-analysis driver (Part 4, .tmp/notes/synthetic_curriculum_plan.md).

Read-only, genre-level statistics only -- see src/style_analysis.py's module docstring
for the constraint this must respect (never persist/reuse real contours/pixels downstream).

Samples chapters across all 5 manhwa series under .tmp/saved/materials/Merged/ (raw,
per-chapter-folder form -- NOT the one already-flattened series, sampled the same way as
the other 4 here for consistency), runs the enclosed-hole extractor + 5-family shape
classifier + frame/bubble split from style_analysis.py, and aggregates: shape-family
frequency, per-family aspect/curvature/jaggedness ranges, bubble-to-page size ratio, frame
border thickness, frame shape frequency, and page width/height distribution.

Usage:
  .venv/bin/python src/run_style_analysis.py smoke   # 1 chapter, report runtime
  .venv/bin/python src/run_style_analysis.py full     # full sample
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from style_analysis import extract_enclosed_holes, save_preview  # noqa: E402

MATERIALS = ROOT / ".tmp" / "saved" / "materials" / "Merged"
OUT = ROOT / ".tmp" / "style_analysis"
PREVIEW_DIR = OUT / "previews"

SERIES = [
    "Авто-охота с клонами",
    "Игрок, поедающий сталь",
    "Мастер меча, живущий на крыше",
    "Новая жизнь убийцы богов",
    "Чтобы покончить с богами, я не стал выбирать класс",
]

SEED = 20260726
CHAPTERS_PER_SERIES = 8
MAX_PAGES_PER_CHAPTER = 15  # runtime safety cap -- some raw "pages" are 40k+ px tall
PREVIEWS_PER_SERIES = 3


def sample_chapters(series_dir: Path, n: int, rng: random.Random) -> list[Path]:
    chapters = sorted(p for p in series_dir.iterdir() if p.is_dir())
    if len(chapters) <= n:
        return chapters
    return rng.sample(chapters, n)


def sample_pages(chapter_dir: Path, n: int, rng: random.Random) -> list[Path]:
    pages = sorted(p for p in chapter_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(pages) <= n:
        return pages
    return rng.sample(pages, n)


def run(mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    t0 = time.time()

    all_shapes: list[dict] = []
    page_dims: list[tuple[int, int]] = []  # (width, height)
    n_pages = 0
    n_chapters = 0
    per_series_preview_count = {s: 0 for s in SERIES}

    series_list = SERIES[:1] if mode == "smoke" else SERIES
    chapters_per = 1 if mode == "smoke" else CHAPTERS_PER_SERIES

    for series_name in series_list:
        series_dir = MATERIALS / series_name
        if not series_dir.exists():
            print(f"WARNING: series dir not found: {series_dir}")
            continue
        chosen_chapters = sample_chapters(series_dir, chapters_per, rng)
        print(f"== {series_name}: sampling {len(chosen_chapters)} chapters ==")
        for chapter_dir in chosen_chapters:
            pages = sample_pages(chapter_dir, MAX_PAGES_PER_CHAPTER, rng)
            if not pages:
                continue
            n_chapters += 1
            for page_path in pages:
                try:
                    rgb = np.asarray(Image.open(page_path).convert("RGB"))
                except Exception as exc:
                    print(f"  skip {page_path.name}: {exc}")
                    continue
                h, w = rgb.shape[:2]
                page_dims.append((w, h))
                n_pages += 1
                shapes = extract_enclosed_holes(rgb)
                for s in shapes:
                    s["series"] = series_name
                    s["page_w"], s["page_h"] = w, h
                all_shapes.extend(shapes)

                if (per_series_preview_count[series_name] < PREVIEWS_PER_SERIES
                        and shapes and h < 6000):  # skip previews on giant pages, too slow to save
                    safe_series = series_name.replace(" ", "_").replace(",", "")
                    save_preview(rgb, shapes, PREVIEW_DIR / f"{safe_series}_{chapter_dir.name}_{page_path.stem}.png")
                    per_series_preview_count[series_name] += 1

        elapsed = time.time() - t0
        print(f"  ({n_pages} pages, {len(all_shapes)} shapes so far, {elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nDone: {n_chapters} chapters, {n_pages} pages, {len(all_shapes)} enclosed shapes, {elapsed:.1f}s")
    if mode == "smoke":
        est_full = elapsed * len(SERIES) * CHAPTERS_PER_SERIES / max(n_chapters, 1)
        print(f"Estimated full-sample runtime: ~{est_full:.0f}s (~{est_full / 60:.1f}min) "
              f"for {len(SERIES)} series x {CHAPTERS_PER_SERIES} chapters")

    # ── aggregate stats ──
    frame_shapes = [s for s in all_shapes if s["is_frame"]]
    bubble_shapes = [s for s in all_shapes if not s["is_frame"]]

    def class_counts(shapes: list[dict]) -> dict:
        counts: dict[str, int] = {}
        for s in shapes:
            counts[s["class"]] = counts.get(s["class"], 0) + 1
        return counts

    def pct(arr: list[float]) -> dict:
        if not arr:
            return {}
        a = np.array(arr)
        return {str(p): float(np.percentile(a, p)) for p in (5, 10, 25, 50, 75, 90, 95)}

    summary = {
        "mode": mode, "elapsed_s": elapsed, "n_chapters": n_chapters, "n_pages": n_pages,
        "n_shapes_total": len(all_shapes),
        "page_width_pct": pct([d[0] for d in page_dims]),
        "page_height_pct": pct([d[1] for d in page_dims]),
        "bubble_class_counts": class_counts(bubble_shapes),
        "frame_class_counts": class_counts(frame_shapes),
        "bubble_aspect_pct": pct([s["aspect"] for s in bubble_shapes]),
        "bubble_solidity_pct": pct([s["solidity"] for s in bubble_shapes]),
        "bubble_jaggedness_pct": pct([s["jaggedness"] for s in bubble_shapes]),
        "bubble_area_frac_of_page_pct": pct(
            [s["area"] / (s["page_w"] * s["page_h"]) for s in bubble_shapes]),
        "frame_border_thickness_pct": pct(
            [s["border_thickness"] for s in frame_shapes if "border_thickness" in s]),
        "per_class_min_radius_pct": {
            cls: pct([s["min_radius"] for s in bubble_shapes if s["class"] == cls])
            for cls in set(s["class"] for s in bubble_shapes)
        },
    }
    out_path = OUT / f"summary_{mode}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    assert mode in ("smoke", "full")
    run(mode)
