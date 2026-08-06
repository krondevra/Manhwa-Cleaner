"""Mission plan v2, Gate 1c (.claude/plans/snazzy-cuddling-creek.md): the black-backdrop class is
a DATA-DOMAIN gap, not a 9th learned-halo mechanism -- Gate 1a's probe found the model has
literally zero per-pixel probability signal on near-black content (recall 0.0000-0.0021 across
every threshold 0.05-0.95), the same saturated-prior signature bubble-halo showed, but here the
root cause is plausible and testable: the Stage-1 curriculum's synthetic backgrounds are white,
so the model never saw "background, but rendered black" as a training example. `black-1.0`'s
earlier failure was a SEPARATE, dedicated black-only model (a from-scratch shift, not a mixed
curriculum); docs/ml_strategy_history.md explicitly lists solid-black background as
UNTESTED-BUT-NOT-IMPLICATED for the mixed-curriculum approach tried here.

No sibling-repo (PepperNCarrotDataset) edits, per standing policy: this is a pure post-generation
RGB transform, reading already-generated Stage-1 pages + their GT read-only, writing a new mixed
dataset entirely under .tmp/datasets/ inside Manhwa-Cleaner. No generator changes, no new
generation run.

Transform (`_blackbg` variant): for each source page, wherever the paired GT says "delete"
(background, alpha<128), recolor that region of the RGB from its original (near-white) tone to a
near-black tone (with mild per-pixel noise and a soft gradient, not a flat fill, so the model
sees some texture variety rather than a single degenerate solid value). GT labels are UNCHANGED
-- the region was already correctly labeled "delete", it just now LOOKS like a black backdrop
instead of a white one, teaching the model backdrop can be either color.

2026-08-05, Attempt 2 (`_darkcontent` variant, after Attempt 1's real-but-flawed result -- see
notes/instance_aware_pivot_2026-08-04.md): Attempt 1 trained ONLY on "darkened region ->
GT-delete", giving the model zero contrastive signal that darkness alone doesn't determine the
label -- confirmed via a direct real-chapter regression (a real dark HUD-splash panel, previously
correctly kept in parts, became 27.66% wrongly deleted). This variant darkens GT-KEEP-labeled
CONTENT instead, via a multiplicative scale that PRESERVES the region's own internal
texture/gradient/linework (not a flat recolor -- flattening content would just teach a different,
equally shallow "flat=background" shortcut instead of "darkness isn't determinative, structure
is"). GT labels unchanged (still keep). Mirrors the real failure case directly: real dark art
(lightning, UI box linework) sitting on a real dark backdrop, same region, different labels.

2026-08-05, Attempt 2 RESULT: regressed past the untouched baseline on BOTH axes (aggregate
25.39% vs. baseline 20.19% vs. Attempt 1's 14.69%) -- root cause (mission plan v3): on this
curriculum "content" is mostly flat fills, so a darkened flat fill at scale 0.15-0.45 could land
visually IDENTICAL to the near-black (0-40 gray) background band -- large volumes of
near-identical patches with OPPOSITE labels, an irresolvable label conflict at patch scale that
degrades the whole black-decision into noise rather than teaching nuance. Attempt 2 also changed
TWO variables at once vs. Attempt 1 (reduced _blackbg count 250->175 AND added _darkcontent),
violating one-variable-per-run.

2026-08-05, Attempt 3 (mission plan v3, Step 2b -- only reached after Step 1's inference-time
fix also failed to generalize, see notes/instance_aware_pivot_2026-08-04.md): ONE variable off
Attempt 1's EXACT recipe (250 orig + 250 _blackbg, unchanged) -- adds a SMALL dose (+50 pages) of
a REVISED _darkcontent transform with a hard floor (`darkcontent_floor`, default 100) keeping
darkened content clearly above the 0-40 background band, avoiding Attempt 2's
identical-patch-opposite-label problem. The goal narrows accordingly: soften the
mid-darkness decision gradient, not teach the patch-irresolvable pure-black-art case (Step 1
already showed that case can't be fixed without hurting gutter recall elsewhere).

2026-08-05, Family B / MISSION PLAN v6 (`_darkpanel` variant, after Probes 0-1 closed the
post-hoc/inference-side options -- see notes/instance_aware_pivot_2026-08-04.md): Probe 0 found
the real dark-panel/HUD defect instances live in near-black connected components that are
PHYSICALLY FUSED to real gutters by flood-fill connectivity (a "mixed" gt_precision population,
not the clean near-zero-deletion dark_art population a pure region vote can already fix) -- so no
post-hoc mechanism can separate them at the component level. Probe 1 found the defect is
context-INDEPENDENT (0.66pp swing across tile_size 512-2048) -- no inference-side lever exists
either. That leaves only the training prior. Neither Attempt 1 (`_blackbg` alone) nor Attempt 3
(`_blackbg` + small floored `_darkcontent` dose) ever constructed the actual missing curriculum
example: a full-width dark PANEL band that is GT-KEEP, containing sparse bright boxes/text,
COEXISTING on the same page with dark GUTTER bands that are GT-DELETE (the real HUD page's
structure). `_darkpanel` authors that: paints a tall band near-black (identical recipe to
`_blackbg`), pastes N sparse bright rounded-rect UI-style boxes with text-like ink-dash noise at
realistic spacing, and authors GT for the whole band as KEEP -- applied ON TOP of an already
`_blackbg`-transformed page, so the page's real (untouched-GT) gutters and the new authored panel
coexist. This authors NEW GT (unlike every prior variant here, which only recolored under
UNCHANGED GT) -- legitimate for a fully-synthetic construction we fully control, per standing
discipline, but requires a rendered visual check of samples before any training spend (see
`.tmp/diagnostics/darkpanel_variant_visual_check.py`). Note `load_pair()` in `src/ml_cleaner.py`
only ever reads the cleaned file's ALPHA channel as the label -- the cleaned file's RGB is never
consumed by training, so it is safe to keep it as the plain original-page RGB for this variant.

Usage:
  .venv/bin/python src/research/make_black_backdrop_variant.py \
      --n-original 250 --n-blackbg 250 --n-darkcontent 50 --darkcontent-floor 100 \
      --out .tmp/datasets/stage1_blackbg_v3 --seed 20260805

  .venv/bin/python src/research/make_black_backdrop_variant.py \
      --n-original 250 --n-blackbg 250 --n-darkpanel 50 \
      --out .tmp/datasets/stage1_blackbg_v4 --seed 20260805
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SOURCE_ROOT = REPO_ROOT.parent / "PepperNCarrotDataset/data/dataset_curriculum/stage1_frames"

ALPHA_THRESHOLD = 128


def make_blackbg_variant(rgb: np.ndarray, gt_delete: np.ndarray, rng: random.Random) -> np.ndarray:
    """Recolors GT-delete-labeled background to a near-black tone with mild noise + a soft
    large-scale gradient (avoids a single degenerate flat value, matches the mild real-world
    variation seen in actual black backdrop scans -- e.g. chapter 001's real components measured
    mean_gray 0-33, std 1.7-10.5 within-component)."""
    h, w = gt_delete.shape
    base = rng.randint(0, 20)  # matches the real measured mean_gray range for backdrop
    # soft large-scale gradient via a low-res random field upsampled -- cheap, avoids a hard seam
    small = np.random.RandomState(rng.randint(0, 2**31 - 1)).uniform(-8, 8, size=(8, 8)).astype(np.float32)
    gradient = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, 4, size=(h, w)).astype(np.float32)
    value = np.clip(base + gradient + noise, 0, 40).astype(np.uint8)

    out = rgb.copy()
    for c in range(3):
        channel = out[:, :, c]
        channel[gt_delete] = value[gt_delete]
    return out


def make_darkcontent_variant(rgb: np.ndarray, gt_keep: np.ndarray, rng: random.Random,
                              floor: int = 100) -> np.ndarray:
    """Darkens GT-KEEP-labeled content via a multiplicative scale (preserves relative internal
    texture/gradient/linework -- a flat recolor here would just teach a different shortcut,
    'flat=background', instead of the actual needed lesson that darkness alone isn't
    determinative). Scale factor randomized per page so the model sees a range of darkness, not
    one fixed shift.

    `floor` (Attempt 3 revision, mission plan v3): hard per-pixel minimum after darkening, kept
    clearly above the 0-40 near-black background band. Attempt 2 used no floor (scale alone,
    0.15-0.45) and produced darkened content visually IDENTICAL to real background on this
    curriculum's flat fills -- large volumes of near-identical patches with opposite labels,
    which regressed training past baseline. The floor keeps the two classes separable at the
    pixel level even in the darkest case, so the softened lesson (mid-darkness gradient) can
    still be learned without the unresolvable pure-black conflict."""
    scale = rng.uniform(0.15, 0.45)  # dark enough to read as "dark art", not just mild shading
    out = rgb.astype(np.float32)
    for c in range(3):
        channel = out[:, :, c]
        channel[gt_keep] = np.maximum(channel[gt_keep] * scale, float(floor))
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_rounded_rect(img: np.ndarray, x: int, y: int, w: int, h: int, r: int,
                        color: tuple, thickness: int) -> None:
    r = max(1, min(r, w // 2, h // 2))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, thickness)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thickness)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness)
        cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness)


def make_darkpanel_variant(rgb: np.ndarray, gt_delete: np.ndarray, rng: random.Random,
                            band_height_range: tuple = (1500, 4500),
                            n_boxes_range: tuple = (3, 8),
                            box_spacing_range: tuple = (300, 600),
                            frame_line: bool = False) -> tuple:
    """Authors a new dark-PANEL band: a tall region painted near-black (identical recipe to
    make_blackbg_variant) containing N sparse bright rounded-rect UI-style boxes with
    text-like ink-dash noise at realistic spacing, GT-authored as whole-band KEEP. Meant to be
    applied to a page that has ALREADY had make_blackbg_variant applied (so real gutters, with
    unchanged GT, and this new authored panel, with new GT, coexist on the same page -- the
    missing curriculum example Probe 0/1 identified: neither prior variant ever taught the
    coexistence of dark-KEEP and dark-DELETE within receptive-field range of real structure).

    2026-08-05, Rung 2 (mission plan v7, Phase 1): Rung 1 (frame_line=False, the original
    construction) trained cleanly but REGRESSED the target class and the aggregate -- the
    authored panel band and a real gutter band share the IDENTICAL near-black paint recipe with
    ZERO structural marker distinguishing "bounded real panel content" from "open gutter" beyond
    the sparse UI boxes themselves. `frame_line=True` adds exactly that missing marker: a thin,
    bright, continuous border line at the band's true top/bottom edges (the real panel<->gutter
    transition), giving the network a hard, consistent structural cue that real gutters never
    have anywhere within their own extent. One isolated variable vs Rung 1 -- everything else
    (paint recipe, box construction, spacing, GT authoring) is unchanged.

    Returns (variant_rgb, new_gt_delete, (y0, y1)) -- new_gt_delete is gt_delete with the band
    forced to False (keep) regardless of what was there before; everywhere outside the band is
    unchanged. (y0, y1) is the band's own bounds, for diagnostics/visual-check tooling.
    """
    h, w = gt_delete.shape
    band_h = min(rng.randint(*band_height_range), h - 100)
    y0 = rng.randint(0, max(1, h - band_h))
    y1 = y0 + band_h

    out = rgb.copy()
    new_gt_delete = gt_delete.copy()

    base = rng.randint(0, 20)
    small = np.random.RandomState(rng.randint(0, 2**31 - 1)).uniform(-8, 8, size=(8, 8)).astype(np.float32)
    gradient = cv2.resize(small, (w, band_h), interpolation=cv2.INTER_CUBIC)
    noise = np.random.RandomState(rng.randint(0, 2**31 - 1)).normal(0, 4, size=(band_h, w)).astype(np.float32)
    value = np.clip(base + gradient + noise, 0, 40).astype(np.uint8)
    for c in range(3):
        out[y0:y1, :, c] = value
    new_gt_delete[y0:y1, :] = False

    if frame_line:
        line_thickness = rng.randint(2, 4)
        line_shade = rng.randint(170, 220)
        line_color = (line_shade, line_shade, line_shade)
        top_y = y0 + 2
        bottom_y = y1 - 2 - line_thickness
        cv2.line(out, (0, top_y), (w - 1, top_y), line_color, line_thickness)
        cv2.line(out, (0, bottom_y), (w - 1, bottom_y), line_color, line_thickness)

    n_boxes = rng.randint(*n_boxes_range)
    cursor_y = y0 + rng.randint(40, 150)
    box_color_base = rng.randint(160, 230)
    for _ in range(n_boxes):
        box_h = rng.randint(60, 160)
        if cursor_y + box_h > y1 - 40:
            break
        box_w = int(w * rng.uniform(0.35, 0.75))
        margin = max(1, int(w * 0.05))
        box_x = rng.randint(margin, max(margin + 1, w - box_w - margin))
        box_y = cursor_y

        shade = box_color_base + rng.randint(-15, 15)
        box_color = (shade, shade, shade)
        radius = min(18, box_h // 4, box_w // 4)
        _draw_rounded_rect(out, box_x, box_y, box_w, box_h, radius, box_color, -1)
        border_shade = min(255, shade + 25)
        _draw_rounded_rect(out, box_x, box_y, box_w, box_h, radius,
                            (border_shade, border_shade, border_shade), 2)

        n_lines = rng.randint(2, 4)
        for li in range(n_lines):
            line_y = box_y + 20 + li * max(1, (box_h - 30) // max(n_lines, 1))
            line_x = box_x + 15
            remaining = box_w - 30
            while remaining > 20:
                dash_w = min(rng.randint(15, 45), remaining)
                ink = rng.randint(40, 90)
                cv2.rectangle(out, (line_x, line_y), (line_x + dash_w, line_y + 6),
                              (ink, ink, ink), -1)
                gap = rng.randint(8, 20)
                line_x += dash_w + gap
                remaining -= dash_w + gap

        cursor_y = box_y + box_h + rng.randint(*box_spacing_range)

    return out, new_gt_delete, (y0, y1)


def build_split(n_original: int, n_blackbg: int, n_darkcontent: int, out_dir: Path, seed: int,
                 source_root: Path, darkcontent_floor: int = 100, n_darkpanel: int = 0,
                 darkpanel_frame_line: bool = False) -> None:
    source_frames = source_root / "frames"
    source_cleaned = source_root / "frames_cleaned"

    rng = random.Random(seed)
    all_stems = sorted(p.stem for p in source_frames.glob("*.png"))
    rng.shuffle(all_stems)

    frames_out = out_dir / "ep1" / "frames"
    cleaned_out = out_dir / "ep1" / "frames_cleaned"
    frames_out.mkdir(parents=True, exist_ok=True)
    cleaned_out.mkdir(parents=True, exist_ok=True)

    total_needed = n_original + n_blackbg + n_darkcontent + n_darkpanel
    if len(all_stems) < total_needed:
        raise ValueError(f"only {len(all_stems)} source pages available, need {total_needed}")

    original_stems = all_stems[:n_original]
    blackbg_stems = all_stems[n_original : n_original + n_blackbg]
    darkcontent_stems = all_stems[n_original + n_blackbg : n_original + n_blackbg + n_darkcontent]
    darkpanel_stems = all_stems[n_original + n_blackbg + n_darkcontent : total_needed]

    for stem in original_stems:
        shutil.copy(source_frames / f"{stem}.png", frames_out / f"{stem}_orig.png")
        shutil.copy(source_cleaned / f"{stem}.png", cleaned_out / f"{stem}_orig.png")

    for stem in blackbg_stems:
        rgb = np.asarray(Image.open(source_frames / f"{stem}.png").convert("RGB"))
        cleaned = Image.open(source_cleaned / f"{stem}.png")
        gt_alpha = np.asarray(cleaned.split()[-1])
        gt_delete = gt_alpha < ALPHA_THRESHOLD

        variant_rgb = make_blackbg_variant(rgb, gt_delete, rng)
        Image.fromarray(variant_rgb).save(frames_out / f"{stem}_blackbg.png")
        shutil.copy(source_cleaned / f"{stem}.png", cleaned_out / f"{stem}_blackbg.png")

    for stem in darkcontent_stems:
        rgb = np.asarray(Image.open(source_frames / f"{stem}.png").convert("RGB"))
        cleaned = Image.open(source_cleaned / f"{stem}.png")
        gt_alpha = np.asarray(cleaned.split()[-1])
        gt_keep = gt_alpha >= ALPHA_THRESHOLD

        variant_rgb = make_darkcontent_variant(rgb, gt_keep, rng, floor=darkcontent_floor)
        Image.fromarray(variant_rgb).save(frames_out / f"{stem}_darkcontent.png")
        shutil.copy(source_cleaned / f"{stem}.png", cleaned_out / f"{stem}_darkcontent.png")

    for stem in darkpanel_stems:
        rgb = np.asarray(Image.open(source_frames / f"{stem}.png").convert("RGB"))
        cleaned = Image.open(source_cleaned / f"{stem}.png").convert("RGBA")
        gt_alpha = np.asarray(cleaned.split()[-1])
        gt_delete = gt_alpha < ALPHA_THRESHOLD

        # apply the existing _blackbg transform first (real gutters, GT unchanged), then carve
        # the authored dark-panel band on top (new GT: whole band keep) -- both labels coexist.
        bg_rgb = make_blackbg_variant(rgb, gt_delete, rng)
        variant_rgb, new_gt_delete, (band_y0, band_y1) = make_darkpanel_variant(
            bg_rgb, gt_delete, rng, frame_line=darkpanel_frame_line)
        Image.fromarray(variant_rgb).save(frames_out / f"{stem}_darkpanel.png")
        print(f"  {stem}_darkpanel: band y={band_y0}-{band_y1} (page height {rgb.shape[0]})")

        cleaned_rgb = np.asarray(cleaned.convert("RGB"))
        new_alpha = np.where(new_gt_delete, 0, 255).astype(np.uint8)
        new_cleaned = np.dstack([cleaned_rgb, new_alpha])
        Image.fromarray(new_cleaned, mode="RGBA").save(cleaned_out / f"{stem}_darkpanel.png")

    print(f"wrote {n_original} original + {n_blackbg} black-bg + {n_darkcontent} dark-content "
          f"(floor={darkcontent_floor}) + {n_darkpanel} dark-panel variant pages to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-original", type=int, required=True)
    ap.add_argument("--n-blackbg", type=int, required=True)
    ap.add_argument("--n-darkcontent", type=int, default=0,
                     help="contrastive dark-but-real-content pages (0 = Attempt 1 behavior)")
    ap.add_argument("--darkcontent-floor", type=int, default=100,
                     help="Attempt 3: hard per-pixel minimum after darkening content, keeps it "
                          "separable from the 0-40 background band (see make_darkcontent_variant)")
    ap.add_argument("--n-darkpanel", type=int, default=0,
                     help="Family B: authored dark-panel-band pages, GT authored as whole-band "
                          "keep, coexisting with real _blackbg gutters (see make_darkpanel_variant)")
    ap.add_argument("--darkpanel-frame-line", action="store_true",
                     help="Rung 2: add a thin bright border line at the panel band's true "
                          "top/bottom edges -- the structural differentiator missing from Rung 1 "
                          "(see make_darkpanel_variant's 2026-08-05 docstring update)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                     help="dir containing frames/ and frames_cleaned/ (default: Stage-1 train split)")
    args = ap.parse_args()
    build_split(args.n_original, args.n_blackbg, args.n_darkcontent, args.out, args.seed,
                args.source_root, args.darkcontent_floor, args.n_darkpanel,
                args.darkpanel_frame_line)


if __name__ == "__main__":
    main()
