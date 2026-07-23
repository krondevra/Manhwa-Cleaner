"""Ensemble combiner: CascadePSP zero-shot + P&C-finetuned refinement.

Phase A of .tmp/notes/manual_clean_quality_plan.md. Goal: get closer to
manual-clean quality by combining each checkpoint's strength --

- CascadePSP zero-shot: excellent gutter/SFX-halo cleanup (near manual-clean),
  but carves up real low-texture art (skies/seas/flat fills) as if background.
- CascadePSP finetuned on P&C (data/models/cascadepsp-pc-finetune-1.0.pth):
  fixes the art-carving problem, but gives back some of zero-shot's halo
  cleanup quality -- a rebalancing, not a clean win (see
  docs/ml_strategy_history.md, search "CascadePSP finetuned on P&C", and
  .tmp/notes/cascadepsp_finetune_next_steps_thinking.md).

Since the two failure modes are spatially near-disjoint (art damage happens
INSIDE real content; leftover halos sit in gutter/background territory), a
deterministic combiner can take the finetuned output as the safe base and
admit zero-shot's extra deletions only where they can't plausibly be real
art. Same philosophy as --reclaim-islands / --repair-frames in ml_cleaner.py:
cheap, deterministic, composable, no new training.

Two rules implemented:
  component -- gate by connected-component of the finetuned KEEP mask. A
    component qualifies for zero-shot's opinion if zero-shot marks >= a
    fraction of it delete AND it's smaller than an area cap (halos are
    small/medium blobs; skies/panels are large -- the area cap is what
    protects real art even where zero-shot wants it gone).
  distance -- gate purely by distance from pixels both checkpoints agree
    are real content (cv2.distanceTransform). Accept zero-shot's extra
    deletions only beyond a radius from any agreed-keep pixel.

Requires .venv-cascadepsp (segmentation_refinement + CPU torch):
  .venv-cascadepsp/bin/python src/ensemble_refine.py screen
  .venv-cascadepsp/bin/python src/ensemble_refine.py full --rule component --frac 0.8 --area 60000
  .venv-cascadepsp/bin/python src/ensemble_refine.py full --rule distance --radius 8

Real-manhwa policy: GT chapters 001/002 are held-out evaluation only, never
training signal (docs/ml_strategy_history.md, "Core architecture").
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import ml_cleaner  # noqa: E402

sys.modules["__main__"].train_command = ml_cleaner.train_command

from ml_cleaner import (  # noqa: E402
    GuidanceParams,
    load_model,
    predict_delete_mask,
    reclaim_landlocked_delete_islands,
    save_red_preview,
)

import segmentation_refinement as sr  # noqa: E402

OUT = ROOT / ".tmp" / "ensemble_refine"
MODEL = ROOT / "data/models/10.0-baseline.pt"
FINETUNE_WEIGHTS = ROOT / "data/models/cascadepsp-pc-finetune-1.0.pth"
GT_DIR = ROOT / ".tmp/saved/chapters"
GT_CHAPTERS = ["001", "002"]
GT_BAND = 4000
MARGIN = 300

# Fixed screening subset: the biggest-diff bands already found by the
# zero-shot probe (gt001_diff0/1/2) and the finetuned probe (gt001_diff0,
# gt002_diff0) -- exact (chapter, y0, height) windows recovered from the
# saved preview filenames/sizes in .tmp/cascadepsp_probe(_finetuned)/.
SCREEN_BANDS = [
    ("001", 7850, 1747, "sky_carving"),      # zero-shot's biggest diff (art damage)
    ("001", 11410, 1241, "zeroshot_diff1"),  # zero-shot's 2nd biggest diff
    ("001", 65037, 2404, "halo_win"),        # the SFX-halo win/damage comparison band
    ("001", 56140, 2080, "mixed_win_damage"),  # finetuned's biggest diff (halo+art mixed)
    ("002", 68389, 2035, "uiboxes_win"),     # finetuned's biggest diff (clean UI-box win)
]


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.4f}%"


class EnsembleProber:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model, self.config = load_model(MODEL, self.device)
        self.threshold = float(self.config.get("threshold", 0.5))
        self.gp = GuidanceParams(
            threshold_value=int(self.config.get("threshold_value", 30)),
            morph_radius=int(self.config.get("morph_radius", 2)),
        )
        self.refiner_zs = sr.Refiner(device="cpu")
        self.refiner_ft = sr.Refiner(device="cpu")
        state = torch.load(str(FINETUNE_WEIGHTS), map_location="cpu", weights_only=False)
        stripped = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        self.refiner_ft.model.load_state_dict(stripped)
        self.refiner_ft.model.eval()

    def delete_mask(self, rgb: np.ndarray) -> np.ndarray:
        raw = predict_delete_mask(
            rgb=rgb, model=self.model, device=self.device, guidance_params=self.gp,
            tile_size=768, overlap=96, threshold=self.threshold, amp=False,
            sdt_fusion=False, sdt_fusion_band_radius=4, sdt_clamp_radius=8.0,
        )
        return reclaim_landlocked_delete_islands(raw)

    @staticmethod
    def _refine(refiner: "sr.Refiner", rgb: np.ndarray, delete: np.ndarray) -> np.ndarray:
        keep = np.where(delete, 0, 255).astype(np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        soft = refiner.refine(bgr, keep, fast=False, L=900)
        return soft <= 127

    def refine_both(self, rgb: np.ndarray, delete: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (delete_zs, delete_ft)."""
        return self._refine(self.refiner_zs, rgb, delete), self._refine(self.refiner_ft, rgb, delete)


def ensemble_component(delete_ft: np.ndarray, delete_zs: np.ndarray,
                        frac_thresh: float, area_max: int | None,
                        erode_px: int = 0) -> np.ndarray:
    """Rule 1: gate by connected component of the finetuned KEEP mask.

    erode_px > 0: label components on an ERODED copy of keep_ft first, not
    the raw mask. Found necessary by direct inspection (gt001 y=65037): an
    SFX halo often has no gap to the real art directly above/beside it in
    finetuned's own keep mask, so raw 8-connectivity fuses "halo + adjacent
    panel" into one component -- large enough (400k+ px here) to fail any
    sane area cap, diluting the zero-shot-delete-fraction test too (a real
    panel's near-zero disagreement swamps the halo's own high disagreement
    once merged). Eroding first breaks these weak bridges (verified: erosion
    10-30px cleanly separated a 33k-area halo, frac=0.64, from a 400k+-area
    panel, frac=0.004, on that exact band). A qualifying eroded component is
    dilated back by the same radius (intersected with the ORIGINAL keep_ft)
    to recover close to its true pre-erosion extent before the pixel swap,
    rather than leaving a shrunk/incomplete cleanup."""
    keep_ft = (~delete_ft).astype(np.uint8)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        label_source = cv2.erode(keep_ft, k)
    else:
        k = None
        label_source = keep_ft
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(label_source, connectivity=8)
    result = delete_ft.copy()
    for label in range(1, n_labels):  # label 0 is the delete-side background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area_max is not None and area >= area_max:
            continue
        comp = labels == label
        frac = np.count_nonzero(delete_zs & comp) / area
        if frac >= frac_thresh:
            if k is not None:
                comp = (cv2.dilate(comp.astype(np.uint8), k) > 0) & (keep_ft > 0)
            result[comp] = delete_zs[comp]
    return result


def ensemble_distance(delete_ft: np.ndarray, delete_zs: np.ndarray, radius: int) -> np.ndarray:
    """Rule 2: gate purely by distance from pixels both checkpoints agree are keep."""
    agreed_keep = (~delete_ft) & (~delete_zs)
    result = delete_ft.copy()
    candidate = delete_zs & ~delete_ft
    if radius <= 0:
        result[candidate] = True
        return result
    src = (~agreed_keep).astype(np.uint8)
    dist = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    result[candidate & (dist > radius)] = True
    return result


# erode=0 is the original (pre-fix) raw-connectivity rule, kept for direct
# before/after comparison. erode>0 breaks halo-to-art bridges (see
# ensemble_component's docstring) -- found necessary by direct inspection of
# the y=65037 band, where a halo was fused into a 400k+px panel component
# and failed every area/fraction threshold until erosion split them.
COMPONENT_COMBOS = [(f, 60000, e) for f in (0.5, 0.6, 0.7) for e in (0, 10, 15, 20)]
DISTANCE_COMBOS = [0, 8, 16]


def window_stats(mask: np.ndarray, gt_delete: np.ndarray) -> tuple[int, int]:
    over = int(np.count_nonzero(mask & ~gt_delete))
    under = int(np.count_nonzero(~mask & gt_delete))
    return over, under


def run_screen() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prober = EnsembleProber()
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # accumulators: combo_key -> [over, under, total_px]
    totals: dict[str, list[int]] = {}
    baseline_ft = [0, 0, 0]
    baseline_zs = [0, 0, 0]

    for ch, y0, h, label in SCREEN_BANDS:
        if ch not in gt_cache:
            rgb_full = np.asarray(Image.open(GT_DIR / f"{ch}.png").convert("RGB"))
            gt_full = np.asarray(Image.open(GT_DIR / f"{ch}_cleaned.png").split()[-1]) < 128
            gt_cache[ch] = (rgb_full, gt_full)
        rgb_full, gt_full = gt_cache[ch]
        H = rgb_full.shape[0]
        ys, ye = max(0, y0 - MARGIN), min(H, y0 + h + MARGIN)
        rgb = rgb_full[ys:ye]
        gt_win = gt_full[y0:y0 + h]
        ly = y0 - ys

        t0 = time.time()
        islands = prober.delete_mask(rgb)
        delete_zs, delete_ft = prober.refine_both(rgb, islands)
        dt = time.time() - t0
        w_zs, w_ft = delete_zs[ly:ly + h], delete_ft[ly:ly + h]
        w_rgb = rgb[ly:ly + h]

        o, u = window_stats(w_ft, gt_win)
        baseline_ft[0] += o; baseline_ft[1] += u; baseline_ft[2] += w_ft.size
        o, u = window_stats(w_zs, gt_win)
        baseline_zs[0] += o; baseline_zs[1] += u; baseline_zs[2] += w_zs.size

        print(f"[{ch} y={y0} {label}] refined both in {dt:.1f}s ({w_ft.size} px window)")

        save_red_preview(OUT / f"{ch}_{label}_finetuned.png", w_rgb, w_ft)
        save_red_preview(OUT / f"{ch}_{label}_zeroshot.png", w_rgb, w_zs)

        for f, a, e in COMPONENT_COMBOS:
            key = f"component_f{f}_a{a}_e{e}"
            result = ensemble_component(w_ft, w_zs, f, a, erode_px=e)
            o, u = window_stats(result, gt_win)
            totals.setdefault(key, [0, 0, 0])
            totals[key][0] += o; totals[key][1] += u; totals[key][2] += result.size
            save_red_preview(OUT / f"{ch}_{label}_{key}.png", w_rgb, result)

        for r in DISTANCE_COMBOS:
            key = f"distance_r{r}"
            result = ensemble_distance(w_ft, w_zs, r)
            o, u = window_stats(result, gt_win)
            totals.setdefault(key, [0, 0, 0])
            totals[key][0] += o; totals[key][1] += u; totals[key][2] += result.size
            save_red_preview(OUT / f"{ch}_{label}_{key}.png", w_rgb, result)

    def row(name: str, stat: list[int]) -> None:
        o, u, t = stat
        print(f"  {name:26s} over-del {pct(o, t):>10}  under-del {pct(u, t):>10}  total {pct(o + u, t):>10}")

    print("\n=== screening totals across 5 fixed bands ===")
    row("finetuned (baseline)", baseline_ft)
    row("zero-shot (baseline)", baseline_zs)
    for key in sorted(totals):
        row(key, totals[key])

    print(f"\nPreviews saved to {OUT}/ -- inspect visually before promoting a combo to 'full'.")


def run_full(rule: str, frac: float, area: int | None, radius: int, erode: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prober = EnsembleProber()
    combo_desc = f"{rule}_f{frac}_a{area}_e{erode}" if rule == "component" else f"{rule}_r{radius}"
    print(f"== full GT eval, combo={combo_desc} ==")

    for ch in GT_CHAPTERS:
        rgb = np.asarray(Image.open(GT_DIR / f"{ch}.png").convert("RGB"))
        gt_delete = np.asarray(Image.open(GT_DIR / f"{ch}_cleaned.png").split()[-1]) < 128
        H = rgb.shape[0]
        total = gt_delete.size

        ensembled = np.zeros((H, rgb.shape[1]), dtype=bool)
        finetuned_only = np.zeros_like(ensembled)
        zeroshot_only = np.zeros_like(ensembled)
        n_bands = (H + GT_BAND - 1) // GT_BAND
        for b in range(n_bands):
            y = b * GT_BAND
            y0, y1 = max(0, y - MARGIN), min(H, y + GT_BAND + MARGIN)
            band_rgb = rgb[y0:y1]
            t0 = time.time()
            islands = prober.delete_mask(band_rgb)
            delete_zs, delete_ft = prober.refine_both(band_rgb, islands)
            if rule == "component":
                combined = ensemble_component(delete_ft, delete_zs, frac, area, erode_px=erode)
            else:
                combined = ensemble_distance(delete_ft, delete_zs, radius)
            dt = time.time() - t0

            ly, ly2 = y - y0, (min(H, y + GT_BAND)) - y0
            ensembled[y:y + (ly2 - ly)] = combined[ly:ly2]
            finetuned_only[y:y + (ly2 - ly)] = delete_ft[ly:ly2]
            zeroshot_only[y:y + (ly2 - ly)] = delete_zs[ly:ly2]
            if b % 5 == 0:
                print(f"  ch{ch} band {b + 1}/{n_bands} ({dt:.0f}s/band)")

        print(f"\n=== chapter {ch} ({rgb.shape[1]}x{H}) ===")
        for tag, m in (("finetuned only  ", finetuned_only),
                       ("zero-shot only  ", zeroshot_only),
                       (f"ensemble {combo_desc}", ensembled)):
            over = int(np.count_nonzero(m & ~gt_delete))
            under = int(np.count_nonzero(~m & gt_delete))
            print(f"  {tag}: over-del {over:>9} ({pct(over, total)})  "
                  f"under-del {under:>9} ({pct(under, total)})  total {pct(over + under, total)}")

        diff = ensembled != finetuned_only
        rows = np.flatnonzero(diff.any(axis=1))
        if len(rows):
            bands, start, prev = [], int(rows[0]), int(rows[0])
            for r in rows[1:]:
                r = int(r)
                if r - prev > 300:
                    bands.append((start, prev))
                    start = r
                prev = r
            bands.append((start, prev))
            bands.sort(key=lambda bb: -int(np.count_nonzero(diff[bb[0]:bb[1] + 1])))
            for i, (r0, r1) in enumerate(bands[:3]):
                p0, p1 = max(0, r0 - 150), min(H, r1 + 150)
                save_red_preview(OUT / f"full_{ch}_diff{i}_y{p0}_finetuned.png", rgb[p0:p1], finetuned_only[p0:p1])
                save_red_preview(OUT / f"full_{ch}_diff{i}_y{p0}_ensemble.png", rgb[p0:p1], ensembled[p0:p1])
            print(f"  saved previews for top {min(3, len(bands))} of {len(bands)} changed band(s)")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["screen", "full"])
    ap.add_argument("--rule", choices=["component", "distance"], default="component")
    ap.add_argument("--frac", type=float, default=0.6, help="component rule: min zero-shot-delete fraction")
    ap.add_argument("--area", type=int, default=60000, help="component rule: max component area px (0 = unlimited)")
    ap.add_argument("--erode", type=int, default=15,
                    help="component rule: erode keep_ft by this many px before labeling components, "
                    "to break weak bridges between a halo and adjacent art (0 = disabled, the "
                    "original raw-connectivity behavior -- found necessary by direct inspection, "
                    "see ensemble_component's docstring)")
    ap.add_argument("--radius", type=int, default=8, help="distance rule: px radius from agreed-keep content")
    args = ap.parse_args()

    area = None if args.area == 0 else args.area
    if args.mode == "screen":
        run_screen()
    else:
        run_full(args.rule, args.frac, area, args.radius, args.erode)


if __name__ == "__main__":
    main()
