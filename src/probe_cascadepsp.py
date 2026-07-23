"""CascadePSP zero-shot refinement probe (2026-07-22).

Feeds 10.0-baseline + --reclaim-islands output (production path, unchanged)
through CascadePSP's pretrained class-agnostic refinement network
(github.com/hkchengrex/CascadePSP, pip package segmentation-refinement),
zero-shot, no finetuning -- to see whether an out-of-trunk refinement
mechanism moves bubble/panel boundaries in the right direction at all on
this content. Probe only: no training, no dataset changes.

Requires the isolated .venv-cascadepsp environment (CPU torch + torchvision
+ segmentation-refinement; pretrained weights auto-download to
~/.segmentation-refinement on first Refiner init):

  .venv-cascadepsp/bin/python src/probe_cascadepsp.py spots
  .venv-cascadepsp/bin/python src/probe_cascadepsp.py clauds
  .venv-cascadepsp/bin/python src/probe_cascadepsp.py gt
  .venv-cascadepsp/bin/python src/probe_cascadepsp.py all

Outputs (previews + stats) go to .tmp/cascadepsp_probe/.

Real-manhwa policy note: the gt set measures against the human-cleaned
manual-reference chapters as held-out evaluation targets -- explicitly
allowed, never training signal (see docs/ml_strategy_history.md, "Core
architecture" policy boundary, clarified 2026-07-22).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import ml_cleaner  # noqa: E402

# Checkpoints pickle their argparse Namespace whose set_defaults(func=...)
# points at __main__.train_command when saved by ml_cleaner.py run directly.
sys.modules["__main__"].train_command = ml_cleaner.train_command

from ml_cleaner import (  # noqa: E402
    GuidanceParams,
    load_model,
    predict_delete_mask,
    reclaim_landlocked_delete_islands,
    save_red_preview,
)

import segmentation_refinement as sr  # noqa: E402

OUT = ROOT / ".tmp" / "cascadepsp_probe"
MODEL = ROOT / "data/models/10.0-baseline.pt"
CHAPTER_085 = ROOT / "data/chapters-initial/085.png"
GT_DIR = ROOT / ".tmp/saved/chapters"

# The 18-coordinate broad spot-check set used for every eval since model 12.0
# (y-offsets on 085.png, recovered from data/compare/screenshots_17.0/).
SPOT_YS = [6800, 13800, 19100, 21700, 33200, 36700, 48800, 54650, 66900,
           112250, 120700, 155050, 161300, 165500, 169300, 172500, 177700, 179450]
SPOT_H = 900          # matches compare_models_video.py --crop-height
MARGIN = 300          # context rows around each window for inference/refinement

# Fixed clauds crop windows (.tmp/notes/clauds_regression_crops.md).
CLAUDS = [("clauds_1", 169250, 450), ("clauds_2", 179450, 450), ("clauds_3", 54550, 500)]

GT_BAND = 4000        # rows per band for full-chapter GT processing
GT_CHAPTERS = ["001", "002"]  # 035 excluded: _cleaned not pixel-aligned (162856 vs 162376)


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.4f}%"


class Prober:
    def __init__(self, fast: bool = False, weights: Path | None = None):
        self.device = torch.device("cpu")
        self.model, self.config = load_model(MODEL, self.device)
        self.threshold = float(self.config.get("threshold", 0.5))
        self.gp = GuidanceParams(
            threshold_value=int(self.config.get("threshold_value", 30)),
            morph_radius=int(self.config.get("morph_radius", 2)),
        )
        self.refiner = sr.Refiner(device="cpu")
        if weights is not None:
            # Our finetune runner (train_cascadepsp_pc.py) saves checkpoints
            # with a 'module.' prefix (matching what nn.DataParallel would
            # produce), the same convention segmentation_refinement.Refiner
            # itself expects and strips -- reuse that exact logic so both
            # the stock release checkpoint and our finetuned ones load the
            # same way.
            state = torch.load(str(weights), map_location="cpu", weights_only=False)
            stripped = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            self.refiner.model.load_state_dict(stripped)
            self.refiner.model.eval()
            print(f"loaded finetuned weights: {weights}")
        self.fast = fast

    def delete_mask(self, rgb: np.ndarray) -> np.ndarray:
        raw = predict_delete_mask(
            rgb=rgb, model=self.model, device=self.device, guidance_params=self.gp,
            tile_size=768, overlap=96, threshold=self.threshold, amp=False,
            sdt_fusion=False, sdt_fusion_band_radius=4, sdt_clamp_radius=8.0,
        )
        return reclaim_landlocked_delete_islands(raw)

    def refine(self, rgb: np.ndarray, delete: np.ndarray, fast: bool | None = None) -> np.ndarray:
        """CascadePSP refines a FOREGROUND mask; foreground = keep = ~delete.
        Package expects an OpenCV-style BGR image. Returns refined delete mask."""
        keep = np.where(delete, 0, 255).astype(np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        soft = self.refiner.refine(bgr, keep, fast=self.fast if fast is None else fast, L=900)
        return soft <= 127  # refined keep<=127 -> delete


def window_eval(prober: Prober, rgb_page: np.ndarray, name: str, y: int, h: int, out_dir: Path = OUT) -> dict:
    """Mask + refine on a margined slice, evaluate/preview the central window."""
    H = rgb_page.shape[0]
    y0, y1 = max(0, y - MARGIN), min(H, y + h + MARGIN)
    rgb = rgb_page[y0:y1]
    islands = prober.delete_mask(rgb)
    t = time.time()
    refined = prober.refine(rgb, islands)
    dt = time.time() - t

    ly = y - y0
    w_isl, w_ref, w_rgb = islands[ly:ly + h], refined[ly:ly + h], rgb[ly:ly + h]
    to_keep = int(np.count_nonzero(w_isl & ~w_ref))    # deletions removed
    to_del = int(np.count_nonzero(~w_isl & w_ref))     # deletions added
    save_red_preview(out_dir / f"{name}_islands.png", w_rgb, w_isl)
    save_red_preview(out_dir / f"{name}_cascadepsp.png", w_rgb, w_ref)
    print(f"  {name}: refine {dt:.1f}s | flips in window: {to_keep} del->keep, "
          f"{to_del} keep->del (window {w_isl.size} px)")
    return {"to_keep": to_keep, "to_del": to_del, "secs": dt}


def run_spots(prober: Prober, out_dir: Path = OUT) -> None:
    print(f"== spots: 18-coordinate broad set on {CHAPTER_085.name} (full-mode) ==")
    page = np.asarray(Image.open(CHAPTER_085).convert("RGB"))
    totals = {"to_keep": 0, "to_del": 0}
    for y in SPOT_YS:
        s = window_eval(prober, page, f"spot_y{y}", y, SPOT_H, out_dir)
        totals["to_keep"] += s["to_keep"]
        totals["to_del"] += s["to_del"]
    print(f"  TOTAL across 18 windows: {totals['to_keep']} del->keep, "
          f"{totals['to_del']} keep->del")


def run_clauds(prober: Prober, out_dir: Path = OUT) -> None:
    print(f"== clauds: 3 fixed crops on {CHAPTER_085.name} (full-mode) ==")
    page = np.asarray(Image.open(CHAPTER_085).convert("RGB"))
    for name, y, h in CLAUDS:
        window_eval(prober, page, name, y, h, out_dir)


def run_gt(prober: Prober, budget_secs: float = 7200.0, out_dir: Path = OUT) -> None:
    print("== gt: manual-reference chapters (held-out evaluation, never training) ==")
    for ch in GT_CHAPTERS:
        rgb = np.asarray(Image.open(GT_DIR / f"{ch}.png").convert("RGB"))
        gt_delete = np.asarray(Image.open(GT_DIR / f"{ch}_cleaned.png").split()[-1]) < 128
        H = rgb.shape[0]
        total = gt_delete.size

        islands = np.zeros((H, rgb.shape[1]), dtype=bool)
        refined = np.zeros_like(islands)
        n_bands = (H + GT_BAND - 1) // GT_BAND
        fast = prober.fast
        for b in range(n_bands):
            y = b * GT_BAND
            y0, y1 = max(0, y - MARGIN), min(H, y + GT_BAND + MARGIN)
            band_rgb = rgb[y0:y1]
            band_isl = prober.delete_mask(band_rgb)
            t = time.time()
            band_ref = prober.refine(band_rgb, band_isl, fast=fast)
            dt = time.time() - t
            if b == 0 and not fast:
                projected = dt * n_bands * len(GT_CHAPTERS)
                if projected > budget_secs:
                    fast = True
                    print(f"  [budget] full-mode band took {dt:.0f}s -> projected "
                          f"{projected/60:.0f}min total; switching gt bands to fast=True")
                    band_ref = prober.refine(band_rgb, band_isl, fast=True)
            ly, ly2 = y - y0, (min(H, y + GT_BAND)) - y0
            islands[y:y + (ly2 - ly)] = band_isl[ly:ly2]
            refined[y:y + (ly2 - ly)] = band_ref[ly:ly2]
            if b % 5 == 0:
                print(f"  ch{ch} band {b + 1}/{n_bands} ({dt:.0f}s/band, fast={fast})")

        print(f"\n=== chapter {ch} ({rgb.shape[1]}x{H}), gt bands fast={fast} ===")
        for tag, m in (("islands           ", islands), ("islands+cascadepsp", refined)):
            over = int(np.count_nonzero(m & ~gt_delete))
            under = int(np.count_nonzero(~m & gt_delete))
            print(f"  {tag}: over-del {over:>9} ({pct(over, total)})  "
                  f"under-del {under:>9} ({pct(under, total)})")

        to_keep = islands & ~refined
        to_del = ~islands & refined
        for nm, flip in (("del->keep", to_keep), ("keep->del", to_del)):
            n = int(flip.sum())
            right_ref = ~gt_delete if nm == "del->keep" else gt_delete
            right = int(np.count_nonzero(flip & right_ref))
            print(f"  {nm}: {n} px flipped, {right} right / {n - right} wrong vs GT")

        # preview the biggest changed bands for visual adjudication
        diff = to_keep | to_del
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
                save_red_preview(out_dir / f"gt{ch}_diff{i}_y{p0}_islands.png", rgb[p0:p1], islands[p0:p1])
                save_red_preview(out_dir / f"gt{ch}_diff{i}_y{p0}_cascadepsp.png", rgb[p0:p1], refined[p0:p1])
            print(f"  saved previews for top {min(3, len(bands))} of {len(bands)} changed band(s)")


def main() -> None:
    # Redirected stdout is fully-buffered, not line-buffered -- a long run's
    # progress prints silently sit unflushed until exit (learned the hard way
    # during the finetune's Phase 2 pilot; see cascadepsp_finetune_plan.md).
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("set", choices=["spots", "clauds", "gt", "all"])
    ap.add_argument("--fast", action="store_true",
                    help="CascadePSP fast mode (global step only) everywhere")
    ap.add_argument("--weights", type=Path, default=None,
                    help="path to a finetuned checkpoint (e.g. from train_cascadepsp_pc.py) "
                    "instead of the stock pretrained CascadePSP release weights")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="override output dir (default: .tmp/cascadepsp_probe, or "
                    ".tmp/cascadepsp_probe_finetuned when --weights is set, so a "
                    "finetuned run never overwrites the zero-shot probe's saved results)")
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = OUT if args.weights is None else ROOT / ".tmp" / "cascadepsp_probe_finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    prober = Prober(fast=args.fast, weights=args.weights)
    if args.set in ("clauds", "all"):
        run_clauds(prober, out_dir)
    if args.set in ("spots", "all"):
        run_spots(prober, out_dir)
    if args.set in ("gt", "all"):
        run_gt(prober, out_dir=out_dir)


if __name__ == "__main__":
    main()
