"""ToonOut/BiRefNet zero-shot probe (.claude/plans/snazzy-cuddling-creek.md).

Mirrors src/probe_cascadepsp.py's structure directly (same spot-check coordinates,
same clauds crops, same banded GT methodology -- MARGIN/GT_BAND, methodology
lesson #10, docs/ml_strategy_history.md) for a standalone segmentation model
(ToonOut) instead of a mask-refiner: ToonOut predicts its own foreground/keep
mask directly from RGB, no coarse input needed.

Weights: joelseytre/toonout (MIT, finetuned on a CC-BY-4.0 1228-image anime
dataset) on top of ZhengPeng7/BiRefNet's architecture (MIT code). Zero-shot --
no P&C exposure at all, no finetuning here.

Requires .venv-toonout (ROCm torch + transformers + timm + huggingface_hub):
  .venv-toonout/bin/python src/probe_toonout.py spots
  .venv-toonout/bin/python src/probe_toonout.py clauds
  .venv-toonout/bin/python src/probe_toonout.py gt
  .venv-toonout/bin/python src/probe_toonout.py all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # visible progress under redirection/backgrounding

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torchvision import transforms  # noqa: E402
import ml_cleaner  # noqa: E402

sys.modules["__main__"].train_command = ml_cleaner.train_command

from ml_cleaner import (  # noqa: E402
    GuidanceParams,
    load_model,
    predict_delete_mask,
    reclaim_landlocked_delete_islands,
    save_red_preview,
)

from transformers import AutoModelForImageSegmentation  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

OUT = ROOT / ".tmp" / "toonout_probe"
MODEL = ROOT / "data/models/10.0-baseline.pt"
CHAPTER_085 = ROOT / "data/chapters-initial/085.png"
GT_DIR = ROOT / ".tmp/saved/chapters"

# Same constants as src/probe_cascadepsp.py -- every quality number this
# project has measured for a refinement/segmentation probe uses these.
SPOT_YS = [6800, 13800, 19100, 21700, 33200, 36700, 48800, 54650, 66900,
           112250, 120700, 155050, 161300, 165500, 169300, 172500, 177700, 179450]
SPOT_H = 900
MARGIN = 300
CLAUDS = [("clauds_1", 169250, 450), ("clauds_2", 179450, 450), ("clauds_3", 54550, 500)]
GT_CHAPTERS = ["001", "002"]
# ToonOut resizes every input to a fixed 1024x1024 square (its own official
# demo notebook does exactly this) -- an extreme aspect-ratio distortion on a
# full GT_BAND=4000-row x 690px-wide band (~6.7:1) that's untested territory
# for this model. Smaller, closer-to-square bands used instead for the GT
# pass here (unlike CascadePSP's GT_BAND=4000) -- verified via a one-band
# sanity check before committing to the full run (see plan's verification
# step); if this still distorts badly, that's itself a real, reportable
# finding about this model's fit for tall manhwa strips.
GT_BAND = 1200

TOONOUT_INPUT_SIZE = 1024


def strip_prefix(k: str) -> str:
    for p in ("module._orig_mod.", "module.", "_orig_mod."):
        if k.startswith(p):
            return k[len(p):]
    return k


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.4f}%"


class Prober:
    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.model, self.config = load_model(MODEL, self.device)
        self.threshold = float(self.config.get("threshold", 0.5))
        self.gp = GuidanceParams(
            threshold_value=int(self.config.get("threshold_value", 30)),
            morph_radius=int(self.config.get("morph_radius", 2)),
        )
        torch.backends.cudnn.enabled = False  # ROCm MIOpen workaround, this session

        self.toonout = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True)
        weights_path = hf_hub_download("joelseytre/toonout", "birefnet_finetuned_toonout.pth")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        self.toonout.load_state_dict({strip_prefix(k): v for k, v in state.items()}, strict=False)
        self.toonout = self.toonout.float().to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((TOONOUT_INPUT_SIZE, TOONOUT_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def islands_mask(self, rgb: np.ndarray) -> np.ndarray:
        raw = predict_delete_mask(
            rgb=rgb, model=self.model, device=self.device, guidance_params=self.gp,
            tile_size=768, overlap=96, threshold=self.threshold, amp=False,
        )
        return reclaim_landlocked_delete_islands(raw)

    def toonout_mask(self, rgb: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(rgb)
        inp = self.transform(pil).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            preds = self.toonout(inp)[-1].sigmoid().cpu()
        mask = (preds[0, 0].numpy() * 255).astype(np.uint8)
        mask_full = np.array(Image.fromarray(mask).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
        keep_prob = mask_full.astype(np.float32) / 255.0
        return keep_prob < 0.5  # delete where NOT confidently foreground/keep


def window_eval(prober: Prober, rgb_page: np.ndarray, name: str, y: int, h: int, out_dir: Path = OUT) -> dict:
    H = rgb_page.shape[0]
    y0, y1 = max(0, y - MARGIN), min(H, y + h + MARGIN)
    rgb = rgb_page[y0:y1]
    islands = prober.islands_mask(rgb)
    t = time.time()
    toonout = prober.toonout_mask(rgb)
    dt = time.time() - t

    ly = y - y0
    w_isl, w_too, w_rgb = islands[ly:ly + h], toonout[ly:ly + h], rgb[ly:ly + h]
    to_keep = int(np.count_nonzero(w_isl & ~w_too))
    to_del = int(np.count_nonzero(~w_isl & w_too))
    save_red_preview(out_dir / f"{name}_islands.png", w_rgb, w_isl)
    save_red_preview(out_dir / f"{name}_toonout.png", w_rgb, w_too)
    print(f"  {name}: toonout {dt:.1f}s | {to_keep} del->keep, {to_del} keep->del (window {w_isl.size} px)")
    return {"to_keep": to_keep, "to_del": to_del}


def run_spots(prober: Prober, out_dir: Path = OUT) -> None:
    print(f"== spots: 18-coordinate broad set on {CHAPTER_085.name} ==")
    page = np.asarray(Image.open(CHAPTER_085).convert("RGB"))
    totals = {"to_keep": 0, "to_del": 0}
    for y in SPOT_YS:
        s = window_eval(prober, page, f"spot_y{y}", y, SPOT_H, out_dir)
        totals["to_keep"] += s["to_keep"]
        totals["to_del"] += s["to_del"]
    print(f"  TOTAL across 18 windows: {totals['to_keep']} del->keep, {totals['to_del']} keep->del")


def run_clauds(prober: Prober, out_dir: Path = OUT) -> None:
    print(f"== clauds: 3 fixed crops on {CHAPTER_085.name} ==")
    page = np.asarray(Image.open(CHAPTER_085).convert("RGB"))
    for name, y, h in CLAUDS:
        window_eval(prober, page, name, y, h, out_dir)


def run_gt(prober: Prober, out_dir: Path = OUT) -> None:
    print("== gt: manual-reference chapters (held-out evaluation, never training) ==")
    for ch in GT_CHAPTERS:
        rgb = np.asarray(Image.open(GT_DIR / f"{ch}.png").convert("RGB"))
        gt_delete = np.asarray(Image.open(GT_DIR / f"{ch}_cleaned.png").split()[-1]) < 128
        H = rgb.shape[0]
        total = gt_delete.size

        islands = np.zeros((H, rgb.shape[1]), dtype=bool)
        toonout = np.zeros_like(islands)
        n_bands = (H + GT_BAND - 1) // GT_BAND
        t_start = time.time()
        for b in range(n_bands):
            y = b * GT_BAND
            y0, y1 = max(0, y - MARGIN), min(H, y + GT_BAND + MARGIN)
            band_rgb = rgb[y0:y1]
            band_isl = prober.islands_mask(band_rgb)
            t = time.time()
            band_too = prober.toonout_mask(band_rgb)
            dt = time.time() - t
            ly, ly2 = y - y0, min(H, y + GT_BAND) - y0
            islands[y:y + (ly2 - ly)] = band_isl[ly:ly2]
            toonout[y:y + (ly2 - ly)] = band_too[ly:ly2]
            if b % 10 == 0:
                elapsed = time.time() - t_start
                eta = (n_bands - b - 1) * (elapsed / (b + 1))
                print(f"  ch{ch} band {b + 1}/{n_bands} ({dt:.1f}s/band, elapsed={elapsed:.0f}s, eta={eta:.0f}s)")

        print(f"\n=== chapter {ch} ({rgb.shape[1]}x{H}) ===")
        for tag, m in (("islands", islands), ("toonout", toonout)):
            over = int(np.count_nonzero(m & ~gt_delete))
            under = int(np.count_nonzero(~m & gt_delete))
            print(f"  {tag:8s}: over-del {over:>9} ({pct(over, total)})  under-del {under:>9} ({pct(under, total)})  "
                  f"total {pct(over + under, total)}")

        to_keep = islands & ~toonout
        to_del = ~islands & toonout
        for nm, flip in (("del->keep", to_keep), ("keep->del", to_del)):
            n = int(flip.sum())
            right_ref = ~gt_delete if nm == "del->keep" else gt_delete
            right = int(np.count_nonzero(flip & right_ref))
            print(f"  {nm}: {n} px flipped, {right} right / {n - right} wrong vs GT")

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
                save_red_preview(out_dir / f"gt{ch}_diff{i}_y{p0}_toonout.png", rgb[p0:p1], toonout[p0:p1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["spots", "clauds", "gt", "all", "sanity"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    prober = Prober(device=args.device)

    if args.mode == "sanity":
        page = np.asarray(Image.open(CHAPTER_085).convert("RGB"))
        window_eval(prober, page, "sanity_gtband", 50000, GT_BAND, OUT)
        return

    if args.mode in ("spots", "all"):
        run_spots(prober)
    if args.mode in ("clauds", "all"):
        run_clauds(prober)
    if args.mode in ("gt", "all"):
        run_gt(prober)


if __name__ == "__main__":
    main()
