"""Two-directional GT eval for a generation-6 pivot checkpoint (Part 7,
notes/synthetic_curriculum_plan.md), against the manual-reference chapters
(001, 002) -- investigation-only use of the human-cleaned references, never training data.

No banding/MARGIN logic needed (Part 3's correction): predict_delete_mask already tiles
arbitrary input sizes internally (pad_image_for_tiling + Hann-window blend), confirmed
working on the full merged GT strips directly by gt_eval_repair_frames.py -- this script
follows that same simple pattern rather than reinventing per-page looping, since merging
was confirmed not to matter to the model itself.

Reports raw and +--reclaim-islands numbers for the gen6 checkpoint, alongside a freshly
computed production (10.0-baseline) number in the same run for a fair, same-instant
comparison rather than trusting an old recorded figure.

Usage:
  .venv-gpu/bin/python src/research/eval_gen6_checkpoint.py --model data/models/18.0-frames.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import ml_cleaner  # noqa: E402

sys.modules["__main__"].train_command = ml_cleaner.train_command

from ml_cleaner import (  # noqa: E402
    GuidanceParams, choose_device, load_model, predict_delete_mask,
    reclaim_landlocked_delete_islands,
)

GT_CHAPTERS = ["001", "002"]
GT_DIR = ROOT / ".tmp/saved/chapters"
PRODUCTION_MODEL = ROOT / "data/models/10.0-baseline.pt"


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.4f}%"


def eval_checkpoint(model_path: Path, device: torch.device, label: str) -> dict:
    model, config = load_model(model_path, device)
    threshold = float(config.get("threshold", 0.5))
    gp = GuidanceParams(threshold_value=int(config.get("threshold_value", 30)),
                         morph_radius=int(config.get("morph_radius", 2)))

    results: dict[str, dict] = {}
    for ch in GT_CHAPTERS:
        rgb = np.asarray(Image.open(GT_DIR / f"{ch}.png").convert("RGB"))
        gt_alpha = np.asarray(Image.open(GT_DIR / f"{ch}_cleaned.png").split()[-1])
        gt_delete = gt_alpha < 128
        total = gt_delete.size

        raw = predict_delete_mask(
            rgb=rgb, model=model, device=device, guidance_params=gp,
            tile_size=768, overlap=96, threshold=threshold, amp=False,
        )
        islands = reclaim_landlocked_delete_islands(raw)

        print(f"\n=== {label} -- chapter {ch} ({rgb.shape[1]}x{rgb.shape[0]}) "
              f"GT delete share {pct(int(gt_delete.sum()), total)} ===")
        ch_result = {}
        for tag, m in (("raw", raw), ("+islands", islands)):
            over = int(np.count_nonzero(m & ~gt_delete))
            under = int(np.count_nonzero(~m & gt_delete))
            total_err = over + under
            print(f"  {tag:10s}: over-del {over:>9} ({pct(over, total)})  "
                  f"under-del {under:>9} ({pct(under, total)})  total {pct(total_err, total)}")
            ch_result[tag] = {"over": over, "under": under, "total_pct": 100.0 * total_err / total}
        results[ch] = ch_result
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path, help="gen6 checkpoint to evaluate")
    ap.add_argument("--label", default=None, help="Label for the checkpoint in output (default: filename)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-production", action="store_true",
                     help="Skip the freshly-computed 10.0-baseline comparison")
    args = ap.parse_args()

    device = choose_device(args.device)
    print(f"device: {device}")

    label = args.label or args.model.stem
    gen6_results = eval_checkpoint(args.model, device, label)

    prod_results = None
    if not args.skip_production:
        prod_results = eval_checkpoint(PRODUCTION_MODEL, device, "10.0-baseline (fresh)")

    print(f"\n=== summary ===")
    for ch in GT_CHAPTERS:
        line = f"  ch{ch}: {label}+islands={gen6_results[ch]['+islands']['total_pct']:.2f}%"
        if prod_results:
            line += f"  vs  production+islands={prod_results[ch]['+islands']['total_pct']:.2f}%"
        print(line)


if __name__ == "__main__":
    main()
