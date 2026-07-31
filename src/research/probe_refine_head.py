"""Qualitative probe for the RefineHead Stage-B pilot (.claude/plans/snazzy-cuddling-creek.md).

Loads a `--refine-head`-enabled checkpoint (e.g. the 400-step pilot,
data/models or .tmp/refine_head_pilot.step400.pt) and runs both its coarse
output (== plain 10.0-baseline, since the coarse decoder is frozen/untouched)
and its refined output (self-consistent inference-time path: refine_head
corrects the model's OWN coarse logits, not a synthetic training-time
substitute) through the same tiled inference + --reclaim-islands pipeline
`ml_cleaner.py process` uses, on the project's established hard-case crop
sets (notes/white_bg_regression_crops.md, sfx_regression_crops.md) --
reused rather than reinvented, per the plan's evaluation section.

Each window is processed with MARGIN=300px of context on each side, matching
methodology lesson #10 (docs/ml_strategy_history.md) and probe_cascadepsp.py's
own window_eval() exactly -- context size changes a refinement stage's
decisions, so this is not optional.

Usage:
  .venv/bin/python src/probe_refine_head.py --checkpoint .tmp/refine_head_pilot.step400.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import ml_cleaner  # noqa: E402

sys.modules["__main__"].train_command = ml_cleaner.train_command

from ml_cleaner import (  # noqa: E402
    GuidanceParams,
    SmallUNet,
    predict_delete_mask,
    reclaim_landlocked_delete_islands,
    save_red_preview,
)

OUT = ROOT / ".tmp" / "refine_head_probe"
CHAPTER_085 = ROOT / "data/chapters-initial/085.png"
MARGIN = 300  # matches probe_cascadepsp.py's MARGIN -- methodology lesson #10

# notes/white_bg_regression_crops.md + sfx_regression_crops.md
CROPS = [
    ("white_A_bubble_on_white", 78000, 1000),
    ("white_B_bubble_panel_gutter", 101000, 1500),
    ("white_C_white_burst_sfx", 160000, 1500),
    ("sfx_A_blue_swoosh", 155050, 600),
    ("sfx_B_red_gradient_text", 177700, 500),
]


class _HeadSelect(nn.Module):
    """Wraps a refine_head-enabled SmallUNet to satisfy predict_delete_mask's
    `logits, sdt_pred = model(image)` 2-tuple contract, selecting either the
    coarse or refined output. Calls the model's real forward() unmodified --
    refine_head sees the model's own coarse logits, exactly as it would at
    real inference time (no synthetic-perturbation shortcut from training)."""

    def __init__(self, model: SmallUNet, head: str) -> None:
        super().__init__()
        assert head in ("coarse", "refined")
        self.model = model
        self.head = head

    def forward(self, x: torch.Tensor):
        coarse, refined, sdt = self.model(x)
        return (coarse if self.head == "coarse" else refined), sdt


def window_eval(coarse_model, refined_model, gp, threshold, rgb_page, name, y, h, out_dir) -> dict:
    H = rgb_page.shape[0]
    y0, y1 = max(0, y - MARGIN), min(H, y + h + MARGIN)
    rgb = rgb_page[y0:y1]

    def run(model):
        raw = predict_delete_mask(
            rgb=rgb, model=model, device=torch.device("cpu"), guidance_params=gp,
            tile_size=768, overlap=96, threshold=threshold, amp=False,
        )
        return reclaim_landlocked_delete_islands(raw)

    coarse = run(coarse_model)
    refined = run(refined_model)

    ly = y - y0
    w_coarse, w_refined, w_rgb = coarse[ly:ly + h], refined[ly:ly + h], rgb[ly:ly + h]
    to_keep = int(np.count_nonzero(w_coarse & ~w_refined))  # delete -> keep
    to_del = int(np.count_nonzero(~w_coarse & w_refined))   # keep -> delete
    save_red_preview(out_dir / f"{name}_coarse.png", w_rgb, w_coarse)
    save_red_preview(out_dir / f"{name}_refined.png", w_rgb, w_refined)
    print(f"  {name}: {to_keep} del->keep, {to_del} keep->del (window {w_coarse.size} px)")
    return {"to_keep": to_keep, "to_del": to_del}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    assert config.get("refine_head"), f"{args.checkpoint} was not saved with refine_head=True"
    model = SmallUNet(
        in_channels=int(config["in_channels"]),
        base=int(config["base_channels"]),
        sdt_head=bool(config.get("sdt_head", False)),
        refine_head=True,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    gp = GuidanceParams(
        threshold_value=int(config.get("threshold_value", 30)),
        morph_radius=int(config.get("morph_radius", 2)),
    )
    threshold = float(config.get("threshold", 0.5))
    coarse_model = _HeadSelect(model, "coarse")
    refined_model = _HeadSelect(model, "refined")

    print(f"== refine_head probe: {args.checkpoint.name} on {CHAPTER_085.name} ==")
    page = np.asarray(Image.open(CHAPTER_085).convert("RGB"))
    totals = {"to_keep": 0, "to_del": 0}
    t0 = time.time()
    for name, y, h in CROPS:
        s = window_eval(coarse_model, refined_model, gp, threshold, page, name, y, h, OUT)
        totals["to_keep"] += s["to_keep"]
        totals["to_del"] += s["to_del"]
    print(f"  TOTAL across {len(CROPS)} windows: {totals['to_keep']} del->keep, "
          f"{totals['to_del']} keep->del ({time.time() - t0:.1f}s)")
    print(f"previews saved under {OUT}")


if __name__ == "__main__":
    main()
