"""Part 2 of the instance-aware architecture pivot smoke test: evaluate the locally-cropped
TinyInstanceNet (instance_sfx_net.py) against REAL SFX-glyph instances -- the actual
decision-gate check -- does bounding the model's input to a 224x224 window (96px margin)
avoid the R>=128px "read as background" context-flip the whole-page production checkpoint
shows (docs/ml_strategy_history.md, 2026-07-31 deep diagnosis: correct at R<=64px, wrong at
R>=128px, saturating ~0.306 by R=256)?

Evaluates the one existing tracked instance (ch1_sfx_text, .tmp/diagnostics/ch002_rois.json)
plus 5 additional real SFX-like instances found by scanning both manual-reference chapters
with the same ink-stroke-connected-components heuristic used for training-crop extraction
(build_sfx_instance_crops.py::find_sfx_instances), filtered to glyph scale and manually
visually confirmed (not just trusted blind) before being included here -- per this project's
own standing rule against n=1 claims.

Real chapters are held-out evaluation only, per permanent project policy -- no weights are
updated from this script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ml_cleaner import GuidanceParams, build_input_tensor  # noqa: E402
from instance_sfx_net import TinyInstanceNet  # noqa: E402

CKPT_DIR = ROOT / ".tmp/checkpoints/instance_sfx_smoke"
FRAME_DARKNESS = 40
CROP_SIZE = 224
PAD = 300

# (label, chapter file, x0, y0, x1, y1) -- x1/y1 are the outer bbox to search for ink within,
# not necessarily the tight glyph box itself (matches ch1_sfx_text's own registry convention:
# a small margin box around the glyph, panel border excluded).
INSTANCES = [
    ("ch1_sfx_text (tracked)", "001.png", 0, 44900, 130, 45080),
    ("real_cand_0 (ch1, Korean glyph+glow)", "001.png", 143, 43920, 243, 44054),
    ("real_cand_1 (ch1, SFX glow strokes)", "001.png", 510, 48750, 610, 48981),
    ("real_cand_4 (ch2, stick-figure SFX icon)", "002.png", 428, 113012, 585, 113205),
    ("real_cand_5 (ch2, Korean SFX text)", "002.png", 502, 25011, 633, 25151),
    ("real_cand_6 (ch2, SFX glyph)", "002.png", 224, 49125, 345, 49282),
]


def eval_instance(model, label: str, chapter_file: str, x0: int, y0: int, x1: int, y1: int) -> None:
    chapter_path = ROOT / "data/chapters-initial" / chapter_file
    full = Image.open(chapter_path)
    W, H = full.size
    rx0, ry0 = max(0, x0 - PAD), max(0, y0 - PAD)
    rx1, ry1 = min(W, x1 + PAD), min(H, y1 + PAD)
    region = np.asarray(full.crop((rx0, ry0, rx1, ry1)).convert("RGB"))

    roi_local = region[y0 - ry0 : y1 - ry0, x0 - rx0 : x1 - rx0]
    gray_roi = cv2.cvtColor(roi_local, cv2.COLOR_RGB2GRAY)
    stroke = gray_roi <= FRAME_DARKNESS
    if not stroke.any():
        print(f"{label}: FAIL -- no ink detected inside the box, skipping")
        return
    ys, xs = np.where(stroke)
    glyph_cx = int(x0 - rx0 + (xs.min() + xs.max()) / 2)
    glyph_cy = int(y0 - ry0 + (ys.min() + ys.max()) / 2)

    input_tensor = build_input_tensor(region, GuidanceParams())

    half = CROP_SIZE // 2
    h, w = input_tensor.shape[:2]
    local_crop = np.zeros((CROP_SIZE, CROP_SIZE, 7), dtype=np.float32)
    sx0, sy0 = max(0, glyph_cx - half), max(0, glyph_cy - half)
    sx1, sy1 = min(w, glyph_cx + half), min(h, glyph_cy + half)
    dx0, dy0 = sx0 - (glyph_cx - half), sy0 - (glyph_cy - half)
    local_crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = input_tensor[sy0:sy1, sx0:sx1]

    with torch.no_grad():
        xt = torch.from_numpy(local_crop).permute(2, 0, 1).unsqueeze(0).float()
        probs = torch.sigmoid(model(xt).squeeze()).numpy()

    roi_x0c = max(0, (x0 - rx0) - (glyph_cx - half))
    roi_x1c = min(CROP_SIZE, (x1 - rx0) - (glyph_cx - half))
    roi_y0c = max(0, (y0 - ry0) - (glyph_cy - half))
    roi_y1c = min(CROP_SIZE, (y1 - ry0) - (glyph_cy - half))
    roi_probs = probs[roi_y0c:roi_y1c, roi_x0c:roi_x1c]

    mean_prob = float(roi_probs.mean())
    delete_fraction = float((roi_probs > 0.5).mean())
    print(f"{label}: mean_prob={mean_prob:.4f}  delete_fraction={delete_fraction:.4f}  "
          f"{'PASS' if mean_prob < 0.30 else 'FAIL'} (ceiling 0.30, expected: keep)")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["glyph_only", "with_bg", "with_bg_light", "with_bg_weighted"])
    args = ap.parse_args()

    ckpt = torch.load(CKPT_DIR / f"instance_sfx_smoke_{args.variant}.pt", map_location="cpu")
    model = TinyInstanceNet(in_ch=ckpt["in_ch"], base=ckpt["base"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"=== [{args.variant}] TinyInstanceNet (local 224px crop, 96px margin) on real SFX-like instances ===")
    for label, chapter_file, x0, y0, x1, y1 in INSTANCES:
        eval_instance(model, label, chapter_file, x0, y0, x1, y1)

    print(f"\nFor comparison (docs/ml_strategy_history.md, whole-page production chain, "
          f"ch1_sfx_text specifically):")
    print(f"  a6_full10k (Stage 1 only):            raw delete_fraction 0.5243")
    print(f"  b2_full2k_finetune (Stage 1+2):        raw delete_fraction 0.3068")
    print(f"  stage3_sfx_2k_resumed (Stage 1+2+3):   raw delete_fraction 0.1987")


if __name__ == "__main__":
    main()
