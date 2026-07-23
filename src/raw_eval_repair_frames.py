"""repair-frames on the RAW mask (no islands): what does it fix on its own?

For manual-reference chapters 001/002 (human GT, investigation-only):
  raw            -- model output, no postprocessing
  raw+frames     -- repair_frame_interiors only
  raw+islands    -- reclaim_landlocked_delete_islands only (production path)
  islands+frames -- sanity (known: identical to raw+islands)

Two-directional vs GT (alpha<128 = delete), plus:
  - flips by frames on raw: right (GT keep) vs wrong (GT delete)
  - overlap of frames' flips with islands' flips (is frames a subset?)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import ml_cleaner  # noqa: E402

sys.modules["__main__"].train_command = ml_cleaner.train_command

from ml_cleaner import (  # noqa: E402
    GuidanceParams, load_model, predict_delete_mask,
    reclaim_landlocked_delete_islands, repair_frame_interiors, save_red_preview,
)

OUT = ROOT / ".tmp" / "repair_frames_eval"
device = torch.device("cpu")
model, config = load_model(ROOT / "data/models/10.0-baseline.pt", device)
threshold = float(config.get("threshold", 0.5))
gp = GuidanceParams(threshold_value=int(config.get("threshold_value", 30)),
                    morph_radius=int(config.get("morph_radius", 2)))

def pct(n, d):
    return f"{100.0 * n / d:.4f}%"

for ch in ["001", "002"]:
    rgb = np.asarray(Image.open(ROOT / f".tmp/saved/chapters/{ch}.png").convert("RGB"))
    gt_delete = np.asarray(Image.open(ROOT / f".tmp/saved/chapters/{ch}_cleaned.png").split()[-1]) < 128
    total = gt_delete.size

    raw = predict_delete_mask(
        rgb=rgb, model=model, device=device, guidance_params=gp,
        tile_size=768, overlap=96, threshold=threshold, amp=False,
        sdt_fusion=False, sdt_fusion_band_radius=4, sdt_clamp_radius=8.0,
    )
    frames = repair_frame_interiors(rgb, raw, 40, 10000, 2)
    islands = reclaim_landlocked_delete_islands(raw)
    both = repair_frame_interiors(rgb, islands, 40, 10000, 2)

    print(f"\n=== chapter {ch} ({rgb.shape[1]}x{rgb.shape[0]}) ===")
    for tag, m in (("raw           ", raw), ("raw+frames    ", frames),
                   ("raw+islands   ", islands), ("islands+frames", both)):
        over = int(np.count_nonzero(m & ~gt_delete))
        under = int(np.count_nonzero(~m & gt_delete))
        print(f"  {tag}: over-del {over:>9} ({pct(over, total)})  "
              f"under-del {under:>9} ({pct(under, total)})")

    f_flip = raw & ~frames
    i_flip = raw & ~islands
    nf, ni = int(f_flip.sum()), int(i_flip.sum())
    inter = int(np.count_nonzero(f_flip & i_flip))
    right = int(np.count_nonzero(f_flip & ~gt_delete))
    wrong = int(np.count_nonzero(f_flip & gt_delete))
    print(f"  frames flipped {nf} px on raw ({right} right / {wrong} wrong vs GT)")
    print(f"  islands flipped {ni} px on raw; frames∩islands = {inter} "
          f"({pct(inter, nf) if nf else 'n/a'} of frames' flips)")

    # preview the largest frames-flip band for visual sanity
    rows = np.flatnonzero(f_flip.any(axis=1))
    if len(rows):
        bands, start, prev = [], int(rows[0]), int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r - prev > 200:
                bands.append((start, prev))
                start = r
            prev = r
        bands.append((start, prev))
        bands.sort(key=lambda b: -int(np.count_nonzero(f_flip[b[0]:b[1]+1])))
        r0, r1 = bands[0]
        y0, y1 = max(0, r0 - 150), min(rgb.shape[0], r1 + 150)
        print(f"  biggest frames-flip band: y={r0}-{r1} "
              f"({int(np.count_nonzero(f_flip[r0:r1+1]))} px) -> previews saved")
        save_red_preview(OUT / f"raw{ch}_band_y{y0}_raw.png", rgb[y0:y1], raw[y0:y1])
        save_red_preview(OUT / f"raw{ch}_band_y{y0}_frames.png", rgb[y0:y1], frames[y0:y1])
