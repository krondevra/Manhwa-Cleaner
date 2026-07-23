"""Two-directional ground-truth eval of --repair-frames on the manual-reference
chapters (001, 002) -- investigation-only use of the human-cleaned references,
same as the 2026-07-10 session. NEVER training data.

For islands-only vs islands+repair_frames, against GT alpha (<128 = delete):
  over-deletion  = mask delete & GT keep   (content wrongly deleted)
  under-deletion = mask keep  & GT delete  (background wrongly kept)
Plus the sharpest measure: of the pixels repair_frames flips delete->keep,
how many are right (GT keep) vs wrong (GT delete)?
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
OUT.mkdir(exist_ok=True)

device = torch.device("cpu")
model, config = load_model(ROOT / "data/models/10.0-baseline.pt", device)
threshold = float(config.get("threshold", 0.5))
gp = GuidanceParams(threshold_value=int(config.get("threshold_value", 30)),
                    morph_radius=int(config.get("morph_radius", 2)))

def pct(n, d):
    return f"{100.0 * n / d:.4f}%"

for ch in ["001", "002"]:
    rgb = np.asarray(Image.open(ROOT / f".tmp/saved/chapters/{ch}.png").convert("RGB"))
    gt_alpha = np.asarray(Image.open(ROOT / f".tmp/saved/chapters/{ch}_cleaned.png").split()[-1])
    gt_delete = gt_alpha < 128
    total = gt_delete.size
    print(f"\n=== chapter {ch} ({rgb.shape[1]}x{rgb.shape[0]}) "
          f"GT delete share {pct(int(gt_delete.sum()), total)} ===")

    mask = predict_delete_mask(
        rgb=rgb, model=model, device=device, guidance_params=gp,
        tile_size=768, overlap=96, threshold=threshold, amp=False,
        sdt_fusion=False, sdt_fusion_band_radius=4, sdt_clamp_radius=8.0,
    )
    islands = reclaim_landlocked_delete_islands(mask)
    repaired = repair_frame_interiors(rgb, islands, frame_darkness=40,
                                      min_interior_px=10000, inset_px=2)
    assert not (repaired & ~islands).any(), "repair added delete pixels"

    for tag, m in (("islands       ", islands), ("islands+frames", repaired)):
        over = int(np.count_nonzero(m & ~gt_delete))
        under = int(np.count_nonzero(~m & gt_delete))
        print(f"  {tag}: over-del {over:>9} ({pct(over, total)})  "
              f"under-del {under:>9} ({pct(under, total)})")

    flipped = islands & ~repaired
    n_flip = int(flipped.sum())
    right = int(np.count_nonzero(flipped & ~gt_delete))
    wrong = int(np.count_nonzero(flipped & gt_delete))
    print(f"  frames flipped {n_flip} px: {right} right (GT keep), {wrong} wrong (GT delete)")

    # save previews of the biggest flip bands for visual inspection
    rows = np.flatnonzero(flipped.any(axis=1))
    if len(rows):
        bands, start, prev = [], int(rows[0]), int(rows[0])
        for r in rows[1:]:
            r = int(r)
            if r - prev > 200:
                bands.append((start, prev))
                start = r
            prev = r
        bands.append((start, prev))
        bands.sort(key=lambda b: -int(np.count_nonzero(flipped[b[0]:b[1]+1])))
        H = rgb.shape[0]
        print(f"  {len(bands)} flip band(s); saving top {min(4, len(bands))}")
        for i, (r0, r1) in enumerate(bands[:4]):
            y0, y1 = max(0, r0 - 150), min(H, r1 + 150)
            n = int(np.count_nonzero(flipped[r0:r1+1]))
            print(f"    band y={r0}-{r1}: {n} px")
            save_red_preview(OUT / f"gt{ch}_band{i}_y{y0}_islands.png", rgb[y0:y1], islands[y0:y1])
            save_red_preview(OUT / f"gt{ch}_band{i}_y{y0}_frames.png", rgb[y0:y1], repaired[y0:y1])
