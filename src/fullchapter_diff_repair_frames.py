"""Whole-chapter test: where does --repair-frames change production output?

Runs 10.0-baseline on all of 085.png (one inference pass), then compares
islands-only vs islands+repair_frames. Saves a red-preview crop pair for
every row-band where the two masks differ, to .tmp/repair_frames_eval/diff_*.

Also asserts repair never adds delete pixels, and reports per-band flip
counts so under-deletion risk spots can be inspected directly.
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

rgb = np.asarray(Image.open(ROOT / "data/chapters-initial/085.png").convert("RGB"))
print(f"chapter: {rgb.shape[1]}x{rgb.shape[0]}")

mask = predict_delete_mask(
    rgb=rgb, model=model, device=device, guidance_params=gp,
    tile_size=768, overlap=96, threshold=threshold, amp=False,
    sdt_fusion=False, sdt_fusion_band_radius=4, sdt_clamp_radius=8.0,
)
islands = reclaim_landlocked_delete_islands(mask)
repaired = repair_frame_interiors(rgb, islands, frame_darkness=40,
                                  min_interior_px=10000, inset_px=2)

added = int(np.count_nonzero(repaired & ~islands))
assert added == 0, f"repair ADDED {added} delete pixels (must be impossible)"

diff = islands & ~repaired
total = int(np.count_nonzero(diff))
print(f"total flipped delete->keep: {total} px ({100.0 * total / diff.size:.4f}% of strip)")

# group differing rows into bands separated by >200 clean rows
rows = np.flatnonzero(diff.any(axis=1))
if len(rows) == 0:
    print("no differences anywhere -- repair is a no-op on this chapter")
    sys.exit(0)

bands = []
start = prev = int(rows[0])
for r in rows[1:]:
    r = int(r)
    if r - prev > 200:
        bands.append((start, prev))
        start = r
    prev = r
bands.append((start, prev))

print(f"{len(bands)} differing band(s):")
H = rgb.shape[0]
for i, (r0, r1) in enumerate(bands):
    y0, y1 = max(0, r0 - 150), min(H, r1 + 150)
    n = int(np.count_nonzero(diff[r0 : r1 + 1]))
    print(f"  band {i}: rows {r0}-{r1} (abs y), {n} px flipped")
    save_red_preview(OUT / f"diff_{i}_y{y0}_islands.png", rgb[y0:y1], islands[y0:y1])
    save_red_preview(OUT / f"diff_{i}_y{y0}_frames.png", rgb[y0:y1], repaired[y0:y1])
