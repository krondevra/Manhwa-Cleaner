"""Plan v13 Phase 0: extract source RGB + GT delete mask from every gold PSD
(.tmp/saved/psd/new-gold/), with per-part sanity stats and the mandated dark-chapter check
for 033-1..4.

All 10 PSDs share the stack red / img[raster mask] / mask-hard / mask-soft (+ stray extras,
ignored): source = img layer raster, GT delete = img mask < 128 (identical convention to the
v12 005-1 extraction, where this mask agreed 1.000000 with the exported cleaned PNG).

Outputs to .tmp/scripts-manual/gold_extracted/: {part}_src.png, {part}_gt.npy.

Usage:
  .venv/bin/python src/spiky/psd_extract_gold.py
"""
from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage

Image.MAX_IMAGE_PIXELS = None

# Promoted from .tmp/scripts-manual (2026-08-10 cleanup): code is git-tracked here,
# PSDs and extraction outputs stay in the gitignored .tmp/.
HERE = Path(__file__).resolve().parent
TMP = HERE.parents[1] / ".tmp"
GOLD = TMP / "saved/psd/new-gold"
OUT = TMP / "scripts-manual/gold_extracted"
OUT.mkdir(parents=True, exist_ok=True)

DARK_T = 64


def main() -> None:
    import subprocess
    print(subprocess.run(["date"], capture_output=True, text=True).stdout.strip(), flush=True)

    for p in sorted(GOLD.glob("*.psd")):
        part = p.stem
        psd = PSDImage.open(p)
        W, H = psd.size
        layers = {layer.name: layer for layer in psd}
        img_layer = layers["img"]

        src = np.zeros((H, W, 3), dtype=np.uint8)
        arr = np.asarray(img_layer.topil().convert("RGB"))
        x0, y0, x1, y1 = img_layer.bbox
        src[y0:y1, x0:x1] = arr

        mask = img_layer.mask
        assert mask is not None, f"{part}: no img mask"
        mask_arr = np.asarray(mask.topil())
        mx0, my0, mx1, my1 = mask.bbox
        full_mask = np.full((H, W), mask.background_color, dtype=np.uint8)
        full_mask[my0:my1, mx0:mx1] = mask_arr
        gt_delete = full_mask < 128

        gray = np.asarray(Image.fromarray(src).convert("L"))
        dark_frac = float((gray <= DARK_T).mean())
        med = float(np.median(gray))
        binary_frac = float(((full_mask <= 5) | (full_mask >= 250)).mean())

        Image.fromarray(src).save(OUT / f"{part}_src.png")
        np.save(OUT / f"{part}_gt.npy", gt_delete)
        print(f"{part}: {W}x{H}  gt_delete={gt_delete.mean():.4f}  "
              f"mask_binary={binary_frac:.4f}  img_median_gray={med:.0f}  "
              f"dark_frac(<= {DARK_T})={dark_frac:.4f}", flush=True)


if __name__ == "__main__":
    main()
