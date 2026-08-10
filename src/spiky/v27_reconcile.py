"""v27 step 1-2: reconcile user's visual review with v12ABES suite numbers.

Renders full disagreement maps (red=over-del, magenta=under-del) for 019_2/3/6 (the
flagged leakage instances) and quantifies class-C glyph dropout: over-deletion pixels
that sit INSIDE the sealed spiky interior's filled-holes version but OUTSIDE its raw
flood-connected version (i.e. isolated pockets text strokes cut off from the main
interior region) across all 12 instances.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
# Render output dir (was a session scratchpad before the 2026-08-10 promotion to src/).
SP = HERE.parents[1] / ".tmp/diagnostics/v27_reconcile"
SP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from v26_fullpage_suite import CACHE, DIAG, INSTANCES, PAGES
from classifiers.background import enclosed as _enclosed
from pipeline import G_TOL


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """cv2-based hole fill (no scipy dep): fill = mask OR NOT(border-reachable ~mask)."""
    H, W = mask.shape
    inv = (~mask).astype(np.uint8)
    padded = np.zeros((H + 2, W + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = inv
    ff = np.zeros((H + 4, W + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 1)
    outside = padded[1:-1, 1:-1] == 1  # border-reachable background
    return mask | (~outside)


def main() -> None:
    import subprocess
    print(subprocess.run(["date"], capture_output=True, text=True).stdout.strip(), flush=True)
    masks = {ch: np.load(CACHE / f"v12ABES_{ch}.npy") for ch in PAGES}
    pages = {ch: np.asarray(Image.open(PAGES[ch]).convert("RGB")) for ch in PAGES}

    print("\n=== C-class dropout scan (all 12) ===", flush=True)
    for name, ch, y0, _init in INSTANCES:
        et = np.asarray(Image.open(DIAG / name / f"{name}_etalon.png"))
        gt_del = et[:, :, 3] < 128
        H = gt_del.shape[0]
        d = masks[ch][y0:y0 + H]
        rgb = pages[ch][y0:y0 + H]
        f = rgb.astype(np.float32)
        gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
        band = rgb.min(axis=2) >= G_TOL
        k3 = np.ones((3, 3), np.uint8)
        barrier = cv2.morphologyEx((~band).astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
        interior_conn = _enclosed(band & ~barrier)
        interior_filled = fill_holes(interior_conn)
        dropout_zone = interior_filled & ~interior_conn  # the pockets glyphs cut off
        over = d & ~gt_del
        c_dropout = int((over & dropout_zone).sum())
        total_over = int(over.sum())
        print(f"  {name:22s} total-over {total_over:6d}  C-dropout {c_dropout:6d}  "
              f"dropout-zone-size {int(dropout_zone.sum()):6d}", flush=True)

    print("\n=== full disagreement renders: 019_2/3/6 ===", flush=True)
    for name, ch, y0, _init in [x for x in INSTANCES if x[0].startswith(("019_2", "019_3", "019_6"))]:
        et = np.asarray(Image.open(DIAG / name / f"{name}_etalon.png"))
        gt_del = et[:, :, 3] < 128
        H = gt_del.shape[0]
        d = masks[ch][y0:y0 + H]
        rgb = pages[ch][y0:y0 + H].copy()
        img = rgb.copy()
        img[d & ~gt_del] = [255, 0, 0]
        img[~d & gt_del] = [255, 0, 255]
        cv2.imwrite(str(SP / f"v27_recon_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # also the RAW v12 output (no GT overlay) for the user's original visual framing
        out = rgb.copy()
        out[d] = [255, 255, 255]
        sep = np.full((H, 8, 3), 128, np.uint8)
        cv2.imwrite(str(SP / f"v27_raw_{name}.png"),
                    cv2.cvtColor(np.hstack([rgb, sep, out]), cv2.COLOR_RGB2BGR))
        print(f"  {name}: renders saved", flush=True)


if __name__ == "__main__":
    main()
