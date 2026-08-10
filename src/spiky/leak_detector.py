"""Plan v13 Phase 2: reference-free flood-leak detector -- barrier-split analysis.

METHOD (stated before use): a deletion region "leaked across a visible border" if, after
subtracting the barrier map (near-black strokes, gray <= 40 -- the project's FRAME_DARKNESS
convention -- morphologically closed 3px to bridge anti-aliasing gaps) from the predicted
DELETE mask, a deleted sub-component remains that (a) does not touch the page edge and
(b) contains no seed pixels. Deletion that got "behind" a visible dark border without a
seeded entry path is, by construction, a flood that crossed the border.

KNOWN LIMITATIONS (stated before use, to be quantified by validation):
- Borders broken in the SOURCE (e.g. white SFX crossing a frame) produce no barrier at the
  break -- leaks through such gaps connect around the barrier and are UNDETECTABLE by this
  method (this is the separately-documented, accepted SFX-leak limitation).
- Dark art enclosed by true background can carve false "behind-barrier" islands (over-report).
- The seed condition uses the chain's own seed; a detector consumer without the chain would
  need page-edge-connectivity only (weaker).

VALIDATION (before any use as a measurement tool): on PSD-backed parts, compare leak-candidate
pixels against actual over-deletion (pred & ~GT). Report per-part candidate px, true-leak px
among candidates (precision) and share of border-crossing over-deletion caught (recall proxy).

Usage:
  .venv/bin/python src/spiky/leak_detector.py --barrier-run none
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Promoted from .tmp/scripts-manual (2026-08-10 cleanup): data stays in gitignored .tmp/.
GOLD = HERE.parents[1] / ".tmp/scripts-manual/gold_extracted"

PARTS = ["001-1", "001-2", "001-3", "002-1", "002-2", "002-3", "033-1"]


def detect_leaks(gray: np.ndarray, pred_delete: np.ndarray, seed: np.ndarray | None):
    stroke = (gray <= 40).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    barrier = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, k) > 0

    interior = pred_delete & ~barrier
    num, labels, stats, _ = cv2.connectedComponentsWithStats(interior.astype(np.uint8), 4)
    H, W = gray.shape
    leak = np.zeros_like(pred_delete)
    n_regions = 0
    for lbl in range(1, num):
        x, y, cw, chh, area = stats[lbl]
        if area < 500:
            continue
        touches_edge = x == 0 or y == 0 or x + cw == W or y + chh == H
        if touches_edge:
            continue
        comp = labels[y : y + chh, x : x + cw] == lbl
        if seed is not None and seed[y : y + chh, x : x + cw][comp].any():
            continue
        leak[y : y + chh, x : x + cw][comp] = True
        n_regions += 1
    return leak, n_regions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--barrier-run", default="none",
                     help="which saved prediction masks to analyze (suffix from generalize_v12)")
    args = ap.parse_args()

    import subprocess
    print(subprocess.run(["date"], capture_output=True, text=True).stdout.strip(), flush=True)

    print("part | leak_px | leak_regions | leak&overdel (precision) | "
          "overdel total | border-crossing overdel caught", flush=True)
    for part in PARTS:
        pred_path = GOLD / f"{part}_pred_{args.barrier_run}.npy"
        if not pred_path.exists():
            print(f"  {part}: no saved prediction, skip", flush=True)
            continue
        pred = np.load(pred_path)
        gt = np.load(GOLD / f"{part}_gt.npy")
        rgb = np.asarray(Image.open(GOLD / f"{part}_src.png").convert("L"))
        gray = rgb

        leak, n_regions = detect_leaks(gray, pred, seed=None)
        n_leak = int(leak.sum())
        overdel = pred & ~gt
        n_over = int(overdel.sum())
        hit = int((leak & overdel).sum())
        precision = hit / max(n_leak, 1)
        print(f"  {part}: leak {n_leak:>9,} px in {n_regions:>3} regions | "
              f"precision {precision:.3f} | overdel {n_over:>9,} | "
              f"caught {100.0 * hit / max(n_over, 1):.1f}% of overdel", flush=True)


if __name__ == "__main__":
    main()
