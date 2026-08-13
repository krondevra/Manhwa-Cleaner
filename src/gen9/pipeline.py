"""Gen9 deterministic core: pixel-exact port of the user's Photopea
algorithm (.tmp/gen9/new-pipeline.md), decoded against their own working
PSD (.tmp/gen9/002_1.psd) -- see decisions.md 2026-08-13 16:17.

The chain is deterministic end to end except two operator judgment calls
(background selection, boundary-glyph selection), which live in
classify_bg.py / classify_glyph.py. Nothing here composes classifiers;
compose_delete takes their outputs as plain masks/ids.

Verified conventions (100.0% vs the PSD's own intermediate layers):
- Levels(lo,1,hi) = per-channel linear ramp, float, no rounding needed.
- Threshold(t)    = luminosity, white where lum >= t. Empirical weights
                    per clone (each 100.0% exact where used, measured
                    2026-08-13): clone-1 matches Rec.709 exactly;
                    clone-2 matches Rec.601 exactly AFTER Min/Max (the
                    3x3 pass washes out the pre-threshold differences, so
                    either could be Photopea's true formula -- we pin the
                    empirically exact pair).
- Minimum/Maximum(1px) = erode/dilate with a 3x3 square (Photopea's
                    square structuring element; blocky steps on curves
                    are a KNOWN accepted artifact this generation).
- Final mask      = dilate1_sq3(selected background comps) AND NOT
                    clone1-black, plus expand-4px glyph fills. (The
                    written steps build the mask in inverted polarity --
                    their "fill black" ops are protective; the PSD file
                    is authoritative.)

Levels history (user-tuned, rejected values kept for the record):
clone-1 tried (38,1,39) and (51,1,52); (33,1,34) is current best.
"""
from __future__ import annotations

import cv2
import numpy as np

# clone-1: near-black ink map (dark strokes survive as black)
LV1 = (33.0, 34.0)
T1 = 226.0
# clone-2: near-white background map (true paper white survives as white)
LV2 = (248.0, 249.0)
T2 = 178.0

REC709 = (0.2126, 0.7152, 0.0722)
REC601 = (0.299, 0.587, 0.114)
SQ3 = np.ones((3, 3), np.uint8)
GLYPH_EXPAND = 4   # step "Select > Modify > Expand > 4 px"


def levels(rgb: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Photopea Levels(lo, 1, hi): per-channel linear ramp to [0,255]."""
    y = (rgb.astype(np.float64) - lo) / (hi - lo)
    np.clip(y, 0.0, 1.0, out=y)
    return y * 255.0


def threshold_lum(rgb255: np.ndarray, t: float, w) -> np.ndarray:
    """Photopea Threshold(t) on weighted luminosity -> uint8 {0, 255}."""
    lum = w[0] * rgb255[..., 0] + w[1] * rgb255[..., 1] + w[2] * rgb255[..., 2]
    return np.where(lum >= t, 255, 0).astype(np.uint8)


def clone1(src_rgb: np.ndarray) -> np.ndarray:
    """Steps 7-10: Levels(33,1,34) -> Threshold(226). Black = dark ink."""
    return threshold_lum(levels(src_rgb, *LV1), T1, REC709)


def clone2(src_rgb: np.ndarray) -> np.ndarray:
    """Steps 14-19: Levels(248,1,249) -> Threshold(178) -> Min(1) -> Max(1).
    White = near-paper-white background candidate."""
    t = threshold_lum(levels(src_rgb, *LV2), T2, REC601)
    return cv2.dilate(cv2.erode(t, SQ3), SQ3)


def bg_components(c2: np.ndarray):
    """4-connected components of clone-2 white. Returns (n, labels, stats)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        (c2 >= 128).astype(np.uint8), connectivity=4)
    return n, lab, st


def compose_delete(shape: tuple[int, int], bg_labels: np.ndarray,
                   selected_ids, c1: np.ndarray,
                   glyph_marks: np.ndarray | None = None) -> np.ndarray:
    """The decoded final-mask formula.

    DELETE = dilate1_sq3(union of selected background components)
             AND NOT clone1-black,
             OR dilate4(selected glyph stroke pixels).
    """
    bg = np.isin(bg_labels, np.asarray(list(selected_ids))).astype(np.uint8)
    delete = cv2.dilate(bg, SQ3).astype(bool) & ~(c1 < 128)
    if glyph_marks is not None and glyph_marks.any():
        k = 2 * GLYPH_EXPAND + 1
        delete |= cv2.dilate(glyph_marks.astype(np.uint8),
                             np.ones((k, k), np.uint8)).astype(bool)
    return delete
