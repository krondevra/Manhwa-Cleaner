"""sfx.py -- prototype automation of the user's manual SFX recipe (gen-8 sfx_glyph
step 4): explicit COMPOSITION of the three classifiers, not a fourth mechanism.

Recipe (decoded pixel-exact from the 6 reference PSDs, 8.6.1):
  pass 1  aggressive binarize: white = G >= 33  (Levels 32,1,33 -> Threshold 140).
          White is the DELETE proposal (the recipe's magic-wand-white raster mask).
  frame   whole-frame-interior keep, the recipe's manual rectangle restore -- here
          from `classifiers.frame`'s validated line inventory via a conservative
          bounding rule (see below), NOT the unvalidated rect grouping.
  sfx     per-object keep: profile detections (classifiers.profiles.sfx_glyph)
          dilated by the measured two-level Expand E = 2 px (stroke width < 2.5)
          else 4 px (8.6.1: E_del ~2-3 thin / ~4-5 thick, prose "2 small / 4+ large";
          delete-over-preserve favors the smaller side).
  pass 2  SFX-preservation rescue inside detection regions only: ink2 = min(R,G,B)
          < 50 (Levels 49,1,50 -> Threshold 230, solved from 005.psd) -- rescues
          saturated mid-tone gradients inside colored SFX that pass 1 bleaches.
  bubble  enclosed light pockets (background.enclosed of the pass-1 white band)
          kept with the SAME Expand mechanism (pocket dilated by E_BUBBLE) -- the
          recipe's wand-ON click inside a pocket + Expand covers interior + wall +
          halo, which is exactly how the reference GT keeps gutter speech bubbles.

Frame keep (conservative, LOGGED simplification): border-quality lines = thin
(<= BORDER_THICK), long (>= BORDER_SPAN_FRAC of the page dimension), not canvas-edge
artifacts. The kept frame region is the bounding band spanned by them (y-extent from
horizontal border lines, x-extent from vertical ones, defaulting to full width /
height when a side has no border line). This over-keeps gutters between panels on
multi-panel pages -- deliberately: the zero-frame-content-loss guard is hard, the
delete bias is soft. Automatic exact panel polygons remain phase-2's open problem
(rect grouping unvalidated).

delete = pass1_white AND NOT (frame_keep OR sfx_keep OR bubble_keep)

Acceptance (sfx_suite.prototype_eval): per-file pixel agreement vs the PSD GT inside
the crop canvas + HARD guard: zero deleted px inside the annotated frame rects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from classifiers.background import enclosed  # noqa: E402
from classifiers.detector_framework import detect  # noqa: E402
from classifiers.frame import bridge_collinear, detect_lines_morph  # noqa: E402
from classifiers.profiles.sfx_glyph import PROFILE as SFX_PROFILE  # noqa: E402
from classifiers.profiles.sfx_glyph import CUT_AGGR  # noqa: E402

CUT_PRESERVE = 50      # 8.6.1: preservation pass = min(R,G,B) >= 50 white
E_THIN, E_THICK = 2, 4  # measured two-level per-object Expand
E_W_SPLIT = 2.5        # stroke-width split between the two levels
E_BUBBLE = 4           # pocket Expand (recipe wand-ON + Expand 4 on bubbles)
POCKET_MIN = 3000      # px: smaller enclosed pockets are glyph loops / small art
                       # holes, not bubbles (GT deletes a glyph loop's interior;
                       # aligned with the spiky candidate band's G_INT_MIN)
BORDER_THICK = 20      # border-quality line: stroke thickness cap (corpus median
                       # ~12px; 004's left border measures 17)
BORDER_SPAN_FRAC = 0.4  # ... and span at least this fraction of the page dimension
EDGE_MARGIN = 3        # lines within this of the canvas edge are crop artifacts


def _border_lines(rgb: np.ndarray):
    f = rgb.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    h_lines, v_lines = detect_lines_morph(gray)
    h_lines, v_lines = bridge_collinear(h_lines), bridge_collinear(v_lines)
    H, W = gray.shape
    hb = [ln for ln in h_lines
          if ln.thick <= BORDER_THICK and ln.span[1] - ln.span[0] >= BORDER_SPAN_FRAC * W
          and EDGE_MARGIN <= ln.pos <= H - 1 - EDGE_MARGIN]
    vb = [ln for ln in v_lines
          if ln.thick <= BORDER_THICK and ln.span[1] - ln.span[0] >= BORDER_SPAN_FRAC * H
          and EDGE_MARGIN <= ln.pos <= W - 1 - EDGE_MARGIN]
    return hb, vb


def frame_keep_mask(rgb: np.ndarray) -> np.ndarray:
    """Conservative whole-frame keep: bounding band of border-quality lines."""
    H, W = rgb.shape[:2]
    hb, vb = _border_lines(rgb)
    keep = np.zeros((H, W), bool)
    if not hb and not vb:
        return keep
    # an axis needs TWO border lines to bound the band; with one line the panel's
    # side is unknowable -> keep the full extent on that axis (conservative: the
    # zero-frame-loss guard is hard, the delete bias soft). Measured failure: a
    # single-h-line page collapsed the band to the line's own 2px and deleted the
    # panel interior.
    y0, y1 = (min(ln.pos for ln in hb),
              max(ln.pos + max(1, ln.thick) for ln in hb)) if len(hb) >= 2 else (0, H)
    x0, x1 = (min(ln.pos for ln in vb),
              max(ln.pos + max(1, ln.thick) for ln in vb)) if len(vb) >= 2 else (0, W)
    keep[y0:y1, x0:x1] = True
    return keep


def clean_sfx_region(rgb: np.ndarray) -> np.ndarray:
    """Returns the delete mask (True = delete) for one page/crop."""
    H, W = rgb.shape[:2]
    white1 = rgb[..., 1] >= CUT_AGGR
    keep = frame_keep_mask(rgb)

    # --- SFX keeps: profile detections, pass-2 rescue inside the region, expand ---
    ink1 = ~white1
    sfx_keep = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in detect(rgb, SFX_PROFILE):
        # a colored SFX glyph extends far beyond its pass-1 dark core (the profile's
        # detection): rescue by CONNECTIVITY under the pass-2 predicate in a
        # generously padded window, seeded from the detection -- measured failure
        # without this: 005's gutter blue-gradient glyph fully deleted because the
        # rescue was confined to the core's bbox.
        pad = max(E_THICK + 6, 2 * max(x1 - x0, y1 - y0))
        wy0, wy1 = max(0, y0 - pad), min(H, y1 + pad)
        wx0, wx1 = max(0, x0 - pad), min(W, x1 + pad)
        win = rgb[wy0:wy1, wx0:wx1]
        ink2 = (win.min(axis=2) < CUT_PRESERVE) | ink1[wy0:wy1, wx0:wx1]
        n2, lab2 = cv2.connectedComponents(ink2.astype(np.uint8), connectivity=8)
        seed = np.zeros_like(ink2)
        seed[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0] = \
            ink1[y0:y1, x0:x1]
        touched = np.unique(lab2[seed])
        obj = np.isin(lab2, touched[touched > 0])
        dc = cv2.distanceTransform(obj.astype(np.uint8), cv2.DIST_L2, 3)
        wvals = dc[dc > 0]
        w_med = float(np.median(wvals) * 2.0) if wvals.size else 0.0
        e = E_THIN if w_med < E_W_SPLIT else E_THICK
        k = 2 * e + 1
        grown = cv2.dilate(obj.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        sfx_keep[wy0:wy1, wx0:wx1] |= grown
    # --- bubble keeps: enclosed pockets + wall + halo via the same Expand ---
    pockets = enclosed(white1)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        pockets.astype(np.uint8), connectivity=8)
    bubble_keep = np.zeros((H, W), bool)
    kb = 2 * E_BUBBLE + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) < POCKET_MIN:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        pad = E_BUBBLE + 2
        wy0, wy1 = max(0, y - pad), min(H, y + h + pad)
        wx0, wx1 = max(0, x - pad), min(W, x + w + pad)
        grown = cv2.dilate((lab[wy0:wy1, wx0:wx1] == i).astype(np.uint8), ker) > 0
        bubble_keep[wy0:wy1, wx0:wx1] |= grown

    return white1 & ~keep & ~sfx_keep & ~bubble_keep
