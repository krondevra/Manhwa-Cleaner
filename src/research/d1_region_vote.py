"""Mission plan v9 Phase F (originally Probe 0's D1, mission plan v6): component-level
deleted_frac band-vote on large full-width near-black components -- the independent ~1pp
aggregate win found during the blocker-#1 investigation and parked for Recipe A assembly.

Validated end-to-end by .tmp/diagnostics/d1_bandvote_simulate.py (2026-08-05): classification
accuracy 89%/92% on held-out ch035; offline aggregate 14.23% -> 13.20% on cached v3+islands
masks. Thresholds were fit on ch001+002 ONLY (component_feature_table.py) and are left
untouched here (the anti-overfit protocol):

  deleted_frac <= T_LO (0.072): the model kept nearly all of a large full-width near-black
    component -> it is a kept dark art panel the model only nibbled at; revert the nibbles
    (delete -> keep).
  deleted_frac >= T_HI (0.602): the component is solidly deleted -> it is true
    backdrop/gutter; complete the remainder (keep -> delete), same effect as
    reclaim_black_backdrop's uniformity path but voted on the model's own output.
  in between: mixed component (real gutters physically fused with dark art) -- NO vote, leave
    unchanged. This band is exactly where blocker #1's class lives; D1 by design does not
    touch it (that is reclaim_patchy_deletion's job).
"""
from __future__ import annotations

import cv2
import numpy as np

FRAME_DARKNESS = 40
MIN_AREA = 50000
MIN_WIDTH_FRAC = 0.85
T_LO = 0.072
T_HI = 0.602


def d1_region_vote(
    rgb: np.ndarray,
    delete_mask: np.ndarray,
    darkness_threshold: int = FRAME_DARKNESS,
    min_area: int = MIN_AREA,
    min_width_frac: float = MIN_WIDTH_FRAC,
    t_lo: float = T_LO,
    t_hi: float = T_HI,
) -> np.ndarray:
    """postprocess(rgb, mask) -> mask. Per-component work inside each component's bounding-box
    crop (project convention for ~150k-px-tall strips)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    near_black = (gray <= darkness_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(near_black, connectivity=8)

    fixed = delete_mask.copy()
    for label in range(1, num_labels):
        x, y, cw, chh, area = stats[label]
        if area < min_area or cw < min_width_frac * w:
            continue
        comp = labels[y : y + chh, x : x + cw] == label
        comp_mask = fixed[y : y + chh, x : x + cw]
        deleted_frac = float((comp & comp_mask).sum() / comp.sum())
        if deleted_frac <= t_lo:
            fixed[y : y + chh, x : x + cw][comp] = False
        elif deleted_frac >= t_hi:
            fixed[y : y + chh, x : x + cw][comp] = True
    return fixed
