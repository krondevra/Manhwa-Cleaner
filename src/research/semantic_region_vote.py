"""Plan v10 Part 1 (.claude/plans/snazzy-cuddling-creek.md): KEEP-SIDE-ONLY semantic region
vote -- the safe half of the region-level signal found by
.tmp/diagnostics/region_semantics_probe.py (2026-08-06).

Probe findings that shape this design (all real per-instance measurements):
- band_density (fraction of non-near-black content in the rows +/-300 around a candidate
  component -- pure layout context, a feature class no closed mechanism ever had) separates
  clean ART components (GT-keep dark art embedded in a scene) from BACKDROP strips with
  non-overlapping IQRs on both the fit (001+002) and held-out (035) chapters.
- The DELETE side is deliberately NOT implemented: the canonical HUD component and the night
  cityscape are region-level indistinguishable from backdrop on every measured feature
  (band_density 0.266/0.377, bright_obj_frac 0.062/0.001 -- inside the BACKDROP distribution).
  Any rule aggressive enough to complete 035's under-deleted strips would wholesale-delete
  those art instances -- the reclaim_black_backdrop rung's catastrophe (class mean 20.7->74.3%)
  reproduced at region granularity. This is a measured two-level indistinguishability, and the
  under-deletion prize stays locked.
- The KEEP action's failure mode (falsely reverting true backdrop -> under-deletion) is the
  cheap direction, and the fit-side BACKDROP maximum band_density is 0.439 (held-out max
  0.715). The default threshold 0.75 = fit-side max x ~1.7 safety margin; the held-out
  distribution confirms zero false positives at this value but was NOT used to choose it.

Action: for each candidate component (d1_region_vote's exact criteria: near-black <=40, area
>=50k, width >=85%) whose band_density >= density_thresh, revert the component's currently-
deleted pixels back to keep. Runs AFTER d1_region_vote (D1's low-band revert already handles
nearly-kept components; this catches art components the model deleted more aggressively).
"""
from __future__ import annotations

import cv2
import numpy as np

FRAME_DARKNESS = 40
MIN_AREA = 50000
MIN_WIDTH_FRAC = 0.85
DENSITY_THRESH = 0.75
BAND_ROWS = 300


def semantic_region_vote(
    rgb: np.ndarray,
    delete_mask: np.ndarray,
    darkness_threshold: int = FRAME_DARKNESS,
    min_area: int = MIN_AREA,
    min_width_frac: float = MIN_WIDTH_FRAC,
    density_thresh: float = DENSITY_THRESH,
    band_rows: int = BAND_ROWS,
) -> np.ndarray:
    """postprocess(rgb, mask) -> mask. Keep-side only by design (see module docstring)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    near_black = (gray <= darkness_threshold).astype(np.uint8)
    row_content_frac = (gray > darkness_threshold).mean(axis=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(near_black, connectivity=8)
    fixed = delete_mask.copy()
    for lbl in range(1, num):
        x, y, cw, chh, area = stats[lbl]
        if area < min_area or cw < min_width_frac * w:
            continue
        b0, b1 = max(0, y - band_rows), min(h, y + chh + band_rows)
        band_density = float(np.mean(row_content_frac[b0:b1]))
        if band_density < density_thresh:
            continue
        comp = labels[y : y + chh, x : x + cw] == lbl
        region = fixed[y : y + chh, x : x + cw]
        region[comp] = False
    return fixed
