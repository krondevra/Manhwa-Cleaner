"""Mission plan Phase 1 (.claude/plans/snazzy-cuddling-creek.md): the dominant lever identified
by the error decomposition (.tmp/diagnostics/error_decomposition.py) -- ~70-90% of under-deletion
on real chapters is large, near-black, geometrically-uniform full-width backdrop strips between
panels (the inter-panel gutter rendered black instead of white in this webtoon's source scans),
not a boundary-precision problem. Purely geometric, no model/training involved -- consistent with
this project's "8 documented failed learned halo mechanisms" conclusion that learned fixes for
this class of error don't work, but the structure itself is trivially detectable from the RGB.

Safety design (calibrated against real measurements 2026-08-05, not guessed):
- Operates on a per-pixel near-black boolean mask (threshold `darkness_threshold`, default 40 --
  this project's established FRAME_DARKNESS convention). A pixel can only be reclaimed if it is
  ITSELF near-black -- bright panel/art content is excluded by construction, confirmed directly:
  a small-panels-in-a-black-margin real region (chapter 001, y=113-8110) has GT delete-fraction
  87-91% in the true near-black margins vs. 0-13% at the rows containing actual panel content
  (gray mean 118-127, nowhere near the threshold) -- the two are cleanly separated by darkness
  alone at the pixel level.
- Per-component uniformity guard (`max_std`): real confirmed backdrop components measure
  std 1.7-10.5 (masked to their own near-black pixels); this guards against dark ART with visible
  detail (screentone, linework, highlights) being mistaken for flat backdrop -- the documented
  false-positive trap for this whole error class.
- Full-width requirement (`min_width_frac`): every real backdrop component found in the
  decomposition spans ≥90% of the page width; narrower dark blobs are more likely to be real
  content (a dark costume, hair, a shadow) and are left alone.
- Large-area requirement (`min_area`): excludes small dark specks/punctuation/thin lines.
"""
from __future__ import annotations

import numpy as np
import cv2


def reclaim_black_backdrop(
    rgb: np.ndarray,
    delete_mask: np.ndarray,
    darkness_threshold: int = 40,
    max_std: float = 20.0,
    min_width_frac: float = 0.85,
    min_area: int = 50000,
) -> np.ndarray:
    """Marks large, uniform, near-black, (near-)full-width connected regions of the RGB for
    deletion, regardless of the model's own (under-confident) prediction there. Per-component
    work inside each component's own bounding-box crop, matching this project's established
    pattern for scalability on ~150k-px-tall images (reclaim_landlocked_delete_islands,
    repair_frame_interiors)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    near_black = (gray <= darkness_threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(near_black, connectivity=8)

    fixed = delete_mask.copy()
    for label in range(1, num_labels):  # label 0 is the non-near-black background, skip
        x, y, cw, ch, area = stats[label]
        if area < min_area:
            continue
        if cw < min_width_frac * w:
            continue

        comp = labels[y : y + ch, x : x + cw] == label
        vals = gray[y : y + ch, x : x + cw][comp]
        if vals.std() > max_std:
            continue

        region = fixed[y : y + ch, x : x + cw]
        region[comp] = True

    return fixed
