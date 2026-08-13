"""Gen9 Classifier A: which clone-2-white components are the inter-frame
background (the step-11 operator clicks). Single-purpose; no interaction
with Classifier B beyond both feeding compose_delete.

Measured GT (002_1.psd): the operator selected exactly the components that
span the FULL strip width (edge-to-edge gutter bands, 689-690 of W=690);
one-side slivers (width <= 303) and panel-interior whites are never
selected. The full-width band touching the page top (margin above the
site-banner/header block) is KEPT by operator convention -- the mask
starts below the header (y160 on 002_1).

Sealed SFX pockets (background enclosed inside SFX shapes): the user's
stated heuristic -- enclosed shape within ~50 px of a background region
whose adjacent contours are thick/connected = SFX pocket (deletable), vs
spiky-cloud interior (protected). NO positive GT exists on 002_1 (the
operator deleted zero non-edge components), so find_sealed_pockets ships
REPORT-ONLY: it lists candidates and never feeds the delete mask. The
50 px constant is the user's estimate, unvalidated -- do not tune blind.
"""
from __future__ import annotations

import cv2
import numpy as np

FULL_WIDTH_SLACK = 2     # comp width >= W - slack counts as full-width
POCKET_NEAR_BG = 50      # user's estimate, unvalidated (no GT positives)
POCKET_RING = 3          # px ring inspected around a pocket candidate
POCKET_RING_INK = 0.8    # ring must be at least this fraction dark ink


def select_background(labels: np.ndarray, stats: np.ndarray,
                      keep_top_band: bool = True) -> list[int]:
    """Return component ids (of clone-2 white) that are background bands.

    Rule A1: a component is background iff it spans the full strip width.
    With keep_top_band, a full-width component touching row 0 is the
    header margin and stays (operator convention on this series).
    """
    W = labels.shape[1]
    ids = []
    for i in range(1, stats.shape[0]):
        x0, y0, w, h, _ = stats[i]
        if w < W - FULL_WIDTH_SLACK:
            continue
        if keep_top_band and y0 == 0:
            continue
        ids.append(i)
    return ids


def find_sealed_pockets(labels: np.ndarray, stats: np.ndarray,
                        selected_ids: list[int], c1: np.ndarray) -> list[dict]:
    """REPORT-ONLY candidates for background pockets sealed inside SFX.

    A candidate is a non-selected component that (a) sits within
    POCKET_NEAR_BG px of the selected background field and (b) is enclosed
    by dark ink (its POCKET_RING-px ring is >= POCKET_RING_INK clone-1
    black). Returned for the report; never fed to compose_delete.
    """
    bg = np.isin(labels, np.asarray(selected_ids))
    k = 2 * POCKET_NEAR_BG + 1
    near_bg = cv2.dilate(bg.astype(np.uint8),
                         np.ones((k, k), np.uint8)).astype(bool)
    ink = c1 < 128
    out = []
    sel = set(selected_ids)
    for i in range(1, stats.shape[0]):
        if i in sel:
            continue
        x0, y0, w, h, area = (int(v) for v in stats[i][:5])
        ys, ye = max(0, y0 - POCKET_RING), y0 + h + POCKET_RING
        xs, xe = max(0, x0 - POCKET_RING), x0 + w + POCKET_RING
        comp = labels[ys:ye, xs:xe] == i
        if not near_bg[ys:ye, xs:xe][comp].any():
            continue
        ring = cv2.dilate(comp.astype(np.uint8),
                          np.ones((2 * POCKET_RING + 1,) * 2,
                                  np.uint8)).astype(bool) & ~comp
        if ring.sum() == 0:
            continue
        ink_frac = float(ink[ys:ye, xs:xe][ring].mean())
        if ink_frac >= POCKET_RING_INK:
            out.append(dict(comp_id=i, bbox=(x0, y0, x0 + w, y0 + h),
                            area=area, ring_ink=round(ink_frac, 3)))
    return out
