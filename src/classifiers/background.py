"""Background classifier: connected-component / reachability primitives.

EXTRACTED (2026-08-10, gen-8 phase 1) verbatim from `src/spiky/pipeline.py` -- the
validated v6/v24-v27 logic, byte-identical behavior confirmed by the equivalence gate
and the full battery at extraction time. This is a mechanical move, not a redesign:
every function body is unchanged from its pipeline.py original; only the names lost
their leading underscore.

Provenance per function:
- `enclosed` / `flood`         -- v8 (plan v21): exact 4-connectivity primitives.
- `protected_interiors`        -- v6 (plan v16): border-disconnected holes >= 10k px
                                  behind 5x5-closed near-black strokes.
- `protected_interiors_v2`     -- v12 Fix A (plan v26): + closed-contour OWNERSHIP test
                                  (>= PROT_DOMINANCE of the hole's ink-adjacent boundary
                                  owned by ONE un-closed stroke component -- a real frame
                                  contour). This is the full-page-context fix: inter-panel
                                  gutters (whose surrounding strokes belong to different
                                  frames) correctly FAIL the test and stay deletable.

All functions take WHOLE-PAGE arrays; any windowing happens at the call site (the
standing full-page-context lesson from the v25/v26 root-cause round -- crop-local
protection detection is exactly the bug class Fix A closed).

Sibling implementations deliberately NOT merged here (different validation lineages):
`ml_cleaner.repair_frame_interiors` (production ML postprocessing) and
`style_analysis.extract_enclosed_holes` (shape-taxonomy detection). Cross-reference
only; do not unify without their own equivalence gates.

Decision-boundary bias (gen-8 standing principle): when uncertain, DELETE background
over PRESERVE frame content -- `classify_background` keeps protection scoped to
interiors that PASS the v2 ownership test (a real, single-contour frame) rather than
protecting every border-disconnected hole; ambiguous enclosures stay deletable.
"""
from __future__ import annotations

import cv2
import numpy as np

PROT_DOMINANCE = 0.90  # v12 Fix A ownership threshold (plan v26)


def enclosed(passable: np.ndarray) -> np.ndarray:
    """Regions of the passable map NOT connected to the image border (exact, 4-conn)."""
    num, lab = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
    border = np.zeros(num, dtype=bool)
    for edge in (lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]):
        border[np.unique(edge)] = True
    border[0] = True
    return ~border[lab] & passable


def flood(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Union of 4-connected components of mask that contain any seed px (exact)."""
    num, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    hit = np.zeros(num, dtype=bool)
    hit[np.unique(lab[seed & mask])] = True
    hit[0] = False
    return hit[lab]


def protected_interiors(gray: np.ndarray) -> np.ndarray:
    """Closed-frame interiors (repair_frame_interiors-style): holes >= 10k px fully enclosed
    by near-black strokes. Pockets inside these are NEVER reclaimed (3(d) negative guard)."""
    stroke = (gray <= 40).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, k)
    H, W = gray.shape
    padded = np.zeros((H + 2, W + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = stroke
    ff = np.zeros((H + 4, W + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 1)
    holes = (padded[1:-1, 1:-1] == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    protected = np.zeros((H, W), dtype=bool)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= 10000:
            protected[labels == lbl] = True
    return protected


def protected_interiors_v2(gray: np.ndarray) -> np.ndarray:
    """Fix A: closed-contour ownership test on top of the v6 hole detection."""
    stroke_closed = (gray <= 40).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stroke_closed = cv2.morphologyEx(stroke_closed, cv2.MORPH_CLOSE, k)
    H, W = gray.shape
    padded = np.zeros((H + 2, W + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = stroke_closed
    ff = np.zeros((H + 4, W + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 1)
    holes = (padded[1:-1, 1:-1] == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)

    raw_stroke = (gray <= 40).astype(np.uint8)  # UN-closed: no cross-frame bridging
    snum, slab = cv2.connectedComponents(raw_stroke, connectivity=8)
    k3 = np.ones((3, 3), np.uint8)
    protected = np.zeros((H, W), dtype=bool)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] < 10000:
            continue
        x, y, w, h = (int(stats[lbl, j]) for j in range(4))
        x0, y0 = max(0, x - 4), max(0, y - 4)
        x1, y1 = min(W, x + w + 4), min(H, y + h + 4)
        hole = (labels[y0:y1, x0:x1] == lbl).astype(np.uint8)
        ring = (cv2.dilate(hole, k3, iterations=3) > 0) & (hole == 0)
        touched = slab[y0:y1, x0:x1][ring]
        touched = touched[touched != 0]
        if len(touched) == 0:
            continue
        counts = np.bincount(touched)
        if counts.max() / len(touched) >= PROT_DOMINANCE:
            protected[labels == lbl] = True
    return protected


def classify_background(page_rgb: np.ndarray) -> np.ndarray:
    """Convenience entry: True = pixel is protectable frame-interior content, i.e. NOT
    safely-deletable background context. Uses the v2 ownership test (delete-over-preserve
    bias: only interiors provably owned by one real frame contour are protected;
    everything ambiguous stays deletable). Built on the same primitives the validated
    pipeline uses -- no separate heuristic."""
    f = page_rgb.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    return protected_interiors_v2(gray)
