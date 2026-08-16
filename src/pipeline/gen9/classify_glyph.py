"""Gen9 Classifier B: which glyph structures at the frame/background
boundary (or floating in background) get the step-15 delete treatment.

Measured GT (002_1.psd): every one of the operator's six expand-4 fills
is a SEALED BACKGROUND POCKET -- the white trapped inside a glyph loop or
between strokes (0% stroke px, 77-95% clone-2-white; each fill maps to one
small non-edge clone-2-white component). The glyph STROKES themselves are
kept by the clone-1-black protection in the formula; what needs deleting
is the enclosed background. So B detects glyph OBJECTS and returns their
enclosed pocket components, which compose_delete treats exactly like
background components (dilate-1 ring, clone-1-black protected).

Glyph-object rule (the user's thickness/connectivity heuristic, measured):
a connected ink structure inside the near-background band that is COMPACT
(bbox <= GLYPH_BBOX_MAX) and THICK (max inscribed radius >= GLYPH_THICK
via distance transform). Long thin panel borders and spiky-cloud contours
fail the compact/thick test; the six GT glyphs measure 3.6-7.4 max
thickness at 38-208 px bbox. Panel-interior glyphs are ignored by
construction: the near-background band never reaches them.

GT recovery on 002_1: 6/6 pockets, plus 19 beyond-etalon candidates (all
visually verified glyph pockets -- the semi-etalon is under-clicked per
the user's 2026-08-13 ruling; they are returned separately so the caller
can include them and report/preview them distinctly).
"""
from __future__ import annotations

import cv2
import numpy as np

NEAR_BG = 25          # band width around background field
GLYPH_MIN_AREA = 80   # ink comp smaller than this is noise/speck
GLYPH_BBOX_MAX = 250  # compact: glyphs, not borders/cloud contours
GLYPH_THICK = 3.0     # max inscribed radius (px) -- user's "thicker" rule
GLYPH_REACH = 12      # pocket must lie within this of a glyph object
POCKET_MIN = 30       # pocket component area bounds
POCKET_MAX = 3000
POCKET_DMIN = 3.0     # sealed: not touching background...
POCKET_DMAX = 30.0    # ...but near it (glyph sits in/at the bg field)


def glyph_objects(c1: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Boolean mask of compact+thick ink structures near the bg field."""
    ink = c1 < 128
    k = np.ones((2 * NEAR_BG + 1,) * 2, np.uint8)
    band = cv2.dilate(bg.astype(np.uint8), k).astype(bool)
    band_ink = (ink & band).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(band_ink, connectivity=8)
    dt = cv2.distanceTransform(band_ink, cv2.DIST_L2, 5)
    out = np.zeros(c1.shape, bool)
    for i in range(1, n):
        _, _, w, h, a = st[i]
        if a < GLYPH_MIN_AREA or w > GLYPH_BBOX_MAX or h > GLYPH_BBOX_MAX:
            continue
        comp = lab == i
        if dt[comp].max() >= GLYPH_THICK:
            out |= comp
    return out


def select_pockets(labels: np.ndarray, stats: np.ndarray,
                   selected_ids: list[int], c1: np.ndarray
                   ) -> list[dict]:
    """Sealed background pockets of glyph objects, as component records.

    Returns dicts (comp_id, bbox, area, dist_bg); caller decides which to
    feed compose_delete (add to the selected background ids -- the ring
    formula then matches the GT fills' 0%-stroke composition).
    """
    bg = np.isin(labels, np.asarray(selected_ids))
    glyph = glyph_objects(c1, bg)
    k = np.ones((2 * GLYPH_REACH + 1,) * 2, np.uint8)
    glyph_near = cv2.dilate(glyph.astype(np.uint8), k).astype(bool)
    dist_bg = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 5)
    sel = set(selected_ids)
    out = []
    for i in range(1, stats.shape[0]):
        if i in sel:
            continue
        x0, y0, w, h, a = (int(v) for v in stats[i][:5])
        if a < POCKET_MIN or a > POCKET_MAX:
            continue
        comp = labels == i
        ys, xs = np.where(comp)
        dmin = float(dist_bg[ys, xs].min())
        if dmin < POCKET_DMIN or dmin > POCKET_DMAX:
            continue
        if not glyph_near[ys, xs].any():
            continue
        out.append(dict(comp_id=i, bbox=(x0, y0, x0 + w, y0 + h),
                        area=a, dist_bg=round(dmin, 1)))
    return out
