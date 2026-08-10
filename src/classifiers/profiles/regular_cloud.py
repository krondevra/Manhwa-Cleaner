"""regular_cloud profile: the existing bubble-family classifier wrapped in the
framework, plus targeted fixes for the phase-3 measured defects.

Gen-8 phase 4 (2026-08-10). Candidate generation = the EXISTING validated classifier
(`style_analysis.extract_enclosed_holes`, Revision-2 state) -- not reimplemented; this
profile adds framework-level signals on top of its output. Baseline it must beat
(cloud_suite, 8.3.1): set A recall 68/92, set B 12/12, set C frame-as-cloud FP pages
4/20 (all 'rectangle'-class panel rects entering the bubble taxonomy).

Attempt 1 (one variable): FRAME-LINE ALIGNMENT rejection. A bubble-family candidate
whose bbox sides coincide with page-scale detected frame lines
(`classifiers.frame.detect_lines_morph` + `bridge_collinear`, the phase-2 VALIDATED
inventory) is a panel/frame artifact, not a bubble -- reject. Delete-over-preserve
consistent: rejecting a false bubble removes a wrongly-protective shape (more deletion,
not less). Real bubbles almost never have sides collinear with long page frame lines;
the suite's set A measures exactly that assumption.
  RESULT (2026-08-10): A/B recall preserved, but 0 of the 4 FP pages fixed --
  measured root cause: line pos = stroke CENTER while the hole's bbox side sits at the
  stroke INNER EDGE; on thick synthetic panel strokes the offset is 19-38px, far past
  the fixed 3px tolerance. Counted failure #1; led directly to attempt 2.

Attempt 2 (one variable): THICKNESS-AWARE alignment -- `Line.thick` (measured stroke
thickness, added to the frame module) widens the coincidence tolerance to
thick/2 + ALIGN_TOL, so a hole edge at a thick stroke's inner boundary is recognized
as lying on that stroke.
  RESULT (2026-08-10): SUCCESS -- set C FP pages 4 -> 2, sets A/B recall preserved.
  Counter reset. Remaining 2 FPs diagnosed: line-inventory gaps on those boxes' own
  strokes + partial span overlap (coverage 0.00/0.44 under the 0.6 bar).

Attempt 3 (one variable): STROKE-THICKNESS signal -- independent of the line
inventory entirely. A bubble-family candidate whose enclosing stroke measures thick
(median 2x distance-transform >= STROKE_THICK_MAX in the ring just outside its bbox)
is a panel box, not a bubble: corpus statistics (notes/style_analysis_findings.md)
put frame border thickness at median ~12px (5th-95th 1.9-143px) while drawn bubble
outlines are thin. Threshold 10px.
  RESULT (2026-08-10): COUNTED FAILURE (regress-elsewhere) -- set C 2 -> 1 but set A
  recall 68 -> 62: the ring measurement conflates enclosing stroke with ANY nearby
  dark mass; the 7 wrongly-rejected reals are thorn/other bubbles adjacent to art/dark
  scenes ("thickness" 12.9-147px is background ink, not outline). All true FPs are
  class 'rectangle'.

Attempt 4 (one variable): CLASS-SCOPED thickness -- the stroke-thickness gate applies
to 'rectangle'-class candidates only (every measured FP is a rectangle panel box;
every measured false rejection was thorn/other). Non-rectangle candidates auto-pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from style_analysis import extract_enclosed_holes  # noqa: E402
from classifiers.frame import bridge_collinear, detect_lines_morph  # noqa: E402
from classifiers.detector_framework import Profile, Region, Signal  # noqa: E402

BUBBLE_FAMILY = {"oval", "cloud", "spiky", "thorn", "rectangle", "other"}
ALIGN_TOL = 3        # px: bbox side to detected-line distance counted as coincident
ALIGN_MAX = 0.6      # reject when >= this fraction of a side's length lies on a line

_line_cache: dict[int, tuple[list, list]] = {}


def _page_lines(page: np.ndarray):
    key = id(page)
    if key not in _line_cache:
        if len(_line_cache) > 8:
            _line_cache.clear()
        if page.ndim == 3:
            f = page.astype(np.float32)
            gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
        else:
            gray = page
        h, v = detect_lines_morph(gray)
        _line_cache[key] = (bridge_collinear(h), bridge_collinear(v))
    return _line_cache[key]


_cls_cache: dict[tuple[int, Region], str] = {}


def _candidates(page: np.ndarray) -> list[Region]:
    rgb = page if page.ndim == 3 else np.stack([page] * 3, axis=-1)
    out: list[Region] = []
    if len(_cls_cache) > 8192:
        _cls_cache.clear()
    for s in extract_enclosed_holes(rgb):
        if s["class"] not in BUBBLE_FAMILY:
            continue
        x, y, w, h = s["bbox"]
        region = (int(x), int(y), int(x + w), int(y + h))
        _cls_cache[(id(page), region)] = s["class"]
        out.append(region)
    return out


def _frame_alignment(page: np.ndarray, region: Region) -> float:
    """Max fraction of any bbox side lying on a detected page-scale frame line."""
    h_lines, v_lines = _page_lines(page)
    x0, y0, x1, y1 = region
    best = 0.0
    for pos, span_lo, span_hi, length in (
        (y0, x0, x1, x1 - x0), (y1, x0, x1, x1 - x0),
    ):
        for ln in h_lines:
            if abs(ln.pos - pos) <= ln.thick / 2 + ALIGN_TOL:
                cover = min(ln.span[1], span_hi) - max(ln.span[0], span_lo)
                if length > 0:
                    best = max(best, cover / length)
    for pos, span_lo, span_hi, length in (
        (x0, y0, y1, y1 - y0), (x1, y0, y1, y1 - y0),
    ):
        for ln in v_lines:
            if abs(ln.pos - pos) <= ln.thick / 2 + ALIGN_TOL:
                cover = min(ln.span[1], span_hi) - max(ln.span[0], span_lo)
                if length > 0:
                    best = max(best, cover / length)
    return best


STROKE_RING = 12        # px outside the bbox sampled for enclosing-stroke thickness
STROKE_DARK = 40        # style_analysis FRAME_DARKNESS ink threshold
STROKE_THICK_MAX = 10   # px: >= this median stroke thickness = panel box, reject
                        # (corpus: frame borders median ~12px; bubble outlines thin)


def _stroke_thickness(page: np.ndarray, region: Region) -> float:
    """Median stroke thickness (2x distance transform) of dark ink in the ring just
    outside the candidate bbox -- the stroke that encloses this hole. Attempt 4:
    applies to 'rectangle'-class candidates ONLY (returns 0.0 = auto-pass otherwise);
    see the attempt log in the module docstring for the measured evidence."""
    import cv2
    if _cls_cache.get((id(page), region)) != "rectangle":
        return 0.0
    x0, y0, x1, y1 = region
    H, W = page.shape[:2]
    pad = STROKE_RING + 20
    wy0, wy1 = max(0, y0 - pad), min(H, y1 + pad)
    wx0, wx1 = max(0, x0 - pad), min(W, x1 + pad)
    win = page[wy0:wy1, wx0:wx1]
    if win.ndim == 3:
        f = win.astype(np.float32)
        gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    else:
        gray = win
    stroke = (gray <= STROKE_DARK).astype(np.uint8)
    if not stroke.any():
        return 0.0
    dist = cv2.distanceTransform(stroke, cv2.DIST_L2, 3)
    # ring band: outside the bbox but within STROKE_RING px of it
    yy, xx = np.mgrid[wy0:wy1, wx0:wx1]
    inside = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    near = (xx >= x0 - STROKE_RING) & (xx < x1 + STROKE_RING) & \
           (yy >= y0 - STROKE_RING) & (yy < y1 + STROKE_RING)
    ring = near & ~inside & (stroke > 0)
    if not ring.any():
        return 0.0
    return float(np.median(dist[ring]) * 2.0)


PROFILE = Profile(
    name="regular_cloud",
    candidates=_candidates,
    signals=[
        Signal("frame_align", _frame_alignment, lambda v: v < ALIGN_MAX),
        Signal("stroke_thickness", _stroke_thickness, lambda v: v < STROKE_THICK_MAX),
    ],
)
