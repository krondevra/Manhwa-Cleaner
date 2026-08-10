"""Frame classifier: panel frame boundaries from GEOMETRY, independent of panel content.

Gen-8 phase 2 (2026-08-10). Motivation: the v27 class-B residual (instances
019_0/3/6, 1.9-3.6% over-deletion) is frame-line occlusion -- a cloud/art element
locally erases the ink line, so window-local run detection (`pipeline._bridged_runs`)
loses the barrier exactly where it matters. Local detection was exhausted there (3
failed attempts, honest negative); the answer class is GLOBAL geometry: detect each
line across the whole page, where its unoccluded majority is plainly visible, and let
that global evidence stand in at occluded spans.

Attempt ladder (docs/decisions.md gen-8 entries track results):
  A1: probabilistic Hough on the full-page dark mask, filtered to long
      near-horizontal/near-vertical segments, merged into maximal lines, grouped
      into (possibly partial) panel rectangles.
  A2 (if needed): extrapolate a panel's missing sides from its confirmed sides.
  A3 (if needed): fuse the threshold-assisted long-run signal
      (`pipeline._bridged_runs` mechanics) with the Hough geometry.

Full-page context ALWAYS (standing lesson: crop-local protection/geometry decisions
are the v25-diagnosed bug class). Decision-boundary bias: when line evidence is
ambiguous, prefer treating px as deletable background over protecting it -- a frame
line is only asserted where the geometric evidence threshold is met.

STATUS AFTER THE PHASE-2 ROUND (2026-08-10, see docs/decisions.md):
- VALIDATED: `detect_lines_morph` + `bridge_collinear` (the A3 page-scale line
  inventory -- deterministic, 0.4s on a 153k-row chapter, agrees with the window-local
  signal everywhere it was checked).
- NOT validated for standalone use: `classify_frames`' panel-rect grouping
  (over-generative -- 791 rects on chapter 019, interiors cover large background areas;
  measured covering up to 100% of correct deletions at the class-B sites). Use the line
  inventory, not the rects, until the grouping gets its own validation round.
- The class-B (019_0/3/6) closure goal itself ended as an HONEST NEGATIVE for both
  classical-geometry families (lines x3 attempts, rects by coverage diagnostic: the
  leaked px produce no line or rectangle evidence at any scale) -- that residual is
  flagged for the GUI/manual track, not for further ladder rounds here.

Constants deliberately reuse the validated classical thresholds: ink `<= FRAME_DARK`
(gray 100) and minimum run length `MIN_LINE` (100 px) are the same values every
frame-interacting mechanism since v10 has used (`_bridged_runs`, A' frame zone,
`clean_spiky_region_clipped`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

FRAME_DARK = 100   # gray <= this counts as frame ink (v10+ standard)
MIN_LINE = 100     # min segment length in px (v10+ standard long-run threshold)
ANGLE_TOL = 2.0    # degrees off horizontal/vertical still counted as h/v
MERGE_POS = 3      # px: same-line position tolerance when merging collinear segments
MERGE_GAP = 12     # px: max endpoint gap bridged when merging collinear segments (A1
                   # keeps this small -- genuine AA/join noise, NOT occlusion bridging;
                   # occlusion spans are hundreds of px and belong to A2)
CORNER_TOL = 14    # px: how close a horizontal and vertical end must be to join a corner


@dataclass
class Line:
    """A maximal merged axis-aligned line. orient 'h': pos=y, span=(x0,x1);
    orient 'v': pos=x, span=(y0,y1). thick = measured stroke thickness in px (pos is
    the stroke CENTER -- consumers matching edges must allow +-thick/2)."""
    orient: str
    pos: int
    span: tuple[int, int]
    thick: int = 1

    @property
    def length(self) -> int:
        return self.span[1] - self.span[0]


@dataclass
class Panel:
    """A detected panel. sides maps 'top'/'bottom'/'left'/'right' to a Line (confirmed)
    or None (undetected -- A2's extrapolation target). rect is set when all 4 sides
    (confirmed or extrapolated) exist."""
    sides: dict = field(default_factory=dict)
    rect: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1)
    extrapolated: list = field(default_factory=list)  # side names filled by A2


def _hough_segments(dark_u8: np.ndarray) -> list[tuple[int, int, int, int]]:
    segs = cv2.HoughLinesP(dark_u8, rho=1, theta=np.pi / 180.0, threshold=80,
                            minLineLength=MIN_LINE, maxLineGap=4)
    if segs is None:
        return []
    return [tuple(int(v) for v in s) for s in segs.reshape(-1, 4)]


def _merge_axis(segments: list[tuple[int, int, int, int]], orient: str) -> list[Line]:
    """Collapse near-collinear segments into maximal Lines. For 'h', a segment
    (x1,y1,x2,y2) contributes pos=(y1+y2)/2, span=(min(x),max(x)); merging joins spans
    whose pos differs <= MERGE_POS and whose gaps are <= MERGE_GAP."""
    items = []
    for x1, y1, x2, y2 in segments:
        if orient == "h":
            pos = (y1 + y2) / 2.0
            span = (min(x1, x2), max(x1, x2))
        else:
            pos = (x1 + x2) / 2.0
            span = (min(y1, y2), max(y1, y2))
        items.append((pos, span))
    items.sort()
    lines: list[Line] = []
    for pos, span in items:
        merged = False
        for ln in lines:
            if abs(ln.pos - pos) <= MERGE_POS:
                if span[0] <= ln.span[1] + MERGE_GAP and span[1] >= ln.span[0] - MERGE_GAP:
                    ln.span = (min(ln.span[0], span[0]), max(ln.span[1], span[1]))
                    ln.pos = int(round((ln.pos + pos) / 2.0))
                    merged = True
                    break
        if not merged:
            lines.append(Line(orient, int(round(pos)), span))
    return [ln for ln in lines if ln.length >= MIN_LINE]


def detect_lines(page_gray: np.ndarray) -> tuple[list[Line], list[Line]]:
    """Full-page maximal near-horizontal and near-vertical frame-ink lines."""
    dark = ((page_gray <= FRAME_DARK).astype(np.uint8)) * 255
    segs = _hough_segments(dark)
    h_raw, v_raw = [], []
    for x1, y1, x2, y2 in segs:
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if ang <= ANGLE_TOL:
            h_raw.append((x1, y1, x2, y2))
        elif ang >= 90.0 - ANGLE_TOL:
            v_raw.append((x1, y1, x2, y2))
    return _merge_axis(h_raw, "h"), _merge_axis(v_raw, "v")


def _group_panels(h: list[Line], v: list[Line]) -> list[Panel]:
    """A1 grouping: pair horizontals (top above bottom) with verticals whose endpoints
    meet the horizontals' ends within CORNER_TOL; emit full rectangles where all 4
    sides confirm, partial Panels otherwise."""
    panels: list[Panel] = []
    used_h: set[int] = set()
    h_sorted = sorted(range(len(h)), key=lambda i: h[i].pos)
    for ii, i in enumerate(h_sorted):
        if i in used_h:
            continue
        top = h[i]
        for j in h_sorted[ii + 1:]:
            if j in used_h:
                continue
            bot = h[j]
            # spans must substantially overlap to be the same panel's top/bottom
            ov = min(top.span[1], bot.span[1]) - max(top.span[0], bot.span[0])
            if ov < 0.6 * min(top.length, bot.length):
                continue
            left = right = None
            for ln in v:
                if not (ln.span[0] <= top.pos + CORNER_TOL and ln.span[1] >= bot.pos - CORNER_TOL):
                    continue
                if abs(ln.pos - max(top.span[0], bot.span[0])) <= CORNER_TOL:
                    left = ln
                elif abs(ln.pos - min(top.span[1], bot.span[1])) <= CORNER_TOL:
                    right = ln
            p = Panel(sides={"top": top, "bottom": bot, "left": left, "right": right})
            if left is not None and right is not None:
                p.rect = (left.pos, top.pos, right.pos, bot.pos)
            panels.append(p)
            used_h.add(i)
            used_h.add(j)
            break
    return panels


BRIDGE_GAP = 600   # A3: max occlusion span bridged between collinear run segments; both
                   # flanks must independently be >= MIN_LINE (the Fix B both-sides
                   # evidence rule, applied page-wide) -- delete-over-preserve bias:
                   # a line is asserted across a gap only on strong two-sided evidence.


def detect_lines_morph(page_gray: np.ndarray) -> tuple[list[Line], list[Line]]:
    """A3 line inventory: deterministic morphological long-run extraction, FULL PAGE.
    Same erode/dilate mechanics as the validated window-local signal
    (`pipeline._bridged_runs` / v10 clip / A' frame zone), promoted to page scale:
    a run only exists where >= MIN_LINE contiguous dark px lie on one row/column --
    immune to the accumulator dilution that sank the Hough attempt (A1) on dense
    full-page ink."""
    dark = (page_gray <= FRAME_DARK).astype(np.uint8)
    out: dict[str, list[Line]] = {}
    for orient, kshape in (("h", (MIN_LINE + 1, 1)), ("v", (1, MIN_LINE + 1))):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, kshape)
        runs = cv2.dilate(cv2.erode(dark, k), k)
        num, lab, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
        lines = []
        for i in range(1, num):
            x, y, w, h = (int(stats[i, j]) for j in range(4))
            if orient == "h":
                if w < MIN_LINE:
                    continue
                lines.append(Line("h", y + h // 2, (x, x + w), thick=h))
            else:
                if h < MIN_LINE:
                    continue
                lines.append(Line("v", x + w // 2, (y, y + h), thick=w))
        out[orient] = lines
    return out["h"], out["v"]


def bridge_collinear(lines: list[Line], max_gap: int = BRIDGE_GAP) -> list[Line]:
    """A3 geometric fusion: join collinear segments across occlusion-scale gaps.
    Requires both flanking segments >= MIN_LINE on their own (two-sided evidence)."""
    by_pos = sorted(lines, key=lambda l: (l.pos, l.span[0]))
    merged: list[Line] = []
    for ln in by_pos:
        joined = False
        for m in merged:
            if m.orient == ln.orient and abs(m.pos - ln.pos) <= MERGE_POS:
                gap = ln.span[0] - m.span[1]
                if gap <= max_gap and ln.span[1] > m.span[1]:
                    if m.length >= MIN_LINE and ln.length >= MIN_LINE:
                        m.span = (m.span[0], ln.span[1])
                        m.thick = max(m.thick, ln.thick)
                        joined = True
                        break
        if not joined:
            merged.append(Line(ln.orient, ln.pos, ln.span, ln.thick))
    return merged


def extrapolate_missing_sides(panels: list[Panel], page_shape: tuple[int, int]) -> list[Panel]:
    """A2: fill a panel's missing sides from its confirmed ones. A panel with confirmed
    top+bottom gets left/right extrapolated at the shared span extremes; a panel with
    3 sides gets the 4th mirrored from the opposite side's extent. Extrapolated sides
    are recorded in panel.extrapolated (never claimed as detected)."""
    H, W = page_shape
    for p in panels:
        top, bot = p.sides.get("top"), p.sides.get("bottom")
        if top is None or bot is None:
            continue  # A2 needs at least the two horizontals
        if p.sides.get("left") is None:
            x = max(top.span[0], bot.span[0])
            p.sides["left"] = Line("v", int(x), (top.pos, bot.pos))
            p.extrapolated.append("left")
        if p.sides.get("right") is None:
            x = min(top.span[1], bot.span[1])
            p.sides["right"] = Line("v", int(x), (top.pos, bot.pos))
            p.extrapolated.append("right")
        p.rect = (p.sides["left"].pos, top.pos, p.sides["right"].pos, bot.pos)
    return panels


def classify_frames(page: np.ndarray) -> list[Panel]:
    """Main entry. page: full-page RGB (HxWx3) or grayscale (HxW) uint8.
    Returns detected Panels (rect set when all sides confirmed; partial otherwise).
    Use `detect_lines` directly when only the line inventory is needed (e.g. the
    occlusion-barrier integration)."""
    if page.ndim == 3:
        f = page.astype(np.float32)
        gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    else:
        gray = page
    # A3 configuration (adopted after A1 Hough + A2 extrapolation-on-A1 both failed on
    # the occlusion sites -- see docs/decisions.md gen-8 phase 2 attempt log):
    # morphological page-wide inventory -> collinear occlusion bridging -> grouping ->
    # missing-side extrapolation.
    h, v = detect_lines_morph(gray)
    h = bridge_collinear(h)
    v = bridge_collinear(v)
    panels = _group_panels(h, v)
    return extrapolate_missing_sides(panels, gray.shape)
