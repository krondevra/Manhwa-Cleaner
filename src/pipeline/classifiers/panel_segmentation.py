"""Chapter-scale panel segmentation -- gen-8 8.11.1.

Removes the ROOT CAUSE behind the 8.10.1 reactive guards: fixed-height windowing
cuts panels mid-body, so every downstream consumer had to guess panel structure
from partial evidence. This module segments a WHOLE chapter strip (logical scope =
full chapter, the Class-A/B/C full-context lesson applied one level up) into typed
segments, so consumers process real panel boundaries.

Primary signal: ROW-BLANKNESS BAND DECOMPOSITION. row_blank = fraction of the
row's px with G >= BLANK_G; smoothed; rows above GUTTER_BLANK form gutter bands,
the rest form content bands. This generalizes sfx.py's A1 far-edge extrapolation:
a content band's far edge is where content actually ends, which IS the border
position when a border stroke is merged into dark art (004(4)-class) -- the case
falls out of the decomposition instead of needing special handling.

Line evidence (frame.py's validated inventory, computed ONCE per chapter) then
classifies each content band:
  panel      -- border-quality h-lines at BOTH band edges;
  partial    -- border line at exactly ONE edge (A1-class; the band edge supplies
                the opposite side);
  borderless -- no border evidence at either edge: full-bleed art or floating
                gutter content (SFX glyphs, bubbles). Explicitly labeled, never
                forced into a false panel -- the y37100/y51300 damage classes of
                the fixed-window era become IDENTIFIABLE here.

Chapter-scale border-quality criteria (NOT sfx.py's page-scale ones): h-lines
thin <= BORDER_THICK and span >= H_SPAN_FRAC of strip width; v-lines thin and
span >= V_SPAN_MIN ABSOLUTE px -- a v border spans its panel's height (~500-2000
rows), so any fraction-of-chapter-height criterion is invalid at this scale
(measured: chapter-scale inventories also contain degenerate thick=690/span=13k
art entries; the thickness cap excludes them, standing lesson).

`units_for_processing`: cuts the strip at GUTTER MIDPOINTS between segments, so
every unit contains complete panels plus their surrounding gutter halves -- the
same unit shape as the user's own manual-workflow crops. Panels sharing a border
line with no gutter between them are cut at the shared line.

Usage:  .venv/bin/python src/classifiers/panel_segmentation.py   (self-check on
the 6 decoded refs + chapters 002/004)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from classifiers.frame import bridge_collinear, detect_lines_morph  # noqa: E402

BLANK_G = 200        # near-blank gutter tone (G channel), same constant as sfx.py
GUTTER_BLANK = 0.90  # smoothed row-blankness at or above this = gutter row
SMOOTH_ROWS = 5      # running-mean window for the row profile
MIN_GUTTER = 80      # rows: shorter blank runs are IN-PANEL whitespace (pale
                     # interiors measure 45-75-row blank gaps; real inter-panel
                     # gutters in this corpus are 150+ rows) -- attempt-2 lesson
MIN_CONTENT = 24     # rows: shorter content runs are noise/specks, merged away
BORDER_THICK = 20    # border-quality stroke cap (shared lesson with sfx.py)
H_SPAN_FRAC = 0.4    # h border line must span this fraction of the strip WIDTH
V_SPAN_MIN = 276     # px: v border line minimum span (absolute -- panel-height
                     # scaled, NOT chapter-height scaled)
EDGE_TOL = 10        # px: line-to-band-edge distance counted as "at the edge"
ABSORB_MAX = 120     # px: a border line at most this far into the adjacent blank
                     # run still closes the band's panel (pale interiors measure
                     # 45-75 blank rows before the true border; real inter-panel
                     # gutters are 150+); requires blank continuing PAST the line
GUTTER_REACH = 300   # px: a panel/partial must have a gutter or strip end within
                     # this reach; otherwise its lines are art-interior edges
                     # (y37100-class) and it demotes to borderless
MIN_XSPAN = 240      # px: fix-3b -- a line-derived x-extent narrower than this
                     # is ONE-SIDED evidence (a border-decoration line cluster
                     # on a single side, e.g. the 004 y78096 diagonal panel's
                     # five right-edge lines at x642-661 -> 21px sliver whose
                     # band then got gutter treatment: 100,966 content px
                     # deleted invisibly to the adversarial guard). Panels in
                     # the gold corpus measure >= ~0.5 W wide; below MIN_XSPAN
                     # the extent falls back to full width (keep-side default,
                     # consistent with the <2-line rule).
X_COL_CONTENT = 0.10  # gen7-regression fix: a column is a CONTENT column of a
                      # band when at least this fraction of its rows is
                      # non-blank (G < BLANK_G).
X_OUT_MAX = 0.03      # gen7-regression fix: a line-derived extent is only
                      # trusted when at most this fraction of the band's
                      # content columns falls OUTSIDE it -- interior art
                      # edges qualify as v-border lines (002 y78891: lines
                      # x353/x651 pass MIN_XSPAN while art spans x0-689 ->
                      # 73k content px gutter-treated = the panel-erasure
                      # class; same mechanism cut captions at 002 y93758,
                      # 001 y74347/y79851). Measured on gold001+gold002:
                      # healthy extents have <= 1% content outside, every
                      # damaged band >= 5% -- bimodal at the bar.


@dataclass
class Segment:
    kind: str          # 'gutter' | 'panel' | 'partial' | 'borderless'
    y0: int
    y1: int            # exclusive
    x0: int = 0
    x1: int = 0        # exclusive; 0/0 for gutters


def _row_blankness(rgb: np.ndarray) -> np.ndarray:
    g = rgb[..., 1] if rgb.ndim == 3 else rgb
    rb = (g >= BLANK_G).mean(axis=1)
    # reflect-pad before smoothing: 'same' zero-padding fabricated 2-row content
    # bands at the strip ends (attempt-1 measured artifact)
    pad = SMOOTH_ROWS // 2
    rbp = np.pad(rb, pad, mode="reflect")
    k = np.ones(SMOOTH_ROWS) / SMOOTH_ROWS
    return np.convolve(rbp, k, mode="valid")


def _relabel_short(runs, target: bool, min_len: int):
    """Flip runs of `target` shorter than min_len (absorbed by their context),
    then coalesce equal neighbors. Handles leading/trailing runs too (the
    attempt-1 merge skipped the first run entirely)."""
    out = []
    for k, y0, y1 in runs:
        if k == target and y1 - y0 < min_len:
            k = not target
        if out and out[-1][0] == k:
            out[-1][2] = y1
        else:
            out.append([k, y0, y1])
    return out


def _bands(rb: np.ndarray):
    """Alternating (is_gutter, y0, y1) runs after two-tier length cleanup."""
    gutter = rb >= GUTTER_BLANK
    runs = []
    y = 0
    for k, length in _runs(gutter):
        runs.append([bool(k), y, y + length])
        y += length
    runs = _relabel_short(runs, target=False, min_len=MIN_CONTENT)
    runs = _relabel_short(runs, target=True, min_len=MIN_GUTTER)
    return [(bool(k), y0, y1) for k, y0, y1 in runs]


def _runs(mask: np.ndarray):
    """(value, run_length) pairs over a 1-D bool array."""
    if len(mask) == 0:
        return
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(mask)]])
    for s, e in zip(starts, ends):
        yield bool(mask[s]), int(e - s)


def chapter_lines(rgb: np.ndarray):
    """frame.py inventory once per chapter, filtered to border quality at
    CHAPTER scale (see module docstring)."""
    f = rgb.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    h_lines, v_lines = detect_lines_morph(gray)
    h_lines, v_lines = bridge_collinear(h_lines), bridge_collinear(v_lines)
    H, W = gray.shape
    hb = [ln for ln in h_lines
          if ln.thick <= BORDER_THICK
          and ln.span[1] - ln.span[0] >= H_SPAN_FRAC * W
          and 3 <= ln.pos <= H - 4]
    vb = [ln for ln in v_lines
          if ln.thick <= BORDER_THICK and ln.span[1] - ln.span[0] >= V_SPAN_MIN
          and 3 <= ln.pos <= W - 4]
    return hb, vb


def _x_extent(vb, y0: int, y1: int, W: int, g: np.ndarray | None = None):
    """Panel x-extent from v border lines overlapping [y0,y1) substantially.
    The extent must be wide enough (MIN_XSPAN, fix-3b) AND corroborated by
    the band's own content columns (X_OUT_MAX, gen7-regression fix): border
    lines BOUND a panel's art, so content outside the extent proves the
    lines are interior art edges, not the bounding borders -> full width
    (keep-side default). `g` is the chapter G channel; None skips the
    content check (compatibility)."""
    vv = [ln for ln in vb
          if min(ln.span[1], y1) - max(ln.span[0], y0) >= 0.6 * (y1 - y0)]
    if len(vv) >= 2:
        x0 = min(ln.pos for ln in vv)
        x1 = max(ln.pos + max(1, ln.thick) for ln in vv)
        if x1 - x0 >= MIN_XSPAN:  # fix-3b: two-sided evidence required
            if g is None:
                return x0, x1
            cols = (g[y0:y1] < BLANK_G).mean(axis=0) > X_COL_CONTENT
            n_all = int(cols.sum())
            outside = cols.copy()
            outside[x0:x1] = False
            if n_all == 0 or int(outside.sum()) / n_all <= X_OUT_MAX:
                return x0, x1
    return 0, W


def segment_chapter(rgb: np.ndarray) -> list[Segment]:
    """Ordered typed segments covering the strip top-to-bottom.

    Attempt-2 design (line-band RECONCILIATION): border lines are not required to
    sit at band edges -- floating gutter content (SFX glyphs above a border)
    merges into the panel's content band and pale interiors blur its edges, so
    lines instead PARTITION each content band from within: consecutive in-band
    lines bound sub-intervals classified by their own blankness (panel vs
    interior gutter); a single in-band line splits panel side (majority content,
    the A1 rule) from floating side; no lines = borderless."""
    H, W = rgb.shape[:2]
    rb = _row_blankness(rgb)
    g_chan = rgb[..., 1]
    hb, vb = chapter_lines(rgb)
    h_pos = sorted(ln.pos for ln in hb)
    # far edges of ALL inventory entries (any thickness): the A1 evidence -- an
    # art-merged border ends where its art-mass entry ends
    f_all = rgb.astype(np.float32)
    gray_all = np.round((f_all.max(axis=2) + f_all.min(axis=2)) / 2.0
                        ).astype(np.uint8)
    h_all, _ = detect_lines_morph(gray_all)
    far_edges = sorted(ln.pos + ln.thick // 2 + 1 for ln in h_all)
    top_edges = sorted(ln.pos - ln.thick // 2 - 1 for ln in h_all)
    segs: list[Segment] = []
    for is_gutter, y0, y1 in _bands(rb):
        if is_gutter:
            segs.append(Segment("gutter", y0, y1))
            continue
        # A4: absorb a border line sitting in the ADJACENT gutter past a pale
        # interior run (004-class: panel bottom is near-blank for 45-75 rows, the
        # thin border line's own rows get swallowed by the gutter run). Cap the
        # search so a neighboring panel's border (real gutters are 150+ rows
        # away) is never absorbed.
        lo, hi = y0 - EDGE_TOL, y1 + EDGE_TOL
        beyond = [p for p in h_pos if y1 < p <= y1 + ABSORB_MAX
                  and float(rb[p + 3:p + 23].mean()) >= GUTTER_BLANK]
        if beyond:  # blank continues past the line = it closes THIS panel
            hi = beyond[-1] + EDGE_TOL
            y1 = min(H, beyond[-1] + 3)
        before = [p for p in h_pos if y0 - ABSORB_MAX <= p < y0
                  and float(rb[max(0, p - 23):max(1, p - 3)].mean())
                  >= GUTTER_BLANK]
        if before:
            lo = before[0] - EDGE_TOL
            y0 = max(0, before[0])
        lines = [p for p in h_pos if lo <= p <= hi]
        # dedupe near-coincident lines (double-stroke borders)
        dl = []
        for p in lines:
            if not dl or p - dl[-1] > EDGE_TOL:
                dl.append(p)
        if not dl:
            segs.append(Segment("borderless", y0, y1, 0, W))
            continue
        # boundaries: band start, in-band lines, band end
        bounds = [y0] + [min(max(p, y0), y1) for p in dl] + [y1]
        n = len(bounds) - 1
        for i in range(n):
            a, b = bounds[i], bounds[i + 1]
            if b - a < MIN_CONTENT:
                continue
            interior_blank = float(rb[a + 2:b - 2].mean()) if b - a > 6 else 1.0
            line_top = i > 0            # bounded above by a detected line
            line_bot = i < n - 1        # bounded below by a detected line
            if interior_blank >= GUTTER_BLANK:
                segs.append(Segment("gutter", a, b))
                continue
            if line_top and line_bot:
                kind = "panel"
            elif line_top or line_bot:
                # one line side; the other side is a raw band edge. Panel if this
                # interval holds the band's content majority (A1 majority-side),
                # floating/borderless otherwise.
                spans = [bounds[j + 1] - bounds[j] for j in range(n)]
                kind = "partial" if (b - a) == max(spans) else "borderless"
            else:
                kind = "borderless"
            if kind == "partial":
                # A3: refine the OPEN side with the A1 inventory far-edge rule --
                # an art-merged border ends where its art-mass entry ends; rows
                # beyond that (floating gutter content, e.g. a glyph below the
                # panel) split off as borderless instead of inflating the panel.
                if line_top and not line_bot:
                    fe = [e for e in far_edges if a < e <= b]
                    if fe and b - max(fe) >= MIN_CONTENT:
                        cut = max(fe)
                        x0, x1 = _x_extent(vb, a, cut, W, g_chan)
                        segs.append(Segment("partial", a, cut, x0, x1))
                        segs.append(Segment("borderless", cut, b, 0, W))
                        continue
                elif line_bot and not line_top:
                    te = [e for e in top_edges if a <= e < b]
                    if te and min(te) - a >= MIN_CONTENT:
                        cut = min(te)
                        segs.append(Segment("borderless", a, cut, 0, W))
                        x0, x1 = _x_extent(vb, cut, b, W, g_chan)
                        segs.append(Segment("partial", cut, b, x0, x1))
                        continue
            x0, x1 = _x_extent(vb, a, b, W, g_chan)
            segs.append(Segment(kind, a, b, x0, x1))
    # coalesce any adjacent gutters produced by interval classification
    out: list[Segment] = []
    for s in segs:
        if out and s.kind == "gutter" and out[-1].kind == "gutter":
            out[-1].y1 = s.y1
        else:
            out.append(s)
    # A5 (blank-neighbor validation): a real border separates content from
    # GUTTER; lines deep inside a continuous content run (caption boxes,
    # building edges on full-bleed art -- the y37100 class) do not. A panel/
    # partial with no gutter or strip end within GUTTER_REACH of either extent
    # is demoted to borderless (floating zones may sit between a panel and its
    # gutter, so adjacency alone is too strict -- 004(4) measures 63 rows).
    gutters = [(s.y0, s.y1) for s in out if s.kind == "gutter"]
    for s in out:
        if s.kind not in ("panel", "partial"):
            continue
        near = (s.y0 <= GUTTER_REACH or s.y1 >= H - GUTTER_REACH
                or any(g1 >= s.y0 - GUTTER_REACH and g0 <= s.y1 + GUTTER_REACH
                       for g0, g1 in gutters))
        if not near:
            s.kind = "borderless"
    return out


def units_for_processing(segs: list[Segment], H: int):
    """Cut points at gutter midpoints -> (y0, y1, kinds) units, each holding the
    content segments between two cuts plus their surrounding gutter halves.
    Consecutive content segments with no gutter between them share one unit cut
    at nothing (they stay in the same unit -- a shared border line is interior
    evidence the consumer's band logic handles)."""
    cuts = [0]
    for s in segs:
        if s.kind == "gutter":
            cuts.append((s.y0 + s.y1) // 2)
    cuts.append(H)
    cuts = sorted(set(cuts))
    units = []
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        kinds = [s.kind for s in segs
                 if s.kind != "gutter" and s.y0 < c1 and s.y1 > c0]
        # pure-gutter slices (chapter head/tail) are included with empty kinds:
        # consumers give them plain gutter treatment (measured otherwise: the
        # strip's head/tail gutters stay uncleaned, a visible seam at the first
        # and last cut)
        units.append((c0, c1, kinds))
    return units


if __name__ == "__main__":
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    REPO = HERE.parents[1]
    sys.path.insert(0, str(HERE / "tests"))
    from sfx_suite import FRAME_RECTS, load_ref  # noqa: E402

    print("=== 6 refs vs hand-annotated FRAME_RECTS ===")
    for stem, rects in FRAME_RECTS.items():
        rgb = load_ref(stem)["raw"]
        segs = segment_chapter(rgb)
        content = [s for s in segs if s.kind != "gutter"]
        print(f"{stem}: {[(s.kind, s.y0, s.y1, s.x0, s.x1) for s in content]}"
              f"  annotated={rects}")

    for name, p in [("002", REPO / ".tmp/eval/002.png"),
                    ("004", REPO / ".tmp/eval/merged/004.png")]:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        import time
        t0 = time.time()
        segs = segment_chapter(rgb)
        units = units_for_processing(segs, rgb.shape[0])
        dt = time.time() - t0
        kinds = {}
        for s in segs:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        print(f"\n=== chapter {name}: {kinds} segments, {len(units)} units, "
              f"{dt:.1f}s ===")
        for tag, y in [("y37100 (full-bleed)", 37100), ("y51300", 51300)]:
            if name == "002":
                hit = [s for s in segs if s.y0 <= y < s.y1]
                print(f"  {tag}: {[(s.kind, s.y0, s.y1) for s in hit]}")
