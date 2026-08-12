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
WALL_MAX = 0.30        # B2: pocket wall fraction on lines/canvas-edge at or above
                       # this = sealed gutter pocket -> delete (threshold set from
                       # the measured per-pocket separation, see decisions.md)
RESCUE_SEED_FRAC = 0.03  # C1: pass-2 rescue admits an ink2 component only if the
                         # detection seed is >= this fraction of it (blocks large
                         # barely-touching neighbors; a colored glyph's dark core
                         # is ~0.09 of its full body on 005)
BLANK_G = 200          # 8.10.1: near-blank gutter tone (G channel)
BLANK_MIN = 0.60       # 8.10.1: outside-band region must be at least this blank
                       # for the band to count as proven panel/gutter structure
                       # (refs 0.94-0.96; damage classes 0.03-0.28)
BORDER_THICK = 20      # border-quality line: stroke thickness cap (corpus median
                       # ~12px; 004's left border measures 17)
BORDER_SPAN_FRAC = 0.4  # ... and span at least this fraction of the page dimension
EDGE_MARGIN = 3        # lines within this of the canvas edge are crop artifacts

# 8.11.2 instrumentation: counts of safety-guard no-ops (defense-in-depth layer;
# on the panel-aware clean_chapter path these are expected to stay near zero --
# every firing there indicates either a genuine edge case or a segmentation miss
# and gets diagnosed). Reset externally; incremented at the guard return sites.
GUARD_STATS = {"zero_line": 0, "blank_evidence": 0, "inversion": 0}
DENSE_INK = 0.15   # 8.11.2: borderless segment ink density at or above this =
                   # full-bleed art island, kept wholesale in clean_chapter
CONTENT_DENSE = 0.20  # fix-3a (case C): borderless bands are ALSO kept when
                      # their CONTENT fraction (G < BLANK_G) reaches this --
                      # light-skin / white-clothing art is midtone, not ink
                      # (the y65368 silhouette band measured ink 0.029), and
                      # the ink-only rule fed it to gutter treatment. Measured
                      # on the gold001/gold002 manual cleans: 0.25 recovers
                      # 1.52M wrongly-deleted content px at an FN cost of 62k
                      # (one 92-row grayscale pale-texture band, gold002
                      # y82920); the A2 step to 0.20 recovers another 133k at
                      # +4.5k FN (30:1) and catches the y65539 silhouette
                      # band (cfrac 0.234). Below 0.20 the ratio collapses to
                      # ~3:1 -- under-keep-preferred stops there.
RC_KEEP_INK_MIN = 0.01  # 8.12.4: a regular_cloud keep must have at least this
                        # INTERIOR ink fraction (G < 100) or it protects an
                        # empty hole (whited-out translation caption boxes,
                        # a280337 diagnosis) and is skipped. Interior = region
                        # inset past its own outline: the flagship empty box
                        # measured 3.3% ink REGION-level purely from its 2px
                        # border, so the region-level metric cannot separate;
                        # interiors measure ~0 (empty) vs >= 5.9% (real text
                        # bubbles) on the 80 measured chapter regions.
RC_INSET_FRAC = 0.12    # inset = max(6 px, this fraction of the smaller dim)
TEXT_SEED_FRAC = 0.15   # fix-1 (case A): a content component (G < BLANK_G) is
                        # rescued from deletion when at least this fraction of
                        # it is already kept (its dark glyph core survives
                        # pass-1) -- the kept core proves it is a text/stroke
                        # structure whose anti-aliased skirt pass-1 eroded
                        # (measured caption erosion 13-17% ink / up to 65%
                        # midtone; with the rescue: 0 on all gold002 captions).
TEXT_COMP_MAX = 2000    # ... and the component is text-scale. Large pale art
                        # components must NOT ride in (they are fix-3a's
                        # domain); measured sweep: without the cap the rescue
                        # adds 147k px of wrongly-kept SFX skirt (gold002),
                        # with it 8.7k px of sub-70px specks.
ISO_RESCUE = 0.30       # sparse-gap fix (H4): a deleted-ink fragment whose
                        # 60px-neighborhood ink density reaches this is part
                        # of a dense kept structure (title art, dark panels,
                        # border clusters) -- rescued. Measured px-weighted on
                        # the gold cleans: GT-deleted (SFX) fragments sit at
                        # iso p90=0.16; above 0.30 the rescue recovers 17,081
                        # px (gold001) + 2,566 (gold002) at FN cost 777+1 px
                        # (16 comps, max 186 px -- speck-scale). Geometry
                        # (area/elong), kept-fraction, and kept-context
                        # density all measured NON-separable; this is the
                        # only clean one-sided signal found (3 families
                        # exhausted first -- see decisions.md).
ISO_BLUR = 121          # box size realizing the 60px neighborhood reach.
SITE_G_TOL = 55         # fix-2: gutter-context site interior band, same
                        # min-channel floor as the production action (wand
                        # tolerance 200 from white; pipeline.G_TOL).
SITE_CLOUD_CLOSE = 25   # fix-2: closing kernel bridging the radial ticks of
                        # a spiky cloud into one silhouette (tick gaps
                        # measure up to ~15 px on the gold002 sites).
SITE_CLOUD_MARGIN = 6   # fix-2: silhouette margin (anti-aliased skirt).
SITE_PANEL_COVER = 0.5  # fix-2: a site bbox at least this covered by
                        # panel/partial rects runs the PANEL-context
                        # production action (12-instance-validated),
                        # otherwise the gutter-context cloud-silhouette
                        # action.


def _border_lines(rgb: np.ndarray):
    """Returns (hb, vb, h_all, v_all): border-quality lines plus the full bridged
    inventory per axis (the latter feeds the 1-line extrapolation)."""
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
    return hb, vb, h_lines, v_lines


def _axis_band(border, all_lines, dim):
    """Band extent for one axis (8.9.1 A1, delete-bias priority). Two border-quality
    lines bound the band exactly. ONE border line: the opposite border is merged
    into dark art and invisible to the inventory as a line -- but the art-mass
    inventory entry's FAR EDGE is where the mass (and with it the border) ends, so
    extrapolate the band from the border line to the farthest far-edge among the
    other inventory entries on the panel side (verified on 004(4): predicts the
    hand-annotated border row exactly). Sides: the panel is on the side of the
    border line holding the majority of other entries. No other entries at all ->
    full extent (the hard frame-loss guard outranks the delete bias where zero
    evidence exists). Zero border lines: full extent (nothing to anchor from)."""
    if len(border) >= 2:
        return (min(ln.pos for ln in border),
                max(ln.pos + max(1, ln.thick) for ln in border))
    if len(border) == 1:
        b = border[0]
        others = [ln for ln in all_lines if ln is not b]
        below = [ln for ln in others if ln.pos > b.pos]
        above = [ln for ln in others if ln.pos < b.pos]
        if len(below) >= len(above) and below:
            return (b.pos, max(ln.pos + ln.thick // 2 + 1 for ln in below))
        if above:
            return (min(ln.pos - ln.thick // 2 - 1 for ln in above),
                    b.pos + max(1, b.thick))
    return (0, dim)


def frame_keep_mask(rgb: np.ndarray, lines=None) -> np.ndarray:
    """Conservative whole-frame keep: bounding band of border-quality lines.
    `lines` may carry a precomputed `_border_lines(rgb)` result."""
    H, W = rgb.shape[:2]
    hb, vb, h_all, v_all = lines if lines is not None else _border_lines(rgb)
    keep = np.zeros((H, W), bool)
    if not hb and not vb:
        return keep
    y0, y1 = _axis_band(hb, h_all, H)
    x0, x1 = _axis_band(vb, v_all, W)
    keep[y0:y1, x0:x1] = True
    return keep


def clean_sfx_region(rgb: np.ndarray, bubble_mode: str = "all",
                     keep_mask: np.ndarray | None = None) -> np.ndarray:
    """Returns the delete mask (True = delete) for one page/crop.

    bubble_mode: 'all' = every pocket >= POCKET_MIN kept (8.8.1 behavior);
    'none' = pockets default to DELETE (the B1 flipped-default measurement);
    'wall' = keep only pockets whose enclosure wall is free-standing ink, not
    inventory lines / canvas edge (B2 wall-material test).

    keep_mask (8.11.2): a precomputed frame-keep (e.g. panel_segmentation
    extents from clean_chapter). When given, the window-local band derivation
    AND its three reactive guards are bypassed -- the caller's segmentation
    knowledge supersedes re-derivation (measured unit-level failures without
    this: A1 unit bands shorter than the true content extent, x-band collapse
    from single v-lines, pseudo-partial furniture lines). None = standalone
    behavior, unchanged."""
    H, W = rgb.shape[:2]
    if keep_mask is not None:
        white1 = rgb[..., 1] >= CUT_AGGR
        keep = keep_mask
        return _delete_with_keeps(rgb, white1, keep, bubble_mode)
    # ZERO-BORDER-LINE SAFETY GUARD (8.10.1): a window with no border-quality line
    # on EITHER axis has zero protective context -- pass-1 would classify colored
    # full-bleed art as deletable background (measured: ~32% of a pure-art window
    # deleted, 002 y37100). Such a window is a NO-OP: degrade to nothing, not to
    # damage, same principle as the chapter-scale band. The threshold is EXACTLY
    # zero lines total: a single line still anchors the A1 far-edge extrapolation
    # (004(4)) and must not be caught here. Guard ordering: hard frame-loss guard
    # > this zero-line no-op guard > delete-bias priority > ambiguous-keep -- the
    # delete bias applies only where SOME frame protection exists.
    lines = _border_lines(rgb)
    hb, vb = lines[0], lines[1]
    if len(hb) + len(vb) == 0:
        GUARD_STATS["zero_line"] += 1
        return np.zeros((H, W), bool)
    white1 = rgb[..., 1] >= CUT_AGGR
    keep = frame_keep_mask(rgb, lines=lines)
    # BLANK-GUTTER EVIDENCE GUARD (8.10.1 attempt 2): border-quality lines alone do
    # not prove a panel/gutter structure -- on full-bleed art, caption-box borders
    # and building edges qualify as lines, the band anchors to them, and everything
    # outside (pure colored art) is deleted (measured: 31.7% of 002 y37100, whose
    # window has FOUR qualifying lines). Corroborate the band with what it claims:
    # the outside-band region must actually look like blank gutter. Measured
    # separation: refs 0.94-0.96 near-blank outside their bands vs 0.03-0.28 on all
    # three damage classes (spurious band, cut panels, dark-scene windows). Below
    # the bar -> no protective context is proven -> NO-OP, same degrade-to-nothing
    # principle and same guard ordering as the zero-line guard above.
    outside = ~keep
    if outside.any():
        g = rgb[..., 1] if rgb.ndim == 3 else rgb
        out_blank = float((g[outside] >= BLANK_G).mean())
        if out_blank < BLANK_MIN:
            GUARD_STATS["blank_evidence"] += 1
            return np.zeros((H, W), bool)
        # BAND-INVERSION GUARD (8.10.1 attempt 3): when a window shows two panels
        # cut at its edges, the only in-window border lines are the panels' facing
        # borders and the band captures the GUTTER between them -- keeping blank
        # gutter and deleting the cut panels' art (measured: 002 y51300 and 3
        # siblings, inside_blank 0.91-0.99 vs outside 0.61-0.66). A band that is
        # BLANKER inside than outside is such an inversion: refs measure inside
        # 0.37-0.55 vs outside 0.94-0.96, never inverted. Inverted -> NO-OP.
        if keep.any():
            in_blank = float((g[keep] >= BLANK_G).mean())
            if in_blank > out_blank:
                GUARD_STATS["inversion"] += 1
                return np.zeros((H, W), bool)

    return _delete_with_keeps(rgb, white1, keep, bubble_mode)


def _delete_with_keeps(rgb, white1, keep, bubble_mode):
    """SFX + bubble keeps and the final delete formula, given a frame keep."""
    H, W = rgb.shape[:2]
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
        # C1 (8.9.2): admit a touched ink2 component only if the seed makes up at
        # least RESCUE_SEED_FRAC of it -- unconditional connectivity grabbed large
        # ADJACENT structures barely touching the seed (frame lines, neighboring
        # art), the measured dominant over-keep source; a colored glyph's own body
        # still passes (its dark core is a substantial fraction of the whole).
        seed_ids, seed_counts = np.unique(lab2[seed], return_counts=True)
        all_counts = np.bincount(lab2.ravel(), minlength=n2)
        admit = [i for i, c in zip(seed_ids, seed_counts)
                 if i > 0 and c / max(1, all_counts[i]) >= RESCUE_SEED_FRAC]
        obj = np.isin(lab2, admit)
        obj |= seed  # the detection's own ink is always kept
        dc = cv2.distanceTransform(obj.astype(np.uint8), cv2.DIST_L2, 3)
        wvals = dc[dc > 0]
        w_med = float(np.median(wvals) * 2.0) if wvals.size else 0.0
        e = E_THIN if w_med < E_W_SPLIT else E_THICK
        k = 2 * e + 1
        grown = cv2.dilate(obj.astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        sfx_keep[wy0:wy1, wx0:wx1] |= grown
    # --- bubble keeps: enclosed pockets + wall + halo via the same Expand ---
    bubble_keep = np.zeros((H, W), bool)
    if bubble_mode != "none":
        pockets = enclosed(white1)
        num, lab, stats, _ = cv2.connectedComponentsWithStats(
            pockets.astype(np.uint8), connectivity=8)
        kb = 2 * E_BUBBLE + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))
        barrier = None
        if bubble_mode == "wall":
            from classifiers.profiles.sfx_glyph import _ctx
            barrier = _ctx(rgb)["barrier"]
        for i in range(1, num):
            if int(stats[i, cv2.CC_STAT_AREA]) < POCKET_MIN:
                continue
            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]),
                          int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]),
                          int(stats[i, cv2.CC_STAT_HEIGHT]))
            pad = E_BUBBLE + 2
            wy0, wy1 = max(0, y - pad), min(H, y + h + pad)
            wx0, wx1 = max(0, x - pad), min(W, x + w + pad)
            pk = lab[wy0:wy1, wx0:wx1] == i
            if bubble_mode == "wall" and _pocket_wall_frac(
                    pk, barrier[wy0:wy1, wx0:wx1],
                    (wy0, wx0, H, W)) >= WALL_MAX:
                continue  # walled by lines/canvas edge = sealed gutter -> delete
            grown = cv2.dilate(pk.astype(np.uint8), ker) > 0
            bubble_keep[wy0:wy1, wx0:wx1] |= grown

    return white1 & ~keep & ~sfx_keep & ~bubble_keep


def _pocket_wall_frac(pocket, barrier_win, geom) -> float:
    """Fraction of a pocket's immediate wall ring lying on inventory-line barrier
    px or the canvas edge -- the B2 wall-material evidence: a real bubble is walled
    by free-standing drawn ink; a sealed GUTTER pocket is walled substantially by
    frame lines and/or the canvas edge."""
    wy0, wx0, H, W = geom
    ring = (cv2.dilate(pocket.astype(np.uint8), _K3) > 0) & ~pocket
    if not ring.any():
        return 0.0
    hw, ww = pocket.shape
    yy, xx = np.nonzero(ring)
    on_edge = ((yy + wy0 <= 1) | (yy + wy0 >= H - 2)
               | (xx + wx0 <= 1) | (xx + wx0 >= W - 2))
    on_line = barrier_win[yy, xx]
    return float((on_edge | on_line).mean())


_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def clean_chapter(rgb: np.ndarray, verbose: bool = False):
    """Panel-aware chapter driver (8.11.2): processes a whole chapter strip by
    REAL panel boundaries instead of fixed-height windows -- the architectural
    fix for the cut-panel damage classes the 8.10.1 guards reacted to.

    Units come from `panel_segmentation.units_for_processing` (gutter-midpoint
    cuts). The frame keep is DRIVEN BY SEGMENTATION, not re-derived per unit
    (attempt B1 -- measured unit-level re-derivation failures: A1 unit bands
    shorter than the true content extent, x-band collapse from single v-lines,
    pseudo-partial furniture lines): keep = all panel/partial rects, plus
    borderless segments whose ink density >= DENSE_INK (full-bleed art islands,
    kept wholesale -- the keep-all answer for content with no panel structure).
    Sparse borderless segments (floating gutter glyphs/bubbles) and gutters get
    gutter treatment: pass-1 delete minus the SFX/bubble keeps. Units whose rows
    are entirely kept are skipped wholesale. The 8.10.1 reactive guards are
    bypassed on this path (superseded by segmentation knowledge) but remain for
    standalone clean_sfx_region use.

    Returns (delete_mask, stats)."""
    from classifiers.panel_segmentation import (segment_chapter,
                                                units_for_processing)
    H, W = rgb.shape[:2]
    segs = segment_chapter(rgb)
    units = units_for_processing(segs, H)
    g = rgb[..., 1]
    keep_all = np.zeros((H, W), bool)
    for s in segs:
        if s.kind in ("panel", "partial"):
            keep_all[s.y0:s.y1, s.x0:s.x1] = True
        elif s.kind == "borderless":
            ink = float((g[s.y0:s.y1] < 100).mean())
            content = float((g[s.y0:s.y1] < BLANK_G).mean())
            if ink >= DENSE_INK or content >= CONTENT_DENSE:
                keep_all[s.y0:s.y1] = True
    delete = np.zeros((H, W), bool)
    stats = {"units": len(units), "processed": 0, "skipped_kept": 0}
    for y0, y1, kinds in units:
        if keep_all[y0:y1].all():
            stats["skipped_kept"] += 1
            continue
        delete[y0:y1] = clean_sfx_region(rgb[y0:y1], keep_mask=keep_all[y0:y1])
        stats["processed"] += 1
    if verbose:
        print(f"clean_chapter: {stats}")
    return delete, stats


def _clean_spiky_site_gutter(rgb: np.ndarray, delete: np.ndarray,
                             bbox: tuple, prot: np.ndarray,
                             panel_keep: np.ndarray) -> np.ndarray:
    """Fix-2 (case B): GUTTER-context site action. The production action
    (validated in PANEL context by the 12-instance suite) wholesale-keeps
    everything but the detected fringe inside its bbox and clears pass-1
    deletions there -- measured at 002 y71625: it resurrected 82,670 gutter
    blank px, manufacturing a kept white rectangle with a torn contour. The
    manual reference at gutter sites is the opposite shape: a smooth balloon
    -- everything in the site zone is deleted EXCEPT the CLOUD SILHOUETTE:
    the sealed interior plus the connected content structure around it
    (outline ring, radial spikes, inter-spike white), morphologically closed
    into one organic shape. Attempt 2-A1 (keep only interior+ring, the May-
    etalon smooth-balloon target) was a COUNTED FAILURE: the full-chapter
    cleans supersede the May spot etalons and KEEP the spiky fringe at all
    10 gold002 + 3 gold001 sites (src ink changed <= 4%), black-filling the
    background up to the silhouette; deleting spikes measured FPink 1.9k ->
    12.1k. 2-A2 keeps the silhouette; protected interiors and panel/partial
    content the bbox dips into are kept as well."""
    x0, y0, x1, y1 = bbox
    H, W = delete.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    sub = rgb[y0:y1, x0:x1]
    band = sub.min(axis=2) >= SITE_G_TOL
    k3 = np.ones((3, 3), np.uint8)
    barrier = cv2.morphologyEx((~band).astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    interior = enclosed(band & ~barrier)
    seed = (interior | ~band).astype(np.uint8)
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (SITE_CLOUD_CLOSE, SITE_CLOUD_CLOSE))
    cloud = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kc) > 0
    # only silhouette components anchored on the interior count as the cloud
    # (stray dark content elsewhere in the bbox must not ride in)
    num, lab = cv2.connectedComponents(cloud.astype(np.uint8), connectivity=8)
    anchored = np.unique(lab[interior])
    cloud = np.isin(lab, anchored[anchored != 0])
    km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * SITE_CLOUD_MARGIN + 1,
                                    2 * SITE_CLOUD_MARGIN + 1))
    cloud = cv2.dilate(cloud.astype(np.uint8), km) > 0
    keep = cloud | prot[y0:y1, x0:x1] | panel_keep[y0:y1, x0:x1]
    out = delete.copy()
    out[y0:y1, x0:x1] = ~keep
    return out


def _text_skirt_rescue(rgb: np.ndarray, delete: np.ndarray,
                       sites: list) -> np.ndarray:
    """Fix-1 (case A, composition policy): un-delete text-scale content
    components whose dark core already survives -- pass-1 erodes the
    anti-aliased skirt and the G 33-99 strokes of translated caption glyphs
    (measured 13-17% ink / up to 65% midtone per caption), leaving broken
    text. A component G < BLANK_G with >= TEXT_SEED_FRAC of itself kept and
    area <= TEXT_COMP_MAX is such a glyph; spiky site bboxes are excluded
    (the site action is the sole authority there). Returns px to un-delete."""
    content = rgb[..., 1] < BLANK_G
    kept = content & ~delete
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        content.astype(np.uint8), connectivity=8)
    insite = np.zeros(delete.shape, bool)
    for (sx0, sy0, sx1, sy1) in sites:
        insite[sy0:sy1, sx0:sx1] = True
    areas = stats[:, cv2.CC_STAT_AREA]
    kept_counts = np.bincount(lab[kept], minlength=num)
    del_counts = np.bincount(lab[content & delete], minlength=num)
    site_counts = np.bincount(lab[insite & content], minlength=num)
    admit = np.zeros(num, bool)
    for i in range(1, num):
        if areas[i] > TEXT_COMP_MAX or del_counts[i] == 0:
            continue
        if site_counts[i] > 0.2 * areas[i]:
            continue
        if kept_counts[i] / areas[i] >= TEXT_SEED_FRAC:
            admit[i] = True
    return admit[lab] & delete


def _ink_context_rescue(rgb: np.ndarray, delete: np.ndarray,
                        sites: list) -> np.ndarray:
    """Sparse-gap fix (H4): un-delete ink fragments embedded in ink-dense
    neighborhoods. The residual sparse-band damage class is mid-gray strokes
    (G 33-99) of KEPT structures (title art, border clusters, art fragments)
    that pass-1 catches; original-SFX strokes -- which must STAY deleted --
    live in sparse fields (measured iso p90=0.16 vs kept-structure fragments
    reaching 0.72+). Fragments = components of the deleted-ink px themselves
    (structure-level components are useless here: content merges through
    gutter pink and ink merges through borders into chapter-spanning
    mega-components, both measured). Site bboxes excluded. Returns px to
    un-delete."""
    ink = rgb[..., 1] < 100
    insite = np.zeros(delete.shape, bool)
    for (sx0, sy0, sx1, sy1) in sites:
        insite[sy0:sy1, sx0:sx1] = True
    cand = delete & ink & ~insite
    if not cand.any():
        return np.zeros(delete.shape, bool)
    dens = cv2.blur(ink.astype(np.float32), (ISO_BLUR, ISO_BLUR))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        cand.astype(np.uint8), connectivity=8)
    out = np.zeros(delete.shape, bool)
    for i in range(1, num):
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        m = lab[y:y + h, x:x + w] == i
        if float(dens[y:y + h, x:x + w][m].mean()) >= ISO_RESCUE:
            out[y:y + h, x:x + w] |= m
    return out


def clean_chapter_full(rgb: np.ndarray, verbose: bool = False):
    """Gen-8 FULL ORCHESTRATION (8.12.1): every gen-8 classifier composed into
    one chapter-scale delete decision. Reference architecture -- see
    docs/gen8_architecture.md for the full description. Order and authority:

      1. clean_chapter(rgb)          panel-aware sfx path: segmentation-driven
                                     keeps (panels/partials/dense-borderless),
                                     gutter treatment with sfx_glyph stroke
                                     keeps + pocket bubble keeps.
      2. regular_cloud keeps         per processing unit: profile-accepted
                                     bubble-family regions (+E_BUBBLE halo)
                                     subtracted from the delete -- classified
                                     bubble protection on top of the pocket
                                     rule. Regions overlapping a spiky site are
                                     EXCLUDED (conflict rule: the separately
                                     validated spiky deletion outranks a cloud
                                     keep -- clouds classify as 'thorn' family,
                                     so overlaps are expected, counted, and
                                     resolved, not ignored).
      3. spiky_cloud site deletions  applied LAST: the equivalence-proven
                                     profile detects the v23 sites; each gets
                                     the production-validated site action
                                     `pipeline.clean_spiky_region_clipped` with
                                     the background classifier's protected
                                     interiors. The ONLY in-panel delete
                                     authority in the composition (carried by
                                     the 12-instance suite validation);
                                     overrides any keep.

    Returns (delete_mask, stats). Does NOT alter clean_chapter / production
    defaults; new additive API."""
    sys.path.insert(0, str(HERE.parents[1] / "src" / "spiky"))
    import pipeline as spiky_pipeline
    from classifiers.background import protected_interiors
    from classifiers.panel_segmentation import (segment_chapter,
                                                units_for_processing)
    from classifiers.profiles import regular_cloud as rc
    from classifiers.profiles import spiky_cloud as sc

    H, W = rgb.shape[:2]
    delete, stats = clean_chapter(rgb)
    sites = detect(rgb, sc.PROFILE)
    stats["spiky_sites"] = len(sites)

    # regular_cloud keeps, per unit (the profile's candidate generator is
    # page-relative; units are the page-scale scope this generation processes at)
    segs = segment_chapter(rgb)
    units = units_for_processing(segs, H)
    rc_regions = 0
    rc_conflicts = 0
    rc_empty_skipped = 0
    rc_keep = np.zeros((H, W), bool)
    for y0, y1, kinds in units:
        for (rx0, ry0, rx1, ry1) in detect(rgb[y0:y1], rc.PROFILE):
            gy0, gy1 = ry0 + y0, ry1 + y0
            overlaps = any(rx0 < sx1 and rx1 > sx0 and gy0 < sy1 and gy1 > sy0
                           for sx0, sy0, sx1, sy1 in sites)
            if overlaps:
                rc_conflicts += 1
                continue
            # 8.12.4 empty-hole filter (composition policy, not detection --
            # the profile stays untouched): keep only regions whose INTERIOR
            # carries ink; see RC_KEEP_INK_MIN.
            inset = max(6, int(RC_INSET_FRAC * min(rx1 - rx0, gy1 - gy0)))
            iy0, iy1 = gy0 + inset, gy1 - inset
            ix0, ix1 = rx0 + inset, rx1 - inset
            if iy1 > iy0 and ix1 > ix0:
                interior_ink = float((rgb[iy0:iy1, ix0:ix1, 1] < 100).mean())
            else:
                interior_ink = 1.0  # region too small to inset: not a box, keep
            if interior_ink < RC_KEEP_INK_MIN:
                rc_empty_skipped += 1
                continue
            rc_regions += 1
            by0 = max(0, gy0 - E_BUBBLE); by1 = min(H, gy1 + E_BUBBLE)
            bx0 = max(0, rx0 - E_BUBBLE); bx1 = min(W, rx1 + E_BUBBLE)
            rc_keep[by0:by1, bx0:bx1] = True
    stats["rc_keep_regions"] = rc_regions
    stats["rc_spiky_conflicts"] = rc_conflicts
    stats["rc_empty_skipped"] = rc_empty_skipped
    kept_from_delete = int((delete & rc_keep).sum())
    stats["rc_kept_px"] = kept_from_delete
    delete &= ~rc_keep

    # spiky site deletions last -- override every keep inside their sites.
    # fix-2 (case B): the action is CONTEXT-DISPATCHED -- panel-context sites
    # get the production action unchanged (12-instance-validated semantics);
    # gutter-context sites get the smooth-balloon action, because the
    # production action's wholesale-keep resurrects gutter blank inside its
    # bbox (measured 82,670 px at 002 y71625 -> torn white rectangles).
    f = rgb.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    prot = protected_interiors(gray)
    panel_keep = np.zeros((H, W), bool)
    for s in segs:
        if s.kind in ("panel", "partial"):
            panel_keep[s.y0:s.y1, s.x0:s.x1] = True
    before = delete.copy()
    gutter_sites = 0
    for bbox in sites:
        bx0, by0, bx1, by1 = bbox
        cover = float(panel_keep[max(0, by0):by1, max(0, bx0):bx1].mean()) \
            if by1 > by0 and bx1 > bx0 else 0.0
        if cover >= SITE_PANEL_COVER:
            delete = spiky_pipeline.clean_spiky_region_clipped(
                rgb, delete, bbox, protected=prot)
        else:
            gutter_sites += 1
            delete = _clean_spiky_site_gutter(rgb, delete, bbox, prot,
                                              panel_keep)
    stats["gutter_sites"] = gutter_sites
    stats["spiky_deleted_px"] = int((delete & ~before).sum())

    # fix-1 (case A): text-skirt rescue, LAST -- runs on the final mask so the
    # site actions' decisions stand (their bboxes are excluded inside).
    rescued = _text_skirt_rescue(rgb, delete, sites)
    stats["text_rescued_px"] = int(rescued.sum())
    delete &= ~rescued

    # sparse-gap fix (H4): ink fragments in ink-dense neighborhoods.
    ink_rescued = _ink_context_rescue(rgb, delete, sites)
    stats["ink_ctx_rescued_px"] = int(ink_rescued.sum())
    delete &= ~ink_rescued
    if verbose:
        print(f"clean_chapter_full: {stats}")
    return delete, stats
