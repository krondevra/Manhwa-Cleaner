"""sfx_glyph profile: isolated thin STROKE STRUCTURES (SFX glyphs and their kin) --
gen-8 sfx_glyph step 3, built from the 6 decoded reference PSDs (8.6.1) and the
measured feature table (sfx_suite.py), NOT from assumed shape rules.

SCOPE (settled by the attempt ladder): the profile detects isolated thin ink stroke
structures ANYWHERE on the page and leaves context to the composition (sfx.py):
in-panel detections are pixel-harmless (frame-keep keeps those regions wholesale),
in-bubble detections are pixel-harmless (bubble-keep is the cloud classifier's job).
Geometry cannot separate SFX from bubble text (both are glyphs) or sealed gutter
pockets from bubble interiors -- measured on the refs, see A2/A3 below.

Attempt ladder (one variable each, measured on 22 labeled ref SFX + 20 synth
frame-only pages; 'harmful extra' = detection whose ink is GT-DELETED):
  A1 baseline: gutter-flood + ring-enclosure + bbox line-collinearity candidate
     filters, 3 signals. Recall 16/22, harmful 0. Miss causes measured: gutter flood
     leaks/over-seals (4 misses), ring-enclosure cannot tell sealed gutter pockets
     from bubble interiors (2 misses at encl_ring=1.0).
  A2 drop gutter filter: recall 20/22. SUCCESS (context moved to composition, where
     over-admission is pixel-harmless).
  A3 drop enclosure filter: recall 22/22, all extras on GT-kept ink. SUCCESS.
  A4 line-structure exclusion by PIXEL coverage under drawn inventory lines
     (replaces bbox collinearity; a panel-rect outline's bbox is not thin but its px
     lie on lines): synth FPs 14 -> 0 BUT recall 22 -> 18 -- COUNTED FAILURE
     (regress-elsewhere): the inventory reports art masses as thick=100-300 "lines";
     drawn at measured thickness they swallow real SFX.
  A5 barrier thickness cap (entries > 30 px never drawn): recall back to 20/22,
     synth 5 (borders the inventory itself missed). SUCCESS on its variable. The 2
     permanent misses are border+steam MERGED components (>= 60% border px), kept via
     frame interior in composition -- pixel-harmless.
  A6 boundary-concentration signal (bconc): border bands / rect outlines live
     entirely on their own bbox boundary (measured 0.93-1.00) while glyph strokes
     cross the interior (<= 0.74 on refs). Synth FPs -> 0/20 pages, recall 20/22
     held, harmful 0. SUCCESS.
Final: recall 20/22 (both misses pixel-harmless), harmful extras 0, synth FP pages
0/20, chapters 002/019 ~1500-1750 detections in 5-8 s each; visual sample audit =
real SFX + bubble text + in-frame art strokes (the two harmless classes).

AND-voted signals (necessary conditions measured on the 22 labeled reference SFX;
precision walls against art structures, which are thick / blobby / dense-context):
  elong      area / w_med^2 (stroke length in width units). SFX observed 41-500;
             blobs are O(1). Threshold >= 25.
  w_p90      90th-pct stroke width. SFX observed <= 12.9 px; art masses are thick.
             Threshold <= 18.
  iso_ink    ink density of OTHER components within 2 stroke widths. SFX observed
             <= 6.4% (the recipe's magic-wand-ON isolation premise). Threshold <= 12%.
  bconc      comp px fraction on its own bbox boundary (A6). Threshold < 0.85.

Sample-size honesty: thresholds come from 22 positive examples in 6 crops, set with
margin from the observed extremes, precision-first; 6-ref numbers are smoke evidence,
not validated rates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from classifiers.background import enclosed  # noqa: E402
from classifiers.frame import bridge_collinear, detect_lines_morph  # noqa: E402
from classifiers.detector_framework import Profile, Region, Signal  # noqa: E402

CUT_AGGR = 33      # 8.6.1: aggressive-pass predicate G >= 33, constant across refs
AREA_MIN = 120     # px: smallest labelable SFX stroke component observed ~150
ELONG_MIN = 25.0   # SFX observed 41-500; blobs O(1)
W_P90_MAX = 18.0   # SFX observed <= 12.9 px
ISO_MAX = 0.12     # SFX observed <= 6.4% ring ink density
LINE_COVER = 0.6   # A4: component px fraction under drawn inventory lines = line
                   # structure (border, border fragment, panel-rect outline), excluded
LINE_THICK_MAX = 30  # A5: inventory entries thicker than this are art masses, not
                     # lines -- they never enter the barrier (frame strokes ~12px)
BCONC_MAX = 0.85   # A6: comp px fraction on its own bbox boundary; straight border
                   # bands / rect outlines 0.93-1.00, glyph strokes <= 0.74 measured


def _gray(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)


def _page_context(rgb: np.ndarray):
    """Lines, gutter mask, enclosed-light mask for one page (cached by id)."""
    gray = _gray(rgb)
    H, W = gray.shape
    h_lines, v_lines = detect_lines_morph(gray)
    h_lines, v_lines = bridge_collinear(h_lines), bridge_collinear(v_lines)
    # A5: only lines up to LINE_THICK_MAX enter the barrier -- the morph inventory
    # also reports dark ART MASSES as "lines" with thick 100-300px; drawing those at
    # measured thickness creates huge barrier patches that swallow real SFX
    # (measured: 2 recall misses on 004_2 under a thick=200 art patch). A real frame
    # border stroke is bounded (corpus median ~12px).
    barrier = np.zeros((H, W), np.uint8)
    for ln in h_lines:
        if ln.thick > LINE_THICK_MAX:
            continue
        t = max(1, ln.thick)
        y0 = max(0, ln.pos - t // 2 - 1); y1 = min(H, ln.pos + t // 2 + 2)
        barrier[y0:y1, ln.span[0]:ln.span[1] + 1] = 1
    for ln in v_lines:
        if ln.thick > LINE_THICK_MAX:
            continue
        t = max(1, ln.thick)
        x0 = max(0, ln.pos - t // 2 - 1); x1 = min(W, ln.pos + t // 2 + 2)
        barrier[ln.span[0]:ln.span[1] + 1, x0:x1] = 1
    free = (barrier == 0).astype(np.uint8)
    mask = np.zeros((H + 2, W + 2), np.uint8)
    for sx, sy in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1),
                   (W // 2, 0), (W // 2, H - 1), (0, H // 2), (W - 1, H // 2)]:
        if free[sy, sx] == 1:
            cv2.floodFill(free, mask, (sx, sy), 2)
    gutter = free == 2
    band = rgb[..., 1] >= CUT_AGGR if rgb.ndim == 3 else rgb >= CUT_AGGR
    encl = enclosed(band)
    return dict(h_lines=h_lines, v_lines=v_lines, gutter=gutter, encl=encl,
                barrier=barrier > 0)


# cache holds a REFERENCE to the page array: id() values are only unique among live
# objects, so an id-keyed cache without the reference can silently serve a previous
# page's context to a newly allocated array at the recycled id (measured failure mode
# during the 6-ref eval loop: 004_4 evaluated with another file's lines/gutter).
_ctx_cache: dict[int, tuple[np.ndarray, dict]] = {}
_feat_cache: dict[tuple[int, Region], dict] = {}


def _ctx(rgb: np.ndarray) -> dict:
    key = id(rgb)
    if key not in _ctx_cache:
        if len(_ctx_cache) > 4:
            _ctx_cache.clear()
            _feat_cache.clear()
        _ctx_cache[key] = (rgb, _page_context(rgb))
    return _ctx_cache[key][1]


def _candidates(rgb: np.ndarray) -> list[Region]:
    ctx = _ctx(rgb)
    ink = (rgb[..., 1] < CUT_AGGR) if rgb.ndim == 3 else (rgb < CUT_AGGR)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    H, W = ink.shape
    out: list[Region] = []
    if len(_feat_cache) > 8192:
        _feat_cache.clear()
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < AREA_MIN:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        # A2: no gutter candidate filter -- the flood-based mask proved fragile on the
        # refs (leaks through border runs broken by bubbles/steam, over-seals pockets;
        # measured cost: 4 of 6 recall misses). Context = frame-keep/bubble-keep is
        # applied by the COMPOSITION (sfx.py), where an over-admitted in-frame
        # detection is pixel-harmless; the gutter mask stays available in _ctx for
        # composition use.
        # A4: line-structure exclusion by PIXEL coverage under the drawn inventory
        # barrier (replaces bbox thin-collinearity, which missed thick synthetic
        # borders and whole panel-rect outlines -- 14 measured synth FPs, all
        # line-structures; a rect outline's bbox is not thin but its px lie on lines).
        comp = lab[y:y + h, x:x + w] == i
        bar_frac = float(ctx["barrier"][y:y + h, x:x + w][comp].mean())
        if bar_frac >= LINE_COVER:
            continue
        # features on a padded window
        pad = 30
        wy0, wy1 = max(0, y - pad), min(H, y + h + pad)
        wx0, wx1 = max(0, x - pad), min(W, x + w + pad)
        wcomp = lab[wy0:wy1, wx0:wx1] == i
        other_ink = ink[wy0:wy1, wx0:wx1] & ~wcomp
        dc = cv2.distanceTransform(wcomp.astype(np.uint8), cv2.DIST_L2, 3)
        dvals = dc[dc > 0]
        w_med = float(np.median(dvals) * 2.0) if dvals.size else 0.0
        w_p90 = float(np.quantile(dvals, 0.90) * 2.0) if dvals.size else 0.0
        elong = a / (w_med * w_med) if w_med > 0 else 0.0
        R = int(max(6, 2 * w_med))
        dout = cv2.distanceTransform((~wcomp).astype(np.uint8), cv2.DIST_L2, 3)
        ring = (dout > 0) & (dout <= R)
        iso = float(other_ink[ring].mean()) if ring.any() else 0.0
        # A6 boundary concentration: fraction of comp px within max(3, w_p90) of the
        # comp's own bbox edge. Border bands / panel-rect outlines live entirely on
        # their bbox boundary (measured 0.93-1.00); glyph strokes cross the interior
        # (measured <= 0.74 on the refs).
        ys, xs = np.nonzero(comp)
        m = max(3, int(w_p90))
        bconc = float(((xs < m) | (xs >= w - m) | (ys < m) | (ys >= h - m)).mean())
        # A3: no enclosure exclusion -- a ring-enclosed test cannot distinguish bubble
        # interiors from SEALED GUTTER POCKETS (canvas edge + frame line + bubble wall
        # close off gutter regions; measured: 2 real SFX at encl_ring=1.0, identical
        # to bubble text). Detections on bubble text/outlines are pixel-harmless in
        # the composition (GT keeps bubbles wholesale; bubble-keep is the cloud
        # classifier's job downstream). encl stays in _ctx for composition use.
        region = (x, y, x + w, y + h)
        _feat_cache[(id(rgb), region)] = dict(
            elong=elong, w_p90=w_p90, iso=iso, area=a, w_med=w_med, bconc=bconc)
        out.append(region)
    return out


def _feat(rgb: np.ndarray, region: Region, key: str) -> float:
    return _feat_cache[(id(rgb), region)][key]


PROFILE = Profile(
    name="sfx_glyph",
    candidates=_candidates,
    signals=[
        Signal("elong", lambda rgb, r: _feat(rgb, r, "elong"),
               lambda v: v >= ELONG_MIN),
        Signal("w_p90", lambda rgb, r: _feat(rgb, r, "w_p90"),
               lambda v: v <= W_P90_MAX),
        Signal("iso_ink", lambda rgb, r: _feat(rgb, r, "iso"),
               lambda v: v <= ISO_MAX),
        Signal("bconc", lambda rgb, r: _feat(rgb, r, "bconc"),
               lambda v: v < BCONC_MAX),
    ],
)
