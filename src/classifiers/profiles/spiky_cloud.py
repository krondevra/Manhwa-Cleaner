"""spiky_cloud profile: the v23 two-signal cascade, PORTED into the framework.

Source of truth for the logic and every threshold: `src/spiky/pipeline.py`
(`find_spiky_sites` + `_rim_runs_and_glyphs`, plan v23 -- 13/13 TP, 0/85 FP on the
reference set incl. holdout). This module RESTRUCTURES that validated code into the
framework's candidates+signals shape without changing any decision: constants are
IMPORTED from pipeline (single source of truth, no number duplication), and the
equivalence gate (gen-8 phase 4) verifies detect(page, PROFILE) returns exactly
`pipeline.find_spiky_sites(page)` on both full suite chapters.

`pipeline.find_spiky_sites` remains the production caller; re-pointing pipeline at this
profile is a separate, explicitly-decided step (per-classifier merge rule).

Signal semantics (from the v23 record):
  rim_runs  -- radial ink-run count around the elliptical rim annulus (scales
               1.02-1.30, 360 angular bins). THE discriminating signal
               (TP 60-103 vs FP max 42; threshold >= 50 centers the empty gap).
  glyphs    -- interior text-glyph count (components 20-4000 px inside the 0.9x
               ellipse at ink < S_TEXT_T). Second wall (TP min 7; threshold >= 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE.parents[2] / "src" / "spiky"))

from pipeline import (G_INT_MAX, G_INT_MIN, G_TOL, S_GLYPHS_MIN, S_MARGIN,  # noqa: E402
                       S_RUNS_MIN, _rim_runs_and_glyphs)
from classifiers.background import enclosed  # noqa: E402
from classifiers.detector_framework import Profile, Region, Signal  # noqa: E402


def _candidates(rgb: np.ndarray) -> list[Region]:
    """Gap-sealed enclosure candidates -- verbatim v23 generation (the pre-cascade part
    of `pipeline.find_spiky_sites`): components newly enclosed by 3x3-closing the
    non-band barrier, area-banded, bbox expanded by S_MARGIN."""
    band = rgb.min(axis=2) >= G_TOL
    k3 = np.ones((3, 3), np.uint8)
    barrier = cv2.morphologyEx((~band).astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    newly = enclosed(band & ~barrier) & ~enclosed(band)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        newly.astype(np.uint8), connectivity=8)
    H, W = band.shape
    out: list[Region] = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if not (G_INT_MIN <= a <= G_INT_MAX):
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        bx0 = max(0, x - S_MARGIN); by0 = max(0, y - S_MARGIN)
        bx1 = min(W, x + w + S_MARGIN); by1 = min(H, y + h + S_MARGIN)
        out.append((bx0, by0, bx1, by1))
    return out


# _rim_runs_and_glyphs computes both signals in one pass; cache per (id(page), region)
# so the two Signal wrappers don't recompute it.
_cache: dict[tuple[int, Region], tuple[int, int]] = {}


def _measure(rgb: np.ndarray, region: Region) -> tuple[int, int]:
    key = (id(rgb), region)
    if key not in _cache:
        if len(_cache) > 4096:
            _cache.clear()
        _cache[key] = _rim_runs_and_glyphs(rgb, *region)
    return _cache[key]


PROFILE = Profile(
    name="spiky_cloud",
    candidates=_candidates,
    signals=[
        Signal("rim_runs", lambda rgb, r: _measure(rgb, r)[0], lambda v: v >= S_RUNS_MIN),
        Signal("glyphs", lambda rgb, r: _measure(rgb, r)[1], lambda v: v >= S_GLYPHS_MIN),
    ],
)
