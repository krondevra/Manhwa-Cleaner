"""Manual-pipeline replication v10 -- plan v23 outcome.

DEFAULT = v9 EXACTLY. steps='S' adds AUTO-SCOPED spiky-cloud reclaim: gap-sealed
enclosure candidates (v21 proposal) filtered by the v23 two-signal cascade, then the
verified action (`clean_spiky_region`, v21) at each accepted site.

== v23 cascade (fit on 001/002 manual etalons; 13 TP / 85 FP / 4 ambiguous) ==

  Signal A -- radial run count: ink runs around the elliptical rim annulus
    (scales 1.02-1.30 of the bbox-inscribed ellipse, 360 angular bins, 24 radial
    samples). TP range 60-103 runs, FP max 42 -> threshold >= 50 centers the empty
    gap. THE discriminating signal.
  Signal B -- spectral periodicity: NO separation (TP peak 0.02-0.05 vs FP 0.01-0.09)
    -- honest negative, not used.
  Signal C -- interior glyph count (area 20-4000 px inside 0.9x ellipse): partial
    separation (TP min 7, FP median 6) -- used as the second wall: >= 5.

  Cascade (A >= 50 AND C >= 5): 13/13 TP, 0/85 FP, 0 FN on the reference set; the
  single FP that passed A alone (42 runs) is killed by both the raised A threshold
  and C.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replicate_pipeline_v6 import _protected_interiors  # noqa: E402
from replicate_pipeline_v8 import (_enclosed, clean_spiky_region,  # noqa: E402
                                    G_TOL, G_INT_MIN, G_INT_MAX)
from replicate_pipeline_v9 import clean_page as clean_page_v9  # noqa: E402

S_MARGIN = 60        # bbox margin around an enclosure (matches the v23 measurement)
S_RUNS_MIN = 50      # signal A threshold (empty gap [43, 59] on the reference set)
S_GLYPHS_MIN = 5     # signal C threshold (TP min 7 at T=100)
S_TEXT_T = 100       # C's ink threshold: cloud text can be dark-gray, not near-black
                     # (019 holdout: 0 glyphs at <55, 33 at <100; reference confusion
                     # unchanged at <100 -- 13/13 TP, 0/85 FP)
NBINS = 360


def _rim_runs_and_glyphs(rgb: np.ndarray, bx0: int, by0: int, bx1: int, by1: int):
    pad = 40
    x0 = max(0, bx0 - pad); y0 = max(0, by0 - pad)
    x1 = min(rgb.shape[1], bx1 + pad); y1 = min(rgb.shape[0], by1 + pad)
    win = rgb[y0:y1, x0:x1]
    ink = win.min(axis=2) < G_TOL
    cx = (bx0 + bx1) / 2 - x0
    cy = (by0 + by1) / 2 - y0
    ax = max((bx1 - bx0) / 2 - S_MARGIN, 12)
    ay = max((by1 - by0) / 2 - S_MARGIN, 12)
    th = np.linspace(0, 2 * np.pi, NBINS, endpoint=False)
    prof = np.zeros(NBINS, np.float32)
    H, W = ink.shape
    scales = np.linspace(1.02, 1.30, 24)
    for s in scales:
        xs = np.clip((cx + s * ax * np.cos(th)).astype(int), 0, W - 1)
        ys = np.clip((cy + s * ay * np.sin(th)).astype(int), 0, H - 1)
        prof += ink[ys, xs]
    prof /= len(scales)
    above = prof > max(prof.mean(), 0.05)
    d = np.diff(above.astype(int))
    n_runs = int(above[0] and not above[-1]) + int((d == 1).sum())
    yy, xx = np.mgrid[0:H, 0:W]
    inside = (((xx - cx) / (0.9 * ax)) ** 2 + ((yy - cy) / (0.9 * ay)) ** 2) <= 1.0
    text_ink = win.min(axis=2) < S_TEXT_T
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        (text_ink & inside).astype(np.uint8), connectivity=8)
    n_gl = sum(1 for i in range(1, num)
               if 20 <= stats[i, cv2.CC_STAT_AREA] <= 4000)
    return n_runs, n_gl


def find_spiky_sites(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Gap-sealed enclosure candidates filtered by the v23 cascade. Returns bboxes."""
    band = rgb.min(axis=2) >= G_TOL
    k3 = np.ones((3, 3), np.uint8)
    barrier = cv2.morphologyEx((~band).astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    newly = _enclosed(band & ~barrier) & ~_enclosed(band)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        newly.astype(np.uint8), connectivity=8)
    H, W = band.shape
    out = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if not (G_INT_MIN <= a <= G_INT_MAX):
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        bx0 = max(0, x - S_MARGIN); by0 = max(0, y - S_MARGIN)
        bx1 = min(W, x + w + S_MARGIN); by1 = min(H, y + h + S_MARGIN)
        n_runs, n_gl = _rim_runs_and_glyphs(rgb, bx0, by0, bx1, by1)
        if n_runs >= S_RUNS_MIN and n_gl >= S_GLYPHS_MIN:
            out.append((bx0, by0, bx1, by1))
    return out


def clean_spiky_region_clipped(rgb: np.ndarray, delete: np.ndarray,
                                bbox: tuple[int, int, int, int],
                                protected: np.ndarray | None = None) -> np.ndarray:
    """The v8 action + panel-line clipping: the deletion is limited to the region
    4-connected to the bbox center without crossing a long horizontal dark run
    (>=100 px, gray<=100) -- the fit-page diagnostic showed the raw bbox margin dips
    into the panel below/above a cloud and blanks a strip of panel art."""
    x0, y0, x1, y1 = bbox
    before = delete
    after = clean_spiky_region(rgb, before, bbox, protected=protected)
    changed = after != before
    if not changed.any():
        return after
    win = (slice(y0, y1), slice(x0, x1))
    f = rgb[win].astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    dark = (gray <= 100).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 1))
    runs = cv2.dilate(cv2.erode(dark, hk), hk) > 0
    passable = ~runs
    seed = np.zeros_like(passable)
    cyy, cxx = passable.shape[0] // 2, passable.shape[1] // 2
    seed[max(0, cyy - 3):cyy + 3, max(0, cxx - 3):cxx + 3] = True
    num, lab = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
    keep_ids = np.unique(lab[seed & passable])
    reach = np.isin(lab, keep_ids[keep_ids != 0])
    out = before.copy()
    ch_win = changed[win] & reach
    reg = out[win]
    reg[ch_win] = after[win][ch_win]
    out[win] = reg
    return out


def clean_page(rgb: np.ndarray, steps: str = "") -> np.ndarray:
    """v10 default == v9 exactly. steps='S' adds cascade-scoped spiky reclaim; 'D' is
    v9's opt-in dark-backdrop track (pass 'SD' for both)."""
    delete = clean_page_v9(rgb, steps="D" if "D" in steps else "")
    if "S" in steps:
        f = rgb.astype(np.float32)
        gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
        prot = _protected_interiors(gray)
        for bbox in find_spiky_sites(rgb):
            delete = clean_spiky_region_clipped(rgb, delete, bbox, protected=prot)
    return delete
