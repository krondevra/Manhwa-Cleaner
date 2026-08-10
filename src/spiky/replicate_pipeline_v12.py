"""Manual-pipeline replication v12 -- plan v26 fix round.

Root cause fixed (Fix A): `_protected_interiors_v2` -- a >= 10k border-disconnected hole
is protected ONLY IF its ink-adjacent boundary is dominated (>= 90%) by ONE connected
stroke component on the UN-closed stroke map: a real frame contour. Inter-panel gutters
(top/bottom strokes = different frames, sides = page edge) fail the test. The v6
original (border-disconnection only, 5x5-closed strokes) mis-protected those gutters at
full-page scale -- the v25-diagnosed root cause behind 002_5/002_6, step Q's page-wide
regression, and the action's silent restores since v8.

Fix B: occlusion bridging in the S-action clip -- collinear frame-run segments are
bridged across the window where the cloud body occludes the panel line (rows must carry
>= 100px of genuine run on BOTH sides of the window's central third), so the clip flood
stops at occluded frame lines (the 019_0/2/3/4/6 leak).

Default steps='Q': v10 default + interior-restore on prot_v2. 'S' adds the v12 action
(bridged clip + frame-run band + prot_v2 restore). 'D' = the v9 opt-in dark track.
FIX_A / FIX_B module flags exist ONLY for the v26 comparison harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replicate_pipeline_v6 import _protected_interiors  # noqa: E402
from replicate_pipeline_v8 import _enclosed, _flood, G_TOL  # noqa: E402
from replicate_pipeline_v9 import (clean_page as clean_page_v9,  # noqa: E402
                                    step_h_dark_backdrop)
from replicate_pipeline_v10 import find_spiky_sites  # noqa: E402
from replicate_pipeline_v11 import FRAME_BAND  # noqa: E402

FIX_A = True   # comparison-harness toggles; production = all True
FIX_B = True
FIX_E = True   # ellipse scope: the action deletes only inside the cloud's elliptical
               # ring (<= ELLIPSE_MAX of the bbox-inscribed ellipse) -- the v23 signal
               # geometry; bbox corners/edges beyond it are art, not soup (the residual
               # leak class on irregular-boundary panels: 019_0/3/6/7 in the v26 table)

FIX_S = True   # saturation gate: soup is achromatic; leaked panel art is colored.
               # cand must be near-gray (max-min <= SAT_MAX) or near-white (min >= 240).
FIX_R = False  # v27 attempt B3: ring-distance gate -- cand must be within RING_PX of
               # ray ink (chamfer distance), replacing unbounded flood reach as the
               # deep-leak stopper. off by default pending measurement.

PROT_DOMINANCE = 0.90
ELLIPSE_MAX = 1.45
SAT_MAX = 40
RING_PX = 80


def _protected_interiors_v2(gray: np.ndarray) -> np.ndarray:
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


def _prot(gray: np.ndarray) -> np.ndarray:
    return _protected_interiors_v2(gray) if FIX_A else _protected_interiors(gray)


RUN_KERNEL_W = 101  # v27 attempt B2: sweep this (min run length survivable by erode+dilate)


def _bridged_runs(gray_win: np.ndarray) -> np.ndarray:
    """Frame runs with Fix B occlusion bridging."""
    dark = (gray_win <= 100).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (RUN_KERNEL_W, 1))
    runs = cv2.dilate(cv2.erode(dark, hk), hk) > 0
    if not FIX_B:
        return runs
    H, W = runs.shape
    third = W // 3
    left = runs[:, :third].sum(axis=1)
    right = runs[:, -third:].sum(axis=1)
    both = (left >= 100) & (right >= 100)
    if both.any():
        wide = cv2.getStructuringElement(cv2.MORPH_RECT, (W, 1))
        closed = cv2.morphologyEx(runs.astype(np.uint8), cv2.MORPH_CLOSE, wide) > 0
        runs = runs | (closed & both[:, None])
    return runs


def clean_spiky_region_v12(rgb: np.ndarray, delete: np.ndarray,
                            bbox: tuple[int, int, int, int],
                            protected: np.ndarray | None = None) -> np.ndarray:
    """v12 action: tol-200 band deletion, sealed-interior + prot_v2 kept, clip flood
    limited by BRIDGED frame runs, +-FRAME_BAND protection around runs."""
    x0, y0, x1, y1 = bbox
    H, W = delete.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    win = (slice(y0, y1), slice(x0, x1))
    sub = rgb[win]
    f = sub.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    prot = _prot(gray) if protected is None else protected[win]
    band = sub.min(axis=2) >= G_TOL
    k3 = np.ones((3, 3), np.uint8)
    barrier = cv2.morphologyEx((~band).astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    interior = _enclosed(band & ~barrier)
    runs = _bridged_runs(gray)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * FRAME_BAND + 1))
    run_band = cv2.dilate(runs.astype(np.uint8), vk) > 0
    # candidate deletions: band, not interior/prot, not in the frame-run band
    cand = band & ~interior & ~prot & ~run_band
    if FIX_E:
        hh, ww = gray.shape
        from replicate_pipeline_v10 import S_MARGIN
        cx, cy = ww / 2.0, hh / 2.0
        ax = max(ww / 2.0 - S_MARGIN, 12.0)
        ay = max(hh / 2.0 - S_MARGIN, 12.0)
        yy, xx = np.mgrid[0:hh, 0:ww]
        ell = (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2) <= ELLIPSE_MAX ** 2
        cand &= ell
    if FIX_S:
        sat = sub.max(axis=2).astype(np.int16) - sub.min(axis=2).astype(np.int16)
        cand &= (sat <= SAT_MAX) | (sub.min(axis=2) >= 240)
    if FIX_R:
        ink = (gray <= G_TOL - 15).astype(np.uint8)  # ray/rim ink, not the soup itself
        dist = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 5)
        cand &= dist <= RING_PX
    # clip: only the region 4-connected to the window center without crossing runs
    passable = ~runs
    seed = np.zeros_like(passable)
    cy, cx = passable.shape[0] // 2, passable.shape[1] // 2
    seed[max(0, cy - 3):cy + 3, max(0, cx - 3):cx + 3] = True
    reach = _flood(seed, passable)
    cand &= reach
    out = delete.copy()
    reg = out[win]
    reg |= cand
    # restore ONLY the sealed spiky interior and true (v2) protected interiors
    reg &= ~(interior | prot)
    out[win] = reg
    return out


def clean_page(rgb: np.ndarray, steps: str = "Q") -> np.ndarray:
    """v12: default = v10-default + step Q on prot_v2; 'S' = v12 action; 'D' opt-in."""
    f = rgb.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    delete = clean_page_v9(rgb, steps="")
    prot = _prot(gray)
    if "Q" in steps:
        delete = delete & ~prot
    if "D" in steps:
        delete = step_h_dark_backdrop(rgb, gray, delete)
    if "S" in steps:
        for bbox in find_spiky_sites(rgb):
            delete = clean_spiky_region_v12(rgb, delete, bbox, protected=prot)
    return delete
