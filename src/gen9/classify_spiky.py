"""Gen9 v2 Classifier D: spiky-cloud detection (algorithm steps 45-54).

A spiky cloud (scream bubble) = a large white interior wrapped in a ring
of dense thin radiating spikes, floating in / at the edge of the
background field. Its interior is CONTENT (kept, text restored); the
whitish halo between/around the spikes is background (deleted).

Detection (measured on the 006 crop, decisions.md 2026-08-14): for each
large non-background cf-white comp, count the connected ink fragments in
an annulus band around it (radial spikes cross the band -> many small
fragments; a smooth bubble border or panel frame crosses it as 1-2 long
pieces). Spiky interior 110 scores 259; every other interior <= 29
except the full-width tinted panel (92, excluded as a panel by the
full-width test -- rule A territory). Threshold 100 sits ~2.6x under
the positive and ~3.4x over the strongest negative.

Library note: no skeletonization dependency -- the annulus-crossing
count is plain cv2.
"""
from __future__ import annotations

import cv2
import numpy as np

INTERIOR_MIN = 10000     # spiky interiors are big (GT 48,318)
HALO_R = 15              # S6 halo zone: dilate of ring (missed 0 at 15)
HALO_TOL = 120           # wand tol 120 vs white -> min channel >= 135
HALO_QUAL = 255 - HALO_TOL
HALO_SAT = 24            # neutral guard (halo is white/gray; tinted
                         # panel px under the lower spikes are frame)
BAND_IN = 10             # annulus: dilate(interior) radius 10..25
BAND_OUT = 25
CROSSINGS_MIN = 100      # spike-fragment count threshold (259 vs <=29)
RING_REACH = 25          # SFX-black comps within this of interior = ring
RECT_PAD = 20            # padded working rectangle around ring+interior
FULL_WIDTH_SLACK = 2
SQ3 = np.ones((3, 3), np.uint8)


def find_spiky(cf: np.ndarray, sfx: np.ndarray, bg_selected: np.ndarray,
               cf_labels: np.ndarray, cf_stats: np.ndarray,
               selected_ids: list[int]) -> list[dict]:
    """Detect spiky clouds. Returns one record per cloud:
    interior (bool mask), ring (bool mask), rect (y0,y1,x0,x1) padded,
    crossings (score), comp_id.
    """
    H, W = cf.shape
    ink = sfx < 128
    sel = set(selected_ids)
    out = []
    for i in range(1, cf_stats.shape[0]):
        x0, y0, w, h, a = (int(v) for v in cf_stats[i][:5])
        if a < INTERIOR_MIN or i in sel:
            continue
        if w >= W - FULL_WIDTH_SLACK:      # full-width = panel (rule A)
            continue
        pad = BAND_OUT + 15
        ys, xs = max(0, y0 - pad), max(0, x0 - pad)
        ye, xe = min(H, y0 + h + pad), min(W, x0 + w + pad)
        comp = (cf_labels[ys:ye, xs:xe] == i).astype(np.uint8)
        din = cv2.dilate(comp, np.ones((2 * BAND_IN + 1,) * 2, np.uint8))
        dout = cv2.dilate(comp, np.ones((2 * BAND_OUT + 1,) * 2, np.uint8))
        band = (dout - din).astype(bool)
        cross = (band & ink[ys:ye, xs:xe]).astype(np.uint8)
        nc, _ = cv2.connectedComponents(cross, 8)
        score = nc - 1
        if score < CROSSINGS_MIN:
            continue
        interior = np.zeros((H, W), bool)
        interior[ys:ye, xs:xe] = comp.astype(bool)
        # ring = SFX ink comps reaching within RING_REACH of the interior
        reach = np.zeros((H, W), bool)
        reach[ys:ye, xs:xe] = cv2.dilate(
            comp, np.ones((2 * RING_REACH + 1,) * 2, np.uint8)).astype(bool)
        if not hasattr(find_spiky, '_ink_cc') or \
                find_spiky._ink_cc[0] is not ink:
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(
                ink.astype(np.uint8), connectivity=8)
            find_spiky._ink_cc = (ink, n2, lab2, st2)
        _, n2, lab2, st2 = find_spiky._ink_cc
        ring = np.zeros((H, W), bool)
        for j in range(1, n2):
            jx, jy, jw, jh, ja = (int(v) for v in st2[j][:5])
            if ja < 500:
                continue
            # bbox intersect with the reach window first (cheap)
            if jx >= xe or jy >= ye or jx + jw <= xs or jy + jh <= ys:
                continue
            box = lab2[jy:jy + jh, jx:jx + jw] == j
            if reach[jy:jy + jh, jx:jx + jw][box].any() and \
                    not interior[jy:jy + jh, jx:jx + jw][box].any():
                ring[jy:jy + jh, jx:jx + jw] |= box
        area = interior | ring
        ysA, xsA = np.where(area)
        rect = (max(0, int(ysA.min()) - RECT_PAD),
                min(H, int(ysA.max()) + 1 + RECT_PAD),
                max(0, int(xsA.min()) - RECT_PAD),
                min(W, int(xsA.max()) + 1 + RECT_PAD))
        out.append(dict(comp_id=i, interior=interior, ring=ring, rect=rect,
                        crossings=score))
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    u = mask.astype(np.uint8)
    n, lab = cv2.connectedComponents((u == 0).astype(np.uint8),
                                     connectivity=4)
    border = np.unique(np.r_[lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    return mask | (~np.isin(lab, border) & (u == 0))


def spiky_zone(cloud: dict, cf_labels: np.ndarray, cf_stats: np.ndarray,
               selected_ids: list[int]) -> np.ndarray:
    """S6 working zone: halo band around the ring + the interior with its
    text region filled, clipped at the host background band's bottom edge
    (the part of the cloud over the frame below is locked territory --
    validated: 0 GT px lost to the clip, 32k saved from over-delete)."""
    ring, interior = cloud['ring'], cloud['interior']
    k = np.ones((2 * HALO_R + 1,) * 2, np.uint8)
    core = fill_holes(cv2.dilate(interior.astype(np.uint8), SQ3).astype(bool))
    zone = cv2.dilate(ring.astype(np.uint8), k).astype(bool) | core
    area = cv2.dilate((ring | interior).astype(np.uint8),
                      np.ones((9, 9), np.uint8)).astype(bool)
    host, best = None, -1
    for i in selected_ids:
        ov = int((area & (cf_labels == i)).sum())
        if ov > best:
            host, best = i, ov
    ymax = int(cf_stats[host, 1] + cf_stats[host, 3]) + 1
    zone[ymax:] = False
    return zone


def halo_classify(src: np.ndarray, zone: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """S6 px rule inside the zone: whitish-neutral = background (delete),
    everything else = content fringe (restore). Returns (bg, not_bg)."""
    whitish = (src.min(axis=2) >= HALO_QUAL) & \
              (src.max(axis=2).astype(int) - src.min(axis=2) < HALO_SAT)
    return zone & whitish, zone & ~whitish
