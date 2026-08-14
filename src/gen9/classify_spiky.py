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
BAND_IN = 10             # annulus: dilate(interior) radius 10..25
BAND_OUT = 25
CROSSINGS_MIN = 100      # spike-fragment count threshold (259 vs <=29)
RING_REACH = 25          # SFX-black comps within this of interior = ring
RECT_PAD = 20            # padded working rectangle around ring+interior
FULL_WIDTH_SLACK = 2


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
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(
            ink.astype(np.uint8), connectivity=8)
        ring = np.zeros((H, W), bool)
        for j in range(1, n2):
            if st2[j, 4] < 500:
                continue
            cj = lab2 == j
            if (cj & reach).any() and not (cj & interior).any():
                ring |= cj
        area = interior | ring
        ysA, xsA = np.where(area)
        rect = (max(0, int(ysA.min()) - RECT_PAD),
                min(H, int(ysA.max()) + 1 + RECT_PAD),
                max(0, int(xsA.min()) - RECT_PAD),
                min(W, int(xsA.max()) + 1 + RECT_PAD))
        out.append(dict(comp_id=i, interior=interior, ring=ring, rect=rect,
                        crossings=score))
    return out
