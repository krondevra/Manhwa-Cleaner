"""Pipeline replication v7 (plan v19, 2026-08-08) = v6 + step E3 ONLY. v6 untouched.

SHIPPING CONFIG: steps="E" (the frame-strip fix). Steps D and F remain in the file as the
measured record of HONEST NEGATIVES (default-off; see decisions.md v19 entries):

  E3 (SHIPPED): the 336x7 gray-191 frame strip on the background side of long horizontal
     frame lines -- band [185,230], within 2px of a >=100px horizontal dark run, reachable
     from deleted background geodesically through the band (<=8px). Etalon crop: -334 px
     under-deletion, zero over-del cost; page-wide flagged pixels visually verified as
     frame-adjacent background AA (the guard proxy penalized v6's own under-deletion).

  D (NEGATIVE, 3 attempts: blur-context / geodesic-25 / geodesic-6+size-cap): mechanizing
     the user's "select by color, threshold 23" halo recipe fails page-wide safety every
     time (17k-301k suspicious px on 007) -- the manual recipe works because a human selects
     the floating-text region first; that locality judgment has no safe classical proxy here.

  F (NEGATIVE, 3 attempts: ring 0.60 / ring 0.55 / margin-adjacency): the page-edge fused
     gap (v16 222-class) -- the safe variant captures nothing (-5 px), the effective
     variants bite bright art near margins (35k suspicious px on 007). Class stays open.

Guards carried: v16's adversarial negatives (kept-bubble text counters), the SFX
punch-through frame guard (v19), and the full gen-7 battery bars.

Usage:
  .venv/bin/python replicate_pipeline_v7.py <src.png> <out_mask.npy>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from replicate_pipeline_v6 import (clean_page as clean_page_v6, _protected_interiors,
                                     step_c_local_reclaim, step_a_border_residue,
                                     step_b_under_frame_line)
from replicate_pipeline_v5 import clean_page as clean_page_v5

# D: local color sweep (user's manual threshold 23 from white)
COLOR_T = 23
D_BAND_LO = 255 - COLOR_T          # 232
D_CTX_RADIUS = 25
D_CTX_BAR = 0.55

# E: frame-strip extension
E_BAND_LO, E_BAND_HI = 185, 230
E_DEPTH = 4
E_NEAR_DEL = 4

# F: edge-margin gap pockets
F_MARGIN_X = 90
F_RING_BAR = 0.55  # F2: 0.60->0.55 measured strictly better (-663 under, zero over cost)
F_MAX_AREA = 10000


def step_d_local_color_sweep(gray, delete, protected):
    """D2 (attempt 2): GEODESIC growth through the >=232 band from deleted-adjacent seeds,
    reach <= D_CTX_RADIUS px. D1's symmetric blur context (17.2k suspicious px on 007 alone)
    bit into panel interiors near borders; geodesic growth cannot cross border ink because
    ink pixels are outside the band."""
    # D3 (final attempt): reach 25 -> 6 px and added-component size cap 2,000 px (halo
    # scale). D2's 25-px growth formed large ribbons along every art/background boundary
    # lacking blocking ink (301k suspicious px on 007); halos are small and hug the deleted
    # region, ribbons are large and thin -- the size cap separates them.
    band = (gray >= D_BAND_LO) & ~protected
    k3 = np.ones((3, 3), np.uint8)
    reach = (cv2.dilate(delete.astype(np.uint8), k3) > 0) & band & ~delete
    for _ in range(6):
        nxt = (cv2.dilate((reach | delete).astype(np.uint8), k3) > 0) & band & ~delete
        if nxt.sum() == reach.sum():
            break
        reach = nxt
    num, labels, stats, _ = cv2.connectedComponentsWithStats(reach.astype(np.uint8), 8)
    keep = np.zeros(num, dtype=bool)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] <= 2000:
            keep[lbl] = True
    out = delete.copy()
    out |= keep[labels]
    return out


def step_e_frame_strip(gray, delete):
    """E2 geometry (E1's any-dark-within-4px trigger cost as much as it fixed, -486/+455;
    E2 requires adjacency within 2px to a LONG horizontal dark run (>=100px) -- a frame
    line, not a text/art stroke -- and lands at -334/+72 on the etalon crop)."""
    band = (gray >= E_BAND_LO) & (gray <= E_BAND_HI)
    dark = (gray <= 64).astype(np.uint8)
    kw = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 1))
    long_dark = cv2.dilate(cv2.erode(dark, kw), kw)
    near_long = np.zeros_like(long_dark)
    for d in range(1, 3):
        near_long[d:, :] |= long_dark[:-d, :]
        near_long[:-d, :] |= long_dark[d:, :]
    # E3 (attempt 3): replace blur-proximity with geodesic connection -- the strip must be
    # reachable from deleted background THROUGH the band (<=8 px), so panel-interior gray
    # near a frame line (separated from deleted bg by the ink line) is untouchable.
    k3 = np.ones((3, 3), np.uint8)
    reach = (cv2.dilate(delete.astype(np.uint8), k3) > 0) & band & ~delete
    for _ in range(8):
        nxt = (cv2.dilate((reach | delete).astype(np.uint8), k3) > 0) & band & ~delete
        if nxt.sum() == reach.sum():
            break
        reach = nxt
    add = reach & (near_long > 0)
    out = delete.copy()
    out |= add
    return out


def step_f_edge_gap_pockets(gray, delete, protected):
    H, W = gray.shape
    pockets = (~delete) & (gray >= 240) & ~protected
    num, labels, stats, _ = cv2.connectedComponentsWithStats(pockets.astype(np.uint8), 4)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ink = gray <= 64
    out = delete.copy()
    for lbl in range(1, num):
        x, y, cw, chh, area = stats[lbl]
        if area >= F_MAX_AREA:
            continue
        near_edge = x <= F_MARGIN_X or (x + cw) >= W - F_MARGIN_X
        if not near_edge:
            continue
        pad = 14
        y0, y1 = max(0, y - pad), min(H, y + chh + pad)
        x0, x1 = max(0, x - pad), min(W, x + cw + pad)
        comp = labels[y0:y1, x0:x1] == lbl
        ring = (cv2.dilate(comp.astype(np.uint8), ring_k) > 0) & ~comp
        n_ring = int(ring.sum())
        if n_ring == 0:
            continue
        bg_like = ring & (out[y0:y1, x0:x1] | ink[y0:y1, x0:x1])
        # F3 (attempt 3): the pocket itself must be DIRECTLY adjacent to deleted MARGIN
        # pixels (dilate-1 touches deletion inside the margin columns) -- the fused-gap
        # signature. F1/F2's "any deleted px in a side strip" was trivially satisfied by
        # full-height margin deletions (35.5k suspicious px on 007), firing on bright art
        # near edges that never touches the margin deletion.
        comp_d1 = cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        touch = comp_d1 & out[y0:y1, x0:x1] & ~comp
        Wm = out.shape[1]
        margin_cols = np.zeros_like(out[y0:y1, x0:x1])
        gx0, gx1 = x0, x1
        left_w = max(0, min(12, gx1) - gx0) if gx0 < 12 else 0
        if gx0 < 12:
            margin_cols[:, :max(0, 12 - gx0)] = True
        if gx1 > Wm - 12:
            margin_cols[:, -(gx1 - (Wm - 12)):] = True
        margin_touch = bool((touch & margin_cols).any())
        if margin_touch and (bg_like.sum() / n_ring) >= F_RING_BAR:
            out[y0:y1, x0:x1][comp] = True
    return out


def clean_page(rgb_u8: np.ndarray, steps: str = "E") -> np.ndarray:
    delete = clean_page_v6(rgb_u8)
    rgb = rgb_u8.astype(np.float32)
    gray = np.round((rgb.max(axis=2) + rgb.min(axis=2)) / 2.0).astype(np.uint8)
    protected = _protected_interiors(gray)
    if "F" in steps:
        delete = step_f_edge_gap_pockets(gray, delete, protected)
    if "D" in steps:
        delete = step_d_local_color_sweep(gray, delete, protected)
    if "E" in steps:
        delete = step_e_frame_strip(gray, delete)
    return delete


if __name__ == "__main__":
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    steps = sys.argv[sys.argv.index("--steps") + 1] if "--steps" in sys.argv else "E"
    rgb = np.asarray(Image.open(src_path).convert("RGB"))
    mask = clean_page(rgb, steps)
    np.save(out_path, mask)
    print(f"deleted {mask.mean():.4f} of page -> {out_path}")
