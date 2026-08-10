"""Pipeline replication v6 (plan v16, 2026-08-07) = v5 + three post-chain steps for the seven
user-found defect classes. v5 untouched.

Measured bands (2026-08-07, ch008 diagnostics -- see plan v16 / ml_strategy_history):
  Cluster 1(a) JPEG border residue: undeleted px adjacent to deletion are 76.6% >= 240
               (band [240,250) sits just under the extent gate's 250).
  Cluster 1(b) under-frame gray line: concentrated at 210-220 (measured 333-region histogram);
               band [200,230], frame-adjacent only -- a DIFFERENT band than (a), applied with
               a different trigger, exactly as the user predicted.

Steps added after the v5 chain:
  A. BORDER RESIDUE SWEEP (1a): undeleted px with gray in [240,250) within 3 px of deleted
     background -> delete. Iterated 3x so multi-px noise fringes collapse.
  B. UNDER-FRAME LINE (1b): px in [200,230] within 4 px BELOW a near-black (<=64) px AND
     within 4 px of deleted background -> delete.
  C. LOCAL BACKGROUND RECLAIM (Cluster 3 core; generalizes the v15 gated-seed idea to local
     pockets): undeleted bright pockets (gray >= 240) whose surrounding ring (dilate radius
     12) consists >= 85% of ink (<=64) + deleted background + page-edge contact -> delete,
     WITH GUARDS:
       - pocket area < 10,000 px (repair_frame_interiors' own min_interior_px convention:
         anything larger is a legitimate frame/bubble interior and is protected -- this is
         the 3(d) adversarial guard, tested in v16_guard_test.py);
       - pockets overlapping detected closed-frame interiors are skipped outright.
     Covers: 3(a) bubble<->frame gap pockets, 3(b) page-edge fused gaps (edge contact counts
     as background in the ring), 3(c) spiky-bubble inter-spike pockets, 3(d) enclosed SFX
     pockets (ring is pure ink; the enclosing stroke floats on deleted field).

Cluster 2 (variable-position edge line): no instance found in available eval data (ch007/008
edge search came up empty) -- FLAGGED as unrepresented; the page-edge trigger in step C is the
designed mechanism but is untested against a real instance of that specific artifact.

Usage:
  .venv/bin/python replicate_pipeline_v6.py <src.png> <out_mask.npy>
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from replicate_pipeline_v5 import clean_page as clean_page_v5

RESIDUE_LO, RESIDUE_HI = 240, 250
RESIDUE_RADIUS = 3
RESIDUE_ITERS = 3

UNDERLINE_LO, UNDERLINE_HI = 200, 230
UNDERLINE_DEPTH = 4
UNDERLINE_NEAR_DEL = 4

POCKET_BRIGHT = 240
POCKET_MAX_AREA = 10000
RING_RADIUS = 12
RING_BG_FRAC = 0.85
OUTER_RADIUS = 30
OUTER_DEL_FRAC = 0.35
INK_T = 64


def _protected_interiors(gray: np.ndarray) -> np.ndarray:
    """Closed-frame interiors (repair_frame_interiors-style): holes >= 10k px fully enclosed
    by near-black strokes. Pockets inside these are NEVER reclaimed (3(d) negative guard)."""
    stroke = (gray <= 40).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, k)
    H, W = gray.shape
    padded = np.zeros((H + 2, W + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = stroke
    ff = np.zeros((H + 4, W + 4), dtype=np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 1)
    holes = (padded[1:-1, 1:-1] == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    protected = np.zeros((H, W), dtype=bool)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= 10000:
            protected[labels == lbl] = True
    return protected


def step_a_border_residue(gray: np.ndarray, delete: np.ndarray) -> np.ndarray:
    band = (gray >= RESIDUE_LO) & (gray < RESIDUE_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (RESIDUE_RADIUS * 2 + 1, RESIDUE_RADIUS * 2 + 1))
    out = delete.copy()
    for _ in range(RESIDUE_ITERS):
        near = cv2.dilate(out.astype(np.uint8), k) > 0
        add = band & near & ~out
        if not add.any():
            break
        out |= add
    return out


def step_b_under_frame_line(gray: np.ndarray, delete: np.ndarray) -> np.ndarray:
    band = (gray >= UNDERLINE_LO) & (gray <= UNDERLINE_HI)
    dark = (gray <= INK_T).astype(np.uint8)
    below_dark = np.zeros_like(dark)
    for d in range(1, UNDERLINE_DEPTH + 1):
        below_dark[d:, :] |= dark[:-d, :]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (UNDERLINE_NEAR_DEL * 2 + 1, UNDERLINE_NEAR_DEL * 2 + 1))
    near_del = cv2.dilate(delete.astype(np.uint8), k) > 0
    add = band & (below_dark > 0) & near_del & ~delete
    out = delete.copy()
    out |= add
    return out


def step_c_local_reclaim(gray: np.ndarray, delete: np.ndarray,
                          protected: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    pockets = (~delete) & (gray >= POCKET_BRIGHT) & ~protected
    num, labels, stats, _ = cv2.connectedComponentsWithStats(pockets.astype(np.uint8), 4)
    out = delete.copy()
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (RING_RADIUS * 2 + 1, RING_RADIUS * 2 + 1))
    outer_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (OUTER_RADIUS * 2 + 1, OUTER_RADIUS * 2 + 1))
    ink = gray <= INK_T
    # NOTE (cluster-3 attempt 3, REJECTED 2026-08-07): dilating ink 2px for ring composition
    # (to absorb text-AA halos, targeting the 666-class note-text pockets) bought only -3%
    # residual there while leaking 9.4k px into kept bubble-text counters on the adversarial
    # test. Raw ink stays; the 666-class (large fragmented pockets around free-floating text)
    # is a documented residual of this round.
    for lbl in range(1, num):
        x, y, cw, chh, area = stats[lbl]
        # Cluster 2 thin-line exemption (mechanism REUSED, not a parallel implementation):
        # a variable-position white line hugging a page edge is a huge-area but tiny-width
        # component (measured: 58-62k rows at width 1-3px on ch007/008 col 688); the area cap
        # exists to protect frame/bubble interiors, which are never this thin.
        thin_line = min(cw, chh) <= 4 and (x == 0 or y == 0 or x + cw == W or y + chh == H)
        if area >= POCKET_MAX_AREA and not thin_line:
            continue
        pad = OUTER_RADIUS + 2
        y0, y1 = max(0, y - pad), min(H, y + chh + pad)
        x0, x1 = max(0, x - pad), min(W, x + cw + pad)
        comp = (labels[y0:y1, x0:x1] == lbl)
        comp_u8 = comp.astype(np.uint8)
        ring = (cv2.dilate(comp_u8, ring_k) > 0) & ~comp
        n_ring = int(ring.sum())
        if n_ring == 0:
            continue
        bg_like = ring & (out[y0:y1, x0:x1] | ink[y0:y1, x0:x1])
        # page-edge contact counts as background (3b fused-gap case)
        touches_edge = (y == 0) or (x == 0) or (y + chh == H) or (x + cw == W)
        frac = bg_like.sum() / n_ring
        if not (frac >= RING_BG_FRAC or (touches_edge and frac >= RING_BG_FRAC * 0.7)):
            continue
        # OUTER-RING guard (attempt 2; 3(d) adversarial fix): the context BEYOND the enclosing
        # ink must be predominantly deleted background (an SFX glyph floats on deleted field)
        # or the page edge. Letter counters inside kept text/panels fail this -- their outer
        # context is kept content -- which is exactly the leak the first guard test caught
        # (1,941 GT-keep px of text counters inside the 005-1 document panel).
        outer = (cv2.dilate(comp_u8, outer_k) > 0) & ~(cv2.dilate(comp_u8, ring_k) > 0) & ~ink[y0:y1, x0:x1]
        n_outer = int(outer.sum())
        outer_del = float((outer & out[y0:y1, x0:x1]).sum() / n_outer) if n_outer else 0.0
        if outer_del >= OUTER_DEL_FRAC or (touches_edge and n_outer < 200):
            out[y0:y1, x0:x1][comp] = True
    return out


def clean_page(rgb_u8: np.ndarray) -> np.ndarray:
    delete = clean_page_v5(rgb_u8)
    rgb = rgb_u8.astype(np.float32)
    gray = np.round((rgb.max(axis=2) + rgb.min(axis=2)) / 2.0).astype(np.uint8)
    protected = _protected_interiors(gray)
    delete = step_c_local_reclaim(gray, delete, protected)
    delete = step_a_border_residue(gray, delete)
    delete = step_b_under_frame_line(gray, delete)
    return delete


if __name__ == "__main__":
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    rgb = np.asarray(Image.open(src_path).convert("RGB"))
    mask = clean_page(rgb)
    np.save(out_path, mask)
    print(f"deleted {mask.mean():.4f} of page -> {out_path}")
