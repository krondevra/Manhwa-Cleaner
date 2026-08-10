"""Manual-pipeline replication v8 -- plan v21 outcome.

DEFAULT BEHAVIOR = v7 EXACTLY (no new automatic steps). What v8 adds is the verified
spiky-cloud ACTION as a region-parameterized utility for the manual/GUI track, plus the
documented-negative automatic variants (off by default).

== The spiky-cloud manual pipeline (spiky-clauds/pipeline-spiky-clouds.md), decoded ==

  1. Magic Wand tol=200, contiguous OFF, feather 0 on the white background, applied
     LOCALLY (human-chosen region) -> raster mask. Verified against the 019-2 etalon:
     100.00% of etalon-deleted px have min-channel >= 55 (= 255 - 200), p0.1 = 57.
  2. img-copy + apply mask + MinMax1 (1px close) -> the spiky contour is sealed.
  3. Wand tol=0 contiguous ON inside -> sealed interior selection; interior wholesale KEPT.

The ACTION given a correct region is fully deterministic (crop result: under-deletion
19,268 -> 96 px, over-deletion ~0 within scope). `clean_spiky_region` implements it.

== v21 automatic-scope ladder (all negatives; see logs/v21_*.log) ==

  a1 unscoped flood         -- INVALID measurement (leaky large-kernel dilation crossed
                               thin ink barriers); corrected geometry becomes a2.
  a2 flood scoped to gap-sealed-enclosure annuli -- SAFE (slab suspicious 2,280 px;
                               crop over 0) but recovers only 31% of the crop target
                               (under 19,268 -> 13,329): sealed wedges are unreachable
                               by any connective flood.
  a3 non-connective deletion in the same annuli  -- solves the crop (under -> 96) but
                               destroys art elsewhere: 53,148 suspicious px in one 35k
                               slab; false-positive "enclosures" include forest art and
                               a character's face+gradient background. Gap-sealed
                               enclosure detection cannot distinguish a floating spiky
                               SFX cloud from ordinary art enclosures.

Conclusion: the SCOPE decision is object-level semantics (the same locality wall as
v20's D/F classes) -> spiky clouds go to the manual/GUI track, with the action below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replicate_pipeline_v6 import _protected_interiors  # noqa: E402
from replicate_pipeline_v7 import clean_page as clean_page_v7  # noqa: E402

G_TOL = 55        # min-channel floor = magic wand tolerance 200 from white
G_SCOPE_R = 150   # documented-negative auto variants: annulus radius
G_INT_MIN = 3000  # minimal interior area for a gap-sealed enclosure
G_INT_MAX = 500000


def _enclosed(passable: np.ndarray) -> np.ndarray:
    """Regions of the passable map NOT connected to the image border (exact, 4-conn)."""
    num, lab = cv2.connectedComponents(passable.astype(np.uint8), connectivity=4)
    border = np.zeros(num, dtype=bool)
    for edge in (lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]):
        border[np.unique(edge)] = True
    border[0] = True
    return ~border[lab] & passable


def _flood(seed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Union of 4-connected components of mask that contain any seed px (exact)."""
    num, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    hit = np.zeros(num, dtype=bool)
    hit[np.unique(lab[seed & mask])] = True
    hit[0] = False
    return hit[lab]


def clean_spiky_region(rgb: np.ndarray, delete: np.ndarray,
                       bbox: tuple[int, int, int, int],
                       protected: np.ndarray | None = None) -> np.ndarray:
    """GUI-track action: apply the verified spiky-cloud recipe inside a HUMAN-CHOSEN bbox.

    bbox = (x0, y0, x1, y1) generously covering the spiky cloud (rim + halo soup).
    Returns the updated delete mask:
      - inside bbox, every px with min-channel >= G_TOL is deleted (wand tol-200,
        contiguous OFF) EXCEPT sealed interiors (MinMax1-closed contours) and existing
        protected frame/bubble interiors, which are wholesale kept.
    """
    x0, y0, x1, y1 = bbox
    H, W = delete.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    sub = rgb[y0:y1, x0:x1]
    f = sub.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    if protected is None:
        prot = _protected_interiors(gray)
    else:
        prot = protected[y0:y1, x0:x1]
    band = sub.min(axis=2) >= G_TOL
    k3 = np.ones((3, 3), np.uint8)
    barrier = cv2.morphologyEx((~band).astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    interior = _enclosed(band & ~barrier)
    out = delete.copy()
    reg = out[y0:y1, x0:x1]
    reg |= band & ~interior & ~prot
    # the sealed interior is wholesale kept, exactly like the manual fill step
    reg &= ~(interior | prot)
    out[y0:y1, x0:x1] = reg
    return out


def step_g_spiky(rgb: np.ndarray, gray: np.ndarray, delete: np.ndarray,
                 protected: np.ndarray | None = None,
                 mode: str = "a3") -> np.ndarray:
    """DOCUMENTED NEGATIVE (default off): automatic-scope variants a2/a3 -- see module
    docstring. Retained for the record; do not enable in production."""
    if protected is None:
        protected = _protected_interiors(gray)
    band = rgb.min(axis=2) >= G_TOL
    k3 = np.ones((3, 3), np.uint8)
    barrier_closed = cv2.morphologyEx((~band).astype(np.uint8), cv2.MORPH_CLOSE, k3) > 0
    enc_before = _enclosed(band)
    enc_after = _enclosed(band & ~barrier_closed)
    newly = enc_after & ~enc_before
    num, lab, stats, _ = cv2.connectedComponentsWithStats(newly.astype(np.uint8), connectivity=8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        if G_INT_MIN <= stats[i, cv2.CC_STAT_AREA] <= G_INT_MAX:
            keep[i] = True
    spiky_int = keep[lab]
    if not spiky_int.any():
        return delete
    ann = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * G_SCOPE_R + 1,) * 2)
    scope = (cv2.dilate(spiky_int.astype(np.uint8), ann) > 0) & ~spiky_int
    if mode == "a2":
        passable = band & ~barrier_closed & ~protected & scope & ~enc_after
        seed = delete | (cv2.dilate(delete.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        return delete | _flood(seed, passable)
    return delete | (band & scope & ~protected)


APRIME_LO, APRIME_HI = 230, 256  # v21 Task 2: the border-residue bands v16-A never covered
APRIME_RADIUS, APRIME_ITERS = 3, 3


def step_a_prime(gray: np.ndarray, delete: np.ndarray) -> np.ndarray:
    """v21 Task 2 fix: same mechanism as v6 step A (iterated 3px adjacency sweep), band
    widened to [230,256). Root cause evidence (logs/v21_combined_chapters.log): border-zone
    under-deletion vs manual chapter GT sits at [250,256) 26.7k/53.7k px and [230,240)
    ~2-3k px on 001/002, untouched by v5->v6->v7 (a GAP, not a regression)."""
    # attempt 3: sandwich geometry -- the residue is a thin white line BETWEEN frame ink
    # and deleted background, so require adjacency to BOTH (<=3px each). Attempt 2's
    # frame-zone gate alone still ate kept white art (net negative, see log).
    dark = (gray <= 100).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 1))
    runs = cv2.dilate(cv2.erode(dark, hk), hk)
    frame_zone = cv2.dilate(runs, np.ones((21, 21), np.uint8)) > 0
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near_ink = cv2.dilate(dark, k7) > 0
    band = (gray >= APRIME_LO) & (gray < APRIME_HI) & frame_zone & near_ink
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    (APRIME_RADIUS * 2 + 1, APRIME_RADIUS * 2 + 1))
    out = delete.copy()
    for _ in range(APRIME_ITERS):
        near = cv2.dilate(out.astype(np.uint8), k) > 0
        add = band & near & ~out
        if not add.any():
            break
        out |= add
    return out


def clean_page(rgb: np.ndarray, steps: str = "P") -> np.ndarray:
    """v8 default == v7 + step A' (P). Pass steps='' for v7-exact behavior; 'G' adds the
    documented-negative automatic spiky variants (not production)."""
    f = rgb.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    delete = clean_page_v7(rgb)
    if "P" in steps:
        delete = step_a_prime(gray, delete)
    if "G" in steps:
        delete = step_g_spiky(rgb, gray, delete)
    return delete
