"""Classical spiky-cloud cleaning pipeline -- consolidated module (2026-08-10).

This file merges the historical replicate_pipeline_v2..v12 chain into one module; each
"vN" section below is byte-identical logic carried over from the file of that name (the
per-version files, their full docstrings, and the evidence trail live in git history --
see docs/decisions.md "Classical replication track" and the 7.12.x/7.13.x commits).

Entry points:
  clean_page(rgb, steps="Q")      -- canonical (v12 candidate; steps: Q interior-restore,
                                     S spiky action, D opt-in dark-backdrop track)
  clean_page_v10(rgb, steps="")   -- v10, production default until v12 confirmed
  clean_page_v7(rgb, steps="E")   -- white-track reference used by v26_battery
  apply_config("ABES")            -- set the FIX_* comparison-harness flags from letters
  clean_spiky_region(...)         -- the verified manual/GUI-track action (v8)

CLI:
  .venv/bin/python src/spiky/pipeline.py <src.png> <out_mask.npy>
      [--steps QS] [--config ABES] [--entry v12|v10|v7]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import ml_cleaner
sys.modules["__main__"].train_command = ml_cleaner.train_command
from ml_cleaner import repair_frame_interiors

from leak_detector import detect_leaks
from band_classifier import find_bands, load_band_net, classify_bands


# ============================================================================
# v2 -- PSD-calibrated white-mask builder (plan v12 steps 4-5).
# Lightness desaturate, direct-gamma levels, > threshold, square-kernel min/max;
# 99.3-99.7% pixel agreement with the reference PSD's own mask rasters.
# (v2's one-off per-page fit tooling was dropped in the consolidation -- git history.)
# ============================================================================

SETTINGS = {
    "soft-white": dict(b=10, g=1.50, w=110, thr=48, order="minmax", r1=4, r2=4),
    "hard-white": dict(b=18, g=0.80, w=70, thr=20, order="minmax", r1=22, r2=10),
}


def build_mask(gray_light: np.ndarray, s: dict) -> np.ndarray:
    lv = np.round(255.0 * np.power(np.clip((gray_light - s["b"]) / float(s["w"] - s["b"]),
                                             0, 1), s["g"]))
    m = ((lv > s["thr"]).astype(np.uint8) * 255)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    if s["order"] == "maxmin":
        m = cv2.dilate(m, k(s["r1"]))
        m = cv2.erode(m, k(s["r2"]))
    else:
        m = cv2.erode(m, k(s["r1"]))
        m = cv2.dilate(m, k(s["r2"]))
    return m >= 128


# ============================================================================
# v5 -- base white-track chain (plan v15): hard/soft white masks, Canny + soft-gradient
# barriers, edge-touching paper-white seed rule, substantial-overlap extent, per-pixel
# >=250 gating, repair_frame_interiors protection, leak-detector subtraction.
# Track 1 micro band classifier hook: OFF by default (honest negative, plan v15).
# ============================================================================

T_FRAC250 = 0.90
T3_EDGE = 0.01
OVERLAP_PX = 5000
OVERLAP_FRAC = 0.2
EXTENT_WHITE = 250
CANNY_LO, CANNY_HI = 60, 120
SOFT_GRAD_SIGMA = 8.0
SOFT_GRAD_T = 1.0
DEFAULT_CKPT = None  # Track 1 hook off by default (honest negative, see v15 record)


def clean_page_v5(rgb_u8: np.ndarray, ckpt: Path | None = DEFAULT_CKPT) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)
    gray_light = (rgb.max(axis=2) + rgb.min(axis=2)) / 2.0
    gray = np.round(gray_light).astype(np.uint8)
    W = gray.shape[1]

    sob = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3) ** 2
                   + cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3) ** 2)
    edge = sob > 30
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    hard = build_mask(gray_light, SETTINGS["hard-white"])
    soft = build_mask(gray_light, SETTINGS["soft-white"])
    barrier = cv2.morphologyEx(cv2.Canny(gray, CANNY_LO, CANNY_HI), cv2.MORPH_CLOSE, k3) > 0
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), SOFT_GRAD_SIGMA)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, 3)
    softgrad = np.sqrt(gx * gx + gy * gy) > SOFT_GRAD_T
    soft_m = soft & ~barrier
    hard_cc = hard & ~(barrier | softgrad)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(hard_cc.astype(np.uint8), 4)
    seed = np.zeros(gray.shape, dtype=bool)
    for lbl in range(1, num):
        x, y, cw, chh, area = stats[lbl]
        if area < 1000 or not (x == 0 or x + cw == W):
            continue
        comp = labels[y : y + chh, :] == lbl
        g = gray[y : y + chh, :][comp]
        if float((g >= 250).mean()) >= T_FRAC250 or \
           float(edge[y : y + chh, :][comp].mean()) <= T3_EDGE:
            seed[y : y + chh, :] |= comp & (gray[y : y + chh, :] >= 250)

    num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(soft_m.astype(np.uint8), 4)
    overlap = np.bincount(labels_s[seed & soft_m], minlength=num_s)
    sel = [l for l in range(1, num_s)
           if overlap[l] >= OVERLAP_PX and overlap[l] >= OVERLAP_FRAC * stats_s[l, 4]]
    extent = np.isin(labels_s, np.array(sel)) if sel else np.zeros_like(seed)

    delete = seed | (extent & (gray >= EXTENT_WHITE))

    # Track 1 hook (opt-in only; off by default -- see v15 record)
    bands = find_bands(gray) if ckpt is not None else []
    if bands and ckpt.exists():
        net = load_band_net(ckpt)
        is_gutter = classify_bands(gray, bands, net)
        for (b0, b1), g in zip(bands, is_gutter):
            if not g:
                delete[b0 + 2 : b1 - 2, :] = False

    delete = repair_frame_interiors(rgb_u8, delete, frame_darkness=40,
                                     min_interior_px=10000, inset_px=2)
    leak, _ = detect_leaks(gray, delete, seed=None)
    return delete & ~leak


# ============================================================================
# v6 -- three post-chain steps for the user-found defect classes (plan v16):
#   A border-residue sweep [240,250), B under-frame gray line [200,230],
#   C local background reclaim with adversarial guards (protected interiors,
#   area cap + thin-line exemption, outer-ring context test).
# ============================================================================

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


def clean_page_v6(rgb_u8: np.ndarray) -> np.ndarray:
    delete = clean_page_v5(rgb_u8)
    rgb = rgb_u8.astype(np.float32)
    gray = np.round((rgb.max(axis=2) + rgb.min(axis=2)) / 2.0).astype(np.uint8)
    protected = _protected_interiors(gray)
    delete = step_c_local_reclaim(gray, delete, protected)
    delete = step_a_border_residue(gray, delete)
    delete = step_b_under_frame_line(gray, delete)
    return delete


# ============================================================================
# v7 -- step E (frame-strip fix, SHIPPED as default) + documented-negative steps
# D (local color sweep) and F (edge-margin gap pockets), default off (plan v19).
# ============================================================================

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


def clean_page_v7(rgb_u8: np.ndarray, steps: str = "E") -> np.ndarray:
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


# ============================================================================
# v8 -- the verified spiky-cloud ACTION as a region-parameterized utility (manual/GUI
# track), documented-negative automatic-scope variants (step G, off), and step A'
# (widened border-residue band [230,256), sandwich geometry) as the new default (plan v21).
# ============================================================================

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
    """DOCUMENTED NEGATIVE (default off): automatic-scope variants a2/a3 -- see the v21
    record. Retained for the record; do not enable in production."""
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


def clean_page_v8(rgb: np.ndarray, steps: str = "P") -> np.ndarray:
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


# ============================================================================
# v9 -- OPT-IN dark-backdrop track (steps='D', plan v22): flat dark px connected to page
# side margins or deleted background; full-auto was an honest negative (dark-art parts
# regress), so a human marks backdrop-bearing parts.
# ============================================================================

DARK_T = 64          # soft candidate threshold (the user's mask-soft T)
FLAT_STD = 2.0       # local flatness gate (std over 21x21)
FLAT_WIN = 21


def step_h_dark_backdrop(rgb: np.ndarray, gray: np.ndarray, delete: np.ndarray,
                          protected: np.ndarray | None = None) -> np.ndarray:
    """OPT-IN dark-backdrop deletion (attempt-3 rule): flat dark px connected to the page
    side margins or already-deleted background, protected interiors excluded. Apply only
    to parts/pages a human marked as backdrop-bearing."""
    if protected is None:
        protected = _protected_interiors(gray)
    g32 = gray.astype(np.float32)
    mu = cv2.blur(g32, (FLAT_WIN, FLAT_WIN))
    var = cv2.blur(g32 * g32, (FLAT_WIN, FLAT_WIN)) - mu * mu
    flat = np.sqrt(np.maximum(var, 0)) <= FLAT_STD
    dark = (gray <= DARK_T) & ~protected & flat
    seed = np.zeros_like(dark)
    seed[:, :3] = True
    seed[:, -3:] = True
    seed |= cv2.dilate(delete.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return delete | _flood(seed, dark)


def clean_page_v9(rgb: np.ndarray, steps: str = "") -> np.ndarray:
    """v9 default == v8 exactly (which defaults to v7 + A'). steps='D' opts in the
    dark-backdrop track for pages a human marked as backdrop-bearing."""
    delete = clean_page_v8(rgb)
    if "D" in steps:
        f = rgb.astype(np.float32)
        gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
        delete = step_h_dark_backdrop(rgb, gray, delete)
    return delete


# ============================================================================
# v10 -- AUTO-SCOPED spiky-cloud reclaim (steps='S', plan v23): gap-sealed enclosure
# candidates filtered by the two-signal cascade (radial rim runs >= 50 AND interior
# glyphs >= 5; 13/13 TP, 0/85 FP on the reference set), the verified action applied
# with panel-line clipping.  << PRODUCTION until v12 is confirmed >>
# ============================================================================

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


def clean_page_v10(rgb: np.ndarray, steps: str = "") -> np.ndarray:
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


# ============================================================================
# v11 -- frame-protected spiky action (plan v24): +-FRAME_BAND px protection band around
# every detected frame run, excluded from the spiky deletion; step Q (protected-interior
# restore) becomes part of the default.
# ============================================================================

FRAME_BAND = 3  # px above/below a detected frame run protected from the spiky deletion


def clean_spiky_region_frameguard(rgb: np.ndarray, delete: np.ndarray,
                                   bbox: tuple[int, int, int, int],
                                   protected: np.ndarray | None = None) -> np.ndarray:
    """v10's clipped action + frame-run protection band (v24 attempt 1)."""
    x0, y0, x1, y1 = bbox
    before = delete
    after = clean_spiky_region_clipped(rgb, before, bbox, protected=protected)
    changed = after != before
    if not changed.any():
        return after
    win = (slice(y0, y1), slice(x0, x1))
    f = rgb[win].astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    dark = (gray <= 100).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (101, 1))
    runs = cv2.dilate(cv2.erode(dark, hk), hk)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * FRAME_BAND + 1))
    run_band = cv2.dilate(runs, vk) > 0
    out = after.copy()
    reg = out[win]
    undo = changed[win] & run_band
    reg[undo] = before[win][undo]
    out[win] = reg
    return out


def clean_page_v11(rgb: np.ndarray, steps: str = "Q") -> np.ndarray:
    """v11 default = v10 default + step Q (protected-interior restore). steps='S' =
    frame-protected spiky action; 'D' = the v9 opt-in dark-backdrop track; combinable.

    Step Q (v24 issue 2): undelete everything inside `_protected_interiors` — the 10-part
    audit measured 25,208 px of frame-interior over-deletion (A' 11,212 / earlier steps
    13,996) vs only 3,122 px of GT-legitimate in-interior deletion (one thin empty box on
    002-2 whose restore is white-on-white, cosmetically nil). Protection exists; steps
    were bypassing it (thin frame lines let dilation-based conditions cross) — enforcing
    it at the end is the reuse the audit validated.
    """
    f = rgb.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    delete = clean_page_v10(rgb, steps="D" if "D" in steps else "")
    prot = _protected_interiors(gray)
    if "Q" in steps:
        delete = delete & ~prot
    if "S" in steps:
        for bbox in find_spiky_sites(rgb):
            delete = clean_spiky_region_frameguard(rgb, delete, bbox, protected=prot)
    return delete


# ============================================================================
# v12 -- the CANDIDATE (plan v26 fix round):
#   Fix A: `_protected_interiors_v2` closed-contour ownership test (>=90% of the hole's
#          ink-adjacent boundary owned by ONE un-closed stroke component -- a real frame
#          contour; inter-panel gutters fail and stop being mis-protected).
#   Fix B: occlusion bridging of collinear frame-run segments in the S-action clip.
#   Fix E: elliptical scope (<= ELLIPSE_MAX of the bbox-inscribed ellipse).
#   Fix S: saturation gate (soup is achromatic).
#   Fix R (v27 attempt B3, off): ring-distance gate -- retained as a documented attempt.
# FIX_* module flags exist ONLY for the comparison harness (see apply_config below).
# ============================================================================

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


def apply_config(letters: str) -> None:
    """Set the FIX_* comparison-harness flags from a letter string, e.g. "A", "AB",
    "ABE", "ABES" (the shipped default), "ABESR". Replaces the old v12_cfg* wrappers."""
    global FIX_A, FIX_B, FIX_E, FIX_S, FIX_R
    FIX_A = "A" in letters
    FIX_B = "B" in letters
    FIX_E = "E" in letters
    FIX_S = "S" in letters
    FIX_R = "R" in letters


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


def clean_page_v12(rgb: np.ndarray, steps: str = "Q") -> np.ndarray:
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


# Canonical entry point: the v12 candidate with the shipped ABES defaults.
clean_page = clean_page_v12


ENTRIES = {"v5": clean_page_v5, "v6": clean_page_v6, "v7": clean_page_v7,
           "v8": clean_page_v8, "v9": clean_page_v9, "v10": clean_page_v10,
           "v11": clean_page_v11, "v12": clean_page_v12}


if __name__ == "__main__":
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    steps = sys.argv[sys.argv.index("--steps") + 1] if "--steps" in sys.argv else None
    entry = sys.argv[sys.argv.index("--entry") + 1] if "--entry" in sys.argv else "v12"
    if "--config" in sys.argv:
        apply_config(sys.argv[sys.argv.index("--config") + 1])
    fn = ENTRIES[entry]
    rgb = np.asarray(Image.open(src_path).convert("RGB"))
    mask = fn(rgb, steps) if steps is not None else fn(rgb)
    np.save(out_path, mask)
    print(f"deleted {mask.mean():.4f} of page -> {out_path}")
