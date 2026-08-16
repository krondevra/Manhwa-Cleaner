"""Gen9 v2 background-area element adjudication (algorithm steps 27-43).

Everything here operates ONLY on elements in the background zone; frames
are already hard-locked when these run (PageState enforces it).

Three decisions, measured against the 006-crop checkpoint PSDs
(decisions.md 2026-08-14):

S3 specks (steps 27-29, "MinMax-defect compensation"): the operator
wand-clicks bright defect remnants floating in the cleaned field. The
wand works on the VISIBLE COMPOSITE -- deleted px show the red fill and
never qualify, so each defect is a contained island. Photopea tolerance
140 vs the clicked white == Chebyshev distance: min channel >= 115
(measured: GT speck tail bottoms out at 116). GT click coverage is
human-imperfect (one 217 px twin sits 3 px from a clicked speck,
unclicked), so the gate reports FN px + beyond-etalon extras instead of
forcing diff 0.

B' SFX outlines (steps 32-43): candidates = SFX-layer black comps in the
bg zone (>= CAND_BG_FRAC of px within NEAR_BG of the deleted field --
panel-interior art never qualifies). SFX iff compact (bbox <= BBOX_MAX)
AND not enclosing a large white interior (<= INTERIOR_MAX). Measured:
15/15 GT stroke comps; caption box (interior 16k) and bubble border
(interior 36-56k) rejected by interior; frame borders and spiky ring
rejected by bbox. Size alone does NOT separate caption from glyph.

C trapped pockets (steps 30-31): background sealed inside a selected SFX
outline = holes of the comp; deleted px = hole AND wand-qualifying AND
currently kept. Measured: 1,640/1,642 GT px (2 px = manual wand AA
floor). Negative interiors (caption 16k, bubble 45k, spiky 48k) never
appear -- they belong to comps B' rejects.
"""
from __future__ import annotations

import cv2
import numpy as np

WAND_TOL = 140            # Photopea magic-wand tolerance (Chebyshev)
QUAL_MIN = 255 - WAND_TOL  # min channel >= this qualifies vs white click
SPECK_MAX = 300           # defect specks are small blobs; border-AA
                          # chains are long and exceed this
SPECK_ADJ = 2             # speck must sit within this of the field
INK_SEAL_MAX = 0.80       # comps whose 2px ring is more SFX-ink than
                          # this are SEALED interiors (thought-bubble
                          # chain circles: 0.85-0.92) -- content, kept;
                          # true bg slivers measure 0.66-0.76
NEAR_BG = 16              # bg-zone membership band (max GT dist 12)
MIN_AREA = 40             # SFX comp minimum (GT min 43)
CAND_BG_FRAC = 0.5        # comp px fraction inside the bg-zone band
BBOX_MAX = 250            # compact rule (spiky ring 346-478 fails)
INTERIOR_MAX = 8000       # enclosed-white ceiling (pocket 1.6k passes,
                          # caption 16k / bubble 36k+ fail)
HOLE_RATIO_MAX = 1.0      # an SFX outline is more ink than enclosed
                          # white (GT max 0.50); bubble-chain circles
                          # (1.49), caption (15), bubble (29) are more
                          # hole than ink -> content, kept
POCKET_MIN = 30
POCKET_MAX = 8000

# Photopea Select>Modify>Expand>4px == octagon radius 4 (Chebyshev+
# Manhattan mix). Kernel ladder vs the before-44 delta: octagon/Euclid4.5
# missed 0 / extra 212; disk9 missed 1,430; sq9 extra 2,221.
EXPAND4 = np.ones((9, 9), np.uint8)
for _i in range(9):
    for _j in range(9):
        if abs(_i - 4) + abs(_j - 4) > 6:
            EXPAND4[_i, _j] = 0


def expand_fringe(sel_mask: np.ndarray) -> np.ndarray:
    """S5: the operator's expand-4 halo around selected SFX strokes."""
    return cv2.dilate(sel_mask.astype(np.uint8), EXPAND4).astype(bool)


def _wand_qualifies(src: np.ndarray) -> np.ndarray:
    """Px within WAND_TOL of pure white (Chebyshev over channels)."""
    return src.min(axis=2) >= QUAL_MIN


def _enclosed(comp_u8: np.ndarray) -> np.ndarray:
    """Holes of a component mask (uint8, cropped): complement comps that
    do not touch the crop border."""
    inv = (comp_u8 == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = np.unique(np.concatenate(
        [lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    hole_ids = np.setdiff1d(np.arange(1, n), border)
    return np.isin(lab, hole_ids)


def find_field_specks(src: np.ndarray, deleted: np.ndarray,
                      exclude: np.ndarray | None = None,
                      sfx_ink: np.ndarray | None = None,
                      forbidden: np.ndarray | None = None) -> np.ndarray:
    """S3: bright defect comps floating at the deleted field's edge.

    Returns the px to delete. `exclude` masks regions handled by later
    stages (the spiky carve -- its interstices are S6's job). `sfx_ink`
    (bool, SFX-layer black) enables the sealed-interior rejection.
    `forbidden` (the frame lock): a comp touching it is a frame-attached
    fragment, not a floating defect -- skipped whole (content-safe; on
    the 006 crop no GT speck touches the lock, so this is a no-op there).
    """
    kept = ~deleted
    qual = _wand_qualifies(src) & kept
    if exclude is not None:
        qual &= ~exclude
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        qual.astype(np.uint8), connectivity=8)
    adj = cv2.dilate(deleted.astype(np.uint8),
                     np.ones((2 * SPECK_ADJ + 1,) * 2, np.uint8)).astype(bool)
    ink = sfx_ink if sfx_ink is not None else None
    out = np.zeros(src.shape[:2], bool)
    H, W = src.shape[:2]
    k5 = np.ones((5, 5), np.uint8)
    for i in range(1, n):
        x0, y0, w, h, a = (int(v) for v in st[i][:5])
        if a > SPECK_MAX:
            continue
        ys, xs = max(0, y0 - 3), max(0, x0 - 3)
        ye, xe = min(H, y0 + h + 3), min(W, x0 + w + 3)
        sub = lab[ys:ye, xs:xe] == i
        if not adj[ys:ye, xs:xe][sub].any():
            continue
        if forbidden is not None and forbidden[ys:ye, xs:xe][sub].any():
            continue    # frame-attached fragment, not a floating defect
        if ink is not None:
            sub8 = sub.astype(np.uint8)
            ring = cv2.dilate(sub8, k5).astype(bool) & ~sub
            if ring.any() and float(ink[ys:ye, xs:xe][ring].mean()) >= INK_SEAL_MAX:
                continue    # sealed interior (bubble-chain circle etc.)
        out[ys:ye, xs:xe] |= sub
    return out


def select_sfx_comps(sfx: np.ndarray, deleted: np.ndarray, cf: np.ndarray
                     ) -> tuple[np.ndarray, list[int], list[dict]]:
    """B': SFX-outline comps in the background zone.

    Returns (labels, selected ids, population report rows).
    """
    sfxb = (sfx < 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sfxb, connectivity=8)
    k = np.ones((2 * NEAR_BG + 1,) * 2, np.uint8)
    near = cv2.dilate(deleted.astype(np.uint8), k).astype(bool)
    cfw = cf >= 128
    sel, rows = [], []
    for i in range(1, n):
        x0, y0, w, h, a = (int(v) for v in st[i][:5])
        if a < MIN_AREA:
            continue
        box = lab[y0:y0 + h, x0:x0 + w] == i
        frac = float(near[y0:y0 + h, x0:x0 + w][box].mean())
        if frac < CAND_BG_FRAC:
            continue
        sub = (lab[max(0, y0 - 1):y0 + h + 1,
                   max(0, x0 - 1):x0 + w + 1] == i).astype(np.uint8)
        holes = _enclosed(sub)
        interior = int(holes.sum())
        compact = w <= BBOX_MAX and h <= BBOX_MAX
        ratio = interior / a
        verdict = (compact and interior < INTERIOR_MAX
                   and ratio < HOLE_RATIO_MAX)
        rows.append(dict(comp_id=i, bbox=(x0, y0, x0 + w, y0 + h), area=a,
                         bg_frac=round(frac, 2), interior=interior,
                         hole_ratio=round(ratio, 2), compact=compact,
                         selected=verdict))
        if verdict:
            sel.append(i)
    return lab, sel, rows


def select_pockets(src: np.ndarray, deleted: np.ndarray,
                   sfx_labels: np.ndarray, sfx_sel: list[int],
                   stats_shape_hint=None) -> tuple[np.ndarray, list[dict]]:
    """C: background trapped inside selected SFX outlines -> px to delete."""
    qual = _wand_qualifies(src)
    kept = ~deleted
    out = np.zeros(src.shape[:2], bool)
    rows = []
    for i in sfx_sel:
        ys, xs = np.where(sfx_labels == i)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        sub = (sfx_labels[max(0, y0 - 1):y1 + 2,
                          max(0, x0 - 1):x1 + 2] == i).astype(np.uint8)
        holes = _enclosed(sub)
        na = int(holes.sum())
        if na < POCKET_MIN or na > POCKET_MAX:
            continue
        full = np.zeros(src.shape[:2], bool)
        full[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] = holes
        px = full & qual & kept
        if int(px.sum()) < POCKET_MIN:
            continue
        out |= px
        rows.append(dict(comp_id=int(i), hole_px=na, delete_px=int(px.sum()),
                         bbox=(int(x0), int(y0), int(x1) + 1, int(y1) + 1)))
    return out, rows
