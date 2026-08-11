"""SFX reference-set harness: labels + per-candidate features on the 6 decoded PSDs
(.tmp/sfx_decode/export/*.npz, produced by sfx_decode.py). Gen-8 sfx_glyph step 3.

Purpose: measure which geometric features actually separate SFX glyph strokes from the
other dark structures that share the gutter (bubble outlines, bubble text, frame border
lines) BEFORE choosing profile signals -- same evidence-first order as v20/v23.

Sample-size honesty: 6 reference crops, ~30 labelable gutter components. Numbers here
are smoke evidence, not validated rates; full-page FP counts on suite chapters 002/019
are the meaningful precision check for whatever profile ships.

Labeling (auto, verified against the decode overlays):
  frame_line  -- near-full-span thin straight component
  bubble_part -- component intersecting an enclosed-hole region (any class incl.
                 frame_* -- the crop canvases misroute big bubbles page-relatively,
                 the cloud_suite 8.3.1 documented artifact) or its enclosing outline
  sfx         -- remaining gutter component KEPT in GT (>=50% of its ink)
  deleted     -- remaining gutter component deleted in GT (negatives)
In-frame components are excluded by the gutter mask entirely (case 3 = frame-keep's
job, per the recipe: no SFX logic inside frames).

Gutter mask: flood from canvas border over non-frame-line px, lines = the phase-2
VALIDATED inventory (detect_lines_morph + bridge_collinear) drawn with their measured
thickness. Interior (unreached) px belong to panels. This is the conservative
"between confirmed lines" reading, not the unvalidated rect grouping.

Usage:  .venv/bin/python src/classifiers/tests/sfx_suite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/spiky"))

from style_analysis import extract_enclosed_holes  # noqa: E402
from classifiers.frame import bridge_collinear, detect_lines_morph  # noqa: E402

EXPORT = REPO / ".tmp/sfx_decode/export"
STEMS = ["004", "004_1", "004_2", "004_3", "004_4", "005"]

CUT_AGGR = 33   # solved 8.6.1: aggressive pass = G >= 33, constant across all 6 refs
MIN_COMP = 120  # px: below this a gutter component is noise, not a labelable object

# Manual reference annotation: panel rects per crop (x0, y0, x1, y1), anchored to the
# VALIDATED line inventory's border-line positions and verified against the decode
# overlays. Used for LABELING only -- the flood-based automatic gutter mask leaks on
# 3-4 of the 6 crops (bubbles sitting on border runs at the canvas edge, white steam
# interrupting the dark line) and would contaminate the 'sfx' label with in-frame art.
# Automatic interior detection stays a production (sfx.py) concern, measured
# separately against these annotations.
FRAME_RECTS = {
    "004":   [(29, 377, 529, 912)],
    "004_1": [(73, 153, 547, 716)],
    "004_2": [(0, 189, 690, 681)],
    "004_3": [(0, 141, 690, 776)],
    "004_4": [(0, 172, 690, 771)],
    "005":   [(0, 199, 690, 848)],
}


def frame_interior_frac(bbox, lab_mask, rects) -> float:
    """Fraction of a component's px inside any annotated panel rect."""
    ys, xs = np.nonzero(lab_mask)
    if not len(ys):
        return 0.0
    inside = np.zeros(len(ys), bool)
    for x0, y0, x1, y1 in rects:
        inside |= (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    return float(inside.mean())


def load_ref(stem: str):
    z = np.load(EXPORT / f"{stem}.npz")
    return {k: z[k] for k in z.files}


def gutter_mask(rgb: np.ndarray) -> np.ndarray:
    """True = gutter (outside every panel). Flood from the canvas border over px not
    covered by a detected page-scale line (drawn at measured thickness)."""
    f = rgb.astype(np.float32)
    gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
    h_lines, v_lines = detect_lines_morph(gray)
    h_lines, v_lines = bridge_collinear(h_lines), bridge_collinear(v_lines)
    H, W = gray.shape
    barrier = np.zeros((H, W), np.uint8)
    for ln in h_lines:
        t = max(1, ln.thick)
        y0 = max(0, ln.pos - t // 2 - 1); y1 = min(H, ln.pos + t // 2 + 2)
        barrier[y0:y1, ln.span[0]:ln.span[1] + 1] = 1
    for ln in v_lines:
        t = max(1, ln.thick)
        x0 = max(0, ln.pos - t // 2 - 1); x1 = min(W, ln.pos + t // 2 + 2)
        barrier[ln.span[0]:ln.span[1] + 1, x0:x1] = 1
    free = (barrier == 0).astype(np.uint8)
    ff = free.copy()
    mask = np.zeros((H + 2, W + 2), np.uint8)
    for seed in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1),
                 (W // 2, 0), (W // 2, H - 1), (0, H // 2), (W - 1, H // 2)]:
        if ff[seed[1], seed[0]] == 1:
            cv2.floodFill(ff, mask, seed, 2)
    return ff == 2


def comp_features(ink: np.ndarray, lab: np.ndarray, i: int, stats_row) -> dict:
    """Geometry of one ink component: stroke width stats, skeleton-ish elongation,
    isolation ring ink density."""
    x, y, w, h = (int(stats_row[cv2.CC_STAT_LEFT]), int(stats_row[cv2.CC_STAT_TOP]),
                  int(stats_row[cv2.CC_STAT_WIDTH]), int(stats_row[cv2.CC_STAT_HEIGHT]))
    a = int(stats_row[cv2.CC_STAT_AREA])
    H, W = ink.shape
    pad = 30
    wy0, wy1 = max(0, y - pad), min(H, y + h + pad)
    wx0, wx1 = max(0, x - pad), min(W, x + w + pad)
    comp = (lab[wy0:wy1, wx0:wx1] == i)
    other_ink = ink[wy0:wy1, wx0:wx1] & ~comp
    dc = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
    dvals = dc[dc > 0]
    w_med = float(np.median(dvals) * 2.0) if dvals.size else 0.0
    w_p90 = float(np.quantile(dvals, 0.90) * 2.0) if dvals.size else 0.0
    # width CV: spread of local width along the body (ridge px approx: local maxima)
    w_cv = float(dvals.std() / dvals.mean()) if dvals.size and dvals.mean() > 0 else 0.0
    # elongation: area / w_med^2 ~ (stroke length / width); blobs ~O(1), strokes >> 1
    elong = a / (w_med * w_med) if w_med > 0 else 0.0
    # isolation: ink density (any ink, incl. other comps) in ring 1..R outside comp
    R = int(max(6, 2 * w_med))
    dout = cv2.distanceTransform((~comp).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (dout > 0) & (dout <= R)
    iso = float(other_ink[ring].mean()) if ring.any() else 0.0
    return dict(area=a, bbox=(x, y, w, h), w_med=w_med, w_p90=w_p90, w_cv=w_cv,
                elong=elong, iso_ink=iso, ring_r=R)


def label_and_measure(stem: str, verbose=True):
    ref = load_ref(stem)
    rgb, gt_del = ref["raw"], ref["gt_delete"]
    H, W = gt_del.shape
    gut = gutter_mask(rgb)
    ink = rgb[..., 1] < CUT_AGGR  # solved aggressive predicate
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), connectivity=8)
    # enclosure regions (bubbles incl. crop-misrouted frame_* classes)
    holes = extract_enclosed_holes(rgb)
    hole_boxes = []
    for s in holes:
        hx, hy, hw, hh = s["bbox"]
        hole_boxes.append((hx - 6, hy - 6, hx + hw + 6, hy + hh + 6, s["class"]))
    keep = ~gt_del
    rects = FRAME_RECTS[stem]
    rows = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < MIN_COMP:
            continue
        comp = lab == i
        gut_frac = float(gut[comp].mean())
        in_frame = frame_interior_frac(None, comp, rects)
        if in_frame > 0.7:  # essentially inside a panel -> frame-keep's territory
            continue
        ft = comp_features(ink, lab, i, stats[i])
        x, y, w, h = ft["bbox"]
        kept = float(keep[comp].mean())
        near_border = any(
            min(abs(y - ry0), abs(y + h - ry1)) <= 6 or
            min(abs(x - rx0), abs(x + w - rx1)) <= 6
            for rx0, ry0, rx1, ry1 in rects)
        thin_straight = (min(w, h) <= 4 and max(w, h) / max(1, min(w, h)) > 12
                         and near_border)
        in_hole = any(x >= bx0 and y >= by0 and x + w <= bx1 and y + h <= by1
                      for bx0, by0, bx1, by1, _ in hole_boxes)
        overlaps_hole = any(x < bx1 and x + w > bx0 and y < by1 and y + h > by0
                            and a > 0 for bx0, by0, bx1, by1, _ in hole_boxes)
        if thin_straight:
            label = "frame_line"
        elif in_hole or overlaps_hole:
            label = "bubble_part"
        elif kept >= 0.5:
            label = "sfx"
        else:
            label = "deleted"
        rows.append(dict(label=label, kept=kept, gut_frac=gut_frac, **ft))
    if verbose:
        print(f"=== {stem}  gutter={gut.mean()*100:.0f}% of canvas  "
              f"holes={len(hole_boxes)}")
        for r in sorted(rows, key=lambda r: (r["label"], -r["area"])):
            print(f"  {r['label']:11} area={r['area']:6d} bbox={r['bbox']} "
                  f"kept={r['kept']*100:3.0f}% w_med={r['w_med']:4.1f} "
                  f"w_p90={r['w_p90']:4.1f} w_cv={r['w_cv']:.2f} "
                  f"elong={r['elong']:6.1f} iso={r['iso_ink']*100:4.1f}%")
    return rows


def eval_profile(verbose=True):
    """Run the sfx_glyph profile against the labeled references: recall on 'sfx'
    components, plus every non-sfx detection listed (ref-level FP evidence -- note
    the auto gutter mask leaks into panels on some refs, so in-frame art DOES enter
    the candidate pool here; a detection on it is a genuine precision failure)."""
    from classifiers.detector_framework import detect
    from classifiers.profiles.sfx_glyph import PROFILE

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
        ix = max(0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
        return inter / ua if ua else 0.0

    tot_sfx = tot_hit = tot_fp = tot_harm = 0
    for stem in STEMS:
        ref = load_ref(stem)
        rows = label_and_measure(stem, verbose=False)
        gt_boxes = [(r["bbox"][0], r["bbox"][1],
                     r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3])
                    for r in rows if r["label"] == "sfx"]
        other_boxes = [(r["bbox"][0], r["bbox"][1],
                        r["bbox"][0] + r["bbox"][2], r["bbox"][1] + r["bbox"][3],
                        r["label"]) for r in rows if r["label"] != "sfx"]
        det = detect(ref["raw"], PROFILE)
        hit = sum(1 for g in gt_boxes if any(iou(g, d) >= 0.5 for d in det))
        fp = [d for d in det if not any(iou(g, d) >= 0.5 for g in gt_boxes)]
        # pixel-harm split: an extra on GT-KEPT ink is harmless in the composition
        # (region kept anyway; halo lands inside kept context); an extra on GT-DELETED
        # ink would wrongly keep deleted structure + halo = real harm.
        keep = ~ref["gt_delete"]
        ink = ref["raw"][..., 1] < 33
        harmful = []
        for d in fp:
            x0, y0, x1, y1 = d
            m = ink[y0:y1, x0:x1]
            if m.any() and float(keep[y0:y1, x0:x1][m].mean()) < 0.5:
                harmful.append(d)
        tot_sfx += len(gt_boxes); tot_hit += hit; tot_fp += len(fp)
        tot_harm += len(harmful)
        if verbose:
            print(f"  {stem}: sfx recall {hit}/{len(gt_boxes)}, extra detections "
                  f"{len(fp)} (harmful: {len(harmful)})")
            for d in harmful:
                print(f"    HARMFUL extra {d}")
    print(f"PROFILE on refs: recall {tot_hit}/{tot_sfx}, extras {tot_fp}, "
          f"HARMFUL extras (on GT-deleted ink) {tot_harm}")
    return tot_hit, tot_sfx, tot_fp, tot_harm


def prototype_eval(verbose=True):
    """sfx.py acceptance: pixel agreement vs PSD GT per reference file + the HARD
    zero-frame-content-loss guard (no deleted px inside the annotated frame rects)."""
    from classifiers.sfx import clean_sfx_region
    print("file        over-del%   over-keep%   frame-loss-px")
    worst = 0.0
    total_frame_loss = 0
    for stem in STEMS:
        ref = load_ref(stem)
        delete = clean_sfx_region(ref["raw"])
        gt = ref["gt_delete"]
        n = gt.size
        over_del = float((delete & ~gt).mean()) * 100   # we delete, GT keeps
        over_keep = float((~delete & gt).mean()) * 100  # we keep, GT deletes
        floss = 0
        for x0, y0, x1, y1 in FRAME_RECTS[stem]:
            floss += int(delete[y0:y1 + 1, x0:x1 + 1].sum())
        total_frame_loss += floss
        worst = max(worst, over_del)
        print(f"{stem:10}  {over_del:8.3f}   {over_keep:9.3f}   {floss}")
    print(f"HARD GUARD frame-loss px total: {total_frame_loss} "
          f"({'PASS' if total_frame_loss == 0 else 'FAIL'})")
    return total_frame_loss


if __name__ == "__main__":
    allrows = []
    for stem in STEMS:
        allrows += label_and_measure(stem)
        print()
    import collections
    print("=== feature ranges by label ===")
    by = collections.defaultdict(list)
    for r in allrows:
        by[r["label"]].append(r)
    for label, rs in sorted(by.items()):
        for key in ("area", "w_med", "w_cv", "elong", "iso_ink"):
            v = sorted(r[key] for r in rs)
            print(f"  {label:11} {key:8} n={len(v):2d} "
                  f"min={v[0]:8.2f} med={v[len(v)//2]:8.2f} max={v[-1]:8.2f}")
        print()
