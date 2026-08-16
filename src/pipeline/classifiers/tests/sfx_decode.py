"""Decode the 6 SFX reference PSDs (.tmp/scripts-manual/SFX/) into measured recipe
parameters + ground truth. Gen-8 sfx_glyph unblock, step 1.

Reverse-engineering discipline as v12/v21/v25: the PSD layer data is ground truth over
the written recipe ("SFX Pipeline" prose) wherever they disagree.

File anatomy (verified): small crop canvas (~690 x 800-970) whose layers are FULL-PAGE
images at large negative offsets -- each PSD is a page-crop window around one SFX case.
  img        pixel layer + layer mask  -> raw art (mask NOT applied) and keep/delete GT
             via the validated `.composite()` path (src/spiky/psd_extract.py). GT is
             only meaningful INSIDE the crop canvas; everything here is canvas-scoped.
  img-clone  baked result of the recipe's Levels(a,1,a+1)->Threshold(t) pass(es); the
             adjustment layers were merged, so the parameters are not stored -- but the
             composition collapses to a single effective gray cutoff, which this script
             SOLVES by sweeping cutoffs against gray(img) and matching img-clone.
  img-copy   (005.psd only) a second preserved threshold layer -- decoded the same way.
  red/Layer1 visualization only, ignored.

Outputs: printed per-file report + overlays in .tmp/sfx_decode/ for visual verification.

Usage:  .venv/bin/python src/classifiers/tests/sfx_decode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/spiky"))

SFX_DIR = REPO / ".tmp/scripts-manual/SFX"
OUT_DIR = REPO / ".tmp/sfx_decode"

FILES = ["004.psd", "004(1).psd", "004(2).psd", "004(3).psd", "004(4).psd", "005.psd"]

# Candidate gray reductions for the effective-cutoff solve. Photoshop's Threshold uses
# luminosity; Levels(a,1,a+1) applied per-channel then Threshold(t) is NOT a plain
# luminosity cutoff on colored pixels (it binarizes channels first), so the G-channel
# predicate is included (G's Rec601 weight 0.587 alone clears Threshold 140/255=0.549).
GRAYS = {
    "rec601": lambda rgb: (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
                           + 0.114 * rgb[..., 2]),
    "green": lambda rgb: rgb[..., 1].astype(np.float64),
    "lightness": lambda rgb: (rgb.max(axis=2).astype(np.float64)
                              + rgb.min(axis=2)) / 2.0,
    "mean": lambda rgb: rgb.mean(axis=2),
    "min": lambda rgb: rgb.min(axis=2).astype(np.float64),
    "max": lambda rgb: rgb.max(axis=2).astype(np.float64),
}


def crop_layer_to_canvas(layer, W: int, H: int, fill: int = 255) -> np.ndarray:
    """Layer's raw pixels (mask NOT applied) in canvas coordinates, padded with
    `fill` where the layer doesn't cover the canvas."""
    raw = np.array(layer.topil().convert("RGB"))
    ox, oy = layer.offset
    out = np.full((H, W, 3), fill, dtype=np.uint8)
    sy0, sy1 = max(0, -oy), min(raw.shape[0], H - oy)
    sx0, sx1 = max(0, -ox), min(raw.shape[1], W - ox)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 + oy:sy1 + oy, sx0 + ox:sx1 + ox] = raw[sy0:sy1, sx0:sx1]
    return out


def gt_delete_mask(img_layer, W: int, H: int) -> np.ndarray:
    """Keep/delete GT from the img layer's properly composited alpha (pixel alpha *
    layer mask at the layer's canvas offset) -- the psd_extract.py-validated path.
    True = delete. Canvas-scoped viewport."""
    rgba = np.array(img_layer.composite(viewport=(0, 0, W, H)).convert("RGBA"))
    return rgba[..., 3] < 128


def solve_cutoff(rgb: np.ndarray, bw_white: np.ndarray):
    """Find (gray_name, cutoff, match_fraction) best reproducing `bw_white`
    (bool, True = white in the baked threshold layer) as gray(rgb) >= cutoff."""
    best = ("?", 0, -1.0)
    n = bw_white.size
    for name, fn in GRAYS.items():
        g = fn(rgb)
        # candidate cutoffs: only values that occur matter; sweep 1..255 is cheap
        # via histogram: match(t) = |white & g>=t| + |black & g<t|
        hw = np.histogram(g[bw_white], bins=256, range=(0, 256))[0]
        hb = np.histogram(g[~bw_white], bins=256, range=(0, 256))[0]
        # for cutoff t: whites correctly classified = sum hw[t:], blacks = sum hb[:t]
        cw = np.concatenate([[hw.sum()], hw.sum() - np.cumsum(hw)])  # cw[t]=sum hw[t:]
        cb = np.concatenate([[0], np.cumsum(hb)])                     # cb[t]=sum hb[:t]
        match = (cw[:256] + cb[:256]) / n
        t = int(match.argmax())
        if match[t] > best[2]:
            best = (name, t, float(match[t]))
    return best


def mismatch_components(rgb, bw_white, gray_name, cutoff, min_area=30):
    g = GRAYS[gray_name](rgb)
    pred = g >= cutoff
    mm = (pred != bw_white).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
    comps = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area:
            comps.append((a, (int(stats[i, cv2.CC_STAT_LEFT]),
                              int(stats[i, cv2.CC_STAT_TOP]),
                              int(stats[i, cv2.CC_STAT_WIDTH]),
                              int(stats[i, cv2.CC_STAT_HEIGHT]))))
    comps.sort(reverse=True)
    return int(mm.sum()), comps, mm.astype(bool)


D_CAP = 25  # px: annulus profile range; a real per-object Expand is far below this


def expand_analysis(clone_white: np.ndarray, gt_delete: np.ndarray, min_area=100):
    """Per SFX ink component in the threshold layer: how far does the GT KEEP region
    extend beyond the ink (the manual per-object Expand E)?

    Two independent estimators (cross-checked):
      E_prof -- annulus keep-profile: r(d) = kept fraction of WHITE px at distance
                (d-1,d] from the component, restricted to px NEARER this component
                than any other ink (so neighboring objects' halos don't pollute);
                E_prof = last d with r(1..d) all >= 0.5. Saturation at D_CAP means
                the component sits inside a wholesale-kept region (frame interior),
                not that a huge Expand was used -- reported as 'keep-all'.
      E_del  -- median, over the component's immediate outer white ring, of the
                distance to the nearest DELETED px (for an isolated dilated-by-E
                object this is ~E; robust to partial frame adjacency via median).
    Plus the component's own stroke width (median 2x distance transform)."""
    ink = (~clone_white).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = ~gt_delete
    # distance to nearest deleted px (0 on deleted px themselves)
    d_del = cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 3)
    H, W = ink.shape
    rows = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        comp = lab == i
        kept_ink = float(keep[comp].mean())
        pad = D_CAP + 15
        wy0, wy1 = max(0, y - pad), min(H, y + h + pad)
        wx0, wx1 = max(0, x - pad), min(W, x + w + pad)
        wlab = lab[wy0:wy1, wx0:wx1]
        d_comp = cv2.distanceTransform((wlab != i).astype(np.uint8), cv2.DIST_L2, 3)
        # d_other: distance to any OTHER ink; computed on mask that is 1 where
        # not-other-ink (own comp px and non-ink px), 0 on other ink
        d_other = cv2.distanceTransform(
            ((wlab == i) | (wlab == 0)).astype(np.uint8), cv2.DIST_L2, 3)
        wwhite = clone_white[wy0:wy1, wx0:wx1]
        wkeep = keep[wy0:wy1, wx0:wx1]
        own = wwhite & (d_comp > 0) & (d_comp < d_other)  # white px owned by this comp
        prof = []
        for d in range(1, D_CAP + 1):
            ring = own & (d_comp > d - 1) & (d_comp <= d)
            prof.append(float(wkeep[ring].mean()) if ring.any() else np.nan)
        e_prof = 0
        for d, r in enumerate(prof, start=1):
            if not np.isnan(r) and r >= 0.5:
                e_prof = d
            else:
                break
        # E_del over the immediate outer ring
        ring1 = own & (d_comp <= 1.5)
        e_del = float(np.median(d_del[wy0:wy1, wx0:wx1][ring1])) if ring1.any() else 0.0
        dcomp_in = cv2.distanceTransform((wlab == i).astype(np.uint8), cv2.DIST_L2, 3)
        width = (float(np.median(dcomp_in[dcomp_in > 0]) * 2.0)
                 if (dcomp_in > 0).any() else 0.0)
        rows.append(dict(area=a, bbox=(x, y, w, h), kept_ink=kept_ink,
                         e_prof=e_prof, e_del=e_del, saturated=e_prof >= D_CAP,
                         stroke_w=width, prof=prof))
    rows.sort(key=lambda r: -r["area"])
    return rows


def two_layer_diff(passes: dict, gt_delete: np.ndarray):
    """005-style two-pass files: where do the aggressive (img-clone) and preservation
    (img-copy) layers differ, and which one agrees with the GT keep there?"""
    if "img-copy" not in passes:
        return None
    w1 = passes["img-clone"]["white"]  # aggressive
    w2 = passes["img-copy"]["white"]   # preservation
    diff = w1 != w2
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        diff.astype(np.uint8), connectivity=8)
    keep = ~gt_delete
    comps = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 30:
            continue
        m = lab == i
        comps.append(dict(
            area=a,
            bbox=(int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                  int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])),
            # px white in aggressive but ink in preservation = what pass 2 rescues
            rescued=int((m & w1 & ~w2).sum()),
            kept=float(keep[m].mean())))
    comps.sort(key=lambda c: -c["area"])
    return dict(diff_px=int(diff.sum()), comps=comps, diff_mask=diff)


def decode_file(name: str, verbose=True):
    from psd_tools import PSDImage
    psd = PSDImage.open(SFX_DIR / name)
    W, H = psd.size
    layers = {ly.name: ly for ly in psd}
    img = layers["img"]
    raw = crop_layer_to_canvas(img, W, H)
    gt_del = gt_delete_mask(img, W, H)
    out = dict(name=name, size=(W, H), raw=raw, gt_delete=gt_del, passes={})
    for lname in ("img-clone", "img-copy"):
        if lname not in layers:
            continue
        bw = crop_layer_to_canvas(layers[lname], W, H)
        white = bw.mean(axis=2) >= 128  # baked layer is pure B/W; midpoint is safe
        gname, cutoff, frac = solve_cutoff(raw, white)
        mm_px, mm_comps, mm_mask = mismatch_components(raw, white, gname, cutoff)
        out["passes"][lname] = dict(white=white, gray=gname, cutoff=cutoff,
                                    match=frac, mm_px=mm_px, mm_comps=mm_comps,
                                    mm_mask=mm_mask)
    # expand analysis against the AGGRESSIVE pass (img-clone = the mask source layer)
    out["expand"] = expand_analysis(out["passes"]["img-clone"]["white"], gt_del)
    out["diff2"] = two_layer_diff(out["passes"], gt_del)
    if verbose:
        print(f"=== {name}  canvas={W}x{H}  gt_delete={gt_del.mean()*100:.1f}% of px")
        for lname, p in out["passes"].items():
            print(f"  {lname}: gray={p['gray']} cutoff>={p['cutoff']} "
                  f"match={p['match']*100:.3f}%  mismatch_px={p['mm_px']}"
                  f"  comps>=30px: {len(p['mm_comps'])}"
                  + (f" largest={p['mm_comps'][0]}" if p['mm_comps'] else ""))
        for r in out["expand"][:14]:
            tag = "KEEP-ALL" if r["saturated"] else f"E_prof={r['e_prof']:2d}"
            print(f"    ink comp area={r['area']:6d} bbox={r['bbox']} "
                  f"kept={r['kept_ink']*100:5.1f}% stroke_w={r['stroke_w']:4.1f} "
                  f"{tag} E_del={r['e_del']:.1f}")
        d2 = out.get("diff2")
        if d2:
            print(f"  two-pass diff: {d2['diff_px']} px, comps>=30px: {len(d2['comps'])}")
            for c in d2["comps"][:8]:
                print(f"    diff comp area={c['area']:6d} bbox={c['bbox']} "
                      f"rescued_by_pass2={c['rescued']} kept={c['kept']*100:.1f}%")
    return out


def save_overlays(out):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = out["name"].replace(".psd", "").replace("(", "_").replace(")", "")
    raw = out["raw"]
    # GT overlay: delete = red tint, keep untouched
    ov = raw.copy()
    ov[out["gt_delete"]] = (0.5 * ov[out["gt_delete"]]
                            + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(ov).save(OUT_DIR / f"{stem}_gt.png")
    for lname, p in out["passes"].items():
        Image.fromarray((p["white"] * 255).astype(np.uint8)).save(
            OUT_DIR / f"{stem}_{lname}.png")
        if p["mm_px"]:
            mv = raw.copy()
            mv[p["mm_mask"]] = [0, 255, 0]
            Image.fromarray(mv).save(OUT_DIR / f"{stem}_{lname}_mismatch.png")


def export_cached(out):
    """Cache raw art + GT + threshold layers as .npz so downstream steps (profile
    eval, sfx.py acceptance) don't re-open the ~90MB PSDs."""
    exp = OUT_DIR / "export"
    exp.mkdir(parents=True, exist_ok=True)
    stem = out["name"].replace(".psd", "").replace("(", "_").replace(")", "")
    arrs = dict(raw=out["raw"], gt_delete=out["gt_delete"])
    for lname, p in out["passes"].items():
        arrs[lname.replace("-", "_")] = p["white"]
    np.savez_compressed(exp / f"{stem}.npz", **arrs)


if __name__ == "__main__":
    for name in FILES:
        out = decode_file(name)
        save_overlays(out)
        export_cached(out)
        print()
