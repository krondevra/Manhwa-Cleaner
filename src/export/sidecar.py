"""Provenance sidecar (gen-8, additive-only) -- feeds the PSD layered-mask
export (the GUI consumer was reversed, see the 2026-08-12 postmortem).

Replays clean_chapter_full's composition stage by stage (importing the
classifiers; sfx.py itself is NOT modified) and records WHERE each
mechanism acted plus the hard-wall CANDIDATE zones the automatic pipeline
cannot resolve. Production defaults untouched.

Outputs under out_dir (default .tmp/gui/<chapter>/):
  delete.npy   final bool delete mask (identical to clean_chapter_full's)
  labels.npz   uint8 mechanism label per px, last authority wins (composition
               order), key 'labels'
  zones.json   list of zone records: {id, kind, bbox [x0,y0,x1,y1],
               geom_hash, meta}; kinds are 'mech:*' (informational) and
               'wall:1..6' (correction candidates)

Label codes (LABELS): 0 none/kept, 1 gutter default-delete, 2 rc keep,
3 rc empty-skip, 4 site action zone, 5 text-skirt rescue, 6 ink-context
rescue, 7 extent-fallback band, 8 CONTENT_DENSE keep band.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from classifiers import sfx  # noqa: E402
from classifiers.detector_framework import detect  # noqa: E402
from classifiers.panel_segmentation import (segment_chapter,  # noqa: E402
                                            units_for_processing)
from classifiers.profiles import regular_cloud as rc  # noqa: E402
from classifiers.profiles import spiky_cloud as sc  # noqa: E402

LABELS = {"gutter_delete": 1, "rc_keep": 2, "rc_empty_skip": 3, "site": 4,
          "text_rescue": 5, "ink_rescue": 6, "extent_fallback": 7,
          "dense_keep": 8}

WALL_NAMES = {1: "semantic", 2: "site", 3: "ring", 4: "card",
              5: "dark", 6: "pale"}


def _hash_bbox_mask(mask: np.ndarray, x0, y0, x1, y1) -> str:
    return hashlib.sha1(np.ascontiguousarray(
        mask[y0:y1, x0:x1]).tobytes()).hexdigest()[:16]


def generate_sidecar(chapter_png: str | Path, out_dir: str | Path,
                     verbose: bool = True) -> dict:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    t0 = time.time()
    rgb = np.array(Image.open(chapter_png).convert("RGB"))
    H, W = rgb.shape[:2]
    g = rgb[..., 1]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = np.zeros((H, W), np.uint8)
    zones: list[dict] = []
    zid = 0

    def add_zone(kind, x0, y0, x1, y1, meta=None, hash_mask=None):
        nonlocal zid
        zid += 1
        zones.append({"id": zid, "kind": kind,
                      "bbox": [int(x0), int(y0), int(x1), int(y1)],
                      "geom_hash": _hash_bbox_mask(
                          hash_mask if hash_mask is not None else g,
                          x0, y0, x1, y1),
                      "meta": meta or {}})
        return zid

    # --- stage 1: clean_chapter (segmentation-driven) ---
    segs = segment_chapter(rgb)
    units = units_for_processing(segs, H)
    delete, _ = sfx.clean_chapter(rgb)
    labels[delete] = LABELS["gutter_delete"]
    for s in segs:
        if s.kind == "borderless":
            win = g[s.y0:s.y1]
            ink = float((win < 100).mean())
            content = float((win < sfx.BLANK_G).mean())
            if ink < sfx.DENSE_INK and content >= sfx.CONTENT_DENSE:
                labels[s.y0:s.y1] = np.where(
                    delete[s.y0:s.y1], labels[s.y0:s.y1],
                    LABELS["dense_keep"])
                add_zone("mech:dense_keep", 0, s.y0, W, s.y1,
                         {"ink": round(ink, 3), "content": round(content, 3)})
                # wall6 candidate: kept band with almost no ink = pale texture
                if ink < 0.05:
                    add_zone("wall:6", 0, s.y0, W, s.y1,
                             {"ink": round(ink, 3)})
        if s.kind in ("panel", "partial") and (s.x0 == 0 and s.x1 == W):
            # extent fallback (full width from corroboration/one-sided rules)
            labels[s.y0:s.y1][~delete[s.y0:s.y1] & (labels[s.y0:s.y1] == 0)] \
                = LABELS["extent_fallback"]
        if s.kind == "partial":
            add_zone("wall:3", s.x0, s.y0, s.x1, s.y1, {"kind": s.kind})
        # wall5: dark-dominant band
        if s.kind in ("panel", "partial", "borderless"):
            win = g[s.y0:s.y1]
            if float((win < 60).mean()) > 0.5:
                add_zone("wall:5", 0, s.y0, W, s.y1,
                         {"dark_frac": round(float((win < 60).mean()), 3)})

    # --- stage 2: rc keeps (replayed with the 8.12.4 policy) ---
    sites = detect(rgb, sc.PROFILE)
    for y0u, y1u, kinds in units:
        for (rx0, ry0, rx1, ry1) in detect(rgb[y0u:y1u], rc.PROFILE):
            gy0, gy1 = ry0 + y0u, ry1 + y0u
            overlaps = any(rx0 < sx1 and rx1 > sx0 and gy0 < sy1 and gy1 > sy0
                           for sx0, sy0, sx1, sy1 in sites)
            inset = max(6, int(sfx.RC_INSET_FRAC * min(rx1 - rx0, gy1 - gy0)))
            iy0, iy1 = gy0 + inset, gy1 - inset
            ix0, ix1 = rx0 + inset, rx1 - inset
            ink = (float((rgb[iy0:iy1, ix0:ix1, 1] < 100).mean())
                   if (iy1 > iy0 and ix1 > ix0) else 1.0)
            card_like = (rx1 - rx0) >= 150 and (gy1 - gy0) >= 60
            if overlaps:
                continue
            if ink < sfx.RC_KEEP_INK_MIN:
                labels[gy0:gy1, rx0:rx1] = LABELS["rc_empty_skip"]
                add_zone("mech:rc_empty_skip", rx0, gy0, rx1, gy1,
                         {"interior_ink": round(ink, 4)})
                if card_like:
                    add_zone("wall:4", rx0, gy0, rx1, gy1,
                             {"interior_ink": round(ink, 4), "src": "empty"})
            else:
                by0 = max(0, gy0 - sfx.E_BUBBLE); by1 = min(H, gy1 + sfx.E_BUBBLE)
                bx0 = max(0, rx0 - sfx.E_BUBBLE); bx1 = min(W, rx1 + sfx.E_BUBBLE)
                labels[by0:by1, bx0:bx1] = np.where(
                    delete[by0:by1, bx0:bx1], labels[by0:by1, bx0:bx1],
                    LABELS["rc_keep"])
                add_zone("mech:rc_keep", bx0, by0, bx1, by1,
                         {"interior_ink": round(ink, 4)})
                if card_like and ink < 0.10:
                    add_zone("wall:4", rx0, gy0, rx1, gy1,
                             {"interior_ink": round(ink, 4), "src": "keep"})

    # --- stage 3: full pipeline final mask (sites + rescues included) ---
    final, stats = sfx.clean_chapter_full(rgb)
    for (sx0, sy0, sx1, sy1) in sites:
        labels[sy0:sy1, sx0:sx1] = LABELS["site"]
        add_zone("mech:site", sx0, sy0, sx1, sy1)
        add_zone("wall:2", sx0, sy0, sx1, sy1)
    # rescues: px kept in final but deleted before the rescue stages
    pre_rescue = delete.copy()
    # reconstruct rc subtraction on the stage-1 mask for accurate diffing
    rc_keep_mask = labels == LABELS["rc_keep"]
    pre_rescue &= ~rc_keep_mask
    text_r = sfx._text_skirt_rescue(rgb, pre_rescue, sites)
    labels[text_r] = LABELS["text_rescue"]
    ink_r = sfx._ink_context_rescue(rgb, pre_rescue & ~text_r, sites)
    labels[ink_r] = LABELS["ink_rescue"]

    # wall1: the honest-negative population -- deleted-ink fragments below
    # the iso bar (still deleted in the FINAL mask)
    ink_mask = g < 100
    insite = np.zeros((H, W), bool)
    for (sx0, sy0, sx1, sy1) in sites:
        insite[sy0:sy1, sx0:sx1] = True
    cand = (final & ink_mask & ~insite).astype(np.uint8)
    n, lb, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 30:
            continue
        x, y, w, h = (st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP],
                      st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
        add_zone("wall:1", x, y, x + w, y + h,
                 {"area": int(st[i, cv2.CC_STAT_AREA])})

    np.save(out / "delete.npy", final)
    np.savez_compressed(out / "labels.npz", labels=labels)
    (out / "zones.json").write_text(json.dumps(
        {"chapter": str(chapter_png), "H": H, "W": W, "stats": stats,
         "zones": zones}, indent=1))
    if verbose:
        walls = {}
        for z in zones:
            if z["kind"].startswith("wall:"):
                walls[z["kind"]] = walls.get(z["kind"], 0) + 1
        print(f"sidecar {out}: {len(zones)} zones ({walls}), "
              f"{time.time()-t0:.0f}s, stats={stats}")
    return {"zones": len(zones), "stats": stats}


if __name__ == "__main__":
    generate_sidecar(sys.argv[1], sys.argv[2])
