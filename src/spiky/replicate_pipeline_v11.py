"""Manual-pipeline replication v11 -- plan v24 outcome.

DEFAULT = v10 EXACTLY (which defaults to v9). steps='S' uses the frame-protected spiky
action: v10's clipped action plus a +-3px protection band around every detected frame
run (horizontal dark run >= 100px at gray <= 100), excluded from the deletion. This is
the A'-style band pattern applied to the frame-junction damage measured in
logs/v24_metric_baseline.log (16/35 runs broken, mean intact 92.48%, min 51.1%).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replicate_pipeline_v6 import _protected_interiors  # noqa: E402
from replicate_pipeline_v10 import (clean_page as clean_page_v10,  # noqa: E402
                                     clean_spiky_region_clipped, find_spiky_sites)

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


def clean_page(rgb: np.ndarray, steps: str = "Q") -> np.ndarray:
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
