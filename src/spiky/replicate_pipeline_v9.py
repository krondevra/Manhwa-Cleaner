"""Manual-pipeline replication v9 -- plan v22 outcome.

DEFAULT = v8 EXACTLY. Adds the DARK-BACKDROP track as an OPT-IN step (steps='D'), the
`--reclaim-islands` precedent: a human decides per part/page whether its dark stratum is
page backdrop (delete) or artwork (keep); the mechanized rule then does the work.

== Method extraction (new-gold PSDs, plan v22 part 1) ==

The user's mask-hard / mask-soft layers in every new-gold PSD are BLACK-track candidate
masks (bright-stratum recall ~0-2%, dark-stratum recall 67-100% vs final GT), decoded as
simple threshold+morphology rules: mask-hard ~ (gray <= 32) dilate 4 (IoU 89.6% on
033-3), mask-soft ~ (gray <= 64) dilate 4 (IoU 92.7%); 033-3's refined black-hard/soft
variants fit at T=32 with symmetric morphology (IoU 69-71% vs final GT directly).
The user applied these candidates SELECTIVELY per part: near-wholesale on backdrop parts
(033-2 dark recall 100%), fully rejected on dark-art parts (001-2 precision 1%).

== Automatic-scope ladder (3 attempts, logs/v22_black_attempt*.log) ==

  a1 margin/deleted-bg connectivity            -- backdrop recall ~complete but dark ART
     eaten wholesale (001-2 over +14.0pp for zero benefit).
  a2 + wraps-panels (protected adjacency)      -- fails: speech bubbles float in dark art
     too; and backdrop/art fuse into single components (033-3 over 18.4pp on FIT).
  a3 + flatness gate (std21 <= 2.0)            -- breakthrough on backdrop parts
     (001-1 28.6->7.3, 033-3 36.6->6.8, 033-2 38.5->4.7, 033-4 44.0->11.9 full-error)
     but still +6.0pp content loss on 001-2: flat black fills that ARE art (silhouettes,
     night scenes) are locally identical to page backdrop. Part-level semantics.

Verdict: full-auto = honest negative (dark-art parts regress); the a3 rule ships OPT-IN
for parts a human marks as backdrop-bearing -- one click replaces the entire manual
black-track pass at ~95% recall.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from replicate_pipeline_v6 import _protected_interiors  # noqa: E402
from replicate_pipeline_v8 import clean_page as clean_page_v8, _flood  # noqa: E402

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


def clean_page(rgb: np.ndarray, steps: str = "") -> np.ndarray:
    """v9 default == v8 exactly (which defaults to v7 + A'). steps='D' opts in the
    dark-backdrop track for pages a human marked as backdrop-bearing."""
    delete = clean_page_v8(rgb)
    if "D" in steps:
        f = rgb.astype(np.float32)
        gray = np.round((f.max(axis=2) + f.min(axis=2)) / 2.0).astype(np.uint8)
        delete = step_h_dark_backdrop(rgb, gray, delete)
    return delete
