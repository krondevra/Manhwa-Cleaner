"""Smoke test for inject_bubble_interior_texture (throwaway, .tmp scratch).

2026-07-31 halo-investigation fallback: bubble/cloud interior texture augmentation,
implemented entirely in Manhwa-Cleaner (see halo_investigation.md) since the bubble generator
itself lives in the sibling PepperNCarrotDataset repo, which is no longer edited. Mirrors
smoke_close_bubble_halo.py's structure since this reuses that function's own hole-detection
technique. Cases:
  (a) prob=0.0 -> true no-op, byte-identical output, same object identity (fast path).
  (b) prob=1.0 on a plain oval bubble -> some interior pixels change, ink stroke stays
      byte-identical, background outside the bubble stays byte-identical, mask_crop is never
      read for writing and never mutated.
  (c) a dark "text glyph" pixel block inside the bubble interior -> must stay byte-identical
      even at prob=1.0 (ink-threshold exclusion).
  (d) changed pixel values stay within [BUBBLE_TEXTURE_DARK_FLOOR, BUBBLE_TEXTURE_LIGHT_CEILING].
  (e) a canvas with no bubble at all (no enclosed hole) -> no-op even at prob=1.0.
"""
import random
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_cleaner import (  # noqa: E402
    inject_bubble_interior_texture,
    BUBBLE_TEXTURE_DARK_FLOOR,
    BUBBLE_TEXTURE_LIGHT_CEILING,
    BUBBLE_TEXTURE_INK_THRESHOLD,
)

FRAME_DARKNESS = 40
MIN_BUBBLE_AREA = 400

failures = []


def draw_oval_bubble(h, w, center, radii):
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.ellipse(rgb, center, radii, 0, 0, 360, (0, 0, 0), thickness=3)
    body = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(body, center, radii, 0, 0, 360, 1, -1)
    return rgb, body


H, W = 300, 300
rng = random.Random(1234)

# --- (a) prob=0.0 must be a true no-op, same object returned
rgb_a, body_a = draw_oval_bubble(H, W, (150, 150), (70, 50))
mask_a = np.zeros((H, W), dtype=bool)
out_a = inject_bubble_interior_texture(rgb_a, mask_a, rng, prob=0.0,
                                        frame_darkness=FRAME_DARKNESS, min_bubble_area=MIN_BUBBLE_AREA)
if out_a is not rgb_a:
    failures.append("(a) prob=0.0 did not return the same object (expected fast-path no-op)")
if not np.array_equal(out_a, rgb_a):
    failures.append("(a) prob=0.0 changed pixels (expected exact no-op)")

# --- (b) prob=1.0 on a plain oval: interior may change, stroke/background must not
rgb_b, body_b = draw_oval_bubble(H, W, (150, 150), (70, 50))
before_b = rgb_b.copy()
mask_b = np.zeros((H, W), dtype=bool)
mask_b_before = mask_b.copy()
out_b = inject_bubble_interior_texture(rgb_b, mask_b, rng, prob=1.0,
                                        frame_darkness=FRAME_DARKNESS, min_bubble_area=MIN_BUBBLE_AREA)
gray_before = cv2.cvtColor(before_b, cv2.COLOR_RGB2GRAY)
stroke_b = gray_before <= FRAME_DARKNESS
interior_b = (body_b > 0) & ~stroke_b
exterior_b = body_b == 0

n_interior_changed = int((out_b != before_b).any(axis=2)[interior_b].sum())
if n_interior_changed == 0:
    failures.append("(b) prob=1.0 changed zero interior pixels (expected texture injection)")

if not np.array_equal(out_b[stroke_b], before_b[stroke_b]):
    failures.append("(b) ink stroke pixels were modified")
if not np.array_equal(out_b[exterior_b], before_b[exterior_b]):
    failures.append("(b) exterior/background pixels were modified")
if not np.array_equal(rgb_b, before_b):
    failures.append("(b) input arr_crop was mutated in place (must return a copy)")
if not np.array_equal(mask_b, mask_b_before):
    failures.append("(b) mask_crop was modified (ground truth must never be touched)")

# --- (c) a dark "text glyph" block inside the interior must be excluded from injection
rgb_c, body_c = draw_oval_bubble(H, W, (150, 150), (70, 50))
cv2.rectangle(rgb_c, (135, 135), (165, 165), (30, 30, 30), thickness=-1)  # dark glyph, gray ~30
before_c = rgb_c.copy()
mask_c = np.zeros((H, W), dtype=bool)
glyph_region = np.zeros((H, W), dtype=bool)
glyph_region[135:165, 135:165] = True
out_c = inject_bubble_interior_texture(rgb_c, mask_c, rng, prob=1.0,
                                        frame_darkness=FRAME_DARKNESS, min_bubble_area=MIN_BUBBLE_AREA)
if not np.array_equal(out_c[glyph_region], before_c[glyph_region]):
    failures.append("(c) dark text-glyph pixels were overwritten by texture (ink-threshold exclusion broken)")

# --- (d) changed pixel values must respect the safety band
gray_out_b = cv2.cvtColor(out_b, cv2.COLOR_RGB2GRAY)
changed_mask_b = (out_b != before_b).any(axis=2) & interior_b
if changed_mask_b.any():
    vals = gray_out_b[changed_mask_b]
    if vals.min() < BUBBLE_TEXTURE_DARK_FLOOR - 1 or vals.max() > BUBBLE_TEXTURE_LIGHT_CEILING:
        failures.append(
            f"(d) texture values out of safety band: min={vals.min()}, max={vals.max()}, "
            f"expected [{BUBBLE_TEXTURE_DARK_FLOOR}, {BUBBLE_TEXTURE_LIGHT_CEILING}]")

# --- (e) no enclosed bubble hole at all -> no-op even at prob=1.0
rgb_e = np.full((H, W, 3), 255, dtype=np.uint8)
cv2.line(rgb_e, (20, 20), (280, 280), (0, 0, 0), thickness=3)  # a stroke with no enclosed hole
before_e = rgb_e.copy()
mask_e = np.zeros((H, W), dtype=bool)
out_e = inject_bubble_interior_texture(rgb_e, mask_e, rng, prob=1.0,
                                        frame_darkness=FRAME_DARKNESS, min_bubble_area=MIN_BUBBLE_AREA)
if not np.array_equal(out_e, before_e):
    failures.append("(e) canvas with no enclosed bubble hole was modified (expected no-op)")

print(f"pixels changed (b, interior): {n_interior_changed}")
if failures:
    print("FAIL:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: all smoke assertions passed")
