"""Smoke test for close_bubble_halo (throwaway, .tmp scratch).

Geometric inverse of repair_frame_interiors (keep->delete in an exterior ring, not
delete->keep in an enclosed interior). Cases:
  (a) an ink-outlined oval "bubble" with a synthetic undeleted halo ring connecting to a
      large true-background delete region -> the halo (excluding the ink stroke itself)
      must close fully, the bubble's own body must stay untouched.
  (b) a bubble whose halo band happens to touch a large separate real-content keep blob
      (simulating nearby art/text, not background) -> that content must be byte-identical
      before/after; the rest of the halo (away from the touching content) must still close.
  (c) a bubble with no halo (ring already correctly deleted) -> zero pixels changed.
  (d) two bubbles whose halos merge (bodies stay separate) -> each bubble's own halo closes
      without cross-bubble interference into the other's core.
  (e) a bubble whose halo touches only a SMALL delete pocket (below min_background_area)
      -> must NOT be reclassified.
Also: universal invariants -- input not mutated, mask only ever moves keep->delete.
"""
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ml_cleaner import close_bubble_halo  # noqa: E402

RING_WIDTH = 24
FRAME_DARKNESS = 40
MIN_BUBBLE_AREA = 2000
MIN_BACKGROUND_AREA = 8000

failures = []


def draw_bubble(rgb, mask_keep_u8, center, radii, halo_width):
    cv2.ellipse(rgb, center, radii, 0, 0, 360, (0, 0, 0), thickness=3)
    body = np.zeros_like(mask_keep_u8)
    cv2.ellipse(body, center, radii, 0, 0, 360, 1, -1)
    if halo_width > 0:
        halo = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_width * 2 + 1,) * 2))
    else:
        halo = body
    mask_keep_u8 |= halo
    return body, halo


# --- (a) + (c) share one canvas: a normal-halo bubble and a no-halo bubble, far apart
H, W = 400, 900
rgb = np.full((H, W, 3), 255, dtype=np.uint8)
keep = np.zeros((H, W), dtype=np.uint8)
body_a, halo_a = draw_bubble(rgb, keep, (100, 100), (40, 30), halo_width=20)
body_c, halo_c = draw_bubble(rgb, keep, (100, 300), (40, 30), halo_width=0)
delete = ~(keep > 0)

before = delete.copy()
after = close_bubble_halo(rgb, delete, RING_WIDTH, FRAME_DARKNESS, MIN_BUBBLE_AREA, MIN_BACKGROUND_AREA)

gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
stroke = gray <= FRAME_DARKNESS
true_halo_a = (halo_a > 0) & (body_a == 0) & ~stroke
if after[true_halo_a].mean() < 0.99:
    failures.append(f"(a) halo not fully closed: {after[true_halo_a].mean():.4f}")
if not (after[body_a > 0] == False).all():
    failures.append("(a) bubble body was touched")
if not (after[body_c > 0] == False).all():
    failures.append("(c) no-halo bubble's own body was touched")
changed_c = int((after != before)[260:340, 20:180].sum())
if changed_c != 0:
    failures.append(f"(c) no-halo case: {changed_c} pixels changed (expected 0)")

# --- (b) a bubble whose halo touches a large separate real-content blob
H2, W2 = 400, 400
rgb2 = np.full((H2, W2, 3), 255, dtype=np.uint8)
keep2 = np.zeros((H2, W2), dtype=np.uint8)
body_b, halo_b = draw_bubble(rgb2, keep2, (100, 100), (40, 30), halo_width=20)
extra = np.zeros((H2, W2), dtype=np.uint8)
cv2.rectangle(extra, (155, 70), (250, 200), 1, -1)
keep2 |= extra
delete2 = ~(keep2 > 0)
before2 = delete2.copy()
after2 = close_bubble_halo(rgb2, delete2, RING_WIDTH, FRAME_DARKNESS, MIN_BUBBLE_AREA, MIN_BACKGROUND_AREA)
if not (after2[extra > 0] == before2[extra > 0]).all():
    failures.append("(b) adjacent real content was modified")
gray2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
true_halo_b = (halo_b > 0) & (body_b == 0) & (extra == 0) & ~(gray2 <= FRAME_DARKNESS)
if after2[true_halo_b].mean() < 0.85:
    failures.append(f"(b) halo (non-touching part) not mostly closed: {after2[true_halo_b].mean():.4f}")

# --- (d) two bubbles whose halos merge, bodies stay separate
H3, W3 = 400, 400
rgb3 = np.full((H3, W3, 3), 255, dtype=np.uint8)
keep3 = np.zeros((H3, W3), dtype=np.uint8)
body_d1, halo_d1 = draw_bubble(rgb3, keep3, (70, 100), (35, 25), halo_width=20)
body_d2, halo_d2 = draw_bubble(rgb3, keep3, (170, 100), (35, 25), halo_width=20)
delete3 = ~(keep3 > 0)
after3 = close_bubble_halo(rgb3, delete3, RING_WIDTH, FRAME_DARKNESS, MIN_BUBBLE_AREA, MIN_BACKGROUND_AREA)
if not (after3[body_d1 > 0] == False).all():
    failures.append("(d) bubble 1 core was touched")
if not (after3[body_d2 > 0] == False).all():
    failures.append("(d) bubble 2 core was touched")
gray3 = cv2.cvtColor(rgb3, cv2.COLOR_RGB2GRAY)
true_halo_d = ((halo_d1 > 0) | (halo_d2 > 0)) & (body_d1 == 0) & (body_d2 == 0) & ~(gray3 <= FRAME_DARKNESS)
if after3[true_halo_d].mean() < 0.99:
    failures.append(f"(d) merged-halo bubbles: halo not fully closed: {after3[true_halo_d].mean():.4f}")

# --- (e) bubble touching only a SMALL background pocket -> must not be reclassified
H4, W4 = 400, 400
rgb4 = np.full((H4, W4, 3), 255, dtype=np.uint8)
keep4 = np.ones((H4, W4), dtype=np.uint8)  # start: everything KEEP (no big background anywhere)
body_e, halo_e = draw_bubble(rgb4, np.zeros((H4, W4), dtype=np.uint8), (100, 100), (40, 30), halo_width=20)
delete4 = np.zeros((H4, W4), dtype=bool)
small_pocket = np.zeros((H4, W4), dtype=np.uint8)
cv2.circle(small_pocket, (160, 100), 10, 1, -1)  # area ~314px, well under MIN_BACKGROUND_AREA
delete4[small_pocket > 0] = True
before4 = delete4.copy()
after4 = close_bubble_halo(rgb4, delete4, RING_WIDTH, FRAME_DARKNESS, MIN_BUBBLE_AREA, MIN_BACKGROUND_AREA)
changed4 = int((after4 != before4).sum())
if changed4 != 0:
    failures.append(f"(e) small background pocket case: {changed4} pixels changed (expected 0)")

# --- universal invariants: no in-place mutation, only ever keep->delete
delete5 = before.copy()
before5 = delete5.copy()
after5 = close_bubble_halo(rgb, delete5, RING_WIDTH, FRAME_DARKNESS, MIN_BUBBLE_AREA, MIN_BACKGROUND_AREA)
if not np.array_equal(delete5, before5):
    failures.append("input delete_mask was mutated in place")
if np.count_nonzero((~after5) & before5):
    failures.append("close_bubble_halo flipped some delete pixels back to keep (must be impossible)")

print(f"pixels changed (a/c pair): {int(np.count_nonzero(before != after))}")
print(f"pixels changed (b): {int(np.count_nonzero(before2 != after2))}")
print(f"pixels changed (d): {int(np.count_nonzero(delete3 != after3))}")
if failures:
    print("FAIL:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: all smoke assertions passed")
