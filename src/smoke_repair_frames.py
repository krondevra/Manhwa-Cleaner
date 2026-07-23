"""Smoke test for repair_frame_interiors (throwaway, .tmp scratch).

Per-hole design. Synthetic 900x600 white canvas with:
  (a) a thin closed black rectangle frame with delete-holes punched inside
      -> holes must be repaired (except within inset of the stroke)
  (b) a filled black rectangle (fake black-bg panel, no enclosed light holes)
      -> delete pixels on/around it untouched
  (c) an open (broken) thin rectangle with a delete-hole inside
      -> must NOT be repaired (interior leaks to exterior)
  (d) a closed thin outline 8-connected to a filled dark blob (the real-world
      bubble-tail-touches-panel case that killed the component-ratio design)
      -> its hole must still be repaired
Also: gutter delete pixels far from any frame must stay deleted.
"""
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ml_cleaner import repair_frame_interiors  # noqa: E402

H, W = 900, 600
rgb = np.full((H, W, 3), 255, dtype=np.uint8)
delete = np.zeros((H, W), dtype=bool)

# --- (a) closed thin frame at rows 50..250, cols 50..550, 3px stroke
cv2.rectangle(rgb, (50, 50), (550, 250), (0, 0, 0), 3)
delete[100:140, 100:200] = True     # interior hole 1
delete[180:220, 400:500] = True     # interior hole 2

# --- (b) filled black rectangle at rows 300..450, cols 50..550
cv2.rectangle(rgb, (50, 300), (550, 450), (0, 0, 0), -1)
delete[320:430, 80:520] = True      # must stay deleted

# --- (c) open frame at rows 500..640, cols 50..550, 3px stroke, 40px gap
cv2.rectangle(rgb, (50, 500), (550, 640), (0, 0, 0), 3)
rgb[500 - 2 : 500 + 4, 250:290] = 255  # break top edge (wider than 5px closing)
delete[540:590, 100:200] = True        # "interior" hole -> must survive

# --- (d) closed thin outline touching a filled blob (merged component)
cv2.rectangle(rgb, (50, 680), (350, 850), (0, 0, 0), 3)   # closed frame
cv2.rectangle(rgb, (350, 700), (560, 830), (0, 0, 0), -1)  # filled blob, shares x=350 edge
delete[720:800, 100:300] = True     # hole inside the closed part -> must be repaired

# --- gutter deletes far from any frame -- must survive
delete[870:895, 50:550] = True

before = delete.copy()
after = repair_frame_interiors(
    rgb, delete, frame_darkness=40, min_interior_px=10000, inset_px=2,
)

failures = []
if after[100:140, 100:200].any():
    failures.append("(a) hole 1 not repaired")
if after[180:220, 400:500].any():
    failures.append("(a) hole 2 not repaired")
if not np.array_equal(after[320:430, 80:520], before[320:430, 80:520]):
    failures.append("(b) filled black rect was modified (false detection!)")
if not np.array_equal(after[540:590, 100:200], before[540:590, 100:200]):
    failures.append("(c) open-frame hole was repaired (should leak out, be rejected)")
if after[720:800, 100:300].any():
    failures.append("(d) hole in merged component not repaired (per-hole rule failed)")
if not np.array_equal(after[870:895, 50:550], before[870:895, 50:550]):
    failures.append("gutter deletes were modified")
if not np.array_equal(delete, before):
    failures.append("input delete_mask was mutated in place")
if np.count_nonzero(after & ~before):
    failures.append("repair ADDED delete pixels (must be impossible)")

print(f"pixels changed by repair: {int(np.count_nonzero(before != after))}")
if failures:
    print("FAIL:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: all smoke assertions passed")
