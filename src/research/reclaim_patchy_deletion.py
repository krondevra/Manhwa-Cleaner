"""Plan v9 Phase B (.claude/plans/snazzy-cuddling-creek.md): per-pixel "patchy deletion around
kept content" reclaim -- release blocker #1's long-documented leading candidate mechanism
(notes/next_session_handoff.md: "true gutters delete near-solidly; this class deletes patchily
around kept content"), implemented at per-pixel granularity after the component-level variant
(Probe 0's D1) failed on fused mixed components.

Phase A probe (2026-08-06, .tmp/diagnostics/patchy_deletion_probe.py) measured, on real cached
v3+islands masks vs manual-clean GT (GT used for fitting only, never at inference):
- WRONG deletions (GT-keep) vs CORRECT deletions (GT-delete) separate in the joint
  (local kept-density, local texture) space for TEXTURED dark art (held-out ch035 capture 0.889,
  dark_cave_bubble 1.000, dark_scene_text 0.946, night_cityscape 0.408) at 0.298pp total
  correct-leak cost (within the 0.3pp under-deletion budget).
- FLAT dark digital paint (hud_panel_canonical capture 0.060, dialogue_lightbeam 0.003) has
  local-max Sobel p75 = 0 within 15px -- locally pixel-identical to true gutter. No local rule
  can capture that sub-class; it is the documented irreducible residual of blocker #1
  (consistent with the tile-size context-independence probe and the fused-component finding).

Fitted defaults below come from that probe's grid search (fit on ch001+002, validated on
held-out ch035 -- the anti-overfit protocol established after the chroma discriminator's
chapter-001-only failure).

A reclaimed pixel must be: currently deleted, near-black (this mechanism never touches bright
content -- same construction guarantee as reclaim_black_backdrop), locally textured (max Sobel
magnitude within texture_radius >= texture_thresh -- true gutters are flat), and in a
neighborhood with real kept content (kept fraction within density_radius >= density_thresh --
deep gutter/backdrop interiors have none). `min_component_px` drops tiny isolated reclaim
specks (mask hygiene, avoids salt-and-pepper keep noise inside gutters).
"""
from __future__ import annotations

import cv2
import numpy as np

FRAME_DARKNESS = 40  # project-standard near-black threshold


def reclaim_patchy_deletion(
    rgb: np.ndarray,
    delete_mask: np.ndarray,
    darkness_threshold: int = FRAME_DARKNESS,
    texture_radius: int = 15,
    texture_thresh: float = 2.0,
    density_radius: int = 200,
    density_thresh: float = 0.10,
    min_component_px: int = 300,
) -> np.ndarray:
    """postprocess(rgb, mask) -> mask, composable with the project's existing chain steps."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    near_black = gray <= darkness_threshold

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    del gx, gy
    mag_u8 = np.clip(mag, 0, 255).astype(np.uint8)
    del mag
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (texture_radius * 2 + 1, texture_radius * 2 + 1)
    )
    tex = cv2.dilate(mag_u8, k)
    del mag_u8

    keep_u8 = (~delete_mask).astype(np.uint8)
    ksize = density_radius * 2 + 1
    dens = cv2.boxFilter(keep_u8, cv2.CV_32F, (ksize, ksize), normalize=True)
    del keep_u8

    reclaim = (
        delete_mask & near_black & (tex >= texture_thresh) & (dens >= density_thresh)
    )
    del tex, dens, near_black

    if min_component_px > 0 and reclaim.any():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            reclaim.astype(np.uint8), connectivity=8
        )
        keep_labels = np.zeros(num, dtype=bool)
        for lbl in range(1, num):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_component_px:
                keep_labels[lbl] = True
        reclaim = keep_labels[labels]

    fixed = delete_mask.copy()
    fixed[reclaim] = False
    return fixed


def _self_test() -> None:
    rng = np.random.default_rng(0)

    # (1) flat synthetic gutter: near-black, zero texture, kept content above. The
    # bright<->dark transition's own Sobel edge bleeds texture_radius px into the gutter's top
    # rows (a thin boundary fringe, same behavior class as protect_frame_borders, already
    # counted in the probe's budget) -- the guarantee under test is that the gutter INTERIOR
    # (beyond texture_radius of the boundary) is NEVER reclaimed.
    h, w = 600, 400
    rgb = np.full((h, w, 3), 10, dtype=np.uint8)          # flat near-black
    rgb[:100] = 200                                        # bright kept band at top
    mask = np.ones((h, w), dtype=bool)
    mask[:100] = False                                     # top band kept
    out = reclaim_patchy_deletion(rgb, mask)
    reclaimed = mask & ~out
    interior_reclaimed = int(reclaimed[100 + 16 :].sum())
    fringe_reclaimed = int(reclaimed[100 : 100 + 16].sum())
    assert interior_reclaimed == 0, \
        f"flat gutter interior must reclaim 0, got {interior_reclaimed}"
    print(f"PASS: flat gutter interior -> zero reclaim (boundary fringe: {fringe_reclaimed} px, "
          f"expected/budgeted)")

    # (2) textured near-black blob surrounded by kept content -> reclaimed
    rgb2 = np.full((h, w, 3), 200, dtype=np.uint8)         # bright kept page
    yy, xx = np.mgrid[200:400, 100:300]
    noise = rng.integers(0, 40, size=(200, 200), dtype=np.uint8)  # textured dark blob
    rgb2[200:400, 100:300] = noise[..., None]
    mask2 = np.zeros((h, w), dtype=bool)
    mask2[200:400, 100:300] = True                         # model wrongly deletes the blob
    out2 = reclaim_patchy_deletion(rgb2, mask2)
    blob_reclaimed = int((mask2 & ~out2)[200:400, 100:300].sum())
    frac = blob_reclaimed / (200 * 200)
    assert frac > 0.5, f"textured blob near kept content should be mostly reclaimed, got {frac:.3f}"
    print(f"PASS: textured near-black blob among kept content -> reclaimed ({frac:.1%})")

    # (3) bright deleted content is never touched (near-black gate)
    rgb3 = np.full((h, w, 3), 200, dtype=np.uint8)
    rgb3[300:350, :] = rng.integers(100, 255, size=(50, w, 3), dtype=np.uint8)  # bright textured
    mask3 = np.zeros((h, w), dtype=bool)
    mask3[300:350] = True
    out3 = reclaim_patchy_deletion(rgb3, mask3)
    assert (mask3 == out3).all(), "bright deleted content must never be reclaimed"
    print("PASS: bright deleted content untouched (near-black gate)")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
