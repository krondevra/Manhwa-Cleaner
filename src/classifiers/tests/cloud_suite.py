"""Regression suite for the EXISTING cloud/bubble classifier
(`style_analysis.extract_enclosed_holes` + `classify_and_measure`).

Gen-8 phase 3 (2026-08-10). Purpose: turn the classifier's user-reported defect list
into measured, repeatable numbers BEFORE the detector-framework work (phase 4) touches
anything -- the framework's regular_cloud profile must beat this baseline, not silently
regress it. The Revision-2 fixes (width-only frame routing, text-plausibility interior
filter -- notes/style_analysis_findings.md limitations 2+3) were verified only by visual
spot-checks on the flagged pages when they shipped; this suite is their first proper
regression harness.

Ground truth sets (coverage stated honestly):
  A. USER CLAUDS CROPS (92): `.tmp/debug/clauds-and-ui/ch{1,2,3,86}/clauds/*.png` --
     every crop is a user-curated positive example of a bubble-family shape (oval,
     cloud, spiky, thorn, rectangle text-box, ...). Expectation: >= 1 bubble-family
     detection per crop. Measures defect (a) recall (incl. text-heavy bubbles ->
     exercises the limitation-3 fix directly) and (b) the crop must NOT come back
     frame-classified only.
  B. SPIKY PSD INSTANCES (12): the standing spiky-cloud diagnostic set. Expectation:
     >= 1 jagged-family ({spiky, cloud, thorn}) detection per instance crop.
  C. SYNTH FRAME-ONLY PAGES (20): `archive/scripts-manual/synth_pages/frames/` --
     synthetic pages with panels/gutters/gradients and ZERO bubbles. Any bubble-family
     detection is a frame-as-cloud false positive (defect (b)).

COVERAGE GAPS (do not over-claim): no pixel-exact shape masks (presence/family-level GT
only); dark-scene recall (limitation 4) is only covered to the extent the user's ch2
crops include dark backgrounds; no labeled examples of the original Revision-1 art
false positives (jewelry/cuffs) -- those classes ride on set C only indirectly.

Crops are pasted onto a white canvas 2.2x their size before detection: the classifier's
frame routing is PAGE-RELATIVE (bbox >= 55% of page width -> frame taxonomy), so running
a tight crop as-is would misroute large bubbles purely as a harness artifact. Padding
keeps every crop shape below the routing threshold; this is a documented harness choice,
NOT a claim about full-page behavior (set C runs full pages unpadded).

Usage:  .venv/bin/python src/classifiers/tests/cloud_suite.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/spiky"))

from style_analysis import extract_enclosed_holes  # noqa: E402

BUBBLE_FAMILY = {"oval", "cloud", "spiky", "thorn", "rectangle", "other"}
JAGGED_FAMILY = {"spiky", "cloud", "thorn"}
PAD = 2.2

CLAUDS_GLOB = str(REPO / ".tmp/debug/clauds-and-ui/ch*/clauds/*.png")
SYNTH_GLOB = str(REPO / "archive/scripts-manual/synth_pages/frames/*.png")
DIAG = REPO / ".tmp/debug/spiky-clouds-diagnostics-psd"


def pad_to_canvas(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    H, W = int(h * PAD), int(w * PAD)
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    y0, x0 = (H - h) // 2, (W - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = rgb
    return canvas


def classes_on(rgb: np.ndarray) -> list[str]:
    return [s["class"] for s in extract_enclosed_holes(rgb)]


def run_suite(verbose: bool = True):
    results = {}

    # --- Set A: user clauds crops ---
    a_rows = []
    for p in sorted(glob.glob(CLAUDS_GLOB)):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        cls = classes_on(pad_to_canvas(rgb))
        fam = [c for c in cls if c in BUBBLE_FAMILY]
        a_rows.append((str(Path(p).relative_to(REPO)), fam, cls))
    a_hit = sum(1 for _, fam, _ in a_rows if fam)
    results["A"] = (a_hit, len(a_rows))

    # --- Set B: spiky PSD instances ---
    b_rows = []
    for d in sorted(DIAG.iterdir()):
        if not d.is_dir():
            continue
        src = d / f"{d.name}.png"
        if not src.exists():
            src = d / f"{d.name}_initial.png"
        rgb = np.asarray(Image.open(src).convert("RGB"))
        cls = classes_on(pad_to_canvas(rgb))
        jag = [c for c in cls if c in JAGGED_FAMILY]
        b_rows.append((d.name, jag, cls))
    b_hit = sum(1 for _, jag, _ in b_rows if jag)
    results["B"] = (b_hit, len(b_rows))

    # --- Set C: synthetic frame-only pages (full pages, unpadded) ---
    c_rows = []
    for p in sorted(glob.glob(SYNTH_GLOB)):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        cls = classes_on(rgb)
        fp = [c for c in cls if c in BUBBLE_FAMILY]
        c_rows.append((Path(p).name, fp, cls))
    c_fp_pages = sum(1 for _, fp, _ in c_rows if fp)
    results["C"] = (c_fp_pages, len(c_rows))

    if verbose:
        print("=== Set A: user clauds crops (positives; expectation: >=1 bubble-family) ===")
        for name, fam, cls in a_rows:
            mark = "ok  " if fam else "MISS"
            print(f"  {mark} {name}: {fam if fam else cls or '(no shapes at all)'}")
        print(f"Set A recall: {a_hit}/{len(a_rows)} ({100*a_hit/max(len(a_rows),1):.1f}%)")
        print("\n=== Set B: spiky PSD instances (expectation: >=1 of {spiky,cloud,thorn}) ===")
        for name, jag, cls in b_rows:
            mark = "ok  " if jag else "MISS"
            print(f"  {mark} {name}: {jag if jag else cls or '(no shapes at all)'}")
        print(f"Set B jagged recall: {b_hit}/{len(b_rows)} ({100*b_hit/max(len(b_rows),1):.1f}%)")
        print("\n=== Set C: synth frame-only pages (expectation: 0 bubble-family) ===")
        for name, fp, _ in c_rows:
            if fp:
                print(f"  FP   {name}: {fp}")
        print(f"Set C frame-as-cloud FP pages: {c_fp_pages}/{len(c_rows)}")
        print("\n=== BASELINE SUMMARY ===")
        print(f"  A (crop recall):        {a_hit}/{len(a_rows)}")
        print(f"  B (spiky-instance):     {b_hit}/{len(b_rows)}")
        print(f"  C (frame FP pages):     {c_fp_pages}/{len(c_rows)}")
    return results, a_rows, b_rows, c_rows


if __name__ == "__main__":
    run_suite()
