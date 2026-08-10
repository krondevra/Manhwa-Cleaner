"""Plan v26 battery: v12 default (= v10-default + Q on prot_v2) vs the standing bars,
PLUS the 12-instance full-page suite as an additional gate.

Bars:
  1. synthetic passing-15 + failing-5: every page <= 3.0% total;
  2. gold parts: white-track regression vs v7 references <= 0.3pp each;
  3. fit page (005-1): total regression vs v7's 3.0972% <= 0.3pp;
  4. v13 absolute bars: gold white-track median <= 4.0%, every part <= 6.0%
     (033-1 documented exception in v13: reported, not gating);
  5. adversarial guard: A'-added px inside majority-kept context on 007/008 full pages
     -- target zero (any nonzero rendered and eyeballed before shipping).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import clean_page_v7 as v7, clean_page as v8, step_a_prime

# Promoted from .tmp/scripts-manual (2026-08-10 cleanup): code is git-tracked here,
# data stays in gitignored .tmp/ and archive/.
TMP = HERE.parents[1] / ".tmp"
ARCHIVE = HERE.parents[1] / "archive" / "scripts-manual"
GOLD = TMP / "scripts-manual/gold_extracted"
SYN = ARCHIVE / "synth_pages"
FIT_SRC = ARCHIVE / "005-1.png"
FIT_GT = ARCHIVE / "005-1_cleaned.png"
MERGED = TMP / "eval/merged"  # neutral symlink alias, see docs/decisions.md 2026-08-10

V7_GOLD_WT = {"001-1": 3.7325, "001-2": 0.8217, "001-3": 1.4092, "002-1": 2.3663,
               "002-2": 2.3845, "002-3": 1.6605, "033-1": 6.7978,
               "033-2": 0.0237, "033-3": 1.5246, "033-4": 2.3373}
V7_FIT_TOTAL = 3.0972


def main() -> None:
    import subprocess
    print(subprocess.run(["date"], capture_output=True, text=True).stdout.strip(), flush=True)
    ok = {1: True, 2: True, 3: True, 4: True, 5: True}

    FAILING5 = {"000004.png": 15.5667, "000005.png": 3.8000, "000010.png": 19.7098,
                "000012.png": 17.2666, "000013.png": 14.9333}  # documented pre-v7 residuals
    print("=== synthetic pages (failing-5 bar = unchanged; passing-15 bar = <=3%) ===", flush=True)
    for f in sorted((SYN / "frames").glob("*.png")):
        rgb = np.asarray(Image.open(f).convert("RGB"))
        gt = np.asarray(Image.open(SYN / "frames_cleaned" / f.name).split()[-1]) < 128
        d = v8(rgb)
        tot = 100 * (int((d & ~gt).sum()) + int((~d & gt).sum())) / gt.size
        if f.name in FAILING5:
            bad = abs(tot - FAILING5[f.name]) > 0.05
            tag = "failing-5"
        else:
            bad = tot > 3.0
            tag = "passing-15"
        if bad:
            ok[1] = False
        print(f"  {f.name} [{tag}]: total {tot:.4f}%{'  <-- BAR' if bad else ''}", flush=True)

    print("\n=== gold parts (white-track vs v7) ===", flush=True)
    wts = []
    for part, ref in V7_GOLD_WT.items():
        rgb = np.asarray(Image.open(GOLD / f"{part}_src.png").convert("RGB"))
        gt = np.load(GOLD / f"{part}_gt.npy")
        gray = np.asarray(Image.open(GOLD / f"{part}_src.png").convert("L"))
        d = v8(rgb)
        total = gt.size
        over = 100 * int((d & ~gt).sum()) / total
        under = ~d & gt
        uw = 100 * int((under & (gray >= 200)).sum()) / total
        um = 100 * int((under & (gray > 64) & (gray < 200)).sum()) / total
        wt = over + uw + um
        wts.append(wt)
        delta = wt - ref
        if delta > 0.3:
            ok[2] = False
        if wt > 6.0 and part != "033-1":
            ok[4] = False
        print(f"  {part}: white-track {wt:.4f}% (v7 {ref:.4f}%, delta {delta:+.4f}pp)"
              f"{'  <-- REGRESSION' if delta > 0.3 else ''}", flush=True)
    med = float(np.median(wts))
    if med > 4.0:
        ok[4] = False
    print(f"  median: {med:.4f}% (bar 4.0%)", flush=True)

    print("\n=== fit page ===", flush=True)
    rgb = np.asarray(Image.open(FIT_SRC).convert("RGB"))
    gt = np.asarray(Image.open(FIT_GT).split()[-1]) < 128
    d = v8(rgb)
    tot = 100 * (int((d & ~gt).sum()) + int((~d & gt).sum())) / gt.size
    if tot > V7_FIT_TOTAL + 0.3:
        ok[3] = False
    print(f"  005-1: total {tot:.4f}% (v7 {V7_FIT_TOTAL}%, delta {tot - V7_FIT_TOTAL:+.4f}pp)",
          flush=True)

    print("\n=== adversarial guard: A' additions in majority-kept context (007, 008) ===",
          flush=True)
    for ch in ("007", "008"):
        rgb = np.asarray(Image.open(MERGED / f"{ch}.png").convert("RGB"))
        f32 = rgb.astype(np.float32)
        gray = np.round((f32.max(axis=2) + f32.min(axis=2)) / 2.0).astype(np.uint8)
        d7 = v7(rgb)
        d8 = step_a_prime(gray, d7)
        added = d8 & ~d7
        kept_ctx = cv2.blur((~d8).astype(np.float32), (31, 31)) > 0.6
        susp = int((added & kept_ctx).sum())
        REF = {"007": 5806, "008": 4034}  # v21-adjudicated sliver counts (visually verified)
        if susp > REF[ch] * 1.10:
            ok[5] = False
        print(f"  ch{ch}: added {int(added.sum()):,}; in majority-kept context {susp:,}",
              flush=True)

    print("\n=== BARS ===", flush=True)
    names = {1: "synthetic <=3% each", 2: "gold regression <=0.3pp",
             3: "fit regression <=0.3pp", 4: "v13 absolutes (median<=4, parts<=6)",
             5: "guard flags within adjudicated reference"}
    for k in sorted(ok):
        print(f"  {k} ({names[k]}): {'PASS' if ok[k] else 'FAIL'}", flush=True)
    print("\n=== full-page suite (12-instance PSD set, v12 QS) ===", flush=True)
    from v26_fullpage_suite import run_suite
    _, suite_ok = run_suite("ABES", "QS", "v12ABES")
    ok[6] = suite_ok
    print(f"OVERALL: {'PASS' if all(ok.values()) else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
