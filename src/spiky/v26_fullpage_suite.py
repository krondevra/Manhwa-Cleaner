"""Standing full-page regression suite over the 12-instance PSD diagnostic set.

Infrastructure (plan v26 Part 1): given a pipeline module name + steps string, runs the
FULL chapters 002 and 019 once each (masks cached in suite_cache/ keyed by config name),
slices the 12 diagnostic crops and scores over%/under% vs the PSD-etalon composited
alpha (<128 = delete; per the v25 019_5 anomaly, never decode layer.mask directly).

Bars (v26): clean sites (019_1/5/7/8/9) over <= 0.3%; defect-class instances must not be
worse than the v10-S reference by > 0.3pp on either axis (and the fix rounds expect them
to IMPROVE). Reference v10-S numbers are embedded from the v25 diagnosis table.

Usage:
  .venv/bin/python v26_fullpage_suite.py <module> <steps> <config_name>
e.g.
  .venv/bin/python v26_fullpage_suite.py replicate_pipeline_v10 S v10S
Importable: run_suite(module_name, steps, config_name) -> list of row dicts.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Promoted from .tmp/scripts-manual (2026-08-10 cleanup): code is git-tracked here,
# data/caches stay in the gitignored .tmp/.
TMP = HERE.parents[1] / ".tmp"
DIAG = TMP / "debug/spiky-clouds-diagnostics-psd"
CACHE = TMP / "scripts-manual/suite_cache"
CACHE.mkdir(parents=True, exist_ok=True)

PAGES = {
    "002": str(TMP / "debug/minmax/other/002.png"),
    "019": str(TMP / "eval/019.png"),
}

# (folder, chapter, crop y0, initial filename)
INSTANCES = [
    ("002_5_y66008", "002", 66008, "002_5_y66008_initial.png"),
    ("002_6_y67551", "002", 67551, "002_6_y67551_initial.png"),
    ("019_0_spiky_y1430", "019", 1430, "019_0_spiky_y1430.png"),
    ("019_1_spiky_y4808", "019", 4808, "019_1_spiky_y4808.png"),
    ("019_2_spiky_y6696", "019", 6696, "019_2_spiky_y6696.png"),
    ("019_3_spiky_y11089", "019", 11089, "019_3_spiky_y11089.png"),
    ("019_4_spiky_y15364", "019", 15364, "019_4_spiky_y15364.png"),
    ("019_5_spiky_y30151", "019", 30151, "019_5_spiky_y30151.png"),
    ("019_6_spiky_y31744", "019", 31744, "019_6_spiky_y31744.png"),
    ("019_7_spiky_y33364", "019", 33364, "019_7_spiky_y33364.png"),
    ("019_8_spiky_y36295", "019", 36295, "019_8_spiky_y36295.png"),
    ("019_9_spiky_y38129", "019", 38129, "019_9_spiky_y38129.png"),
]
CLEAN_SITES = {"019_1_spiky_y4808", "019_5_spiky_y30151", "019_7_spiky_y33364",
               "019_8_spiky_y36295", "019_9_spiky_y38129"}
# v10-S reference (v25 diagnosis table): name -> (over%, under%)
V10S_REF = {
    "002_5_y66008": (0.133, 21.530), "002_6_y67551": (0.125, 18.342),
    "019_0_spiky_y1430": (11.179, 0.039), "019_1_spiky_y4808": (0.137, 1.075),
    "019_2_spiky_y6696": (5.152, 0.111), "019_3_spiky_y11089": (5.339, 0.352),
    "019_4_spiky_y15364": (2.872, 0.127), "019_5_spiky_y30151": (0.059, 0.186),
    "019_6_spiky_y31744": (10.486, 0.129), "019_7_spiky_y33364": (0.294, 0.143),
    "019_8_spiky_y36295": (0.000, 0.139), "019_9_spiky_y38129": (0.000, 0.173),
}


def _page_mask(module_name: str, steps: str, config_name: str, ch: str) -> np.ndarray:
    p = CACHE / f"{config_name}_{ch}.npy"
    if p.exists():
        return np.load(p)
    mod = importlib.import_module(module_name)
    rgb = np.asarray(Image.open(PAGES[ch]).convert("RGB"))
    d = mod.clean_page(rgb, steps=steps)
    np.save(p, d)
    return d


def run_suite(module_name: str, steps: str, config_name: str, verbose: bool = True):
    masks = {ch: _page_mask(module_name, steps, config_name, ch) for ch in PAGES}
    rows = []
    ok = True
    for name, ch, y0, _init in INSTANCES:
        et = np.asarray(Image.open(DIAG / name / f"{name}_etalon.png"))
        gt_del = et[:, :, 3] < 128
        H = gt_del.shape[0]
        d = masks[ch][y0:y0 + H]
        total = gt_del.size
        over = 100 * int((d & ~gt_del).sum()) / total
        under = 100 * int((~d & gt_del).sum()) / total
        ro, ru = V10S_REF[name]
        flags = []
        if name in CLEAN_SITES and over > 0.3:
            flags.append("CLEAN-BAR")
            ok = False
        if over > ro + 0.3 or under > ru + 0.3:
            flags.append("WORSE-THAN-v10S")
            ok = False
        rows.append(dict(name=name, over=over, under=under, flags=flags))
        if verbose:
            print(f"  {name:22s} over {over:7.3f}%  under {under:7.3f}%  "
                  f"(v10S {ro:.3f}/{ru:.3f}) {' '.join(flags)}", flush=True)
    if verbose:
        print(f"SUITE [{config_name}]: {'PASS' if ok else 'FAIL'}", flush=True)
    return rows, ok


if __name__ == "__main__":
    import subprocess
    print(subprocess.run(["date"], capture_output=True, text=True).stdout.strip(), flush=True)
    run_suite(sys.argv[1], sys.argv[2], sys.argv[3])
