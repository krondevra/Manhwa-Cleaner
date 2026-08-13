"""Gen9 end-to-end: automated port of the user's Photopea algorithm.

Usage: .venv/bin/python src/gen9/run.py <page.png> [out_dir=.tmp/gen9/out]

Outputs (named after the input stem):
  <stem>_gen9_delete.npy   boolean delete mask
  <stem>_gen9_red.png      source with deleted px red-tinted; Classifier
                           B pocket zones outlined blue for review
  <stem>_gen9_clean.png    deleted px filled white (release style is the
                           user's call; red preview is the review artifact)

Pipeline: deterministic chain (pipeline.py) + Classifier A background
selection + Classifier B glyph-pocket selection -> compose_delete.
No other decision-makers exist; adding one is a stop-and-report event.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def run_page(page_png: str | Path, out_dir: str | Path = None):
    import cv2
    from PIL import Image

    from gen9 import classify_bg, classify_glyph
    from gen9 import pipeline as pl
    Image.MAX_IMAGE_PIXELS = None

    page_png = Path(page_png)
    out = Path(out_dir) if out_dir else page_png.parent / "out"
    out.mkdir(parents=True, exist_ok=True)
    stem = page_png.stem

    src = np.array(Image.open(page_png).convert("RGB"))
    c1 = pl.clone1(src)
    c2 = pl.clone2(src)
    n, lab, st = pl.bg_components(c2)
    sel = classify_bg.select_background(lab, st)
    pockets = classify_glyph.select_pockets(lab, st, sel, c1)
    pocket_ids = [p["comp_id"] for p in pockets]
    delete = pl.compose_delete(src.shape[:2], lab, sel + pocket_ids, c1)

    np.save(out / f"{stem}_gen9_delete.npy", delete)

    red = src.copy()
    red[delete] = (red[delete] * 0.45 + np.array([255, 0, 0]) * 0.55
                   ).astype(np.uint8)
    for p in pockets:
        x0, y0, x1, y1 = p["bbox"]
        cv2.rectangle(red, (max(0, x0 - 3), max(0, y0 - 3)),
                      (min(src.shape[1] - 1, x1 + 3),
                       min(src.shape[0] - 1, y1 + 3)), (0, 80, 255), 2)
    Image.fromarray(red).save(out / f"{stem}_gen9_red.png")

    clean = src.copy()
    clean[delete] = 255
    Image.fromarray(clean).save(out / f"{stem}_gen9_clean.png")

    stats = dict(page=str(page_png), bg_comps=len(sel),
                 pockets=len(pockets), deleted_px=int(delete.sum()),
                 pocket_list=[dict(bbox=p["bbox"], area=p["area"])
                              for p in pockets])
    print(f"{stem}: {len(sel)} bg comps, {len(pockets)} glyph pockets, "
          f"{int(delete.sum())} px deleted -> {out}")
    return delete, stats


if __name__ == "__main__":
    run_page(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
