"""Plan v12 steps 4-5: assemble the deterministic replication of pipeline-v2 and measure it
against the manual reference (005-1_cleaned.png alpha == PSD img mask, verified identical).

Chain (deterministic candidate):
  1. Build hard-white + soft-white masks with the PSD-calibrated builder (lightness
     desaturate, direct-gamma levels, > threshold, square-kernel min/max) -- 99.3-99.7%
     pixel agreement with the PSD's own mask rasters (residual = Photopea's merge-time
     anti-aliased threshold edges under grayscale morphology; diagnosed, documented).
  2. WAND RULE (replaces magic-wand steps 5-9): a mask component is "white background" iff it
     touches the page's left/right edge AND its SOURCE-IMAGE content is paper-white by a
     2-feature rule (frac of px >= 250, edge density) fitted by threshold search on this
     page's own supervised component table (fit-to-this-page is in scope: the deliverable is
     replication of THIS reference). Delete = union of selected components, hard-white stage
     then soft-white stage.
  3. Everything the rules DON'T reproduce vs the reference is measured and decomposed --
     that remainder is the judgment-call share (steps 10-20: tol-200 img wand + frame-protect
     rectangles, expand+lasso, SFX-outline brush, spiky-cloud text brush, misc).

Usage:
  .venv/bin/python src/spiky/replicate_pipeline_v2.py
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
OUT = HERE / "extracted"

SETTINGS = {
    "soft-white": dict(b=10, g=1.50, w=110, thr=48, order="minmax", r1=4, r2=4),
    "hard-white": dict(b=18, g=0.80, w=70, thr=20, order="minmax", r1=22, r2=10),
}


def build_mask(gray_light: np.ndarray, s: dict) -> np.ndarray:
    lv = np.round(255.0 * np.power(np.clip((gray_light - s["b"]) / float(s["w"] - s["b"]),
                                             0, 1), s["g"]))
    m = ((lv > s["thr"]).astype(np.uint8) * 255)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
    if s["order"] == "maxmin":
        m = cv2.dilate(m, k(s["r1"]))
        m = cv2.erode(m, k(s["r2"]))
    else:
        m = cv2.erode(m, k(s["r1"]))
        m = cv2.dilate(m, k(s["r2"]))
    return m >= 128


def component_table(mask: np.ndarray, gray: np.ndarray, edge: np.ndarray,
                     gt_delete: np.ndarray):
    """Edge-touching components >=1000px with source-image features + GT selection label."""
    H, W = mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 4)
    rows = []
    for lbl in range(1, num):
        x, y, cw, chh, area = stats[lbl]
        if area < 1000:
            continue
        if not (x == 0 or x + cw == W):
            continue
        comp = labels[y : y + chh, x : x + cw] == lbl
        g = gray[y : y + chh, x : x + cw][comp]
        frac250 = float((g >= 250).mean())
        edge_frac = float(edge[y : y + chh, x : x + cw][comp].mean())
        dfrac = float(gt_delete[y : y + chh, x : x + cw][comp].mean())
        rows.append(dict(lbl=lbl, area=int(area), frac250=frac250, edge_frac=edge_frac,
                          dfrac=dfrac))
    return num, labels, stats, rows


def fit_rule(rows) -> tuple[float, float, float, float]:
    """Threshold search: selected iff frac250 >= t1 OR (frac250 >= t2 AND edge <= t3).
    Minimizes PER-PIXEL cost: selecting a component costs its GT-keep pixels (over-deletion),
    skipping it costs its GT-delete pixels (under-deletion). Partial/merged components are
    thereby handled correctly instead of being excluded (the attempt-1 bug: a replica
    component bridging a selected+unselected pair was excluded from the fit yet still
    predicted, wholesale-deleting its kept half)."""
    best = None
    f_grid = sorted(set(round(r["frac250"], 3) for r in rows)) + [1.001]
    e_grid = sorted(set(round(r["edge_frac"], 4) for r in rows)) + [1.0]
    for t1 in f_grid:
        for t2 in [t for t in f_grid if t <= t1]:
            for t3 in e_grid:
                cost = 0.0
                for r in rows:
                    pred = (r["frac250"] >= t1) or (r["frac250"] >= t2 and
                                                     r["edge_frac"] <= t3)
                    cost += (1.0 - r["dfrac"]) * r["area"] if pred else r["dfrac"] * r["area"]
                if best is None or cost < best[0]:
                    best = (cost, t1, t2, t3)
    return best


def main() -> None:
    """FINAL chain (2026-08-07, measured -- see pipeline-v2.md addendum):
      1. hard-white mask -> edge-touching components passing the paper-white rule = SEED
         (replicates wand steps 5-7; seed over-deletion measured at 215 px on this page).
      2. soft-white mask -> components substantially overlapping the seed (>=5000 px AND
         >=20%) define the candidate EXTENT (replicates the wand's contiguous grab; these
         components genuinely merge background with panel interiors through bright bridges --
         the same over-selection the prose's step 11 corrects by hand).
      3. Deletion = seed + (extent AND gray >= 250): per-pixel paper-white gating replicates
         the deletion the reference actually shows inside those merged regions.
      4. repair_frame_interiors (production machinery) protects fully-enclosed panel
         interiors (deterministic replacement for half of the manual step-11 rectangles).
    Result vs the manual reference: over 2.32% / under 0.49% / TOTAL 2.82% of page.
    The residual is the measured judgment share (broken-frame leaks incl. the white document
    panel, SFX-outline and cloud-text brushwork)."""
    import subprocess
    import sys as _sys
    print(subprocess.run(["date"], capture_output=True, text=True).stdout.strip(), flush=True)
    _sys.path.insert(0, str(HERE.parents[1] / "src"))
    import ml_cleaner
    _sys.modules["__main__"].train_command = ml_cleaner.train_command
    from ml_cleaner import repair_frame_interiors

    rgb_u8 = np.asarray(Image.open(HERE / "005-1.png").convert("RGB"))
    rgb = rgb_u8.astype(np.float32)
    gray_light = (rgb.max(axis=2) + rgb.min(axis=2)) / 2.0
    gray = np.round(gray_light).astype(np.uint8)
    gt_delete = np.asarray(Image.open(HERE / "005-1_cleaned.png").split()[-1]) < 128
    total = gt_delete.size

    sob = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3) ** 2
                   + cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3) ** 2)
    edge = sob > 30

    # 1. hard-white seed
    hard = build_mask(gray_light, SETTINGS["hard-white"])
    num, labels, stats, rows = component_table(hard, gray, edge, gt_delete)
    cost, t1, t2, t3 = fit_rule(rows)
    seed = np.zeros_like(gt_delete)
    for r in rows:
        if (r["frac250"] >= t1) or (r["frac250"] >= t2 and r["edge_frac"] <= t3):
            lbl = r["lbl"]
            x, y, cw, chh, _ = stats[lbl]
            seed[y : y + chh, :][labels[y : y + chh, :] == lbl] = True
    print(f"seed rule: frac250>={t1} OR (frac250>={t2} AND edge<={t3});  "
          f"seed px {int(seed.sum()):,}, over {int((seed & ~gt_delete).sum())} px", flush=True)

    # 2. soft-white extent (substantial-overlap gating)
    soft = build_mask(gray_light, SETTINGS["soft-white"])
    num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(soft.astype(np.uint8), 4)
    overlap = np.bincount(labels_s[seed & soft], minlength=num_s)
    sel_labels = [l for l in range(1, num_s)
                  if overlap[l] >= 5000 and overlap[l] >= 0.2 * stats_s[l, 4]]
    extent = np.isin(labels_s, np.array(sel_labels))

    # 3. per-pixel paper-white deletion inside the extent
    delete = seed | (extent & (gray >= 250))

    # 4. enclosed-interior protection
    delete = repair_frame_interiors(rgb_u8, delete, frame_darkness=40,
                                     min_interior_px=10000, inset_px=2)

    over = int((delete & ~gt_delete).sum())
    under = int((~delete & gt_delete).sum())
    print(f"\nFINAL vs manual reference: over {100*over/total:.4f}%  "
          f"under {100*under/total:.4f}%  total {100*(over+under)/total:.4f}%", flush=True)
    np.save(OUT / "final_delete.npy", delete)

    red_view = rgb_u8.copy()
    red_view[delete] = (255, 0, 0)
    Image.fromarray(red_view).save(OUT / "replica_cleaned_red.png")
    print(f"red view: {OUT / 'replica_cleaned_red.png'}", flush=True)

    for tag, m in (("over(we deleted, ref kept)", delete & ~gt_delete),
                    ("under(ref deleted, we kept)", ~delete & gt_delete)):
        n = int(m.sum())
        num2, _, stats2, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        areas = sorted(stats2[1:, 4], reverse=True)
        print(f"  {tag}: {n:,} px ({100*n/total:.4f}%)  components {num2-1}  "
              f"top {areas[:6]}", flush=True)


if __name__ == "__main__":
    main()
