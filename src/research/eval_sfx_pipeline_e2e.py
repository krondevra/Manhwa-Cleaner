"""Part 1.4: end-to-end whole-page test. Unlike last night's eval_instance_sfx_real.py (which
hand-specified each instance's bounding box and centered a crop via a simple ink-bbox lookup),
this runs the REAL find_sfx_instances() detector on a region around each of the 6 known real
instances and checks: does the detector actually find it (recall), and if so, does the
end-to-end pipeline (detect -> crop -> TinyInstanceNet -> paste back) produce the same quality
result last night's hand-specified-box evaluation showed?

Also reports the dense-checkpoint baseline (stage3_sfx_2k_resumed) at the same bbox for direct
comparison, and the changed_frac_in_crop diagnostic from apply_sfx_instance_refine so any
unexpected change in the surrounding margin (not just the object's own bbox) is visible, not
just assumed benign.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_cleaner  # noqa: E402
sys.modules["__main__"].train_command = ml_cleaner.train_command
from ml_cleaner import GuidanceParams, choose_device, load_model, predict_delete_mask  # noqa: E402
from sfx_instance_pipeline import apply_sfx_instance_refine  # noqa: E402
from build_sfx_instance_crops import find_sfx_instances  # noqa: E402

DENSE_CKPT = ROOT / ".tmp/checkpoints/stage3/stage3_sfx_2k/out/stage3_sfx_2k_resumed.pt"
PAD = 300

# Same 6 instances as last night's eval_instance_sfx_real.py, (label, chapter, x0,y0,x1,y1) --
# x0..y1 is the outer search box, not the tight glyph box (matches ch1_sfx_text's own registry
# convention).
INSTANCES = [
    ("ch1_sfx_text (tracked)", "001.png", 0, 44900, 130, 45080),
    ("real_cand_0 (ch1, Korean glyph+glow)", "001.png", 143, 43920, 243, 44054),
    ("real_cand_1 (ch1, SFX glow strokes)", "001.png", 510, 48750, 610, 48981),
    ("real_cand_4 (ch2, stick-figure SFX icon)", "002.png", 428, 113012, 585, 113205),
    ("real_cand_5 (ch2, Korean SFX text)", "002.png", 502, 25011, 633, 25151),
    ("real_cand_6 (ch2, SFX glyph)", "002.png", 224, 49125, 345, 49282),
]


def boxes_overlap(b1, b2) -> bool:
    x0, y0, w0, h0 = b1
    x1, y1, w1, h1 = b2
    return not (x0 + w0 < x1 or x1 + w1 < x0 or y0 + h0 < y1 or y1 + h1 < y0)


def main() -> None:
    device = choose_device("auto")
    dense_model, config = load_model(DENSE_CKPT, device)
    threshold = float(config.get("threshold", 0.5))
    gp = GuidanceParams(threshold_value=int(config.get("threshold_value", 30)),
                         morph_radius=int(config.get("morph_radius", 2)))

    print("=== Part 1.4: end-to-end whole-page SFX pipeline, real detector (not hand-specified boxes) ===\n")
    n_found, n_total = 0, len(INSTANCES)

    for label, chapter_file, x0, y0, x1, y1 in INSTANCES:
        chapter_path = ROOT / "data/chapters-initial" / chapter_file
        full = Image.open(chapter_path)
        W, H = full.size
        rx0, ry0 = max(0, x0 - PAD), max(0, y0 - PAD)
        rx1, ry1 = min(W, x1 + PAD), min(H, y1 + PAD)
        region = np.asarray(full.crop((rx0, ry0, rx1, ry1)).convert("RGB"))

        target_box = (x0 - rx0, y0 - ry0, x1 - x0, y1 - y0)
        boxes = find_sfx_instances(region)
        match = next((b for b in boxes if boxes_overlap(b, target_box)), None)

        if match is None:
            print(f"{label}: NOT FOUND by detector ({len(boxes)} other candidates in region) -- RECALL MISS")
            n_total_matches = 0
            continue
        n_found += 1

        dense_raw = predict_delete_mask(rgb=region, model=dense_model, device=device,
                                         guidance_params=gp, tile_size=768, overlap=96,
                                         threshold=threshold, amp=False)
        mx, my, mw, mh = match
        dense_frac = float(dense_raw[my:my + mh, mx:mx + mw].mean())

        refined, info = apply_sfx_instance_refine(region, dense_raw)
        this_info = next((i for i in info if i["bbox"] == match), None)

        print(f"{label}:")
        print(f"  detector found bbox {match} (target was {target_box})")
        print(f"  dense baseline delete_frac at bbox:     {dense_frac:.4f}")
        if this_info:
            print(f"  instance-model delete_frac at bbox:     {this_info['instance_delete_frac_at_bbox']:.4f}")
            print(f"  changed_frac across whole 224px crop:   {this_info['changed_frac_in_crop']:.4f}")
        print()

    print(f"=== Recall: {n_found}/{n_total} known real instances found by the detector ===")
    print("\nFor comparison, last night's hand-specified-box eval (eval_instance_sfx_real.py, "
          "with_bg_weighted): 6/6 PASS, mean_prob 0.09-0.29 (measured at the exact ink bbox, "
          "not the detector's own proposal box -- may differ slightly in extent).")


if __name__ == "__main__":
    main()
