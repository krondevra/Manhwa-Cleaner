"""Two-directional GT eval against the manual-reference chapters (001, 002, 035 -- see
GT_HEIGHT_OVERRIDE for 035's prefix-crop) -- investigation-only use of the human-cleaned
references, never training data.

2026-08-05 (halo investigation, Part A of .claude/plans/snazzy-cuddling-creek.md): this is now
the PRIMARY success metric for any halo-mechanism attempt (previously the 5-instance
ring-distance check was primary; that check stays as a secondary diagnostic, proven good at the
specific context-dependence/independence question, but a single-instance-scale check was never a
substitute for measuring against the actual manual-clean quality bar). Extended from the
original raw/+islands-only version: `eval_checkpoint` now accepts an arbitrary
`postprocess(rgb, mask) -> mask` callable (same convention as
`real_boundary_probe.py::measure_instance` / `bubble_instance_real_instance_check.py`) instead of
hardcoding islands as the only option. Chapter 035 (initially excluded for an apparent
misalignment) was re-diagnosed and added 2026-08-05 (mission plan Phase 0b) -- see
GT_HEIGHT_OVERRIDE's comment for the real explanation (a prefix relationship, not a shift).

No banding/MARGIN logic needed (Part 3's correction, prior session): predict_delete_mask already
tiles arbitrary input sizes internally (pad_image_for_tiling + Hann-window blend), confirmed
working on full merged GT strips directly.

Usage:
  # evaluate one checkpoint+chain against all 3 chapters, with aggregate
  .venv/bin/python src/research/eval_gen6_checkpoint.py --model PATH --chain {none,islands,full}

  # run both standing reference configs together (10.0-baseline+islands,
  # gen6-combined+full-chain) for a side-by-side manual-clean baseline comparison
  .venv/bin/python src/research/eval_gen6_checkpoint.py --baselines
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import ml_cleaner  # noqa: E402

sys.modules["__main__"].train_command = ml_cleaner.train_command

from ml_cleaner import (  # noqa: E402
    GuidanceParams, choose_device, load_model, predict_delete_mask,
    reclaim_landlocked_delete_islands, repair_frame_interiors, close_bubble_halo,
)

# 2026-08-05 (mission plan Phase 0b): chapter 035's apparent 480px "misalignment"
# (.tmp/saved/chapters/035.png=162856px tall vs 035_cleaned.png=162376px) was mis-diagnosed in an
# earlier session as a shift/seam problem from a coarse 200-row-sample probe. A proper masked
# kept-pixel-only row comparison (.tmp/diagnostics/error_decomposition.py's approach, verified
# separately) found NO shift needed at all: 99.99% of valid (kept-content) rows match at shift=0
# with near-zero diff (mean 0.23/255, 99th pct 1.12/255 -- compression noise, not misalignment).
# The cleaned reference is simply a PREFIX of the original: 035_cleaned.png = 035.png[:162376].
# The extra 480 rows are a real, un-cleaned promotional cover/credits page appended after the
# actual chapter content (confirmed visually -- title text + a VK.com social link), which the
# manual cleaner correctly never touched. GT_HEIGHT_OVERRIDE crops the original to the cleaned
# reference's own height before comparison for any chapter where this is verified true.
GT_CHAPTERS = ["001", "002", "035"]
GT_HEIGHT_OVERRIDE = {"035": 162376}
GT_DIR = ROOT / ".tmp/saved/chapters"
PRODUCTION_MODEL = ROOT / "data/models/10.0-baseline.pt"
GEN6_COMBINED_MODEL = ROOT / ".tmp/checkpoints/stage3/stage3_sfx_2k/out/stage3_sfx_2k_resumed.pt"

# Established defaults from this project's own recommended postprocess chain
# (process_command / real_boundary_probe.py), reused verbatim, not reinvented.
REPAIR_FRAMES_KWARGS = dict(frame_darkness=40, min_interior_px=10000, inset_px=2)
CLOSE_BUBBLE_HALO_KWARGS = dict(ring_width=24, frame_darkness=40, min_bubble_area=2000,
                                 min_background_area=8000)


def chain_none(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return mask


def chain_islands(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return reclaim_landlocked_delete_islands(mask)


def chain_islands_repair(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mission plan v5, Phase 2b: islands -> repair-frames only, no close-bubble-halo -- isolates
    repair_frame_interiors' own marginal contribution (including its effect on release blocker #1,
    the HUD-panel over-deletion class, whose UI text boxes are exactly the enclosed-interior
    content this step protects)."""
    m = reclaim_landlocked_delete_islands(mask)
    m = repair_frame_interiors(rgb, m, **REPAIR_FRAMES_KWARGS)
    return m


def chain_full(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The full recommended gen-6 chain: islands -> repair-frames -> close-bubble-halo, same
    order process_command and real_boundary_probe.py's own postprocess chain already use."""
    m = reclaim_landlocked_delete_islands(mask)
    m = repair_frame_interiors(rgb, m, **REPAIR_FRAMES_KWARGS)
    m = close_bubble_halo(rgb, m, **CLOSE_BUBBLE_HALO_KWARGS)
    return m


def chain_islands_backdrop(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mission plan Phase 1: islands -> reclaim_black_backdrop (the dominant under-deletion
    lever identified by error_decomposition.py -- large uniform near-black backdrop strips)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reclaim_black_backdrop import reclaim_black_backdrop
    m = reclaim_landlocked_delete_islands(mask)
    m = reclaim_black_backdrop(rgb, m)
    return m


def chain_recipe_a(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mission plan v9 Phase F (2026-08-06): the adopted Recipe A chain, assembled by offline
    ablation (.tmp/diagnostics/recipe_a_ablation.py, one step at a time on cached v3+islands
    masks): islands -> reclaim_patchy_deletion -> d1_region_vote -> repair_frame_interiors ->
    close_bubble_halo. Offline: total 14.2294% -> 12.4955% (over 2.5841%, under 9.9114%),
    class mean 36.37% -> 20.71%. SFX instance protect measured and EXCLUDED (+1.50pp aggregate
    regression on this checkpoint -- it keeps material the manual reference deletes); P+D order
    beat D+P (13.0608% vs 13.1193%)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reclaim_patchy_deletion import reclaim_patchy_deletion
    from d1_region_vote import d1_region_vote
    m = reclaim_landlocked_delete_islands(mask)
    m = reclaim_patchy_deletion(rgb, m)
    m = d1_region_vote(rgb, m)
    m = repair_frame_interiors(rgb, m, **REPAIR_FRAMES_KWARGS)
    m = close_bubble_halo(rgb, m, **CLOSE_BUBBLE_HALO_KWARGS)
    return m


def chain_islands_patchy(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mission plan v9 (blocker #1, adopted 2026-08-06): islands ->
    reclaim_patchy_deletion -- per-pixel texture+kept-density gated reclaim of wrongly-deleted
    TEXTURED dark art (the sub-class of the dark-panel defect that is locally separable from
    true flat gutters; the flat-digital-paint sub-class is a documented irreducible residual).
    Offline gate on cached v3+islands masks: class mean 36.37%->20.72%, aggregate total
    14.23%->14.08%, under-del +0.298pp (within the 0.3pp budget)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reclaim_patchy_deletion import reclaim_patchy_deletion
    m = reclaim_landlocked_delete_islands(mask)
    m = reclaim_patchy_deletion(rgb, m)
    return m


CHAINS = {"none": chain_none, "islands": chain_islands, "full": chain_full,
          "islands_backdrop": chain_islands_backdrop, "islands_repair": chain_islands_repair,
          "islands_patchy": chain_islands_patchy, "recipe_a": chain_recipe_a}


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.4f}%"


def eval_checkpoint(model_path: Path, device: torch.device, label: str, chain_name: str,
                     crf_weights: Path | None = None, sfx_weights: Path | None = None) -> dict:
    """Returns {chapter: {"raw": {...}, tag_name: {...}}} plus a pixel-weighted "aggregate"
    key computed across all chapters for each of "raw" and tag_name.

    `crf_weights` (attempt 8, 2026-08-05) / `sfx_weights` (2026-08-05, the already-proven SFX
    instance-scoped refiner, `sfx_instance_pipeline.py`): both optional and orthogonal to
    `chain_name` -- when given, wrap whichever chain was selected with an additional refine step
    (not a new CHAINS enum value), exactly the postprocess(rgb, mask) -> mask generalization this
    function was already built to support. Both may be combined."""
    model, config = load_model(model_path, device)
    threshold = float(config.get("threshold", 0.5))
    gp = GuidanceParams(threshold_value=int(config.get("threshold_value", 30)),
                         morph_radius=int(config.get("morph_radius", 2)))
    base_chain_fn = CHAINS[chain_name]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    suffix = ""
    steps = []
    if crf_weights is not None:
        from crf_instance_pipeline import apply_crf_refine
        steps.append(lambda rgb, m: apply_crf_refine(rgb, m, crf_weights, device))
        suffix += "+crf"
    if sfx_weights is not None:
        from sfx_instance_pipeline import apply_sfx_instance_refine, load_sfx_instance_model
        sfx_model = load_sfx_instance_model(sfx_weights)
        steps.append(lambda rgb, m: apply_sfx_instance_refine(rgb, m, sfx_model)[0])
        suffix += "+sfx"

    if steps:
        def chain_fn(rgb, mask):
            m = base_chain_fn(rgb, mask) if chain_name != "none" else mask
            for step in steps:
                m = step(rgb, m)
            return m

        tag_name = f"{chain_name}{suffix}"
    else:
        chain_fn = base_chain_fn
        tag_name = chain_name

    results: dict[str, dict] = {}
    agg = {"raw": {"over": 0, "under": 0, "total": 0}, tag_name: {"over": 0, "under": 0, "total": 0}}

    for ch in GT_CHAPTERS:
        rgb = np.asarray(Image.open(GT_DIR / f"{ch}.png").convert("RGB"))
        if ch in GT_HEIGHT_OVERRIDE:
            rgb = rgb[: GT_HEIGHT_OVERRIDE[ch]]
        gt_alpha = np.asarray(Image.open(GT_DIR / f"{ch}_cleaned.png").split()[-1])
        gt_delete = gt_alpha < 128
        total = gt_delete.size

        raw = predict_delete_mask(
            rgb=rgb, model=model, device=device, guidance_params=gp,
            tile_size=768, overlap=96, threshold=threshold, amp=False,
        )
        chained = chain_fn(rgb, raw) if (chain_name != "none" or steps) else raw

        print(f"\n=== {label} -- chapter {ch} ({rgb.shape[1]}x{rgb.shape[0]}) "
              f"GT delete share {pct(int(gt_delete.sum()), total)} ===")
        ch_result = {}
        for tag, m in (("raw", raw), (tag_name, chained)):
            over = int(np.count_nonzero(m & ~gt_delete))
            under = int(np.count_nonzero(~m & gt_delete))
            total_err = over + under
            print(f"  {tag:10s}: over-del {over:>9} ({pct(over, total)})  "
                  f"under-del {under:>9} ({pct(under, total)})  total {pct(total_err, total)}")
            ch_result[tag] = {"over": over, "under": under, "total_pct": 100.0 * total_err / total}
            agg[tag]["over"] += over
            agg[tag]["under"] += under
            agg[tag]["total"] += total
        results[ch] = ch_result

    results["aggregate"] = {}
    print(f"\n=== {label} -- AGGREGATE (pixel-weighted across {', '.join(GT_CHAPTERS)}) ===")
    for tag in ("raw", tag_name):
        a = agg[tag]
        total_err = a["over"] + a["under"]
        total_pct = 100.0 * total_err / a["total"]
        print(f"  {tag:10s}: over-del {pct(a['over'], a['total'])}  "
              f"under-del {pct(a['under'], a['total'])}  total {total_pct:.4f}%")
        results["aggregate"][tag] = {"over": a["over"], "under": a["under"], "total_pct": total_pct}

    return results


def run_baselines(device: torch.device) -> None:
    """Both standing reference configs together: 10.0-baseline+islands (current production
    recipe) and the gen-6 Stage1+2+3 combined checkpoint + the full recommended chain."""
    prod = eval_checkpoint(PRODUCTION_MODEL, device, "10.0-baseline", "islands")
    gen6 = eval_checkpoint(GEN6_COMBINED_MODEL, device, "gen6-combined", "full")

    print(f"\n=== side-by-side: absolute distance from manual-clean quality ===")
    for ch in GT_CHAPTERS + ["aggregate"]:
        prod_pct = prod[ch]["islands"]["total_pct"]
        gen6_pct = gen6[ch]["full"]["total_pct"]
        print(f"  {ch:10s}: 10.0-baseline+islands={prod_pct:.4f}%   "
              f"gen6-combined+full={gen6_pct:.4f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, help="checkpoint to evaluate")
    ap.add_argument("--label", default=None, help="Label for the checkpoint in output (default: filename)")
    ap.add_argument("--chain", default="islands", choices=list(CHAINS),
                     help="postprocess chain to apply on top of the raw prediction")
    ap.add_argument("--crf-weights", type=Path, default=None,
                     help="attempt 8: optional CRFRefineNet checkpoint applied on top of "
                          "--chain (orthogonal to --chain, not a new chain choice)")
    ap.add_argument("--sfx-weights", type=Path, default=None,
                     help="optional SFX instance-scoped refiner checkpoint (sfx_instance_pipeline.py) "
                          "applied on top of --chain (and --crf-weights, if both given)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--baselines", action="store_true",
                     help="Run both standing reference configs (10.0-baseline+islands, "
                          "gen6-combined+full) together instead of evaluating --model")
    args = ap.parse_args()

    device = choose_device(args.device)
    print(f"device: {device}")

    if args.baselines:
        run_baselines(device)
        return

    if not args.model:
        ap.error("--model is required unless --baselines is given")

    label = args.label or args.model.stem
    eval_checkpoint(args.model, device, label, args.chain,
                     crf_weights=args.crf_weights, sfx_weights=args.sfx_weights)


if __name__ == "__main__":
    main()
