#!/usr/bin/env python3
"""Single CLI dispatch wrapper for the halo investigation's repeated actions
(.claude/plans/snazzy-cuddling-creek.md Part 0.2, built 2026-08-04) -- replaces re-explaining
the same few commands in prose each session with one flagged entry point. Same spirit as
.tmp/diagnostics/run_ladder.sh's "single entry point for a repeated ladder" convention, scoped
to this investigation's own tools. Thin dispatch layer: shells out to the existing, already-
tested scripts with the right arguments -- does not reimplement their logic.

Usage:
  # manual-clean reference comparison (chapters 001/002, arbitrary --chain since Part A;
  # optional --crf-weights layers attempt 8's CRF refiner on top of --chain, since 2026-08-05)
  python halo_tools.py --eval-manual-clean --checkpoint PATH [--label NAME] [--chain none|islands|full] [--crf-weights PATH]

  # 5-instance ring-distance / point-boundary check -- --mechanism selects which real-instance
  # script/checkpoint-naming-convention to use (contour: attempt 7, crf: attempt 8; default contour)
  python halo_tools.py --eval-ring-distance --tag CHECKPOINT_TAG [--mechanism contour|crf]

  # build training data (if needed) + train a contour-deformation checkpoint at a given scale
  python halo_tools.py --train --tag NAME --scale {smoke250,1k,2k}
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH = ROOT / "src/research"
DIAGNOSTICS = ROOT / ".tmp/diagnostics"
CHECKPOINT_DIR = ROOT / ".tmp/checkpoints/contour_deform_smoke"

# scale -> (limit_examples, limit_pages, val_examples, val_pages), matching this week's
# established convention (notes/instance_aware_pivot_2026-08-04.md)
SCALE_PARAMS = {
    "smoke250": (250, 150, 60, 40),
    "1k": (1000, 700, 200, 150),
    "2k": (2000, 1400, 400, 300),
}


def run(cmd: list[str], **kwargs) -> int:
    print(f"[halo_tools] running: {' '.join(str(c) for c in cmd)}")
    return subprocess.call(cmd, **kwargs)


def eval_manual_clean(checkpoint: Path, label: str | None, chain: str | None,
                       crf_weights: Path | None = None) -> int:
    """Wraps eval_gen6_checkpoint.py as-is. As of 2026-08-05 (Part A landed) that script
    supports chapters 001/002 (035 excluded, dimension mismatch -- see its own module
    docstring) and an arbitrary --chain in {none, islands, full}. --crf-weights (attempt 8,
    2026-08-05) layers CRFRefineNet on top of --chain, orthogonal to the chain choice."""
    cmd = [sys.executable, str(RESEARCH / "eval_gen6_checkpoint.py"), "--model", str(checkpoint)]
    if label:
        cmd += ["--label", label]
    if chain:
        cmd += ["--chain", chain]
    if crf_weights:
        cmd += ["--crf-weights", str(crf_weights)]
    return run(cmd)


def eval_ring_distance(tag: str, mechanism: str = "contour") -> int:
    """Wraps the mechanism-appropriate real-instance ring-distance script via its own
    established CKPT_TAG env-var interface (unchanged) -- 'contour' (attempt 7, default,
    preserves prior behavior) wraps contour_deform_real_instance_check.py (CONTOUR_CKPT_TAG);
    'crf' (attempt 8, 2026-08-05) wraps crf_refine_real_instance_check.py (CRF_CKPT_TAG),
    resolving .tmp/checkpoints/crf_refine_{tag}/crf_{tag}.pt, same naming convention already
    used for the crf smoke/1k runs."""
    import os
    env = dict(os.environ)
    if mechanism == "crf":
        env["CRF_CKPT_TAG"] = tag
        cmd = [sys.executable, str(DIAGNOSTICS / "crf_refine_real_instance_check.py")]
    else:
        env["CONTOUR_CKPT_TAG"] = tag
        cmd = [sys.executable, str(DIAGNOSTICS / "contour_deform_real_instance_check.py")]
    return run(cmd, env=env)


def train(tag: str, scale: str) -> int:
    """Builds training data at the given scale (if not already on disk) then trains a
    contour-deformation checkpoint with that tag -- wraps build_contour_training_data.py +
    contour_deform_net.py, both unchanged."""
    limit_ex, limit_pages, val_ex, val_pages = SCALE_PARAMS[scale]
    train_npz = CHECKPOINT_DIR / f"train_{tag}.npz"
    val_npz = CHECKPOINT_DIR / f"val_{tag}.npz"

    if train_npz.exists() and val_npz.exists():
        print(f"[halo_tools] {train_npz.name}/{val_npz.name} already exist, skipping data build "
              f"(delete them first to force a rebuild).")
    else:
        rc = run([
            sys.executable, str(RESEARCH / "build_contour_training_data.py"),
            "--tag", tag,
            "--limit-examples", str(limit_ex), "--limit-pages", str(limit_pages),
            "--val-examples", str(val_ex), "--val-pages", str(val_pages),
        ])
        if rc != 0:
            print(f"[halo_tools] data build failed (exit {rc}), not proceeding to training.")
            return rc

    epochs = {"smoke250": 15, "1k": 20, "2k": 25}[scale]
    return run([
        sys.executable, "-u", str(RESEARCH / "contour_deform_net.py"),
        "--tag", tag, "--epochs", str(epochs), "--batch-size", "8",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval-manual-clean", action="store_true")
    mode.add_argument("--eval-ring-distance", action="store_true")
    mode.add_argument("--train", action="store_true")

    ap.add_argument("--checkpoint", type=Path, help="checkpoint path (--eval-manual-clean)")
    ap.add_argument("--label", default=None, help="display label (--eval-manual-clean)")
    ap.add_argument("--tag", help="checkpoint/data tag (--eval-ring-distance, --train)")
    ap.add_argument("--scale", choices=list(SCALE_PARAMS), help="data scale (--train)")
    ap.add_argument("--chain", default=None, choices=["none", "islands", "full"],
                     help="postprocess chain (--eval-manual-clean); default matches "
                          "eval_gen6_checkpoint.py's own default (islands)")
    ap.add_argument("--crf-weights", type=Path, default=None,
                     help="attempt 8: CRFRefineNet checkpoint, layered on top of --chain "
                          "(--eval-manual-clean only)")
    ap.add_argument("--mechanism", default="contour", choices=["contour", "crf"],
                     help="which real-instance script/naming-convention to use "
                          "(--eval-ring-distance only; default contour, attempt 7)")
    args = ap.parse_args()

    if args.eval_manual_clean:
        if not args.checkpoint:
            ap.error("--eval-manual-clean requires --checkpoint")
        return eval_manual_clean(args.checkpoint, args.label, args.chain, args.crf_weights)

    if args.eval_ring_distance:
        if not args.tag:
            ap.error("--eval-ring-distance requires --tag")
        return eval_ring_distance(args.tag, args.mechanism)

    if args.train:
        if not args.tag or not args.scale:
            ap.error("--train requires --tag and --scale")
        return train(args.tag, args.scale)

    return 1


if __name__ == "__main__":
    sys.exit(main())
