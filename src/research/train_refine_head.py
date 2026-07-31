"""Bounded pilot: train `RefineHead` (SmallUNet's self-contained coarse+refine
second stage, src/ml_cleaner.py) via synthetic ground-truth-mask perturbation,
warm-started from 10.0-baseline.pt's coarse decoder. Stage B of the plan in
.claude/plans/snazzy-cuddling-creek.md -- no third-party weights, P&C only.

Freezes everything except refine_head: the coarse decoder is 10.0-baseline's
own already-trained, already-validated weights, untouched (Stage A = just
loading it). Each step feeds the model a real P&C crop, generates a synthetic
"coarse mistake" version of its real ground-truth mask via
synthesize_coarse_mask_perturbation() (own from-scratch implementation, not
CascadePSP's code), and trains refine_head alone to map (image,
synthetic-coarse-mask) -> real ground-truth mask.

Usage:
  # pilot: bounded, a few hundred steps, confirm loss decreases before any
  # further commitment (see plan's explicit stop condition)
  .venv/bin/python src/train_refine_head.py --steps 400 \
      --pilot-out .tmp/refine_head_pilot.pt
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Works around a real ROCm/MIOpen codegen bug hit on this GPU specifically at
# (batch=2, 512x512, base=24) -- BatchNorm2d's spatial-train MIOpen kernel
# fails inline-asm compilation (RuntimeError: miopenStatusUnknownError,
# "cannot compile inline asm"), confirmed via a standalone repro. Disabling
# cudnn (which routes to MIOpen on ROCm) makes PyTorch fall back to its
# portable batchnorm kernel instead -- confirmed still runs on cuda:0, not a
# silent CPU fallback, and confirmed cheap (20 iters of the failing shape in
# ~0.05s). Only changes which BatchNorm kernel implementation runs, not the
# math, so no accuracy impact.
torch.backends.cudnn.enabled = False

import ml_cleaner  # noqa: E402

# Checkpoints pickle their argparse Namespace's `func` default, which points
# at __main__.train_command when saved by running ml_cleaner.py directly
# (see probe_cascadepsp.py, same trick) -- needed to torch.load 10.0-baseline.pt
# from this separate entry point.
sys.modules["__main__"].train_command = ml_cleaner.train_command

from ml_cleaner import (  # noqa: E402
    BASE_VARIANTS,
    OVERLAY_VARIANTS,
    DiceBCELoss,
    GuidanceParams,
    PatchDataset,
    SmallUNet,
    choose_device,
    expand_path,
    find_dataset_pairs,
    log,
    seed_everything,
    synthesize_coarse_mask_perturbation,
)

BASELINE = ROOT / "data/models/10.0-baseline.pt"
DATASET_DIR = ROOT / "data/dataset_split_scaled/train"


def perturb_batch(masks: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """masks: (B,1,H,W) float {0,1} real GT delete-mask. Returns synthetic
    'coarse' pseudo-logits of the same shape: a confident +-constant matching
    a perturbed (eroded/dilated/bite-holed) version of the real mask, standing
    in for a coarse model's confident-but-sometimes-wrong prediction --
    refine_head only ever sees logits, never a raw 0/1 mask, matching how
    it's fed from coarse_forward() in SmallUNet.forward()."""
    arr = masks.detach().cpu().numpy().astype(bool)
    out = np.empty_like(arr, dtype=np.float32)
    for b in range(arr.shape[0]):
        out[b, 0] = synthesize_coarse_mask_perturbation(arr[b, 0], rng).astype(np.float32)
    logits = (out * 2.0 - 1.0) * 8.0  # +-8 logit ~= sigmoid 0.9997/0.0003, a confident coarse call
    return torch.from_numpy(logits).to(masks.device)


def main() -> None:
    # Redirected stdout (nohup/background runs) is fully-buffered by default --
    # force line-buffering so a long pilot stays observable in real time
    # (same lesson as train_cascadepsp_pc.py).
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--steps", type=int, required=True, help="bounded pilot: a few hundred, not a full training-scale run")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--patch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dice-weight", type=float, default=0.65)
    ap.add_argument(
        "--boundary-patch-ratio", type=float, default=0.5,
        help="of positive patches, fraction centered on a mask-boundary pixel rather than "
        "any delete pixel -- refine_head's whole job is boundary correction, so bias much "
        "more toward boundaries than 10.0-baseline's own recipe did (0.0)",
    )
    ap.add_argument("--positive-patch-ratio", type=float, default=0.85)
    ap.add_argument("--alpha-threshold", type=int, default=128)
    ap.add_argument("--threshold-value", type=int, default=30)
    ap.add_argument("--morph-radius", type=int, default=2)
    ap.add_argument("--variants", default=",".join(BASE_VARIANTS + OVERLAY_VARIANTS))
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--renders-cleaned", type=Path, default=ROOT / "data/renders_cleaned")
    ap.add_argument("--baseline", type=Path, default=BASELINE, help="warm-start checkpoint (Stage A)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--cache-size", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "data/models/refine-head-pilot.pt")
    ap.add_argument("--pilot-out", type=Path, default=None,
                     help="if set, save/overwrite here instead of --out (keeps pilot runs out of the real checkpoint slot)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = choose_device(args.device)
    log(f"device: {device}")

    if not args.baseline.exists():
        sys.exit(f"baseline checkpoint not found: {args.baseline}")
    checkpoint = torch.load(str(args.baseline), map_location="cpu", weights_only=False)
    base_config = checkpoint["config"]
    log(f"baseline config: {base_config}")

    guidance_params = GuidanceParams(threshold_value=args.threshold_value, morph_radius=args.morph_radius)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    pairs = find_dataset_pairs(expand_path(str(args.dataset)), expand_path(str(args.renders_cleaned)), variants)
    log(f"found {len(pairs)} training pairs")

    dataset = PatchDataset(
        pairs=pairs,
        alpha_threshold=args.alpha_threshold,
        guidance_params=guidance_params,
        patch_size=args.patch_size,
        patches_per_epoch=args.steps * args.batch_size,
        positive_patch_ratio=args.positive_patch_ratio,
        min_positive_pixels=1,
        augment=True,
        cache_size=args.cache_size,
        boundary_patch_ratio=args.boundary_patch_ratio,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)

    model = SmallUNet(
        in_channels=int(base_config["in_channels"]),
        base=int(base_config["base_channels"]),
        sdt_head=bool(base_config.get("sdt_head", False)),
        refine_head=True,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    log(f"warm-start load: missing={missing} unexpected={unexpected}")
    assert all(k.startswith("refine_head.") for k in missing), f"unexpected missing keys outside refine_head: {missing}"
    assert not unexpected, f"unexpected keys in baseline checkpoint: {unexpected}"

    # Freeze everything except refine_head -- Stage B is a clean, isolated
    # single-variable test of the refine head alone; the coarse decoder is
    # 10.0-baseline's own already-trained, already-validated weights.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("refine_head.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = DiceBCELoss(pos_weight=1.0, dice_weight=args.dice_weight)

    out_path = args.pilot_out if args.pilot_out is not None else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = dict(base_config)
    config["refine_head"] = True

    model.train()
    step = 0
    running = 0.0
    t_start = time.time()
    for images, masks, _weights, _sdts in loader:
        step += 1
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.no_grad():
            u1, mid, _coarse_logits = model.coarse_forward(images)
            synthetic_logits = perturb_batch(masks, rng)

        refined_logits = model.refine_head(u1, synthetic_logits, mid)
        loss = criterion(refined_logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        if step % args.log_every == 0:
            avg = running / args.log_every
            elapsed = time.time() - t_start
            log(f"step {step}/{args.steps} loss={avg:.5f} elapsed={elapsed:.1f}s")
            running = 0.0

        if step % args.save_every == 0 or step == args.steps:
            torch.save({"state_dict": model.state_dict(), "config": config}, out_path)
            stepped_path = out_path.with_suffix(f".step{step}{out_path.suffix}")
            torch.save({"state_dict": model.state_dict(), "config": config}, stepped_path)
            log(f"saved checkpoint: {out_path} and {stepped_path} (step {step})")

        if step >= args.steps:
            break

    log(f"done. total time: {(time.time() - t_start) / 60:.1f}min for {args.steps} steps")


if __name__ == "__main__":
    main()
