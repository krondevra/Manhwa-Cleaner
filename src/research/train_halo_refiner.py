"""Halo fix, 5th mechanism (.claude/plans/snazzy-cuddling-creek.md), Part 3: train the small,
from-scratch, trunk-independent HaloRefinerNet on calibrated halo-perturbation crops.

Usage:
  python3 src/train_halo_refiner.py --n-train 300 --n-val 60 --epochs 5 \
      --out .tmp/halo_refiner_smoke/refiner.pt --device auto

Smoke-scale first (per Part 5's verification ladder): small --n-train/--n-val, few --epochs,
then check the keep<->delete flip ratio specifically (model 18.0's fatal failure signature was
a wildly lopsided 70,430:33 ratio) before any larger commitment.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
REPO_ROOT = SRC_ROOT.parent

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from ml_cleaner import (
    DiceBCELoss, GuidanceParams, choose_device, find_dataset_pairs, load_model, log, seed_everything,
)
from halo_refiner import (
    HaloRefinerCropDataset, HaloRefinerRealErrorCropDataset, HaloRefinerNet, count_parameters,
    precompute_model_predictions,
)
import ml_cleaner
import __main__ as _main_mod
_main_mod.train_command = ml_cleaner.train_command


def flip_ratio_report(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """model 18.0's RefineHead failure signature: a wildly lopsided keep<->delete flip ratio
    (70,430 keep->delete flips against only 33 delete->keep, on held-out data). Measures the
    same two directions here: how many pixels does the refiner flip relative to its OWN input
    mask, split by direction, and separately how many of the refiner's flips actually MATCH
    the true correction needed (input->target) vs. don't."""
    model.eval()
    keep_to_delete = 0
    delete_to_keep = 0
    correct_fixes = 0
    incorrect_flips = 0
    total_px = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            input_mask = images[:, 3:4, :, :]
            logits = model(images)
            pred = (torch.sigmoid(logits) > 0.5).float()

            flip_k2d = (input_mask < 0.5) & (pred > 0.5)  # keep -> delete
            flip_d2k = (input_mask > 0.5) & (pred < 0.5)  # delete -> keep
            keep_to_delete += int(flip_k2d.sum().item())
            delete_to_keep += int(flip_d2k.sum().item())

            any_flip = flip_k2d | flip_d2k
            matches_target = (pred == targets)
            correct_fixes += int((any_flip & matches_target).sum().item())
            incorrect_flips += int((any_flip & ~matches_target).sum().item())
            total_px += input_mask.numel()

    return {
        "keep_to_delete": keep_to_delete,
        "delete_to_keep": delete_to_keep,
        "correct_fixes": correct_fixes,
        "incorrect_flips": incorrect_flips,
        "total_px": total_px,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=REPO_ROOT / ".tmp/datasets/b2_bubbles_2k_prestage/train_root")
    ap.add_argument("--variant", type=str, default="bubbles")
    ap.add_argument("--n-train", type=int, default=300, help="number of source pairs for train split")
    ap.add_argument("--n-val", type=int, default=60, help="number of source pairs for val split")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-channels", type=int, default=24)
    ap.add_argument("--negative-frac", type=float, default=0.25)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--data-source", choices=["perturbation", "real-error"], default="perturbation",
                     help="'perturbation' = synthesize_halo_perturbation on clean GT (original "
                     "Part 1 approach). 'real-error' = train on the b2 checkpoint's own "
                     "predicted errors vs. clean GT (Priority 1, autonomous continuation, "
                     "2026-08-01) -- requires --source-checkpoint.")
    ap.add_argument("--source-checkpoint", type=Path,
                     default=REPO_ROOT / ".tmp/checkpoints/stage2/b2_bubbles_2k_prestage/b2_full2k_finetune.pt",
                     help="Stage1+2 checkpoint to generate real-error pairs from, only used "
                     "when --data-source=real-error.")
    ap.add_argument("--pred-cache", type=Path, default=None,
                     help="Cache dir for precomputed real-error predictions (default: "
                     "<out_dir>/pred_cache).")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = choose_device(args.device)
    log(f"device: {device}")

    all_pairs = find_dataset_pairs(args.dataset, Path(""), [args.variant])
    log(f"found {len(all_pairs)} total source pairs")
    train_pairs = all_pairs[: args.n_train]
    val_pairs = all_pairs[args.n_train: args.n_train + args.n_val]
    log(f"train pairs: {len(train_pairs)}, val pairs: {len(val_pairs)}")

    if args.data_source == "real-error":
        log(f"data source: real-error, generating from {args.source_checkpoint}")
        gp = GuidanceParams()
        source_model, source_config = load_model(args.source_checkpoint, device)
        log(f"loaded source checkpoint (in_channels={source_config.get('in_channels')})")
        pred_cache = args.pred_cache or (args.out.parent / "pred_cache")
        t_pre = time.time()
        train_manifest = precompute_model_predictions(train_pairs, source_model, device, gp, pred_cache)
        val_manifest = precompute_model_predictions(val_pairs, source_model, device, gp, pred_cache)
        log(f"precomputed {len(train_manifest)}+{len(val_manifest)} page predictions in "
            f"{time.time()-t_pre:.1f}s (cached to {pred_cache})")
        del source_model
        train_ds = HaloRefinerRealErrorCropDataset(train_manifest, seed=args.seed)
        val_ds = HaloRefinerRealErrorCropDataset(val_manifest, seed=args.seed + 1)
    else:
        train_ds = HaloRefinerCropDataset(train_pairs, negative_frac=args.negative_frac, seed=args.seed)
        val_ds = HaloRefinerCropDataset(val_pairs, negative_frac=args.negative_frac, seed=args.seed + 1)
    log(f"train crops: {len(train_ds)} ({sum(1 for s in train_ds.samples if s['hard'])} hard), "
        f"val crops: {len(val_ds)} ({sum(1 for s in val_ds.samples if s['hard'])} hard)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HaloRefinerNet(in_channels=4, base=args.base_channels).to(device)
    n_params = count_parameters(model)
    log(f"model params: {n_params:,} ({n_params/1e6:.3f}M)")

    criterion = DiceBCELoss(pos_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                val_losses.append(criterion(logits, targets).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        elapsed = time.time() - t0
        log(f"epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"({elapsed:.1f}s)")

        epoch_path = args.out.with_name(f"{args.out.stem}.epoch{epoch}{args.out.suffix}")
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"in_channels": 4, "base_channels": args.base_channels},
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
        }, epoch_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "config": {"in_channels": 4, "base_channels": args.base_channels},
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            }, args.out)
            log(f"  new best (val_loss={val_loss:.4f}), saved to {args.out}")

    log("\n--- flip-ratio report on held-out val crops (model 18.0 failure-signature check) ---")
    model.load_state_dict(torch.load(args.out, map_location=device, weights_only=False)["state_dict"])
    report = flip_ratio_report(model, val_loader, device)
    log(f"keep->delete flips: {report['keep_to_delete']:,}")
    log(f"delete->keep flips: {report['delete_to_keep']:,}")
    ratio = (report['keep_to_delete'] / max(1, report['delete_to_keep']))
    log(f"ratio (keep->delete : delete->keep) = {ratio:.2f} : 1 "
        f"(model 18.0's fatal signature was ~2134:1 -- 70,430:33)")
    log(f"of all flips, correct (matches target): {report['correct_fixes']:,}, "
        f"incorrect: {report['incorrect_flips']:,}")
    log(f"total pixels evaluated: {report['total_px']:,}")


if __name__ == "__main__":
    main()
