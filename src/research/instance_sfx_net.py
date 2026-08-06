"""Part 1 steps 2-3 of the instance-aware architecture pivot (see
notes/synthetic_curriculum_plan.md and docs/ml_strategy_history.md for the project record;
.claude/plans/snazzy-cuddling-creek.md has the full session plan).

A minimal, from-scratch, per-instance local network: takes a fixed 224x224 crop centered on a
detected SFX instance (7 channels: RGB + the 4 make_guidance_channels guidance channels, same
input convention as the production SmallUNet) and predicts a same-size dense keep/delete mask
IN LOCAL CROP COORDINATES ONLY. Deliberately much smaller than SmallUNet (this is a
proof-of-mechanism run, not a production model) and deliberately dense-per-pixel rather than a
parametric contour (see the plan's Finding 3 for why -- time budget, not a claim contour is
worse). The whole point being tested: does bounding the model's input to a 224x224 window
(96px margin around the glyph, chosen from the occlusion probe's own "still correct" R<=64px
band in docs/ml_strategy_history.md) structurally prevent the R>=128px "read as background"
context-flip the whole-page model shows on ch1_sfx_text -- not by learning anything different,
but by construction, since the trigger radius literally isn't visible from this crop.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / ".tmp/checkpoints/instance_sfx_smoke"


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyInstanceNet(nn.Module):
    """3-level encoder/decoder, ~1/10th SmallUNet's channel width -- a proof-of-mechanism
    network, not a production candidate."""

    def __init__(self, in_ch: int = 7, base: int = 12) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)  # logits, (N, 1, H, W)


def load_split(name: str, variant: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = np.load(DATA_DIR / f"crops_{name}_{variant}.npz")
    x = torch.from_numpy(d["x"]).permute(0, 3, 1, 2).float()  # (N,7,H,W)
    y = torch.from_numpy(d["y"]).float()  # (N,H,W), 1.0 = delete
    is_bg = torch.from_numpy(d["is_bg"]) if "is_bg" in d else torch.zeros(x.shape[0], dtype=torch.bool)
    return x, y, is_bg


def dice_bce(logits: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    if sample_weight is None:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum()
        dice = 1.0 - (2 * inter + 1.0) / (probs.sum() + targets.sum() + 1.0)
        return bce + dice
    # per-sample weighting (used by with_bg_weighted: background-only crops contribute less to
    # the gradient than instance crops, a loss-side fix instead of the data-ratio/output-bias
    # fixes already ruled out -- see notes/instance_aware_pivot_2026-08-03.md).
    per_px_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    w = sample_weight.view(-1, 1, 1)
    bce = (per_px_bce * w).sum() / (w.expand_as(per_px_bce).sum() + 1e-8)
    probs = torch.sigmoid(logits)
    inter = (probs * targets * w).sum()
    dice = 1.0 - (2 * inter + 1.0) / ((probs * w).sum() + (targets * w).sum() + 1.0)
    return bce + dice


BG_LOSS_WEIGHT = 0.2  # with_bg_weighted only: background-crop loss contribution relative to instance crops


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                     choices=["glyph_only", "with_bg", "with_bg_light", "with_bg_weighted"])
    args = ap.parse_args()
    weighted = args.variant == "with_bg_weighted"

    torch.manual_seed(0)
    device = torch.device("cpu")

    train_x, train_y, train_bg = load_split("train", args.variant)
    val_x, val_y, val_bg = load_split("val", args.variant)
    print(f"[{args.variant}] train: {tuple(train_x.shape)}  val: {tuple(val_x.shape)}"
          + (f"  (bg_loss_weight={BG_LOSS_WEIGHT})" if weighted else ""))

    model = TinyInstanceNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 8
    batch_size = 16
    n = train_x.shape[0]

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = train_x[idx].to(device), train_y[idx].to(device)
            opt.zero_grad()
            logits = model(xb).squeeze(1)
            if weighted:
                bg_batch = train_bg[idx].to(device)
                sw = torch.where(bg_batch, torch.tensor(BG_LOSS_WEIGHT), torch.tensor(1.0))
                loss = dice_bce(logits, yb, sample_weight=sw)
            else:
                loss = dice_bce(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device)).squeeze(1)
            val_loss = dice_bce(val_logits, val_y.to(device)).item()
            val_probs = torch.sigmoid(val_logits)
            frac_pred_delete = (val_probs > 0.5).float().mean().item()

        print(f"epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_pred_delete_frac={frac_pred_delete:.4f}")

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            print("FAIL: non-finite loss (NaN/Inf) -- architecture/data pipeline bug, stopping.")
            sys.exit(1)

    # degeneracy check: output should not collapse to all-0 or all-1
    with torch.no_grad():
        val_probs = torch.sigmoid(model(val_x.to(device)).squeeze(1))
    p_min, p_max, p_std = val_probs.min().item(), val_probs.max().item(), val_probs.std().item()
    print(f"final val prob range: min={p_min:.4f} max={p_max:.4f} std={p_std:.4f}")
    if p_std < 1e-3:
        print("FAIL: degenerate output (near-zero variance across predictions).")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"instance_sfx_smoke_{args.variant}.pt"
    torch.save({"model_state": model.state_dict(), "base": 12, "in_ch": 7}, out_path)
    print(f"OK: smoke check passed. Checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
