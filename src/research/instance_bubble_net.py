"""Part 2 training step: same TinyInstanceNet architecture and dice_bce loss as
instance_sfx_net.py (reused, not reimplemented -- content-type-agnostic), trained on bubble/
cloud instance crops with the PROVEN with_bg_weighted approach from last night (0.2x loss
weight on background-only crops) applied directly -- not re-running the 2 already-failed
data-ratio attempts, per the plan's explicit instruction to skip straight to what worked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from instance_sfx_net import TinyInstanceNet, dice_bce  # noqa: E402

DATA_DIR = ROOT / ".tmp/checkpoints/instance_bubble_smoke"
BG_LOSS_WEIGHT = 0.2  # same value that worked for SFX -- confirmed via evaluation, not assumed


def load_split(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = np.load(DATA_DIR / f"crops_{name}_bubble.npz")
    x = torch.from_numpy(d["x"]).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(d["y"]).float()
    is_bg = torch.from_numpy(d["is_bg"])
    return x, y, is_bg


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    train_x, train_y, train_bg = load_split("train")
    val_x, val_y, val_bg = load_split("val")
    print(f"[bubble] train: {tuple(train_x.shape)}  val: {tuple(val_x.shape)}  "
          f"(bg_loss_weight={BG_LOSS_WEIGHT})")

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
            bg_batch = train_bg[idx].to(device)
            sw = torch.where(bg_batch, torch.tensor(BG_LOSS_WEIGHT), torch.tensor(1.0))
            loss = dice_bce(logits, yb, sample_weight=sw)
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

    with torch.no_grad():
        val_probs = torch.sigmoid(model(val_x.to(device)).squeeze(1))
    p_min, p_max, p_std = val_probs.min().item(), val_probs.max().item(), val_probs.std().item()
    print(f"final val prob range: min={p_min:.4f} max={p_max:.4f} std={p_std:.4f}")
    if p_std < 1e-3:
        print("FAIL: degenerate output (near-zero variance across predictions).")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "instance_bubble_with_bg_weighted.pt"
    torch.save({"model_state": model.state_dict(), "base": 12, "in_ch": 7}, out_path)
    print(f"OK: smoke check passed. Checkpoint saved to {out_path}")


if __name__ == "__main__":
    main()
