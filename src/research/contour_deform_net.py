"""Attempt 7: Deep-Snake-style contour deformation network. Backbone CNN extracts a multi-scale
feature map from the 320px crop; per-vertex features are bilinear-sampled (F.grid_sample) at
each of N_VERTICES points around the initial (coarse ellipse) contour; a circular 1D conv stack
(the graph-conv analog for a simple-cycle contour graph -- each vertex's neighbors on the cycle
are its immediate predecessor/successor, exactly what a circularly-padded Conv1d convolves over)
predicts a per-vertex RADIAL offset. No dense per-pixel output anywhere in this network -- the
one property every one of the 6 previously-tried mechanisms shared and this one structurally
lacks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / ".tmp/checkpoints/contour_deform_smoke"
CROP_SIZE = 512  # 2026-08-04: bumped from 320 to cover the synthetic pool's real bubble-scale
                  # distribution (p90 diag ~299px) -- see build_contour_training_data.py's note
N_VERTICES = 64


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContourFeatureBackbone(nn.Module):
    """3-level encoder, feature-pyramid-sampled (all 3 scales upsampled back to full crop
    resolution and concatenated) -- same multi-scale-sampling idea Deep Snake itself uses,
    simplified to a plain concat rather than a learned fusion."""

    def __init__(self, in_ch: int = 7, base: int = 12) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.out_ch = base + base * 2 + base * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        size = e1.shape[2:]
        e2_up = F.interpolate(e2, size=size, mode="bilinear", align_corners=False)
        e3_up = F.interpolate(e3, size=size, mode="bilinear", align_corners=False)
        return torch.cat([e1, e2_up, e3_up], dim=1)


class CircularConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=0)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # circular padding: each vertex's neighbors on the closed contour cycle
        x_pad = torch.cat([x[:, :, -1:], x, x[:, :, :1]], dim=2)
        return F.relu(self.bn(self.conv(x_pad)))


class ContourDeformNet(nn.Module):
    def __init__(self, in_ch: int = 7, base: int = 12, n_vertices: int = N_VERTICES) -> None:
        super().__init__()
        self.backbone = ContourFeatureBackbone(in_ch, base)
        feat_ch = self.backbone.out_ch
        gconv_ch = 64
        self.proj = nn.Conv1d(feat_ch + 2, gconv_ch, 1)  # +2 for normalized vertex xy
        self.gconv1 = CircularConv1d(gconv_ch, gconv_ch)
        self.gconv2 = CircularConv1d(gconv_ch, gconv_ch)
        self.gconv3 = CircularConv1d(gconv_ch, gconv_ch)
        self.out = nn.Conv1d(gconv_ch, 1, 1)
        self.n_vertices = n_vertices

    def forward(self, crop: torch.Tensor, vertex_xy_norm: torch.Tensor) -> torch.Tensor:
        """crop: (N,7,H,W). vertex_xy_norm: (N,n_vertices,2) in [-1,1] grid_sample coords.
        Returns predicted per-vertex radial offset in PIXELS, (N, n_vertices)."""
        feat_map = self.backbone(crop)
        grid = vertex_xy_norm.unsqueeze(1)  # (N,1,n_vertices,2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        sampled = sampled.squeeze(2)  # (N,C,n_vertices)
        vxy = vertex_xy_norm.permute(0, 2, 1)  # (N,2,n_vertices)
        h = torch.cat([sampled, vxy], dim=1)
        h = self.proj(h)
        h = self.gconv1(h)
        h = self.gconv2(h)
        h = self.gconv3(h)
        return self.out(h).squeeze(1)


def vertex_xy_norm_from_radii(radii: torch.Tensor, angles: torch.Tensor, crop_size: int) -> torch.Tensor:
    """radii: (N, n_vertices) in pixels, from the crop's own center (crops are always built
    centered on the instance centroid). angles: (n_vertices,). Returns grid_sample-ready
    normalized coords, (N, n_vertices, 2)."""
    half = crop_size / 2.0
    x_px = half + radii * torch.cos(angles)
    y_px = half + radii * torch.sin(angles)
    x_norm = 2.0 * x_px / (crop_size - 1) - 1.0
    y_norm = 2.0 * y_px / (crop_size - 1) - 1.0
    return torch.stack([x_norm, y_norm], dim=-1)


def load_split(tag: str, name: str) -> dict:
    d = np.load(DATA_DIR / f"{name}_{tag}.npz")
    return {
        "crops": torch.from_numpy(d["crops"]).permute(0, 3, 1, 2).float(),
        "init_radii": torch.from_numpy(d["init_radii"]).float(),
        "true_radii": torch.from_numpy(d["true_radii"]).float(),
        "angles": torch.from_numpy(d["angles"]).float(),
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="smoke250")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")

    train = load_split(args.tag, "train")
    val = load_split(args.tag, "val")
    print(f"[{args.tag}] train: {tuple(train['crops'].shape)}  val: {tuple(val['crops'].shape)}", flush=True)

    model = ContourDeformNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 2026-08-05 Part B fix (refinement of attempt 7, notes/instance_aware_pivot_2026-08-04.md):
    # no LR schedule existed before -- flat lr for every epoch. Cosine decay so late-epoch
    # updates shrink instead of continuing to move the weights at full magnitude, which the
    # 1k/2k training logs showed producing a final-epoch checkpoint measurably worse than the
    # run's own best (val_loss bottoming out several epochs before the run ended, then getting
    # worse) -- diagnosed directly from those logs, not guessed.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = train["crops"].shape[0]
    batch_size = args.batch_size
    angles = train["angles"].to(device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    def run_batch(idx: torch.Tensor, split: dict) -> torch.Tensor:
        crop = split["crops"][idx].to(device)
        init_r = split["init_radii"][idx].to(device)
        true_r = split["true_radii"][idx].to(device)
        vxy = vertex_xy_norm_from_radii(init_r, angles, CROP_SIZE)
        pred_dr = model(crop, vxy)
        target_dr = true_r - init_r
        loss = F.smooth_l1_loss(pred_dr, target_dr)
        return loss, pred_dr, target_dr

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss, _, _ = run_batch(idx, train)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / n
        scheduler.step()

        model.eval()
        n_val = val["crops"].shape[0]
        val_loss_sum = 0.0
        pred_chunks, target_chunks = [], []
        with torch.no_grad():
            # batched, same as training -- an earlier unbatched version (whole val set, all
            # 200 examples at once at CROP_SIZE=512) silently OOM-killed the process before
            # this epoch's print ever ran (no traceback, SIGKILL isn't catchable) -- root
            # caused via direct memory-footprint estimation, not guessed.
            for i in range(0, n_val, batch_size):
                val_idx = torch.arange(i, min(i + batch_size, n_val))
                loss, pred_dr, target_dr = run_batch(val_idx, val)
                val_loss_sum += loss.item() * len(val_idx)
                pred_chunks.append(pred_dr)
                target_chunks.append(target_dr)
        val_loss = val_loss_sum / n_val
        pred_dr = torch.cat(pred_chunks, dim=0)
        target_dr = torch.cat(target_chunks, dim=0)
        # "does the deformed contour move in a plausible direction" check: correlation
        # between predicted and true radial offset (0 = no signal, 1 = perfect direction+magnitude)
        pred_flat = pred_dr.flatten().numpy()
        target_flat = target_dr.flatten().numpy()
        corr = np.corrcoef(pred_flat, target_flat)[0, 1] if pred_flat.std() > 1e-6 else float("nan")
        mean_abs_err_before = float(target_dr.abs().mean())  # error if predicting zero deformation
        mean_abs_err_after = float((pred_dr - target_dr).abs().mean())

        cur_lr = opt.param_groups[0]["lr"]
        print(f"epoch {epoch}: lr={cur_lr:.2e}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"pred_vs_true_corr={corr:.4f}  mae_zero_baseline={mean_abs_err_before:.2f}px  "
              f"mae_model={mean_abs_err_after:.2f}px", flush=True)

        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            print("FAIL: non-finite loss (NaN/Inf) -- architecture/data pipeline bug, stopping.")
            sys.exit(1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"contour_deform_{args.tag}.pt"
    torch.save({"model_state": model.state_dict(), "base": 12, "in_ch": 7, "n_vertices": N_VERTICES,
                "crop_size": CROP_SIZE, "epoch": args.epochs}, out_path)
    print(f"OK: smoke check passed. Final-epoch checkpoint saved to {out_path}")

    best_path = DATA_DIR / f"contour_deform_{args.tag}_best.pt"
    torch.save({"model_state": best_state, "base": 12, "in_ch": 7, "n_vertices": N_VERTICES,
                "crop_size": CROP_SIZE, "epoch": best_epoch, "val_loss": best_val_loss}, best_path)
    print(f"Best-val-loss checkpoint (epoch {best_epoch}, val_loss={best_val_loss:.4f}) "
          f"saved to {best_path}")


if __name__ == "__main__":
    main()
