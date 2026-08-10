"""Plan v15 Track 1: micro band classifier -- gutter (delete) vs white-panel-interior (keep)
for full-width white bands, the class where every classical mechanism failed (v14: band-height
rule broke both distributions; the band is locally information-free by the structural-identity
finding, so the net sees the SURROUNDING LAYOUT).

Architecture (attempt 1): BandNet -- 4 conv blocks (8/16/32/64ch, stride 2) over a
band-centered grayscale context window (page downscaled to width 128, window 256 rows tall,
band midline centered), GAP, then a linear head that also takes 2 scalars (band height px,
band height / page median band height). Param count printed at train time (~63k).

Training data: 100% synthetic (P&C stage1 generator, fresh seeds 20260808+); the 20 Phase-A
eval pages (seed 20260807), gold parts and the 005-1 fit page are NEVER trained on.
Bands extracted with the SAME detector used at inference (row white-frac >= 0.95, height >= 8);
label = GT majority over band rows (>=0.8 delete -> gutter=1, <=0.2 -> interior=0, else drop).

Checkpoint: .tmp/checkpoints/band_classifier/ (never committed).

Usage:
  train:  .venv/bin/python band_classifier.py train --pages DIR --out CKPT [--epochs 12]
  (inference API: load_band_net(ckpt), classify_bands(gray, bands, net) -> list[bool gutter])
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

WIN_H = 256
WIN_W = 128
MIN_BAND_H = 8
WHITE_ROW_FRAC = 0.95


def find_bands(gray: np.ndarray) -> list[tuple[int, int]]:
    """Full-width white bands: maximal runs of rows with white-frac >= 0.95. Same detector at
    train and inference."""
    white_row = (gray >= 250).mean(axis=1) >= WHITE_ROW_FRAC
    bands = []
    r, H = 0, len(white_row)
    while r < H:
        if not white_row[r]:
            r += 1
            continue
        r0 = r
        while r < H and white_row[r]:
            r += 1
        if r - r0 >= MIN_BAND_H:
            bands.append((r0, r))
    return bands


def band_window(gray: np.ndarray, b0: int, b1: int) -> np.ndarray:
    """Band-centered context window, page downscaled to width WIN_W, WIN_H rows tall in
    downscaled space (scale factor = WIN_W / page width, so the window sees the same
    PROPORTIONAL context on every page)."""
    H, W = gray.shape
    scale = WIN_W / W
    small = cv2.resize(gray, (WIN_W, max(1, int(round(H * scale)))), interpolation=cv2.INTER_AREA)
    mid = int(round(((b0 + b1) / 2) * scale))
    half = WIN_H // 2
    win = np.zeros((WIN_H, WIN_W), dtype=np.uint8)
    s0 = max(0, mid - half)
    s1 = min(small.shape[0], mid + half)
    d0 = half - (mid - s0)
    win[d0 : d0 + (s1 - s0)] = small[s0:s1]
    return win


def band_features(gray: np.ndarray, bands: list[tuple[int, int]], i: int) -> np.ndarray:
    heights = np.array([b1 - b0 for b0, b1 in bands], dtype=np.float32)
    med = float(np.median(heights)) if len(heights) else 1.0
    h = float(bands[i][1] - bands[i][0])
    return np.array([h / 1000.0, h / max(med, 1.0)], dtype=np.float32)


def build_net():
    import torch.nn as nn

    class BandNet(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(ci, co):
                return nn.Sequential(nn.Conv2d(ci, co, 3, stride=2, padding=1),
                                      nn.BatchNorm2d(co), nn.ReLU())
            self.conv = nn.Sequential(blk(1, 8), blk(8, 16), blk(16, 32), blk(32, 64))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(64 + 2, 1)

        def forward(self, x, feats):
            z = self.pool(self.conv(x)).flatten(1)
            import torch
            return self.head(torch.cat([z, feats], dim=1)).squeeze(1)

    return BandNet()


def extract_dataset(pages_dir: Path):
    xs, fs, ys = [], [], []
    src_dir = pages_dir / "frames"
    tgt_dir = pages_dir / "frames_cleaned"
    n_pages = 0
    for f in sorted(src_dir.glob("*.png")):
        gray = np.asarray(Image.open(f).convert("L"))
        alpha = np.asarray(Image.open(tgt_dir / f.name).split()[-1])
        gt_delete = alpha < 128
        bands = find_bands(gray)
        if not bands:
            continue
        n_pages += 1
        for i, (b0, b1) in enumerate(bands):
            dfrac = float(gt_delete[b0:b1].mean())
            if 0.2 < dfrac < 0.8:
                continue
            xs.append(band_window(gray, b0, b1))
            fs.append(band_features(gray, bands, i))
            ys.append(1.0 if dfrac >= 0.8 else 0.0)
    return np.stack(xs), np.stack(fs), np.array(ys, dtype=np.float32), n_pages


def train(pages_dir: Path, out: Path, epochs: int, lr: float = 3e-4,
           use_pos_weight: bool = True, use_sched: bool = True,
           use_best_ckpt: bool = True) -> None:
    import torch
    import torch.nn as nn

    xs, fs, ys = None, None, None
    xs, fs, ys, n_pages = extract_dataset(pages_dir)
    n = len(ys)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_val = max(1, n // 10)
    vi, ti = idx[:n_val], idx[n_val:]
    print(f"pages {n_pages}  bands {n}  gutter {int(ys.sum())}  interior {int((1-ys).sum())}  "
          f"train {len(ti)}  val {len(vi)}", flush=True)

    net = build_net()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params:,}", flush=True)
    # All historical configurations are reproducible via flags (plan v17 rigor correction --
    # the original attempt 2 conflated data-scale AND loss-balancing in one run; the isolated
    # re-runs 2a/2b use these flags):
    #   attempt 1:  --lr 1e-3 --no-pos-weight --no-sched --save-last   (250 pages)
    #   attempt 2a: --lr 1e-3 --pos-weight    --no-sched --save-last   (250 pages, loss only)
    #   attempt 2b: --lr 1e-3 --no-pos-weight --no-sched --save-last   (1k pages, data only)
    #   attempt 3:  --lr 3e-4 --pos-weight    --sched    --best-ckpt   (1k pages)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.3) if use_sched else None
    if use_pos_weight:
        n_pos = float(ys[ti].sum())
        n_neg = float(len(ti) - n_pos)
        pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"pos_weight (gutter): {float(pos_weight):.3f}", flush=True)
    else:
        lossf = nn.BCEWithLogitsLoss()
        print("pos_weight: OFF (plain BCE)", flush=True)

    def batches(indices, bs=64, shuffle=True):
        order = rng.permutation(indices) if shuffle else indices
        for k in range(0, len(order), bs):
            sel = order[k : k + bs]
            x = torch.from_numpy(xs[sel]).float().unsqueeze(1) / 255.0
            f = torch.from_numpy(fs[sel])
            y = torch.from_numpy(ys[sel])
            yield x, f, y

    best_bal, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        net.train()
        tot = 0.0
        for x, f, y in batches(ti):
            opt.zero_grad()
            loss = lossf(net(x, f), y)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(y)
        net.eval()
        with torch.no_grad():
            tp = fp = tn = fn = 0
            for x, f, y in batches(vi, shuffle=False):
                pred = (torch.sigmoid(net(x, f)) >= 0.5).float()
                tp += int(((pred == 1) & (y == 1)).sum())
                fp += int(((pred == 1) & (y == 0)).sum())
                tn += int(((pred == 0) & (y == 0)).sum())
                fn += int(((pred == 0) & (y == 1)).sum())
        gut_rec = tp / max(tp + fn, 1)
        int_rec = tn / max(tn + fp, 1)
        bal = (gut_rec + int_rec) / 2
        print(f"epoch {ep}: train_loss {tot/len(ti):.4f}  "
              f"val gutter-recall {gut_rec:.3f}  interior-recall {int_rec:.3f}  "
              f"balanced {bal:.3f}", flush=True)
        if bal > best_bal:
            best_bal = bal
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_ep = ep
        if sched is not None:
            sched.step()

    if not use_best_ckpt:
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
        best_ep = epochs
        print("(save-last mode: keeping the final epoch's weights)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "n_params": n_params,
                 "best_epoch": best_ep, "best_balanced_acc": best_bal}, out)
    print(f"saved BEST epoch {best_ep} (balanced acc {best_bal:.3f}) -> {out}", flush=True)


_net_cache = {}


def load_band_net(ckpt: Path):
    import torch
    key = str(ckpt)
    if key not in _net_cache:
        net = build_net()
        net.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
        net.eval()
        _net_cache[key] = net
    return _net_cache[key]


def classify_bands(gray: np.ndarray, bands: list[tuple[int, int]], net) -> list[bool]:
    """Returns per-band is_gutter (True = delete-side)."""
    import torch
    if not bands:
        return []
    xs = np.stack([band_window(gray, b0, b1) for b0, b1 in bands])
    fs = np.stack([band_features(gray, bands, i) for i in range(len(bands))])
    with torch.no_grad():
        logits = net(torch.from_numpy(xs).float().unsqueeze(1) / 255.0, torch.from_numpy(fs))
        return (torch.sigmoid(logits) >= 0.5).tolist()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--pages", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", dest="pos_weight", action="store_true", default=True)
    ap.add_argument("--no-pos-weight", dest="pos_weight", action="store_false")
    ap.add_argument("--sched", dest="sched", action="store_true", default=True)
    ap.add_argument("--no-sched", dest="sched", action="store_false")
    ap.add_argument("--best-ckpt", dest="best_ckpt", action="store_true", default=True)
    ap.add_argument("--save-last", dest="best_ckpt", action="store_false")
    a = ap.parse_args()
    train(a.pages, a.out, a.epochs, lr=a.lr, use_pos_weight=a.pos_weight,
           use_sched=a.sched, use_best_ckpt=a.best_ckpt)
