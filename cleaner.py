#!/usr/bin/env python3
"""
v1.0.0 — full ML pivot: PyTorch SmallUNet binary segmentation.

Why the rewrite:
  Random Trees (v4–v7) failed to generalise from 1 training example to other
  chapters. The classifier can only learn per-pixel statistics; it has no spatial
  understanding of panel structure, speech bubbles, or SFX context.

  A convolutional U-Net, trained on manually cleaned chapters, can learn to
  understand these spatial relationships: it sees a 512×512 context window and
  can correlate a white pixel with the frame boundary above it, the speech bubble
  around it, etc.

Architecture: SmallUNet
  Input:  3 channels (RGB)
  Enc1:   3 → 32 (conv 3×3, BN, ReLU ×2) → MaxPool
  Enc2:   32 → 64                          → MaxPool
  Enc3:   64 → 128                         → MaxPool
  Bottleneck: 128 → 256                    (no pool)
  Dec3:   256+128 → 128 (skip connection from Enc3)
  Dec2:   128+ 64 →  64 (skip from Enc2)
  Dec1:    64+ 32 →  32 (skip from Enc1)
  Out:     32 → 1  (sigmoid → alpha probability)

Loss: DiceBCELoss = 0.5 * BCE + 0.5 * Dice
Optimizer: AdamW (lr=1e-3)
Training: random 512×512 patches from long-strip samples.

Commands:
  python manhwa_ml_cleaner.py train \
      --samples samples --model models/manhwa_cleaner.pt \
      --epochs 20 --steps-per-epoch 500 --batch-size 2 --patch-size 512 --device cpu

  python manhwa_ml_cleaner.py process \
      chapters/003.png chapters-results/003_result.png \
      --model models/manhwa_cleaner.pt --device cpu

  python manhwa_ml_cleaner.py process-folder \
      --input chapters --output chapters-results \
      --model models/manhwa_cleaner.pt --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CLEAN_SUFFIX = "_cleaned"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = _Block(in_channels, 32)
        self.enc2 = _Block(32, 64)
        self.enc3 = _Block(64, 128)
        self.bottleneck = _Block(128, 256)
        self.dec3 = _Block(256 + 128, 128)
        self.dec2 = _Block(128 + 64, 64)
        self.dec1 = _Block(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.out(d1)


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        num = 2 * (probs * targets).sum() + smooth
        den = probs.sum() + targets.sum() + smooth
        dice = 1 - num / den
        return 0.5 * bce + 0.5 * dice


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    original: np.ndarray   # H×W×3 uint8
    target: np.ndarray     # H×W float32 in [0,1]  (1 = delete)


def load_samples(samples_dir: Path, alpha_threshold: int = 128) -> List[Sample]:
    results = []
    files = sorted(p for p in samples_dir.glob("*.png") if not p.stem.endswith(CLEAN_SUFFIX))
    for orig_path in files:
        clean_path = orig_path.with_name(orig_path.stem + CLEAN_SUFFIX + ".png")
        if not clean_path.exists():
            continue
        _log(f"loading sample: {orig_path.name} + {clean_path.name}")
        original = np.array(Image.open(orig_path).convert("RGB"))
        cleaned = np.array(Image.open(clean_path).convert("RGBA"))
        if original.shape[:2] != cleaned.shape[:2]:
            raise ValueError(f"Size mismatch: {orig_path.name} and {clean_path.name}")
        target = (cleaned[:, :, 3] < alpha_threshold).astype(np.float32)
        delete_ratio = target.mean()
        _log(f"  size={original.shape[1]}x{original.shape[0]}, delete_ratio={delete_ratio:.4f}")
        results.append(Sample(original=original, target=target))
    return results


def random_patch(
    sample: Sample,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = sample.original.shape[:2]
    y = random.randint(0, max(0, h - patch_size))
    x = random.randint(0, max(0, w - patch_size))
    ph, pw = min(patch_size, h - y), min(patch_size, w - x)
    img_patch = sample.original[y:y+ph, x:x+pw]
    tgt_patch = sample.target[y:y+ph, x:x+pw]
    # Pad to patch_size if at boundary
    if ph < patch_size or pw < patch_size:
        img_out = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        tgt_out = np.zeros((patch_size, patch_size), dtype=np.float32)
        img_out[:ph, :pw] = img_patch
        tgt_out[:ph, :pw] = tgt_patch
        return img_out, tgt_out
    return img_patch, tgt_patch


def to_tensor(img: np.ndarray, device: "torch.device") -> "torch.Tensor":
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    if t.ndim == 3:
        t = t.permute(2, 0, 1)
    return t.unsqueeze(0).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model: SmallUNet, path: Path, config: dict) -> None:
    torch.save({"model": model.state_dict(), "config": config}, str(path))
    path.with_suffix(".json").write_text(json.dumps(config, indent=2))


def load_model(path: Path, device: "torch.device") -> Tuple[SmallUNet, dict]:
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    config = checkpoint.get("config", {"in_channels": 3})
    model = SmallUNet(in_channels=config.get("in_channels", 3))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, config


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    _log(f"device: {device}")

    samples = load_samples(Path(args.samples), args.alpha_threshold)
    _log(f"found {len(samples)} sample pairs")
    if not samples:
        raise SystemExit("No valid sample pairs found.")

    total_pixels = sum(s.target.size for s in samples)
    total_delete = sum(s.target.sum() for s in samples)
    delete_ratio = total_delete / total_pixels
    pos_weight = (1 - delete_ratio) / max(delete_ratio, 1e-6)
    _log(f"training pixels: delete={int(total_delete):,}, total={total_pixels:,}, delete_ratio={delete_ratio:.4f}")
    _log(f"pos_weight={pos_weight:.3f}")

    config = {"in_channels": 3}
    model = SmallUNet(in_channels=3).to(device)

    if args.resume:
        _log(f"loading checkpoint for resume: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    criterion = DiceBCELoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(args, "lr", 1e-3))
    scaler = GradScaler() if (args.amp and device.type == "cuda") else None

    best_loss = math.inf
    _log("training started")
    t_total = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t_epoch = time.time()

        for step in range(1, args.steps_per_epoch + 1):
            sample = random.choice(samples)
            imgs, tgts = [], []
            for _ in range(args.batch_size):
                img_p, tgt_p = random_patch(sample, args.patch_size)
                imgs.append(to_tensor(img_p, device))
                tgts.append(torch.from_numpy(tgt_p).unsqueeze(0).unsqueeze(0).to(device))

            imgs_t = torch.cat(imgs, dim=0)
            tgts_t = torch.cat(tgts, dim=0)

            optimizer.zero_grad()
            if scaler:
                with autocast():
                    loss = criterion(model(imgs_t), tgts_t)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs_t), tgts_t)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            if step % 25 == 0:
                elapsed = time.time() - t_epoch
                _log(f"epoch {epoch}/{args.epochs}, step {step}/{args.steps_per_epoch}, "
                     f"loss={epoch_loss/step:.5f}, elapsed={elapsed:.1f}s")

        epoch_loss /= args.steps_per_epoch
        elapsed = time.time() - t_epoch
        _log(f"epoch {epoch}/{args.epochs} done, loss={epoch_loss:.5f}, time={elapsed:.1f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_model(model, Path(args.model), config)
            _log(f"saved best model: {args.model}")

    _log(f"training complete in {time.time()-t_total:.1f}s, best_loss={best_loss:.5f}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def infer_tiled(
    model: SmallUNet,
    rgb: np.ndarray,
    device: "torch.device",
    tile_size: int = 512,
    overlap: int = 64,
    threshold: float = 0.5,
    amp: bool = False,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    # Pad to multiple of tile_size
    pad_h = math.ceil(h / tile_size) * tile_size
    pad_w = math.ceil(w / tile_size) * tile_size
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    padded[:h, :w] = rgb

    prob_map = np.zeros((pad_h, pad_w), dtype=np.float32)
    count_map = np.zeros((pad_h, pad_w), dtype=np.float32)

    stride = tile_size - overlap
    ys = list(range(0, pad_h - tile_size + 1, stride))
    xs = list(range(0, pad_w - tile_size + 1, stride))
    if not ys or ys[-1] + tile_size < pad_h:
        ys.append(pad_h - tile_size)
    if not xs or xs[-1] + tile_size < pad_w:
        xs.append(pad_w - tile_size)

    tiles = [(y, x) for y in ys for x in xs]
    total = len(tiles)
    _log(f"inference: image={w}x{h}, padded={pad_w}x{pad_h}, tiles={total}, "
         f"tile_size={tile_size}, overlap={overlap}")
    t0 = time.time()

    for i, (ty, tx) in enumerate(tiles):
        patch = padded[ty:ty+tile_size, tx:tx+tile_size]
        t = to_tensor(patch, device)

        with torch.no_grad():
            if amp and device.type == "cuda":
                with autocast():
                    logits = model(t)
            else:
                logits = model(t)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        prob_map[ty:ty+tile_size, tx:tx+tile_size] += prob
        count_map[ty:ty+tile_size, tx:tx+tile_size] += 1.0

        if (i + 1) % 20 == 0 or (i + 1) == total:
            _log(f"inference tile {i+1}/{total} ({(i+1)/total*100:.1f}%), elapsed={time.time()-t0:.1f}s")

    avg_prob = prob_map[:h, :w] / np.maximum(count_map[:h, :w], 1.0)
    return avg_prob > threshold


def process_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    _log(f"device: {device}")
    model, _ = load_model(Path(args.model), device)

    _log(f"loading image: {args.input}")
    rgb = np.array(Image.open(args.input).convert("RGB"))
    h, w = rgb.shape[:2]
    _log(f"loaded {w}x{h}")

    remove_mask = infer_tiled(model, rgb, device, threshold=args.threshold, amp=args.amp)

    alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])

    out_path = Path(args.output)
    _log(f"saved: {out_path}")
    Image.fromarray(rgba, mode="RGBA").save(str(out_path))
    preview = rgb.copy()
    preview[remove_mask] = [255, 0, 0]
    Image.fromarray(preview).save(str(out_path.with_name(out_path.stem + "_red_preview.png")))
    _log("done")


def process_folder_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, _ = load_model(Path(args.model), device)
    in_dir, out_dir = Path(args.input), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(in_dir.glob("*.png")):
        out_path = out_dir / img_path.name
        _log(f"processing {img_path.name}...")
        rgb = np.array(Image.open(img_path).convert("RGB"))
        remove_mask = infer_tiled(model, rgb, device, threshold=args.threshold, amp=args.amp)
        alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
        Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(str(out_path))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not HAS_TORCH:
        raise SystemExit("PyTorch not installed. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    # train
    train = sub.add_parser("train")
    train.add_argument("--samples", default="samples")
    train.add_argument("--model", required=True)
    train.add_argument("--resume")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--steps-per-epoch", type=int, default=500, dest="steps_per_epoch")
    train.add_argument("--batch-size", type=int, default=2, dest="batch_size")
    train.add_argument("--patch-size", type=int, default=512, dest="patch_size")
    train.add_argument("--device", default="cpu")
    train.add_argument("--amp", action="store_true")
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--alpha-threshold", type=int, default=128, dest="alpha_threshold")
    train.set_defaults(func=train_command)

    # process
    proc = sub.add_parser("process")
    proc.add_argument("input")
    proc.add_argument("output")
    proc.add_argument("--model", default="models/manhwa_cleaner.pt")
    proc.add_argument("--device", default="cpu")
    proc.add_argument("--threshold", type=float, default=0.5)
    proc.add_argument("--amp", action="store_true")
    proc.set_defaults(func=process_command)

    # process-folder
    pf = sub.add_parser("process-folder")
    pf.add_argument("--input", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--model", default="models/manhwa_cleaner.pt")
    pf.add_argument("--device", default="cpu")
    pf.add_argument("--threshold", type=float, default=0.5)
    pf.add_argument("--amp", action="store_true")
    pf.set_defaults(func=process_folder_command)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
