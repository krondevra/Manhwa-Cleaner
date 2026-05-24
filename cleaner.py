#!/usr/bin/env python3
"""
cleaner.py — production-ready manhwa background cleaner.

Renamed from manhwa_ml_cleaner_v2.py. Changes:
  - default model path: models/r_cleaner.pt
  - --debug flag on process command (saves *_debug_*.png masks)
  - --postprocess flag for light morphological smoothing of output alpha
  - process command always saves both result.png and result_red_preview.png

Architecture: SmallUNet (7-channel input: RGB + threshold + morph_close +
morph_open + Canny edges). See manhwa_ml_cleaner_v2.py for full rationale.

Training commands:
  # fresh start:
  python cleaner.py train \
      --samples samples --model models/r_cleaner.pt \
      --epochs 20 --steps-per-epoch 300 --batch-size 2 --patch-size 512 \
      --device cpu 2>&1 | tee logs/train.log

  # resume from existing checkpoint:
  python cleaner.py train \
      --samples samples --model models/cleaner_next.pt \
      --resume models/r_cleaner.pt \
      --epochs 20 --steps-per-epoch 300 --batch-size 2 --patch-size 512 \
      --device cpu 2>&1 | tee logs/train_resume.log

Inference commands:
  python cleaner.py process chapters-long/005.png chapters-results/005_result.png \
      --model models/r_cleaner.pt --device cpu

  python cleaner.py process chapters-long/005.png chapters-results/005_result.png \
      --model models/r_cleaner.pt --device cpu --threshold 0.40   # more aggressive
      --model models/r_cleaner.pt --device cpu --threshold 0.60   # more conservative

  python cleaner.py process-folder \
      --input chapters-long --output chapters-results \
      --model models/r_cleaner.pt --device cpu
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
    from torch.cuda.amp import GradScaler, autocast
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CLEAN_SUFFIX = "_cleaned"
IN_CHANNELS = 7
DEFAULT_MODEL = "models/r_cleaner.pt"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def build_guidance_channels(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    thr_f = thr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel).astype(np.float32) / 255.0
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel).astype(np.float32) / 255.0
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    return np.stack([thr_f, closed, opened, edges], axis=2)


def rgb_to_input(rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([rgb.astype(np.float32) / 255.0, build_guidance_channels(rgb)], axis=2)


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
    def __init__(self, in_channels: int = IN_CHANNELS) -> None:
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


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        dice = 1 - (2 * (probs * targets).sum() + smooth) / (probs.sum() + targets.sum() + smooth)
        return 0.5 * bce + 0.5 * dice


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    input_arr: np.ndarray   # H×W×7 float32
    target: np.ndarray      # H×W float32


def load_samples(samples_dir: Path, alpha_threshold: int = 128) -> List[Sample]:
    results = []
    files = sorted(p for p in samples_dir.glob("*.png") if not p.stem.endswith(CLEAN_SUFFIX))
    for orig_path in files:
        clean_path = orig_path.with_name(orig_path.stem + CLEAN_SUFFIX + ".png")
        if not clean_path.exists():
            continue
        _log(f"loading sample: {orig_path.name} + {clean_path.name}")
        rgb = np.array(Image.open(orig_path).convert("RGB"))
        cleaned = np.array(Image.open(clean_path).convert("RGBA"))
        if rgb.shape[:2] != cleaned.shape[:2]:
            raise ValueError(f"Size mismatch: {orig_path.name} and {clean_path.name}")
        inp = rgb_to_input(rgb)
        target = (cleaned[:, :, 3] < alpha_threshold).astype(np.float32)
        delete_ratio = target.mean()
        _log(f"  size={rgb.shape[1]}x{rgb.shape[0]}, delete_ratio={delete_ratio:.4f}")
        results.append(Sample(input_arr=inp, target=target))
    return results


def random_patch(sample: Sample, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = sample.input_arr.shape[:2]
    y = random.randint(0, max(0, h - patch_size))
    x = random.randint(0, max(0, w - patch_size))
    ph, pw = min(patch_size, h - y), min(patch_size, w - x)
    img_p = sample.input_arr[y:y+ph, x:x+pw]
    tgt_p = sample.target[y:y+ph, x:x+pw]
    if ph < patch_size or pw < patch_size:
        img_out = np.zeros((patch_size, patch_size, IN_CHANNELS), dtype=np.float32)
        tgt_out = np.zeros((patch_size, patch_size), dtype=np.float32)
        img_out[:ph, :pw] = img_p
        tgt_out[:ph, :pw] = tgt_p
        return img_out, tgt_out
    return img_p, tgt_p


def to_tensor(arr: np.ndarray, device: "torch.device") -> "torch.Tensor":
    t = torch.from_numpy(arr)
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
    config = checkpoint.get("config", {"in_channels": IN_CHANNELS})
    model = SmallUNet(in_channels=config.get("in_channels", IN_CHANNELS))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config


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

    config = {"in_channels": IN_CHANNELS}
    model = SmallUNet(in_channels=IN_CHANNELS).to(device)

    if args.resume:
        _log(f"loading checkpoint for resume: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    criterion = DiceBCELoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler() if (args.amp and device.type == "cuda") else None

    n_samples = len(samples)
    versioned_name = Path(args.model).stem + f"_{n_samples}chapters"
    versioned_path = Path(args.model).with_name(versioned_name + ".pt")

    best_loss = math.inf
    _log(f"training started — {n_samples} sample pair(s)")
    _log(f"versioned checkpoint will save to: {versioned_path}")
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

            imgs_t = torch.cat(imgs)
            tgts_t = torch.cat(tgts)

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
            save_model(model, versioned_path, config)
            _log(f"saved best model: {args.model} + {versioned_path}")

    _log(f"training complete in {time.time()-t_total:.1f}s, best_loss={best_loss:.5f}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def infer_tiled(
    model: SmallUNet,
    rgb: np.ndarray,
    device: "torch.device",
    tile_size: int = 768,
    overlap: int = 96,
    threshold: float = 0.5,
    amp: bool = False,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    pad_h = math.ceil(h / tile_size) * tile_size
    pad_w = math.ceil(w / tile_size) * tile_size

    inp = rgb_to_input(rgb)
    padded = np.zeros((pad_h, pad_w, IN_CHANNELS), dtype=np.float32)
    padded[:h, :w] = inp

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
        t_in = to_tensor(patch, device)
        with torch.no_grad():
            if amp and device.type == "cuda":
                with autocast():
                    logits = model(t_in)
            else:
                logits = model(t_in)
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
        _log(f"processing {img_path.name}...")
        rgb = np.array(Image.open(img_path).convert("RGB"))
        remove_mask = infer_tiled(model, rgb, device, threshold=args.threshold, amp=args.amp)
        alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
        Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(str(out_dir / img_path.name))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not HAS_TORCH:
        raise SystemExit("PyTorch not installed. Run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    train = sub.add_parser("train")
    train.add_argument("--samples", default="samples")
    train.add_argument("--model", required=True)
    train.add_argument("--resume")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--steps-per-epoch", type=int, default=300, dest="steps_per_epoch")
    train.add_argument("--batch-size", type=int, default=2, dest="batch_size")
    train.add_argument("--patch-size", type=int, default=512, dest="patch_size")
    train.add_argument("--device", default="cpu")
    train.add_argument("--amp", action="store_true")
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--alpha-threshold", type=int, default=128, dest="alpha_threshold")
    train.set_defaults(func=train_command)

    proc = sub.add_parser("process")
    proc.add_argument("input")
    proc.add_argument("output")
    proc.add_argument("--model", default=DEFAULT_MODEL)
    proc.add_argument("--device", default="cpu")
    proc.add_argument("--threshold", type=float, default=0.5)
    proc.add_argument("--amp", action="store_true")
    proc.add_argument("--postprocess", action="store_true")
    proc.add_argument("--debug", action="store_true")
    proc.set_defaults(func=process_command)

    pf = sub.add_parser("process-folder")
    pf.add_argument("--input", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--model", default=DEFAULT_MODEL)
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
