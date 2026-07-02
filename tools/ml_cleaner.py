#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:
    raise SystemExit("PyTorch is required to run this script.") from exc


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CLEAN_SUFFIX = "_cleaned"

DEFAULT_DATASET_DIR = Path("data/dataset_split/train")
DEFAULT_VAL_DATASET_DIR = Path("data/dataset_split/val")
# Fallback only: pre-split PepperNCarrotDataset layouts (data/dataset_split/*)
# are self-contained and keep the universal clean target as an
# "initial_cleaned" sibling folder inside each episode. This dir is only
# consulted when that sibling folder is missing (older, non-split dataset
# layouts that keep a single shared clean-render tree instead).
DEFAULT_RENDERS_CLEANED_DIR = Path("data/renders_cleaned")
# Base variants have no usable "cleaned" counterpart of their own (their
# *_cleaned siblings, where present, keep a frame outline or jpeg-degraded
# edge rather than being fully clean) -- they all pair against the one true
# fully-clean render, "initial_cleaned" (see DEFAULT_RENDERS_CLEANED_DIR
# fallback above for non-split layouts).
BASE_VARIANTS = ["initial", "black", "framed", "framed_jpeg", "gradient_border", "gradient_border_inv", "jpeg"]
# Overlay variants pair against their own *_cleaned sibling folder inside
# the dataset dir, since the overlay content (SFX/speech bubble) must be
# kept, not removed like a border.
OVERLAY_VARIANTS = ["sfx_overlay", "bubble_overlay"]
DEFAULT_CHAPTERS_LONG_DIR = Path("data/chapters-long")
DEFAULT_CHAPTERS_RESULTS_DIR = Path("data/chapters-results")
# Model path intentionally has no default.
# It must be provided explicitly with --model.


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def chapter_sort_key(path_or_stem: str | Path) -> tuple[int, str]:
    stem = Path(str(path_or_stem)).stem

    if stem.endswith("_result"):
        stem = stem[: -len("_result")]

    if stem.isdigit():
        return (0, f"{int(stem):09d}")

    return (1, stem)


def parse_chapter_list(value: str) -> list[str]:
    """
    Parse selected chapters without adding zero-padding.

    Examples:
        003,034,024 -> ["003", "034", "024"]
        3,34,24     -> ["3", "34", "24"]
        003-005,012 -> ["003", "004", "005", "012"]
        3-5,12      -> ["3", "4", "5", "12"]
    """
    import re

    if not value:
        return []

    raw_items = re.split(r"[,\s]+", value.strip())
    chapters: list[str] = []

    for item in raw_items:
        if not item:
            continue

        if re.fullmatch(r"\d+\s*-\s*\d+", item):
            left, right = re.split(r"\s*-\s*", item)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            width = max(len(left), len(right))

            for number in range(start, end + step, step):
                if width > 1:
                    chapters.append(f"{number:0{width}d}")
                else:
                    chapters.append(str(number))
            continue

        chapters.append(Path(item).stem)

    seen: set[str] = set()
    unique: list[str] = []

    for chapter in chapters:
        if chapter in seen:
            continue
        seen.add(chapter)
        unique.append(chapter)

    return unique


def looks_like_chapter_id(value: str | Path) -> bool:
    path = Path(str(value))
    return path.parent == Path(".") and path.suffix == ""


def resolve_chapter_input(value: str | Path, chapters_dir: Path = DEFAULT_CHAPTERS_LONG_DIR) -> Path:
    path = expand_path(value)

    if looks_like_chapter_id(value):
        return chapters_dir / f"{Path(str(value)).stem}.png"

    return path


def resolve_chapter_output(
    input_path: Path,
    output_value: str | None,
    output_dir: Path = DEFAULT_CHAPTERS_RESULTS_DIR,
) -> Path:
    if output_value:
        return expand_path(output_value)

    return output_dir / f"{input_path.stem}_result.png"


def discover_input_files(
    input_dir: Path,
    chapters: str | None = None,
    chapter_from: str | None = None,
    chapter_to: str | None = None,
) -> list[Path]:
    if chapters:
        selected = parse_chapter_list(chapters)
        files: list[Path] = []
        missing: list[Path] = []

        for chapter in selected:
            path = input_dir / f"{chapter}.png"
            if path.exists():
                files.append(path)
            else:
                missing.append(path)

        if missing:
            raise FileNotFoundError(
                "Requested chapter(s) not found:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )

        return files

    files = [
        path
        for path in sorted(input_dir.iterdir(), key=chapter_sort_key)
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
        and not path.stem.endswith(CLEAN_SUFFIX)
        and not path.stem.endswith("_result")
        and not path.stem.endswith("_red_preview")
    ]

    if chapter_from:
        from_key = chapter_sort_key(chapter_from)
        files = [path for path in files if chapter_sort_key(path.stem) >= from_key]

    if chapter_to:
        to_key = chapter_sort_key(chapter_to)
        files = [path for path in files if chapter_sort_key(path.stem) <= to_key]

    return files



def now_str() -> str:
    return time.strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_str()}] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def read_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)


def save_rgba(path: Path, rgb: np.ndarray, delete_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha = np.full(delete_mask.shape, 255, dtype=np.uint8)
    alpha[delete_mask] = 0
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)


def save_red_preview(path: Path, rgb: np.ndarray, delete_mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = rgb.copy()
    preview[delete_mask] = (255, 0, 0)
    Image.fromarray(preview, "RGB").save(path)


def maybe_save_red_preview(enabled: bool, output_path: Path, rgb: np.ndarray, delete_mask: np.ndarray) -> None:
    if not enabled:
        return

    preview = output_path.with_name(output_path.stem + "_red_preview.png")
    save_red_preview(preview, rgb, delete_mask)
    log(f"saved preview: {preview}")


def find_dataset_pairs(
    dataset_dir: Path,
    renders_cleaned_dir: Path,
    variants: List[str],
) -> List[Tuple[Path, Path]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder was not found: {dataset_dir.resolve()}")

    pairs: List[Tuple[Path, Path]] = []
    for episode_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        self_contained_target = episode_dir / f"initial{CLEAN_SUFFIX}"
        for variant in variants:
            variant_dir = episode_dir / variant
            if not variant_dir.is_dir():
                continue

            if variant in OVERLAY_VARIANTS:
                target_dir = episode_dir / f"{variant}{CLEAN_SUFFIX}"
            elif self_contained_target.is_dir():
                target_dir = self_contained_target
            else:
                target_dir = renders_cleaned_dir / episode_dir.name
            if not target_dir.is_dir():
                continue

            for src in sorted(variant_dir.iterdir()):
                if src.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                target = target_dir / src.name
                if target.exists():
                    pairs.append((src, target))

    if not pairs:
        raise FileNotFoundError(
            f"No dataset pairs were found under {dataset_dir.resolve()}\n"
            f"(target render dir: {renders_cleaned_dir.resolve()}, variants: {variants})."
        )
    return pairs


@dataclass
class GuidanceParams:
    threshold_value: int = 90
    morph_radius: int = 2


def make_guidance_channels(rgb: np.ndarray, params: GuidanceParams) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold_bin = (gray <= params.threshold_value).astype(np.uint8) * 255

    radius = max(0, int(params.morph_radius))
    if radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        closed = cv2.morphologyEx(threshold_bin, cv2.MORPH_CLOSE, k, iterations=1)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k, iterations=1)
    else:
        closed = threshold_bin.copy()
        opened = threshold_bin.copy()

    edges = cv2.Canny(gray, 40, 120)

    t = threshold_bin.astype(np.float32) / 255.0
    c = closed.astype(np.float32) / 255.0
    o = opened.astype(np.float32) / 255.0
    e = edges.astype(np.float32) / 255.0
    return np.stack([t, c, o, e], axis=2)


def build_input_tensor(rgb: np.ndarray, guidance_params: GuidanceParams) -> np.ndarray:
    guidance = make_guidance_channels(rgb, guidance_params)
    image = rgb.astype(np.float32) / 255.0
    return np.concatenate([image, guidance], axis=2).astype(np.float32)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 7, base: int = 24) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.down4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 8, base * 12)

        self.up4 = nn.ConvTranspose2d(base * 12, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        c2 = self.down2(self.pool(c1))
        c3 = self.down3(self.pool(c2))
        c4 = self.down4(self.pool(c3))
        m = self.mid(self.pool(c4))

        u4 = self.up4(m)
        u4 = self._resize_like(u4, c4)
        u4 = self.conv4(torch.cat([u4, c4], dim=1))

        u3 = self.up3(u4)
        u3 = self._resize_like(u3, c3)
        u3 = self.conv3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(u3)
        u2 = self._resize_like(u2, c2)
        u2 = self.conv2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(u2)
        u1 = self._resize_like(u1, c1)
        u1 = self.conv1(torch.cat([u1, c1], dim=1))

        return self.out(u1)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, dice_weight: float = 0.65) -> None:
        super().__init__()
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32))
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight_tensor.to(logits.device))
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
        return bce * (1.0 - self.dice_weight) + dice * self.dice_weight


def crop_with_padding(arr: np.ndarray, mask: np.ndarray, x0: int, y0: int, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    x1 = min(w, x0 + patch_size)
    y1 = min(h, y0 + patch_size)
    arr_crop = arr[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]

    if arr_crop.shape[0] == patch_size and arr_crop.shape[1] == patch_size:
        return arr_crop.copy(), mask_crop.copy()

    out_arr = np.zeros((patch_size, patch_size, arr.shape[2]), dtype=arr.dtype)
    out_arr[:, :, :3] = 1.0
    out_mask = np.zeros((patch_size, patch_size), dtype=bool)
    out_arr[:arr_crop.shape[0], :arr_crop.shape[1]] = arr_crop
    out_mask[:mask_crop.shape[0], :mask_crop.shape[1]] = mask_crop
    return out_arr, out_mask


def augment_patch(arr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        arr = np.ascontiguousarray(arr[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    if random.random() < 0.6:
        arr = arr.copy()
        rgb = arr[:, :, :3]
        contrast = random.uniform(0.92, 1.08)
        brightness = random.uniform(-0.04, 0.04)
        rgb = np.clip((rgb - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        arr[:, :, :3] = rgb

    return arr, mask


def load_pair(original: Path, cleaned: Path, alpha_threshold: int, guidance_params: GuidanceParams) -> Tuple[np.ndarray, np.ndarray]:
    rgb = read_rgb(original)
    rgba = read_rgba(cleaned)
    if rgb.shape[:2] != rgba.shape[:2]:
        raise ValueError(f"Size mismatch: {original} and {cleaned}")
    delete_mask = rgba[:, :, 3] < alpha_threshold
    model_input = build_input_tensor(rgb, guidance_params)
    return model_input, delete_mask


def estimate_delete_ratio(pairs: List[Tuple[Path, Path]], alpha_threshold: int) -> Tuple[int, int]:
    """One lightweight pass over target alpha channels only (no guidance
    channels, nothing retained across iterations) to size the loss pos_weight
    without preloading the full dataset into memory."""
    total_delete = 0
    total_pixels = 0
    for _, cleaned in pairs:
        alpha = np.asarray(Image.open(cleaned).convert("RGBA"), dtype=np.uint8)[:, :, 3]
        mask = alpha < alpha_threshold
        total_delete += int(mask.sum())
        total_pixels += int(mask.size)
    return total_delete, total_pixels


class PatchDataset(Dataset):
    """Draws random training patches from `pairs` without preloading the
    whole dataset: each (original, cleaned) pair is read from disk and
    converted to a model-input tensor only the first time it's picked, then
    kept in a small LRU cache (bounded by `cache_size`) so memory use stays
    constant regardless of dataset size. Note: with DataLoader `--workers` >
    0, each worker process gets its own separate cache.
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        alpha_threshold: int,
        guidance_params: GuidanceParams,
        patch_size: int,
        patches_per_epoch: int,
        positive_patch_ratio: float,
        min_positive_pixels: int,
        augment: bool,
        cache_size: int = 8,
    ) -> None:
        self.pairs = pairs
        self.alpha_threshold = alpha_threshold
        self.guidance_params = guidance_params
        self.patch_size = patch_size
        self.patches_per_epoch = patches_per_epoch
        self.positive_patch_ratio = positive_patch_ratio
        self.min_positive_pixels = min_positive_pixels
        self.augment = augment
        self.cache_size = max(1, cache_size)
        self._cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]]" = OrderedDict()

    def __len__(self) -> int:
        return self.patches_per_epoch

    def _get(self, sample_index: int):
        if sample_index in self._cache:
            self._cache.move_to_end(sample_index)
            return self._cache[sample_index]

        original, cleaned = self.pairs[sample_index]
        arr, mask = load_pair(original, cleaned, self.alpha_threshold, self.guidance_params)
        ys, xs = np.where(mask)
        positive_coords = (ys, xs) if len(xs) else None

        self._cache[sample_index] = (arr, mask, positive_coords)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arr, mask, positive_coords

    def __getitem__(self, index: int):
        ps = self.patch_size
        for _ in range(40):
            sample_index = random.randrange(len(self.pairs))
            arr, mask, coords = self._get(sample_index)
            h, w = mask.shape
            want_positive = random.random() < self.positive_patch_ratio

            if want_positive and coords is not None:
                ys, xs = coords
                p = random.randrange(len(xs))
                cy = int(ys[p])
                cx = int(xs[p])
                y0 = cy - random.randint(0, ps - 1)
                x0 = cx - random.randint(0, ps - 1)
            else:
                y0 = random.randint(0, max(0, h - ps))
                x0 = random.randint(0, max(0, w - ps))

            y0 = max(0, min(y0, max(0, h - ps)))
            x0 = max(0, min(x0, max(0, w - ps)))
            arr_crop, mask_crop = crop_with_padding(arr, mask, x0, y0, ps)

            if want_positive and int(mask_crop.sum()) < self.min_positive_pixels:
                continue

            if self.augment:
                arr_crop, mask_crop = augment_patch(arr_crop, mask_crop)

            image = torch.from_numpy(arr_crop.transpose(2, 0, 1).astype(np.float32))
            target = torch.from_numpy(mask_crop.astype(np.float32)[None, :, :])
            return image, target

        sample_index = random.randrange(len(self.pairs))
        arr, mask, _ = self._get(sample_index)
        h, w = mask.shape
        y0 = random.randint(0, max(0, h - ps))
        x0 = random.randint(0, max(0, w - ps))
        arr_crop, mask_crop = crop_with_padding(arr, mask, x0, y0, ps)
        if self.augment:
            arr_crop, mask_crop = augment_patch(arr_crop, mask_crop)
        image = torch.from_numpy(arr_crop.transpose(2, 0, 1).astype(np.float32))
        target = torch.from_numpy(mask_crop.astype(np.float32)[None, :, :])
        return image, target


def save_checkpoint(path: Path, model: nn.Module, config: dict, args: argparse.Namespace) -> None:
    checkpoint = {"state_dict": model.state_dict(), "config": config, "args": vars(args)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    path.with_suffix(".json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = SmallUNet(in_channels=int(config["in_channels"]), base=int(config["base_channels"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


def train_command(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    log(f"device: {device}")

    guidance_params = GuidanceParams(threshold_value=args.threshold_value, morph_radius=args.morph_radius)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    pairs = find_dataset_pairs(expand_path(args.dataset), expand_path(args.renders_cleaned), variants)
    log(f"found {len(pairs)} sample pairs")

    val_pairs: Optional[List[Tuple[Path, Path]]] = None
    if args.val_dataset:
        val_dir = expand_path(args.val_dataset)
        if val_dir.exists():
            val_pairs = find_dataset_pairs(val_dir, expand_path(args.renders_cleaned), variants)
            log(f"found {len(val_pairs)} validation pairs")
        else:
            log(f"validation dataset not found, skipping: {val_dir}")

    log("estimating delete ratio (single pass over target alpha channels)...")
    total_delete, total_pixels = estimate_delete_ratio(pairs, args.alpha_threshold)
    delete_ratio = total_delete / max(1, total_pixels)
    keep_ratio = 1.0 - delete_ratio
    raw_pos_weight = keep_ratio / max(delete_ratio, 1e-6)
    pos_weight = float(np.clip(raw_pos_weight, 0.5, args.max_pos_weight))

    log(f"training pixels: delete={total_delete:,}, total={total_pixels:,}, delete_ratio={delete_ratio:.4f}")
    log(f"pos_weight={pos_weight:.3f}")

    dataset = PatchDataset(
        pairs=pairs,
        alpha_threshold=args.alpha_threshold,
        guidance_params=guidance_params,
        patch_size=args.patch_size,
        patches_per_epoch=args.steps_per_epoch * args.batch_size,
        positive_patch_ratio=args.positive_patch_ratio,
        min_positive_pixels=args.min_positive_pixels,
        augment=not args.no_augment,
        cache_size=args.cache_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True)

    val_loader = None
    if val_pairs:
        val_dataset = PatchDataset(
            pairs=val_pairs,
            alpha_threshold=args.alpha_threshold,
            guidance_params=guidance_params,
            patch_size=args.patch_size,
            patches_per_epoch=args.val_steps * args.batch_size,
            positive_patch_ratio=args.positive_patch_ratio,
            min_positive_pixels=args.min_positive_pixels,
            augment=False,
            cache_size=args.cache_size,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

    in_channels = load_pair(*pairs[0], args.alpha_threshold, guidance_params)[0].shape[2]
    model = SmallUNet(in_channels=in_channels, base=args.base_channels).to(device)
    if args.resume and expand_path(args.resume).exists():
        log(f"loading checkpoint for resume: {args.resume}")
        checkpoint = torch.load(expand_path(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = DiceBCELoss(pos_weight=pos_weight, dice_weight=args.dice_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    model_path = expand_path(args.model)
    config = {
        "model_type": "SmallUNet",
        "in_channels": in_channels,
        "base_channels": args.base_channels,
        "patch_size": args.patch_size,
        "threshold": args.threshold,
        "alpha_threshold": args.alpha_threshold,
        "threshold_value": args.threshold_value,
        "morph_radius": args.morph_radius,
    }

    best_loss = float("inf")
    log("training started")
    t_all = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        t_epoch = time.time()

        for step, (images, masks) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())
            seen += 1

            if step % args.log_every == 0 or step == args.steps_per_epoch:
                avg = running / max(1, seen)
                elapsed = time.time() - t_epoch
                log(f"epoch {epoch}/{args.epochs}, step {step}/{args.steps_per_epoch}, loss={avg:.5f}, elapsed={elapsed:.1f}s")

            if step >= args.steps_per_epoch:
                break

        avg_epoch = running / max(1, seen)
        log(f"epoch {epoch}/{args.epochs} done, loss={avg_epoch:.5f}, time={time.time() - t_epoch:.1f}s")

        tracked_loss = avg_epoch
        if val_loader is not None and epoch % args.val_every == 0:
            model.eval()
            val_running = 0.0
            val_seen = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    logits = model(images)
                    val_running += float(criterion(logits, masks).item())
                    val_seen += 1
            tracked_loss = val_running / max(1, val_seen)
            log(f"epoch {epoch}/{args.epochs} val_loss={tracked_loss:.5f}")

        if tracked_loss < best_loss:
            best_loss = tracked_loss
            save_checkpoint(model_path, model, config, args)
            log(f"saved best model: {model_path}")

    log(f"training finished in {(time.time() - t_all) / 60.0:.1f} min")


def make_weight_window(tile_size: int) -> np.ndarray:
    one = np.hanning(tile_size)
    if one.max() <= 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    one = np.maximum(one, 0.05)
    window = np.outer(one, one).astype(np.float32)
    return window / window.max()


def pad_image_for_tiling(arr: np.ndarray, tile_size: int, stride: int):
    h, w = arr.shape[:2]

    if h <= tile_size:
        new_h = tile_size
    else:
        steps_h = math.ceil((h - tile_size) / stride)
        new_h = steps_h * stride + tile_size

    if w <= tile_size:
        new_w = tile_size
    else:
        steps_w = math.ceil((w - tile_size) / stride)
        new_w = steps_w * stride + tile_size

    padded = np.zeros((new_h, new_w, arr.shape[2]), dtype=arr.dtype)
    padded[:, :, :3] = 1.0
    padded[:h, :w] = arr
    return padded, (h, w)


@torch.no_grad()
def predict_delete_mask(rgb: np.ndarray, model: nn.Module, device: torch.device, guidance_params: GuidanceParams, tile_size: int, overlap: int, threshold: float, amp: bool):
    full = build_input_tensor(rgb, guidance_params)

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("--overlap must be >= 0 and smaller than --tile-size")

    stride = tile_size - overlap
    padded, (oh, ow) = pad_image_for_tiling(full, tile_size, stride)
    ph, pw = padded.shape[:2]

    prob_sum = np.zeros((ph, pw), dtype=np.float32)
    weight_sum = np.zeros((ph, pw), dtype=np.float32)
    window = make_weight_window(tile_size)

    y_positions = list(range(0, ph - tile_size + 1, stride))
    x_positions = list(range(0, pw - tile_size + 1, stride))
    total_tiles = len(y_positions) * len(x_positions)

    log(f"inference: image={ow}x{oh}, padded={pw}x{ph}, tiles={total_tiles}, tile_size={tile_size}, overlap={overlap}")
    t0 = time.time()
    tile_index = 0

    for y in y_positions:
        for x in x_positions:
            tile_index += 1
            tile = padded[y:y + tile_size, x:x + tile_size]
            image = torch.from_numpy(tile.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and amp)):
                logits = model(image)
                probs = torch.sigmoid(logits)[0, 0].float().cpu().numpy()

            prob_sum[y:y + tile_size, x:x + tile_size] += probs * window
            weight_sum[y:y + tile_size, x:x + tile_size] += window

            if tile_index % 20 == 0 or tile_index == total_tiles:
                percent = 100.0 * tile_index / max(1, total_tiles)
                elapsed = time.time() - t0
                log(f"inference tile {tile_index}/{total_tiles} ({percent:.1f}%), elapsed={elapsed:.1f}s")

    probability = prob_sum / np.maximum(weight_sum, 1e-6)
    probability = probability[:oh, :ow]
    return probability >= threshold


def postprocess_delete_mask(delete_mask: np.ndarray, close_radius: int, open_radius: int) -> np.ndarray:
    mask = delete_mask.astype(np.uint8)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if open_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius * 2 + 1, open_radius * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask.astype(bool)


def process_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    log(f"device: {device}")

    model_path = expand_path(args.model)
    model, config = load_model(model_path, device)
    threshold = args.threshold if args.threshold is not None else float(config.get("threshold", 0.5))
    guidance_params = GuidanceParams(
        threshold_value=int(config.get("threshold_value", args.threshold_value)),
        morph_radius=int(config.get("morph_radius", args.morph_radius)),
    )

    input_path = resolve_chapter_input(args.input, DEFAULT_CHAPTERS_LONG_DIR)
    output_path = resolve_chapter_output(input_path, args.output, DEFAULT_CHAPTERS_RESULTS_DIR)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    log(f"model: {model_path}")
    log(f"loading image: {input_path}")
    rgb = read_rgb(input_path)
    log(f"loaded {rgb.shape[1]}x{rgb.shape[0]}")

    delete_mask = predict_delete_mask(
        rgb=rgb,
        model=model,
        device=device,
        guidance_params=guidance_params,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=threshold,
        amp=args.amp,
    )

    if args.postprocess:
        delete_mask = postprocess_delete_mask(delete_mask, args.close_radius, args.open_radius)

    save_rgba(output_path, rgb, delete_mask)
    log(f"saved: {output_path}")

    maybe_save_red_preview(
        enabled=args.red_preview,
        output_path=output_path,
        rgb=rgb,
        delete_mask=delete_mask,
    )

    log("done")

def process_folder_command(args: argparse.Namespace) -> None:
    input_dir = expand_path(args.input)
    output_dir = expand_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    selected_chapters = getattr(args, "chapters", None)
    selected_from = getattr(args, "from_chapter", None)
    selected_to = getattr(args, "to_chapter", None)

    selection_modes = [
        bool(selected_chapters),
        bool(selected_from or selected_to),
    ]

    if sum(selection_modes) > 1:
        raise ValueError("Use either --chapters or --from-chapter/--to-chapter, not both.")

    files = discover_input_files(
        input_dir=input_dir,
        chapters=selected_chapters,
        chapter_from=selected_from,
        chapter_to=selected_to,
    )

    if not files:
        raise FileNotFoundError(f"No images found in: {input_dir.resolve()}")

    device = choose_device(args.device)
    log(f"device: {device}")

    model_path = expand_path(args.model)
    model, config = load_model(model_path, device)
    threshold = args.threshold if args.threshold is not None else float(config.get("threshold", 0.5))
    guidance_params = GuidanceParams(
        threshold_value=int(config.get("threshold_value", args.threshold_value)),
        morph_radius=int(config.get("morph_radius", args.morph_radius)),
    )

    log(f"model: {model_path}")
    log(f"input: {input_dir}")
    log(f"output: {output_dir}")
    log(f"processing {len(files)} files")

    for i, input_path in enumerate(files, start=1):
        output_path = output_dir / f"{input_path.stem}_result.png"
        log(f"[{i}/{len(files)}] {input_path.name} -> {output_path.name}")

        rgb = read_rgb(input_path)
        delete_mask = predict_delete_mask(
            rgb=rgb,
            model=model,
            device=device,
            guidance_params=guidance_params,
            tile_size=args.tile_size,
            overlap=args.overlap,
            threshold=threshold,
            amp=args.amp,
        )

        if args.postprocess:
            delete_mask = postprocess_delete_mask(delete_mask, args.close_radius, args.open_radius)

        save_rgba(output_path, rgb, delete_mask)

        maybe_save_red_preview(
            enabled=args.red_preview,
            output_path=output_path,
            rgb=rgb,
            delete_mask=delete_mask,
        )

    log("folder processing done")

def add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Required model path. Example: --model data/models/2.1.pt")
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=None, help="Override model threshold. Example: 0.45 or 0.60")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda")
    parser.add_argument("--amp", action="store_true", help="Mixed precision on CUDA/ROCm")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--open-radius", type=int, default=0)
    parser.add_argument(
        "--red-preview",
        action="store_true",
        help="Also save *_result_red_preview.png. Disabled by default.",
    )
    parser.add_argument("--threshold-value", type=int, default=90)
    parser.add_argument("--morph-radius", type=int, default=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manhwa ML cleaner for training and batch inference.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train from the Pepper & Carrot dataset under data/dataset_split/train/.")
    p_train.add_argument("--dataset", default=str(DEFAULT_DATASET_DIR), help="Dataset root. Example: data/dataset_split/train")
    p_train.add_argument(
        "--val-dataset",
        default=str(DEFAULT_VAL_DATASET_DIR),
        help="Held-out split used to pick the best checkpoint. Example: data/dataset_split/val. Pass '' to disable.",
    )
    p_train.add_argument(
        "--renders-cleaned",
        default=str(DEFAULT_RENDERS_CLEANED_DIR),
        help="Fallback universal clean-render target, only used when an episode has no self-contained "
        "initial_cleaned/ folder. Example: data/renders_cleaned",
    )
    p_train.add_argument(
        "--variants",
        default=",".join(BASE_VARIANTS + OVERLAY_VARIANTS),
        help="Comma-separated variant folders to train on. Example: framed,jpeg,sfx_overlay",
    )
    p_train.add_argument("--val-steps", type=int, default=50, help="Validation patches per epoch = val-steps * batch-size")
    p_train.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs")
    p_train.add_argument("--model", required=True, help="Required output model path. Example: --model data/models/2.1.pt")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--steps-per-epoch", type=int, default=300)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--patch-size", type=int, default=512)
    p_train.add_argument("--base-channels", type=int, default=24)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--dice-weight", type=float, default=0.65)
    p_train.add_argument("--max-pos-weight", type=float, default=4.0)
    p_train.add_argument("--positive-patch-ratio", type=float, default=0.70)
    p_train.add_argument("--min-positive-pixels", type=int, default=256)
    p_train.add_argument("--threshold", type=float, default=0.50)
    p_train.add_argument("--alpha-threshold", type=int, default=128)
    p_train.add_argument("--threshold-value", type=int, default=90, help="Threshold value used for guidance channels")
    p_train.add_argument("--morph-radius", type=int, default=2, help="Morphological radius used for guidance channels")
    p_train.add_argument(
        "--cache-size",
        type=int,
        default=8,
        help="Full images kept in the lazy-loading LRU cache (per DataLoader worker). Bounds memory regardless of dataset size.",
    )
    p_train.add_argument("--workers", type=int, default=0)
    p_train.add_argument("--device", default="auto", help="auto, cpu, cuda")
    p_train.add_argument("--seed", type=int, default=7)
    p_train.add_argument("--log-every", type=int, default=25)
    p_train.add_argument("--amp", action="store_true")
    p_train.add_argument("--no-augment", action="store_true")
    p_train.add_argument("--resume", default="", help="Optional checkpoint to resume from")
    p_train.set_defaults(func=train_command)

    p_process = sub.add_parser("process", help="Clean one chapter")
    p_process.add_argument("input")
    p_process.add_argument("output", nargs="?", help="Optional output path. Default: data/chapters-results/<input>_result.png")
    add_inference_args(p_process)
    p_process.set_defaults(func=process_command)

    p_folder = sub.add_parser("process-folder", help="Clean all chapters, selected chapters, or a chapter range")
    p_folder.add_argument("--input", default=str(DEFAULT_CHAPTERS_LONG_DIR))
    p_folder.add_argument("--output", default=str(DEFAULT_CHAPTERS_RESULTS_DIR))
    p_folder.add_argument("--chapters", default=None, help="Selected chapters. Example: 003,034,024 or 003-005,012")
    p_folder.add_argument("--from-chapter", default=None, help="Start chapter for range processing. Example: 003")
    p_folder.add_argument("--to-chapter", default=None, help="End chapter for range processing. Example: 175")
    add_inference_args(p_folder)
    p_folder.set_defaults(func=process_folder_command)

    p_range = sub.add_parser("process-range", help="Clean chapters by range from data/chapters-long")
    p_range.add_argument("--input", default=str(DEFAULT_CHAPTERS_LONG_DIR))
    p_range.add_argument("--output", default=str(DEFAULT_CHAPTERS_RESULTS_DIR))
    p_range.add_argument("--from-chapter", required=True, help="Start chapter. Example: 003")
    p_range.add_argument("--to-chapter", required=True, help="End chapter. Example: 175")
    p_range.set_defaults(chapters=None)
    add_inference_args(p_range)
    p_range.set_defaults(func=process_folder_command)

    p_list = sub.add_parser("process-list", help="Clean selected chapters from data/chapters-long")
    p_list.add_argument("--input", default=str(DEFAULT_CHAPTERS_LONG_DIR))
    p_list.add_argument("--output", default=str(DEFAULT_CHAPTERS_RESULTS_DIR))
    p_list.add_argument("--chapters", required=True, help="Selected chapters. Example: 003,034,024 or 003-005,012")
    p_list.set_defaults(from_chapter=None, to_chapter=None)
    add_inference_args(p_list)
    p_list.set_defaults(func=process_folder_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
