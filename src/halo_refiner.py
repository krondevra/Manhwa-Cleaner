"""Halo fix, 5th mechanism (.claude/plans/snazzy-cuddling-creek.md): a small, from-scratch,
TRUNK-INDEPENDENT crop-based refiner network, inspired by CascadePSP's design principle
(independent learned mask refinement, trained on perturbed ground truth) but using none of its
code or weights. See the plan file for full context: 4 in-trunk mechanisms (capacity increase,
boundary-weighted loss, an SDT head, and model 18.0's self-contained RefineHead) all made
boundary precision worse on this project's SmallUNet; the one mechanism that ever meaningfully
helped a related defect (CascadePSP) was trunk-independent. This module keeps that property
while fixing CascadePSP's two disqualifiers: 100% from-scratch training on this project's own
synthetic data (no third-party weights), and a narrow, bubble-crop-only scope (not full-page
multi-stage refinement) to keep inference cheap.

Part 0 (already completed, see halo_investigation.md) found the remaining `--close-bubble-halo`
failure case (inst3) isn't a small ink-outline gap -- it's the bubble's tail touching an
unrelated panel-divider line, merging two structurally different objects into one connected
component that hand-coded flood-fill geometry cannot safely separate. That's exactly the kind
of semantic ("is this connection part of the same object?") vs. topological ("are these pixels
connected?") distinction a learned refiner is suited for and pure geometry isn't.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from ml_cleaner import DoubleConv, crop_with_padding, find_dataset_pairs, read_rgb, read_rgba  # noqa: E402
from style_analysis import extract_enclosed_holes  # noqa: E402

CROP = 512  # 2026-08-02 Part A.4 (.claude/plans/snazzy-cuddling-creek.md): raised from 224
            # after Part A.2's inference-only 512px test showed no meaningful transfer either,
            # to isolate crop-size as a trained (not just inference-time) variable. Divisible
            # by 8 given HaloRefinerNet's 3 downsampling stages; covers the largest real
            # instance (398x260) with wide margin.
BUBBLE_SHAPE_CLASSES = {"oval", "thorn", "cloud", "spiky"}
DEFAULT_ALPHA_THRESHOLD = 128


# ---------------------------------------------------------------------------
# Part 2 -- architecture: small, separate, from scratch. Same DoubleConv/ConvTranspose2d
# building blocks as the main SmallUNet (ml_cleaner.py:493-572), reused directly since it's a
# generic, bubble-agnostic building block -- but only 3 downsampling levels (not 4) and a much
# narrower channel count, since a 224px crop needs less depth/receptive field than the main
# model's full 512px training patch. No import from data/CascadePSP or any third-party model
# code anywhere in this file.
# ---------------------------------------------------------------------------
class HaloRefinerNet(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 24) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 4, base * 6)

        self.up3 = nn.ConvTranspose2d(base * 6, base * 4, 2, stride=2)
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
        m = self.mid(self.pool(c3))

        u3 = self.up3(m)
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
        import torch.nn.functional as F
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Part 1 -- calibrated halo perturbation.
#
# NOT a reuse of ml_cleaner.synthesize_coarse_mask_perturbation as-is: that one is generic
# erode/dilate/random-holes with no outward-directional or curvature bias. This mimics the
# MEASURED halo shape specifically (halo_investigation.md's ring-distance data): a
# false-"keep" band starting right at the true boundary, most severe close in, decaying
# outward over a calibrated width -- e.g. inst1 pre-fix showed delete-probability 0/0/0/0/0.09
# at +2/+4/+8/+16/+32px, i.e. the ring is almost entirely wrongly "kept" through 16px, only
# slightly recovering by 32px. Severity and width are randomized per sample so the refiner
# sees a distribution of halo severities, not one fixed pattern.
# ---------------------------------------------------------------------------
def synthesize_halo_perturbation(
    delete_mask_crop: np.ndarray,
    contour_local: np.ndarray,
    rng: random.Random,
    severity: Optional[float] = None,
    halo_width: Optional[int] = None,
) -> np.ndarray:
    """Flips a calibrated outward ring of `delete_mask_crop` (True=delete) from delete->keep,
    starting at the bubble's own true contour, simulating the halo defect. Never touches
    pixels inside the contour itself. Returns a new array; input is not mutated."""
    if severity is None:
        severity = rng.uniform(0.5, 1.0)
    if halo_width is None:
        halo_width = rng.randint(8, 32)

    h, w = delete_mask_crop.shape
    interior = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(interior, [contour_local], -1, 1, thickness=-1)

    # Distance from each pixel to the nearest interior pixel -- 0 inside the contour, growing
    # outward. cv2.distanceTransform requires a single-channel 8-bit source where the "0"
    # pixels are the targets to measure distance to, so invert the interior mask.
    dist_outside = cv2.distanceTransform((1 - interior).astype(np.uint8), cv2.DIST_L2, 5)

    ring = (dist_outside > 0) & (dist_outside <= halo_width) & delete_mask_crop
    if not ring.any():
        return delete_mask_crop.copy()

    flip_prob = severity * np.clip(1.0 - dist_outside / halo_width, 0.0, 1.0)
    local_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    draw = local_rng.random_sample((h, w))
    flip = ring & (draw < flip_prob)

    out = delete_mask_crop.copy()
    out[flip] = False
    return out


# ---------------------------------------------------------------------------
# Crop-based training dataset. Structurally templated on
# src/train_cascadepsp_pc.py::StratifiedRefinementDataset (a much closer fit than the main
# PatchDataset, which is full-resolution-page-oriented with an LRU page cache) -- centers
# each crop on a detected bubble/cloud contour instead of a stratified sample point.
# ---------------------------------------------------------------------------
class HaloRefinerCropDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
        negative_frac: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.alpha_threshold = alpha_threshold
        self.negative_frac = negative_frac
        self.rng = random.Random(seed)
        # Pre-scan every pair for bubble/cloud contours once at construction time (cheap
        # relative to training itself, and lets __len__ reflect real bubble instances rather
        # than an arbitrary steps-per-epoch count). Each entry also carries a "hard" flag
        # (bbox touches/overlaps another accepted hole or a wide, panel-border-like stroke
        # component) -- directly targets Part 0's inst3 finding (tail touching an unrelated
        # structure), oversampled below.
        self.samples: list[dict] = []
        for original, cleaned in pairs:
            rgb = read_rgb(original)
            rgba = read_rgba(cleaned)
            delete_mask = rgba[:, :, 3] < alpha_threshold
            holes = extract_enclosed_holes(rgb)
            candidates = [h for h in holes if h["class"] in BUBBLE_SHAPE_CLASSES and not h["is_frame"]]
            # extract_enclosed_holes finds ink-darkness-bounded regions in the RGB alone --
            # it has no notion of the ground-truth mask, so it can (and does, confirmed on
            # this exact corpus) find a "hole" that ISN'T actually a protected keep region at
            # all -- e.g. the same dark-scene-contamination pattern already documented for
            # this function elsewhere in the project (a whole dark background scene can trip
            # the same gray<=FRAME_DARKNESS threshold a real ink stroke does). Only accept a
            # hole as a genuine bubble-training-target if its own interior is actually mostly
            # "keep" in the ground truth -- otherwise it's a false positive for our purposes,
            # regardless of how it's shape-classified.
            accepted = []
            for hole in candidates:
                interior = np.zeros(delete_mask.shape, dtype=np.uint8)
                cv2.drawContours(interior, [hole["contour"]], -1, 1, thickness=-1)
                interior_bool = interior.astype(bool)
                if interior_bool.sum() == 0:
                    continue
                delete_frac = delete_mask[interior_bool].mean()
                if delete_frac > 0.3:
                    continue
                # extract_enclosed_holes' own "bbox" field is stale/region-local (computed by
                # classify_and_measure BEFORE "contour" is offset to page-absolute coordinates
                # -- same bug already found and worked around this session in
                # halo_instance_crop_check.py). Always recompute bbox from the (correctly
                # offset) contour itself, never trust hole["bbox"] directly.
                bbox = cv2.boundingRect(hole["contour"])
                accepted.append({**hole, "bbox": bbox})
            for i, hole in enumerate(accepted):
                hard = self._touches_other(hole, accepted[:i] + accepted[i + 1:])
                self.samples.append({
                    "original": original, "cleaned": cleaned,
                    "contour": hole["contour"], "bbox": hole["bbox"], "hard": hard,
                })
        # Oversample hard cases (2x) so they're meaningfully represented despite being rarer.
        hard_samples = [s for s in self.samples if s["hard"]]
        self.samples.extend(hard_samples)
        self.rng.shuffle(self.samples)

    @staticmethod
    def _touches_other(hole: dict, others: list[dict], margin: int = 16) -> bool:
        x, y, cw, ch = hole["bbox"]
        bx0, by0, bx1, by1 = x - margin, y - margin, x + cw + margin, y + ch + margin
        for other in others:
            ox, oy, ocw, och = other["bbox"]
            ox0, oy0, ox1, oy1 = ox, oy, ox + ocw, oy + och
            if bx0 < ox1 and bx1 > ox0 and by0 < oy1 and by1 > oy0:
                return True
        return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rgb = read_rgb(s["original"])
        rgba = read_rgba(s["cleaned"])
        delete_mask = rgba[:, :, 3] < self.alpha_threshold

        x, y, cw, ch = s["bbox"]
        cx, cy = x + cw // 2, y + ch // 2
        h, w = delete_mask.shape
        # crop_with_padding does not clamp negative x0/y0 itself (its other caller,
        # PatchDataset, always passes values >= 0 by construction) -- a bubble near a page
        # edge can easily put cx-CROP//2 below 0, and negative numpy slice indices silently
        # wrap from the array's end instead of clamping, producing a garbage crop. Clamp here.
        x0 = max(0, min(cx - CROP // 2, max(0, w - CROP)))
        y0 = max(0, min(cy - CROP // 2, max(0, h - CROP)))
        arr_crop, mask_crop, _ = crop_with_padding(rgb, delete_mask, x0, y0, CROP)

        contour_local = s["contour"] - np.array([x0, y0])

        is_negative = self.rng.random() < self.negative_frac
        if is_negative:
            input_mask = mask_crop.copy()
        else:
            input_mask = synthesize_halo_perturbation(mask_crop, contour_local, self.rng)

        rgb_f = arr_crop.astype(np.float32) / 255.0
        input_mask_f = input_mask.astype(np.float32)[None, :, :]
        model_input = np.concatenate([rgb_f.transpose(2, 0, 1), input_mask_f], axis=0)

        image = torch.from_numpy(model_input)
        target = torch.from_numpy(mask_crop.astype(np.float32)[None, :, :])
        return image, target


# ---------------------------------------------------------------------------
# Priority 1 (2026-08-01 autonomous continuation, .tmp/notes/halo_refiner_autonomous_log.md):
# the perturbation-based dataset above trained a refiner that perfectly reversed its OWN
# synthesize_halo_perturbation function but showed ZERO transfer to a differently-constructed
# synthetic test or any real instance -- diagnosed as overfitting to that function's specific
# statistical signature, not the real halo defect. This class instead trains on the b2
# checkpoint's ACTUAL predicted errors (verified present on synthetic data too via
# verify_synthetic_halo_presence.py: mean predicted delete-frac 0.0138/0.0681/0.368/0.594/0.822
# at +2/+4/+8/+16/+32px, the same qualitative shape as real instances) against clean synthetic
# ground truth -- removing the "wrong proxy" problem entirely.
# ---------------------------------------------------------------------------
def precompute_model_predictions(
    pairs: list[tuple[Path, Path]],
    model: nn.Module,
    device: torch.device,
    guidance_params,
    cache_dir: Path,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> list[dict]:
    """Runs the (already-loaded) Stage1+2 model once per page, caching each predicted delete
    mask to `cache_dir` as a compressed boolean .npz (so repeated crop sampling during training
    doesn't re-run full-page inference every epoch). Returns a manifest list of
    {"original", "cleaned", "pred_path"} dicts."""
    from ml_cleaner import predict_delete_mask

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for original, cleaned in pairs:
        pred_path = cache_dir / f"{original.stem}_pred.npz"
        if not pred_path.exists():
            rgb = read_rgb(original)
            pred_delete = predict_delete_mask(
                rgb=rgb, model=model, device=device, guidance_params=guidance_params,
                tile_size=768, overlap=96, threshold=0.5, amp=False,
            )
            np.savez_compressed(pred_path, pred_delete=pred_delete)
        manifest.append({"original": original, "cleaned": cleaned, "pred_path": pred_path})
    return manifest


class HaloRefinerRealErrorCropDataset(Dataset):
    """Same crop-centering/hard-case/false-positive-hole filtering logic as
    HaloRefinerCropDataset, but the INPUT mask is the model's own precomputed real prediction
    (from precompute_model_predictions), not a synthetically perturbed version of the ground
    truth. Target is still the clean ground truth. No `negative_frac`/perturbation knobs --
    the model's own behavior already provides the full natural mix of correct and incorrect
    predictions."""

    def __init__(
        self,
        manifest: list[dict],
        alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
        seed: int = 0,
    ) -> None:
        self.alpha_threshold = alpha_threshold
        self.rng = random.Random(seed)

        self.samples: list[dict] = []
        for entry in manifest:
            original, cleaned, pred_path = entry["original"], entry["cleaned"], entry["pred_path"]
            rgb = read_rgb(original)
            rgba = read_rgba(cleaned)
            delete_mask = rgba[:, :, 3] < alpha_threshold
            holes = extract_enclosed_holes(rgb)
            candidates = [h for h in holes if h["class"] in BUBBLE_SHAPE_CLASSES and not h["is_frame"]]

            accepted = []
            for hole in candidates:
                interior = np.zeros(delete_mask.shape, dtype=np.uint8)
                cv2.drawContours(interior, [hole["contour"]], -1, 1, thickness=-1)
                interior_bool = interior.astype(bool)
                if interior_bool.sum() == 0:
                    continue
                if delete_mask[interior_bool].mean() > 0.3:
                    continue
                bbox = cv2.boundingRect(hole["contour"])
                accepted.append({**hole, "bbox": bbox})

            for i, hole in enumerate(accepted):
                hard = HaloRefinerCropDataset._touches_other(hole, accepted[:i] + accepted[i + 1:])
                self.samples.append({
                    "original": original, "cleaned": cleaned, "pred_path": pred_path,
                    "contour": hole["contour"], "bbox": hole["bbox"], "hard": hard,
                })
        hard_samples = [s for s in self.samples if s["hard"]]
        self.samples.extend(hard_samples)
        self.rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        rgb = read_rgb(s["original"])
        rgba = read_rgba(s["cleaned"])
        gt_delete = rgba[:, :, 3] < self.alpha_threshold
        pred_delete = np.load(s["pred_path"])["pred_delete"]

        x, y, cw, ch = s["bbox"]
        cx, cy = x + cw // 2, y + ch // 2
        h, w = gt_delete.shape
        x0 = max(0, min(cx - CROP // 2, max(0, w - CROP)))
        y0 = max(0, min(cy - CROP // 2, max(0, h - CROP)))

        arr_crop, gt_crop, _ = crop_with_padding(rgb, gt_delete, x0, y0, CROP)
        _, pred_crop, _ = crop_with_padding(rgb, pred_delete, x0, y0, CROP)

        rgb_f = arr_crop.astype(np.float32) / 255.0
        input_mask_f = pred_crop.astype(np.float32)[None, :, :]
        model_input = np.concatenate([rgb_f.transpose(2, 0, 1), input_mask_f], axis=0)

        image = torch.from_numpy(model_input)
        target = torch.from_numpy(gt_crop.astype(np.float32)[None, :, :])
        return image, target
