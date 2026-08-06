"""Part 2's whole-page pipeline for bubbles -- mirrors sfx_instance_pipeline.py exactly, swapping
find_sfx_instances() for find_bubble_instances() (build_bubble_instance_crops.py, itself a thin
wrapper around the existing, already-validated _find_bubble_interior_holes detector)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_cleaner import GuidanceParams, build_input_tensor  # noqa: E402
from instance_sfx_net import TinyInstanceNet  # noqa: E402
from build_sfx_instance_crops import crop_and_pad, CROP_SIZE  # noqa: E402
from build_bubble_instance_crops import find_bubble_instances, MARGIN  # noqa: E402

CKPT_PATH = ROOT / ".tmp/checkpoints/instance_bubble_smoke/instance_bubble_with_bg_weighted.pt"

_model_cache: dict[str, TinyInstanceNet] = {}


def load_bubble_instance_model(ckpt_path: Path = CKPT_PATH) -> TinyInstanceNet:
    key = str(ckpt_path)
    if key not in _model_cache:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = TinyInstanceNet(in_ch=ckpt["in_ch"], base=ckpt["base"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def apply_bubble_instance_refine(
    rgb: np.ndarray, delete_mask: np.ndarray, model: TinyInstanceNet | None = None,
) -> tuple[np.ndarray, list[dict]]:
    if model is None:
        model = load_bubble_instance_model()

    boxes = find_bubble_instances(rgb)
    out = delete_mask.copy()
    if not boxes:
        return out, []

    input_tensor = build_input_tensor(rgb, GuidanceParams())
    h, w = delete_mask.shape
    half = CROP_SIZE // 2
    info = []

    for (x, y, bw, bh) in boxes:
        if max(bw, bh) + 2 * MARGIN > 4 * CROP_SIZE:
            continue
        cx, cy = x + bw // 2, y + bh // 2
        crop = crop_and_pad(input_tensor, cx, cy, CROP_SIZE)

        with torch.no_grad():
            xt = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float()
            probs = torch.sigmoid(model(xt).squeeze(0).squeeze(0)).numpy()
        pred_delete = probs > 0.5

        x0, y0 = cx - half, cy - half
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w, x0 + CROP_SIZE), min(h, y0 + CROP_SIZE)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        dx0, dy0 = sx0 - x0, sy0 - y0

        before = out[sy0:sy1, sx0:sx1].copy()
        after = pred_delete[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)]
        out[sy0:sy1, sx0:sx1] = after

        info.append({
            "bbox": (x, y, bw, bh),
            "changed_frac_in_crop": float((before != after).mean()),
        })

    return out, info
