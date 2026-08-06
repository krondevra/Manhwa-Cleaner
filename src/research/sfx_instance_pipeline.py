"""Part 1.2/1.4 of the object-proposal-detector plan (.claude/plans/snazzy-cuddling-creek.md,
see notes/instance_aware_pivot_2026-08-03.md for the prior night's pre-cut-crop-only result).

Whole-page pipeline: detect SFX instance proposals -> crop -> TinyInstanceNet forward pass ->
paste back into the full-page delete mask. Mirrors ml_cleaner.py::apply_halo_refine's exact
shape (detect via an existing contour/stroke detector, crop, run a small per-instance model,
paste the valid region back) -- reused as the structural pattern, not reinvented.

One deliberate deviation from apply_halo_refine's own crop_with_padding call: that function
clamps the crop window to stay fully in-bounds when possible (shifting off-center near image
edges instead of zero-padding), which would put an edge-adjacent object off-center relative to
what the model was actually trained on (crop_and_pad in build_sfx_instance_crops.py always
centers exactly on the object, zero-padding past the edge). This matters concretely here --
the tracked ch1_sfx_text instance sits at x0=0, right at the real page's left edge. Centered
zero-padding is used instead, to stay consistent with the training distribution, at the cost of
diverging from the crop_with_padding convention other refiners use.
"""
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
from build_sfx_instance_crops import find_sfx_instances, crop_and_pad, MARGIN, CROP_SIZE  # noqa: E402

CKPT_PATH = ROOT / ".tmp/checkpoints/instance_sfx_smoke/instance_sfx_smoke_with_bg_weighted.pt"

_model_cache: dict[str, TinyInstanceNet] = {}


def load_sfx_instance_model(ckpt_path: Path = CKPT_PATH) -> TinyInstanceNet:
    key = str(ckpt_path)
    if key not in _model_cache:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = TinyInstanceNet(in_ch=ckpt["in_ch"], base=ckpt["base"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def apply_sfx_instance_refine(
    rgb: np.ndarray, delete_mask: np.ndarray, model: TinyInstanceNet | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Detect SFX proposals on rgb, run each through the instance-scoped model, paste the
    predicted crop back into delete_mask. Returns (refined_mask, per_instance_info) -- the
    info list lets a caller inspect what changed, not just trust the merged output blindly."""
    if model is None:
        model = load_sfx_instance_model()

    boxes = find_sfx_instances(rgb)
    out = delete_mask.copy()
    if not boxes:
        return out, []

    input_tensor = build_input_tensor(rgb, GuidanceParams())
    h, w = delete_mask.shape
    half = CROP_SIZE // 2
    info = []

    for (x, y, bw, bh) in boxes:
        if max(bw, bh) + 2 * MARGIN > 4 * CROP_SIZE:
            continue  # same size filter used when building training crops
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

        changed_frac = float((before != after).mean())
        obj_bx0, obj_by0 = max(0, x - sx0), max(0, y - sy0)
        obj_bx1, obj_by1 = min(sx1 - sx0, x + bw - sx0), min(sy1 - sy0, y + bh - sy0)
        obj_before = before[obj_by0:obj_by1, obj_bx0:obj_bx1]
        obj_after = after[obj_by0:obj_by1, obj_bx0:obj_bx1]
        info.append({
            "bbox": (x, y, bw, bh),
            "changed_frac_in_crop": changed_frac,
            "dense_delete_frac_at_bbox": float(obj_before.mean()) if obj_before.size else None,
            "instance_delete_frac_at_bbox": float(obj_after.mean()) if obj_after.size else None,
        })

    return out, info
