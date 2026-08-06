"""Attempt 8 (CRF layer): whole-page pipeline for applying CRFRefineNet to a delete-mask,
adapted directly from `ml_cleaner.apply_halo_refine`'s exact structure (mechanism 5's own
instance-scoped application) -- CRFRefineNet was deliberately built with the same (rgb,mask)-crop
-> mask I/O contract as HaloRefinerNet, so the detect -> crop -> model -> paste-back shape carries
over unchanged; only the model class and checkpoint config differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_cleaner import _find_bubble_interior_holes, crop_with_padding  # noqa: E402
from halo_refiner import CROP  # noqa: E402
from crf_refine_net import CRFRefineNet  # noqa: E402

FRAME_DARKNESS = 40
MIN_BUBBLE_AREA = 2000

_crf_cache: dict[tuple[str, str], CRFRefineNet] = {}


def load_crf_model(weights_path: Path, device: torch.device) -> CRFRefineNet:
    cache_key = (str(weights_path), str(device))
    model = _crf_cache.get(cache_key)
    if model is None:
        checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = CRFRefineNet(in_channels=config["in_channels"], base=config["base_channels"],
                              window=config["window"], n_iters=config["n_iters"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _crf_cache[cache_key] = model
    return model


def apply_crf_refine(
    rgb: np.ndarray,
    delete_mask: np.ndarray,
    weights_path: Path,
    device: torch.device,
    frame_darkness: int = FRAME_DARKNESS,
    min_bubble_area: float = MIN_BUBBLE_AREA,
) -> np.ndarray:
    """Applies CRFRefineNet to every detected bubble/cloud contour's crop, pasting the corrected
    result back into the full mask -- same detection (`_find_bubble_interior_holes` on the source
    RGB, not the predicted mask), same CROP=512 crop size (halo_refiner.CROP, matching what
    CRFRefineNet was trained on via HaloRefinerCropDataset), same crop-with-padding + paste-back
    pattern as `apply_halo_refine`. No-op if no contour is found."""
    model = load_crf_model(weights_path, device)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    holes = _find_bubble_interior_holes(gray, frame_darkness, min_bubble_area)
    if not holes:
        return delete_mask.copy()

    h, w = delete_mask.shape
    out = delete_mask.copy()
    for hole in holes:
        contour = hole["contour"]
        x, y, cw, ch = cv2.boundingRect(contour)
        cx, cy = x + cw // 2, y + ch // 2
        x0 = max(0, min(cx - CROP // 2, max(0, w - CROP)))
        y0 = max(0, min(cy - CROP // 2, max(0, h - CROP)))
        x1, y1 = min(w, x0 + CROP), min(h, y0 + CROP)

        arr_crop, mask_crop, _ = crop_with_padding(rgb, out, x0, y0, CROP)
        rgb_f = arr_crop.astype(np.float32) / 255.0
        mask_f = mask_crop.astype(np.float32)[None, :, :]
        model_input = np.concatenate([rgb_f.transpose(2, 0, 1), mask_f], axis=0)

        with torch.no_grad():
            tensor = torch.from_numpy(model_input).unsqueeze(0).to(device)
            logits = model(tensor)
            pred = (torch.sigmoid(logits) > 0.5).squeeze(0).squeeze(0).cpu().numpy()

        valid_h, valid_w = y1 - y0, x1 - x0
        out[y0:y1, x0:x1] = pred[:valid_h, :valid_w].astype(bool)

    return out
