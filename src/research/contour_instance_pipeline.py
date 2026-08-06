"""Attempt 7: whole-page pipeline for the contour deformation net -- converts a deformed contour
into an actual delete-mask correction, for regression-testing against the existing Stage 1/2/3
ROI battery (ch002_rois.json) and eventual CLI integration, mirroring sfx_instance_pipeline.py/
bubble_instance_pipeline.py's detect->crop->model->paste-back shape.

Semantics differ from the dense instance nets by necessity: a contour model defines a boundary
for exactly ONE object, not a dense prediction over the whole crop. The correction is therefore
scoped to a bounded "region of influence" around the object (the union of the initial and
deformed contours, dilated by a small margin) -- pixels outside that region are left completely
untouched, not overwritten by an undefined dense-field guess the way the dense nets' whole-crop
paste-back does.
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

from ml_cleaner import GuidanceParams, build_input_tensor, _find_bubble_interior_holes  # noqa: E402
from contour_common import build_angles, fit_init_ellipse_radii  # noqa: E402
from contour_deform_net import ContourDeformNet, vertex_xy_norm_from_radii, CROP_SIZE, N_VERTICES  # noqa: E402

FRAME_DARKNESS = 40
MIN_BUBBLE_AREA = 2000
REGION_MARGIN = 12  # px dilation around the influence region, avoids a hard-edge seam

_model_cache: dict[str, ContourDeformNet] = {}


def load_contour_model(ckpt_path: Path) -> ContourDeformNet:
    key = str(ckpt_path)
    if key not in _model_cache:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = ContourDeformNet(in_ch=ckpt["in_ch"], base=ckpt["base"], n_vertices=ckpt["n_vertices"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[key] = model
    return _model_cache[key]


def radii_to_polygon(cx: float, cy: float, radii: np.ndarray, angles: np.ndarray) -> np.ndarray:
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return np.stack([xs, ys], axis=1).astype(np.int32)


def apply_contour_instance_refine(
    rgb: np.ndarray, delete_mask: np.ndarray, model: ContourDeformNet,
) -> tuple[np.ndarray, list[dict]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    holes = _find_bubble_interior_holes(gray, FRAME_DARKNESS, MIN_BUBBLE_AREA)
    out = delete_mask.copy()
    if not holes:
        return out, []

    input_tensor = build_input_tensor(rgb, GuidanceParams())
    h, w = delete_mask.shape
    angles_np = build_angles(N_VERTICES)
    angles_t = torch.from_numpy(angles_np).float()
    half = CROP_SIZE // 2
    info = []

    for hole in holes:
        x, y, bw, bh = hole["bbox"]
        diag = float(np.hypot(bw, bh))
        if diag > 0.75 * CROP_SIZE:  # matches build_contour_training_data.py's MAX_BBOX_DIAG_FRAC
            continue
        m = cv2.moments(hole["contour"])
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        icx, icy = int(round(cx)), int(round(cy))

        init_radii_np = fit_init_ellipse_radii(angles_np, bw, bh)

        crop = np.zeros((CROP_SIZE, CROP_SIZE, 7), dtype=np.float32)
        sx0, sy0 = max(0, icx - half), max(0, icy - half)
        sx1, sy1 = min(w, icx + half), min(h, icy + half)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        dx0, dy0 = sx0 - (icx - half), sy0 - (icy - half)
        crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = input_tensor[sy0:sy1, sx0:sx1]

        init_radii_t = torch.from_numpy(init_radii_np).float().unsqueeze(0)
        vxy = vertex_xy_norm_from_radii(init_radii_t, angles_t, CROP_SIZE)
        crop_t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            pred_dr = model(crop_t, vxy).squeeze(0).numpy()
        deformed_radii = init_radii_np + pred_dr

        # rasterize in crop-local coords (crop is centered on (icx,icy))
        local_cx, local_cy = half, half
        deformed_poly = radii_to_polygon(local_cx, local_cy, deformed_radii, angles_np)
        init_poly = radii_to_polygon(local_cx, local_cy, init_radii_np, angles_np)

        deformed_mask = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        cv2.fillPoly(deformed_mask, [deformed_poly], 1)
        influence = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        cv2.fillPoly(influence, [init_poly], 1)
        influence = np.maximum(influence, deformed_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REGION_MARGIN * 2 + 1,) * 2)
        influence = cv2.dilate(influence, k)

        new_delete_local = (deformed_mask == 0)  # inside deformed contour = keep, outside = delete
        influence_bool = influence.astype(bool)

        # paste back only within [sy0:sy1, sx0:sx1] (the valid, non-padded region) AND the
        # influence mask -- never touch content outside this object's own bounded region.
        region_y0, region_y1 = dy0, dy0 + (sy1 - sy0)
        region_x0, region_x1 = dx0, dx0 + (sx1 - sx0)
        local_influence = influence_bool[region_y0:region_y1, region_x0:region_x1]
        local_new_delete = new_delete_local[region_y0:region_y1, region_x0:region_x1]

        target = out[sy0:sy1, sx0:sx1]
        before = target.copy()
        target[local_influence] = local_new_delete[local_influence]
        out[sy0:sy1, sx0:sx1] = target

        info.append({
            "bbox": (x, y, bw, bh),
            "changed_frac_in_influence": float((before[local_influence] != target[local_influence]).mean())
            if local_influence.any() else 0.0,
        })

    return out, info
