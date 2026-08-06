"""Attempt 7 (halo investigation, .claude/plans/snazzy-cuddling-creek.md): shared utilities for
a Deep-Snake-style parametric contour deformation network -- the first mechanism tried that has
no dense per-pixel probability field anywhere in its output, structurally removing the substrate
the halo investigation's occlusion probe showed a smooth, context-independent "keep" prior can
leak across (docs/ml_strategy_history.md's halo closure).

Deliberate simplification, disclosed not hidden: predicts a per-vertex RADIAL offset (one scalar
per angle, from a fixed centroid) rather than free 2D per-vertex offsets. This keeps the deformed
contour star-shaped-consistent by construction (no self-intersection risk) and halves the output
dimensionality -- a reasonable scope-down for a single-pass first attempt, still fundamentally
vertex/contour-based, not a dense field. Both the initial coarse contour and the training target
are parametrized as N equally-angle-spaced radii from a centroid, reusing the same ray-walking
technique real_boundary_probe.py already validated for real-instance boundary measurement in this
project's halo work (there: walking outward through real ink-darkness thresholds; here: walking
outward through an exact synthetic ground-truth mask, or -- for real-instance inference -- the
same real ink-darkness technique, so training and inference share one code path).
"""
from __future__ import annotations

import math

import numpy as np


def build_angles(n_vertices: int) -> np.ndarray:
    return np.linspace(0, 2 * math.pi, n_vertices, endpoint=False)


def ray_radius_mask(mask: np.ndarray, cx: float, cy: float, angle: float, max_r: int) -> float:
    """Walk outward from (cx,cy) along `angle` through a boolean mask (True = inside the shape).
    Returns the radius (px) of the last True pixel before the first False, i.e. where the ray
    exits the shape -- sub-pixel-free, nearest-integer-step version (matches
    real_boundary_probe.py's own step granularity)."""
    h, w = mask.shape
    dx, dy = math.cos(angle), math.sin(angle)
    last_inside = 0.0
    for r in range(1, max_r):
        x, y = int(round(cx + dx * r)), int(round(cy + dy * r))
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            return last_inside
        last_inside = float(r)
    return last_inside


def ray_radius_ink(gray: np.ndarray, cx: float, cy: float, angle: float, max_r: int,
                    dark_thresh: float = 110.0, min_run: int = 2) -> float | None:
    """Real-instance variant (same technique as real_boundary_probe.py::find_per_angle_boundary,
    reused not reimplemented from scratch -- inlined here since that script's version is scoped
    to its own CLI, not imported as a library function). Returns None if no dark-ink transition
    is found within max_r (a genuine no-marker angle, same semantics as the original)."""
    h, w = gray.shape
    dx, dy = math.cos(angle), math.sin(angle)
    dark_run = 0
    for r in range(1, max_r):
        x, y = int(round(cx + dx * r)), int(round(cy + dy * r))
        if not (0 <= x < w and 0 <= y < h):
            return None
        if gray[y, x] < dark_thresh:
            dark_run += 1
            if dark_run >= min_run:
                return float(r - min_run + 1)
        else:
            dark_run = 0
    return None


def fit_init_ellipse_radii(angles: np.ndarray, bbox_w: float, bbox_h: float,
                            shrink: float = 0.85) -> np.ndarray:
    """Coarse initial-contour radii: an axis-aligned ellipse inscribed in the object's own
    bounding box, per-angle radius via the standard polar ellipse formula. `shrink` pulls the
    ellipse slightly inside the bbox (a bbox-circumscribed ellipse typically overshoots a
    rounded/organic bubble shape at the corners) -- a coarse, deliberately imperfect starting
    guess, not the true contour, matching Deep Snake's own "coarse detection to deform" premise."""
    a, b = (bbox_w / 2.0) * shrink, (bbox_h / 2.0) * shrink
    cos_t, sin_t = np.cos(angles), np.sin(angles)
    denom = np.sqrt((b * cos_t) ** 2 + (a * sin_t) ** 2)
    denom = np.maximum(denom, 1e-6)
    return (a * b) / denom
