"""Reusable style-analysis primitives (Part 4, .tmp/notes/synthetic_curriculum_plan.md).

Lifted and extended from src/probe_bubble_curvature.py's one-off inline logic (the
flood-fill-from-corner enclosed-hole detector, shared with src/ml_cleaner.py's
repair_frame_interiors, and the curvature/contour measurement code), plus a new frame/
bubble split and a 5-family shape taxonomy (oval, cloud/thought, spiky/"sea urchin", thorn,
rectangle -- standard in published comic/balloon-segmentation work, see
.tmp/notes/inspiration_papers_review.md's Dubray & Laubrock reference) needed for the
generation-6 synthetic-generator style analysis.

Read-only analysis code: extracts AGGREGATE STATISTICS only. Callers must never persist or
reuse the actual contours/pixels this module returns beyond a single run's in-memory
aggregation and QA-preview rendering -- see the plan's Part 4 constraint.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image
from pathlib import Path

FRAME_DARKNESS = 40  # matches --frame-darkness's own default (src/ml_cleaner.py)
MIN_SHAPE_AREA = 400  # px^2, at native resolution

# An enclosed component is treated as a PANEL FRAME, not a bubble/text candidate, if its
# bounding box covers a large fraction of the page in either axis -- real panels almost
# always span most of the page width and/or a substantial height; bubbles/text boxes don't.
# Heuristic, not exact -- the QA preview step (save_preview) exists specifically so this
# split can be sanity-checked visually before trusting any aggregate built on top of it.
FRAME_WIDTH_FRAC = 0.55
FRAME_HEIGHT_FRAC = 0.20


# ── contour geometry helpers ─────────────────────────────────────────────────

def resample_contour(pts: np.ndarray, n_samples: int) -> Optional[np.ndarray]:
    pts = np.vstack([pts, pts[0]])
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-6:
        return None
    targets = np.linspace(0, total, n_samples, endpoint=False)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, len(seg_len) - 1)
    seg_t = (targets - cum[idx]) / np.maximum(seg_len[idx], 1e-6)
    out = pts[idx] + seg_t[:, None] * (pts[idx + 1] - pts[idx])
    return out


def curvature_radii(resampled: np.ndarray, arc_step: float, window_px: float) -> np.ndarray:
    """Discrete radius-of-curvature estimate: turning angle over arc length
    between two points a window on either side of each sample."""
    n = len(resampled)
    k = max(1, int(round(window_px / max(arc_step, 1e-6))))
    idx = np.arange(n)
    a = resampled[(idx - k) % n]
    b = resampled[idx]
    c = resampled[(idx + k) % n]
    v1 = b - a
    v2 = c - b
    len1 = np.hypot(v1[:, 0], v1[:, 1])
    len2 = np.hypot(v2[:, 0], v2[:, 1])
    dot = (v1 * v2).sum(axis=1)
    cos_ang = np.clip(dot / np.maximum(len1 * len2, 1e-6), -1.0, 1.0)
    ang = np.arccos(cos_ang)
    arc = len1 + len2
    radii = np.where(ang > 1e-4, arc / np.maximum(ang, 1e-6), 1e4)
    radii = np.where((len1 < 1e-6) | (len2 < 1e-6), 1e4, radii)
    return np.clip(radii, 0.0, 1e4)


def radial_bump_count(resampled: np.ndarray) -> int:
    """Count outward protrusions ("bumps"/rays) via local maxima of
    distance-from-centroid over the resampled contour, non-max-suppressed
    against neighbors within ~8% of the contour's own sample count so nearby
    noisy peaks don't get double-counted."""
    c = resampled.mean(axis=0)
    d = np.hypot(resampled[:, 0] - c[0], resampled[:, 1] - c[1])
    n = len(d)
    thresh = np.median(d) * 1.08  # only count protrusions clearly beyond the typical radius
    win = max(2, n // 12)
    peaks = []
    for i in range(n):
        lo, hi = i - win, i + win
        idxs = [j % n for j in range(lo, hi + 1)]
        if d[i] >= thresh and d[i] == max(d[j] for j in idxs):
            peaks.append(i)
    # de-duplicate peaks that are still adjacent after the window check
    deduped = []
    for i in peaks:
        if all(min((i - p) % n, (p - i) % n) > win for p in deduped):
            deduped.append(i)
    return len(deduped)


# ── shape classification (5-family taxonomy) ────────────────────────────────

def classify_and_measure(contour: np.ndarray, page_w: int, page_h: int,
                          min_area: float = MIN_SHAPE_AREA) -> Optional[dict]:
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1e-6)
    (_, (rw, rh), _) = cv2.minAreaRect(contour)
    if min(rw, rh) < 1e-6:
        return None
    aspect = max(rw, rh) / min(rw, rh)

    x, y, cw, ch = cv2.boundingRect(contour)
    is_frame = (cw >= FRAME_WIDTH_FRAC * page_w) or (ch >= FRAME_HEIGHT_FRAC * page_h)

    pts = contour.reshape(-1, 2).astype(np.float64)
    resampled = resample_contour(pts, n_samples=150)
    if resampled is None:
        return None
    perim = cv2.arcLength(contour, True)
    hull_perim = cv2.arcLength(hull, True)
    if perim < 1e-6 or hull_perim < 1e-6:
        return None
    jaggedness = perim / hull_perim
    arc_step = perim / 150
    radii = curvature_radii(resampled, arc_step, window_px=max(6.0, perim * 0.02))
    bumps = radial_bump_count(resampled)

    approx = cv2.approxPolyDP(contour, 0.02 * perim, True)
    n_vertices = len(approx)

    if is_frame:
        # rect (near-4-vertex, low jaggedness) vs angled (4-6 vertices, tilted/trapezoid,
        # still fairly low jaggedness) vs irregular (many vertices / high jaggedness --
        # triangular cuts, multi-panel-merged regions, non-convex outlines)
        if n_vertices <= 4 and jaggedness < 1.05:
            shape_class = "frame_rect"
        elif n_vertices <= 6 and jaggedness < 1.15:
            shape_class = "frame_angled"
        else:
            shape_class = "frame_irregular"
    elif n_vertices <= 6 and solidity > 0.92 and jaggedness < 1.05:
        shape_class = "rectangle"
    elif solidity > 0.88 and jaggedness < 1.15 and bumps <= 1 and aspect < 3.5:
        shape_class = "oval"
    elif solidity > 0.75 and bumps in (1, 2) and np.max(radii) > 1.6 * np.median(radii):
        shape_class = "thorn"
    elif 0.55 <= solidity <= 0.90 and bumps >= 3 and jaggedness < 1.45:
        shape_class = "cloud"
    elif solidity < 0.65 or (bumps >= 6 and jaggedness > 1.4):
        shape_class = "spiky"
    else:
        shape_class = "other"

    return {
        "area": float(area), "aspect": float(aspect), "solidity": float(solidity),
        "jaggedness": float(jaggedness), "bumps": int(bumps), "n_vertices": int(n_vertices),
        "min_radius": float(np.min(radii)), "p10_radius": float(np.percentile(radii, 10)),
        "p50_radius": float(np.percentile(radii, 50)),
        "bbox": (int(x), int(y), int(cw), int(ch)), "is_frame": bool(is_frame),
        "class": shape_class, "contour": contour,
    }


def extract_enclosed_holes(rgb: np.ndarray, min_area: float = MIN_SHAPE_AREA) -> list[dict]:
    """Same flood-fill-from-padded-corner idea as
    src/ml_cleaner.py::repair_frame_interiors -- returns each enclosed hole's
    own contour/stats instead of a repaired delete mask."""
    page_h, page_w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    stroke = (gray <= FRAME_DARKNESS).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stroke = cv2.morphologyEx(stroke, cv2.MORPH_CLOSE, k, iterations=1)
    num_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(stroke, connectivity=8)

    out = []
    for label in range(1, num_labels):
        x, y, cw, ch, comp_area = comp_stats[label]
        if cw * ch < min_area + comp_area:
            continue
        comp = (labels[y:y + ch, x:x + cw] == label).astype(np.uint8)
        padded = np.zeros((ch + 2, cw + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = comp
        ff_mask = np.zeros((ch + 4, cw + 4), dtype=np.uint8)
        cv2.floodFill(padded, ff_mask, (0, 0), 1)
        holes = (padded[1:-1, 1:-1] == 0).astype(np.uint8)
        if not holes.any():
            continue
        hole_contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in hole_contours:
            stats = classify_and_measure(c, page_w, page_h, min_area=min_area)
            if stats:
                stats["contour"] = stats["contour"] + np.array([x, y])
                stats["stroke_bbox"] = (int(x), int(y), int(cw), int(ch))
                # border-thickness estimate for frame-classified strokes: median distance
                # transform value within the stroke component *2 (radius -> width)
                if stats["is_frame"]:
                    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
                    nz = dist[dist > 0]
                    stats["border_thickness"] = float(np.median(nz) * 2) if nz.size else 0.0
                out.append(stats)
    return out


# ── QA preview ────────────────────────────────────────────────────────────────

PREVIEW_COLORS = {
    "oval": (0, 255, 0), "cloud": (255, 200, 0), "spiky": (255, 0, 0),
    "thorn": (255, 0, 255), "rectangle": (0, 128, 255),
    "frame_rect": (0, 200, 200), "frame_angled": (0, 150, 255), "frame_irregular": (150, 0, 255),
    "other": (128, 128, 128),
}


def save_preview(rgb: np.ndarray, shapes: list[dict], path: Path) -> None:
    preview = rgb.copy()
    for s in shapes:
        color = PREVIEW_COLORS.get(s["class"], (128, 128, 128))
        cv2.drawContours(preview, [s["contour"]], -1, color, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(path)
