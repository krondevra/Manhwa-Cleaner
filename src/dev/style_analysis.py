"""Reusable style-analysis primitives (Part 4, notes/synthetic_curriculum_plan.md).

Lifted and extended from src/probe_bubble_curvature.py's one-off inline logic (the
flood-fill-from-corner enclosed-hole detector, shared with src/ml_cleaner.py's
repair_frame_interiors, and the curvature/contour measurement code), plus a new frame/
bubble split and a 5-family shape taxonomy (oval, cloud/thought, spiky/"sea urchin", thorn,
rectangle -- standard in published comic/balloon-segmentation work, see
notes/inspiration_papers_review.md's Dubray & Laubrock reference) needed for the
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

# Text-plausibility check for bubble-family interiors (extract_enclosed_holes) -- see the
# fix note there, 2026-07-26, for why a blanket interior-variance cutoff isn't enough.
TEXT_INK_THRESHOLD = 140  # gray value below which an interior pixel counts as "ink"
MIN_INK_FRAC = 0.003  # some ink expected -- a perfectly flat highlight/decal has none
MAX_INK_FRAC = 0.40  # above this it's a solid dark fill, not sparse text
MAX_INTERIOR_SATURATION = 40  # 0-255 HSV S; real bubble/text-box fills are ~grayscale

# Skip stroke components whose bounding box covers most of the page -- see the fix note
# in extract_enclosed_holes for why (dark scene backgrounds, not ink strokes, produce
# these). Legitimate panel-divider components measured during calibration stayed well
# under this (typically <5% of page area even on giant multi-panel strips).
MAX_STROKE_BBOX_FRAC = 0.60

# An enclosed component is treated as a PANEL FRAME, not a bubble/text candidate, if its
# bounding box covers a large fraction of the page WIDTH -- real panel dividers in this
# corpus (vertical-scroll webtoons) are consistently near-full-width horizontal bars,
# empirically checked across ~60 real frame instances spanning page heights from 1.4k to
# 32k px (wfrac clustered 57%-99.6%, none below 57%). A HEIGHT-based fraction was tried
# first and DROPPED, 2026-07-26 (user-caught bug): it's page-relative, so on short pages
# (common -- page height 5th pctile is 1264px per style_analysis_findings.md) a perfectly
# normal 300-370px-tall bubble already exceeds a 20%-of-height threshold and gets
# misrouted to the frame taxonomy. Confirmed on real examples: 4 separate genuine bubbles
# across 2 flagged pages were ALL misrouted by the height criterion alone (wfrac 15.8%-
# 47.1%, well under the width threshold below) while zero real frame instances in the
# calibration sample were ever triggered by height rather than width. Width-only is not
# just a patch -- it's what the calibration data actually supports for this corpus's
# layout convention (panels stack vertically, dividers are horizontal bars); heuristic,
# not exact -- the QA preview step (save_preview) exists specifically so this split can be
# sanity-checked visually before trusting any aggregate built on top of it.
FRAME_WIDTH_FRAC = 0.55


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


def find_bump_peaks(resampled: np.ndarray) -> list[int]:
    """Indices into `resampled` of local maxima of distance-from-centroid,
    non-max-suppressed against neighbors within ~8% of the contour's own sample
    count so nearby noisy peaks don't get double-counted. Shared peak-finding
    logic behind radial_bump_count and measure_tail (2026-07-28)."""
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
    return deduped


def radial_bump_count(resampled: np.ndarray) -> int:
    """Count outward protrusions ("bumps"/rays) via local maxima of
    distance-from-centroid over the resampled contour, non-max-suppressed
    against neighbors within ~8% of the contour's own sample count so nearby
    noisy peaks don't get double-counted."""
    return len(find_bump_peaks(resampled))


def measure_tail(resampled: np.ndarray, radii: np.ndarray) -> Optional[dict]:
    """For a shape with exactly one dominant bump (the tail-candidate signature
    already used by the 'thorn' classification: bumps in (1,2) with one radius
    clearly dominant), measure the tail's length and tip sharpness -- the two axes
    explicitly requested for the new bubble generator's tail variety (2026-07-28):
    short vs. long (tail_length_ratio: tip distance from centroid, normalized by
    the body's own typical radius so it's comparable across bubble sizes) and
    pointed vs. rounded tip (tip_curvature_radius_norm: the SAME discrete
    radius-of-curvature already computed for classification, evaluated at the
    peak sample -- a sharp point turns direction quickly over a short arc, which
    curvature_radii's arc/angle formula reports as a SMALL radius; a rounded tip
    turns more gradually, reporting a LARGER radius. No new geometry primitive
    needed -- both measurements reuse resample_contour/curvature_radii's existing
    output, just evaluated at the bump peak instead of aggregated across the
    whole contour.

    find_bump_peaks' centroid (the plain mean of resampled points, shared with
    the already-calibrated classification logic -- deliberately not changed
    here) shifts noticeably toward a long tail, which can make the body's own
    far side register as a second, spurious "peak" simply because it's now
    farther from the shifted mean, not because it's a real second protrusion.
    Tested directly against synthetic shapes with a known single tail
    (2026-07-28): the spurious peak's raw DISTANCE can be nearly identical to
    the real tail tip's (confirmed one case at a 1.02x ratio -- a distance-ratio
    filter would have rejected it), but its local CURVATURE never is -- the
    spurious peak sits on the body's ordinary smooth boundary (measured
    curvature radius ~150px in testing) while a genuine tail tip is a real
    directional discontinuity (~7-9px in the same tests), independent of how
    close the two points' distances happen to be. Real thorn shapes are already
    classified allowing exactly this `bumps in (1, 2)` case, so when there are
    exactly 2 peaks, treat the one with clearly sharper curvature (<0.5x the
    other's radius) as the tail; if neither is clearly sharper, the shape
    genuinely has two comparable protrusions and isn't safely reducible to one
    tail measurement, so still return None rather than guessing."""
    peaks = find_bump_peaks(resampled)
    if len(peaks) == 1:
        peak_idx = peaks[0]
    elif len(peaks) == 2:
        r_peaks = [float(radii[p]) for p in peaks]
        lo, hi = sorted(r_peaks)
        if hi < 1e-6 or lo / hi > 0.5:
            return None
        peak_idx = peaks[r_peaks.index(lo)]
    else:
        return None
    c = resampled.mean(axis=0)
    d = np.hypot(resampled[:, 0] - c[0], resampled[:, 1] - c[1])
    n = len(d)
    win = max(2, n // 12)
    body_idx = [i for i in range(n) if min((i - peak_idx) % n, (peak_idx - i) % n) > win]
    if not body_idx:
        return None
    body_radius = float(np.median(d[body_idx]))
    if body_radius < 1e-6:
        return None
    return {
        "tail_length_ratio": float(d[peak_idx] / body_radius),
        "tip_curvature_radius_norm": float(radii[peak_idx] / body_radius),
    }


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
    is_frame = cw >= FRAME_WIDTH_FRAC * page_w

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

    tail = measure_tail(resampled, radii) if not is_frame and shape_class in ("thorn", "oval", "cloud") else None

    return {
        "area": float(area), "aspect": float(aspect), "solidity": float(solidity),
        "jaggedness": float(jaggedness), "bumps": int(bumps), "n_vertices": int(n_vertices),
        "min_radius": float(np.min(radii)), "p10_radius": float(np.percentile(radii, 10)),
        "p50_radius": float(np.percentile(radii, 50)),
        "tail_length_ratio": tail["tail_length_ratio"] if tail else None,
        "tip_curvature_radius_norm": tail["tip_curvature_radius_norm"] if tail else None,
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

    page_area = page_w * page_h
    out = []
    for label in range(1, num_labels):
        x, y, cw, ch, comp_area = comp_stats[label]
        if cw * ch < min_area + comp_area:
            continue
        if cw * ch >= MAX_STROKE_BBOX_FRAC * page_area:
            # A dark SCENE background (night scenes, shadow, etc.) falls under the same
            # gray<=FRAME_DARKNESS threshold as an ink stroke and can connect into one
            # enormous component spanning most of the page -- confirmed, 2026-07-26
            # (user-flagged QA finding): on such a page, the corner-flood-fill's "hole"
            # detection stops being reliable, because light regions (a character's face,
            # a bubble sitting inside the dark scene) can leak through other unrelated
            # light regions elsewhere in the same giant bounding box to reach the corner,
            # instead of being cleanly isolated as their own hole -- this is why two real,
            # clearly-drawn jagged speech bubbles on a dark night-scene page were silently
            # missed entirely (not misclassified -- never even produced as a candidate
            # hole), while nearby unrelated light slivers on the character's face/clothing
            # showed up as spurious partial "holes" instead. Skipping components this
            # large is an honest, bounded limitation, not a fix for the general case: real
            # per-bubble detection on dark-background pages needs a different algorithm
            # (test each light blob for local enclosure directly, not a global corner
            # flood-fill over a page-spanning dark component) -- out of scope here. This
            # cap only excludes giant pathological components; legitimate large panel
            # dividers/frames measured during calibration stayed well under it.
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
        region = rgb[y:y + ch, x:x + cw]
        for c in hole_contours:
            stats = classify_and_measure(c, page_w, page_h, min_area=min_area)
            if not stats:
                continue
            if stats["is_frame"]:
                dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
                nz = dist[dist > 0]
                stats["border_thickness"] = float(np.median(nz) * 2) if nz.size else 0.0
            else:
                # Real speech-bubble/text-box interiors are a flat, light fill with dark
                # TEXT ink on top -- not uniformly flat, and not continuously shaded like
                # real artwork. Confirmed visually (user-flagged QA findings, 2026-07-26):
                # unfiltered, small enclosed ink loops elsewhere in the artwork (jewelry,
                # clothing seams, decorative patterns, glyph counters in SFX/logo text)
                # get swept in as false positives; a first fix (blanket
                # interior_std<=35 over the WHOLE interior) stopped that but
                # over-corrected -- it also rejected real bubbles with enough text ink to
                # push interior variance up (confirmed: a clean, correctly-shaped
                # "ПРАВДА?" oval bubble, solidity=0.995, was rejected at
                # interior_std=50.5 despite interior_mean=242.6, purely from its own
                # text). Replaced with an explicit text-plausibility check: split the
                # interior into "ink" (dark) vs "background" pixels, and require the
                # background ALONE -- not the whole interior -- to be a flat light fill,
                # plus a plausible (not zero, not solid-fill) ink coverage fraction. A low
                # mean-saturation check catches colorful decorative fills a
                # brightness-only test could still miss.
                mask_local = np.zeros((ch, cw), dtype=np.uint8)
                cv2.drawContours(mask_local, [c], -1, 1, thickness=-1)
                mask_bool = mask_local.astype(bool)
                interior = region[mask_bool]
                if interior.size == 0:
                    continue
                interior_gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)[mask_bool]
                ink = interior_gray < TEXT_INK_THRESHOLD
                ink_frac = float(np.mean(ink))
                bg_pixels = interior[~ink]
                if bg_pixels.size == 0:
                    continue
                bg_mean = float(bg_pixels.mean())
                bg_std = float(bg_pixels.std())
                mean_sat = float(cv2.cvtColor(region, cv2.COLOR_RGB2HSV)[mask_bool][:, 1].mean())
                stats["interior_mean"] = bg_mean
                stats["interior_std"] = bg_std
                stats["ink_frac"] = ink_frac
                stats["mean_saturation"] = mean_sat
                if not (MIN_INK_FRAC <= ink_frac <= MAX_INK_FRAC):
                    continue  # no plausible text ink, or a solid dark fill -- not a bubble/text box
                if bg_mean < 170.0 or bg_std > 20.0:
                    continue  # non-ink background isn't a flat light fill -> artwork detail
                if mean_sat > MAX_INTERIOR_SATURATION:
                    continue  # colorful -- real bubble/text-box fills are ~grayscale
            stats["contour"] = stats["contour"] + np.array([x, y])
            stats["stroke_bbox"] = (int(x), int(y), int(cw), int(ch))
            out.append(stats)
    return out


# ── QA preview ────────────────────────────────────────────────────────────────

# Bubble-family: saturated primary/secondary hues. Frame-family: desaturated brown/gray
# tones, deliberately far from every bubble color -- frame_irregular's old color
# (150,0,255) was close enough to thorn's (255,0,255) to fool a visual QA spot-check at a
# glance (2026-07-26, this is how a frame-misrouting bug got misread as a correctly
# classified bubble). Frame and bubble classes must never be confusable by hue alone.
PREVIEW_COLORS = {
    "oval": (0, 255, 0), "cloud": (255, 200, 0), "spiky": (255, 0, 0),
    "thorn": (255, 0, 255), "rectangle": (0, 128, 255),
    "frame_rect": (140, 140, 140), "frame_angled": (180, 140, 80), "frame_irregular": (100, 60, 20),
    "other": (0, 0, 0),
}


def save_preview(rgb: np.ndarray, shapes: list[dict], path: Path) -> None:
    preview = rgb.copy()
    for s in shapes:
        color = PREVIEW_COLORS.get(s["class"], (128, 128, 128))
        cv2.drawContours(preview, [s["contour"]], -1, color, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(path)
