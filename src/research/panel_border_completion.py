"""Plan v8 Phase 1 (.claude/plans/snazzy-cuddling-creek.md): geometric boundary-tracing +
completion for dark-on-dark panel borders. Classical CV, no model, no training -- a
`postprocess(rgb, mask) -> mask` step following the same shape as `protect_frame_borders` /
`repair_frame_interiors` (src/ml_cleaner.py) and `reclaim_black_backdrop.py`.

Motivated by Phase 0's signal probe (`.tmp/diagnostics/border_signal_probe.py`), which confirmed
on 6 real instances that dark panel borders carry PARTIAL signal -- strong along some stretches,
weak-but-present along others -- rather than being uniformly undetectable, and that the strong
stretches are blindly Hough-detectable without ground truth.

Pipeline:
  1. Candidate segments: Canny edges (thresholds derived from a measured noise floor) restricted
     to dark-adjacent pixels (one side near-black), fed to HoughLinesP (any orientation).
  2. Collinear grouping: segments clustered by (angle, perpendicular offset) into groups
     representing a single real boundary line/curve.
  3. Interpolation-only completion (`_fill_gaps`, unit-tested in isolation): within a group's
     merged anchor runs (sorted along the line), a gap is completed ONLY when it sits strictly
     between two real anchor runs (both >= min_anchor_len_px) and is no longer than
     max_gap_px. Never extrapolates past the outermost anchor; a group with 0 or 1 anchor runs
     produces zero completions.
  4. Panel-side decision + bounded reclaim: for a completed chain spanning >= min_span_frac of
     the relevant page dimension, sample the CURRENT delete_mask on both sides to identify which
     side is "panel" (low delete fraction) vs "gutter" (near-solid delete fraction); skip
     entirely if the two sides aren't clearly separated (safety default: do nothing). On the
     panel side, walk perpendicular to the boundary column-by-column; reclaim near-black delete
     pixels back to keep ONLY up to the point where a genuine already-kept run is found (i.e.
     enclosure is actually established for that column) -- a column that never finds a kept run
     within reclaim_depth_px is left untouched rather than blindly reclaimed to the depth cap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

FRAME_DARKNESS = 40  # project-standard near-black threshold (reclaim_black_backdrop.py,
                      # repair_frame_interiors)


# ---------------------------------------------------------------------------
# Core safety-critical logic: gap-filling between anchor runs. Pure, testable
# in isolation from any image/CV code -- see _self_test() at the bottom.
# ---------------------------------------------------------------------------

@dataclass
class AnchorRun:
    t_min: float
    t_max: float
    pt_at_min: tuple[float, float]   # 2D image point achieving t_min
    pt_at_max: tuple[float, float]   # 2D image point achieving t_max


@dataclass
class Completion:
    kind: str  # "interpolated"
    p0: tuple[float, float]
    p1: tuple[float, float]
    gap_len: float


def _fill_gaps(anchor_runs: list[AnchorRun], max_gap_px: float) -> list[Completion]:
    """Given anchor runs already sorted by t_min (position along the shared line), return
    interpolated straight-line completions for every gap that sits strictly BETWEEN two anchor
    runs and is no longer than max_gap_px. Never extrapolates past the first/last anchor's own
    extent -- a list of 0 or 1 anchor runs always returns []."""
    completions: list[Completion] = []
    for i in range(len(anchor_runs) - 1):
        a, b = anchor_runs[i], anchor_runs[i + 1]
        gap = b.t_min - a.t_max
        if gap <= 0:
            continue  # overlapping/touching, nothing to fill
        if gap > max_gap_px:
            continue  # too big to trust a straight-line guess -- leave untouched
        completions.append(Completion(kind="interpolated", p0=a.pt_at_max, p1=b.pt_at_min,
                                       gap_len=float(gap)))
    return completions


# ---------------------------------------------------------------------------
# Segment detection + collinear grouping
# ---------------------------------------------------------------------------

def _compute_noise_floor(gray: np.ndarray, delete_mask: np.ndarray) -> float:
    grad = _gradient_magnitude(gray)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
    deep_delete = cv2.erode(delete_mask.astype(np.uint8), k, iterations=1) > 0
    if not deep_delete.any():
        return float(np.percentile(grad, 99))
    return float(np.percentile(grad[deep_delete], 99))


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _detect_segments(rgb: np.ndarray, delete_mask: np.ndarray, darkness_threshold: int,
                      floor: float, min_anchor_len_px: int) -> np.ndarray:
    """Returns an (N,4) array of Hough line segments [x1,y1,x2,y2] restricted to dark-adjacent
    edges. Any orientation -- real gutters in this corpus can be diagonal."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    near_black = (gray <= darkness_threshold).astype(np.uint8)
    dark_adjacent = cv2.dilate(near_black, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    edges = cv2.Canny(gray, max(floor, 1.0), max(floor, 1.0) * 2.0)
    edges = cv2.bitwise_and(edges, edges, mask=dark_adjacent)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                             minLineLength=min_anchor_len_px, maxLineGap=10)
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def _circular_angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _group_collinear(segments: np.ndarray, max_angle_deg: float,
                      max_perp_dist_px: float) -> list[np.ndarray]:
    """Greedy clustering of Hough segments into groups sharing (angle mod 180, perpendicular
    offset from origin) within tolerance -- i.e. segments plausibly on the same real line."""
    groups: list[dict] = []  # each: {"angle": deg, "rho": float, "members": [seg_idx,...]}
    for idx, (x1, y1, x2, y2) in enumerate(segments):
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        # normal direction consistent sign: rotate direction by 90deg, normalize
        nx, ny = -dy / length, dx / length
        if nx < 0 or (nx == 0 and ny < 0):
            nx, ny = -nx, -ny
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rho = nx * mx + ny * my

        placed = False
        for g in groups:
            if _circular_angle_diff(angle, g["angle"]) <= max_angle_deg and \
               abs(rho - g["rho"]) <= max_perp_dist_px:
                g["members"].append(idx)
                n = len(g["members"])
                g["angle"] = (g["angle"] * (n - 1) + angle) / n
                g["rho"] = (g["rho"] * (n - 1) + rho) / n
                placed = True
                break
        if not placed:
            groups.append({"angle": angle, "rho": rho, "members": [idx]})

    return [segments[g["members"]] for g in groups if len(g["members"]) >= 1]


def _build_anchor_runs(group_segs: np.ndarray, min_anchor_len_px: int,
                        merge_tol_px: float = 5.0) -> list[AnchorRun]:
    """Project a collinear group's segment endpoints onto the group's averaged direction and
    merge overlapping/near-touching segments into anchor runs (a single real stroke can produce
    several short Hough sub-segments)."""
    pts = group_segs.reshape(-1, 2, 2)  # (N, endpoint_idx, xy)
    all_pts = pts.reshape(-1, 2)
    p0 = all_pts.mean(axis=0)

    # averaged direction via unit vectors, sign-normalized so cos>=0
    dirs = pts[:, 1] - pts[:, 0]
    lens = np.linalg.norm(dirs, axis=1, keepdims=True)
    lens[lens < 1e-6] = 1.0
    unit_dirs = dirs / lens
    unit_dirs[unit_dirs[:, 0] < 0] *= -1
    d = unit_dirs.mean(axis=0)
    d = d / (np.linalg.norm(d) + 1e-9)

    intervals = []  # (t_min, t_max, pt_at_min, pt_at_max, seg_len)
    for (p1, p2) in pts:
        seg_len = float(np.linalg.norm(p2 - p1))
        if seg_len < min_anchor_len_px:
            continue
        t1 = float(np.dot(p1 - p0, d))
        t2 = float(np.dot(p2 - p0, d))
        if t1 <= t2:
            intervals.append([t1, t2, tuple(p1), tuple(p2), seg_len])
        else:
            intervals.append([t2, t1, tuple(p2), tuple(p1), seg_len])

    if not intervals:
        return []

    intervals.sort(key=lambda iv: iv[0])
    merged = [intervals[0]]
    for iv in intervals[1:]:
        last = merged[-1]
        if iv[0] <= last[1] + merge_tol_px:
            if iv[1] > last[1]:
                last[1] = iv[1]
                last[3] = iv[3]
        else:
            merged.append(iv)

    return [AnchorRun(t_min=m[0], t_max=m[1], pt_at_min=m[2], pt_at_max=m[3]) for m in merged]


# ---------------------------------------------------------------------------
# Panel-side decision + bounded reclaim
# ---------------------------------------------------------------------------

def _chain_points(anchor_runs: list[AnchorRun], completions: list[Completion],
                   step_px: float = 3.0) -> list[tuple[float, float]]:
    """Densely-sampled points along the full completed chain (real anchors + interpolated
    gaps), in order."""
    segs = []
    for r in anchor_runs:
        segs.append((r.pt_at_min, r.pt_at_max))
    for c in completions:
        segs.append((c.p0, c.p1))
    # order segments by their first point's projection isn't guaranteed here since anchors and
    # completions are separate lists; caller passes them already interleaved in position order.
    points = []
    for p0, p1 in segs:
        p0 = np.array(p0, dtype=np.float64)
        p1 = np.array(p1, dtype=np.float64)
        length = float(np.linalg.norm(p1 - p0))
        n = max(2, int(length / step_px))
        for i in range(n):
            t = i / (n - 1)
            points.append(tuple(p0 + t * (p1 - p0)))
    return points


def _chain_span_and_orientation(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Returns (span_px, dx, dy) where (dx,dy) is the unit tangent direction from first to last
    point (approximation of overall chain orientation)."""
    p_first = np.array(points[0])
    p_last = np.array(points[-1])
    span = float(np.linalg.norm(p_last - p_first))
    d = p_last - p_first
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return span, 1.0, 0.0
    return span, float(d[0] / norm), float(d[1] / norm)


def _reclaim_side(rgb: np.ndarray, delete_mask: np.ndarray, points: list[tuple[float, float]],
                   nx: float, ny: float, darkness_threshold: int, reclaim_depth_px: int,
                   kept_run_px: int, sign: float) -> tuple[np.ndarray, int]:
    """Walks perpendicular to the chain (direction sign*(nx,ny)) from each sampled point,
    reclaiming near-black delete pixels back to keep ONLY up to a genuinely-found run of
    already-kept pixels (enclosure established for that column). A column that never finds such
    a run within reclaim_depth_px is left untouched entirely -- no blind reclaim to the cap."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    fixed = delete_mask.copy()
    n_reclaimed = 0
    for (px, py) in points:
        stop_d = None
        run = 0
        for d in range(1, reclaim_depth_px + 1):
            x = int(round(px + sign * nx * d))
            y = int(round(py + sign * ny * d))
            if not (0 <= x < w and 0 <= y < h):
                break
            if not fixed[y, x]:  # already kept
                run += 1
                if run >= kept_run_px:
                    stop_d = d - kept_run_px + 1
                    break
            else:
                run = 0
        if stop_d is None:
            continue  # enclosure not established for this column -- do nothing (safety default)
        for d in range(1, stop_d + 1):
            x = int(round(px + sign * nx * d))
            y = int(round(py + sign * ny * d))
            if not (0 <= x < w and 0 <= y < h):
                continue
            if fixed[y, x] and gray[y, x] <= darkness_threshold:
                fixed[y, x] = False
                n_reclaimed += 1
    return fixed, n_reclaimed


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class CompletionReport:
    chains: list[dict] = field(default_factory=list)
    n_groups_detected: int = 0
    n_completions: int = 0
    n_pixels_reclaimed: int = 0


def complete_panel_borders(
    rgb: np.ndarray,
    delete_mask: np.ndarray,
    darkness_threshold: int = FRAME_DARKNESS,
    floor: float | None = None,
    max_angle_deg: float = 20.0,
    max_perp_dist_px: float = 8.0,
    max_gap_px: float = 400.0,
    min_anchor_len_px: int = 40,
    min_span_frac: float = 0.6,
    reclaim_depth_px: int = 250,
    kept_run_px: int = 15,
    solid_gutter_thresh: float = 0.85,
    panel_side_thresh: float = 0.5,
    probe_dist_px: int = 15,
) -> tuple[np.ndarray, CompletionReport]:
    """Postprocess(rgb, mask) -> mask, following the CHAINS-composable shape used throughout
    this project (eval_gen6_checkpoint.py's crf/sfx wrapper precedent)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if floor is None:
        floor = _compute_noise_floor(gray, delete_mask)

    segments = _detect_segments(rgb, delete_mask, darkness_threshold, floor, min_anchor_len_px)
    report = CompletionReport()
    if len(segments) == 0:
        return delete_mask.copy(), report

    groups = _group_collinear(segments, max_angle_deg, max_perp_dist_px)
    report.n_groups_detected = len(groups)

    fixed = delete_mask.copy()
    for group_segs in groups:
        anchor_runs = _build_anchor_runs(group_segs, min_anchor_len_px)
        if len(anchor_runs) < 2:
            continue  # 0 or 1 anchor runs -> nothing to fill (case b/c)
        completions = _fill_gaps(anchor_runs, max_gap_px)
        if not completions:
            continue

        points = _chain_points(anchor_runs, completions)
        span, dx, dy = _chain_span_and_orientation(points)
        # relevant dimension: mostly-horizontal chain checked against width, mostly-vertical
        # checked against height
        relevant_dim = w if abs(dx) >= abs(dy) else h
        if span < min_span_frac * relevant_dim:
            continue  # too short to trust as a real panel boundary

        # perpendicular (normal) direction
        nx, ny = -dy, dx

        def side_delete_frac(sign: float) -> float:
            vals = []
            for (px, py) in points[::4]:
                x = int(round(px + sign * nx * probe_dist_px))
                y = int(round(py + sign * ny * probe_dist_px))
                if 0 <= x < w and 0 <= y < h:
                    vals.append(1.0 if fixed[y, x] else 0.0)
            return float(np.mean(vals)) if vals else 0.5

        frac_pos = side_delete_frac(1.0)
        frac_neg = side_delete_frac(-1.0)
        if frac_pos <= frac_neg:
            panel_sign, panel_frac, gutter_frac = 1.0, frac_pos, frac_neg
        else:
            panel_sign, panel_frac, gutter_frac = -1.0, frac_neg, frac_pos

        if not (gutter_frac >= solid_gutter_thresh and panel_frac <= panel_side_thresh):
            continue  # sides not clearly separated -- ambiguous, do nothing (safety default)

        fixed, n_reclaimed = _reclaim_side(rgb, fixed, points, nx, ny, darkness_threshold,
                                            reclaim_depth_px, kept_run_px, panel_sign)

        report.n_completions += len(completions)
        report.n_pixels_reclaimed += n_reclaimed
        report.chains.append({
            "n_anchor_runs": len(anchor_runs),
            "n_completions": len(completions),
            "gap_lengths": [c.gap_len for c in completions],
            "span_px": span,
            "panel_side_delete_frac": panel_frac,
            "gutter_side_delete_frac": gutter_frac,
            "pixels_reclaimed": n_reclaimed,
            "chain_bbox": (
                min(p[0] for p in points), min(p[1] for p in points),
                max(p[0] for p in points), max(p[1] for p in points),
            ),
        })

    return fixed, report


# ---------------------------------------------------------------------------
# Self-tests for the safety-critical gap-filling logic (Plan v8 Phase 1 explicit requirement:
# "must be verified as an actual implemented constraint, not just described").
# ---------------------------------------------------------------------------

def _self_test() -> None:
    # (a) two anchors + gap -> completed
    runs = [
        AnchorRun(t_min=0, t_max=100, pt_at_min=(0, 0), pt_at_max=(100, 0)),
        AnchorRun(t_min=150, t_max=250, pt_at_min=(150, 0), pt_at_max=(250, 0)),
    ]
    out = _fill_gaps(runs, max_gap_px=100)
    assert len(out) == 1, f"expected 1 completion, got {len(out)}"
    assert out[0].p0 == (100, 0) and out[0].p1 == (150, 0)
    assert out[0].gap_len == 50
    print("PASS: two anchors + gap within max_gap_px -> completed")

    # gap too large -> not completed
    out2 = _fill_gaps(runs, max_gap_px=40)
    assert len(out2) == 0, f"expected 0 completions (gap too large), got {len(out2)}"
    print("PASS: gap larger than max_gap_px -> NOT completed")

    # (b) one-sided anchor -> NOT completed (single run, nothing to pair with)
    single = [AnchorRun(t_min=0, t_max=100, pt_at_min=(0, 0), pt_at_max=(100, 0))]
    out3 = _fill_gaps(single, max_gap_px=1000)
    assert len(out3) == 0, f"expected 0 completions (single anchor), got {len(out3)}"
    print("PASS: one-sided anchor (single run) -> NOT completed")

    # (c) zero anchors -> untouched
    out4 = _fill_gaps([], max_gap_px=1000)
    assert len(out4) == 0, f"expected 0 completions (zero anchors), got {len(out4)}"
    print("PASS: zero anchors -> untouched")

    # extrapolation-past-outermost-anchor guard: 3 runs, only interior gaps ever fillable
    three = [
        AnchorRun(t_min=0, t_max=50, pt_at_min=(0, 0), pt_at_max=(50, 0)),
        AnchorRun(t_min=60, t_max=110, pt_at_min=(60, 0), pt_at_max=(110, 0)),
        AnchorRun(t_min=120, t_max=170, pt_at_min=(120, 0), pt_at_max=(170, 0)),
    ]
    out5 = _fill_gaps(three, max_gap_px=1000)
    assert len(out5) == 2, f"expected 2 interior gaps filled, got {len(out5)}"
    # no completion point ever lies before t=0 or after t=170 (the outermost anchor extents)
    for c in out5:
        assert 0 <= c.p0[0] <= 170 and 0 <= c.p1[0] <= 170
    print("PASS: only interior gaps filled, never extrapolating past outermost anchors")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
