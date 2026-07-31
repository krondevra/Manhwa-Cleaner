"""Probe: real-vs-training bubble curvature distribution check.

Plan: .claude/plans/snazzy-cuddling-creek.md. Context: the recurring "clauds" defect
(scalloped red intrusions eating into oval speech-bubble content at curved boundaries,
docs/ml_strategy_history.md "model 15.0") has survived every training-mechanism-side fix
tried (models 12.0, 13.0, 15.0, 18.0-refinehead). Two bubble-shape *diversity* hypotheses
were already ruled out this session (docs/ml_strategy_history.md, model 16.0's finding
plus the 2026-07-25 correction to model 18.0): make_bubbles.py's static templates are dead
code, and the real SVG-derived bubble source isn't diversity-starved either (~2,475
distinct hand-drawn shapes across 39 episodes). This probe checks a different,
still-untested hypothesis: not "how many distinct shapes" but "what range of *curvature*
do they cover" -- a distribution-shift question.

No training, no dataset changes, no production code touched. Downloads a handful of P&C
episode lang-pack SVGs (cached under .tmp/bubble_curvature/svgs/), isolates the bubble/SFX
fill layer the same way PepperNCarrotDataset/src/process/process_speechbubbles.py does
(artwork/txt layers dropped -- reimplemented here with stdlib xml.etree.ElementTree
instead of that script's lxml, to avoid adding a new dependency for a one-off analysis),
renders via the system `rsvg-convert` binary (the same fallback path
process_speechbubbles.py::render_svg uses when cairosvg is unavailable), and measures
contour curvature directly. Real manhwa bubble interiors are found the same way
--reclaim-frames/repair_frame_interiors (src/ml_cleaner.py) finds enclosed panel/bubble
interiors: flood-fill a dark-ink stroke component's padded bounding box from its corner;
pixels the fill can't reach are holes fully enclosed by ink.

Curvature is measured in raw pixels at each side's *native* resolution (P&C pages ~2481px
wide, real chapters ~690-720px wide) -- deliberately NOT scale-normalized. A CNN's
convolutional kernels have a fixed absolute-pixel receptive field (established in model
16.0's investigation), so this is the comparison that actually matters for what the model
can represent -- it also naturally folds in the already-known ~3.6x page-scale mismatch
rather than needing separate handling.

Usage:
  .venv/bin/python src/probe_bubble_curvature.py
"""
from __future__ import annotations

import io
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
from ml_cleaner import log  # noqa: E402

OUT = ROOT / ".tmp" / "bubble_curvature"
SVG_DIR = OUT / "svgs"
PREVIEW_DIR = OUT / "previews"

PC_BASE = "https://www.peppercarrot.com"
# Bubble-densest episodes found this session (path counts, model-18.0-era corpus scan)
# plus 2 more mid-density picks for a broader sample -- see the plan for the full list.
EPISODE_SLUGS = [
    "ep36_The-Surprise-Attack",
    "ep39_The-Tavern",
    "ep34_The-Knighting-of-Shichimi",
    "ep33_Spell-of-War",
    "ep31_The-Fight",
]

GT_CHAPTERS = ["001", "002"]
GT_DIR = ROOT / ".tmp" / "saved" / "chapters"

CHAPTER_085 = ROOT / "data" / "chapters-initial" / "085.png"
# notes/clauds_regression_crops.md -- 3 confirmed, historically-documented defect
# bubble instances. No manual GT exists for chapter 085 (checked: no
# .tmp/saved/chapters/085*), so these are measured directly from the raw source art's
# ink outline, same as the P&C/GT-chapter side.
CLAUDS_CROPS = [
    ("clauds_1", 169250, 450),
    ("clauds_2", 179450, 450),
    ("clauds_3", 54550, 500),
]

NS_SVG = "http://www.w3.org/2000/svg"
NS_INK = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", NS_SVG)
ET.register_namespace("inkscape", NS_INK)

FRAME_DARKNESS = 40  # matches --frame-darkness's own default (src/ml_cleaner.py)
MIN_SHAPE_AREA = 400  # px^2, at each side's native resolution
# Real-chapter enclosed-hole extraction (unlike the P&C side, which renders an
# isolated bubble-only SVG layer with no panel lines at all) also picks up
# rounded-corner PANEL frames -- solid, convex, moderate-aspect, easily
# passing the oval filter on shape alone. Checked and rejected a size-based
# cutoff to filter these out: real bubble area and panel area overlap
# directly (confirmed clauds instances measure 79k-110k px^2, well inside the
# range other real panels occupy: 106k-347k px^2 in chapter 001 alone) --
# there is no clean size threshold. Left unfiltered; see the whole-page GT
# aggregate section below, which is flagged unreliable for this reason. The
# CLAUDS_CROPS measurement doesn't need a classifier fix: each crop is a
# small, single-bubble-focused window (per clauds_regression_crops.md) where
# a panel border is unlikely to close into its own contour at all, so
# "largest oval in the crop" already reliably means "the bubble" -- verified
# visually against the crop notes' own bubble-text descriptions.


# ── SVG download + bubble-layer isolation ───────────────────────────────────

def download_episode_svgs(slug: str) -> list[Path]:
    ep_dir = SVG_DIR / slug
    existing = sorted(ep_dir.glob("*.svg")) if ep_dir.exists() else []
    if existing:
        return existing
    ep_dir.mkdir(parents=True, exist_ok=True)
    url = f"{PC_BASE}/0_sources/{slug}/zip/{slug}_lang-pack.zip"
    log(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        en_names = [n for n in z.namelist() if n.startswith("lang/en/") and n.endswith(".svg")]
        for n in en_names:
            (ep_dir / Path(n).name).write_bytes(z.read(n))
    return sorted(ep_dir.glob("*.svg"))


def isolate_bubble_layer(svg_path: Path) -> Optional[tuple[bytes, int, int]]:
    """Drop the artwork/txt layer <g> groups, keep everything else (bubble +
    SFX fills) -- same layer-selection rule as
    process_speechbubbles.py::load_bubble_svg_bytes, reimplemented with
    stdlib ElementTree instead of lxml."""
    tree = ET.parse(svg_path)
    root = tree.getroot()
    try:
        width = int(round(float(root.get("width"))))
        height = int(round(float(root.get("height"))))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    for g in list(root.findall(f"{{{NS_SVG}}}g")):
        label = g.get(f"{{{NS_INK}}}label")
        if label in ("artwork", "txt"):
            root.remove(g)
    buf = io.BytesIO()
    tree.write(buf)
    return buf.getvalue(), width, height


def render_svg_alpha(svg_bytes: bytes, width: int, height: int) -> np.ndarray:
    result = subprocess.run(
        ["rsvg-convert", "-w", str(width), "-h", str(height)],
        input=svg_bytes, capture_output=True, check=True,
    )
    img = Image.open(io.BytesIO(result.stdout)).convert("RGBA")
    return np.array(img)[:, :, 3]


# ── curvature measurement ────────────────────────────────────────────────────

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


def classify_and_measure(contour: np.ndarray, min_area: float = MIN_SHAPE_AREA) -> Optional[dict]:
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

    pts = contour.reshape(-1, 2).astype(np.float64)
    resampled = resample_contour(pts, n_samples=150)
    if resampled is None:
        return None
    perim = cv2.arcLength(contour, True)
    if perim < 1e-6:
        return None
    arc_step = perim / 150
    radii = curvature_radii(resampled, arc_step, window_px=max(6.0, perim * 0.02))

    if solidity > 0.90 and aspect < 3.0:
        shape_class = "oval"
    elif solidity < 0.65:
        shape_class = "spiky"
    else:
        shape_class = "other"

    return {
        "area": float(area), "aspect": float(aspect), "solidity": float(solidity),
        "min_radius": float(np.min(radii)), "p10_radius": float(np.percentile(radii, 10)),
        "p50_radius": float(np.percentile(radii, 50)), "class": shape_class,
        "contour": contour,
    }


def extract_pc_shapes(alpha: np.ndarray) -> list[dict]:
    binmask = (alpha > 10).astype(np.uint8)
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        stats = classify_and_measure(c)
        if stats:
            out.append(stats)
    return out


def extract_enclosed_holes(rgb: np.ndarray, min_area: float = MIN_SHAPE_AREA) -> list[dict]:
    """Same flood-fill-from-padded-corner idea as
    src/ml_cleaner.py::repair_frame_interiors -- adapted to return each
    enclosed hole's own contour/stats instead of a repaired delete mask."""
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
            stats = classify_and_measure(c, min_area=min_area)
            if stats:
                stats["contour"] = stats["contour"] + np.array([x, y])
                stats["bbox"] = (int(x), int(y), int(cw), int(ch))
                out.append(stats)
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def percentile_summary(shapes: list[dict], label: str) -> None:
    ovals = [s for s in shapes if s["class"] == "oval"]
    print(f"  {label}: {len(shapes)} total shapes, {len(ovals)} classified oval")
    if not ovals:
        return
    min_radii = np.array([s["min_radius"] for s in ovals])
    aspects = np.array([s["aspect"] for s in ovals])
    for name, arr in (("min_radius", min_radii), ("aspect", aspects)):
        pcts = np.percentile(arr, [5, 10, 25, 50, 75, 90])
        print(f"    {name} percentiles [5,10,25,50,75,90]: "
              + ", ".join(f"{v:.1f}" for v in pcts))


def rank_against(value: float, reference: np.ndarray) -> float:
    return float(100.0 * np.mean(reference <= value))


def save_preview(rgb: np.ndarray, shapes: list[dict], path: Path) -> None:
    preview = rgb.copy()
    for s in shapes:
        color = {"oval": (0, 255, 0), "spiky": (255, 0, 0), "other": (0, 128, 255)}[s["class"]]
        cv2.drawContours(preview, [s["contour"]], -1, color, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(path)


def main() -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. Pepper & Carrot side ──
    pc_shapes: list[dict] = []
    for slug in EPISODE_SLUGS:
        svgs = download_episode_svgs(slug)
        log(f"{slug}: {len(svgs)} English SVG pages")
        for svg_path in svgs:
            isolated = isolate_bubble_layer(svg_path)
            if isolated is None:
                continue
            svg_bytes, width, height = isolated
            try:
                alpha = render_svg_alpha(svg_bytes, width, height)
            except subprocess.CalledProcessError as exc:
                log(f"  render failed for {svg_path.name}: {exc}")
                continue
            shapes = extract_pc_shapes(alpha)
            pc_shapes.extend(shapes)
            if shapes and svg_path.stem.endswith("P05"):  # one preview per episode
                rgb_preview = np.dstack([255 - alpha, 255 - alpha, 255 - alpha])
                save_preview(rgb_preview.astype(np.uint8), shapes, PREVIEW_DIR / f"pc_{slug}_{svg_path.stem}.png")

    print(f"\n=== Pepper & Carrot bubble/SFX shapes ({len(pc_shapes)} total) ===")
    percentile_summary(pc_shapes, "P&C corpus")
    pc_oval_min_radii = np.array([s["min_radius"] for s in pc_shapes if s["class"] == "oval"])

    # ── 2. real GT chapters ──
    # NOTE: whole-page enclosed-hole extraction also catches rounded-corner
    # PANEL frames (dark strokes enclosing a large light interior, same as a
    # bubble geometrically) -- confirmed by inspecting the largest "oval"
    # candidates by hand (see script comment above MIN_SHAPE_AREA). No clean
    # size/shape cutoff separates them from real bubbles (areas overlap
    # directly). So the whole-page aggregate below is NOT a reliable
    # bubble-only measurement and is reported for descriptive context only --
    # it is not used for the percentile ranking below, which instead uses the
    # clean P&C corpus distribution and the 3 individually-verified clauds
    # instances.
    print("\n=== Real manhwa GT chapters (001, 002) -- descriptive only, panel-contaminated, see note ===")
    for ch in GT_CHAPTERS:
        rgb = np.asarray(Image.open(GT_DIR / f"{ch}.png").convert("RGB"))
        shapes = extract_enclosed_holes(rgb)
        percentile_summary(shapes, f"chapter {ch}")

    # ── 3. the 3 confirmed clauds defect instances ──
    print("\n=== Confirmed 'clauds' defect instances (data/chapters-initial/085.png) ===")
    page = np.asarray(Image.open(CHAPTER_085).convert("RGB"))
    for name, y, h in CLAUDS_CROPS:
        y0, y1 = max(0, y - 40), min(page.shape[0], y + h + 40)
        crop = page[y0:y1]
        shapes = extract_enclosed_holes(crop)
        ovals = sorted([s for s in shapes if s["class"] == "oval"], key=lambda s: -s["area"])
        if not ovals:
            print(f"  {name}: no oval-classified enclosed shape found in crop")
            continue
        target = ovals[0]  # largest enclosed oval in the crop == the bubble itself
        pct = rank_against(target["min_radius"], pc_oval_min_radii)
        print(f"  {name}: min_radius={target['min_radius']:.1f}px, aspect={target['aspect']:.2f} "
              f"-> {pct:.1f}th percentile of P&C oval min_radius distribution "
              f"({'TIGHTER than most P&C training bubbles' if pct < 25 else 'within normal P&C range' if pct > 25 else 'borderline'})")
        save_preview(crop, [target], PREVIEW_DIR / f"{name}_measured.png")

    print(f"\nDone in {time.time() - t0:.1f}s. Previews under {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
