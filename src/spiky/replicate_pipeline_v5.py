"""Pipeline replication v5 (plan v15, 2026-08-07) = v4 + two additions. v4 untouched.

1. TRACK 2 — soft-gradient barrier (classical): the hard-mask split barrier now also includes
   the LOW-FREQUENCY gradient (|grad(Gaussian(gray, sigma=8))| > 1.0), cutting the soft
   gradient bridges (title-transition fades, glow rims) that Canny(60,120) cannot see and
   that kept page-scale merged components below the seed rule's whiteness bar. Measured
   (gold): 001-1 under-white 8.33% -> 1.14%, 001-2 6.62% -> 0.50%; cost +0.85pp over-del on
   001-1 (guard letter exceeded, accepted with the 8.5:1 exchange flagged -- the new over-del
   is visually confirmed to be the KNOWN white-inside-panel ambiguity class, not new damage).

2. TRACK 1 — micro band classifier (learned, synthetic-only): OFF BY DEFAULT after an honest
   negative (2026-08-07). BandNet (24,691 params, stable training, val balanced-acc 0.874,
   interior-recall 0.957 / gutter-recall 0.792 at tau=0.5) has NO viable operating point for
   this integration: protecting a misclassified gutter band costs its entire area, so the
   hook needs gutter-recall >= 0.995 -- and at that recall the val interior-recall collapses
   to 0.00-0.11 (operating-point sweep on synthetic val bands). Battery with the hook ON:
   fixed 2 of the 5 target pages but collapsed 7 previously-passing synthetic pages to
   15-33% under-deletion. The hook remains available via an explicit --ckpt for future
   experiments (e.g. more data / richer context); the class stays OPEN.

Usage:
  .venv/bin/python replicate_pipeline_v5.py <src.png> <out_mask.npy> [--ckpt CKPT]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import ml_cleaner
sys.modules["__main__"].train_command = ml_cleaner.train_command
from ml_cleaner import repair_frame_interiors

from replicate_pipeline_v2 import build_mask, SETTINGS
from leak_detector import detect_leaks
from band_classifier import find_bands, load_band_net, classify_bands

T_FRAC250 = 0.90
T3_EDGE = 0.01
OVERLAP_PX = 5000
OVERLAP_FRAC = 0.2
EXTENT_WHITE = 250
CANNY_LO, CANNY_HI = 60, 120
SOFT_GRAD_SIGMA = 8.0
SOFT_GRAD_T = 1.0
DEFAULT_CKPT = None  # Track 1 hook off by default (honest negative, see module docstring)


def clean_page(rgb_u8: np.ndarray, ckpt: Path | None = DEFAULT_CKPT) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32)
    gray_light = (rgb.max(axis=2) + rgb.min(axis=2)) / 2.0
    gray = np.round(gray_light).astype(np.uint8)
    W = gray.shape[1]

    sob = np.sqrt(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3) ** 2
                   + cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3) ** 2)
    edge = sob > 30
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    hard = build_mask(gray_light, SETTINGS["hard-white"])
    soft = build_mask(gray_light, SETTINGS["soft-white"])
    barrier = cv2.morphologyEx(cv2.Canny(gray, CANNY_LO, CANNY_HI), cv2.MORPH_CLOSE, k3) > 0
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), SOFT_GRAD_SIGMA)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, 3)
    softgrad = np.sqrt(gx * gx + gy * gy) > SOFT_GRAD_T
    soft_m = soft & ~barrier
    hard_cc = hard & ~(barrier | softgrad)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(hard_cc.astype(np.uint8), 4)
    seed = np.zeros(gray.shape, dtype=bool)
    for lbl in range(1, num):
        x, y, cw, chh, area = stats[lbl]
        if area < 1000 or not (x == 0 or x + cw == W):
            continue
        comp = labels[y : y + chh, :] == lbl
        g = gray[y : y + chh, :][comp]
        if float((g >= 250).mean()) >= T_FRAC250 or \
           float(edge[y : y + chh, :][comp].mean()) <= T3_EDGE:
            seed[y : y + chh, :] |= comp & (gray[y : y + chh, :] >= 250)

    num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(soft_m.astype(np.uint8), 4)
    overlap = np.bincount(labels_s[seed & soft_m], minlength=num_s)
    sel = [l for l in range(1, num_s)
           if overlap[l] >= OVERLAP_PX and overlap[l] >= OVERLAP_FRAC * stats_s[l, 4]]
    extent = np.isin(labels_s, np.array(sel)) if sel else np.zeros_like(seed)

    delete = seed | (extent & (gray >= EXTENT_WHITE))

    # Track 1 hook (opt-in only; off by default -- see module docstring)
    bands = find_bands(gray) if ckpt is not None else []
    if bands and ckpt.exists():
        net = load_band_net(ckpt)
        is_gutter = classify_bands(gray, bands, net)
        for (b0, b1), g in zip(bands, is_gutter):
            if not g:
                delete[b0 + 2 : b1 - 2, :] = False

    delete = repair_frame_interiors(rgb_u8, delete, frame_darkness=40,
                                     min_interior_px=10000, inset_px=2)
    leak, _ = detect_leaks(gray, delete, seed=None)
    return delete & ~leak


if __name__ == "__main__":
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    ckpt = Path(sys.argv[sys.argv.index("--ckpt") + 1]) if "--ckpt" in sys.argv else DEFAULT_CKPT
    rgb = np.asarray(Image.open(src_path).convert("RGB"))
    mask = clean_page(rgb, ckpt)
    np.save(out_path, mask)
    print(f"deleted {mask.mean():.4f} of page -> {out_path}")
