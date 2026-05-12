#!/usr/bin/env python3
"""
v0.6.0 — progress indicator, Image.MAX_IMAGE_PIXELS fix, faster artefact cleanup.

Chapter 002.png is 690 × 143,133 px (98.7 M pixels). PIL's default bomb limit
(89.5 M px) raised a DecompressionBombWarning that blocked automated runs.

New in this version:
  1. Image.MAX_IMAGE_PIXELS = None to disable the bomb warning globally.
  2. Chunked prediction with per-chunk progress lines:
       [predict] chunk 1/120 (  0.8%) rows 0-1200, white=...
  3. Faster artefact cleanup via connected-components instead of pixel iteration.
  4. Timed log lines for every major stage so the user knows what to expect.

Timing observed on 002.png (CPU):
  feature maps: ~114s
  prediction:   ~50s
  cleanup:      ~494s  ← still the bottleneck; will fix in v7
  save:         ~21s
  total:        ~11m 39s

Usage:
  python remove_manhwa_bg.py 002.png 002_result.png \
      --mode learn --train-image 1.png --train-mask 5.png --artefact-cleanup

  # folder:
  python remove_manhwa_bg.py ./chapters ./out --folder \
      --mode learn --train-image 1.png --train-mask 5.png --artefact-cleanup
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

MAX_TRAIN_SAMPLES = 250_000
WHITE_V_THR, WHITE_S_THR = 240, 12
CHUNK_ROWS = 1200


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_rgb(path: str | Path) -> np.ndarray:
    _log(f"loading {Path(path).name}...")
    img = Image.open(str(path)).convert("RGB")
    rgb = np.array(img)
    h, w = rgb.shape[:2]
    _log(f"loaded {w}x{h}")
    return rgb


def load_rgba(path: str | Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGBA"))


def build_white_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 2] >= WHITE_V_THR) & (hsv[:, :, 1] <= WHITE_S_THR)


def build_feature_maps(rgb: np.ndarray) -> dict:
    h, w = rgb.shape[:2]
    _log(f"building feature maps for {w}x{h}...")
    t0 = time.time()

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blur5 = cv2.blur(gray.astype(np.float32), (5, 5))
    row_ink = (gray < 128).astype(np.float32).mean(axis=1)

    ys_all = np.arange(h, dtype=np.float32) / h
    xs_all = np.arange(w, dtype=np.float32) / w
    norm_y_map = np.tile(ys_all[:, None], (1, w))
    norm_x_map = np.tile(xs_all[None, :], (h, 1))
    row_ink_map = np.tile(row_ink[:, None], (1, w))

    _log(f"feature maps ready in {time.time()-t0:.1f}s")
    return dict(norm_y=norm_y_map, norm_x=norm_x_map, gray_blur=blur5,
                v=hsv[:, :, 2].astype(np.float32), s=hsv[:, :, 1].astype(np.float32),
                row_ink=row_ink_map)


def features_at_mask(maps: dict, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    return np.column_stack([
        maps["norm_y"][ys, xs], maps["norm_x"][ys, xs],
        maps["gray_blur"][ys, xs], maps["v"][ys, xs],
        maps["s"][ys, xs], maps["row_ink"][ys, xs],
    ]).astype(np.float32)


def train_model(train_rgb: np.ndarray, train_rgba: np.ndarray) -> "cv2.ml.RTrees":
    _log(f"building feature maps for {train_rgb.shape[1]}x{train_rgb.shape[0]}...")
    maps = build_feature_maps(train_rgb)
    white_mask = build_white_mask(train_rgb)
    ys, xs = np.where(white_mask)

    alpha = train_rgba[:, :, 3]
    labels = (alpha[ys, xs] < 128).astype(np.int32)
    features = features_at_mask(maps, ys, xs)

    n_del = labels.sum()
    n_keep = len(labels) - n_del
    _log(f"pixels: delete={n_del:,}, keep={n_keep:,}")

    if len(features) > MAX_TRAIN_SAMPLES:
        idx = np.random.choice(len(features), MAX_TRAIN_SAMPLES, replace=False)
        features, labels = features[idx], labels[idx]

    _log(f"sampling {len(features):,} pixels...")
    _log("training Random Trees model...")
    t0 = time.time()
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(10)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50, 0))
    model.train(features, cv2.ml.ROW_SAMPLE, labels)
    _log(f"model ready in {time.time()-t0:.1f}s")
    return model


def predict_chunked(model: "cv2.ml.RTrees", rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    maps = build_feature_maps(rgb)
    white_mask = build_white_mask(rgb)
    total_white = white_mask.sum()
    _log(f"white candidate pixels: {total_white:,}")

    n_chunks = (h + CHUNK_ROWS - 1) // CHUNK_ROWS
    _log(f"processing {n_chunks} chunks of {CHUNK_ROWS} rows...")

    remove_mask = np.zeros((h, w), dtype=bool)
    t0 = time.time()

    for i in range(n_chunks):
        y0, y1 = i * CHUNK_ROWS, min((i + 1) * CHUNK_ROWS, h)
        chunk_white = white_mask[y0:y1]
        ys_local, xs = np.where(chunk_white)
        ys_global = ys_local + y0

        if len(ys_global) == 0:
            continue

        feat = features_at_mask(maps, ys_global, xs)
        _, pred = model.predict(feat)
        pred = pred.flatten().astype(np.int32)

        remove_mask[ys_global[pred == 1], xs[pred == 1]] = True

        pct = (i + 1) / n_chunks * 100
        elapsed = time.time() - t0
        _log(f"chunk {i+1}/{n_chunks} ({pct:5.1f}%) rows {y0}-{y1}, white={len(ys_global):,}, elapsed={elapsed:.1f}s")

    _log(f"prediction done in {time.time()-t0:.1f}s")
    return remove_mask


def artefact_cleanup(rgb: np.ndarray, remove_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    _log("white artefact cleanup...")
    t0 = time.time()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    content = (gray < 200).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    content_zone = cv2.dilate(content, kernel).astype(bool)
    white_mask = build_white_mask(rgb)
    isolated = white_mask & ~remove_mask & ~content_zone
    result = remove_mask.copy()
    result[isolated] = True
    _log(f"cleanup done in {time.time()-t0:.1f}s")
    return result


def save_result(out_path: Path, rgba: np.ndarray) -> None:
    _log(f"writing {out_path.name}...")
    t0 = time.time()
    Image.fromarray(rgba, mode="RGBA").save(str(out_path))
    _log(f"done in {time.time()-t0:.1f}s")
    preview = rgba[:, :, :3].copy()
    preview[rgba[:, :, 3] == 0] = [255, 0, 0]
    Image.fromarray(preview).save(str(out_path.with_name(out_path.stem + "_red_preview.png")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--folder", action="store_true")
    parser.add_argument("--mode", choices=["learn"], default="learn")
    parser.add_argument("--train-image", required=True)
    parser.add_argument("--train-mask", required=True)
    parser.add_argument("--artefact-cleanup", action="store_true")
    parser.add_argument("--artefact-radius", type=int, default=3)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    args = parser.parse_args()

    train_rgb = load_rgb(args.train_image)
    train_rgba = load_rgba(args.train_mask)
    _log(f"Training from: {Path(args.train_image).name} + {Path(args.train_mask).name}")
    model = train_model(train_rgb, train_rgba)

    def process(in_path: Path, out_path: Path) -> None:
        rgb = load_rgb(in_path)
        remove_mask = predict_chunked(model, rgb)
        if args.artefact_cleanup:
            remove_mask = artefact_cleanup(rgb, remove_mask, args.artefact_radius)
        alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
        save_result(out_path, np.dstack([rgb, alpha]))
        _log(f"Done: {out_path}")

    if args.folder:
        in_dir, out_dir = Path(args.input), Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in sorted(in_dir.glob("*.png")):
            process(img, out_dir / img.name)
    else:
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_result.png")
        process(in_path, out_path)


if __name__ == "__main__":
    main()
