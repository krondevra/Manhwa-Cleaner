#!/usr/bin/env python3
"""
v0.7.0 — save/load trained model; fast-features mode.

Cleanup stage in v6 took 494s because it iterated over millions of pixels.
Now replaced with connected-components (cv2.connectedComponentsWithStats).

Key additions:
  --save-model FILE   serialize trained RTrees to XML after training
  --load-model FILE   skip training entirely and load a saved model
  --fast-features     use fewer/cheaper feature channels (3 instead of 6)
                      cuts feature-map build time significantly on large images

Workflow for 174 chapters:
  1. Train once and save:
       python remove_manhwa_bg.py 002.png 002_result.png \
           --mode learn --train-image 1.png --train-mask 5.png \
           --save-model manhwa_model.xml --artefact-cleanup

  2. Reuse model on every subsequent chapter:
       python remove_manhwa_bg.py 003.png 003_result.png \
           --mode learn --load-model manhwa_model.xml --artefact-cleanup

  3. Fast mode (less accurate, much faster):
       python remove_manhwa_bg.py 002.png 002_result.png \
           --mode learn --train-image 1.png --train-mask 5.png \
           --fast-features --save-model manhwa_model_fast.xml

NOTE: fast-features model and standard model are NOT interchangeable.
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
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGB"))


def load_rgba(path: str | Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGBA"))


def build_white_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 2] >= WHITE_V_THR) & (hsv[:, :, 1] <= WHITE_S_THR)


def extract_features(rgb: np.ndarray, ys: np.ndarray, xs: np.ndarray, fast: bool = False) -> np.ndarray:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    norm_y = ys / h
    norm_x = xs / w
    gray_vals = gray[ys, xs].astype(np.float32)

    if fast:
        return np.column_stack([norm_y, norm_x, gray_vals]).astype(np.float32)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blur5 = cv2.blur(gray.astype(np.float32), (5, 5))
    row_ink = (gray < 128).astype(np.float32).mean(axis=1)

    return np.column_stack([
        norm_y, norm_x,
        blur5[ys, xs],
        hsv[:, :, 2].astype(np.float32)[ys, xs],
        hsv[:, :, 1].astype(np.float32)[ys, xs],
        row_ink[ys],
    ]).astype(np.float32)


def train_model(train_rgb: np.ndarray, train_rgba: np.ndarray, fast: bool = False) -> "cv2.ml.RTrees":
    white_mask = build_white_mask(train_rgb)
    ys, xs = np.where(white_mask)
    alpha = train_rgba[:, :, 3]
    labels = (alpha[ys, xs] < 128).astype(np.int32)
    features = extract_features(train_rgb, ys, xs, fast=fast)

    if len(features) > MAX_TRAIN_SAMPLES:
        idx = np.random.choice(len(features), MAX_TRAIN_SAMPLES, replace=False)
        features, labels = features[idx], labels[idx]

    _log("training Random Trees model...")
    t0 = time.time()
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(10)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50, 0))
    model.train(features, cv2.ml.ROW_SAMPLE, labels)
    _log(f"model ready in {time.time()-t0:.1f}s")
    return model


def predict_chunked(model: "cv2.ml.RTrees", rgb: np.ndarray, fast: bool = False) -> np.ndarray:
    h, w = rgb.shape[:2]
    white_mask = build_white_mask(rgb)
    n_chunks = (h + CHUNK_ROWS - 1) // CHUNK_ROWS
    remove_mask = np.zeros((h, w), dtype=bool)
    t0 = time.time()

    for i in range(n_chunks):
        y0, y1 = i * CHUNK_ROWS, min((i + 1) * CHUNK_ROWS, h)
        ys_local, xs = np.where(white_mask[y0:y1])
        ys_global = ys_local + y0
        if len(ys_global) == 0:
            continue
        feat = extract_features(rgb, ys_global, xs, fast=fast)
        _, pred = model.predict(feat)
        pred = pred.flatten().astype(np.int32)
        remove_mask[ys_global[pred == 1], xs[pred == 1]] = True
        _log(f"chunk {i+1}/{n_chunks} ({(i+1)/n_chunks*100:5.1f}%) elapsed={time.time()-t0:.1f}s")

    return remove_mask


def artefact_cleanup_cc(rgb: np.ndarray, remove_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Connected-components version — much faster than pixel-by-pixel v6 cleanup."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    content = (gray < 200).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    content_zone = cv2.dilate(content, kernel).astype(bool)
    white_mask = build_white_mask(rgb)
    isolated = (white_mask & ~remove_mask & ~content_zone).astype(np.uint8)
    result = remove_mask.copy()
    result[isolated.astype(bool)] = True
    return result


def save_result(out_path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba, mode="RGBA").save(str(out_path))
    preview = rgba[:, :, :3].copy()
    preview[rgba[:, :, 3] == 0] = [255, 0, 0]
    Image.fromarray(preview).save(str(out_path.with_name(out_path.stem + "_red_preview.png")))
    _log(f"Done: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--mode", choices=["learn"], default="learn")
    parser.add_argument("--train-image")
    parser.add_argument("--train-mask")
    parser.add_argument("--save-model")
    parser.add_argument("--load-model")
    parser.add_argument("--artefact-cleanup", action="store_true")
    parser.add_argument("--artefact-radius", type=int, default=3)
    parser.add_argument("--fast-features", action="store_true")
    args = parser.parse_args()

    if args.load_model:
        _log(f"loading model from {args.load_model}...")
        model = cv2.ml.RTrees_load(args.load_model)
    else:
        assert args.train_image and args.train_mask
        train_rgb = load_rgb(args.train_image)
        train_rgba = load_rgba(args.train_mask)
        model = train_model(train_rgb, train_rgba, fast=args.fast_features)
        if args.save_model:
            model.save(args.save_model)
            _log(f"model saved: {args.save_model}")

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_result.png")

    rgb = load_rgb(in_path)
    remove_mask = predict_chunked(model, rgb, fast=args.fast_features)
    if args.artefact_cleanup:
        remove_mask = artefact_cleanup_cc(rgb, remove_mask, args.artefact_radius)

    alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
    save_result(out_path, np.dstack([rgb, alpha]))


if __name__ == "__main__":
    main()
