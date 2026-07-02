#!/usr/bin/env python3
"""
v0.4.0 — supervised pixel classifier via OpenCV Random Trees.

Manual rule-based restore keeps failing because identical #FFFFFF appears
both in the gutter and inside panels. The right fix is supervised learning:
give the algorithm one manually cleaned example and let it learn the rule.

Approach:
  1. Load train_image + train_mask (manually cleaned PNG with alpha).
  2. Build per-pixel features for all white pixels: local texture statistics,
     distance to nearest dark pixel, row/column position, HSV values.
  3. From train_mask alpha: label white pixels as delete (0) or keep (255).
  4. Train cv2.ml.RTrees on sampled white pixels.
  5. For a new image: build features for its white pixels, predict delete/keep.
  6. Apply predicted mask as alpha channel.

Usage:
  python remove_manhwa_bg.py 1.png 1_result.png \
      --mode learn --train-image 1.png --train-mask 5.png

  # folder mode:
  python remove_manhwa_bg.py ./chapters ./out --folder \
      --mode learn --train-image 1.png --train-mask 5.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

MAX_TRAIN_SAMPLES = 250_000
WHITE_V_THR = 240
WHITE_S_THR = 12


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGB"))


def load_rgba(path: str | Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGBA"))


def build_white_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[:, :, 2] >= WHITE_V_THR) & (hsv[:, :, 1] <= WHITE_S_THR)


def build_features(rgb: np.ndarray, white_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix for all white pixels.
    Returns (features [N, F], yx_coords [N, 2]).
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    ys, xs = np.where(white_mask)

    # Normalised position
    norm_y = ys / h
    norm_x = xs / w

    # Local grayscale mean in 5×5 window
    blur5 = cv2.blur(gray.astype(np.float32), (5, 5))
    mean5 = blur5[ys, xs]

    # HSV values
    v_channel = hsv[:, :, 2].astype(np.float32)
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_vals = v_channel[ys, xs]
    s_vals = s_channel[ys, xs]

    # Row ink density (fraction of dark pixels per row)
    ink_row = (gray < 128).astype(np.float32).mean(axis=1)
    row_ink = ink_row[ys]

    features = np.column_stack([norm_y, norm_x, mean5, v_vals, s_vals, row_ink]).astype(np.float32)
    coords = np.column_stack([ys, xs])

    return features, coords


def train_model(train_rgb: np.ndarray, train_rgba: np.ndarray) -> "cv2.ml.RTrees":
    white_mask = build_white_mask(train_rgb)
    features, coords = build_features(train_rgb, white_mask)

    alpha = train_rgba[:, :, 3]
    ys, xs = coords[:, 0], coords[:, 1]
    labels = (alpha[ys, xs] < 128).astype(np.int32)  # 1 = delete, 0 = keep

    # Downsample for speed
    if len(features) > MAX_TRAIN_SAMPLES:
        idx = np.random.choice(len(features), MAX_TRAIN_SAMPLES, replace=False)
        features = features[idx]
        labels = labels[idx]

    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(10)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50, 0))
    model.train(features, cv2.ml.ROW_SAMPLE, labels)
    return model


def apply_model(model: "cv2.ml.RTrees", rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    white_mask = build_white_mask(rgb)
    features, coords = build_features(rgb, white_mask)

    _, predicted = model.predict(features)
    predicted = predicted.flatten().astype(np.int32)

    remove_mask = np.zeros((h, w), dtype=bool)
    ys, xs = coords[:, 0], coords[:, 1]
    remove_mask[ys[predicted == 1], xs[predicted == 1]] = True

    alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def save_result(out_path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba, mode="RGBA").save(str(out_path))
    preview = rgba[:, :, :3].copy()
    preview[rgba[:, :, 3] == 0] = [255, 0, 0]
    Image.fromarray(preview).save(str(out_path.with_name(out_path.stem + "_red_preview.png")))


def process_single(
    in_path: Path,
    out_path: Path,
    train_image: Path,
    train_mask: Path,
) -> None:
    train_rgb = load_rgb(train_image)
    train_rgba = load_rgba(train_mask)
    print(f"Training from: {train_image.name} + {train_mask.name}")
    model = train_model(train_rgb, train_rgba)

    rgb = load_rgb(in_path)
    rgba = apply_model(model, rgb)
    save_result(out_path, rgba)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--folder", action="store_true")
    parser.add_argument("--mode", choices=["learn"], default="learn")
    parser.add_argument("--train-image", required=True)
    parser.add_argument("--train-mask", required=True)
    args = parser.parse_args()

    train_image = Path(args.train_image)
    train_mask = Path(args.train_mask)

    if args.folder:
        in_dir = Path(args.input)
        out_dir = Path(args.output) if args.output else in_dir.parent / (in_dir.name + "_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(in_dir.glob("*.png")):
            process_single(img_path, out_dir / img_path.name, train_image, train_mask)
    else:
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_result.png")
        process_single(in_path, out_path, train_image, train_mask)


if __name__ == "__main__":
    main()
