#!/usr/bin/env python3
"""
v0.5.0 — add artefact cleanup; exact-copy mode for same-file train/apply.

New features vs v4:
  1. exact-copy shortcut: if input == train-image, skip prediction and copy alpha
     directly from train-mask. Avoids model approximation on the training image itself.
  2. --artefact-cleanup: after prediction, remove small isolated white blobs that
     are not connected to any significant content region (kills white halos around
     SFX and isolated trapped-white islands).
  3. --artefact-radius controls dilation radius for content protection zone (default 3).
  4. --no-exact-train-copy forces prediction even on the training image (for diagnostics).

Usage:
  # same-file (exact copy of 5.png alpha):
  python remove_manhwa_bg.py 1.png 1_result.png \
      --mode learn --train-image 1.png --train-mask 5.png

  # other chapter + cleanup:
  python remove_manhwa_bg.py other.png other_result.png \
      --mode learn --train-image 1.png --train-mask 5.png --artefact-cleanup
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
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ys, xs = np.where(white_mask)

    norm_y = ys / h
    norm_x = xs / w
    blur5 = cv2.blur(gray.astype(np.float32), (5, 5))
    mean5 = blur5[ys, xs]
    v_vals = hsv[:, :, 2].astype(np.float32)[ys, xs]
    s_vals = hsv[:, :, 1].astype(np.float32)[ys, xs]
    row_ink = (gray < 128).astype(np.float32).mean(axis=1)[ys]

    features = np.column_stack([norm_y, norm_x, mean5, v_vals, s_vals, row_ink]).astype(np.float32)
    return features, np.column_stack([ys, xs])


def train_model(train_rgb: np.ndarray, train_rgba: np.ndarray) -> "cv2.ml.RTrees":
    white_mask = build_white_mask(train_rgb)
    features, coords = build_features(train_rgb, white_mask)
    alpha = train_rgba[:, :, 3]
    ys, xs = coords[:, 0], coords[:, 1]
    labels = (alpha[ys, xs] < 128).astype(np.int32)

    if len(features) > MAX_TRAIN_SAMPLES:
        idx = np.random.choice(len(features), MAX_TRAIN_SAMPLES, replace=False)
        features, labels = features[idx], labels[idx]

    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(10)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 50, 0))
    model.train(features, cv2.ml.ROW_SAMPLE, labels)
    return model


def predict_remove_mask(model: "cv2.ml.RTrees", rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    white_mask = build_white_mask(rgb)
    features, coords = build_features(rgb, white_mask)
    _, predicted = model.predict(features)
    predicted = predicted.flatten().astype(np.int32)

    mask = np.zeros((h, w), dtype=bool)
    ys, xs = coords[:, 0], coords[:, 1]
    mask[ys[predicted == 1], xs[predicted == 1]] = True
    return mask


def artefact_cleanup(rgb: np.ndarray, remove_mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """
    Remove small isolated white blobs that are not near any content.
    White blobs still touching content are preserved as potential speech bubbles.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    content = (gray < 200).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    content_zone = cv2.dilate(content, kernel).astype(bool)

    # Any white pixel not inside the content zone and not yet removed is trapped.
    white_mask = build_white_mask(rgb)
    not_removed = white_mask & ~remove_mask
    isolated = not_removed & ~content_zone

    result = remove_mask.copy()
    result[isolated] = True
    return result


def apply_exact_training_mask_if_same_input(
    in_path: Path, train_image: Path, train_mask: Path, no_exact: bool
) -> np.ndarray | None:
    if no_exact:
        return None
    try:
        if in_path.resolve() == train_image.resolve():
            return load_rgba(train_mask)
    except Exception:
        pass
    return None


def save_result(out_path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba, mode="RGBA").save(str(out_path))
    preview = rgba[:, :, :3].copy()
    preview[rgba[:, :, 3] == 0] = [255, 0, 0]
    Image.fromarray(preview).save(str(out_path.with_name(out_path.stem + "_red_preview.png")))


def process_image(
    in_path: Path,
    out_path: Path,
    train_image: Path,
    train_mask: Path,
    artefact: bool = False,
    artefact_radius: int = 3,
    no_exact: bool = False,
) -> None:
    exact = apply_exact_training_mask_if_same_input(in_path, train_image, train_mask, no_exact)
    if exact is not None:
        save_result(out_path, exact)
        return

    train_rgb = load_rgb(train_image)
    train_rgba = load_rgba(train_mask)
    model = train_model(train_rgb, train_rgba)

    rgb = load_rgb(in_path)
    remove_mask = predict_remove_mask(model, rgb)

    if artefact:
        remove_mask = artefact_cleanup(rgb, remove_mask, radius=artefact_radius)

    alpha = np.where(remove_mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    save_result(out_path, rgba)


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
    parser.add_argument("--no-exact-train-copy", action="store_true")
    args = parser.parse_args()

    train_image = Path(args.train_image)
    train_mask = Path(args.train_mask)

    if args.folder:
        in_dir, out_dir = Path(args.input), Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(in_dir.glob("*.png")):
            out_path = out_dir / img_path.name
            print(f"processing {img_path.name}...")
            process_image(img_path, out_path, train_image, train_mask,
                          args.artefact_cleanup, args.artefact_radius, args.no_exact_train_copy)
    else:
        in_path = Path(args.input)
        out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_result.png")
        process_image(in_path, out_path, train_image, train_mask,
                      args.artefact_cleanup, args.artefact_radius, args.no_exact_train_copy)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
