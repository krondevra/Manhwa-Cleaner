"""Export (image.jpg, mask.png) pairs for CascadePSP finetuning, from the
already-existing data/dataset_split_scaled/{train,val} tier -- P&C only,
never real manhwa (see docs/ml_strategy_history.md "Core architecture"
policy boundary).

CascadePSP's own OnlineTransformDataset (util/boundary_modification.py)
already synthesizes the "coarse seg mask" from a clean GT mask ONLINE, per
epoch, via random dilate/erode/hole-punching targeting a random IoU -- so
this script does NOT need to generate perturbed pairs itself, only the
(image, clean GT keep-mask) pairs in the flat directory layout their
method=1 loader expects (XXX.jpg / XXX.png, same basename).

Only exports the variants SmallUNet is actually trained on
(ml_cleaner.py::BASE_VARIANTS/OVERLAY_VARIANTS, minus "initial" -- real
chapters always have a frame, "initial" is the pre-frame raw render and
not representative of what CascadePSP will be asked to refine), so the
refiner sees the same visual domain as the model it's refining.

Also writes a <id>.strata.json sidecar per exported page with candidate
crop-center coordinates for two content strata, for the stratified-sampling
dataset in train_cascadepsp_pc.py (the single most important design
detail in .tmp/notes/cascadepsp_finetune_plan.md -- the finetune must see
both the low-texture-interior failure mode and the boundary/gutter win
case, not a content-blind uniform crop distribution):
  - "low_texture": centroids of large, low-local-contrast KEEP regions
    (flat art interiors -- sky/sea/solid fills) using the same 7x7
    MORPH_GRADIENT local-contrast measure ml_cleaner.py's guidance
    channels use, restricted to real content (keep_mask==255) so gutter/
    background (also low-contrast, but delete==0) is excluded.
  - "boundary": subsampled points along the keep/delete mask contour.

Usage:
  python3 src/export_cascadepsp_pairs.py                  # all episodes, both splits
  python3 src/export_cascadepsp_pairs.py --episode ep01    # one episode (smoke test)
  python3 src/export_cascadepsp_pairs.py --split train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml_cleaner import BASE_VARIANTS, OVERLAY_VARIANTS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_SPLIT_SCALED = REPO_ROOT / "data" / "dataset_split_scaled"
OUT_DIR = REPO_ROOT / "data" / "refinement_pairs"

# "initial" excluded: real chapters always arrive framed; refining against
# the unframed raw render would train on a visual domain production never
# actually presents at inference.
VARIANTS = [v for v in BASE_VARIANTS if v != "initial"] + OVERLAY_VARIANTS

LOCAL_CONTRAST_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
LOCAL_CONTRAST_THRESHOLD = 30  # matches ml_cleaner.py's GuidanceParams default
MIN_LOW_TEXTURE_AREA = 224 * 224 // 2  # generous enough that a 224 crop centered here stays mostly flat
BOUNDARY_STRIDE_PX = 40  # arc-length stride between sampled boundary points

JPEG_QUALITY = 95  # high, to stay close to lossless for the non-_jpeg variants


def find_low_texture_regions(gray: np.ndarray, keep_mask: np.ndarray) -> list[tuple[int, int]]:
    local_contrast = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, LOCAL_CONTRAST_KERNEL)
    flat_keep = ((local_contrast < LOCAL_CONTRAST_THRESHOLD) & (keep_mask > 0)).astype(np.uint8)
    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(flat_keep, connectivity=8)
    points = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= MIN_LOW_TEXTURE_AREA:
            cx, cy = centroids[label]
            points.append((int(round(cx)), int(round(cy))))
    return points


def find_boundary_points(keep_mask: np.ndarray) -> list[tuple[int, int]]:
    contours, _ = cv2.findContours(keep_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    points = []
    for contour in contours:
        pts = contour.reshape(-1, 2)
        if len(pts) < 2:
            continue
        # subsample by arc length, not just vertex index, so dense/sparse
        # contour segments get roughly even coverage
        seg_lens = np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum[-1]
        if total < 1:
            continue
        n_samples = max(1, int(total // BOUNDARY_STRIDE_PX))
        targets = np.linspace(0, total, n_samples, endpoint=False)
        idx = np.searchsorted(cum, targets)
        idx = np.clip(idx, 0, len(pts) - 1)
        for i in idx:
            x, y = pts[i]
            points.append((int(x), int(y)))
    return points


def export_page(rgb_path: Path, cleaned_path: Path, out_id: str, out_dir: Path) -> tuple[int, int]:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    alpha = np.asarray(Image.open(cleaned_path).convert("RGBA"))[:, :, 3]
    keep_mask = (alpha >= 128).astype(np.uint8) * 255

    Image.fromarray(rgb).save(out_dir / f"{out_id}.jpg", quality=JPEG_QUALITY)
    Image.fromarray(keep_mask, mode="L").save(out_dir / f"{out_id}.png")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    low_texture = find_low_texture_regions(gray, keep_mask)
    boundary = find_boundary_points(keep_mask)
    h, w = keep_mask.shape
    with open(out_dir / f"{out_id}.strata.json", "w") as f:
        json.dump({"w": w, "h": h, "low_texture": low_texture, "boundary": boundary}, f)

    return len(low_texture), len(boundary)


def export_split(split: str, episode_filter: str | None) -> None:
    split_dir = DATASET_SPLIT_SCALED / split
    if not split_dir.is_dir():
        print(f"skip: {split_dir} not found")
        return
    out_dir = OUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = sorted(d for d in split_dir.iterdir() if d.is_dir())
    if episode_filter:
        episodes = [d for d in episodes if episode_filter in d.name]
    if not episodes:
        print(f"no episodes matching '{episode_filter}' in {split_dir}")
        return

    n_pages = 0
    n_low_texture = 0
    n_boundary = 0
    per_variant_pages: dict[str, int] = {}

    for ep_dir in episodes:
        for variant in VARIANTS:
            src_dir = ep_dir / variant
            cleaned_dir = ep_dir / f"{variant}_cleaned"
            if not src_dir.is_dir() or not cleaned_dir.is_dir():
                continue
            for src_png in sorted(src_dir.glob("*.png")):
                cleaned_png = cleaned_dir / src_png.name
                if not cleaned_png.exists():
                    continue
                out_id = f"{split}__{ep_dir.name}__{variant}__{src_png.stem}"
                lt, bd = export_page(src_png, cleaned_png, out_id, out_dir)
                n_pages += 1
                n_low_texture += lt
                n_boundary += bd
                per_variant_pages[variant] = per_variant_pages.get(variant, 0) + 1

    print(f"[{split}] exported {n_pages} pairs to {out_dir}")
    for variant, count in sorted(per_variant_pages.items()):
        print(f"  {variant}: {count} pages")
    print(f"  strata totals: {n_low_texture} low_texture regions, {n_boundary} boundary points "
          f"({n_low_texture / max(n_pages, 1):.1f} / {n_boundary / max(n_pages, 1):.1f} per page)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--episode", default=None, help="substring filter, e.g. ep01 (for smoke-testing one episode first)")
    args = ap.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        export_split(split, args.episode)


if __name__ == "__main__":
    main()
