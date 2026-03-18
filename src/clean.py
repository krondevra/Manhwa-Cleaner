import cv2
import numpy as np
import sys
import os

# Usage:
#   python clean.py input.png
#
# Output:
#   *_gutters_removed.png
#   *_debug_threshold.png
#   *_debug_black_barrier.png
#   *_debug_white_safe.png
#   *_debug_remove_mask.png

THRESH = 210

# Strengthen black contours a little
CLOSE_KERNEL = (3, 3)
CLOSE_ITERS = 1
DILATE_KERNEL = (3, 3)
DILATE_ITERS = 1

# White component filters
MIN_AREA = 1500
MIN_BBOX_W = 40
MIN_BBOX_H = 8

# Gutter heuristics
MIN_WIDTH_RATIO_TOP_BOTTOM = 0.60   # delete if touches top/bottom and wide enough
MIN_WIDTH_RATIO_ANY = 0.92          # delete if almost full width anywhere
MIN_STRIP_HEIGHT = 10               # minimum strip height
MAX_COMPACT_FILL = 0.98             # optional safety for very box-like regions


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def build_barrier_map(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Threshold: white=255, dark/content=0
    _, bw = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)

    # Strengthen black contours/barriers
    black = np.where(bw == 0, 255, 0).astype(np.uint8)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)
    black_barrier = cv2.morphologyEx(
        black, cv2.MORPH_CLOSE, close_kernel, iterations=CLOSE_ITERS
    )

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KERNEL)
    black_barrier = cv2.dilate(
        black_barrier, dilate_kernel, iterations=DILATE_ITERS
    )

    # Rebuild safe white map
    white_safe = np.where(black_barrier > 0, 0, 255).astype(np.uint8)
    return bw, black_barrier, white_safe


def component_touches(x, y, w, h, img_w, img_h):
    touch_top = (y == 0)
    touch_bottom = (y + h >= img_h)
    touch_left = (x == 0)
    touch_right = (x + w >= img_w)
    return touch_top, touch_bottom, touch_left, touch_right


def main():
    if len(sys.argv) < 2:
        print("Usage: python remove_manhwa_gutters.py input.png")
        sys.exit(1)

    input_path = sys.argv[1]
    img = load_image(input_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bw, black_barrier, white_safe = build_barrier_map(gray)

    # Find white connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        white_safe, connectivity=8
    )

    remove_mask = np.zeros((h, w), dtype=np.uint8)
    debug_info = []

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        ww = stats[label, cv2.CC_STAT_WIDTH]
        hh = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_AREA:
            continue
        if ww < MIN_BBOX_W or hh < MIN_BBOX_H:
            continue

        touch_top, touch_bottom, touch_left, touch_right = component_touches(
            x, y, ww, hh, w, h
        )

        width_ratio = ww / float(w)
        bbox_area = max(1, ww * hh)
        fill_ratio = area / float(bbox_area)

        # Main rule:
        # remove only regions that look like horizontal gutters,
        # not generic white connected-to-edge regions.
        is_top_bottom_gutter = (
            (touch_top or touch_bottom)
            and width_ratio >= MIN_WIDTH_RATIO_TOP_BOTTOM
            and hh >= MIN_STRIP_HEIGHT
        )

        is_almost_full_width_strip = (
            width_ratio >= MIN_WIDTH_RATIO_ANY
            and hh >= MIN_STRIP_HEIGHT
        )

        # Optional safety:
        # compact filled boxes are often speech boxes, do not remove them
        looks_like_compact_box = (
            fill_ratio >= MAX_COMPACT_FILL
            and ww < 0.95 * w
            and hh < 0.35 * h
            and not (touch_top or touch_bottom)
        )

        is_gutter = (
            (is_top_bottom_gutter or is_almost_full_width_strip)
            and not looks_like_compact_box
        )

        if not is_gutter:
            continue

        remove_mask[labels == label] = 255
        debug_info.append({
            "label": label,
            "x": x, "y": y, "w": ww, "h": hh,
            "area": area,
            "width_ratio": round(width_ratio, 3),
            "fill_ratio": round(fill_ratio, 3),
            "touch_top": touch_top,
            "touch_bottom": touch_bottom,
            "touch_left": touch_left,
            "touch_right": touch_right,
        })

    # Apply transparency
    b, g, r = cv2.split(img)
    alpha = np.full((h, w), 255, dtype=np.uint8)
    alpha[remove_mask == 255] = 0
    out = cv2.merge([b, g, r, alpha])

    # Save files
    base, _ = os.path.splitext(input_path)
    out_path = base + "-auto-cleaned.png"
    # dbg_thresh = base + "_debug_threshold.png"
    # dbg_barrier = base + "_debug_black_barrier.png"
    # dbg_white = base + "_debug_white_safe.png"
    # dbg_mask = base + "_debug_remove_mask.png"

    cv2.imwrite(out_path, out)
    cv2.imwrite(dbg_thresh, bw)
    cv2.imwrite(dbg_barrier, black_barrier)
    cv2.imwrite(dbg_white, white_safe)
    cv2.imwrite(dbg_mask, remove_mask)

    print("Done.")
    print(f"Saved result:        {out_path}")
    # print(f"Saved threshold:     {dbg_thresh}")
    # print(f"Saved black barrier: {dbg_barrier}")
    # print(f"Saved white map:     {dbg_white}")
    # print(f"Saved remove mask:   {dbg_mask}")
    # print(f"Removed components:  {len(debug_info)}")

    for item in debug_info[:50]:
        print(item)


if __name__ == "__main__":
    main()
