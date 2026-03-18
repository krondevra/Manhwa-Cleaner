import cv2
import numpy as np
import os
import sys

# Usage:
#   python cut_frames.py input.png

MIN_AREA = 20000          # Minimum frame area
MERGE_GAP_X = 25          # Merge nearby parts horizontally
MERGE_GAP_Y = 25          # Merge nearby parts vertically
PAD = 4                   # Padding around frame


def boxes_close(a, b, gap_x=25, gap_y=25):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return not (
        ax2 < bx1 - gap_x or
        bx2 < ax1 - gap_x or
        ay2 < by1 - gap_y or
        by2 < ay1 - gap_y
    )


def merge_boxes(boxes, gap_x=25, gap_y=25):
    boxes = [list(b) for b in boxes]
    changed = True

    while changed:
        changed = False
        new_boxes = []
        used = [False] * len(boxes)

        for i in range(len(boxes)):
            if used[i]:
                continue

            x1, y1, x2, y2 = boxes[i]
            used[i] = True

            merged = True
            while merged:
                merged = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue
                    if boxes_close((x1, y1, x2, y2), boxes[j], gap_x, gap_y):
                        bx1, by1, bx2, by2 = boxes[j]
                        x1 = min(x1, bx1)
                        y1 = min(y1, by1)
                        x2 = max(x2, bx2)
                        y2 = max(y2, by2)
                        used[j] = True
                        merged = True
                        changed = True

            new_boxes.append([x1, y1, x2, y2])

        boxes = new_boxes

    return [tuple(b) for b in boxes]


def main(input_path):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {input_path}")

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = alpha > 0
    else:
        # fallback if no alpha: detect non-white
        bgr = img[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = gray < 250

    mask = mask.astype(np.uint8) * 255

    # Clean tiny noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        boxes.append((x, y, x + w, y + h))

    boxes = merge_boxes(boxes, MERGE_GAP_X, MERGE_GAP_Y)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right

    os.makedirs("frames", exist_ok=True)

    h, w = img.shape[:2]
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        x1 = max(0, x1 - PAD)
        y1 = max(0, y1 - PAD)
        x2 = min(w, x2 + PAD)
        y2 = min(h, y2 + PAD)

        crop = img[y1:y2, x1:x2]
        out_path = os.path.join("frames", f"frame_{idx:03d}.png")
        cv2.imwrite(out_path, crop)
        print(f"Saved: {out_path}")

    print(f"Done. Total frames: {len(boxes)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cut_frames.py merged.png")
        sys.exit(1)

    main(sys.argv[1])
