import cv2
import numpy as np
import sys


def fill_white_holes_inside_black_objects(
    image_path: str,
    output_path: str = "filled_output.png",
    threshold_value: int = 250,
    grow_px: int = 1,
    min_black_area: int = 80,
) -> None:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    blue = img[:, :, 0]

    # Blue channel threshold
    _, bw = cv2.threshold(blue, threshold_value, 255, cv2.THRESH_BINARY)

    # Black mask
    black = cv2.bitwise_not(bw)

    # Grow black a little to close tiny gaps
    if grow_px > 0:
        kernel = np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)
        black = cv2.dilate(black, kernel, iterations=1)

    result_black = black.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black, connectivity=8)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area < min_black_area:
            continue

        component_mask = (labels == label).astype(np.uint8) * 255

        # ROI of this black object
        roi_mask = component_mask[y:y+h, x:x+w]

        # Invert inside ROI: black object -> 0, surrounding white -> 255
        roi_inv = cv2.bitwise_not(roi_mask)

        # Flood-fill white from ROI border
        flood = roi_inv.copy()
        fh, fw = flood.shape
        ff_mask = np.zeros((fh + 2, fw + 2), np.uint8)

        border_points = []
        for xx in range(fw):
            border_points.append((xx, 0))
            border_points.append((xx, fh - 1))
        for yy in range(fh):
            border_points.append((0, yy))
            border_points.append((fw - 1, yy))

        for px, py in border_points:
            if flood[py, px] == 255:
                cv2.floodFill(flood, ff_mask, (px, py), 128)

        # Remaining white = enclosed white holes inside this black object
        holes = (flood == 255).astype(np.uint8) * 255

        # Fill those holes with black
        result_black[y:y+h, x:x+w][holes == 255] = 255

    result = cv2.bitwise_not(result_black)
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py input.png [output.png]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "filled_output.png"

    fill_white_holes_inside_black_objects(
        input_file,
        output_file,
        threshold_value=250,
        grow_px=1,
        min_black_area=80,
    )
