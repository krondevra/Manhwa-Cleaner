import cv2
import numpy as np
import sys


def fill_inside_closed_frames(
    image_path: str,
    output_path: str = "filled_frames.png",
    threshold_value: int = 250,
    grow_px: int = 1,
) -> None:
    """
    Goal:
    - use BLUE channel only
    - threshold at 250
    - grow black by 1 px
    - fill only white regions fully enclosed by black outlines
    - keep white gaps connected to image border white
    """

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # OpenCV is BGR, blue channel = 0
    blue = img[:, :, 0]

    # 250 threshold on blue channel
    # >= 250 -> white, < 250 -> black
    _, bw = cv2.threshold(blue, threshold_value, 255, cv2.THRESH_BINARY)

    # Grow black by 1 px
    black = cv2.bitwise_not(bw)
    kernel = np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)
    black = cv2.dilate(black, kernel, iterations=1)

    # Back to normal: black outlines / white background
    bw = cv2.bitwise_not(black)

    h, w = bw.shape

    # Flood fill ALL white connected to borders.
    # This preserves white gaps between frames if they are open to border.
    outside = bw.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)

    border_points = []

    for x in range(w):
        border_points.append((x, 0))
        border_points.append((x, h - 1))
    for y in range(h):
        border_points.append((0, y))
        border_points.append((w - 1, y))

    for x, y in border_points:
        if outside[y, x] == 255:
            cv2.floodFill(outside, mask, (x, y), 128)

    # White regions NOT connected to border = closed frames/boxes/bubbles
    enclosed_white = (outside == 255)

    # Fill enclosed white regions with black
    result = bw.copy()
    result[enclosed_white] = 0

    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py input.png [output.png]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "filled_frames.png"

    fill_inside_closed_frames(
        image_path=input_file,
        output_path=output_file,
        threshold_value=250,
        grow_px=1,
    )
