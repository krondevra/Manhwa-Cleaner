import cv2
import numpy as np
import os
import glob
import sys

# Usage:
#   python montage.py frames out.mp4

W, H = 1080, 1920
FPS = 30

SHOW_SEC = 1.8
TRANS_SEC = 0.45
BG = (0, 0, 0)

def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def fit_image(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h)
    nw, nh = int(w * s), int(h * s)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def alpha_of(img):
    if img.shape[2] == 4:
        return img[:, :, 3] / 255.0, img[:, :, :3]
    return np.ones(img.shape[:2], np.float32), img[:, :, :3]

def overlay(bg, fg, x, y, opacity=1.0):
    out = bg.copy()
    a, rgb = alpha_of(fg)
    a = (a * opacity)[..., None]
    h, w = rgb.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(out.shape[1], x + w), min(out.shape[0], y + h)
    if x1 >= x2 or y1 >= y2:
        return out
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    fg_crop = rgb[fy1:fy2, fx1:fx2].astype(np.float32)
    a_crop = a[fy1:fy2, fx1:fx2]
    bg_crop = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (fg_crop * a_crop + bg_crop * (1 - a_crop)).astype(np.uint8)
    return out

def main(frames_dir, out_path):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise FileNotFoundError("No PNG frames found")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    show_n = int(SHOW_SEC * FPS)
    trans_n = int(TRANS_SEC * FPS)

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        img = fit_image(img, int(W * 0.9), int(H * 0.9))

        for i in range(trans_n):
            t = ease_out_cubic((i + 1) / trans_n)
            scale = 0.85 + 0.15 * t
            opacity = t

            h, w = img.shape[:2]
            sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
            fr = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

            canvas = np.full((H, W, 3), BG, dtype=np.uint8)
            x = (W - sw) // 2
            y = (H - sh) // 2
            canvas = overlay(canvas, fr, x, y, opacity)
            vw.write(canvas)

        for _ in range(show_n):
            canvas = np.full((H, W, 3), BG, dtype=np.uint8)
            x = (W - img.shape[1]) // 2
            y = (H - img.shape[0]) // 2
            canvas = overlay(canvas, img, x, y, 1.0)
            vw.write(canvas)

    vw.release()
    print("Saved:", out_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python frames_to_mp4.py frames out.mp4")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
