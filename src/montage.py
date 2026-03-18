import cv2
import numpy as np
import os
import glob
import sys

# Usage:
#   python montage.py frames out.mp4

W, H = 1080, 1920
FPS = 30
BG = (0, 0, 0)

# Timing
INTRO_SEC = 1.2          # frame appearance
HOLD_SEC = 1.8           # hold after intro
LONG_SCROLL_SEC = 6.5    # scroll time for very tall frames
OUTRO_SEC = 0.45         # slight fade before next frame

# Long-frame rule
LONG_BASE_W = 690
LONG_BASE_H = 1800
LONG_RATIO_THRESHOLD = LONG_BASE_H / LONG_BASE_W  # ~3.22

def ease_out_cubic(t):
    return 1 - (1 - t) ** 3

def ease_in_out_sine(t):
    return -(np.cos(np.pi * t) - 1) / 2

def alpha_of(img):
    if img.shape[2] == 4:
        return img[:, :, 3].astype(np.float32) / 255.0, img[:, :, :3]
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

def fit_normal(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h)
    nw, nh = int(w * s), int(h * s)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def resize_by_width(img, target_w):
    h, w = img.shape[:2]
    s = target_w / w
    nw, nh = int(w * s), int(h * s)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def is_long_frame(img):
    h, w = img.shape[:2]
    return (h / w) > LONG_RATIO_THRESHOLD

def write_frame(vw, frame):
    vw.write(frame)

def render_normal_frame(vw, img):
    intro_n = int(INTRO_SEC * FPS)
    hold_n = int(HOLD_SEC * FPS)
    outro_n = int(OUTRO_SEC * FPS)

    fitted = fit_normal(img, int(W * 0.9), int(H * 0.9))
    fh, fw = fitted.shape[:2]

    for i in range(intro_n):
        t = ease_out_cubic((i + 1) / max(1, intro_n))
        scale = 0.90 + 0.10 * t
        opacity = t

        sw, sh = max(1, int(fw * scale)), max(1, int(fh * scale))
        fr = cv2.resize(fitted, (sw, sh), interpolation=cv2.INTER_AREA)

        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        x = (W - sw) // 2
        y = (H - sh) // 2
        canvas = overlay(canvas, fr, x, y, opacity)
        write_frame(vw, canvas)

    for _ in range(hold_n):
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        x = (W - fw) // 2
        y = (H - fh) // 2
        canvas = overlay(canvas, fitted, x, y, 1.0)
        write_frame(vw, canvas)

    for i in range(outro_n):
        t = 1.0 - ((i + 1) / max(1, outro_n)) * 0.18
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        x = (W - fw) // 2
        y = (H - fh) // 2
        canvas = overlay(canvas, fitted, x, y, t)
        write_frame(vw, canvas)

def render_long_frame(vw, img):
    intro_n = int(INTRO_SEC * FPS)
    hold_n = int(0.8 * FPS)
    scroll_n = int(LONG_SCROLL_SEC * FPS)
    outro_n = int(OUTRO_SEC * FPS)

    target_w = int(W * 0.82)
    fitted = resize_by_width(img, target_w)
    fh, fw = fitted.shape[:2]

    x = (W - fw) // 2

    # Intro: appear from slight zoom + fade
    for i in range(intro_n):
        t = ease_out_cubic((i + 1) / max(1, intro_n))
        scale = 0.94 + 0.06 * t
        opacity = t

        sw, sh = max(1, int(fw * scale)), max(1, int(fh * scale))
        fr = cv2.resize(fitted, (sw, sh), interpolation=cv2.INTER_AREA)

        sx = (W - sw) // 2
        sy = int(H * 0.08)
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        canvas = overlay(canvas, fr, sx, sy, opacity)
        write_frame(vw, canvas)

    # Small hold at top
    for _ in range(hold_n):
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        canvas = overlay(canvas, fitted, x, int(H * 0.08), 1.0)
        write_frame(vw, canvas)

    # If not actually taller than viewport after resize, just hold
    visible_h = H - int(H * 0.16)
    if fh <= visible_h:
        for _ in range(int(2.0 * FPS)):
            canvas = np.full((H, W, 3), BG, dtype=np.uint8)
            y = (H - fh) // 2
            canvas = overlay(canvas, fitted, x, y, 1.0)
            write_frame(vw, canvas)
    else:
        start_y = int(H * 0.08)
        end_y = H - int(H * 0.08) - fh
        for i in range(scroll_n):
            t = ease_in_out_sine((i + 1) / max(1, scroll_n))
            y = int(start_y + (end_y - start_y) * t)

            canvas = np.full((H, W, 3), BG, dtype=np.uint8)
            canvas = overlay(canvas, fitted, x, y, 1.0)
            write_frame(vw, canvas)

    for i in range(outro_n):
        opacity = 1.0 - ((i + 1) / max(1, outro_n)) * 0.18
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        y = min(int(H * 0.08), H - fh)
        canvas = overlay(canvas, fitted, x, y, opacity)
        write_frame(vw, canvas)

def main(frames_dir, out_path):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise FileNotFoundError("No PNG frames found")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skip unreadable file: {p}")
            continue

        if is_long_frame(img):
            render_long_frame(vw, img)
        else:
            render_normal_frame(vw, img)

    vw.release()
    print("Saved:", out_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python frames_to_mp4.py frames out.mp4")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
