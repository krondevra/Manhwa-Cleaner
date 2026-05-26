#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------

FPS = 30
VIDEO_W = 720
VIDEO_H = 1280

ZOOM = 0.80

TOP_BAR_H = 0
BOTTOM_BAR_H = 95
VIEW_H = VIDEO_H - TOP_BAR_H - BOTTOM_BAR_H

HOLD_SEC = 2.0
TRANSITION_SEC = 0.60
GROUP_PAUSE_SEC = 0.35

# ---------------- TEXT ----------------

BOTTOM_TITLE_SCALE = 1.00
BOTTOM_SUB_SCALE = 0.85
BOTTOM_INFO_SCALE = 0.75

BOTTOM_TITLE_THICKNESS = 2
BOTTOM_SUB_THICKNESS = 2
BOTTOM_INFO_THICKNESS = 2

# ---------------- HOTSPOTS ----------------
# Это ЦЕНТРЫ нужных фрагментов по вертикали в длинной главе.
# Если фрагмент в видео оказался слишком высоко — увеличь число.
# Если фрагмент в видео оказался слишком низко — уменьши число.
#
# Подбирай шагами:
# грубо: +/- 3000
# точнее: +/- 1000
# финально: +/- 300

HOTSPOT_CENTER_Y = [
    18000,
    45950,
    59100,
    85200,
    113000,
]

# ---------------- INPUT / OUTPUT ----------------

BEFORE_PATH = "chapters-long/009.png"
MODEL1_PATH = "compare/009_cleaner4_red_preview.png"
MODEL2_PATH = "compare/009_cleaner5_red_preview.png"

OUTPUT_PATH = "compare/009_compare_static.mp4"

BG_COLOR = (10, 14, 20)
TEXT_COLOR = (255, 255, 255)
SUBTEXT_COLOR = (220, 220, 220)

# ---------------------------------------


def load_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def put_centered_text(
    img: np.ndarray,
    text: str,
    y: int,
    scale: float,
    thickness: int,
    color: tuple[int, int, int],
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, (VIDEO_W - text_w) // 2)

    cv2.putText(
        img,
        text,
        (x + 2, y + 2),
        font,
        scale,
        (0, 0, 0),
        thickness + 1,
        cv2.LINE_AA,
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_bottom_panel(
    frame_bgr: np.ndarray,
    title: str,
    subtitle: str,
    hotspot_index: int,
) -> np.ndarray:
    frame = frame_bgr.copy()

    cv2.rectangle(
        frame,
        (0, VIDEO_H - BOTTOM_BAR_H),
        (VIDEO_W, VIDEO_H),
        (0, 0, 0),
        -1,
    )

    info_text = f"CHAPTER 009  |  FRAGMENT {hotspot_index + 1}  |  ZOOM {int(ZOOM * 100)}%"

    put_centered_text(
        frame,
        title,
        VIDEO_H - 62,
        BOTTOM_TITLE_SCALE,
        BOTTOM_TITLE_THICKNESS,
        TEXT_COLOR,
    )

    put_centered_text(
        frame,
        subtitle,
        VIDEO_H - 34,
        BOTTOM_SUB_SCALE,
        BOTTOM_SUB_THICKNESS,
        SUBTEXT_COLOR,
    )

    put_centered_text(
        frame,
        info_text,
        VIDEO_H - 10,
        BOTTOM_INFO_SCALE,
        BOTTOM_INFO_THICKNESS,
        SUBTEXT_COLOR,
    )

    return frame


def crop_view(img_rgb: np.ndarray, y_top: int) -> np.ndarray:
    h, w, _ = img_rgb.shape

    source_view_h = int(round(VIEW_H / ZOOM))
    source_view_w = int(round(VIDEO_W / ZOOM))

    if h <= source_view_h:
        y_top = 0
    else:
        y_top = max(0, min(y_top, h - source_view_h))

    crop = img_rgb[y_top:y_top + min(source_view_h, h), :, :]

    _, cw = crop.shape[:2]

    if cw > source_view_w:
        x0 = (cw - source_view_w) // 2
        crop = crop[:, x0:x0 + source_view_w, :]

    new_w = max(1, int(round(crop.shape[1] * ZOOM)))
    new_h = max(1, int(round(crop.shape[0] * ZOOM)))

    crop_resized = cv2.resize(
        crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full((VIDEO_H, VIDEO_W, 3), BG_COLOR, dtype=np.uint8)

    x = (VIDEO_W - new_w) // 2
    y = TOP_BAR_H

    crop_resized = crop_resized[:VIEW_H, :, :]
    new_h, new_w = crop_resized.shape[:2]

    canvas[y:y + new_h, x:x + new_w] = crop_resized

    return rgb_to_bgr(canvas)


def make_still_frame(
    img_rgb: np.ndarray,
    y_top: int,
    title: str,
    subtitle: str,
    hotspot_index: int,
) -> np.ndarray:
    frame = crop_view(img_rgb, y_top)
    frame = draw_bottom_panel(frame, title, subtitle, hotspot_index)
    return frame


def crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return cv2.addWeighted(a, 1.0 - t, b, t, 0.0)


def repeat_frame(writer: cv2.VideoWriter, frame: np.ndarray, sec: float):
    count = int(round(sec * FPS))
    for _ in range(count):
        writer.write(frame)


def transition(writer: cv2.VideoWriter, a: np.ndarray, b: np.ndarray, sec: float):
    count = int(round(sec * FPS))

    for i in range(count):
        t = (i + 1) / count
        writer.write(crossfade(a, b, t))


def make_hotspots_from_centers(image_height: int) -> list[int]:
    source_view_h = int(round(VIEW_H / ZOOM))
    max_y = max(0, image_height - source_view_h)

    hotspots = []

    for center_y in HOTSPOT_CENTER_Y:
        y_top = int(center_y - source_view_h / 2)
        y_top = max(0, min(y_top, max_y))
        hotspots.append(y_top)

    return hotspots


def main():
    before = load_rgb(BEFORE_PATH)
    model1 = load_rgb(MODEL1_PATH)
    model2 = load_rgb(MODEL2_PATH)

    if before.shape != model1.shape or before.shape != model2.shape:
        raise ValueError("All images must have the same dimensions.")

    image_height = before.shape[0]
    hotspots = make_hotspots_from_centers(image_height)

    print("Image height:", image_height)
    print("Source viewport height:", int(round(VIEW_H / ZOOM)))
    print("Hotspot top Y values:", hotspots)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        FPS,
        (VIDEO_W, VIDEO_H),
    )

    for idx, y in enumerate(hotspots):
        frame_before = make_still_frame(before, y, "BEFORE", "Original", idx)
        frame_m1 = make_still_frame(model1, y, "MODEL 1", "5 samples", idx)
        frame_m2 = make_still_frame(model2, y, "MODEL 2", "9 samples", idx)

        repeat_frame(writer, frame_before, HOLD_SEC)

        transition(writer, frame_before, frame_m1, TRANSITION_SEC)
        repeat_frame(writer, frame_m1, HOLD_SEC)

        transition(writer, frame_m1, frame_m2, TRANSITION_SEC)
        repeat_frame(writer, frame_m2, HOLD_SEC)

        repeat_frame(writer, frame_m2, GROUP_PAUSE_SEC)

    writer.release()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
