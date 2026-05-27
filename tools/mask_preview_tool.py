#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


# ==================================================
# DEFAULT SETTINGS
# ==================================================

DEFAULT_BLACK = 24
DEFAULT_WHITE = 31
DEFAULT_GAMMA_X100 = 100
DEFAULT_THRESHOLD = 20
DEFAULT_MIN_RADIUS = 18
DEFAULT_MAX_RADIUS = 18
DEFAULT_ZOOM_X100 = 100
DEFAULT_OVERLAY_ALPHA_X100 = 65

# 0 = grayscale
# 1 = min RGB
# 2 = max RGB
# 3 = red
# 4 = green
# 5 = blue
DEFAULT_CHANNEL_MODE = "grayscale"

# 0 = original
# 1 = levels
# 2 = mask
# 3 = red overlay
# 4 = split original/mask
DEFAULT_PREVIEW_MODE = "red overlay"

BG_COLOR = "#1b1f24"
PANEL_BG = "#252a30"


# ==================================================


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resolve_image_path(image_arg: str) -> Path:
    """
    Resolve image argument for the current project layout.

    Examples:
        033
        033.png
        data/chapters-long/033.png

    Default project path:
        data/chapters-long/<CH>.png
    """
    p = Path(image_arg)

    if p.exists():
        return p

    if p.suffix.lower() == ".png":
        candidate = Path("data") / "chapters-long" / p.name
    else:
        candidate = Path("data") / "chapters-long" / f"{image_arg}.png"

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Image not found: {image_arg}\n"
        f"Also tried: {candidate}"
    )


def default_output_dir_for_image(image_path: Path) -> Path:
    return Path("data") / "temp" / image_path.stem / "mask-preview"


def clamp(value: int | float, low: int | float, high: int | float):
    return max(low, min(value, high))


def apply_channel(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "grayscale":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    if mode == "min RGB":
        return np.minimum(np.minimum(r, g), b)

    if mode == "max RGB":
        return np.maximum(np.maximum(r, g), b)

    if mode == "red":
        return r

    if mode == "green":
        return g

    if mode == "blue":
        return b

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def apply_levels(channel: np.ndarray, black: int, white: int, gamma_x100: int) -> np.ndarray:
    black = int(clamp(black, 0, 254))
    white = int(clamp(white, black + 1, 255))

    gamma = max(0.05, gamma_x100 / 100.0)

    x = channel.astype(np.float32)
    x = (x - black) / (white - black)
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, 1.0 / gamma)
    x = x * 255.0

    return x.astype(np.uint8)


def threshold_mask(levels_channel: np.ndarray, threshold_value: int) -> np.ndarray:
    return np.where(levels_channel < threshold_value, 0, 255).astype(np.uint8)


def apply_minimum_maximum(mask: np.ndarray, min_radius: int, max_radius: int) -> np.ndarray:
    out = mask

    if min_radius > 0:
        k = min_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.erode(out, kernel, iterations=1)

    if max_radius > 0:
        k = max_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel, iterations=1)

    return out


def make_red_overlay(rgb: np.ndarray, mask: np.ndarray, alpha_x100: int) -> np.ndarray:
    alpha = clamp(alpha_x100 / 100.0, 0.0, 1.0)

    out = rgb.copy()
    black = mask < 128

    red = np.zeros_like(out)
    red[:, :, 0] = 255

    out[black] = (
        out[black].astype(np.float32) * (1.0 - alpha)
        + red[black].astype(np.float32) * alpha
    ).astype(np.uint8)

    return out


def make_split_view(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    out = rgb.copy()

    split_x = out.shape[1] // 2
    out[:, split_x:] = mask_rgb[:, split_x:]

    cv2.line(out, (split_x, 0), (split_x, out.shape[0] - 1), (255, 0, 0), 2)

    return out


class ValueSlider:
    def __init__(
        self,
        parent,
        label: str,
        from_: int,
        to: int,
        value: int,
        command,
    ):
        self.command = command
        self.var = tk.IntVar(value=value)

        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=10, pady=5)

        top = ttk.Frame(frame)
        top.pack(fill="x")

        ttk.Label(top, text=label).pack(side="left")

        self.entry = ttk.Entry(top, width=7, textvariable=self.var)
        self.entry.pack(side="right")
        self.entry.bind("<Return>", self.on_entry)
        self.entry.bind("<FocusOut>", self.on_entry)

        self.scale = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            orient="horizontal",
            command=self.on_scale,
        )
        self.scale.set(value)
        self.scale.pack(fill="x")

    def on_scale(self, value):
        self.var.set(int(float(value)))
        self.command()

    def on_entry(self, _event=None):
        try:
            value = int(self.entry.get())
        except ValueError:
            value = self.var.get()

        value = int(clamp(value, int(float(self.scale.cget("from"))), int(float(self.scale.cget("to")))))
        self.var.set(value)
        self.scale.set(value)
        self.command()

    def get(self) -> int:
        return int(self.var.get())

    def set(self, value: int):
        self.var.set(value)
        self.scale.set(value)


class LiveThresholdGUI:
    def __init__(self, image_path: Path, output_dir: Path | None = None):
        self.image_path = image_path
        self.output_dir = output_dir
        self.rgb = load_rgb(image_path)
        self.img_h, self.img_w = self.rgb.shape[:2]

        self.root = tk.Tk()
        self.root.title(f"Live Threshold GUI - {image_path.name}")
        self.root.geometry("1500x950")
        self.root.configure(bg=BG_COLOR)

        self.y_offset = 0.0
        self.x_offset = 0.0
        self.zoom = DEFAULT_ZOOM_X100 / 100.0

        self.tk_image = None
        self.last_crop = None
        self.last_preview = None
        self.last_mask = None
        self.last_levels = None

        self.build_ui()
        self.bind_events()

        self.root.after(100, self.update_preview)

    def build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        panel = ttk.Frame(main, width=330)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        ttk.Label(panel, text="Live Threshold Lab", font=("Arial", 16, "bold")).pack(pady=(12, 8))

        self.black_slider = ValueSlider(panel, "Levels black", 0, 255, DEFAULT_BLACK, self.update_preview)
        self.white_slider = ValueSlider(panel, "Levels white", 1, 255, DEFAULT_WHITE, self.update_preview)
        self.gamma_slider = ValueSlider(panel, "Gamma x100", 5, 300, DEFAULT_GAMMA_X100, self.update_preview)
        self.threshold_slider = ValueSlider(panel, "Threshold", 0, 255, DEFAULT_THRESHOLD, self.update_preview)
        self.min_slider = ValueSlider(panel, "Minimum radius", 0, 40, DEFAULT_MIN_RADIUS, self.update_preview)
        self.max_slider = ValueSlider(panel, "Maximum radius", 0, 40, DEFAULT_MAX_RADIUS, self.update_preview)
        self.overlay_slider = ValueSlider(panel, "Overlay alpha %", 0, 100, DEFAULT_OVERLAY_ALPHA_X100, self.update_preview)

        ttk.Separator(panel).pack(fill="x", padx=10, pady=10)

        ttk.Label(panel, text="Channel").pack(anchor="w", padx=10)
        self.channel_var = tk.StringVar(value=DEFAULT_CHANNEL_MODE)
        channel_box = ttk.Combobox(
            panel,
            textvariable=self.channel_var,
            values=["grayscale", "min RGB", "max RGB", "red", "green", "blue"],
            state="readonly",
        )
        channel_box.pack(fill="x", padx=10, pady=5)
        channel_box.bind("<<ComboboxSelected>>", lambda _e: self.update_preview())

        ttk.Label(panel, text="Preview").pack(anchor="w", padx=10)
        self.preview_var = tk.StringVar(value=DEFAULT_PREVIEW_MODE)
        preview_box = ttk.Combobox(
            panel,
            textvariable=self.preview_var,
            values=["original", "levels", "mask", "red overlay", "split original/mask"],
            state="readonly",
        )
        preview_box.pack(fill="x", padx=10, pady=5)
        preview_box.bind("<<ComboboxSelected>>", lambda _e: self.update_preview())

        ttk.Separator(panel).pack(fill="x", padx=10, pady=10)

        button_row = ttk.Frame(panel)
        button_row.pack(fill="x", padx=10, pady=5)

        ttk.Button(button_row, text="Save", command=self.save_current).pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(button_row, text="Print", command=self.print_settings).pack(side="left", fill="x", expand=True, padx=(5, 0))

        ttk.Button(panel, text="Reset view", command=self.reset_view).pack(fill="x", padx=10, pady=5)

        self.info_label = ttk.Label(panel, text="", justify="left")
        self.info_label.pack(fill="x", padx=10, pady=10)

        help_text = (
            "Controls:\n"
            "Mouse wheel = scroll\n"
            "Alt + wheel = zoom\n"
            "s = save\n"
            "p = print settings\n"
            "q / Esc = quit\n"
        )
        ttk.Label(panel, text=help_text, justify="left").pack(fill="x", padx=10, pady=10)

    def bind_events(self):
        self.root.bind("<Configure>", lambda _e: self.update_preview())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        self.root.bind("q", lambda _e: self.root.destroy())
        self.root.bind("s", lambda _e: self.save_current())
        self.root.bind("p", lambda _e: self.print_settings())

        # Windows/macOS wheel
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # Linux wheel
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

    def is_alt_pressed(self, event) -> bool:
        # Tk Mod1Mask is usually Alt.
        return bool(event.state & 0x0008)

    def wheel_direction(self, event) -> int:
        if hasattr(event, "delta") and event.delta != 0:
            return 1 if event.delta > 0 else -1

        if hasattr(event, "num"):
            if event.num == 4:
                return 1
            if event.num == 5:
                return -1

        return 0

    def on_mouse_wheel(self, event):
        direction = self.wheel_direction(event)
        if direction == 0:
            return

        if self.is_alt_pressed(event):
            self.zoom_at_mouse(event.x, event.y, direction)
        else:
            self.scroll_vertical(direction)

        self.update_preview()

    def scroll_vertical(self, direction: int):
        step = 140 / self.zoom
        self.y_offset -= direction * step
        self.clamp_offsets()

    def zoom_at_mouse(self, mouse_x: int, mouse_y: int, direction: int):
        old_zoom = self.zoom

        if direction > 0:
            new_zoom = old_zoom * 1.12
        else:
            new_zoom = old_zoom / 1.12

        new_zoom = clamp(new_zoom, 0.20, 8.00)

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        crop_w = min(self.img_w, int(canvas_w / old_zoom))
        crop_h = min(self.img_h, int(canvas_h / old_zoom))

        draw_w = int(crop_w * old_zoom)
        draw_h = int(crop_h * old_zoom)

        img_x = max(0, (canvas_w - draw_w) // 2)
        img_y = max(0, (canvas_h - draw_h) // 2)

        src_x_under_mouse = self.x_offset + (mouse_x - img_x) / old_zoom
        src_y_under_mouse = self.y_offset + (mouse_y - img_y) / old_zoom

        self.zoom = new_zoom

        self.x_offset = src_x_under_mouse - (mouse_x - img_x) / new_zoom
        self.y_offset = src_y_under_mouse - (mouse_y - img_y) / new_zoom

        self.clamp_offsets()

    def clamp_offsets(self):
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        crop_w = min(self.img_w, int(canvas_w / self.zoom))
        crop_h = min(self.img_h, int(canvas_h / self.zoom))

        self.x_offset = clamp(self.x_offset, 0, max(0, self.img_w - crop_w))
        self.y_offset = clamp(self.y_offset, 0, max(0, self.img_h - crop_h))

    def reset_view(self):
        self.zoom = 1.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.update_preview()

    def get_settings(self) -> dict:
        black = self.black_slider.get()
        white = self.white_slider.get()

        if white <= black:
            white = min(255, black + 1)

        return {
            "image": str(self.image_path),
            "levels_black": black,
            "levels_white": white,
            "gamma": self.gamma_slider.get() / 100.0,
            "gamma_x100": self.gamma_slider.get(),
            "threshold": self.threshold_slider.get(),
            "minimum_radius": self.min_slider.get(),
            "maximum_radius": self.max_slider.get(),
            "overlay_alpha": self.overlay_slider.get() / 100.0,
            "overlay_alpha_x100": self.overlay_slider.get(),
            "channel": self.channel_var.get(),
            "preview": self.preview_var.get(),
            "zoom": self.zoom,
            "x_offset": int(self.x_offset),
            "y_offset": int(self.y_offset),
        }

    def process_crop(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        settings = self.get_settings()

        channel = apply_channel(crop, settings["channel"])
        levels = apply_levels(
            channel,
            settings["levels_black"],
            settings["levels_white"],
            settings["gamma_x100"],
        )

        mask = threshold_mask(levels, settings["threshold"])
        mask = apply_minimum_maximum(
            mask,
            settings["minimum_radius"],
            settings["maximum_radius"],
        )

        preview_mode = settings["preview"]

        if preview_mode == "original":
            preview = crop.copy()
        elif preview_mode == "levels":
            preview = cv2.cvtColor(levels, cv2.COLOR_GRAY2RGB)
        elif preview_mode == "mask":
            preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        elif preview_mode == "split original/mask":
            preview = make_split_view(crop, mask)
        else:
            preview = make_red_overlay(crop, mask, settings["overlay_alpha_x100"])

        return preview, mask, levels

    def update_preview(self):
        self.clamp_offsets()

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())

        crop_w = min(self.img_w, max(1, int(canvas_w / self.zoom)))
        crop_h = min(self.img_h, max(1, int(canvas_h / self.zoom)))

        x0 = int(clamp(self.x_offset, 0, max(0, self.img_w - crop_w)))
        y0 = int(clamp(self.y_offset, 0, max(0, self.img_h - crop_h)))

        crop = self.rgb[y0:y0 + crop_h, x0:x0 + crop_w].copy()

        preview, mask, levels = self.process_crop(crop)

        new_w = max(1, int(preview.shape[1] * self.zoom))
        new_h = max(1, int(preview.shape[0] * self.zoom))

        interpolation = cv2.INTER_NEAREST if self.zoom >= 1.0 else cv2.INTER_AREA
        display = cv2.resize(preview, (new_w, new_h), interpolation=interpolation)

        self.tk_image = ImageTk.PhotoImage(Image.fromarray(display))

        self.canvas.delete("all")
        draw_x = max(0, (canvas_w - new_w) // 2)
        draw_y = max(0, (canvas_h - new_h) // 2)
        self.canvas.create_image(draw_x, draw_y, image=self.tk_image, anchor="nw")

        self.last_crop = crop
        self.last_preview = preview
        self.last_mask = mask
        self.last_levels = levels

        settings = self.get_settings()
        info = (
            f"Image: {self.img_w}x{self.img_h}\n"
            f"View: x={x0}, y={y0}, zoom={self.zoom:.2f}\n"
            f"Levels: {settings['levels_black']} / "
            f"{settings['gamma']:.2f} / {settings['levels_white']}\n"
            f"Threshold: {settings['threshold']}\n"
            f"Min/Max: {settings['minimum_radius']} / {settings['maximum_radius']}\n"
            f"Channel: {settings['channel']}\n"
            f"Preview: {settings['preview']}"
        )
        self.info_label.config(text=info)

    def print_settings(self):
        settings = self.get_settings()
        print("\nCurrent settings:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

    def save_current(self):
        if self.last_crop is None or self.last_preview is None or self.last_mask is None or self.last_levels is None:
            return

        out_dir = self.output_dir if self.output_dir is not None else self.image_path.with_suffix("").parent / f"{self.image_path.stem}_live_gui"
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.image_path.stem}_{stamp}"

        Image.fromarray(self.last_crop).save(out_dir / f"{prefix}_crop.png")
        Image.fromarray(self.last_preview).save(out_dir / f"{prefix}_preview.png")
        Image.fromarray(self.last_mask).save(out_dir / f"{prefix}_mask.png")
        Image.fromarray(self.last_levels).save(out_dir / f"{prefix}_levels.png")

        settings = self.get_settings()
        with (out_dir / f"{prefix}_settings.txt").open("w", encoding="utf-8") as f:
            for key, value in settings.items():
                f.write(f"{key}: {value}\n")

        print(f"Saved to: {out_dir}")

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Live GUI for Levels + Threshold + Minimum/Maximum.")
    parser.add_argument("image", help="Chapter id or image path. Example: 033, 033.png, data/chapters-long/033.png")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saved previews/settings. Default: data/temp/<CH>/mask-preview",
    )
    args = parser.parse_args()

    image_path = resolve_image_path(args.image)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir_for_image(image_path)

    print(f"Image: {image_path}")
    print(f"Output dir: {output_dir}")

    app = LiveThresholdGUI(image_path, output_dir=output_dir)
    app.run()


if __name__ == "__main__":
    main()
