#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

Image.MAX_IMAGE_PIXELS = None

BLACK_LABEL = "black"  # target mask black = delete/remove
WHITE_LABEL = "white"  # target mask white = keep/protect
ROI_PAD = 64
TOP_N = 30


@dataclass
class ROI:
    x0: int
    y0: int
    x1: int
    y1: int
    label: str


@dataclass
class TonalParams:
    channel: str
    black: int
    white: int
    gamma_x100: int
    threshold: int
    score: float
    black_error: float
    white_error: float


@dataclass
class FinalParams:
    channel: str
    black: int
    white: int
    gamma_x100: int
    threshold: int
    min_radius: int
    max_radius: int
    morph_order: str
    score: float
    black_error: float
    white_error: float


@dataclass
class SearchProfile:
    priority: str
    black_weight: float
    white_weight: float
    channels: list[str]
    level_black_values: list[int]
    level_white_values: list[int]
    gamma_x100_values: list[int]
    threshold_values: list[int]
    min_radius_values: list[int]
    max_radius_values: list[int]
    morph_orders: list[str]


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    if label in {BLACK_LABEL, WHITE_LABEL}:
        return label
    raise ValueError(f"Unknown ROI label: {label}")


def normalize_roi(x0: int, y0: int, x1: int, y1: int, label: str) -> ROI:
    rx0, rx1 = sorted([int(x0), int(x1)])
    ry0, ry1 = sorted([int(y0), int(y1)])
    return ROI(rx0, ry0, rx1, ry1, normalize_label(label))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def parse_rois_from_text(text: str) -> list[ROI]:
    rois: list[ROI] = []
    pattern = re.compile(
        r"(?P<label>BLACK|WHITE|black|white).*?"
        r"x\s*(?P<x0>\d+)\s*-\s*(?P<x1>\d+).*?"
        r"y\s*(?P<y0>\d+)\s*-\s*(?P<y1>\d+)"
    )
    for line in text.splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue
        roi = normalize_roi(
            int(match.group("x0")),
            int(match.group("y0")),
            int(match.group("x1")),
            int(match.group("y1")),
            match.group("label"),
        )
        if roi.x1 > roi.x0 and roi.y1 > roi.y0:
            rois.append(roi)
    return rois


def load_rois(path: Path) -> list[ROI]:
    rois = parse_rois_from_text(path.read_text(encoding="utf-8", errors="ignore"))
    if not rois:
        raise ValueError(f"No ROIs found in: {path}")
    return rois


def save_rois(path: Path, rois: list[ROI]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, roi in enumerate(rois):
            f.write(f"{i:03d} | {roi.label.upper()} | x {roi.x0}-{roi.x1} | y {roi.y0}-{roi.y1}\n")


def validate_rois(rois: list[ROI]) -> None:
    b = sum(1 for r in rois if r.label == BLACK_LABEL)
    w = sum(1 for r in rois if r.label == WHITE_LABEL)
    if b == 0 or w == 0:
        raise ValueError("Need at least one BLACK ROI and one WHITE ROI.")


class RoiMarker:
    """
    Tk-based ROI marker with a persistent right panel and middle-mouse panning.

    Labels:
      BLACK = should become black in generated mask = delete/remove candidate.
      WHITE = should become white in generated mask = keep/protect candidate.
    """

    def __init__(self, image_path: Path, output_path: Path):
        self.image_path = image_path
        self.output_path = output_path

        self.rgb = load_rgb(image_path)
        self.pil = Image.fromarray(self.rgb)
        self.h, self.w = self.rgb.shape[:2]

        self.rois: list[ROI] = []
        self.label = BLACK_LABEL

        self.zoom = 1.0
        self.offset_x = 0.0  # image pixels
        self.offset_y = 0.0  # image pixels

        self.drag_start_img: tuple[int, int] | None = None
        self.drag_current_img: tuple[int, int] | None = None
        self.pan_start: tuple[int, int, float, float] | None = None

        self.saved = False
        self.photo = None

        self.root = tk.Tk()
        self.root.title(f"ROI Marker - {image_path.name}")
        self.root.geometry("1450x900")

        self.main = ttk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.main,
            bg="#151515",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.panel = ttk.Frame(self.main, width=330)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.panel.pack_propagate(False)

        self._build_panel()
        self._bind_events()

        self.root.after(50, self.update_view)

    # ------------------------------
    # UI
    # ------------------------------

    def _build_panel(self) -> None:
        title = ttk.Label(self.panel, text="ROI marker", font=("TkDefaultFont", 14, "bold"))
        title.pack(anchor="w", padx=10, pady=(10, 2))

        info = ttk.Label(
            self.panel,
            text=f"{self.image_path.name}\n{self.w} × {self.h}",
            justify="left",
        )
        info.pack(anchor="w", padx=10, pady=(0, 10))

        self.mode_var = tk.StringVar(value=BLACK_LABEL)

        mode_box = ttk.LabelFrame(self.panel, text="Current label")
        mode_box.pack(fill=tk.X, padx=10, pady=6)

        ttk.Radiobutton(
            mode_box,
            text="BLACK / delete / remove",
            variable=self.mode_var,
            value=BLACK_LABEL,
            command=self.on_mode_change,
        ).pack(anchor="w", padx=8, pady=4)

        ttk.Radiobutton(
            mode_box,
            text="WHITE / keep / protect",
            variable=self.mode_var,
            value=WHITE_LABEL,
            command=self.on_mode_change,
        ).pack(anchor="w", padx=8, pady=4)

        buttons = ttk.Frame(self.panel)
        buttons.pack(fill=tk.X, padx=10, pady=8)

        ttk.Button(buttons, text="Undo", command=self.undo).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        save_buttons = ttk.Frame(self.panel)
        save_buttons.pack(fill=tk.X, padx=10, pady=4)

        ttk.Button(save_buttons, text="Save + start search", command=self.save_and_close).pack(fill=tk.X)

        self.status_var = tk.StringVar()
        ttk.Label(self.panel, textvariable=self.status_var, justify="left").pack(anchor="w", padx=10, pady=8)

        roi_box = ttk.LabelFrame(self.panel, text="ROIs")
        roi_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.roi_list = tk.Listbox(roi_box, height=16)
        self.roi_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(roi_box, orient=tk.VERTICAL, command=self.roi_list.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.roi_list.configure(yscrollcommand=scroll.set)

        help_box = ttk.LabelFrame(self.panel, text="Controls")
        help_box.pack(fill=tk.X, padx=10, pady=8)

        help_text = (
            "Left drag: create ROI\n"
            "Middle drag: pan image\n"
            "Mouse wheel: vertical scroll\n"
            "Shift + wheel: horizontal scroll\n"
            "Ctrl/Alt + wheel: zoom\n"
            "B: BLACK mode\n"
            "W: WHITE mode\n"
            "U: undo\n"
            "S / Enter: save + start\n"
            "Esc: close without saving"
        )
        ttk.Label(help_box, text=help_text, justify="left").pack(anchor="w", padx=8, pady=8)

    def _bind_events(self) -> None:
        self.canvas.bind("<Configure>", lambda e: self.update_view())

        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)

        self.canvas.bind("<ButtonPress-2>", self.on_middle_down)
        self.canvas.bind("<B2-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_middle_up)

        # Some Linux configs map middle-drag to Button-3 on touchpads/mice.
        self.canvas.bind("<ButtonPress-3>", self.on_middle_down)
        self.canvas.bind("<B3-Motion>", self.on_middle_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_middle_up)

        self.canvas.bind("<MouseWheel>", self.on_mousewheel)   # Windows / some Linux Tk builds
        self.canvas.bind("<Button-4>", self.on_wheel_up)       # X11
        self.canvas.bind("<Button-5>", self.on_wheel_down)     # X11

        self.root.bind("<KeyPress-b>", lambda e: self.set_mode(BLACK_LABEL))
        self.root.bind("<KeyPress-B>", lambda e: self.set_mode(BLACK_LABEL))
        self.root.bind("<KeyPress-w>", lambda e: self.set_mode(WHITE_LABEL))
        self.root.bind("<KeyPress-W>", lambda e: self.set_mode(WHITE_LABEL))
        self.root.bind("<KeyPress-u>", lambda e: self.undo())
        self.root.bind("<KeyPress-U>", lambda e: self.undo())
        self.root.bind("<KeyPress-s>", lambda e: self.save_and_close())
        self.root.bind("<KeyPress-S>", lambda e: self.save_and_close())
        self.root.bind("<Return>", lambda e: self.save_and_close())
        self.root.bind("<Escape>", lambda e: self.close_without_saving())

    # ------------------------------
    # Coordinate transforms
    # ------------------------------

    def canvas_size(self) -> tuple[int, int]:
        cw = max(1, int(self.canvas.winfo_width()))
        ch = max(1, int(self.canvas.winfo_height()))
        return cw, ch

    def clamp_view(self) -> None:
        cw, ch = self.canvas_size()
        view_w_img = cw / self.zoom
        view_h_img = ch / self.zoom
        self.offset_x = clamp(self.offset_x, 0, max(0, self.w - view_w_img))
        self.offset_y = clamp(self.offset_y, 0, max(0, self.h - view_h_img))

    def image_to_screen(self, ix: float, iy: float) -> tuple[int, int]:
        sx = int(round((ix - self.offset_x) * self.zoom))
        sy = int(round((iy - self.offset_y) * self.zoom))
        return sx, sy

    def screen_to_image(self, sx: float, sy: float) -> tuple[int, int]:
        ix = int(round(self.offset_x + sx / self.zoom))
        iy = int(round(self.offset_y + sy / self.zoom))
        return int(clamp(ix, 0, self.w - 1)), int(clamp(iy, 0, self.h - 1))

    # ------------------------------
    # Rendering
    # ------------------------------

    def update_status(self) -> None:
        b = sum(1 for r in self.rois if r.label == BLACK_LABEL)
        w = sum(1 for r in self.rois if r.label == WHITE_LABEL)
        self.status_var.set(
            f"Mode: {self.label.upper()}\n"
            f"ROIs: {len(self.rois)}  BLACK: {b}  WHITE: {w}\n"
            f"Zoom: {self.zoom:.2f}\n"
            f"Offset: x={int(self.offset_x)} y={int(self.offset_y)}\n"
            f"Save to:\n{self.output_path}"
        )

    def refresh_list(self) -> None:
        self.roi_list.delete(0, tk.END)
        for i, roi in enumerate(self.rois):
            self.roi_list.insert(
                tk.END,
                f"{i:03d} {roi.label.upper()}  x {roi.x0}-{roi.x1}  y {roi.y0}-{roi.y1}",
            )

    def update_view(self) -> None:
        if not self.root.winfo_exists():
            return

        self.clamp_view()
        cw, ch = self.canvas_size()

        x0 = int(self.offset_x)
        y0 = int(self.offset_y)
        x1 = int(min(self.w, self.offset_x + cw / self.zoom + 2))
        y1 = int(min(self.h, self.offset_y + ch / self.zoom + 2))

        if x1 <= x0 or y1 <= y0:
            return

        crop = self.pil.crop((x0, y0, x1, y1))
        disp_w = max(1, int(round((x1 - x0) * self.zoom)))
        disp_h = max(1, int(round((y1 - y0) * self.zoom)))
        crop = crop.resize((disp_w, disp_h), Image.Resampling.BILINEAR)

        self.photo = ImageTk.PhotoImage(crop)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        # Existing ROIs.
        for roi in self.rois:
            sx0, sy0 = self.image_to_screen(roi.x0, roi.y0)
            sx1, sy1 = self.image_to_screen(roi.x1, roi.y1)
            if sx1 < 0 or sy1 < 0 or sx0 > cw or sy0 > ch:
                continue
            color = "#ff3333" if roi.label == BLACK_LABEL else "#00d7ff"
            self.canvas.create_rectangle(sx0, sy0, sx1, sy1, outline=color, width=2)
            self.canvas.create_text(
                sx0 + 5,
                sy0 + 5,
                text=roi.label.upper(),
                fill=color,
                anchor="nw",
                font=("TkDefaultFont", 10, "bold"),
            )

        # Drag ROI.
        if self.drag_start_img and self.drag_current_img:
            x0i, y0i = self.drag_start_img
            x1i, y1i = self.drag_current_img
            sx0, sy0 = self.image_to_screen(x0i, y0i)
            sx1, sy1 = self.image_to_screen(x1i, y1i)
            color = "#ff3333" if self.label == BLACK_LABEL else "#00d7ff"
            self.canvas.create_rectangle(sx0, sy0, sx1, sy1, outline=color, width=3, dash=(4, 3))

        self.update_status()
        self.refresh_list()

    # ------------------------------
    # Events
    # ------------------------------

    def on_mode_change(self) -> None:
        self.label = self.mode_var.get()
        self.update_view()

    def set_mode(self, label: str) -> None:
        self.label = label
        self.mode_var.set(label)
        self.update_view()

    def on_left_down(self, event) -> None:
        self.drag_start_img = self.screen_to_image(event.x, event.y)
        self.drag_current_img = self.drag_start_img
        self.update_view()

    def on_left_drag(self, event) -> None:
        if self.drag_start_img is None:
            return
        self.drag_current_img = self.screen_to_image(event.x, event.y)
        self.update_view()

    def on_left_up(self, event) -> None:
        if self.drag_start_img is None:
            return
        end = self.screen_to_image(event.x, event.y)
        roi = normalize_roi(
            self.drag_start_img[0],
            self.drag_start_img[1],
            end[0],
            end[1],
            self.label,
        )
        if roi.x1 > roi.x0 and roi.y1 > roi.y0:
            self.rois.append(roi)
        self.drag_start_img = None
        self.drag_current_img = None
        self.update_view()

    def on_middle_down(self, event) -> None:
        self.pan_start = (event.x, event.y, self.offset_x, self.offset_y)
        self.canvas.configure(cursor="fleur")

    def on_middle_drag(self, event) -> None:
        if self.pan_start is None:
            return
        sx, sy, ox, oy = self.pan_start
        dx = (event.x - sx) / self.zoom
        dy = (event.y - sy) / self.zoom
        self.offset_x = ox - dx
        self.offset_y = oy - dy
        self.update_view()

    def on_middle_up(self, event) -> None:
        self.pan_start = None
        self.canvas.configure(cursor="crosshair")

    def zoom_at(self, factor: float, sx: int, sy: int) -> None:
        before_x, before_y = self.screen_to_image(sx, sy)
        old_zoom = self.zoom
        self.zoom = float(clamp(self.zoom * factor, 0.10, 8.0))
        if self.zoom == old_zoom:
            return
        # Keep cursor point stable.
        self.offset_x = before_x - sx / self.zoom
        self.offset_y = before_y - sy / self.zoom
        self.update_view()

    def on_mousewheel(self, event) -> None:
        state = int(event.state)
        ctrl_or_alt = bool(state & 0x0004) or bool(state & 0x0008) or bool(state & 0x20000)
        shift = bool(state & 0x0001)
        direction = 1 if event.delta > 0 else -1

        if ctrl_or_alt:
            self.zoom_at(1.12 if direction > 0 else 1 / 1.12, event.x, event.y)
            return

        if shift:
            self.offset_x -= direction * 120 / self.zoom
        else:
            self.offset_y -= direction * 420 / self.zoom
        self.update_view()

    def on_wheel_up(self, event) -> None:
        state = int(event.state)
        ctrl_or_alt = bool(state & 0x0004) or bool(state & 0x0008) or bool(state & 0x20000)
        shift = bool(state & 0x0001)
        if ctrl_or_alt:
            self.zoom_at(1.12, event.x, event.y)
        elif shift:
            self.offset_x -= 120 / self.zoom
            self.update_view()
        else:
            self.offset_y -= 420 / self.zoom
            self.update_view()

    def on_wheel_down(self, event) -> None:
        state = int(event.state)
        ctrl_or_alt = bool(state & 0x0004) or bool(state & 0x0008) or bool(state & 0x20000)
        shift = bool(state & 0x0001)
        if ctrl_or_alt:
            self.zoom_at(1 / 1.12, event.x, event.y)
        elif shift:
            self.offset_x += 120 / self.zoom
            self.update_view()
        else:
            self.offset_y += 420 / self.zoom
            self.update_view()

    def undo(self) -> None:
        if self.rois:
            self.rois.pop()
        self.update_view()

    def clear(self) -> None:
        if not self.rois:
            return
        if messagebox.askyesno("Clear ROIs", "Remove all ROIs?"):
            self.rois.clear()
            self.update_view()

    def save_and_close(self) -> None:
        try:
            validate_rois(self.rois)
        except Exception as e:
            messagebox.showerror("ROIs not ready", str(e))
            return
        save_rois(self.output_path, self.rois)
        print(f"Saved ROIs: {self.output_path}")
        self.saved = True
        self.root.destroy()

    def close_without_saving(self) -> None:
        self.saved = False
        self.root.destroy()

    def run(self) -> list[ROI]:
        self.root.mainloop()
        if self.saved:
            return self.rois
        return []


def get_search_profile(priority: str, wide: bool, morph_order: str) -> SearchProfile:
    if morph_order not in {"minmax", "maxmin", "both"}:
        raise ValueError("--morph-order must be minmax, maxmin or both")
    orders = ["minmax", "maxmin"] if morph_order == "both" else [morph_order]
    priority = priority.strip().lower()

    if priority == "black-bg":
        if wide:
            return SearchProfile(
                priority="black-bg",
                black_weight=4.0,
                white_weight=5.0,
                channels=["grayscale", "min RGB", "max RGB", "value", "saturation"],
                level_black_values=list(range(0, 91, 3)),
                level_white_values=list(range(30, 226, 5)),
                gamma_x100_values=[25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150, 180],
                threshold_values=[8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160],
                min_radius_values=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24],
                max_radius_values=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24],
                morph_orders=orders,
            )
        return SearchProfile(
            priority="black-bg",
            black_weight=4.0,
            white_weight=5.0,
            channels=["grayscale", "max RGB", "value", "saturation"],
            level_black_values=[0, 6, 12, 18, 24, 30, 36, 42, 50, 60, 72],
            level_white_values=[40, 55, 70, 85, 100, 120, 140, 170, 200],
            gamma_x100_values=[30, 40, 50, 60, 70, 80, 100, 120, 150],
            threshold_values=[8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128],
            min_radius_values=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
            max_radius_values=[0, 1, 2, 3, 4, 6, 8, 10, 12, 16],
            morph_orders=orders,
        )

    if priority == "hard":
        return SearchProfile(
            priority="hard",
            black_weight=4.0,
            white_weight=1.5,
            channels=["grayscale", "min RGB", "value"],
            level_black_values=[18, 20, 22, 24, 26, 28, 30, 34, 38, 42],
            level_white_values=[70, 80, 90, 100, 110, 120, 130, 150],
            gamma_x100_values=[30, 35, 40, 45, 48, 55, 65, 80],
            threshold_values=[20, 24, 28, 29, 32, 36, 40, 48, 56, 64],
            min_radius_values=[10, 12, 14, 16, 18, 20, 22],
            max_radius_values=[10, 12, 14, 16, 18, 20, 22],
            morph_orders=orders,
        )

    if priority == "soft":
        return SearchProfile(
            priority="soft",
            black_weight=1.0,
            white_weight=4.0,
            channels=["grayscale", "max RGB", "value"],
            level_black_values=[10, 14, 18, 22, 26, 30, 34, 38, 42],
            level_white_values=[90, 110, 130, 150, 170, 190, 210, 230],
            gamma_x100_values=[60, 80, 100, 120, 150],
            threshold_values=[48, 64, 80, 96, 112, 128, 141, 160],
            min_radius_values=[0, 2, 4, 6, 8, 10],
            max_radius_values=[0, 2, 4, 6, 8, 10],
            morph_orders=orders,
        )

    raise ValueError("--priority must be hard, soft or black-bg")


def apply_channel(rgb: np.ndarray, mode: str) -> np.ndarray:
    if mode == "grayscale":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    if mode == "min RGB":
        return np.minimum(np.minimum(r, g), b)
    if mode == "max RGB":
        return np.maximum(np.maximum(r, g), b)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if mode == "value":
        return hsv[:, :, 2]
    if mode == "saturation":
        return hsv[:, :, 1]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def apply_levels(channel: np.ndarray, black: int, white: int, gamma_x100: int) -> np.ndarray:
    black = int(clamp(black, 0, 254))
    white = int(clamp(white, black + 1, 255))
    gamma = max(0.05, gamma_x100 / 100.0)
    x = channel.astype(np.float32)
    x = (x - black) / (white - black)
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, 1.0 / gamma)
    return (x * 255.0).astype(np.uint8)


def threshold_mask(levels_channel: np.ndarray, threshold_value: int) -> np.ndarray:
    return np.where(levels_channel < threshold_value, 0, 255).astype(np.uint8)


def apply_minimum(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Photoshop Minimum expands dark areas. On 0/255 masks this is erosion.
    return cv2.erode(mask, kernel, iterations=1)


def apply_maximum(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Photoshop Maximum expands light areas. On 0/255 masks this is dilation.
    return cv2.dilate(mask, kernel, iterations=1)


def apply_morphology(mask: np.ndarray, min_radius: int, max_radius: int, morph_order: str) -> np.ndarray:
    if morph_order == "minmax":
        return apply_maximum(apply_minimum(mask, min_radius), max_radius)
    if morph_order == "maxmin":
        return apply_minimum(apply_maximum(mask, max_radius), min_radius)
    raise ValueError(f"Unknown morph_order: {morph_order}")


def build_tonal_mask(rgb, channel_mode, black, white, gamma_x100, threshold_value) -> np.ndarray:
    channel = apply_channel(rgb, channel_mode)
    levels = apply_levels(channel, black, white, gamma_x100)
    return threshold_mask(levels, threshold_value)


def build_final_mask(rgb, channel_mode, black, white, gamma_x100, threshold_value, min_radius, max_radius, morph_order) -> np.ndarray:
    mask = build_tonal_mask(rgb, channel_mode, black, white, gamma_x100, threshold_value)
    return apply_morphology(mask, min_radius, max_radius, morph_order)


def crop_for_roi(rgb: np.ndarray, roi: ROI, pad: int = 0):
    h, w = rgb.shape[:2]
    x0 = max(0, roi.x0 - pad)
    y0 = max(0, roi.y0 - pad)
    x1 = min(w, roi.x1 + pad)
    y1 = min(h, roi.y1 + pad)
    crop = rgb[y0:y1, x0:x1].copy()
    local = (roi.x0 - x0, roi.y0 - y0, roi.x1 - x0, roi.y1 - y0)
    return crop, local


def make_preview_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.70) -> np.ndarray:
    out = rgb.copy()
    deleted = mask < 128
    red = np.zeros_like(out)
    red[:, :, 0] = 255
    out[deleted] = (out[deleted].astype(np.float32) * (1.0 - alpha) + red[deleted].astype(np.float32) * alpha).astype(np.uint8)
    return out


def score_mask_patch(mask_patch: np.ndarray, label: str) -> float:
    if mask_patch.size == 0:
        return 1.0
    if label == BLACK_LABEL:
        # Should be black/deleted.
        return float((mask_patch > 127).mean())
    if label == WHITE_LABEL:
        # Should be white/kept.
        return float((mask_patch < 128).mean())
    return 1.0


def calculate_weighted_score(black_errors, white_errors, profile: SearchProfile):
    black_error = float(np.mean(black_errors)) if black_errors else 0.0
    white_error = float(np.mean(white_errors)) if white_errors else 0.0
    score = black_error * profile.black_weight + white_error * profile.white_weight
    return score, black_error, white_error


def evaluate_tonal_params(rgb, rois, profile, channel, black, white, gamma_x100, threshold_value) -> TonalParams:
    black_errors, white_errors = [], []
    for roi in rois:
        crop, local = crop_for_roi(rgb, roi, pad=0)
        lx0, ly0, lx1, ly1 = local
        mask = build_tonal_mask(crop, channel, black, white, gamma_x100, threshold_value)
        err = score_mask_patch(mask[ly0:ly1, lx0:lx1], roi.label)
        if roi.label == BLACK_LABEL:
            black_errors.append(err)
        else:
            white_errors.append(err)
    score, be, we = calculate_weighted_score(black_errors, white_errors, profile)
    return TonalParams(channel, black, white, gamma_x100, threshold_value, score, be, we)


def evaluate_final_params(rgb, rois, profile, tonal, min_radius, max_radius, morph_order) -> FinalParams:
    black_errors, white_errors = [], []
    for roi in rois:
        crop, local = crop_for_roi(rgb, roi, pad=ROI_PAD)
        lx0, ly0, lx1, ly1 = local
        mask = build_final_mask(crop, tonal.channel, tonal.black, tonal.white, tonal.gamma_x100, tonal.threshold, min_radius, max_radius, morph_order)
        err = score_mask_patch(mask[ly0:ly1, lx0:lx1], roi.label)
        if roi.label == BLACK_LABEL:
            black_errors.append(err)
        else:
            white_errors.append(err)
    score, be, we = calculate_weighted_score(black_errors, white_errors, profile)
    return FinalParams(tonal.channel, tonal.black, tonal.white, tonal.gamma_x100, tonal.threshold, min_radius, max_radius, morph_order, score, be, we)


def search_stage1_tonal(rgb, rois, profile) -> list[TonalParams]:
    combos = []
    for channel in profile.channels:
        for black in profile.level_black_values:
            for white in profile.level_white_values:
                if white <= black + 1:
                    continue
                for gamma_x100 in profile.gamma_x100_values:
                    for threshold in profile.threshold_values:
                        combos.append((channel, black, white, gamma_x100, threshold))
    total = len(combos)
    results = []
    print(f"\nStage 1: Levels + Threshold [{profile.priority}], combinations={total}")
    start = time.time()
    for i, args in enumerate(combos, 1):
        p = evaluate_tonal_params(rgb, rois, profile, *args)
        results.append(p)
        if i % 250 == 0 or i == total:
            best = min(results, key=lambda x: x.score)
            print(f"[stage1 {i}/{total}] {i/total*100:.1f}% elapsed={time.time()-start:.1f}s best={best.score:.5f} {best.channel} L {best.black}/{best.gamma_x100/100:.2f}/{best.white} T{best.threshold} black={best.black_error:.3f} white={best.white_error:.3f}", flush=True)
    results.sort(key=lambda x: x.score)
    return results


def search_stage2_morphology(rgb, rois, profile, tonal_results, top_tonal) -> list[FinalParams]:
    tonal_candidates = tonal_results[:top_tonal]
    combos = []
    for tonal in tonal_candidates:
        for min_radius in profile.min_radius_values:
            for max_radius in profile.max_radius_values:
                for order in profile.morph_orders:
                    combos.append((tonal, min_radius, max_radius, order))
    total = len(combos)
    results = []
    print(f"\nStage 2: Morphology, candidates={len(tonal_candidates)}, orders={profile.morph_orders}, combinations={total}")
    start = time.time()
    for i, (tonal, min_radius, max_radius, order) in enumerate(combos, 1):
        p = evaluate_final_params(rgb, rois, profile, tonal, min_radius, max_radius, order)
        results.append(p)
        if i % 100 == 0 or i == total:
            best = min(results, key=lambda x: x.score)
            print(f"[stage2 {i}/{total}] {i/total*100:.1f}% elapsed={time.time()-start:.1f}s best={best.score:.5f} {best.channel} L {best.black}/{best.gamma_x100/100:.2f}/{best.white} T{best.threshold} Min{best.min_radius} Max{best.max_radius} {best.morph_order} black={best.black_error:.3f} white={best.white_error:.3f}", flush=True)
    results.sort(key=lambda x: x.score)
    return results


def save_stage1_csv(out_dir: Path, results: list[TonalParams]) -> None:
    with (out_dir / "stage1_tonal_top.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "score", "black_error", "white_error", "channel", "levels_black", "levels_white", "gamma", "threshold"])
        for rank, p in enumerate(results[:TOP_N], 1):
            wr.writerow([rank, f"{p.score:.8f}", f"{p.black_error:.8f}", f"{p.white_error:.8f}", p.channel, p.black, p.white, f"{p.gamma_x100/100:.2f}", p.threshold])


def save_final_csv(out_dir: Path, results: list[FinalParams]) -> None:
    with (out_dir / "top_results.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "score", "black_error", "white_error", "channel", "levels_black", "levels_white", "gamma", "threshold", "minimum_radius", "maximum_radius", "morph_order"])
        for rank, p in enumerate(results[:TOP_N], 1):
            wr.writerow([rank, f"{p.score:.8f}", f"{p.black_error:.8f}", f"{p.white_error:.8f}", p.channel, p.black, p.white, f"{p.gamma_x100/100:.2f}", p.threshold, p.min_radius, p.max_radius, p.morph_order])


def save_best_settings(out_dir: Path, best: FinalParams, profile: SearchProfile) -> None:
    with (out_dir / "best_settings.txt").open("w", encoding="utf-8") as f:
        f.write(f"priority: {profile.priority}\n")
        f.write(f"black_weight: {profile.black_weight}\n")
        f.write(f"white_weight: {profile.white_weight}\n")
        f.write(f"channel: {best.channel}\n")
        f.write(f"levels_black: {best.black}\n")
        f.write(f"levels_white: {best.white}\n")
        f.write(f"gamma: {best.gamma_x100/100:.2f}\n")
        f.write(f"threshold: {best.threshold}\n")
        f.write(f"minimum_radius: {best.min_radius}\n")
        f.write(f"maximum_radius: {best.max_radius}\n")
        f.write(f"morph_order: {best.morph_order}\n")
        f.write(f"score: {best.score:.8f}\n")
        f.write(f"black_error: {best.black_error:.8f}\n")
        f.write(f"white_error: {best.white_error:.8f}\n")


def _safe_jpeg_save(img: Image.Image, path: Path, quality: int = 92) -> None:
    """
    Save JPEG safely.
    JPEG encoders commonly fail when one dimension is too large.
    This function prevents the whole optimization run from crashing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    max_dim = 60000
    w, h = img.size

    if w > max_dim or h > max_dim:
        scale = min(max_dim / w, max_dim / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img.save(path, quality=quality)


def _build_contact_page(
    rgb: np.ndarray,
    rois: list[ROI],
    params: FinalParams,
    rank: int,
    title_suffix: str,
    cell_w: int = 300,
    cell_h: int = 230,
    label_h: int = 96,
    pad: int = 12,
    cols: int = 4,
) -> Image.Image:
    rows = max(1, math.ceil(len(rois) / cols))

    header_h = 78
    sheet_w = cols * cell_w + (cols + 1) * pad
    sheet_h = header_h + rows * (cell_h + label_h) + (rows + 1) * pad

    sheet = Image.new("RGB", (sheet_w, sheet_h), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            15,
        )
    except Exception:
        font = ImageFont.load_default()

    title = (
        f"Rank #{rank} | {title_suffix}\n"
        f"score={params.score:.4f} black={params.black_error:.3f} white={params.white_error:.3f} | "
        f"{params.channel} L {params.black}/{params.gamma_x100 / 100:.2f}/{params.white} "
        f"T{params.threshold} Min{params.min_radius} Max{params.max_radius} {params.morph_order}"
    )

    draw.text((pad, pad), title, font=font, fill=(255, 255, 255))

    for idx, roi in enumerate(rois):
        row_i = idx // cols
        col_i = idx % cols

        crop, _ = crop_for_roi(rgb, roi, pad=ROI_PAD)

        mask = build_final_mask(
            crop,
            params.channel,
            params.black,
            params.white,
            params.gamma_x100,
            params.threshold,
            params.min_radius,
            params.max_radius,
            params.morph_order,
        )

        preview = Image.fromarray(make_preview_overlay(crop, mask, 0.70))
        preview.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

        cell = Image.new("RGB", (cell_w, cell_h + label_h), (0, 0, 0))
        cell_draw = ImageDraw.Draw(cell)

        label_text = (
            f"{roi.label.upper()} | x {roi.x0}-{roi.x1} y {roi.y0}-{roi.y1}\n"
            f"#{rank} score={params.score:.4f}\n"
            f"black={params.black_error:.3f} white={params.white_error:.3f}\n"
            f"{params.channel} T{params.threshold} {params.morph_order}"
        )

        cell_draw.text((6, 5), label_text, font=font, fill=(255, 255, 255))
        cell.paste(preview, ((cell_w - preview.width) // 2, label_h))

        x = pad + col_i * (cell_w + pad)
        y = header_h + pad + row_i * (cell_h + label_h + pad)
        sheet.paste(cell, (x, y))

    return sheet


def _sample_rois_evenly(rois: list[ROI], max_count: int) -> list[ROI]:
    if len(rois) <= max_count:
        return rois[:]

    # Keep vertical distribution stable. This is better than only taking the first N.
    indexes = np.linspace(0, len(rois) - 1, max_count).round().astype(int).tolist()
    seen = set()
    sampled = []
    for i in indexes:
        if i in seen:
            continue
        seen.add(i)
        sampled.append(rois[i])
    return sampled


def save_contact_sheet(out_dir: Path, rgb: np.ndarray, rois: list[ROI], results: list[FinalParams]) -> None:
    """
    Safe contact-sheet writer.

    Old behavior created one enormous JPG:
      rows = top10 results * all ROIs
    With 360 boundary ROIs this exceeded JPEG dimension limits and crashed after
    the full search had already finished.

    New behavior:
      1. Always saves a small overview as top10_contact_sheet.jpg.
      2. Saves detailed contact sheets as multiple pages in contact_pages/.
      3. Never lets contact-sheet generation crash the completed search.
    """
    if not results or not rois:
        return

    params_list = results[:10]
    contact_dir = out_dir / "contact_pages"
    contact_dir.mkdir(parents=True, exist_ok=True)

    # 1) Small overview file with sampled ROIs for each top result.
    # This keeps the old expected filename, but avoids huge dimensions.
    try:
        overview_rois = _sample_rois_evenly(rois, 32)
        overview_pages = []

        for rank, params in enumerate(params_list, start=1):
            page = _build_contact_page(
                rgb=rgb,
                rois=overview_rois,
                params=params,
                rank=rank,
                title_suffix=f"overview sample | {len(overview_rois)} of {len(rois)} ROIs",
                cell_w=280,
                cell_h=200,
                label_h=90,
                cols=4,
            )
            overview_pages.append(page)

        # Stack overview pages vertically, but still keep it below safe size.
        if overview_pages:
            total_w = max(p.width for p in overview_pages)
            total_h = sum(p.height for p in overview_pages)

            # If still too high, save separate overview pages and create rank1 as main sheet.
            if total_h <= 60000:
                sheet = Image.new("RGB", (total_w, total_h), (25, 25, 25))
                y = 0
                for page in overview_pages:
                    sheet.paste(page, (0, y))
                    y += page.height
                _safe_jpeg_save(sheet, out_dir / "top10_contact_sheet.jpg", quality=92)
            else:
                _safe_jpeg_save(overview_pages[0], out_dir / "top10_contact_sheet.jpg", quality=92)
                for rank, page in enumerate(overview_pages, start=1):
                    _safe_jpeg_save(
                        page,
                        contact_dir / f"overview_rank{rank:02d}.jpg",
                        quality=92,
                    )

    except Exception as e:
        print(f"WARNING: overview contact sheet failed: {e}")

    # 2) Detailed paged sheets.
    # 40 ROIs per page keeps image dimensions safe and reviewable.
    rois_per_page = 40

    try:
        page_count = 0

        for rank, params in enumerate(params_list, start=1):
            for start in range(0, len(rois), rois_per_page):
                page_rois = rois[start:start + rois_per_page]
                end = start + len(page_rois)

                page = _build_contact_page(
                    rgb=rgb,
                    rois=page_rois,
                    params=params,
                    rank=rank,
                    title_suffix=f"ROIs {start + 1}-{end} of {len(rois)}",
                    cell_w=300,
                    cell_h=230,
                    label_h=96,
                    cols=4,
                )

                page_path = contact_dir / f"contact_rank{rank:02d}_rois{start + 1:04d}-{end:04d}.jpg"
                _safe_jpeg_save(page, page_path, quality=92)
                page_count += 1

        print(f"Saved contact sheets: {page_count} pages in {contact_dir}")

    except Exception as e:
        # Do not kill the already finished search.
        print(f"WARNING: detailed contact sheets failed: {e}")

def save_all(image_path: Path, rgb, rois, stage1, final, profile) -> None:
    out_dir = Path(f"{image_path.stem}_{profile.priority}_params")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rois(out_dir / "used_rois.txt", rois)
    save_stage1_csv(out_dir, stage1)
    save_final_csv(out_dir, final)
    save_best_settings(out_dir, final[0], profile)
    save_contact_sheet(out_dir, rgb, rois, final)
    b = final[0]
    print("\nBest final settings:")
    print(f"  priority: {profile.priority}")
    print(f"  channel: {b.channel}")
    print(f"  levels: {b.black} / {b.gamma_x100/100:.2f} / {b.white}")
    print(f"  threshold: {b.threshold}")
    print(f"  minimum_radius: {b.min_radius}")
    print(f"  maximum_radius: {b.max_radius}")
    print(f"  morph_order: {b.morph_order}")
    print(f"  score: {b.score:.8f}")
    print(f"  black_error: {b.black_error:.8f}")
    print(f"  white_error: {b.white_error:.8f}")
    print(f"\nSaved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto finder for Levels + Threshold + Minimum/Maximum with black-background mode and ROI marking.")
    parser.add_argument("image", help="Input image, e.g. 033-1.png")
    parser.add_argument("--rois", default=None, help="ROI file. If omitted, ROI marker opens first.")
    parser.add_argument("--mark-rois", action="store_true", help="Force interactive ROI marking before search.")
    parser.add_argument("--roi-output", default=None, help="Where to save marked ROIs. Default: used_rois_<image>.txt")
    parser.add_argument("--priority", choices=["hard", "soft", "black-bg"], default="black-bg", help="Default: black-bg")
    parser.add_argument("--morph-order", choices=["minmax", "maxmin", "both"], default="both", help="minmax=Minimum then Maximum; maxmin=Maximum then Minimum; both=test both. Default: both")
    parser.add_argument("--wide", action="store_true", help="Wider and slower search")
    parser.add_argument("--top-tonal", type=int, default=30, help="Best tonal candidates for morphology search")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.mark_rois or args.rois is None:
        roi_path = Path(args.roi_output) if args.roi_output else Path(f"used_rois_{image_path.stem}.txt")
        marker = RoiMarker(image_path, roi_path)
        rois = marker.run()
        if roi_path.exists():
            rois = load_rois(roi_path)
    else:
        roi_path = Path(args.rois)
        if not roi_path.exists():
            raise FileNotFoundError(f"ROI file not found: {roi_path}")
        rois = load_rois(roi_path)

    validate_rois(rois)

    rgb = load_rgb(image_path)
    profile = get_search_profile(args.priority, args.wide, args.morph_order)

    print(f"Image: {image_path}")
    print(f"Size: {rgb.shape[1]}x{rgb.shape[0]}")
    print(f"ROIs: {len(rois)}")
    print(f"BLACK/remove: {sum(1 for r in rois if r.label == BLACK_LABEL)}")
    print(f"WHITE/keep:   {sum(1 for r in rois if r.label == WHITE_LABEL)}")
    print(f"Priority: {profile.priority}")
    print(f"Weights: BLACK={profile.black_weight}, WHITE={profile.white_weight}")
    print(f"Morph order search: {profile.morph_orders}")
    print(f"Mode: {'WIDE' if args.wide else 'FAST'}")

    stage1 = search_stage1_tonal(rgb, rois, profile)
    final = search_stage2_morphology(rgb, rois, profile, stage1, args.top_tonal)
    save_all(image_path, rgb, rois, stage1, final, profile)


if __name__ == "__main__":
    main()
