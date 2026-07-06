"""
Build a video comparing model versions side by side.

Source data: data/chapters-results/{version}-{chapter}_result_red_preview.png
in this repo -- each file is one model version's full chapter rendered as a
single tall vertical strip (all pages stacked). This script crops a handful
of horizontal slices out of that strip, tiles each slice's per-version crops
into a row (labeled with the version tag), and writes a video that holds on
each slice in turn so each shot is "all models, same spot".

Picking where those slices come from is manual, not evenly auto-spaced --
run with --contact-sheet first to get a scrollable index of y-offsets, pick
the interesting ones by eye, then pass them with --y-offsets.

Output goes to data/compare/ (gitignored) by default.

Usage (run from repo root):
  # 1) generate an index of thumbnails with their y-offset labeled
  python3 src/compare_models_video.py --contact-sheet

  # 2) look at data/compare/contact_sheet.png, then build the video from chosen spots
  python3 src/compare_models_video.py --y-offsets 12000 48000 91000 130000

  # 3) also write a compressed, capped-width copy for easier sharing
  python3 src/compare_models_video.py --y-offsets 12000 48000 91000 130000 --compressed
"""
import argparse
import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None  # these are large but trusted, locally-generated strips

DEFAULT_DATA_DIR = Path("data/chapters-results")
DEFAULT_OUT_DIR = Path("data/compare")
# Matches both the plain "v5.0-075_result_red_preview.png" and a variant
# with an infix like "v5.0-islands-075_result_red_preview.png" -- e.g. the
# same checkpoint's output with --reclaim-islands postprocessing applied,
# saved under a distinct name so it doesn't overwrite the plain run. The
# infix (if present) becomes part of the displayed tag ("v5.0-islands"),
# not a separate model version.
FNAME_RE = re.compile(r"^(v\d+(?:\.\d+)*)(-[a-zA-Z][a-zA-Z0-9]*)?-(\d+)_result_red_preview\.png$")

LABEL_H = 90
GAP = 12
BG = (30, 30, 30)
FG = (255, 255, 255)
FONT_SIZE = 64

# Truthful, boring reason this needs a search list at all: PIL's
# ImageFont.truetype("name.ttf", ...) only resolves bare names against
# fontconfig on systems that have one set up for it; a bundled path is the
# reliable fallback everywhere else (this box has no ~/.fonts DejaVu alias).
FONT_CANDIDATES = [
    "/usr/lib/python3.14/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
]


def version_key(tag):
    """Sort by numeric version first, then by infix (plain "" sorts before
    any suffix like "-islands") so e.g. v5.0, v5.0-islands, v6.0,
    v6.0-islands come out in that order rather than alphabetically."""
    base, _, suffix = tag.partition("-")
    return (tuple(int(p) for p in base.lstrip("v").split(".")), suffix)


def discover(data_dir, chapter=None):
    """Return {chapter_id: [(version_tag, path), ...]} sorted by version.
    version_tag includes any infix, e.g. "v5.0" or "v5.0-islands"."""
    chapters = {}
    for p in sorted(data_dir.glob("*_result_red_preview.png")):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        version, suffix, ch = m.groups()
        tag = version + (suffix or "")
        if chapter and ch != chapter:
            continue
        chapters.setdefault(ch, []).append((tag, p))
    for ch in chapters:
        chapters[ch].sort(key=lambda t: version_key(t[0]))
    return chapters


def label_font(size=FONT_SIZE):
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def make_contact_sheet(entries, out_path, n_tiles=40, tile_w=260, cols=5):
    """Chop the first version's strip into n_tiles labeled thumbnails (with
    their y-offset) tiled into a grid, so you can pick real --y-offsets by
    eye instead of guessing blind against a 100k+px-tall image."""
    tag, path = entries[0]
    img = Image.open(path).convert("RGB")
    w, h = img.size
    tile_h_src = h // n_tiles
    font = label_font(20)

    thumbs = []
    for i in range(n_tiles):
        y0 = i * tile_h_src
        y1 = h if i == n_tiles - 1 else y0 + tile_h_src
        crop = img.crop((0, y0, w, y1))
        scale = tile_w / w
        thumb = crop.resize((tile_w, max(1, int((y1 - y0) * scale))))
        labeled = Image.new("RGB", (tile_w, thumb.height + 26), BG)
        labeled.paste(thumb, (0, 26))
        ImageDraw.Draw(labeled).text((4, 2), f"y={y0}", fill=FG, font=font)
        thumbs.append(labeled)
    img.close()

    rows = math.ceil(len(thumbs) / cols)
    row_heights = [max(t.height for t in thumbs[r * cols:(r + 1) * cols]) for r in range(rows)]
    sheet = Image.new("RGB", (tile_w * cols, sum(row_heights)), BG)
    y = 0
    for r in range(rows):
        x = 0
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(thumbs):
                break
            sheet.paste(thumbs[idx], (x, y))
            x += tile_w
        y += row_heights[r]
    sheet.save(out_path)
    print(f"Wrote contact sheet ({tag}, {n_tiles} tiles, full height {h}px) to {out_path}")
    print("Pick a few 'y=' values from it and pass them via --y-offsets.")


def build_frames(entries, y_offsets, crop_h):
    """For each requested y-offset, crop that band from every version's
    strip and return a list of {tag: labeled_crop_image}."""
    font = label_font()
    slots = [dict() for _ in y_offsets]

    for tag, path in entries:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        for i, y0 in enumerate(y_offsets):
            y0 = max(0, min(y0, h - 1))
            crop = img.crop((0, y0, w, min(y0 + crop_h, h)))
            labeled = Image.new("RGB", (w, crop_h + LABEL_H), BG)
            labeled.paste(crop, (0, LABEL_H))
            ImageDraw.Draw(labeled).text((10, 10), tag, fill=FG, font=font)
            slots[i][tag] = labeled
        img.close()

    return slots


def compose_rows(slots, entries):
    order = [tag for tag, _ in entries]
    frames = []
    for slot in slots:
        cols = [slot[tag] for tag in order]
        total_w = sum(c.width for c in cols) + GAP * (len(cols) - 1)
        row_h = max(c.height for c in cols)
        row = Image.new("RGB", (total_w, row_h), BG)
        x = 0
        for c in cols:
            row.paste(c, (x, 0))
            x += c.width + GAP
        frames.append(row)
    return frames


def write_video(frames, out_path, fps, hold_sec):
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer for {out_path}")

    hold_count = max(1, round(hold_sec * fps))
    try:
        for frame in frames:
            bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            for _ in range(hold_count):
                writer.write(bgr)
    finally:
        writer.release()

    print(f"Wrote {out_path} ({len(frames)} slices, {w}x{h}, {hold_count} frames/slice @ {fps}fps)")


def write_compressed(in_path, out_path, max_width=1920):
    """
    Re-encode the raw mp4v video to H.264 with a silent audio track and a
    capped width. Several video players/previewers (this started as a fix
    for one such client) only treat a file as an inline-playable video
    rather than a generic document if it has an audio stream and isn't
    absurdly wide -- these comparison videos tile N versions side by side at
    full crop width (e.g. 6306px for an 8-version comparison), far outside
    what a normal video pipeline expects. H.264 + yuv420p + faststart also
    gets far better compression than raw mp4v on video that holds each
    slice static for several seconds.
    """
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.2",
        "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Wrote {out_path} (compressed, max width {max_width}px)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--chapter", default=None, help="chapter id, e.g. 075 (default: first one found)")
    ap.add_argument("--versions", nargs="+", default=None,
                     help="limit to these version tags, e.g. --versions v2.1 v6.0 (default: all found)")
    ap.add_argument("--y-offsets", type=int, nargs="+", default=None,
                     help="pixel y-offset (top of crop) for each slice you want -- pick these from --contact-sheet")
    ap.add_argument("--crop-height", type=int, default=900, help="px height of each sampled slice")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold-sec", type=float, default=3.0, help="seconds each slice is held on screen")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "model_comparison.mp4")
    ap.add_argument("--contact-sheet", action="store_true",
                     help="instead of building the video, dump a labeled thumbnail index to pick --y-offsets from")
    ap.add_argument("--contact-sheet-out", type=Path, default=DEFAULT_OUT_DIR / "contact_sheet.png")
    ap.add_argument("--contact-sheet-tiles", type=int, default=40)
    ap.add_argument("--compressed", action="store_true",
                     help="also write a compressed, capped-width H.264 copy alongside --out "
                          "(suffixed _compressed) for easier sharing")
    ap.add_argument("--max-width", type=int, default=1920, help="width cap used by --compressed")
    args = ap.parse_args()

    chapters = discover(args.data_dir, args.chapter)
    if not chapters:
        raise SystemExit(f"No matching files found in {args.data_dir}")

    chapter_id = args.chapter or sorted(chapters)[0]
    entries = chapters[chapter_id]

    if args.versions:
        wanted = set(args.versions)
        missing = wanted - {tag for tag, _ in entries}
        if missing:
            raise SystemExit(f"Requested version(s) not found for chapter {chapter_id}: {sorted(missing)}")
        entries = [(tag, path) for tag, path in entries if tag in wanted]

    print(f"Chapter {chapter_id}: {len(entries)} model versions -> {[t for t, _ in entries]}")

    if args.contact_sheet:
        args.contact_sheet_out.parent.mkdir(parents=True, exist_ok=True)
        make_contact_sheet(entries, args.contact_sheet_out, n_tiles=args.contact_sheet_tiles)
        return

    if not args.y_offsets:
        raise SystemExit(
            "No --y-offsets given. Run with --contact-sheet first to pick some, "
            "e.g.: --y-offsets 12000 48000 91000 130000"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    slots = build_frames(entries, args.y_offsets, args.crop_height)
    frames = compose_rows(slots, entries)
    write_video(frames, args.out, args.fps, args.hold_sec)

    if args.compressed:
        compressed_out = args.out.with_name(args.out.stem + "_compressed" + args.out.suffix)
        write_compressed(args.out, compressed_out, max_width=args.max_width)


if __name__ == "__main__":
    main()
