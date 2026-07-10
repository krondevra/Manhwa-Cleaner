"""
Build a video comparing model versions side by side.

Source data: data/chapters-results/{version}-{chapter}_result_red_preview.png
in this repo -- each file is one model version's full chapter rendered as a
single tall vertical strip (all pages stacked). This script crops a handful
of horizontal slices out of that strip, tiles each slice's per-version crops
into a row (labeled with the version tag), and writes a video that holds on
each slice in turn so each shot is "all models, same spot". The original,
uncleaned source chapter (data/chapters-initial/{chapter}.png) is prepended
as a leading "original" column when present, so each shot reads as "before,
then every model's after".

Picking where those slices come from is manual, not evenly auto-spaced --
run with --contact-sheet first to get a full continuous strip with a
Y-coordinate ruler alongside it, read off the offsets you want, then pass
them with --y-offsets.

Output goes to data/compare/ (gitignored) by default.

Usage (run from repo root):
  # 1) generate a ruler-labeled strip to read y-offsets from
  python3 src/compare_models_video.py --contact-sheet

  # 2) look at data/compare/contact_sheet.png, then build the video from chosen spots
  python3 src/compare_models_video.py --y-offsets 12000 48000 91000 130000

  # 3) also write a compressed, capped-width copy for easier sharing
  python3 src/compare_models_video.py --y-offsets 12000 48000 91000 130000 --compressed

  # 4) also dump each slice as its own PNG (one per --y-offsets entry) --
  #    handy when you have many coordinates and want to flip through stills
  #    rather than scrub a video
  python3 src/compare_models_video.py --y-offsets 12000 48000 91000 130000 --screenshots
"""
import argparse
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None  # these are large but trusted, locally-generated strips

DEFAULT_DATA_DIR = Path("data/chapters-results")
DEFAULT_CHAPTERS_INITIAL_DIR = Path("data/chapters-initial")
DEFAULT_OUT_DIR = Path("data/compare")
ORIGINAL_TAG = "original"
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


def make_ruler_strip(entries, out_path, label_interval=1000, major_every=5, ruler_w=150):
    """
    Full continuous strip of the first version -- not chopped into a grid --
    with Y-coordinate ruler ticks/labels down the left margin every
    label_interval px (a bolder tick + full-width line every major_every
    labels). Chopping into a shuffled grid of separate thumbnails (the old
    behavior) breaks continuity right where you need it most -- you can't
    tell where one tile ends and the next begins, or see context across a
    tile boundary. This instead reads exactly like the source page, just
    with a precise ruler alongside it, so picking accurate --y-offsets is a
    matter of scrolling and reading a number, not guessing from a tile index.
    """
    tag, path = entries[0]
    img = Image.open(path).convert("RGB")
    w, h = img.size
    minor_font = label_font(30)
    major_font = label_font(38)

    canvas = Image.new("RGB", (w + ruler_w, h), BG)
    canvas.paste(img, (ruler_w, 0))
    img.close()
    draw = ImageDraw.Draw(canvas)

    n_labels = 0
    y = 0
    while y < h:
        is_major = (y // label_interval) % major_every == 0
        if is_major:
            draw.line([(0, y), (w + ruler_w, y)], fill=(255, 210, 0), width=1)
            draw.line([(ruler_w - 30, y), (ruler_w, y)], fill=(255, 210, 0), width=3)
            draw.text((4, min(h - 26, max(0, y - 16))), str(y), fill=(255, 230, 80), font=major_font)
        else:
            draw.line([(ruler_w - 20, y), (ruler_w, y)], fill=(180, 180, 180), width=1)
            draw.text((4, min(h - 22, max(0, y - 12))), str(y), fill=(190, 190, 190), font=minor_font)
        n_labels += 1
        y += label_interval

    canvas.save(out_path)
    print(f"Wrote ruler strip ({tag}, {h}px tall, {n_labels} labels every {label_interval}px, "
          f"bold every {major_every * label_interval}px) to {out_path}")
    print("Scroll through it and read off 'y=' values to pass via --y-offsets.")


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
    """
    Writes MJPG-in-.avi rather than mp4v-in-.mp4: mp4v (this ffmpeg build's
    only software mpeg4 encoder, no libx264) hard-fails with "dimensions too
    large for MPEG-4" once enough model versions are tiled side by side
    (hit at 12 versions / 8412px wide -- more versions only ever grows this
    total width over time, so the raw writer needs a codec without that
    ceiling; --compressed still produces a normal, widely-playable .mp4 via
    the system ffmpeg binary's libx264, unaffected by this).
    """
    w, h = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
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


def write_screenshots(frames, y_offsets, out_dir):
    """Save each composed slice (one row = every version at one y-offset) as
    its own PNG, named with its y-offset so many coordinates stay easy to
    tell apart without scrubbing through the video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame, y0 in zip(frames, y_offsets):
        path = out_dir / f"slice_y{y0}.png"
        frame.save(path)
        paths.append(path)
    print(f"Wrote {len(paths)} screenshot(s) to {out_dir}")


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
    ap.add_argument("--chapters-initial-dir", type=Path, default=DEFAULT_CHAPTERS_INITIAL_DIR,
                     help="where to look for the original, uncleaned {chapter}.png "
                          "(default: data/chapters-initial) -- prepended as a leading "
                          "'original' column when present")
    ap.add_argument("--no-original", action="store_true",
                     help="don't prepend the original uncleaned source image column")
    ap.add_argument("--chapter", default=None, help="chapter id, e.g. 075 (default: first one found)")
    ap.add_argument("--versions", nargs="+", default=None,
                     help="limit to these version tags, e.g. --versions v2.1 v6.0 (default: all found)")
    ap.add_argument("--y-offsets", type=int, nargs="+", default=None,
                     help="pixel y-offset (top of crop) for each slice you want -- pick these from --contact-sheet")
    ap.add_argument("--crop-height", type=int, default=900, help="px height of each sampled slice")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold-sec", type=float, default=3.0, help="seconds each slice is held on screen")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "model_comparison.avi",
                     help="Raw MJPG output path (default: data/compare/model_comparison.avi). "
                          "Use --compressed for a normal, widely-playable .mp4.")
    ap.add_argument("--contact-sheet", action="store_true",
                     help="instead of building the video, dump a full continuous strip with a "
                          "Y-coordinate ruler alongside it, to pick --y-offsets from")
    ap.add_argument("--contact-sheet-out", type=Path, default=DEFAULT_OUT_DIR / "contact_sheet.png")
    ap.add_argument("--contact-sheet-interval", type=int, default=100,
                     help="px spacing between ruler labels (default: 100, e.g. 100 200 300 ... 900 1000)")
    ap.add_argument("--contact-sheet-major-every", type=int, default=10,
                     help="every Nth label is bolded with a full-width line (default: 10, i.e. every 1000px)")
    ap.add_argument("--compressed", action="store_true",
                     help="also write a compressed, capped-width H.264 copy alongside --out "
                          "(suffixed _compressed) for easier sharing")
    ap.add_argument("--max-width", type=int, default=1920, help="width cap used by --compressed")
    ap.add_argument("--screenshots", action="store_true",
                     help="also save each slice (one row = every version at one y-offset) as "
                          "its own PNG -- handy for flipping through many coordinates as "
                          "stills instead of scrubbing the video")
    ap.add_argument("--screenshots-dir", type=Path, default=DEFAULT_OUT_DIR / "screenshots",
                     help="output dir for --screenshots (default: data/compare/screenshots)")
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

    # Prepend the original, uncleaned source chapter as a leading "before"
    # column -- not passed through version_key() (it isn't a version tag),
    # so it's added directly rather than sorted in with the model entries.
    if not args.no_original:
        original_path = args.chapters_initial_dir / f"{chapter_id}.png"
        if original_path.exists():
            entries = [(ORIGINAL_TAG, original_path)] + entries
        else:
            print(f"Note: no original source image at {original_path}, skipping '{ORIGINAL_TAG}' column")

    print(f"Chapter {chapter_id}: {len(entries)} columns -> {[t for t, _ in entries]}")

    if args.contact_sheet:
        args.contact_sheet_out.parent.mkdir(parents=True, exist_ok=True)
        make_ruler_strip(entries, args.contact_sheet_out,
                          label_interval=args.contact_sheet_interval,
                          major_every=args.contact_sheet_major_every)
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

    if args.screenshots:
        write_screenshots(frames, args.y_offsets, args.screenshots_dir)

    if args.compressed:
        # Always .mp4 regardless of --out's extension: write_compressed
        # always produces H.264 + AAC + faststart, which belongs in an mp4
        # container, not whatever raw-writer extension --out happens to use.
        compressed_out = args.out.with_name(args.out.stem + "_compressed.mp4")
        write_compressed(args.out, compressed_out, max_width=args.max_width)


if __name__ == "__main__":
    main()
