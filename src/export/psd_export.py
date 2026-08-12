"""PSD layered-mask export: package pipeline detection as Photopea-editable
layers instead of an interactive GUI (2026-08-12 direction).

Per chapter, parts of <= PART_ROWS rows (the user's own proven 50k-part PSD
convention). pytoshop enforces the 30k-row PSD spec limit, so parts are
written as PSB (version 2, limit 300k rows -- Photopea-native):
`<chapter>_touchup-N.psb` with layers (bottom-up):

  base              original art, visible
  pipeline_result   white px + alpha where the automatic delete mask fires
  wall1_semantic .. wall6_pale   white + alpha candidate masks, hidden

wall:1 is px-precise (the measured honest-negative fragment rule recomputed
from delete.npy + rgb); walls 2-6 are bbox fills from zones.json. Toggle a
layer to see its flags over the art; Ctrl-click its thumbnail in Photopea to
load the alpha as a selection; paint into it to adjust.

Writer: pytoshop 1.2.1 with three py3.14/numpy2 compatibility shims (see
_shim_pytoshop): negative constant-channel scalars, the broken RLE
constant path (zip compression avoids it), and the NUL-terminated unicode
layer names. Round-trip verification uses psd-tools (reader only).

Usage: .venv/bin/python src/export/psd_export.py <chapter.png> <sidecar_dir>
       [out_dir=<sidecar_dir>] [name=<chapter>]
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

PART_ROWS = 50000
WALL_LAYERS = {1: "wall1_semantic", 2: "wall2_spiky_remnant",
               3: "wall3_broken_ring", 4: "wall4_ui_card",
               5: "wall5_dark_zone", 6: "wall6_pale_band"}


def _shim_pytoshop():
    import pytoshop.codecs as codecs
    import pytoshop.util as util

    _orig = codecs.compress_image

    def _fixed(fd, image, compression, shape, num_channels, depth, version):
        if np.isscalar(image) and image < 0:
            image = (1 << depth) + image   # numpy2: no negative wraparound
        return _orig(fd, image, compression, shape, num_channels, depth,
                     version)
    codecs.compress_image = _fixed

    def _enc(s):   # drop the NUL terminator pytoshop bakes into layer names
        return struct.pack(">L", len(s)) + s.encode("utf_16_be")
    util.encode_unicode_string = _enc


def build_wall_masks(rgb: np.ndarray, delete: np.ndarray,
                     zones: list[dict]) -> dict[int, np.ndarray]:
    """Per-class boolean masks. wall:1 px-precise; others bbox fills."""
    from classifiers import sfx
    H, W = delete.shape
    masks = {i: np.zeros((H, W), bool) for i in WALL_LAYERS}
    site_boxes = [z["bbox"] for z in zones if z["kind"] == "wall:2"]
    insite = np.zeros((H, W), bool)
    for (x0, y0, x1, y1) in site_boxes:
        insite[y0:y1, x0:x1] = True
    ink = rgb[..., 1] < 100
    cand = (delete & ink & ~insite).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep_ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 30]
    masks[1] = np.isin(lab, keep_ids)
    for z in zones:
        k = z["kind"]
        if not k.startswith("wall:") or k == "wall:1":
            continue
        cls = int(k.split(":")[1])
        x0, y0, x1, y1 = z["bbox"]
        masks[cls][y0:y1, x0:x1] = True
    return masks


def export_chapter(chapter_png: str | Path, sidecar_dir: str | Path,
                   out_dir: str | Path | None = None,
                   name: str | None = None) -> list[Path]:
    _shim_pytoshop()
    from PIL import Image
    from pytoshop import enums
    from pytoshop.user import nested_layers
    Image.MAX_IMAGE_PIXELS = None

    side = Path(sidecar_dir)
    out = Path(out_dir) if out_dir else side
    out.mkdir(parents=True, exist_ok=True)
    name = name or side.name

    rgb = np.array(Image.open(chapter_png).convert("RGB"))
    delete = np.load(side / "delete.npy")
    zones = json.loads((side / "zones.json").read_text())["zones"]
    walls = build_wall_masks(rgb, delete, zones)
    H = rgb.shape[0]

    written = []
    for pi, y0 in enumerate(range(0, H, PART_ROWS), start=1):
        y1 = min(H, y0 + PART_ROWS)

        def mask_layer(lname, m):
            a = (m[y0:y1] * np.uint8(255))
            if not a.any():
                # a wholly-empty layer is dropped by the writer; plant one
                # 1/255-alpha corner px so every part carries the SAME layer
                # set (Photopea layer-panel consistency)
                a = a.copy()
                a[0, 0] = 1
            w = np.full(a.shape, 255, np.uint8)
            return nested_layers.Image(
                name=lname, visible=False,
                channels={0: w, 1: w, 2: w, -1: np.ascontiguousarray(a)})

        layers = [mask_layer(WALL_LAYERS[c], walls[c])
                  for c in sorted(WALL_LAYERS, reverse=True)]
        layers.append(mask_layer("pipeline_result", delete))
        sub = rgb[y0:y1]
        layers.append(nested_layers.Image(
            name="base", visible=True,
            channels={0: np.ascontiguousarray(sub[..., 0]),
                      1: np.ascontiguousarray(sub[..., 1]),
                      2: np.ascontiguousarray(sub[..., 2])}))
        psd = nested_layers.nested_layers_to_psd(
            layers, color_mode=3, compression=enums.Compression.zip,
            version=enums.Version.psb)
        path = out / f"{name}_touchup-{pi}.psb"
        with open(path, "wb") as f:
            psd.write(f)
        written.append(path)
        print(f"wrote {path} rows y{y0}-{y1}", flush=True)
    return written


def verify_roundtrip(chapter_png: str | Path, sidecar_dir: str | Path,
                     parts: list[Path]) -> bool:
    from PIL import Image
    from psd_tools import PSDImage
    Image.MAX_IMAGE_PIXELS = None
    side = Path(sidecar_dir)
    rgb = np.array(Image.open(chapter_png).convert("RGB"))
    delete = np.load(side / "delete.npy")
    zones = json.loads((side / "zones.json").read_text())["zones"]
    walls = build_wall_masks(rgb, delete, zones)
    ok = True
    expect_names = {"base", "pipeline_result", *WALL_LAYERS.values()}
    for pi, path in enumerate(parts, start=1):
        y0 = (pi - 1) * PART_ROWS
        y1 = min(rgb.shape[0], y0 + PART_ROWS)
        p = PSDImage.open(path)
        names = {l.name for l in p}
        if names != expect_names:
            print(f"  {path.name}: LAYER NAMES MISMATCH {names ^ expect_names}")
            ok = False
            continue
        for l in p:
            if l.bbox != (0, 0, rgb.shape[1], y1 - y0):
                print(f"  {path.name}/{l.name}: bbox {l.bbox} != full part")
                ok = False
            arr = l.numpy()   # float32 [0,1], RGB(A)
            if l.name == "base":
                got = (arr[..., :3] * 255).round().astype(np.uint8)
                if not np.array_equal(got, rgb[y0:y1]):
                    print(f"  {path.name}/base: PIXELS DIFFER"); ok = False
            else:
                a = (arr[..., -1] * 255).round().astype(np.uint8) > 127
                # the planted 1/255 corner px thresholds to False -- no
                # special-casing needed
                ref = delete[y0:y1] if l.name == "pipeline_result" else \
                    walls[[k for k, v in WALL_LAYERS.items()
                           if v == l.name][0]][y0:y1]
                if not np.array_equal(a, ref):
                    d = int((a != ref).sum())
                    print(f"  {path.name}/{l.name}: ALPHA DIFFERS {d} px")
                    ok = False
        print(f"  {path.name}: OK" if ok else f"  {path.name}: FAIL")
    return ok


if __name__ == "__main__":
    png, side = sys.argv[1], sys.argv[2]
    outd = sys.argv[3] if len(sys.argv) > 3 else None
    parts = export_chapter(png, side, outd)
    good = verify_roundtrip(png, side, parts)
    print("ROUNDTRIP:", "ALL PASS" if good else "FAIL")
    sys.exit(0 if good else 1)
