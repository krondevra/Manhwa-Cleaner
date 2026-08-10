"""Reusable PSD-only ground-truth extraction (verification round, see chat).

Confirmed against production references (chapters 001, 002) and the 019_5
x-offset anomaly: `.composite()` on a single named layer is the only extraction
path that exactly reproduces the already-validated *_cleaned.png / *_etalon.png
files -- `layer.topil()` and manual `layer.mask` decoding both silently drop
information (the layer's own non-destructive mask, and/or its canvas offset)
and do NOT match.

Usage: point LAYER_NAME at whichever layer in the PSD carries both the real
artwork pixels and a non-destructive layer mask marking keep/delete (named
'img' or 'Background' in the reference sets seen so far).
"""
from __future__ import annotations

import numpy as np
from psd_tools import PSDImage
from psd_tools.api.layers import Layer

MAX_CHUNK = 25000  # composite()'s hard per-axis limit is 30000px; stay safely under it


def extract_original_and_etalon(layer: Layer) -> tuple[np.ndarray, np.ndarray]:
    """Returns (original_rgb, delete_mask) for one PSD layer.

    original_rgb: HxWx3 uint8, the layer's own raw pixel data (mask NOT applied) --
        this is the true pre-clean source art, since a non-destructive PS mask never
        touches the underlying pixels.
    delete_mask: HxW bool, True = delete. Derived from the layer's PROPERLY COMPOSITED
        alpha (pixel alpha * layer mask, positioned at the layer's own canvas offset) --
        the only path that matches production ground truth exactly. threshold: alpha<128.
    """
    psd = layer._psd
    W, H = psd.size
    raw_rgb = np.array(layer.topil().convert("RGB"))
    chunks = []
    for y0 in range(0, H, MAX_CHUNK):
        y1 = min(y0 + MAX_CHUNK, H)
        chunks.append(np.array(layer.composite(viewport=(0, y0, W, y1)).convert("RGBA")))
    alpha = np.concatenate(chunks, axis=0)[..., 3]
    return raw_rgb, alpha < 128
