"""Calibration sanity check for instance_sfx_net.py's TinyInstanceNet: a pure blank-white
crop (no ink anywhere) should be predicted confidently "delete" (mean prob > 0.5), not just
technically-under-threshold. Added after the first smoke checkpoint (trained on
ink-glyph-centered crops only) showed mean_prob=0.20 on this exact input -- correct side of
the 0.5 decision boundary, but weaker than a clean "delete" case should be, traced to the
crop dataset under-representing plain background (see build_sfx_instance_crops.py's
sample_background_crops docstring for the fix)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_cleaner import GuidanceParams, build_input_tensor  # noqa: E402
from instance_sfx_net import TinyInstanceNet  # noqa: E402

CKPT_DIR = ROOT / ".tmp/checkpoints/instance_sfx_smoke"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["glyph_only", "with_bg", "with_bg_light", "with_bg_weighted"])
    args = ap.parse_args()

    ckpt = torch.load(CKPT_DIR / f"instance_sfx_smoke_{args.variant}.pt", map_location="cpu")
    model = TinyInstanceNet(in_ch=ckpt["in_ch"], base=ckpt["base"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    blank_rgb = np.full((224, 224, 3), 255, dtype=np.uint8)
    inp = build_input_tensor(blank_rgb, GuidanceParams())
    with torch.no_grad():
        x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float()
        probs = torch.sigmoid(model(x).squeeze()).numpy()

    mean_prob = float(probs.mean())
    print(f"[{args.variant}] pure blank crop mean delete-prob: {mean_prob:.4f} "
          f"({'OK, confidently delete' if mean_prob > 0.5 else 'weak/under-confident'})")


if __name__ == "__main__":
    main()
