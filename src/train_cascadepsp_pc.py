"""Finetune pretrained CascadePSP on Pepper & Carrot refinement pairs.

Follow-up to the zero-shot CascadePSP probe (docs/ml_strategy_history.md,
search "CascadePSP"; src/probe_cascadepsp.py). Full experiment design in
.tmp/notes/cascadepsp_finetune_plan.md.

Data: data/refinement_pairs/{train,val}/*.jpg + *.png + *.strata.json,
produced by src/export_cascadepsp_pairs.py from data/dataset_split_scaled/
(P&C only -- never real manhwa; policy: docs/ml_strategy_history.md "Core
architecture").

Reuses CascadePSP's own training code (cloned to .tmp/CascadePSP, NOT
committed -- third-party code, fetched via
`git clone https://github.com/hkchengrex/CascadePSP.git .tmp/CascadePSP`)
for the model (models/psp/pspnet.py::PSPNet), the online IoU-targeted
boundary-perturbation engine (util/boundary_modification.py::modify_boundary
-- generates the coarse "seg" input from a clean GT mask on the fly, so we
never need to synthesize perturbed pairs ourselves), the loss
(util/metrics_compute.py) and the Sobel gradient-loss operator
(models/sobel_op.py, patched: two hardcoded `.cuda()` calls removed --
see cascadepsp_finetune_plan.md Phase 0, this is the only patch needed for
the whole training path to run on CPU or ROCm/ HIP).

Only new code here is StratifiedRefinementDataset: unlike
CascadePSP's own OnlineTransformDataset (which crops uniformly at random),
this centers each 224x224 training crop on one of three content strata:
  - "low_texture": flat/low-local-contrast KEEP interior regions (sky, sea,
    solid fills) -- must learn to be RESTORED, not deleted. This targets
    the zero-shot probe's dominant failure mode directly (it carved these
    regions out as if they were background).
  - "boundary": points along the real keep/delete mask contour (panel
    gutters, bubble/SFX outlines) -- preserves the probe's genuine win
    (stranded gutter keep-specks wiped to match the human etalon).
  - "uniform": ordinary random crop, the remainder.
Ratios configurable via --strata-ratios (default 0.3/0.4/0.3). Per-stratum
pick counts are logged periodically so the intended mix is verifiable in
the actual training stream, not just assumed from the sidecar files.

Usage:
  # pilot: measure real s/step, confirm loss decreases, verify checkpoint round-trip
  python3 src/train_cascadepsp_pc.py --device cuda --steps 400 --pilot-out .tmp/cascadepsp_pilot.pth

  # full finetune (after a pilot confirms feasibility)
  python3 src/train_cascadepsp_pc.py --device cuda --steps 6000 \
      --out data/models/cascadepsp-pc-finetune-1.0.pth
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
CASCADEPSP = REPO_ROOT / ".tmp" / "CascadePSP"
if not CASCADEPSP.is_dir():
    sys.exit(
        f"CascadePSP training code not found at {CASCADEPSP}.\n"
        "Clone it first (third-party code, not committed to this repo):\n"
        "  git clone https://github.com/hkchengrex/CascadePSP.git .tmp/CascadePSP\n"
        "Then apply the CPU/ROCm compat patch to models/sobel_op.py (removes two\n"
        "hardcoded .cuda() calls) -- see .tmp/notes/cascadepsp_finetune_plan.md Phase 0."
    )
sys.path.insert(0, str(CASCADEPSP))

from models.psp.pspnet import PSPNet  # noqa: E402
from models.sobel_op import SobelComputer  # noqa: E402
from util.metrics_compute import compute_loss_and_metrics  # noqa: E402
from util.boundary_modification import modify_boundary  # noqa: E402
from dataset.reseed import reseed  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "refinement_pairs"
CROP = 224
JITTER = 32  # px, random offset applied around a chosen stratum center

im_transform = transforms.Compose([
    transforms.ColorJitter(0.2, 0.05, 0.05, 0),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
gt_transform = transforms.Compose([transforms.ToTensor()])
seg_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class StratifiedRefinementDataset(Dataset):
    def __init__(self, root: Path, strata_ratios: tuple[float, float, float] = (0.3, 0.4, 0.3)):
        self.root = root
        self.ids = sorted(p.stem for p in root.glob("*.jpg"))
        if not self.ids:
            raise FileNotFoundError(f"No .jpg files found in {root}")
        self.ratios = strata_ratios
        self.counts = {"low_texture": 0, "boundary": 0, "uniform": 0}

    def __len__(self) -> int:
        return len(self.ids)

    def _pick_center(self, strata: dict, w: int, h: int) -> tuple[str, int, int]:
        stratum = random.choices(["low_texture", "boundary", "uniform"], weights=self.ratios, k=1)[0]
        points = strata.get(stratum) if stratum != "uniform" else None
        if not points:
            stratum = "uniform"
            cx = random.randint(0, max(0, w - 1))
            cy = random.randint(0, max(0, h - 1))
        else:
            cx, cy = random.choice(points)
            cx += random.randint(-JITTER, JITTER)
            cy += random.randint(-JITTER, JITTER)
        self.counts[stratum] += 1
        return stratum, cx, cy

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        img = Image.open(self.root / f"{stem}.jpg").convert("RGB")
        mask = Image.open(self.root / f"{stem}.png").convert("L")
        strata = json.loads((self.root / f"{stem}.strata.json").read_text())
        w, h = strata["w"], strata["h"]

        _, cx, cy = self._pick_center(strata, w, h)
        left = min(max(0, cx - CROP // 2), max(0, w - CROP))
        top = min(max(0, cy - CROP // 2), max(0, h - CROP))
        img = img.crop((left, top, left + CROP, top + CROP))
        mask = mask.crop((left, top, left + CROP, top + CROP))
        if img.size != (CROP, CROP):  # page smaller than crop on some axis: pad
            img = transforms.functional.pad(img, (0, 0, CROP - img.width, CROP - img.height))
            mask = transforms.functional.pad(mask, (0, 0, CROP - mask.width, CROP - mask.height))

        seed = np.random.randint(2147483647)
        reseed(seed)
        if random.random() < 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        gt_arr = (np.array(mask) > 127).astype("uint8") * 255
        iou_target = np.random.rand() * 0.2 + 0.8  # 0.8-1.0, matches OnlineTransformDataset
        seg_arr = modify_boundary(gt_arr, iou_target=iou_target)

        im_t = im_transform(img)
        gt_t = gt_transform(Image.fromarray(gt_arr))
        seg_t = seg_transform(Image.fromarray(seg_arr))
        return im_t, seg_t, gt_t


def strip_module_prefix(state_dict: dict) -> dict:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def add_module_prefix(state_dict: dict) -> dict:
    """Save with a 'module.' prefix so probe_cascadepsp.py / segmentation_refinement's
    Refiner (which always strips 'module.' on load, matching a DataParallel-saved
    checkpoint) can load this finetuned checkpoint with no code changes."""
    return {f"module.{k}": v for k, v in state_dict.items()}


def load_pretrained_weights(model: torch.nn.Module) -> None:
    from segmentation_refinement.download import download_and_or_check_model_file
    model_folder = Path.home() / ".segmentation-refinement"
    model_folder.mkdir(exist_ok=True)
    model_path = model_folder / "model"
    download_and_or_check_model_file(str(model_path))
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_module_prefix(state))


def main() -> None:
    # Redirected stdout (nohup/background runs) is fully-buffered by default, not
    # line-buffered -- progress prints silently sit unflushed for the entire run
    # (confirmed via a real pilot: CPU time tracked elapsed wall time almost exactly
    # while the log file showed nothing for 20+ minutes). Force line-buffering so a
    # long Phase 3 run stays observable in real time, matching this project's own
    # "silent monitoring gap" lesson for OOM kills -- this is the same class of risk.
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--steps", type=int, required=True, help="gradient steps (finetune-scale, not their 45k from-scratch default)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-weight", type=float, default=5.0)
    ap.add_argument("--strata-ratios", type=float, nargs=3, default=[0.3, 0.4, 0.3],
                     help="low_texture boundary uniform")
    ap.add_argument("--workers", type=int, default=0,
                     help="DataLoader workers. Default 0 (single-process) is deliberate: "
                     "GPU compute (~9s/step measured) completely dominates the cost of "
                     "decoding a 224px JPEG crop, so extra workers buy nothing here -- and "
                     "with workers>0 each gets its own forked/spawned copy of the dataset, "
                     "so StratifiedRefinementDataset.counts mutations never reach the main "
                     "process (confirmed: a 60-step pilot with workers=2 logged all-zero "
                     "strata_counts the entire run). Only raise this if a future profiling "
                     "pass shows data loading actually becomes the bottleneck.")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/models/cascadepsp-pc-finetune.pth")
    ap.add_argument("--pilot-out", type=Path, default=None,
                     help="if set, save/overwrite here instead of --out (keeps pilot runs out of the real checkpoint slot)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"device: {device}")

    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend="resnet50")
    print("loading stock pretrained CascadePSP weights...")
    load_pretrained_weights(model)
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sobel = SobelComputer()
    sobel.sobel = sobel.sobel.to(device)

    train_ds = StratifiedRefinementDataset(DATA_DIR / "train", tuple(args.strata_ratios))
    print(f"train pairs: {len(train_ds)}")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, drop_last=True)

    para = dict(
        ce_weight=[0.0, 1.0, 0.5, 1.0, 1.0, 0.5],
        l1_weight=[1.0, 0.0, 0.25, 0.0, 0.0, 0.25],
        l2_weight=[1.0, 0.0, 0.25, 0.0, 0.0, 0.25],
        grad_weight=args.grad_weight,
    )

    out_path = args.pilot_out if args.pilot_out is not None else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    step = 0
    t_start = time.time()
    running_loss = 0.0
    loader_iter = iter(loader)
    while step < args.steps:
        try:
            im, seg, gt = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            im, seg, gt = next(loader_iter)

        im, seg, gt = im.to(device), seg.to(device), gt.to(device)
        t0 = time.time()
        images = model(im, seg)
        images["im"], images["seg"], images["gt"] = im, seg, gt
        sobel.compute_edges(images)
        loss_and_metrics = compute_loss_and_metrics(images, para)
        optimizer.zero_grad()
        loss_and_metrics["total_loss"].backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_time = time.time() - t0

        running_loss += float(loss_and_metrics["total_loss"])
        step += 1

        if step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            elapsed = time.time() - t_start
            eta = (args.steps - step) * (elapsed / step)
            print(f"step {step}/{args.steps}  loss={avg_loss:.4f}  step_time={step_time:.2f}s  "
                  f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min  "
                  f"strata_counts={train_ds.counts}")
            running_loss = 0.0

        if step % args.save_every == 0 or step == args.steps:
            torch.save(add_module_prefix(model.state_dict()), out_path)
            print(f"saved checkpoint: {out_path} (step {step})")

    print(f"done. total time: {(time.time()-t_start)/60:.1f}min for {args.steps} steps")
    print(f"final strata pick counts: {train_ds.counts}")


if __name__ == "__main__":
    main()
