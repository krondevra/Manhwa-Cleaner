# Manhwa Cleaner
ML pipeline for turning saved manhwa/webtoon chapters into cleaned, transparent
long-strip PNGs: merge pages → remove background (white/black/gray, gradients,
artifacts) while preserving frames, speech bubbles, SFX and text → cut into
frames for downstream use.

## Approach
The background-removal problem is context-sensitive: the same pixel color can
be either removable background or content depending on structure, so
rule-based heuristics (flood fill, panel detection) were not enough on their
own. The project moved through:

1. **rule-based** — flood fill + panel detection (early prototype, since removed)
2. **classical ML** — OpenCV Random Trees pixel classifier, single-example
   training (did not generalise past the training image)
3. **deep learning** — `SmallUNet` binary segmentation in PyTorch, 7-channel
   input (RGB + threshold/morphology/Canny guidance channels derived from the
   manual Photoshop workflow this project automates)
4. **production tooling** — dataset prep, heuristic evaluation without ground
   truth, hard-case mining
5. **learned refinement (experimental)** — CascadePSP (Cheng et al.), a
   class-agnostic refinement network, finetuned on Pepper & Carrot pairs to
   correct SmallUNet's raw output. Best net pixel-error of any tested config
   against real ground truth, but trades some gutter/SFX cleanup quality for
   fixing over-deletion of real artwork — not yet adopted for production.

**Current production**: `data/models/10.0-baseline.pt` + `src/ml_cleaner.py
process ... --reclaim-islands`.

Full history of that iteration — including abandoned approaches and why they
were abandoned — is in the git log (`git log --oneline`) and, curated,
`docs/ml_strategy_history.md` (what's been tried for background isolation
specifically, what worked, what didn't, and why — check it before starting
a new ML experiment).

## Layout
```text
src/        current pipeline scripts (longify, split, merge, cut_samples,
            ml_cleaner, evaluate, compare, compare_models_video)
docs/       command reference (docs/readme.md), strategy history
            (docs/ml_strategy_history.md)
```

`data/` (dataset, chapter images, trained checkpoints under `data/models/`)
and `reports/` are generated/copied locally (gitignored) — not tracked.
See "Training data" below for the expected `data/` layout.

## Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy
```

See `docs/readme.md` for the current command reference for every tool.

## Training data
Earlier model checkpoints (`models/1.0`–`2.1`) were trained on copyrighted
manhwa chapters for research/prototyping only and have been removed from this
repository's history. Training now uses the open, reproducible
[Pepper & Carrot dataset](https://www.peppercarrot.com/) (CC BY 4.0), copied
locally (gitignored, not tracked) as:

```text
data/dataset_split/train/   per-episode input variants + *_cleaned targets (training split)
data/dataset_split/val/     same layout, held out for checkpoint selection
data/models/                trained checkpoints (.pt + .json config)
```

Each episode folder is self-contained: every input variant (`initial`,
`framed_speechbubles_w`, `framed_speechbubles_shapes_bw`, ...) pairs against
an `initial_cleaned/` sibling folder for the universal fully-clean target,
plus its own `<variant>_cleaned/` sibling folder wherever the ground truth
legitimately differs (frame/bubble outline kept, SFX/bubble/shape marks
kept). `src/ml_cleaner.py train` reads
`data/dataset_split/train` and `data/dataset_split/val` by default; see
`docs/readme.md` for selecting a subset of variants.

`data/dataset_split/` and `data/dataset_split_scaled/` aren't kept on disk
between sessions (regenerable, not required for inference) — regenerate from
the PepperNCarrotDataset repo's `src/tools/cut_dataset.py` before training.

## Checkpoints and releases
Small checkpoints (SmallUNet, ~14MB each) are tracked directly in
`data/models/`. Larger third-party-architecture checkpoints (e.g. the
CascadePSP finetune above, ~260MB) exceed GitHub's 100MB per-file limit and
are distributed as [GitHub Release](../../releases) assets instead — download
and place under `data/models/` manually if needed;
`data/models/cascadepsp-*` is gitignored for this reason.

## License

**Pipeline code** (all `.py` files) — [MIT License](LICENSE) © 2026 Devids Kronbergs.

**Artwork and generated dataset** — derived from [Pepper & Carrot](https://www.peppercarrot.com/) by [David Revoy](https://www.davidrevoy.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: **"Pepper & Carrot" by David Revoy**.
