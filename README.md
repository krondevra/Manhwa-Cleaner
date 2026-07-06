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

Full history of that iteration — including abandoned approaches and why they
were abandoned — is in the git log (`git log --oneline`).

## Layout
```text
src/        current pipeline scripts (longify, split, merge, cut_samples,
            ml_cleaner, evaluate, compare, compare_models_video)
scripts/    Photopea/Photoshop JSX scripts for manual mask creation
docs/       command reference (docs/readme.md) and manual cleaning
            workflow (docs/pipeline.md)
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

## License

**Pipeline code** (all `.py` files) — [MIT License](LICENSE) © 2026 Devids Kronbergs.

**Artwork and generated dataset** — derived from [Pepper & Carrot](https://www.peppercarrot.com/) by [David Revoy](https://www.davidrevoy.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: **"Pepper & Carrot" by David Revoy**.
