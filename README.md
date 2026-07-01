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

1. **rule-based** — flood fill + panel detection (`remove_manhwa_bg.py`)
2. **classical ML** — OpenCV Random Trees pixel classifier, single-example
   training (did not generalise past the training image)
3. **deep learning** — `SmallUNet` binary segmentation in PyTorch, 7-channel
   input (RGB + threshold/morphology/Canny guidance channels derived from the
   manual Photoshop workflow this project automates)
4. **production tooling** — dataset prep, heuristic evaluation without ground
   truth, hard-case mining, semi-automatic mask/ROI generation, parameter
   search for Photoshop-style levels/threshold profiles

Full history of that iteration — including abandoned approaches and why they
were abandoned — is in the git log (`git log --oneline`).

## Layout
```text
tools/      current pipeline scripts (longify, split, merge, cut_samples,
            mask_preview_tool, mask_boundary_roi, mask_parameter_search,
            ml_cleaner, evaluate, compare)
scripts/    Photopea/Photoshop JSX scripts for manual/semi-auto mask creation
docs/       command reference (docs/readme.md) and manual cleaning
            workflow (docs/pipeline.md)
src/        earlier standalone merge/clean/cutframes/montage pipeline
remove_manhwa_bg.py   earliest rule-based prototype, kept for history
```

`data/`, `models/`, `reports/` are generated locally (gitignored) — chapter
images, trained checkpoints and evaluation reports are not tracked.

## Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy
```

See `docs/readme.md` for the current command reference for every tool.

## Note on training data
Earlier model checkpoints (`models/1.0`–`2.1`) were trained on copyrighted
manhwa chapters for research/prototyping only and have been removed from this
repository's history. The pipeline and methodology are unaffected — going
forward, training uses an open, reproducible dataset instead.
