# Manhwa Cleaner
Pipeline for turning saved manhwa/webtoon chapters into cleaned, transparent
long-strip PNGs: merge pages → remove background (near-white panels, gutters,
paper texture, and light artifacts) while preserving frames, speech bubbles,
SFX and text → cut into frames for downstream use. Dark/black and gray
backgrounds are not reliably handled by either pipeline in this repo yet —
see "Known limitations" below.

<p align="center">
  <img src="https://raw.githubusercontent.com/krondevra/Manhwa-Cleaner/main/assets/manhwa_cleaner_pipeline_showdown.gif" alt="Old ML vs current gen9 pipeline on a composited stress-test page">
</p>

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
5. **classical spiky-cloud pipeline** — an OpenCV replication of the manual
   Photoshop spiky-cloud cleaning workflow (`clean_page_v10` production /
   `clean_page` v12 candidate), superseded by gen9 below; its regression
   battery and PSD ground-truth extractors carried forward.
6. **gen9** (`src/pipeline/gen9/`) — a deterministic, no-model port of the
   full manual Photoshop/Photopea cleaning algorithm: layered Levels/
   Threshold/Minimum-Maximum image derivatives feed four write-once
   classifiers (background panel/gutter selection, SFX outline recovery,
   trapped-pocket detection, spiky-cloud/scream-burst handling), each
   locking its territory so no later stage can re-touch it. Validated
   pixel-exact against hand-cleaned checkpoint PSDs across multiple real
   chapters; on a held-out hand-cleaned test page it differs from the human
   reference by well under 1% of page pixels, against ~8–10% for the ML
   baseline on the same page (see the demo above — the background halo
   around bubbles/SFX is the ML pipeline's dominant remaining defect class).

A learned CascadePSP refinement stage was evaluated and removed (2026-08-10):
its base weights' upstream license provenance is incompatible with this
project's MIT policy, and results did not justify keeping it — the full record
stays in `docs/ml_strategy_history.md`.

**Current recommended pipeline**: `src/pipeline/gen9/run_hierarchy.py
<page.png> [out_dir]` — no trained model required, deterministic, and the
best-measured of the two (above). The earlier ML pipeline
(`data/models/10.0-baseline.pt` + `src/pipeline/ml_cleaner.py process ...
--reclaim-islands`) remains available and is documented in
`docs/ml_strategy_history.md`.

## Known limitations
Both pipelines target **near-white** backgrounds specifically:
- **Dark/black and gray backgrounds are not removed.** Neither pipeline
  reliably distinguishes gutter-black from dark art — measured
  near-zero deletion on black-background pages (documented as
  "zero-signal-on-black" in `docs/ml_strategy_history.md`). Treat this as
  a paused/unsupported domain, not a bug to work around per-page.
- **Borderless neutral or gradient-tinted panels can be over-deleted.**
  A full-bleed panel with no drawn frame and a flat or gently graded near-
  white tone reads the same as page background to the panel-selection
  classifier, in both pipelines — the panel's own content is at risk of
  being swept away entirely. Panels with a drawn border, or clear non-white
  content, are unaffected.

See `docs/decisions.md` and `docs/ml_strategy_history.md` for the full,
measured defect-class history (what's been tried, what worked, what didn't,
and why) — check both before starting new work in either pipeline.

## Layout
```text
src/pipeline/            tracked production code
  gen9/                    current recommended pipeline (deterministic, no
                           trained model) -- run_hierarchy.py is the entry point
  ml_cleaner.py            SmallUNet ML pipeline (train / process)
  longify.py, split.py,
  merge.py                 page-merge and long-strip chunking utilities
  classifiers/, export/     panel/frame/SFX detector profiles and PSD/sidecar
                           export helpers
  jsx/                     Photopea/Photoshop automation scripts
src/dev/                 experiment scripts, harnesses, and one-off tools --
                         gitignored, not part of the public repo
docs/                    command reference (docs/readme.md), decision log
                         (docs/decisions.md), strategy history
                         (docs/ml_strategy_history.md, docs/history.md)
assets/                  README media
```

`data/` (dataset, chapter images, trained checkpoints under `data/models/`)
and all working/notes directories (`.tmp/`) are generated or private
(gitignored) — not tracked. See "Training data" below for the expected
`data/` layout.

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
kept). `src/pipeline/ml_cleaner.py train` reads
`data/dataset_split/train` and `data/dataset_split/val` by default; see
`docs/readme.md` for selecting a subset of variants.

`data/dataset_split/` and `data/dataset_split_scaled/` aren't kept on disk
between sessions (regenerable, not required for inference) — regenerate from
the PepperNCarrotDataset repo's `src/tools/cut_dataset.py` before training.

## Checkpoints and releases
Small checkpoints (SmallUNet, ~14MB each) are tracked directly in
`data/models/`.

## License

**Pipeline code** (all `.py` files) — [MIT License](LICENSE) © 2026 Devids Kronbergs.

**Artwork and generated dataset** — derived from [Pepper & Carrot](https://www.peppercarrot.com/) by [David Revoy](https://www.davidrevoy.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution: **"Pepper & Carrot" by David Revoy**.
