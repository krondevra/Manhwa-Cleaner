# Commands
## longify.py
```bash
python src/longify.py data/chapters-initial \
  --output-dir data/chapters-long \
  --name-mode index \
  --start-index 1
```

---

## split.py
```bash
CH=033 && python src/split.py "data/chapters-long/${CH}.png" \
  --chunk 50000 \
  --output-dir "data/temp/${CH}"
```

---

## merge.py
```bash
CH=033 && python src/merge.py "$CH" \
  --dir "data/temp/${CH}" \
  --output "data/temp/${CH}/${CH}_cleaned.png"
```

---

## cut_samples.py
```bash
CH=033 && python src/cut_samples.py "$CH" \
  --output-dir "data/temp/${CH}/cut-samples" \
  --pad-top 60 \
  --pad-bottom 50 \
  --boundary-margin 4 \
  --post-merge-gap 90 \
  --post-merge-max-height 1600 \
  --min-standalone-height 600 \
  --clear
```

---

## evaluate.py
### Range
```bash
python src/evaluate.py \
  --from-chapter 003 \
  --to-chapter 005
```

### List
```bash
python src/evaluate.py \
  --chapters 162,119,123,118,122,096,120,121 \
```

Optional:
```bash
  --allow-missing
```

---

## ml_cleaner.py
### Train model from 0
Trains on the Pepper & Carrot dataset split (`data/dataset_split/train/`,
validated against `data/dataset_split/val/` for checkpoint selection)
instead of manually cleaned manhwa samples. Each episode is self-contained:
base variants (including `initial`) pair against their episode's own
`initial_cleaned/` folder.
```bash
python src/ml_cleaner.py train \
  --dataset data/dataset_split/train \
  --val-dataset data/dataset_split/val \
  --model data/models/4.0.pt
```

### Train on selected variants only
```bash
python src/ml_cleaner.py train \
  --dataset data/dataset_split/train \
  --val-dataset data/dataset_split/val \
  --variants framed_speechbubles_w,framed_speechbubles_w_jpeg,framed_speechbubles_shapes_bw,framed_speechbubles_shapes_mixed \
  --model data/models/4.0.pt
```

### Continue training from model
```bash
python src/ml_cleaner.py train \
  --dataset data/dataset_split/train \
  --val-dataset data/dataset_split/val \
  --resume data/models/4.0.pt \
  --model data/models/4.1.pt
```

### Disable validation (train on old, non-split dataset layouts)
```bash
python src/ml_cleaner.py train \
  --dataset data/dataset \
  --val-dataset "" \
  --renders-cleaned data/renders_cleaned \
  --model data/models/3.0.pt
```

### Clean chapter (single)
```bash
CH=003 && python src/ml_cleaner.py process "$CH" \
  --model data/models/2.0.pt \
  --red-preview
```

### Inference postprocessing flags (all process* commands)
```bash
  --reclaim-islands   # un-delete regions not connected to any image edge
                      # (production default -- see docs/ml_strategy_history.md)
  --repair-frames     # un-delete pixels inside regions fully enclosed by
                      # near-black strokes (closed panel frames, bubble
                      # outlines); tune with --frame-darkness (40),
                      # --frame-min-interior (10000), --frame-inset (2)
  --protect-borders   # re-keep thin near-black frame lines near kept content;
                      # tune with --border-band (3), --border-darkness (40)
```
Output filenames encode the combination (`v10.0-baseline-islands-...`,
`...-frames-...`, `...-islandsframes-...`) so configurations never collide.
Note for `compare_models_video.py`: version tags support only ONE hyphenated
suffix segment -- `islandsframes` is deliberately one word.

### list
```bash
python src/ml_cleaner.py process-list \
  --chapters 162,119,123,118,122,96,120,121,117,124 \
  --model data/models/2.1.pt \
  --red-preview
```

### Clean chapters (range)
```bash
python src/ml_cleaner.py process-range \
  --from-chapter 003 \
  --to-chapter 004 \
  --model data/models/2.0.pt \
  --red-preview
```

---

## compare.py
```bash
python src/compare.py 009 \
  --results \
    data/compare/009_cleaner4_red_preview.png \
    data/compare/009_cleaner5_red_preview.png \
    data/compare/009_result_red_preview.png \
  --labels "MODEL 1,MODEL 1.1, MODEL 2.0" \
  --centers 18000,45950,59100,85200,113000
```
