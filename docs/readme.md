# Commands
## longify.py
```bash
python tools/longify.py data/chapters-initial \
  --output-dir data/chapters-long \
  --name-mode index \
  --start-index 1
```

---

## split.py
```bash
CH=033 && python tools/split.py "data/chapters-long/${CH}.png" \
  --chunk 50000 \
  --output-dir "data/temp/${CH}"
```

---

## merge.py
```bash
CH=033 && python tools/merge.py "$CH" \
  --dir "data/temp/${CH}" \
  --output "data/temp/${CH}/${CH}_cleaned.png"
```

---

## cut_samples.py
```bash
CH=033 && python tools/cut_samples.py "$CH" \
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
python tools/evaluate.py \
  --from-chapter 003 \
  --to-chapter 005
```

### List
```bash
python tools/evaluate.py \
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
python tools/ml_cleaner.py train \
  --dataset data/dataset_split/train \
  --val-dataset data/dataset_split/val \
  --model data/models/4.0.pt
```

### Train on selected variants only
```bash
python tools/ml_cleaner.py train \
  --dataset data/dataset_split/train \
  --val-dataset data/dataset_split/val \
  --variants framed,framed_jpeg,jpeg,sfx_overlay,bubble_overlay \
  --model data/models/4.0.pt
```

### Continue training from model
```bash
python tools/ml_cleaner.py train \
  --dataset data/dataset_split/train \
  --val-dataset data/dataset_split/val \
  --resume data/models/4.0.pt \
  --model data/models/4.1.pt
```

### Disable validation (train on old, non-split dataset layouts)
```bash
python tools/ml_cleaner.py train \
  --dataset data/dataset \
  --val-dataset "" \
  --renders-cleaned data/renders_cleaned \
  --model data/models/3.0.pt
```

### Clean chapter (single)
```bash
CH=003 && python tools/ml_cleaner.py process "$CH" \
  --model data/models/2.0.pt \
  --red-preview
```

### list
```bash
python tools/ml_cleaner.py process-list \
  --chapters 162,119,123,118,122,96,120,121,117,124 \
  --model data/models/2.1.pt \
  --red-preview
```

### Clean chapters (range)
```bash
python tools/ml_cleaner.py process-range \
  --from-chapter 003 \
  --to-chapter 004 \
  --model data/models/2.0.pt \
  --red-preview
```

---

## compare.py
```bash
python tools/compare.py 009 \
  --results \
    data/compare/009_cleaner4_red_preview.png \
    data/compare/009_cleaner5_red_preview.png \
    data/compare/009_result_red_preview.png \
  --labels "MODEL 1,MODEL 1.1, MODEL 2.0" \
  --centers 18000,45950,59100,85200,113000
```
