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

## mask_preview_tool.py
```bash
CH=033 && python tools/mask_preview_tool.py "$CH"
```

---

## mask_boundary_roi.py
```bash
CH=033 && python tools/mask_boundary_roi.py "$CH"
```

---

## mask_parameter_search.py
```bash
CH=033 && python tools/mask_parameter_search.py "$CH" \
  --profile black-hard \
  --top-tonal 60
```

Profiles: 
```bash
--profile black-soft 
--profile black-hard
--profile white-soft
--profile white-hard
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
  --chapters 002,034,024 \
```

Optional:
```bash
  --allow-missing
```

---

## ml_cleaner.py
### Train model from 0
```bash
python tools/ml_cleaner.py train \
  --samples data/samples \
  --model models/2.1.pt
```

### Continue training from model
```bash
python tools/ml_cleaner.py train \
  --samples data/samples \
  --resume models/2.0.pt \
  --model models/2.1.pt
```

### Clean chapter (single)
```bash
CH=003 && python tools/ml_cleaner.py process "$CH" \
  --model models/2.0.pt \
  --red-preview
```

### Clean chapters (range)
```bash
python tools/ml_cleaner.py process-range \
  --from-chapter 003 \
  --to-chapter 004 \
  --model models/2.0.pt \
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
