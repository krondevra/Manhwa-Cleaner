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
