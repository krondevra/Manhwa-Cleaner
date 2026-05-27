# Commands
## longify.py


```bash
python tools/longify.py data/chapters-initial \
  --output-dir data/chapters-long \
  --name-mode index \
  --start-index 0
```

---

## split.py

```bash
CH=033

python tools/split.py "data/chapters-long/${CH}.png" \
  --chunk 50000 \
  --output-dir "data/temp/${CH}"
```

---

## merge.py

```bash
CH=033

python tools/merge.py "$CH" \
  --dir "data/temp/${CH}" \
  --output "data/temp/${CH}/${CH}_cleaned.png"
```
