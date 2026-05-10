# ML Manhwa Cleaner

Remove white/black background from manhwa chapters automatically.

## Problem

Long manhwa chapters (~690×50,000–143,000 px) contain white gutters between frames
that must be made transparent. The challenge: the same `#FFFFFF` appears both in
the background *and* inside frames, speech bubbles, and text boxes.

Manual cleanup in Photoshop: 2–9 hours per chapter. Goal: automate 174 chapters.

## Pipeline

```
chapters-initial/   raw per-page PNGs per chapter
chapters-long/      merged long-strip PNGs (one file per chapter)
samples/            manually cleaned pairs (*.png + *_cleaned.png)
models/             trained PyTorch checkpoints
chapters-results/   ML-cleaned output PNGs
logs/               training logs
```

## Approach

1. **rule-based v1** — flood fill from edges through near-white pixels
2. **rule-based v2/v3** — panel detection + keep_mask + safe flood fill
3. **Random Trees** — supervised pixel classifier trained on one manual example
4. **SmallUNet** — full binary segmentation model trained on manually cleaned chapters

## Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy
```

## Commands

### Train
```bash
python cleaner.py train \
  --samples samples \
  --model models/cleaner.pt \
  --epochs 20 --steps-per-epoch 300 --batch-size 2 --patch-size 512 --device cpu \
  2>&1 | tee logs/train.log
```

### Resume training
```bash
python cleaner.py train \
  --samples samples \
  --model models/cleaner_next.pt \
  --resume models/cleaner.pt \
  --epochs 20 --steps-per-epoch 300 --batch-size 2 --patch-size 512 --device cpu
```

### Clean one chapter
```bash
python cleaner.py process chapters-long/005.png chapters-results/005_result.png \
  --model models/cleaner.pt --device cpu
```

### Clean all chapters
```bash
python cleaner.py process-folder \
  --input chapters-long --output chapters-results \
  --model models/cleaner.pt --device cpu
```
