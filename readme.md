# Commands
## Start train from the beggining
```bash
cd ~/Desktop && source /home/user/Desktop/.venv/bin/activate && cd ~/Desktop/Manhwa-cleaner

mkdir -p models logs

python cleaner.py train \
  --samples samples \
  --model models/2.pt \
  --epochs 30 \
  --steps-per-epoch 400 \
  --batch-size 2 \
  --patch-size 512 \
  --device cpu \
  2>&1 | tee logs/2.log
```

## Continue train from stopped point
```bash
cd ~/Desktop && source /home/user/Desktop/.venv/bin/activate && cd ~/Desktop/Manhwa-cleaner

mkdir -p models logs

test -f models/cleaner_4chapters.pt || { echo "ERROR: resume model not found"; exit 1; }

python cleaner.py train \
  --samples samples \
  --resume models/cleaner_4chapters.pt \
  --model models/cleaner_5chapters.pt \
  --epochs 20 \
  --steps-per-epoch 300 \
  --batch-size 2 \
  --patch-size 512 \
  --device cpu \
  2>&1 | tee logs/train_5chapters_from_4chapters.log
```

## Clean chapter with ML
```bash
cd ~/Desktop && source /home/user/Desktop/.venv/bin/activate && cd ~/Desktop/Manhwa-cleaner

CH=009
MODEL=models/2.0.pt

python cleaner.py process \
  "chapters-long/${CH}.png" \
  "chapters-results/${CH}_result.png" \
  --model "$MODEL" \
  --device cpu
```

```bash
cd ~/Desktop && source /home/user/Desktop/.venv/bin/activate && cd ~/Desktop/Manhwa-cleaner

FROM=009
TO=038
MODEL=models/2.0.pt

mkdir -p chapters-results logs

for CH in $(seq -w "$FROM" "$TO"); do
  echo "===== Processing chapter $CH ====="

  python cleaner.py process \
    "chapters-long/${CH}.png" \
    "chapters-results/${CH}_result.png" \
    --model "$MODEL" \
    --device cpu \
    2>&1 | tee "logs/process_${CH}_model2.log"
done
```
