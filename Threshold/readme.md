# ThresholdGUI-Manual.py
```python
python ThresholdGUI-Manual.py 012-1.png
```

# ThresholdGUI-Auto.py
## Hard mask search
```python
python ThresholdGUI-Auto.py 012-1.png --rois used_rois.txt --priority hard
```

## Soft mask search
```python
python ThresholdGUI-Auto.py 012-1.png --rois used_rois.txt --priority soft --top-tonal 40
```
