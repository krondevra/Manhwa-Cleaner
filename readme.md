# Prompt
## Erase background
```md
I need automatic erase of white background out of frame. In frame white background, in frame from up and bottom narrow border. out of frame white background which must be erased. I attached manual erase (manual-015.png). I got such result with 3 click of magic wand from up, bottom and bottom korean symbol. Keep in mind code on python should clean image automatically. I expect get .png. write a python code. DO NOT CLEAN BACKGROUND IN FRAME AND IN CLOUDS WITH TEXT. code should work without reference. Reference was for you. only 1 image input
```

## Effects
```md
1. наклонение вправо/влево на протяжение фрейма. фрейм длиться меньше
2. длинный фрейм камера сверху/вниз и снизу/вверх
3. фрейм появляеться резко в кадре (быстрое появление с blur от движения
4. уход в затемнение не делаем
5. фрейм растворяеться
6. фрейм появляеться (проявление)
7. фреймы когда появились, движутся ближе к камере на протяжении всего фрейма
8. длинные фреймы без пролёта камеры приближаются к камере

```

# Run
## Create and activate venv
```python
python -m venv .venv && source .venv/bin/activate
```

## Code brief description
`merge.py` - merge all images into one
`mergeandclean.py` - merge all images into one + clean white background
`cutframes` - cut cleaned long image by frames
`montage` - create a video from cutted frames

## Libraries
```bash
pip3 install pillow numpy opencv-python
```


# temp

python code7.py original-015.jpg

python code12.py frames demo.mp4
