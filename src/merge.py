from PIL import Image
import os
import glob

folder = "chapters"
output = "merged.jpg"

images = sorted(glob.glob(os.path.join(folder, "*.jpg")))
imgs = [Image.open(img).convert("RGB") for img in images]

width = imgs[0].width
height = sum(img.height for img in imgs)

result = Image.new("RGB", (width, height), (255, 255, 255))

y_offset = 0
for img in imgs:
    result.paste(img, (0, y_offset))
    y_offset += img.height

result.save(output, quality=95)
print("Saved:", output)
