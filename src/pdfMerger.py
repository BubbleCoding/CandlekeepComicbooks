from PIL import Image
import os

image_dir = "comicPages"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
image_files.sort()

images = []
for file in image_files:
    img_path = os.path.join(image_dir, file)
    img = Image.open(img_path).convert("RGB")
    images.append(img)

if images:
    images[0].save("comic.pdf", save_all=True, append_images=images[1:])