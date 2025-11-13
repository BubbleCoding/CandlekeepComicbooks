from PIL import Image
import os

def main():
    image_dir = "output/comicPages"
    
    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"⚠️ Directory not found: {image_dir}")
        print("   No comic pages to merge. Run the full pipeline first.")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort()

    if not image_files:
        print(f"⚠️ No image files found in {image_dir}")
        return

    images = []
    for file in image_files:
        img_path = os.path.join(image_dir, file)
        img = Image.open(img_path).convert("RGB")
        images.append(img)

    if images:
        output_path = "output/comic.pdf"
        images[0].save(output_path, save_all=True, append_images=images[1:])
        print(f"✅ Created PDF with {len(images)} pages: {output_path}")
    else:
        print("⚠️ No images loaded to merge into PDF")

if __name__ == "__main__":
    main()