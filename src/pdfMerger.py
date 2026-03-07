from PIL import Image
import os


def main(output_dir: str = "output"):
    image_dir = os.path.join(output_dir, "comicPages")

    if not os.path.exists(image_dir):
        print(f"Directory not found: {image_dir}")
        return

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and any(c.isdigit() for c in f)
    ]

    if not image_files:
        print(f"No image files found in {image_dir}")
        return

    # Numeric sort so page 10 comes after page 9
    image_files.sort(key=lambda name: int("".join(filter(str.isdigit, name))))

    images = [Image.open(os.path.join(image_dir, f)).convert("RGB") for f in image_files]

    output_path = os.path.join(output_dir, "comic.pdf")
    images[0].save(output_path, save_all=True, append_images=images[1:])
    print(f"Created PDF with {len(images)} pages: {output_path}")


if __name__ == "__main__":
    main()
