from PIL import Image
import os


def resize_to_target(image, target_size):
    result = Image.new("RGB", target_size, "black")
    result.paste(image, ((target_size[0] - image.width) // 2, (target_size[1] - image.height) // 2))
    return result


def load_images(output_dir: str = "output"):
    image_dir = os.path.join(output_dir, "images", "imagesWithText")
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and any(c.isdigit() for c in f)
    ]
    sorted_files = sorted(image_files, key=lambda name: int("".join(filter(str.isdigit, name))))

    images = []
    for filename in sorted_files:
        images.append(Image.open(os.path.join(image_dir, filename)))
    return images


def main(output_dir: str = "output"):
    images = load_images(output_dir)
    columns, rows = 2, 3
    images_per_page = columns * rows

    pages_dir = os.path.join(output_dir, "comicPages")
    os.makedirs(pages_dir, exist_ok=True)

    for i in range(0, len(images), images_per_page):
        page_images = images[i:i + images_per_page]
        panel_w, panel_h = page_images[0].width, page_images[0].height

        output_width = columns * panel_w + (columns - 1) * 10
        output_height = rows * panel_h + (rows - 1) * 10

        result_image = Image.new("RGB", (output_width, output_height), "black")

        for j, img in enumerate(page_images):
            x = (j % columns) * (panel_w + 10)
            y = (j // columns) * (panel_h + 10)
            result_image.paste(resize_to_target(img, (panel_w, panel_h)), (x, y))

        result_image = result_image.resize((1024, 1536))
        page_path = os.path.join(pages_dir, f"comicPage_{i // images_per_page}.jpg")
        result_image.save(page_path)
        print(f"Saved {page_path}")


if __name__ == "__main__":
    main()
