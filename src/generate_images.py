from diffusers import DiffusionPipeline
import torch
import AddTextToImage
import json

testing = False

def generate_images_from_script(script_path):
    # Load comic panel script with LLM-generated prompts
    with open(script_path, "r", encoding="utf-8") as f:
        comic_script = json.load(f)

    # Load SDXL pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.load_lora_weights("../lora/Fantasy_art_XL_V1.safetensors")
    pipe.to("cuda")

    for i, panel in enumerate(comic_script):
        prompt = "Comic book illustration. Vivid fantasy." + panel.get("sdxl_prompt", "")
        negative_prompt = panel.get("negative_prompt", "")

        if not prompt:
            print(f"⚠️ Skipping panel {i}: no prompt found.")
            continue

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
        ).images[0]

        image.save(f"output/images/imagesWithoutText/panel_{i}.png")
        image_with_text = AddTextToImage.add_text_to_panel(panel["text"], image)
        image_with_text.save(f"output/images/imagesWithText/panel_{i}.png")

        if i == 5 and testing:
            return

def main():
    generate_images_from_script("output/adaptive_comic_script_with_prompts.json")

if __name__ == "__main__":
    main()
