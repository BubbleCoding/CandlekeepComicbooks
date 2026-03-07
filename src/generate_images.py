"""Generate comic panel images using FLUX.1.

Auto-detects device (CUDA / MPS / CPU) and configures the pipeline accordingly:
  - CUDA (e.g. RTX 3080 Ti, 12GB): loads with bfloat16 + CPU offload to fit in VRAM.
  - MPS (Apple Silicon): loads with float16, uses Metal backend.
  - CPU: loads with float32, slow but functional.

Set FLUX_MODEL env var to override the default model repo.
Set TESTING=1 env var (or testing=True below) to generate only 6 panels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline

from src import AddTextToImage

testing = False  # set True or TESTING=1 env var for 6-panel preview runs

FLUX_MODEL = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
LORA_PATH = "assets/lora/comic_book_illustration.safetensors"


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(device: str) -> FluxPipeline:
    print(f"Loading FLUX pipeline on {device} ({FLUX_MODEL})...")

    if device == "cuda":
        # bfloat16 + CPU offload fits FLUX.1-dev in 12GB VRAM
        pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
    elif device == "mps":
        pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.float16,
        )
        pipe.to("mps")
    else:
        pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.float32,
        )
        pipe.to("cpu")

    # Load comic style LoRA if present
    if Path(LORA_PATH).exists():
        print(f"Loading LoRA: {LORA_PATH}")
        pipe.load_lora_weights(LORA_PATH)
    else:
        print(f"No LoRA found at {LORA_PATH}, skipping.")

    return pipe


def generate_images_from_script(
    script_path: str,
    output_dir: str = "output",
    max_panels: int | None = None,
) -> None:
    with open(script_path, "r", encoding="utf-8") as f:
        comic_script = json.load(f)

    Path(f"{output_dir}/images/imagesWithoutText").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/images/imagesWithText").mkdir(parents=True, exist_ok=True)

    is_testing = testing or os.environ.get("TESTING", "0") == "1"
    if is_testing:
        comic_script = comic_script[:6]
        print("Testing mode: generating first 6 panels only.")

    if max_panels:
        comic_script = comic_script[:max_panels]

    device = detect_device()
    pipe = load_pipeline(device)

    for i, panel in enumerate(comic_script):
        prompt = panel.get("flux_prompt", "")
        if not prompt:
            print(f"  Skipping panel {i}: no flux_prompt found.")
            continue

        print(f"  Generating panel {i + 1}/{len(comic_script)}...")

        # FLUX does not use negative prompts natively, but pass it for
        # models/pipelines that support guidance (e.g. FLUX.1-dev with CFG).
        negative_prompt = panel.get("negative_prompt", "")

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            height=1024,
            width=768,
            num_inference_steps=28,
            guidance_scale=3.5,
        )
        image = result.images[0]

        raw_path = f"{output_dir}/images/imagesWithoutText/panel_{i}.png"
        text_path = f"{output_dir}/images/imagesWithText/panel_{i}.png"

        image.save(raw_path)

        panel_text = panel.get("text", "")
        image_with_text = AddTextToImage.add_text_to_panel(panel_text, image)
        image_with_text.save(text_path)

        print(f"    Saved panel {i}")


def main(
    script_path: str = "output/adaptive_comic_script_with_prompts.json",
    output_dir: str = "output",
) -> None:
    generate_images_from_script(script_path, output_dir)


if __name__ == "__main__":
    main()
