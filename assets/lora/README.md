# LoRA Weights

This directory contains LoRA (Low-Rank Adaptation) weights for Stable Diffusion XL.

## Required Files

The following files are required but not included in the repository due to their large size:

1. **Fantasy_art_XL_V1.safetensors** (~435 MB)
2. **80sFantasyMovieMJ7SDXL.safetensors** (~218 MB)
3. **comic_book_illustration.safetensors** (~8 MB)

## Where to Download

Download these LoRA weights from:
- [CivitAI](https://civitai.com/)
- [HuggingFace](https://huggingface.co/)

Place the downloaded `.safetensors` files in this directory (`assets/lora/`).

## Usage

The pipeline will automatically load `Fantasy_art_XL_V1.safetensors` when generating images.
You can modify `src/generate_images.py` to use different LoRA weights.
