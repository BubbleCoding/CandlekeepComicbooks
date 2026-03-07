"""Generate FLUX image prompts for each comic panel.

Uses Claude to write rich natural-language prompts. Character appearance
descriptions from characters.json are injected per panel so visual
consistency is maintained across all generated images.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import anthropic


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_character_lookup(characters_path: str) -> Dict[str, str]:
    """Return {name_lowercase: visual_description} for prompt injection."""
    if not Path(characters_path).exists():
        return {}
    data = load_json(characters_path)
    lookup = {}
    for char in data.get("characters", []):
        name = char.get("name", "").strip()
        appearance = char.get("appearance", "").strip()
        features = char.get("distinctive_features", "").strip()
        if name and (appearance or features):
            desc = ", ".join(filter(None, [appearance, features]))
            lookup[name.lower()] = f"{name}: {desc}"
    return lookup


def get_style_config(characters_path: str) -> Dict[str, str]:
    if not Path(characters_path).exists():
        return {}
    data = load_json(characters_path)
    return data.get("style", {})


def resolve_character_descriptions(
    panel_characters: List[Any],
    character_lookup: Dict[str, str],
) -> str:
    """Match panel character names to their visual descriptions."""
    descriptions = []
    for entry in panel_characters:
        name = entry if isinstance(entry, str) else entry.get("name", "")
        desc = character_lookup.get(name.lower())
        if desc:
            descriptions.append(desc)
        elif name:
            descriptions.append(name)
    return "; ".join(descriptions)


def generate_flux_prompt(
    panel: Dict[str, Any],
    character_descriptions: str,
    art_style: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
) -> str:
    user_message = f"""Write a single image generation prompt for this comic panel.

Panel details:
- Setting: {panel.get('setting', '')}
- Characters: {character_descriptions or ', '.join(str(c) for c in panel.get('characters', []))}
- Objects: {', '.join(str(o) for o in panel.get('objects', []))}
- Action: {panel.get('action', '')}
- Mood: {panel.get('mood', '')}
- Camera view: {panel.get('camera_view', '')}
- Lighting: {panel.get('lighting', '')}

Art style to use: {art_style}

Requirements:
- Write one fluent, descriptive paragraph.
- Include character appearances exactly as described — this ensures visual consistency.
- Emphasize the most visually striking elements.
- Mention the art style, lighting, and camera angle naturally.
- Do NOT include dialogue or text — that is added separately.
- Return only the prompt text, no commentary or quotes."""

    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text.strip()


def process_script(
    input_path: str,
    output_path: str,
    characters_path: str = "assets/characters.json",
    model: str = "claude-sonnet-4-6",
) -> None:
    client = anthropic.Anthropic()
    script = load_json(input_path)
    character_lookup = build_character_lookup(characters_path)
    style_config = get_style_config(characters_path)
    art_style = style_config.get(
        "art_style",
        "comic book panel, bold ink outlines, flat colors, high contrast, D&D fantasy illustration"
    )
    negative_prompt = style_config.get(
        "negative_prompt",
        "deformed, blurry, extra limbs, poorly drawn hands, extra fingers, mutated, ugly, watermark, photorealistic"
    )

    for i, panel in enumerate(script):
        print(f"Generating prompt for panel {i + 1}/{len(script)}...")
        try:
            char_descs = resolve_character_descriptions(
                panel.get("characters", []),
                character_lookup,
            )
            prompt = generate_flux_prompt(panel, char_descs, art_style, client, model)
            panel["flux_prompt"] = prompt
            panel["negative_prompt"] = negative_prompt
        except Exception as e:
            print(f"  Failed for panel {i}: {e}")
            panel["flux_prompt"] = ""
            panel["negative_prompt"] = negative_prompt
        time.sleep(0.2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    print(f"Prompts saved to {output_path}")


def main(
    input_path: str = "output/adaptive_comic_script.json",
    output_path: str = "output/adaptive_comic_script_with_prompts.json",
    characters_path: str = "assets/characters.json",
) -> None:
    process_script(input_path, output_path, characters_path)


if __name__ == "__main__":
    main()
