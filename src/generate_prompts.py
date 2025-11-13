import json
import ollama
import time

def normalize_panel(panel):
    def normalize_entry(entry):
        if isinstance(entry, str):
            return entry
        elif isinstance(entry, dict):
            name = entry.get("name", "Unknown")
            extras = [v for k, v in entry.items() if k != "name" and v]
            return f"{name} ({', '.join(extras)})" if extras else name
        return str(entry)

    panel["characters"] = [normalize_entry(c) for c in panel.get("characters", [])]
    panel["objects"] = [normalize_entry(o) for o in panel.get("objects", [])]

    return panel

def generate_prompt_with_llama(panel):
    system_message = "You are a prompt engineer for Stable Diffusion XL (SDXL)."

    user_message = f"""
Create a visually descriptive SDXL prompt based on the following panel description:

Setting: {panel.get('setting', '')}
Characters: {", ".join(panel.get("characters", []))}
Objects: {", ".join(panel.get("objects", []))}
Action: {panel.get('action', '')}
Mood: {panel.get('mood', '')}
Camera View: {panel.get('camera_view', '')}
Lighting: {panel.get('lighting', '')}

Requirements:
- Combine all of the above into a single natural-language description.
- Maximum length: 75 tokens.
- Do NOT add quotes or commentary, just return the prompt text only.
- The prompts need to be short, concise, and suitable for generating a single comic book panel.
"""

    response = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        stream=False
    )

    prompt = response["message"]["content"].strip()
    return prompt

def process_script(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    for i, panel in enumerate(script):
        print(f"🔧 Generating SDXL prompt for panel {i}...")
        try:
            panel = normalize_panel(panel)
            prompt = generate_prompt_with_llama(panel)
            panel["sdxl_prompt"] = prompt
            panel["negative_prompt"] = (
                "deformed, blurry, extra limbs, poorly drawn hands, extra fingers, "
                "mutated, out of frame, ugly, tiling, low resolution, bad anatomy, watermark"
            )
        except Exception as e:
            print(f"⚠️ Failed for panel {i}: {e}")
            panel["sdxl_prompt"] = ""
            panel["negative_prompt"] = ""
        time.sleep(0.3)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    print(f"✅ SDXL prompts saved to {output_path}")

def main():
    process_script(
        input_path="output/adaptive_comic_script.json",
        output_path="output/adaptive_comic_script_with_prompts.json"
    )

if __name__ == "__main__":
    main()
