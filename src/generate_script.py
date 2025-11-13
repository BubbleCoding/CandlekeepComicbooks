import ollama
import json
import re
import time

def load_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def adaptive_panel_generator(transcript, chunk_size=15, max_retries=3):
    panels = []
    position = 0
    memory_buffer = []
    story_so_far = ""

    while position < len(transcript):
        chunk = transcript[position : position + chunk_size]
        print(f"🧠 Processing chunk at line {position}...")

        retries = 0
        while retries < max_retries:
            try:
                used_indices, panel, summary = generate_panel_from_chunk(chunk, story_so_far)
                if panel:
                    panels.append(panel)

                    memory_buffer.append(summary)
                    if len(memory_buffer) > 10:
                        memory_buffer.pop(0)
                    story_so_far = "\n".join(memory_buffer)

                    num_used = max(used_indices) + 1 if used_indices else 1
                    position += num_used
                    break  # success, break out of retry loop
                else:
                    retries += 1
                    print(f"⚠️ No panel returned, retrying... ({retries}/{max_retries})")
                    time.sleep(0.3)
            except Exception as e:
                retries += 1
                print(f"⚠️ Panel generation error: {e} (retry {retries}/{max_retries})")
                time.sleep(0.3)

        else:
            print("❌ Failed after maximum retries, skipping chunk.")
            position += 1  # give up and move on

    return panels

def generate_panel_from_chunk(chunk, story_so_far):
    content = "\n".join(f"{i}: {seg['speaker']}: {seg['text']}" for i, seg in enumerate(chunk))

    prompt = f"""
You are an AI comic scriptwriter adapting a Dungeons & Dragons session.

Here is what has happened so far:
{story_so_far.strip()}

Here is the next part of the transcript:
{content}

IMPORTANT FILTERING:

- Ignore any rule talk, dice rolls, or out-of-character chatter.
- Focus only on in-character dialogue, world-building, and story-driving actions.

TASK:

Generate ONE new panel based on a selection of lines from this chunk. Your job is to condense meaningful story progression into a single visual scene.

Output JSON in this format:
{{
  "used_segments": [list of indices],
  "panel": {{
    "setting": "...",
    "characters": [Character names],
    "objects": [Important objects],
    "action": "...",
    "mood": "...",
    "camera_view": "...",
    "lighting": "...",
    "text": "..."
  }},
  "summary": "One sentence summary of the panel for story memory."
}}

- Every panel must include all 9 fields and cannot be empty.
- Output only valid JSON — no extra text, no commentary.
"""

    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    output = response["message"]["content"]
    return parse_adaptive_json(output)

def parse_adaptive_json(text):
    try:
        match = re.search(r'{.*}', text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")
        json_text = match.group(0)
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*\]", "]", json_text)
        parsed = json.loads(json_text)

        used = parsed.get("used_segments", [])
        panel = parsed.get("panel", {})
        summary = parsed.get("summary", "")

        return used, panel, summary
    except Exception as e:
        print("⚠️ JSON parsing failed:", e)
        return [], None, ""

def save_panels(panels, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(panels, f, indent=2, ensure_ascii=False)

def main():
    transcript_path = "output/clean_transcript_with_speakers.json"
    output_path = "output/adaptive_comic_script.json"

    transcript = load_transcript(transcript_path)
    panels = adaptive_panel_generator(transcript)
    save_panels(panels, output_path)
    print(f"✅ Script saved to {output_path}")

if __name__ == "__main__":
    main()
