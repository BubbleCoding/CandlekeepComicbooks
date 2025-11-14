"""Adaptive comic panel script generation.

This module ingests a diarized & punctuated transcript and incrementally
derives structured comic panels using an LLM. It maintains a rolling
"story memory" summary to give the model continuity while avoiding feeding
the entire transcript every time.

Key improvements added:
 - Robust panel validation (required fields + defaults)
 - Safer position advancement using max(used_indices) + 1
 - Adaptive retry strategy with prompt reinforcement & partial chunk shrink
 - Optional limits (max_panels, max_seconds) to prevent runaway processing
 - Clear type hints & docstrings for maintainability
 - More defensive JSON parsing (direct attempt before regex fallback)
"""

from __future__ import annotations

import ollama
import json
import re
import time
import math
from typing import List, Dict, Any, Tuple, Optional

# Required panel fields (panel is a dict with these keys)
REQUIRED_FIELDS = [
    "setting",
    "characters",
    "objects",
    "action",
    "mood",
    "camera_view",
    "lighting",
    "text",
]

def validate_panel(panel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and normalize a panel structure.

    Ensures required fields exist; fills missing with empty types.
    Returns cleaned panel or None if fatally invalid (e.g. empty dict).
    """
    if not panel or not isinstance(panel, dict):
        return None
    cleaned: Dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        value = panel.get(field)
        if value is None:
            # Provide sensible defaults: lists for list-like fields, empty string otherwise
            if field in ("characters", "objects"):
                value = []
            else:
                value = ""
        cleaned[field] = value
    # Basic heuristic: discard panels with completely empty textual content
    if not any(str(cleaned.get(k)).strip() for k in ("setting", "action", "text")):
        return None
    return cleaned

def load_transcript(path: str) -> List[Dict[str, Any]]:
    """Load transcript JSON (list of segments)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Transcript root must be a list of segments")
    return data

def adaptive_panel_generator(
    transcript: List[Dict[str, Any]],
    chunk_size: int = 15,
    max_retries: int = 3,
    max_panels: Optional[int] = None,
    max_seconds: Optional[float] = None,
    model: str = "llama3.1",
    summary_buffer_limit: int = 10,
) -> List[Dict[str, Any]]:
    """Generate panels adaptively over a transcript.

    Args:
        transcript: List of segment dicts (each with speaker, text, timestamps).
        chunk_size: Number of transcript segments to consider per attempt.
        max_retries: How many LLM retries per chunk before skipping.
        max_panels: Optional hard cap on total panels generated.
        max_seconds: Optional wall time limit (seconds) for generation loop.
        model: Ollama model name.
        summary_buffer_limit: Max number of summary lines kept for continuity.

    Returns:
        List of validated panel dicts.
    """
    panels: List[Dict[str, Any]] = []
    position = 0
    memory_buffer: List[str] = []
    story_so_far = ""
    start_time = time.time()

    while position < len(transcript):
        if max_seconds and (time.time() - start_time) > max_seconds:
            print("⏱️ Time limit reached; stopping generation.")
            break
        if max_panels and len(panels) >= max_panels:
            print("📏 Panel limit reached; stopping generation.")
            break

        chunk = transcript[position : position + chunk_size]
        if not chunk:
            break
        print(f"🧠 Processing chunk starting at index {position} (size={len(chunk)})...")

        retries = 0
        used_indices: List[int] = []
        panel: Optional[Dict[str, Any]] = None
        summary: str = ""

        while retries < max_retries:
            try:
                # On final retry, shrink chunk to increase chance of parse success.
                effective_chunk = chunk if retries < max_retries - 1 else chunk[: max(1, len(chunk)//2)]
                used_indices, raw_panel, summary = generate_panel_from_chunk(
                    effective_chunk,
                    story_so_far,
                    model=model,
                    retry_number=retries,
                )
                validated = validate_panel(raw_panel)
                if validated:
                    panel = validated
                    panels.append(panel)
                    memory_buffer.append(summary or panel.get("action", ""))
                    if len(memory_buffer) > summary_buffer_limit:
                        memory_buffer.pop(0)
                    story_so_far = "\n".join(memory_buffer)

                    # Decide advancement
                    if used_indices:
                        # Guard against indices outside effective_chunk
                        clamped = [i for i in used_indices if 0 <= i < len(effective_chunk)]
                        advance = (max(clamped) + 1) if clamped else 1
                    else:
                        advance = 1
                    position += advance
                    break
                else:
                    retries += 1
                    print(f"⚠️ Invalid/empty panel, retrying... ({retries}/{max_retries})")
                    time.sleep(0.4 + 0.1 * retries)
            except Exception as e:
                retries += 1
                print(f"⚠️ Panel generation error: {e} (retry {retries}/{max_retries})")
                time.sleep(0.4 + 0.1 * retries)

        else:  # exhausted retries without success
            print("❌ Failed after maximum retries, adaptively skipping forward.")
            # Adaptive skip: half the chunk, at least 1.
            skip = max(1, math.ceil(chunk_size / 2))
            position += skip

    return panels

def generate_panel_from_chunk(
    chunk: List[Dict[str, Any]],
    story_so_far: str,
    model: str = "llama3.1",
    retry_number: int = 0,
) -> Tuple[List[int], Optional[Dict[str, Any]], str]:
    """Send chunk + accumulated story to LLM and parse JSON response.

    Returns tuple (used_segment_indices, panel_dict_or_None, summary_str).
    """
    content = "\n".join(
        f"{i}: {seg.get('speaker', 'Unknown')}: {seg.get('text', '').strip()}" for i, seg in enumerate(chunk)
    )

    reinforcement = "" if retry_number == 0 else f"(Retry {retry_number}) Return ONLY JSON. Ensure all required keys present."

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

- Every panel must include all 8 fields (setting, characters, objects, action, mood, camera_view, lighting, text) and cannot be empty.
- Output only valid JSON — no extra text, no commentary.
{reinforcement}
"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    output = response["message"]["content"]
    return parse_adaptive_json(output)

def parse_adaptive_json(text: str) -> Tuple[List[int], Optional[Dict[str, Any]], str]:
    """Parse model output into (used_indices, panel, summary).

    Tries direct json load first; falls back to regex extraction & cleanup.
    """
    cleaned = text.strip()
    # First direct attempt
    try:
        direct = json.loads(cleaned)
        if isinstance(direct, dict):
            return (
                direct.get("used_segments", []),
                direct.get("panel", {}),
                direct.get("summary", ""),
            )
    except Exception:
        pass

    # Regex fallback
    try:
        match = re.search(r"{.*}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in text")
        json_text = match.group(0)
        # Basic trailing comma cleanup
        json_text = re.sub(r",\s*}\s*", "}", json_text)
        json_text = re.sub(r",\s*]\s*", "]", json_text)
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed root is not a dict")
        return (
            parsed.get("used_segments", []),
            parsed.get("panel", {}),
            parsed.get("summary", ""),
        )
    except Exception as e:
        print("⚠️ JSON parsing failed (fallback):", e)
        return [], None, ""

def save_panels(panels: List[Dict[str, Any]], output_path: str) -> None:
    """Persist generated panels as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(panels, f, indent=2, ensure_ascii=False)

def main() -> None:
    transcript_path = "output/clean_transcript_with_speakers.json"
    output_path = "output/adaptive_comic_script.json"

    transcript = load_transcript(transcript_path)
    panels = adaptive_panel_generator(
        transcript,
        chunk_size=15,
        max_retries=3,
        max_panels=None,  # set e.g. 40 to cap
        max_seconds=None,  # set e.g. 300 to cap time
    )
    save_panels(panels, output_path)
    print(f"✅ Script saved to {output_path} (total panels: {len(panels)})")

if __name__ == "__main__":
    main()
