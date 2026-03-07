"""Adaptive comic panel script generation.

Ingests a diarized & punctuated transcript and incrementally derives structured
comic panels using Claude. Maintains hierarchical story memory:
  - Recent panel summaries (last N panels, detailed)
  - Act summaries (compressed every ACT_SIZE panels, long-range continuity)

Campaign context and character roster from assets/ are injected into every call.
"""

from __future__ import annotations

import json
import re
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import anthropic

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

# Hierarchical memory settings
RECENT_BUFFER_LIMIT = 8   # panel summaries kept verbatim
ACT_SIZE = 10             # compress into act summary every N panels


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transcript(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Transcript root must be a list of segments")
    return data


def load_context(
    characters_path: str = "assets/characters.json",
    session_path: str = "assets/session_brief.json",
) -> str:
    """Build a campaign context block to prepend to every LLM call."""
    lines = []

    if Path(session_path).exists():
        brief = load_json(session_path)
        if brief.get("campaign_name"):
            lines.append(f"Campaign: {brief['campaign_name']}")
        if brief.get("setting"):
            lines.append(f"Setting: {brief['setting']}")
        if brief.get("tone"):
            lines.append(f"Tone: {brief['tone']}")
        if brief.get("current_arc"):
            lines.append(f"Current arc: {brief['current_arc']}")
        if brief.get("recent_events"):
            lines.append(f"Recent events: {brief['recent_events']}")

    if Path(characters_path).exists():
        chars = load_json(characters_path)
        char_lines = []
        for c in chars.get("characters", []):
            name = c.get("name", "")
            cls = c.get("class", "")
            appearance = c.get("appearance", "")
            features = c.get("distinctive_features", "")
            if name and appearance:
                desc = f"  - {name} ({cls}): {appearance}"
                if features:
                    desc += f", {features}"
                char_lines.append(desc)
        if char_lines:
            lines.append("Characters in this campaign:")
            lines.extend(char_lines)

    return "\n".join(lines)


def validate_panel(panel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not panel or not isinstance(panel, dict):
        return None
    cleaned: Dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        value = panel.get(field)
        if value is None:
            value = [] if field in ("characters", "objects") else ""
        cleaned[field] = value
    if not any(str(cleaned.get(k)).strip() for k in ("setting", "action", "text")):
        return None
    return cleaned


def compress_to_act_summary(
    panel_summaries: List[str],
    campaign_context: str,
    client: anthropic.Anthropic,
    model: str,
) -> str:
    """Summarize a batch of panel summaries into a single act summary."""
    content = "\n".join(f"- {s}" for s in panel_summaries)
    prompt = f"""{campaign_context}

The following panels have just been generated for this D&D session comic:
{content}

Write a single concise paragraph (3-5 sentences) summarizing the story arc these panels represent.
Focus on narrative progression, character development, and key events.
Return only the paragraph, no commentary."""

    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def build_story_context(
    act_summaries: List[str],
    recent_summaries: List[str],
) -> str:
    parts = []
    if act_summaries:
        parts.append("Story so far (earlier acts):")
        parts.extend(f"  Act {i+1}: {s}" for i, s in enumerate(act_summaries))
    if recent_summaries:
        parts.append("Recent panels:")
        parts.extend(f"  - {s}" for s in recent_summaries)
    return "\n".join(parts)


def generate_panel_from_chunk(
    chunk: List[Dict[str, Any]],
    story_context: str,
    campaign_context: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    retry_number: int = 0,
) -> Tuple[List[int], Optional[Dict[str, Any]], str]:
    content = "\n".join(
        f"{i}: {seg.get('speaker', 'Unknown')}: {seg.get('text', '').strip()}"
        for i, seg in enumerate(chunk)
    )

    reinforcement = (
        "" if retry_number == 0
        else f"\n(Retry {retry_number}) Return ONLY valid JSON. All required keys must be present."
    )

    prompt = f"""{campaign_context}

{story_context}

Here is the next part of the session transcript:
{content}

FILTERING RULES:
- Skip rule discussions, dice rolls, and out-of-character chatter.
- Focus only on in-character dialogue, world-building, and story-driving actions.

TASK:
Generate ONE comic panel from the most dramatically interesting moment in this chunk.
The panel should feel like a single visual scene from a comic book.

Output this exact JSON format:
{{
  "used_segments": [list of segment indices used],
  "panel": {{
    "setting": "Physical location and environment description",
    "characters": ["Character names present in this panel"],
    "objects": ["Important objects visible in the scene"],
    "action": "What is happening in the scene",
    "mood": "Emotional tone of the scene",
    "camera_view": "e.g. close-up, wide shot, over-the-shoulder, bird's eye",
    "lighting": "Lighting description",
    "text": "The dialogue or caption that appears in this panel (keep under 20 words)"
  }},
  "summary": "One sentence describing what happened in this panel, for story continuity."
}}

Every field is required. Return only valid JSON.{reinforcement}"""

    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    output = response.content[0].text
    return parse_adaptive_json(output)


def parse_adaptive_json(text: str) -> Tuple[List[int], Optional[Dict[str, Any]], str]:
    cleaned = text.strip()
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

    try:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")
        json_text = match.group(0)
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*]", "]", json_text)
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("Root is not a dict")
        return (
            parsed.get("used_segments", []),
            parsed.get("panel", {}),
            parsed.get("summary", ""),
        )
    except Exception as e:
        print(f"  JSON parse failed: {e}")
        return [], None, ""


def adaptive_panel_generator(
    transcript: List[Dict[str, Any]],
    campaign_context: str,
    chunk_size: int = 15,
    max_retries: int = 3,
    max_panels: Optional[int] = None,
    max_seconds: Optional[float] = None,
    model: str = "claude-sonnet-4-6",
) -> List[Dict[str, Any]]:
    client = anthropic.Anthropic()
    panels: List[Dict[str, Any]] = []
    position = 0
    recent_summaries: List[str] = []
    act_summaries: List[str] = []
    pending_for_act: List[str] = []
    start_time = time.time()

    while position < len(transcript):
        if max_seconds and (time.time() - start_time) > max_seconds:
            print("Time limit reached; stopping.")
            break
        if max_panels and len(panels) >= max_panels:
            print("Panel limit reached; stopping.")
            break

        chunk = transcript[position: position + chunk_size]
        if not chunk:
            break

        progress = f"{position}/{len(transcript)}"
        print(f"Processing chunk at index {position} ({progress})...")

        story_context = build_story_context(act_summaries, recent_summaries)

        retries = 0
        success = False

        while retries < max_retries:
            try:
                effective_chunk = chunk if retries < max_retries - 1 else chunk[:max(1, len(chunk) // 2)]
                used_indices, raw_panel, summary = generate_panel_from_chunk(
                    effective_chunk,
                    story_context,
                    campaign_context,
                    client,
                    model=model,
                    retry_number=retries,
                )
                validated = validate_panel(raw_panel)
                if validated:
                    panels.append(validated)
                    entry = summary or validated.get("action", "")

                    # Update recent buffer
                    recent_summaries.append(entry)
                    if len(recent_summaries) > RECENT_BUFFER_LIMIT:
                        recent_summaries.pop(0)

                    # Update act compression
                    pending_for_act.append(entry)
                    if len(pending_for_act) >= ACT_SIZE:
                        print(f"  Compressing {ACT_SIZE} panels into act summary...")
                        act_sum = compress_to_act_summary(pending_for_act, campaign_context, client, model)
                        act_summaries.append(act_sum)
                        pending_for_act.clear()

                    # Advance position
                    if used_indices:
                        clamped = [i for i in used_indices if 0 <= i < len(effective_chunk)]
                        advance = (max(clamped) + 1) if clamped else 1
                    else:
                        advance = 1
                    position += advance
                    success = True
                    break
                else:
                    retries += 1
                    print(f"  Invalid panel, retrying ({retries}/{max_retries})...")
                    time.sleep(0.5 * retries)
            except Exception as e:
                retries += 1
                print(f"  Error: {e} (retry {retries}/{max_retries})")
                time.sleep(0.5 * retries)

        if not success:
            print("  Failed after max retries, skipping forward.")
            position += max(1, math.ceil(chunk_size / 2))

    return panels


def save_panels(panels: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(panels, f, indent=2, ensure_ascii=False)


def main(
    transcript_path: str = "output/resolved_transcript.json",
    output_path: str = "output/adaptive_comic_script.json",
    characters_path: str = "assets/characters.json",
    session_path: str = "assets/session_brief.json",
) -> None:
    # Fall back to unresolved transcript if resolved doesn't exist
    if not Path(transcript_path).exists():
        transcript_path = "output/clean_transcript_with_speakers.json"
        print(f"  resolved_transcript.json not found, using {transcript_path}")

    transcript = load_transcript(transcript_path)
    campaign_context = load_context(characters_path, session_path)

    print(f"Loaded {len(transcript)} segments.")
    if campaign_context:
        print("Campaign context loaded.")

    panels = adaptive_panel_generator(
        transcript,
        campaign_context,
        chunk_size=15,
        max_retries=3,
        max_panels=None,
        max_seconds=None,
    )
    save_panels(panels, output_path)
    print(f"Script saved to {output_path} ({len(panels)} panels)")


if __name__ == "__main__":
    main()
