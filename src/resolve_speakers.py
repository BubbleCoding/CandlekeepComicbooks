"""Speaker resolution: maps pyannote SPEAKER_XX labels to real character names.

Two strategies:
1. Direct mapping via characters.json (speaker_id field) - preferred.
2. LLM inference: scan transcript for characters addressing each other by name
   and infer the mapping automatically.

Output is the same transcript format with speaker fields replaced by real names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import anthropic


def load_characters(characters_path: str) -> Dict[str, Any]:
    with open(characters_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_direct_mapping(characters: Dict[str, Any]) -> Dict[str, str]:
    """Build SPEAKER_XX -> name map from characters.json speaker_id fields."""
    mapping = {}
    for char in characters.get("characters", []):
        sid = char.get("speaker_id", "").strip()
        name = char.get("name", "").strip()
        if sid and name:
            mapping[sid] = name
    return mapping


def infer_mapping_with_llm(
    transcript: List[Dict[str, Any]],
    character_names: List[str],
    model: str = "claude-sonnet-4-6",
) -> Dict[str, str]:
    """Use Claude to infer speaker->character mapping from transcript context."""
    client = anthropic.Anthropic()

    # Sample up to 80 segments spread across the session for context
    step = max(1, len(transcript) // 80)
    sample = transcript[::step][:80]

    lines = "\n".join(
        f"{seg['speaker']}: {seg['text'].strip()}" for seg in sample
    )

    known_speakers = sorted({seg["speaker"] for seg in transcript})
    names_list = ", ".join(character_names)

    prompt = f"""You are analyzing a Dungeons & Dragons session transcript.

The speaker labels are: {', '.join(known_speakers)}
The known character/player names are: {names_list}

Here is a sample of the transcript:
{lines}

Based on the dialogue, infer which speaker label corresponds to which character or player name.
The Dungeon Master narrates scenes and voices NPCs — look for descriptive narration or NPC dialogue.
Players typically speak in first person as their characters.

Return ONLY a JSON object mapping speaker labels to names, like:
{{"SPEAKER_00": "Dungeon Master", "SPEAKER_01": "Aria", "SPEAKER_02": "Theron"}}

If you cannot confidently identify a speaker, use their original label as the value.
Return only valid JSON, no commentary."""

    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"Warning: LLM mapping parse failed: {e}")

    return {}


def resolve_speakers(
    transcript: List[Dict[str, Any]],
    mapping: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Apply speaker mapping to transcript, leaving unmapped speakers as-is."""
    resolved = []
    for seg in transcript:
        new_seg = dict(seg)
        new_seg["speaker"] = mapping.get(seg["speaker"], seg["speaker"])
        resolved.append(new_seg)
    return resolved


def main(
    transcript_path: str = "output/clean_transcript_with_speakers.json",
    characters_path: str = "assets/characters.json",
    output_path: str = "output/resolved_transcript.json",
) -> None:
    print("Loading transcript and character data...")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    characters_data = load_characters(characters_path)
    characters = characters_data.get("characters", [])
    character_names = [c["name"] for c in characters if c.get("name")]

    # Strategy 1: direct mapping from characters.json
    mapping = build_direct_mapping(characters_data)
    unmapped = {seg["speaker"] for seg in transcript} - set(mapping.keys())

    # Strategy 2: LLM inference for anything not manually mapped
    if unmapped:
        print(f"Unmapped speakers: {unmapped}. Inferring with Claude...")
        inferred = infer_mapping_with_llm(transcript, character_names)
        # Only use inferred for speakers not already in manual mapping
        for speaker, name in inferred.items():
            if speaker not in mapping:
                mapping[speaker] = name

    print(f"Speaker mapping: {mapping}")
    resolved = resolve_speakers(transcript, mapping)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=False)

    print(f"Resolved transcript saved to {output_path}")


if __name__ == "__main__":
    main()
