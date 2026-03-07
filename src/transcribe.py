import whisper
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pyannote.audio import Pipeline
import json
import re
from typing import List, Dict, Any


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models():
    """Load all models lazily (called once at transcription time, not at import)."""
    device = detect_device()
    print(f"Loading models on {device}...")

    whisper_model = whisper.load_model("turbo")

    tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
    punct_model = AutoModelForTokenClassification.from_pretrained(
        "oliverguhr/fullstop-punctuation-multilang-large"
    )
    punct_model.eval()

    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diarization_pipeline.to(torch.device(device))

    return whisper_model, tokenizer, punct_model, diarization_pipeline


def transcribe_with_segments(audio_path, whisper_model):
    """Transcribe audio and return timestamped segments."""
    result = whisper_model.transcribe(audio_path)
    return result.get("segments", [])


def restore_punctuation(text: str, tokenizer, punct_model) -> str:
    """Restore punctuation & capitalization.

    Uses token classification model labels (PERIOD, COMMA, QUESTION_MARK). Falls back
    gracefully to simple heuristics if model inference fails.
    """
    original = text
    try:
        if not text or not text.strip():
            return text
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = punct_model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        id2label = getattr(punct_model.config, "id2label", {})

        words: List[str] = []
        current = ""
        for tok, pid in zip(tokens, pred_ids):
            if tok in {"<s>", "</s>", "<pad>", "<unk>"}:
                continue

            label = id2label.get(pid, "NONE")

            if tok.startswith("▁") or tok.startswith("\u2581"):
                # New word boundary — apply punctuation to current word before saving
                if label == "PERIOD" and not current.endswith('.'):
                    current += "."
                elif label == "COMMA" and not current.endswith(','):
                    current += ","
                elif label == "QUESTION_MARK" and not current.endswith('?'):
                    current += "?"
                if current:
                    words.append(current)
                current = tok.lstrip("▁\u2581")
            elif tok.startswith("##"):
                current += tok[2:]
                if label == "PERIOD" and not current.endswith('.'):
                    current += "."
                elif label == "COMMA" and not current.endswith(','):
                    current += ","
                elif label == "QUESTION_MARK" and not current.endswith('?'):
                    current += "?"
            else:
                if current:
                    current += tok
                else:
                    current = tok
                if label == "PERIOD" and not current.endswith('.'):
                    current += "."
                elif label == "COMMA" and not current.endswith(','):
                    current += ","
                elif label == "QUESTION_MARK" and not current.endswith('?'):
                    current += "?"

        if current:
            words.append(current)
        result = " ".join(words)
        result = re.sub(r"\s+", " ", result).strip()

        # Capitalize first char of each sentence
        sentences = re.split(r"([.!?])", result)
        rebuilt = ""
        for i in range(0, len(sentences), 2):
            fragment = sentences[i].strip()
            end = sentences[i + 1] if i + 1 < len(sentences) else ""
            if fragment:
                fragment = fragment[0].upper() + fragment[1:]
            rebuilt += fragment + end + (" " if end else "")
        rebuilt = rebuilt.strip()
        return rebuilt or original
    except Exception as e:
        print(f"Punctuation restore failed, fallback used: {e}")
        cleaned = original.strip()
        if cleaned and cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def diarize_audio(audio_path, diarization_pipeline):
    """Run diarization and return speaker segments with timestamps."""
    diarization = diarization_pipeline(audio_path)
    diarized_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarized_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return diarized_segments


def merge_transcript_and_diarization(transcript_segments, diarized_segments):
    """Match each transcript segment to a speaker based on overlap."""
    merged = []
    for t_seg in transcript_segments:
        candidates = [
            d for d in diarized_segments
            if not (d["end"] < t_seg["start"] or d["start"] > t_seg["end"])
        ]
        if candidates:
            best_speaker = None
            max_overlap = 0
            for c in candidates:
                overlap = min(t_seg["end"], c["end"]) - max(t_seg["start"], c["start"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = c["speaker"]
        else:
            best_speaker = "Unknown"

        merged.append({
            "start": t_seg["start"],
            "end": t_seg["end"],
            "speaker": best_speaker,
            "text": t_seg["text"]
        })
    return merged


def combine_same_speaker_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge consecutive segments from the same speaker into longer utterances."""
    if not segments:
        return []

    combined = []
    current_speaker = segments[0]["speaker"]
    current_text = segments[0]["text"]
    current_start = segments[0]["start"]
    current_end = segments[0]["end"]

    for seg in segments[1:]:
        if seg["speaker"] == current_speaker:
            current_text += " " + seg["text"]
            current_end = seg["end"]
        else:
            combined.append({
                "start": current_start,
                "end": current_end,
                "speaker": current_speaker,
                "text": current_text.strip()
            })
            current_speaker = seg["speaker"]
            current_text = seg["text"]
            current_start = seg["start"]
            current_end = seg["end"]

    combined.append({
        "start": current_start,
        "end": current_end,
        "speaker": current_speaker,
        "text": current_text.strip()
    })
    return combined


def create_clean_transcript_with_speakers(audio_path, output_json):
    whisper_model, tokenizer, punct_model, diarization_pipeline = load_models()

    print("Transcribing audio with Whisper...")
    transcript_segments = transcribe_with_segments(audio_path, whisper_model)

    print("Running speaker diarization...")
    diarized_segments = diarize_audio(audio_path, diarization_pipeline)

    print("Merging transcript and diarization...")
    merged = merge_transcript_and_diarization(transcript_segments, diarized_segments)

    print("Combining consecutive segments from same speakers...")
    combined = combine_same_speaker_segments(merged)

    print("Restoring punctuation on combined segments...")
    for seg in combined:
        seg["text"] = restore_punctuation(seg["text"], tokenizer, punct_model)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"Transcript saved to {output_json}")


def main(audio_file=None, output_dir=None):
    if audio_file is None:
        audio_file = "audio/recording.wav"
    if output_dir is None:
        output_dir = "output"

    output_file = f"{output_dir}/clean_transcript_with_speakers.json"
    create_clean_transcript_with_speakers(audio_file, output_file)


if __name__ == "__main__":
    main()
