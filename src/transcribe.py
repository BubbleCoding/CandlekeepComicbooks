import whisper
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pyannote.audio import Pipeline
import json
import re

# Load Whisper model
whisper_model = whisper.load_model("turbo")

# Load punctuation restoration model
tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
punct_model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")
punct_model.eval()

# Load pyannote diarization pipeline (this requires your HuggingFace token)
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization_pipeline.to(torch.device("cuda"))


def transcribe_with_segments(audio_path):
    """Transcribe audio and return timestamped segments."""
    result = whisper_model.transcribe(audio_path)
    return result.get("segments", [])

def restore_punctuation(text):
    """Restore punctuation and capitalization using the punctuation model."""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = punct_model(**inputs).logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Reconstruct text properly handling SentencePiece tokens
    words = []
    current_word = ""
    
    for i, (token, label) in enumerate(zip(tokens, predictions)):
        # Skip special tokens
        if token in ['<s>', '</s>', '<pad>', '<unk>']:
            continue
            
        # Handle SentencePiece tokens (they start with ▁ which is \u2581)
        if token.startswith('▁') or token.startswith('\u2581'):
            # Start of new word
            if current_word:
                words.append(current_word)
            current_word = token.replace('▁', '').replace('\u2581', '')
        elif token.startswith('##'):
            # BERT-style subword continuation
            current_word += token[2:]
        else:
            # Regular token or continuation
            if current_word and not token.startswith('▁') and not token.startswith('\u2581'):
                current_word += token
            else:
                if current_word:
                    words.append(current_word)
                current_word = token.replace('▁', '').replace('\u2581', '')
        
        # Add punctuation based on label
        if label == 1:  # PERIOD
            current_word += "."
        elif label == 2:  # COMMA
            current_word += ","
        elif label == 3:  # QUESTION_MARK
            current_word += "?"
    
    # Don't forget the last word
    if current_word:
        words.append(current_word)
    
    # Join words and clean up
    punctuated_text = " ".join(words)
    
    # Clean up any remaining special characters and normalize spaces
    punctuated_text = re.sub(r'[▁\u2581]', '', punctuated_text)
    punctuated_text = re.sub(r'\s+', ' ', punctuated_text)
    punctuated_text = punctuated_text.strip()
    
    # Capitalize first letter
    if punctuated_text:
        punctuated_text = punctuated_text[0].upper() + punctuated_text[1:]
    
    return punctuated_text

def diarize_audio(audio_path):
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
        # Find diarization segments overlapping this transcript segment
        candidates = [d for d in diarized_segments if not (d["end"] < t_seg["start"] or d["start"] > t_seg["end"])]

        if candidates:
            # Pick the speaker with the longest overlap
            best_speaker = None
            max_overlap = 0
            for c in candidates:
                overlap_start = max(t_seg["start"], c["start"])
                overlap_end = min(t_seg["end"], c["end"])
                overlap = overlap_end - overlap_start
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

def create_clean_transcript_with_speakers(audio_path, output_json):
    print("Transcribing audio with Whisper...")
    transcript_segments = transcribe_with_segments(audio_path)

    print("Restoring punctuation...")
    for seg in transcript_segments:
        seg["text"] = restore_punctuation(seg["text"])

    print("Running speaker diarization...")
    diarized_segments = diarize_audio(audio_path)

    print("Merging transcript and diarization...")
    merged = merge_transcript_and_diarization(transcript_segments, diarized_segments)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Transcript with speakers saved to {output_json}")

def main():
    audio_file = "../audio/fullSession4.wav"
    output_file = "output/clean_transcript_with_speakers.json"
    create_clean_transcript_with_speakers(audio_file, output_file)

if __name__ == "__main__":
    main()
