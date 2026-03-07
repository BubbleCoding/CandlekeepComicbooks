# Candlekeep Comicbooks

An automated pipeline that converts TTRPG (Tabletop Role-Playing Game) audio recordings into illustrated comic books. This project uses AI for transcription, script generation, image generation, and layout to create a visual retelling of your D&D sessions.

**Master's Thesis Project**

## Features

- **Audio Transcription** - Converts session recordings to text with speaker diarization via Whisper + pyannote
- **Speaker Resolution** - Maps diarization labels (SPEAKER_00 etc.) to real character names via config or Claude inference
- **Script Generation** - Transforms transcripts into structured comic panels using Claude with hierarchical story memory
- **Prompt Engineering** - Generates rich FLUX image prompts with character appearance descriptions injected per panel
- **Image Generation** - Creates fantasy art using FLUX.1-dev with LoRA support; auto-detects CUDA / MPS / CPU
- **Speech Bubbles** - Adds oval speech bubbles (dialogue) and caption boxes (narration) directly on panel images
- **Page Layout** - Arranges panels into comic book pages (2x3 grid)
- **PDF Export** - Combines all pages into a final PDF comic book

## Requirements

- Python 3.10+
- GPU recommended:
  - NVIDIA (CUDA): 12GB+ VRAM — FLUX.1-dev runs via CPU offload on 12GB
  - Apple Silicon (MPS): 24GB unified memory recommended for FLUX.1-dev
- Anthropic API key (script generation + prompt engineering)
- HuggingFace account (pyannote diarization + FLUX.1-dev model access)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/BubbleCoding/CandlekeepComicbooks.git
cd CandlekeepComicbooks
```

### 2. Create virtual environment
```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables
```bash
# Required
export ANTHROPIC_API_KEY="your_key_here"
export HF_TOKEN="your_huggingface_token_here"

# Optional: override default FLUX model
export FLUX_MODEL="black-forest-labs/FLUX.1-dev"
```

On Windows PowerShell replace `export` with `$env:`.

### 5. Accept model terms on HuggingFace
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

Then log in on your machine:
```bash
huggingface-cli login
```

### 6. Configure your campaign
Edit `assets/characters.json` with your party's characters and speaker IDs:
```json
{
  "characters": [
    {
      "name": "Aria",
      "speaker_id": "SPEAKER_01",
      "appearance": "tall elf ranger with silver hair and green eyes",
      "class": "Ranger",
      "distinctive_features": "worn leather armor, longbow, fox companion",
      "color_palette": "green and brown"
    }
  ],
  "style": {
    "art_style": "comic book panel, bold ink outlines, flat colors, high contrast, D&D fantasy illustration",
    "negative_prompt": "deformed, blurry, extra limbs, poorly drawn hands, watermark, photorealistic"
  }
}
```

Edit `assets/session_brief.json` with session context before each run:
```json
{
  "campaign_name": "Curse of Strahd",
  "setting": "Dark gothic fantasy. The land of Barovia.",
  "tone": "Horror, mystery",
  "current_arc": "The party investigates the village of Barovia.",
  "party_members": ["Aria", "Theron"],
  "recent_events": "The party defeated wolves on the road and found a mysterious letter."
}
```

### 7. Add your LoRA (optional)
Place a FLUX-compatible comic style LoRA in `assets/lora/comic_book_illustration.safetensors`.
Good options available on [CivitAI](https://civitai.com/).

### 8. Add your audio file
Place your session recording in the `audio/` directory as a `.wav` file.

## Project Structure

```
CandlekeepComicbooks/
├── main.py                          # Main pipeline runner
├── requirements.txt                 # Python dependencies
├── assets/
│   ├── characters.json              # Character registry (fill in per campaign)
│   ├── session_brief.json           # Session context (fill in per session)
│   ├── fonts/
│   │   └── manga-temple.ttf         # Font for text overlays
│   └── lora/
│       └── *.safetensors            # LoRA weights (not in repo)
├── audio/                           # Place your .wav files here
├── output/                          # Generated outputs (gitignored)
│   ├── clean_transcript_with_speakers.json
│   ├── resolved_transcript.json
│   ├── adaptive_comic_script.json
│   ├── adaptive_comic_script_with_prompts.json
│   ├── images/
│   │   ├── imagesWithText/          # Final panel images with speech bubbles
│   │   └── imagesWithoutText/       # Raw generated images
│   ├── comicPages/                  # Combined page layouts
│   └── comic.pdf                    # Final output
└── src/
    ├── transcribe.py                # Audio → transcript with speaker labels
    ├── resolve_speakers.py          # SPEAKER_XX → character names
    ├── generate_script.py           # Transcript → comic panel script (Claude)
    ├── generate_prompts.py          # Panels → FLUX image prompts (Claude)
    ├── generate_images.py           # Prompts → images (FLUX.1-dev)
    ├── AddTextToImage.py            # Speech bubbles and caption boxes
    ├── CreateComicPage.py           # Panels → comic pages (2x3 grid)
    └── pdfMerger.py                 # Pages → PDF
```

## Usage

### Full pipeline
```bash
python main.py audio/session.wav
```

### Preview mode (first 6 panels only — fast iteration)
```bash
python main.py audio/session.wav --preview
```

### Resume from a specific step (skips steps whose output already exists)
```bash
# Redo everything from script generation onwards
python main.py audio/session.wav --skip-transcribe --skip-resolve

# Redo only image generation and layout
python main.py audio/session.wav --skip-transcribe --skip-resolve --skip-script --skip-prompts

# Force re-run all steps even if output exists
python main.py audio/session.wav --force
```

### Available flags
| Flag | Effect |
|---|---|
| `--preview` | Generate only first 6 panels end-to-end |
| `--skip-transcribe` | Reuse existing transcript |
| `--skip-resolve` | Reuse existing resolved transcript |
| `--skip-script` | Reuse existing comic script |
| `--skip-prompts` | Reuse existing image prompts |
| `--skip-images` | Skip image generation |
| `--skip-layout` | Skip page layout and PDF steps |
| `--force` | Re-run all steps regardless of existing output |
| `--output-dir` | Custom output directory (default: `output`) |

### Pipeline steps
1. **Transcribe audio** → `output/clean_transcript_with_speakers.json`
2. **Resolve speakers** → `output/resolved_transcript.json`
3. **Generate comic script** → `output/adaptive_comic_script.json`
4. **Generate FLUX prompts** → `output/adaptive_comic_script_with_prompts.json`
5. **Generate panel images** → `output/images/`
6. **Create comic pages** → `output/comicPages/`
7. **Merge into PDF** → `output/comic.pdf`

## Notes

- First run downloads FLUX.1-dev (~24GB) and other models — subsequent runs use the cache
- On NVIDIA with 12GB VRAM, FLUX runs via CPU offload (slower but fits)
- Image generation is the slowest step (~20-60 seconds per panel depending on GPU)
- A 1.5h session typically produces 50-100 panels
- Speaker IDs (SPEAKER_00 etc.) from pyannote may shift between sessions — check `resolved_transcript.json` if character attribution looks wrong

## Troubleshooting

**Out of memory (CUDA)**: FLUX CPU offload is enabled automatically; if still failing, try `FLUX_MODEL=black-forest-labs/FLUX.1-schnell` for a lighter model

**Diarization fails**: Check `HF_TOKEN` and that you accepted pyannote model terms on HuggingFace

**Wrong character names in panels**: Check `output/resolved_transcript.json` to verify speaker mapping, then adjust `speaker_id` in `assets/characters.json`

**Missing font errors**: Ensure `manga-temple.ttf` is in `assets/fonts/`

**Import errors**: Run from the project root directory, not from inside `src/`

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- FLUX.1 by Black Forest Labs
- Whisper by OpenAI
- pyannote for speaker diarization
- Claude by Anthropic

## Contact

Master's Thesis Project - Feel free to open issues or contribute!
