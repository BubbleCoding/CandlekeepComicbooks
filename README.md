# Candlekeep Comicbooks

An automated pipeline that converts TTRPG (Tabletop Role-Playing Game) audio recordings into illustrated comic books. This project uses AI for transcription, script generation, image generation, and layout to create a visual retelling of your D&D sessions.

**Master's Thesis Project**

## 🎯 Features

- **Audio Transcription** - Converts session recordings to text with speaker diarization
- **Script Generation** - Transforms transcripts into structured comic panels using LLMs
- **Prompt Engineering** - Generates optimized SDXL prompts for each panel
- **Image Generation** - Creates fantasy art using Stable Diffusion XL with LoRA weights
- **Text Overlay** - Adds dialogue to generated images
- **Page Layout** - Arranges panels into comic book pages (2x3 grid)
- **PDF Export** - Combines all pages into a final PDF comic book

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (for image generation)
- NVIDIA GPU with 12GB+ VRAM recommended
- HuggingFace account (for pyannote diarization)
- Ollama with llama3.1 model installed

## 🚀 Installation

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

### 4. Download LoRA weights
Download the required LoRA weights (if desired) and place them in `assets/lora/`:
- **Fantasy_art_XL_V1.safetensors** (~435 MB) - Primary LoRA for fantasy art
- Download from [CivitAI](https://civitai.com/) or [HuggingFace](https://huggingface.co/)

See `assets/lora/README.md` for more details.

### 5. Install Ollama
Download and install [Ollama](https://ollama.ai/), then pull the llama3.1 model:
```bash
ollama pull llama3.1
```

### 6. Setup HuggingFace token
For speaker diarization, you need a HuggingFace token with access to pyannote models:
1. Create account at [HuggingFace](https://huggingface.co/)
2. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Set your token as environment variable:
```bash
# Windows PowerShell
$env:HF_TOKEN = "your_token_here"
# Linux/Mac
export HF_TOKEN="your_token_here"
```

### 7. Add your audio file
Place your TTRPG session recording in the `audio/` directory as a `.wav` file.

## 📁 Project Structure

```
CandlekeepComicbooks/
├── main.py                      # Main pipeline runner
├── requirements.txt             # Python dependencies
├── assets/
│   ├── fonts/
│   │   └── manga-temple.ttf    # Font for text overlays
│   └── lora/
│       └── *.safetensors       # LoRA weights (not in repo)
├── audio/                       # Place your .wav files here
├── output/                      # Generated outputs (gitignored)
│   ├── *.json                  # Intermediate script files
│   ├── images/
│   │   ├── imagesWithText/     # Final panel images
│   │   └── imagesWithoutText/  # Raw generated images
│   ├── comicPages/             # Combined page layouts
│   └── comic.pdf               # Final output
└── src/
    ├── transcribe.py           # Audio → Text with speakers
    ├── generate_script.py      # Text → Comic script
    ├── generate_prompts.py     # Script → SDXL prompts
    ├── generate_images.py      # Prompts → Images
    ├── AddTextToImage.py       # Add dialogue to images
    ├── CreateComicPage.py      # Combine panels into pages
    └── pdfMerger.py           # Pages → PDF
```

## 🎮 Usage

### Run the complete pipeline:
```bash
python main.py
```

This will execute all 6 steps:
1. **Transcribe audio** → `output/clean_transcript_with_speakers.json`
2. **Generate comic script** → `output/adaptive_comic_script.json`
3. **Generate SDXL prompts** → `output/adaptive_comic_script_with_prompts.json`
4. **Generate panel images** → `output/images/`
5. **Create comic pages** → `output/comicPages/`
6. **Merge into PDF** → `output/comic.pdf`

### Run individual steps:
```bash
# Just transcription
python -m src.transcribe

# Just image generation (requires existing script)
python -m src.generate_images
```

## ⚙️ Configuration

### Modify generation settings:
- **Audio file**: Edit path in `src/transcribe.py` (line 165)
- **Chunk size**: Adjust in `src/generate_script.py` (`chunk_size` parameter)
- **LoRA model**: Change in `src/generate_images.py` (line 21)
- **Page layout**: Modify `columns, rows` in `src/CreateComicPage.py` (line 29)
- **Image quality**: Adjust negative prompts in `src/generate_prompts.py` (line 66)

## 🧪 Testing Mode

Enable testing mode in `src/generate_images.py` to generate only 6 panels:
```python
testing = True  # Line 6
```

## 📝 Notes

- First run will download SDXL model (~7GB) and other models
- Image generation is the slowest step (~30-60 seconds per panel)
- GPU memory usage peaks at ~10GB during image generation
- Average session (1 hour audio) produces 20-40 panels

## 🐛 Troubleshooting

**Out of memory errors**: Reduce batch size or use smaller models
**Diarization fails**: Check HuggingFace token and model access
**Missing font errors**: Ensure `manga-temple.ttf` is in `assets/fonts/`
**Import errors**: Make sure you're running from project root directory

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stable Diffusion XL by Stability AI
- Whisper by OpenAI
- pyannote for speaker diarization
- Ollama for local LLM inference

## 📧 Contact

Master's Thesis Project - Feel free to open issues or contribute!
