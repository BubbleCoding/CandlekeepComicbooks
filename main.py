import argparse
from pathlib import Path
from typing import Optional
from src import transcribe, CreateComicPage, generate_script, generate_images, generate_prompts, pdfMerger


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser (kept separate for reuse/testing)."""
    parser = argparse.ArgumentParser(
        description='Candlekeep Comicbooks - Generate comic books from audio recordings'
    )
    parser.add_argument(
        'audio_file',
        nargs='?',
        default='audio/recording.wav',
        help='Path to audio file (default: audio/recording.wav)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='output',
        help='Output directory for generated files (default: output)'
    )
    # Future flags could be added here (e.g., --skip-images, --max-panels)
    return parser


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments (argv injected for testing)."""
    parser = build_parser()
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full comic generation pipeline given parsed args."""
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        print("   Please place your audio file in the audio/ directory or pass a valid path.")
        return

    print(' Starting Candlekeep Comicbooks Pipeline...')
    print(f' Audio file: {args.audio_file}')
    print('\n' + '=' * 50)

    print('\n Step 1: Transcribing audio...')
    transcribe.main(args.audio_file, args.output_dir)

    print('\n Step 2: Generating comic script...')
    generate_script.main()

    print('\n Step 3: Generating SDXL prompts...')
    generate_prompts.main()

    print('\n Step 4: Generating panel images...')
    generate_images.main()

    print('\n Step 5: Creating comic pages...')
    CreateComicPage.main()

    print('\n Step 6: Merging pages into PDF...')
    pdfMerger.main()

    print('\n' + '=' * 50)
    print(f' Pipeline complete! Your comic is ready at {args.output_dir}/comic.pdf')


def main(argv: Optional[list] = None) -> None:
    """Entry point: parse arguments then run pipeline."""
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == '__main__':
    main()
