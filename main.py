import argparse
import os
from pathlib import Path
from typing import Optional
from src import transcribe, resolve_speakers, CreateComicPage, generate_script, generate_images, generate_prompts, pdfMerger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Candlekeep Comicbooks - Generate comic books from D&D audio recordings"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="audio/recording.wav",
        help="Path to audio file (default: audio/recording.wav)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Output directory (default: output)",
    )

    # Step skipping — if output already exists, skip that step unless --force
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription (reuse existing transcript)")
    parser.add_argument("--skip-resolve", action="store_true", help="Skip speaker resolution")
    parser.add_argument("--skip-script", action="store_true", help="Skip comic script generation")
    parser.add_argument("--skip-prompts", action="store_true", help="Skip image prompt generation")
    parser.add_argument("--skip-images", action="store_true", help="Skip image generation")
    parser.add_argument("--skip-layout", action="store_true", help="Skip page layout and PDF steps")
    parser.add_argument("--force", action="store_true", help="Re-run all steps even if outputs exist")

    # Preview mode: only generate first 6 panels end-to-end
    parser.add_argument("--preview", action="store_true", help="Preview mode: generate only first 6 panels")

    return parser


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def should_run(flag: bool, output_file: str, force: bool) -> bool:
    """Run step if not explicitly skipped AND (output missing OR force)."""
    if flag:
        return False
    if force:
        return True
    return not Path(output_file).exists()


def run_pipeline(args: argparse.Namespace) -> None:
    o = args.output_dir
    force = args.force

    if args.preview:
        os.environ["TESTING"] = "1"
        print("Preview mode: pipeline will generate only the first 6 panels.")

    # Step 1: Transcribe
    transcript_path = f"{o}/clean_transcript_with_speakers.json"
    if should_run(args.skip_transcribe, transcript_path, force):
        audio_path = Path(args.audio_file)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {args.audio_file}")
            return
        print(f"\nStep 1: Transcribing {args.audio_file}...")
        transcribe.main(args.audio_file, o)
    else:
        print("\nStep 1: Skipping transcription (output exists).")

    # Step 2: Resolve speakers
    resolved_path = f"{o}/resolved_transcript.json"
    if should_run(args.skip_resolve, resolved_path, force):
        print("\nStep 2: Resolving speaker labels to character names...")
        resolve_speakers.main(
            transcript_path=transcript_path,
            characters_path="assets/characters.json",
            output_path=resolved_path,
        )
    else:
        print("\nStep 2: Skipping speaker resolution (output exists).")

    # Step 3: Generate script
    script_path = f"{o}/adaptive_comic_script.json"
    transcript_input = resolved_path if Path(resolved_path).exists() else transcript_path
    if should_run(args.skip_script, script_path, force):
        print("\nStep 3: Generating comic script...")
        generate_script.main(
            transcript_path=transcript_input,
            output_path=script_path,
        )
    else:
        print("\nStep 3: Skipping script generation (output exists).")

    # Step 4: Generate prompts
    prompts_path = f"{o}/adaptive_comic_script_with_prompts.json"
    if should_run(args.skip_prompts, prompts_path, force):
        print("\nStep 4: Generating FLUX image prompts...")
        generate_prompts.main(
            input_path=script_path,
            output_path=prompts_path,
        )
    else:
        print("\nStep 4: Skipping prompt generation (output exists).")

    # Step 5: Generate images
    if not args.skip_images:
        print("\nStep 5: Generating panel images with FLUX...")
        generate_images.main(
            script_path=prompts_path,
            output_dir=o,
        )
    else:
        print("\nStep 5: Skipping image generation.")

    if args.skip_layout:
        print("\nSteps 6-7: Skipping layout and PDF.")
        return

    # Step 6: Create comic pages
    print("\nStep 6: Creating comic pages...")
    CreateComicPage.main()

    # Step 7: Merge into PDF
    print("\nStep 7: Merging into PDF...")
    pdfMerger.main()

    print(f"\nDone! Comic saved to {o}/comic.pdf")


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
