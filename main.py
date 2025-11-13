from src import transcribe
from src import CreateComicPage
from src import generate_script
from src import generate_images
from src import generate_prompts
from src import pdfMerger

def main():
    print(' Starting Candlekeep Comicbooks Pipeline...')
    print('\n' + '='*50)
    
    print('\n Step 1: Transcribing audio...')
    transcribe.main()
    
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
    
    print('\n' + '='*50)
    print(' Pipeline complete! Your comic is ready at output/comic.pdf')

if __name__ == '__main__':
    main()
