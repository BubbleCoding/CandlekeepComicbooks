from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

FONT_PATH = "assets/fonts/manga-temple.ttf"
FONT_SIZE = 22
PADDING = 16
LINE_SPACING = 6
CAPTION_HEIGHT = 120


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if Path(FONT_PATH).exists():
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    return ImageFont.load_default()


def add_text_to_panel(text: str, panel_image: Image.Image) -> Image.Image:
    """Append a white caption strip below the panel image with centered text."""
    font = _load_font()
    width = panel_image.width

    # Word wrap
    draw_measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = (text or "").split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw_measure.textlength(candidate, font=font) <= width - PADDING * 2:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    # Build caption strip
    caption = Image.new("RGB", (width, CAPTION_HEIGHT), color="white")
    draw = ImageDraw.Draw(caption)

    line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1] + LINE_SPACING
    total_h = line_height * len(lines)
    y = (CAPTION_HEIGHT - total_h) // 2

    for line in lines:
        lw = draw.textlength(line, font=font)
        draw.text(((width - lw) // 2, y), line, fill="black", font=font)
        y += line_height

    result = Image.new("RGB", (width, panel_image.height + CAPTION_HEIGHT))
    result.paste(panel_image, (0, 0))
    result.paste(caption, (0, panel_image.height))
    return result
