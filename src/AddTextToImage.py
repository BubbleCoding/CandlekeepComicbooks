"""Add speech bubbles and narration captions to comic panels.

Two text modes:
  - Dialogue: oval speech bubble with a downward tail, positioned in the
    upper portion of the image so the character below appears to be speaking.
  - Narration: rectangular caption box with yellow background in the
    upper-left corner, mimicking classic comic caption boxes.

Detection: if the text contains a colon (e.g. "Aria: Let's go!"), it's
treated as dialogue and the speaker name is stripped. Otherwise narration.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "assets/fonts/manga-temple.ttf"
FALLBACK_FONT_SIZE = 22
BUBBLE_PADDING = 18
BUBBLE_MARGIN = 20        # distance from image edge
TAIL_HEIGHT = 22
LINE_SPACING = 6
MAX_BUBBLE_WIDTH_RATIO = 0.72   # bubble can use up to 72% of image width


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if Path(FONT_PATH).exists():
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _text_block_size(lines: list[str], font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lh = font.getbbox("Ay")[3] - font.getbbox("Ay")[1] + LINE_SPACING
    w = max(int(draw.textlength(l, font=font)) for l in lines)
    h = lh * len(lines)
    return w, h


def _draw_speech_bubble(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    image_width: int,
) -> Tuple[int, int, int, int]:
    """Draw a speech bubble in the upper-center of the image. Returns bubble bbox."""
    text_w, text_h = _text_block_size(lines, font)
    bubble_w = text_w + BUBBLE_PADDING * 2
    bubble_h = text_h + BUBBLE_PADDING * 2

    # Center horizontally
    bx = (image_width - bubble_w) // 2
    by = BUBBLE_MARGIN

    # Oval bubble
    draw.ellipse(
        [bx, by, bx + bubble_w, by + bubble_h],
        fill="white",
        outline="black",
        width=3,
    )

    # Tail pointing downward from bubble center
    tail_cx = bx + bubble_w // 2
    tail_top = by + bubble_h - 4
    tail_bottom = tail_top + TAIL_HEIGHT
    draw.polygon(
        [
            (tail_cx - 10, tail_top),
            (tail_cx + 10, tail_top),
            (tail_cx, tail_bottom),
        ],
        fill="white",
        outline="black",
    )
    # Cover the ellipse border inside the tail base
    draw.polygon(
        [
            (tail_cx - 8, tail_top),
            (tail_cx + 8, tail_top),
            (tail_cx, tail_bottom - 2),
        ],
        fill="white",
    )

    # Draw text
    lh = font.getbbox("Ay")[3] - font.getbbox("Ay")[1] + LINE_SPACING
    draw_obj = draw
    for i, line in enumerate(lines):
        lw = int(draw_obj.textlength(line, font=font))
        tx = bx + BUBBLE_PADDING + (text_w - lw) // 2
        ty = by + BUBBLE_PADDING + i * lh
        draw_obj.text((tx, ty), line, fill="black", font=font)

    return bx, by, bx + bubble_w, by + bubble_h


def _draw_caption_box(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    image_width: int,
) -> None:
    """Draw a narration caption box in the upper-left corner."""
    text_w, text_h = _text_block_size(lines, font)
    box_w = min(text_w + BUBBLE_PADDING * 2, image_width - BUBBLE_MARGIN * 2)
    box_h = text_h + BUBBLE_PADDING * 2
    bx, by = BUBBLE_MARGIN, BUBBLE_MARGIN

    draw.rectangle(
        [bx, by, bx + box_w, by + box_h],
        fill="#FFFACD",   # lemon chiffon yellow
        outline="black",
        width=3,
    )

    lh = font.getbbox("Ay")[3] - font.getbbox("Ay")[1] + LINE_SPACING
    for i, line in enumerate(lines):
        tx = bx + BUBBLE_PADDING
        ty = by + BUBBLE_PADDING + i * lh
        draw.text((tx, ty), line, fill="black", font=font)


def add_text_to_panel(text: str, panel_image: Image.Image) -> Image.Image:
    """Overlay speech bubble or caption on the panel image.

    Returns a new image with text overlaid (same dimensions as input).
    If text is empty, returns the original image unchanged.
    """
    if not text or not text.strip():
        return panel_image

    image = panel_image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(FALLBACK_FONT_SIZE)

    max_bubble_w = int(image.width * MAX_BUBBLE_WIDTH_RATIO) - BUBBLE_PADDING * 2

    # Detect dialogue vs narration
    dialogue_match = re.match(r"^([^:]{1,30}):\s*(.+)$", text.strip(), re.DOTALL)
    if dialogue_match:
        speech_text = dialogue_match.group(2).strip()
        lines = _wrap_text(speech_text, font, max_bubble_w)
        _draw_speech_bubble(draw, lines, font, image.width)
    else:
        lines = _wrap_text(text.strip(), font, max_bubble_w)
        _draw_caption_box(draw, lines, font, image.width)

    return image
