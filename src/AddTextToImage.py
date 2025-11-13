from PIL import Image, ImageDraw, ImageFont

def add_text_to_panel(text, panel_image):
    text_image = generate_text_image(text, width=panel_image.width, height=120)

    result_image = Image.new('RGB', (panel_image.width, panel_image.height + text_image.height))

    # Merge the panel image and text image
    result_image.paste(panel_image, (0, 0))
    result_image.paste(text_image, (0, panel_image.height))
    return result_image

def generate_text_image(text, width=1144, height=120):
    # Required variables
    font = ImageFont.truetype(font="manga-temple.ttf", size=30)
    words = text.split()
    lines = []
    current_line = ""

    # Create a PIL drawing space for measuring
    temp_image = Image.new('RGB', (width, height), color='white')
    temp_draw = ImageDraw.Draw(temp_image)
    
    # Build lines by checking if adding each word would exceed width
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        if temp_draw.textlength(test_line, font=font) <= width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Single word is too long, add it anyway
                lines.append(word)
                current_line = ""
    
    # Don't forget the last line
    if current_line:
        lines.append(current_line)
    
    # Create the actual image
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Calculate vertical centering
    line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]  # Height of typical characters
    total_text_height = len(lines) * line_height
    y_start = (height - total_text_height) // 2
    
    # Draw each line centered horizontally
    for i, line in enumerate(lines):
        line_width = draw.textlength(line, font=font)
        x = (width - line_width) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
    
    return image