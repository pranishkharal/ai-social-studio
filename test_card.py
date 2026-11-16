import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

def generate_tiktok_screen_caption_card(text: str, output_path: str = "caption_card.png"):
    width = 1080
    height = 1920
    
    # Create transparent canvas for 1080x1920 video overlay
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    font_path = "C:/Windows/Fonts/arialbd.ttf"
    if not os.path.exists(font_path):
        font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, 40)
    
    # Strip emojis or replace for font rendering if needed, or render cleanly
    import re
    # We will sanitize text to render cleanly with standard fonts
    # Word wrap text so lines fit nicely within max_width (e.g. 740px)
    words = text.split()
    lines = []
    cur_line = []
    max_text_width = 740
    
    for word in words:
        test_line = " ".join(cur_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_text_width:
            cur_line.append(word)
        else:
            if cur_line:
                lines.append(" ".join(cur_line))
            cur_line = [word]
    if cur_line:
        lines.append(" ".join(cur_line))
        
    line_spacing = 16
    line_height = 46
    total_text_height = len(lines) * line_height + (len(lines) - 1) * line_spacing
    
    # Measure max line width for the box
    actual_max_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        actual_max_w = max(actual_max_w, bbox[2] - bbox[0])
        
    pad_x = 36
    pad_y = 28
    box_w = actual_max_w + (pad_x * 2)
    box_h = total_text_height + (pad_y * 2)
    
    # Placement: Safe zone just above TikTok username & caption
    # TikTok username is around y = 1530-1750
    # Safe zone for the bottom of the card is y = 1470
    box_y = 1470 - box_h
    # Center or align left with margin
    box_x = 75  # leaves plenty of room on right (ends around 75 + 800 = 875, well before 920px like icons)
    
    # Draw subtle drop shadow for card depth
    shadow_offset = 6
    draw.rounded_rectangle(
        [box_x + shadow_offset, box_y + shadow_offset, box_x + box_w + shadow_offset, box_y + box_h + shadow_offset],
        radius=20,
        fill=(0, 0, 0, 90)
    )
    
    # Draw pure white rounded card
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=20,
        fill=(255, 255, 255, 255)
    )
    
    # Draw text lines in pitch black
    cur_y = box_y + pad_y
    for line in lines:
        draw.text((box_x + pad_x, cur_y), line, font=font, fill=(10, 10, 10, 255))
        cur_y += line_height + line_spacing
        
    canvas.save(output_path, "PNG")
    print(f"Generated caption card saved to {output_path} at box_y={box_y}, box_h={box_h}")
    return output_path

if __name__ == "__main__":
    generate_tiktok_screen_caption_card(
        "N3on was SHOCKED after Ryan Garcia just ANNOUNCED that he's F1GHTING Boxing Champ Floyd Mayweather"
    )
