"""Scratch script: render emoji text with the platform colour-emoji font.

Safe to run on Windows, macOS and Linux — when no colour-emoji font is
installed it prints a note and exits instead of crashing.
"""
from PIL import Image, ImageDraw, ImageFont
from platform_utils import find_emoji_font

img = Image.new("RGBA", (400, 100), (255, 255, 255, 255))
draw = ImageDraw.Draw(img)

emoji_path = find_emoji_font()
if not emoji_path:
    print("No colour-emoji font installed on this platform; skipping emoji test.")
else:
    # NotoColorEmoji is a bitmap font: it only renders at its embedded
    # size (109 px), so fall back to that when the requested size fails.
    rendered = False
    for size in (36, 109):
        try:
            emoji_font = ImageFont.truetype(emoji_path, size)
            draw.text((10, 20), "Hello 😳🥊", font=emoji_font, fill=(0, 0, 0, 255), embedded_color=True)
            img.save("test_emoji.png")
            print(f"Emoji rendered with embedded_color successfully! ({emoji_path}, size {size})")
            rendered = True
            break
        except Exception as e:
            last_error = e
    if not rendered:
        print(f"Emoji test exception: {last_error}")
