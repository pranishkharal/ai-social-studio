import os
from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGBA", (400, 100), (255, 255, 255, 255))
draw = ImageDraw.Draw(img)

try:
    emoji_font = ImageFont.truetype("C:/Windows/Fonts/seguiemj.ttf", 36)
    draw.text((10, 20), "Hello 😳🥊", font=emoji_font, fill=(0, 0, 0, 255), embedded_color=True)
    img.save("test_emoji.png")
    print("Emoji rendered with embedded_color successfully!")
except Exception as e:
    print(f"Emoji test exception: {e}")
