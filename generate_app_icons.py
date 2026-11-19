from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from platform_utils import load_font

static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# Generate 512x512 icon
size = 512
img = Image.new("RGBA", (size, size), (11, 14, 20, 255))
draw = ImageDraw.Draw(img)

# Rounded rectangle background badge
margin = 24
draw.rounded_rectangle(
    [margin, margin, size - margin, size - margin],
    radius=80,
    fill=(15, 23, 42, 255),
    outline=(83, 252, 24, 255),
    width=12
)

# Glow effect around K
draw.rounded_rectangle(
    [margin + 20, margin + 20, size - margin - 20, size - margin - 20],
    radius=60,
    fill=(0, 0, 0, 255)
)

# Draw bold Kick "K" (cross-platform font lookup; falls back to Pillow's
# built-in default when no bold font is installed)
font = load_font(320, bold=True)
bbox = draw.textbbox((0, 0), "K", font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]
x = (size - text_w) / 2 - bbox[0]
y = (size - text_h) / 2 - bbox[1]

# Neon green glow drop
draw.text((x + 6, y + 6), "K", font=font, fill=(30, 100, 10, 255))
# Main Kick Green K
draw.text((x, y), "K", font=font, fill=(83, 252, 24, 255))

# Save 512x512
icon512_path = static_dir / "icon-512.png"
img.save(icon512_path, "PNG")

# Save 192x192
icon192_path = static_dir / "icon-192.png"
img192 = img.resize((192, 192), Image.Resampling.LANCZOS)
img192.save(icon192_path, "PNG")

# Save .ICO with multiple sizes (Windows icon, also used as favicon)
ico_path = static_dir / "app_icon.ico"
img.save(ico_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])

print(f"Generated icons: {icon512_path}, {icon192_path}, {ico_path}")
