import os
from PIL import Image, ImageDraw, ImageFont
from config import BASE_DIR

# Official Creator Rates & Commands Database
CREATOR_RATES = {
    # $80 / 100K ($800 / 1M)
    "stake": {"cpm": 80, "cmd": "/stake", "tier": "Tier 1 Ultra"},
    "moon": {"cpm": 80, "cmd": "/moon", "tier": "Tier 1 Ultra"},
    "davido": {"cpm": 80, "cmd": "/davido", "tier": "Tier 1 Ultra"},

    # $60 / 100K ($600 / 1M)
    "musicstreamers": {"cpm": 60, "cmd": "/musicstreamers", "tier": "Tier 2 High"},
    "patriceevra": {"cpm": 60, "cmd": "/patriceevra", "tier": "Tier 2 High"},

    # $40 / 100K ($400 / 1M)
    "cuffem": {"cpm": 40, "cmd": "/cuffem", "tier": "Standard $40"},
    "cheesur": {"cpm": 40, "cmd": "/cheesur", "tier": "Standard $40"},
    "samfrank": {"cpm": 40, "cmd": "/samfrank", "tier": "Standard $40"},
    "other": {"cpm": 40, "cmd": "/other", "tier": "Standard $40"},
    "iamtiagz": {"cpm": 40, "cmd": "/iamtiagz", "tier": "Standard $40"},
    "konvy": {"cpm": 40, "cmd": "/konvy", "tier": "Standard $40"},
    "trainwreckstv": {"cpm": 40, "cmd": "/trainwreckstv", "tier": "Standard $40"},
    "messymaj": {"cpm": 40, "cmd": "/messymaj", "tier": "Standard $40"},
    "odablock": {"cpm": 40, "cmd": "/odablock", "tier": "Standard $40"},
    "mizkif": {"cpm": 40, "cmd": "/mizkif", "tier": "Standard $40"},
    "nelkboys": {"cpm": 40, "cmd": "/nelkboys", "tier": "Standard $40"},
    "kodakblack": {"cpm": 40, "cmd": "/kodakblack", "tier": "Standard $40"},
    "donfather404": {"cpm": 40, "cmd": "/donfather404", "tier": "Standard $40"},

    # $30 / 100K ($300 / 1M)
    "adinross": {"cpm": 30, "cmd": "/adinross", "tier": "High Volume $30"},
    "n3on": {"cpm": 30, "cmd": "/n3on", "tier": "High Volume $30"},
    "lospollostv": {"cpm": 30, "cmd": "/lospollostv", "tier": "High Volume $30"},
    "jidion": {"cpm": 30, "cmd": "/jidion", "tier": "High Volume $30"},
    "jackdoherty": {"cpm": 30, "cmd": "/jackdoherty", "tier": "High Volume $30"},
    "iceposeidon": {"cpm": 30, "cmd": "/iceposeidon", "tier": "High Volume $30"},

    # $27 / 100K ($270 / 1M)
    "edmatthews": {"cpm": 27, "cmd": "/edmatthews", "tier": "Tier $27"},

    # $25 / 100K ($250 / 1M)
    "hstikkytokky": {"cpm": 25, "cmd": "/hstikkytokky", "tier": "Tier $25"},
    "flaminhotmitts": {"cpm": 25, "cmd": "/flaminhotmitts", "tier": "Tier $25"},
    "sneako": {"cpm": 25, "cmd": "/sneako", "tier": "Tier $25"},
    "rampagejackson": {"cpm": 25, "cmd": "/rampagejackson", "tier": "Tier $25"},

    # $20 / 100K ($200 / 1M)
    "akademiks": {"cpm": 20, "cmd": "/akademiks", "tier": "Tier $20"},
    "6ix9ine": {"cpm": 20, "cmd": "/6ix9ine", "tier": "Tier $20"},
    "ryangarcia": {"cpm": 20, "cmd": "/ryangarcia", "tier": "Tier $20"},

    # $15 / 100K ($150 / 1M)
    "xqc": {"cpm": 15, "cmd": "/xqc", "tier": "Mega Streamer $15"},
    "clavicular": {"cpm": 15, "cmd": "/clavicular", "tier": "Mega Streamer $15"},
    "larrywheels": {"cpm": 15, "cmd": "/larrywheels", "tier": "Mega Streamer $15"},

    # $10 / 100K ($100 / 1M)
    "asmongold": {"cpm": 10, "cmd": "/asmongold", "tier": "Tier $10"},
    "blueface": {"cpm": 10, "cmd": "/blueface", "tier": "Tier $10"}
}

def get_creator_info(username: str) -> dict:
    """Returns rate and command. Defaults to /other ($40/100K) if creator is on whitelist."""
    clean = username.lower().strip().replace("@", "")
    if clean in CREATOR_RATES:
        return CREATOR_RATES[clean]
    return {"cpm": 40, "cmd": "/other", "tier": "Standard Whitelist ($40)"}

def resolve_multi_streamer_pov(streamer_list: list) -> dict:
    """
    Enforces the Multi-Streamer POV Rule:
    'If multiple streamers appear in the same clip, the clip must be submitted
    under the lowest CPM streamer shown.'
    """
    if not streamer_list:
        return get_creator_info("other")

    lowest_cpm = 999999
    selected = None
    for s in streamer_list:
        info = get_creator_info(s)
        if info["cpm"] < lowest_cpm:
            lowest_cpm = info["cpm"]
            selected = info

    return selected or get_creator_info("other")

def format_bot_submission_command(streamer: str, video_urls: list) -> str:
    """
    Formats the exact Discord bot command for payouts.
    Supports up to 10 video links separated by commas!
    """
    info = get_creator_info(streamer)
    clean_urls = [u.strip() for u in video_urls if u.strip()]
    if not clean_urls:
        return f"{info['cmd']} <PASTE_TIKTOK_OR_REELS_LINK_HERE>"
    urls_str = ",".join(clean_urls[:10])
    return f"{info['cmd']} {urls_str}"

def _candidate_display_fonts():
    """Bold display-font paths, in preference order.

    Built from the platform font directories (platform_utils): Windows
    Fonts, macOS Fonts (+ Supplemental, where Impact and Arial Bold
    live), then the Linux truetype dirs. The Docker image installs
    fonts-dejavu-core and fonts-liberation, so DejaVuSans-Bold and
    LiberationSans-Bold are available there.
    """
    from platform_utils import font_directories
    names = [
        "impact.ttf", "Impact.ttf",
        "arialbd.ttf", "Arial_Bold.ttf", "Arial Bold.ttf",
        "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf",
        "NotoSans-Bold.ttf", "FreeSansBold.ttf",
    ]
    return [str(d / name) for d in font_directories() for name in names]


def resolve_display_font_path():
    """Return the first bold display font file that exists, or None."""
    for path in _candidate_display_fonts():
        if path and os.path.exists(path):
            return path
    return None


def load_display_font(size: int):
    """Load a bold display font at `size`; never raises.

    Falls back to Pillow's built-in default font when no display font is
    installed, so overlay generation works on any platform.
    """
    font_path = resolve_display_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def get_or_create_streamer_overlay(streamer_name: str) -> str:
    """
    Returns the full-width side-to-side Kick watermark overlay (prxno / GitHub standard):
    - Width: 1082px (spans the entire horizontal width)
    - Height: 108px
    - Pixel-font neon green #53FC18 KICK logo on the left
    - Solid black bar across the entire width
    - Centered KICK.COM/{STREAMER} bold text
    - Safe from TikTok captions, icons, and description
    """
    overlays_dir = BASE_DIR / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    clean_name = streamer_name.lower().strip().replace("@", "")

    # 1. Check existing pre-made overlay (.webp or .png)
    for ext in [".webp", ".png"]:
        existing = overlays_dir / f"{clean_name}{ext}"
        if existing.exists():
            return str(existing)

    # 2. Generate dynamic pixel-perfect overlay
    logo_path = overlays_dir / "kick_pixel_logo.png"
    if not logo_path.exists():
        # Fallback to generating logo crop if not present
        adin_sample = overlays_dir / "adinross.webp"
        if adin_sample.exists():
            adin_img = Image.open(adin_sample).convert("RGBA")
            logo_crop = adin_img.crop((0, 0, 335, 108))
            logo_crop.save(logo_path)

    canvas = Image.new("RGBA", (1082, 108), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    # Draw full-width horizontal black bar
    draw.rectangle([0, 28, 1082, 104], fill=(0, 0, 0, 255))

    # Paste official pixel KICK logo on the left
    if logo_path.exists():
        logo = Image.open(logo_path).convert("RGBA")
        canvas.paste(logo, (0, 0), logo)

    # Draw centered streamer text (cross-platform font resolution; never crashes)
    font = load_display_font(42)
    
    text = f"KICK.COM/{clean_name.upper()}"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = 335 + (1082 - 335 - tw) // 2
    ty = 28 + (76 - th) // 2 - 4
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))

    out_file = str(overlays_dir / f"{clean_name}.png")
    canvas.save(out_file)
    return out_file

def generate_kick_watermark(streamer_name: str, output_path: str = None) -> str:
    """Delegates to official full-width overlay system."""
    return get_or_create_streamer_overlay(streamer_name)
