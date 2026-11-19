import os
from PIL import Image, ImageDraw, ImageFont

def render_preview():
    # 1080x1920 black canvas representing vertical video
    img = Image.new("RGBA", (1080, 1920), (20, 24, 33, 255))
    draw = ImageDraw.Draw(img)
    
    # 1. Simulate 16:9 center video area (from y = 656 to 1264)
    draw.rectangle([0, 656, 1080, 1264], fill=(40, 50, 70, 255))
    draw.text((450, 940), "CENTER 16:9 VIDEO", fill=(255, 255, 255, 180))
    
    # 2. Kick Watermark Bar at y = 1160 (flush with bottom edge of center video)
    # 1082x108 bar
    draw.rectangle([0, 1160, 1080, 1268], fill=(0, 0, 0, 230))
    draw.text((100, 1195), "[KICK PIXEL LOGO]   KICK.COM/ADINROSS", fill=(83, 252, 24, 255))
    
    # 3. White On-Screen Caption Card at y = 1290 to 1480
    from platform_utils import load_font
    font = load_font(38, bold=True)
    text = "Adin Ross was SHOCKED after Ryan Garcia just ANNOUNCED that he's F1GHTING Boxing Champ Floyd Mayweather"
    
    words = text.split()
    lines = []
    cur_line = []
    for w in words:
        test = " ".join(cur_line + [w])
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= 740:
            cur_line.append(w)
        else:
            if cur_line: lines.append(" ".join(cur_line))
            cur_line = [w]
    if cur_line: lines.append(" ".join(cur_line))
    
    actual_max_w = max(draw.textbbox((0, 0), l, font=font)[2] - draw.textbbox((0, 0), l, font=font)[0] for l in lines)
    box_w = actual_max_w + 64
    box_h = len(lines) * 48 + 48
    
    box_x = 75
    box_y = 1480 - box_h  # bottom is at 1480
    
    # Drop shadow
    draw.rounded_rectangle([box_x + 6, box_y + 6, box_x + box_w + 6, box_y + box_h + 6], radius=20, fill=(0, 0, 0, 120))
    # White card
    draw.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h], radius=20, fill=(255, 255, 255, 255))
    
    cy = box_y + 24
    for l in lines:
        draw.text((box_x + 32, cy), l, font=font, fill=(10, 10, 10, 255))
        cy += 48
        
    # 4. Simulate TikTok UI
    # TikTok account name at y = 1530
    draw.text((75, 1530), "@Prada Clips", fill=(255, 255, 255, 255), font=font)
    draw.text((75, 1590), "N3on was SHOCKED after Ryan Garcia just ANNOUNCED...", fill=(220, 220, 220, 220))
    draw.rectangle([0, 1780, 1080, 1920], fill=(0, 0, 0, 200))
    draw.text((100, 1830), "Home       Friends       (+)       Inbox       Profile", fill=(200, 200, 200, 255))
    
    # TikTok right action icons (Like, comment, bookmark, share)
    for icon_y in [1100, 1240, 1380, 1520]:
        draw.ellipse([940, icon_y, 1010, icon_y + 70], fill=(50, 50, 50, 200))
        draw.text((965, icon_y + 25), "♥", fill=(255, 255, 255, 255))
        
    from pathlib import Path
    out_path = Path(__file__).resolve().parent / "test_safe_zone_preview.png"
    img.save(out_path)
    print(f"Safe zone preview rendered: box_y={box_y}, box_h={box_h}, bottom={box_y+box_h}")

if __name__ == "__main__":
    render_preview()
