import os
import sys
import subprocess
from pathlib import Path
from config import EDITED_DIR, VIDEO_WIDTH, VIDEO_HEIGHT
from database import log_event
from kick_program import generate_kick_watermark

class ClipEditor:
    def __init__(self):
        EDITED_DIR.mkdir(parents=True, exist_ok=True)
        self.font_path = "C:/Windows/Fonts/impact.ttf"
        if not os.path.exists(self.font_path):
            self.font_path = "C:/Windows/Fonts/arial.ttf"

    def escape_path(self, path: str) -> str:
        """Escapes Windows path for FFmpeg filter syntax."""
        p = str(Path(path).resolve()).replace("\\", "/")
        if len(p) > 1 and p[1] == ":":
            p = p[0] + "\\:" + p[2:]
        return p

    def generate_screen_caption_card(self, text: str, output_path: str, position: str = "top", layout: str = "blurred_backdrop") -> str:
        """
        Renders the viral Prestige White Sticker Caption Card:
        - Pure white rounded rectangle with subtle drop shadow
        - High-contrast black bold typography
        - Word wrapped cleanly within safe horizontal margins (max_width 720px)
        - Safe vertical positioning:
          * "top": Safe top zone (y: 260-450, blue circled area in screenshot)
          * "bottom": Stacked directly above Kick watermark bar (y: 1410 - box_h)
          * "none": No caption
        """
        from PIL import Image, ImageDraw, ImageFont
        
        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        font_main_path = "C:/Windows/Fonts/arialbd.ttf"
        if not os.path.exists(font_main_path):
            font_main_path = "C:/Windows/Fonts/arial.ttf"
        font_main = ImageFont.truetype(font_main_path, 38)

        font_emoji_path = "C:/Windows/Fonts/seguiemj.ttf"
        font_emoji = ImageFont.truetype(font_emoji_path, 36) if os.path.exists(font_emoji_path) else font_main

        def measure_char(c):
            if ord(c) > 0x2000:
                return draw.textlength(c, font=font_emoji)
            return draw.textlength(c, font=font_main)

        def measure_string(s):
            return sum(measure_char(c) for c in s)

        # Word wrap text to fit within max_text_width
        words = text.split()
        lines = []
        cur_line = []
        max_text_width = 720
        
        for word in words:
            test_line = " ".join(cur_line + [word])
            w = measure_string(test_line)
            if w <= max_text_width:
                cur_line.append(word)
            else:
                if cur_line:
                    lines.append(" ".join(cur_line))
                cur_line = [word]
        if cur_line:
            lines.append(" ".join(cur_line))
            
        line_height = 48
        pad_x = 34
        pad_y = 26
        
        actual_max_w = max((measure_string(l) for l in lines), default=400)
        box_w = int(actual_max_w + (pad_x * 2))
        box_h = int(len(lines) * line_height + (pad_y * 2))
        
        # Positioning: Flexible Top / Bottom placement
        if position == "top":
            # Top Banner placement (as circled in user screenshot)
            if layout == "blurred_backdrop":
                # In blurred_backdrop, center 16:9 video starts at y = 656
                # TikTok top header tabs occupy y = 0 to 220
                # Vertically center the caption card in this safe top region:
                box_y = int(230 + (410 - box_h) // 2)
            else:
                # Full vertical crop / speaker tracking: place cleanly below top TikTok tabs
                box_y = 260
            # Center horizontally for clean headline banner look
            box_x = int((VIDEO_WIDTH - box_w) // 2)
        elif position == "bottom":
            # Stacked directly above Kick watermark bar (at bottom, safe from TikTok username y: 1530)
            box_y = 1410 - box_h
            # Left margin 75px leaves >200px from right edge (safe from like/comment buttons)
            box_x = 75
        else:
            box_y = 260
            box_x = int((VIDEO_WIDTH - box_w) // 2)
        
        # Drop shadow for clean separation over video
        shadow_offset = 6
        draw.rounded_rectangle(
            [box_x + shadow_offset, box_y + shadow_offset, box_x + box_w + shadow_offset, box_y + box_h + shadow_offset],
            radius=20,
            fill=(0, 0, 0, 110)
        )
        
        # Pure white rounded card
        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_w, box_y + box_h],
            radius=20,
            fill=(255, 255, 255, 255)
        )
        
        # Draw text lines with dual-font support (Bold black text + full color emojis)
        cur_y = box_y + pad_y
        for l in lines:
            cur_x = box_x + pad_x
            for char in l:
                if ord(char) > 0x2000:
                    draw.text((cur_x, cur_y), char, font=font_emoji, embedded_color=True)
                    cur_x += draw.textlength(char, font=font_emoji)
                else:
                    draw.text((cur_x, cur_y), char, font=font_main, fill=(10, 10, 10, 255))
                    cur_x += draw.textlength(char, font=font_main)
            cur_y += line_height
            
        canvas.save(output_path, "PNG")
        return output_path

    def edit_clip(self, raw_clip_path: str, output_path: str, streamer: str = "adinross",
                  layout: str = "blurred_backdrop", ass_subtitle_path: str = None, 
                  hook_banner: str = None, mirror: bool = True, hdr_sharpness: bool = True,
                  auto_captions: bool = False, caption_position: str = "top") -> str:
        """
        Converts horizontal stream clip to 9:16 vertical TikTok clip (1080x1920)
        Adheres 100% to Official Kick Clipping Program Rules & prxno standard:
        - Full-width side-to-side Kick watermark bar (KICK.COM/{STREAMER})
        - Compliant overlay placement (>= 25% above bottom, safe from TikTok UI)
        - Anti-Saturation Mirroring (bypasses unoriginal content flag)
        - HDR Sharpness & Contrast Boost
        - Clean On-Screen Text (Top Hook Banner)
        - Auto-captions OFF by default (clean cinematic look)
        """
        if not os.path.exists(raw_clip_path):
            raise FileNotFoundError(f"Source clip not found: {raw_clip_path}")

        # 0. Get official full-width Kick Watermark overlay for this streamer
        watermark_png = generate_kick_watermark(streamer)
        escaped_wm = self.escape_path(watermark_png)

        filter_complex = []
        
        # 1. Anti-Saturation Mirror Filter
        source_label = "[0:v]"
        if mirror:
            filter_complex.append(f"[0:v]hflip[mirrored]")
            source_label = "[mirrored]"

        # 2. Layout Transformation (Split or Blurred Backdrop)
        split_divider_y = 960  # Clean 50/50 split
        if layout == "speaker_tracking" or layout == "smart_crop":
            # Active Speaker Tracking & Dynamic Face Auto-Framing (Follows the talking person)
            from speaker_tracker import SpeakerTracker
            tracker = SpeakerTracker()
            try:
                smoothed_x, rw, rh, cw = tracker.analyze_video_trajectory(raw_clip_path)
                crop_filter = tracker.get_ffmpeg_crop_filter(smoothed_x, 30.0, cw, rh)
                filter_complex.append(f"{source_label}{crop_filter}[base]")
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Speaker tracking fallback to smart crop: {e}")
                filter_complex.append(
                    f"{source_label}scale=-1:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}[base]"
                )
            # Full-width Kick Watermark bar placed flush right above TikTok username (y: 1530)
            wm_y = 1420
        elif layout == "split":
            filter_complex.append(
                f"{source_label}split=2[cam][game];"
                f"[cam]crop=iw*0.5:ih*0.6:0:0,scale={VIDEO_WIDTH}:{split_divider_y}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{split_divider_y}[top];"
                f"[game]crop=iw:ih*0.7:0:ih*0.3,scale={VIDEO_WIDTH}:{VIDEO_HEIGHT - split_divider_y}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT - split_divider_y}[bottom];"
                f"[top][bottom]vstack[base]"
            )
            # Watermark flush across the middle divider seam
            wm_y = split_divider_y - 28
        elif layout == "blurred_backdrop":
            filter_complex.append(
                f"{source_label}split=2[bg_raw][fg_raw];"
                f"[bg_raw]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},boxblur=25:5[bg];"
                f"[fg_raw]scale={VIDEO_WIDTH}:-1[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
            )
            # Full-width watermark placed right above TikTok username (y: 1530)
            wm_y = 1420
        else:
            filter_complex.append(
                f"{source_label}scale=-1:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}[base]"
            )
            wm_y = 1420

        current_label = "[base]"

        # 3. HDR Sharpness & Contrast Boost
        if hdr_sharpness:
            filter_complex.append(
                f"{current_label}unsharp=3:3:1.2:3:3:0.0,eq=contrast=1.06:saturation=1.12[hdr]"
            )
            current_label = "[hdr]"

        # 4. Burn Full-Width Kick Watermark Bar (Spans 100% across the screen side-to-side)
        filter_complex.append(
            f"movie='{escaped_wm}',scale={VIDEO_WIDTH}:-1[wm];"
            f"{current_label}[wm]overlay=0:{wm_y}[with_wm]"
        )
        current_label = "[with_wm]"

        # 5. Add Clean On-Screen Story Caption Card (Prestige White Sticker Card in Safe Zone)
        card_png = None
        if hook_banner and caption_position != "none":
            import time
            card_png = str(EDITED_DIR / f"temp_card_{streamer}_{int(time.time()*1000)}.png")
            self.generate_screen_caption_card(hook_banner, card_png, position=caption_position, layout=layout)
            escaped_card = self.escape_path(card_png)
            filter_complex.append(
                f"movie='{escaped_card}'[card];"
                f"{current_label}[card]overlay=0:0[with_card]"
            )
            current_label = "[with_card]"

        # 6. Dynamic Subtitles (OFF by default as requested by user)
        if auto_captions and ass_subtitle_path and os.path.exists(ass_subtitle_path):
            escaped_ass = self.escape_path(ass_subtitle_path)
            filter_complex.append(f"{current_label}subtitles='{escaped_ass}':fontsdir='C\\:/Windows/Fonts'[v_final]")
            final_v_out = "[v_final]"
        else:
            final_v_out = current_label

        full_filter_str = ";".join(filter_complex)

        cmd = [
            "ffmpeg", "-y",
            "-i", raw_clip_path,
            "-filter_complex", full_filter_str,
            "-map", final_v_out,
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "21",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            output_path
        ]

        try:
            log_event("INFO", "ClipEditor", f"Rendering compliant clip for {streamer}: {output_path}")
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                timeout=180
            )
            if res.returncode != 0:
                log_event("ERROR", "ClipEditor", f"FFmpeg render error: {res.stderr[-300:]}")
                raise RuntimeError(f"FFmpeg failed: {res.stderr[-200:]}")
                
            return output_path
        except Exception as e:
            log_event("ERROR", "ClipEditor", f"Clip rendering error: {e}")
            raise e
        finally:
            if card_png and os.path.exists(card_png):
                try:
                    os.remove(card_png)
                except Exception:
                    pass
