import os
import sys
import subprocess
from pathlib import Path
from config import EDITED_DIR, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_ENCODER
from database import log_event
from kick_program import generate_kick_watermark
from retry_utils import retry

# Hook banner timing (Sara's rule): the white hook banner shows for the
# first BANNER_DURATION_SECONDS of a clip, then fades out over
# BANNER_FADE_SECONDS and stays hidden for the rest of the clip. The
# Kick watermark bar is branding, not the banner, and stays visible
# throughout. The banner card is a single still image, so the overlay
# input is looped to give the fade filter a running timeline — the loop
# count is finite on purpose (loop=-1 stalls the filter graph) and far
# more frames than the 4.5 s window needs at any sane image fps.
BANNER_DURATION_SECONDS = 4.0
BANNER_FADE_SECONDS = 0.5
BANNER_LOOP_FRAMES = 600


def resolve_font_path(preferred_names):
    """
    Cross-platform font lookup: Windows, macOS and Linux font folders
    (see platform_utils.find_font). Returns the first existing path as a
    string, or None when no candidate font is installed. Windows names
    fall back to platform equivalents (DejaVu / Liberation / Noto on
    Linux, Arial / Helvetica on macOS).
    """
    from platform_utils import find_font
    return find_font(preferred_names)


def load_font(size, bold=True, emoji=False):
    """
    Load a Pillow font cross-platform: Windows fonts first, then macOS
    and Linux font dirs, then Pillow's built-in default font. Never
    crashes when no font file is installed (falls back to the default
    font). Emoji requests use the colour-emoji font when one is
    installed (Segoe UI Emoji / Apple Color Emoji / Noto Color Emoji).
    """
    from PIL import ImageFont

    if emoji:
        from platform_utils import find_emoji_font
        emoji_path = find_emoji_font()
        if emoji_path:
            try:
                return ImageFont.truetype(emoji_path, size)
            except Exception:
                pass
        # No colour-emoji font: fall through to the regular bold font.
    preferred = ["arialbd.ttf", "arial.ttf", "impact.ttf"] if bold \
        else ["arial.ttf", "arialbd.ttf"]
    font_path = resolve_font_path(preferred)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    # Final fallback: Pillow's built-in default font (always available).
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: load_default() takes no size
        return ImageFont.load_default()


def resolve_fonts_dir():
    """
    Font directory for ffmpeg's subtitles filter: the first existing
    platform font directory (Windows Fonts, macOS Fonts, or a Linux
    truetype dir), else None (filter then omits fontsdir and ffmpeg
    uses its default fontconfig lookup).
    """
    from platform_utils import font_directories
    for d in font_directories():
        if d.is_dir():
            return str(d)
    return None


def _probe_duration(path):
    """Clip duration in seconds via ffprobe, or None when unavailable."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return float(res.stdout.strip())
    except Exception:
        return None


class ClipEditor:
    def __init__(self):
        EDITED_DIR.mkdir(parents=True, exist_ok=True)
        # Cross-platform: may be None on a system with no matching font;
        # generate_screen_caption_card falls back to Pillow's default.
        self.font_path = resolve_font_path(["impact.ttf", "arial.ttf"])

    def escape_path(self, path: str) -> str:
        """Escapes a filesystem path for FFmpeg filter syntax: backslashes
        become forward slashes and a Windows drive-letter colon is escaped.
        A POSIX path passes through unchanged."""
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
            for blurred_backdrop / center_crop; for face-tracking layouts
            (speaker_tracking, camera_switch, split_screen) the banner
            drops to y 800, below the facecam zone, so it never covers
            the streamer's face.
          * "bottom": Stacked directly above Kick watermark bar (y: 1410 - box_h)
          * "none": No caption
        """
        from PIL import Image, ImageDraw, ImageFont
        
        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        font_main = load_font(38, bold=True)
        font_emoji = load_font(36, emoji=True)

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
            elif layout in ("speaker_tracking", "smart_crop", "camera_switch", "split_screen"):
                # Face-tracking layouts: the streamer's facecam fills the
                # top of the 9:16 frame (roughly y 0-760), so a banner at
                # y 260 would cover their face. Drop it below the facecam
                # zone, over the gameplay — still well above the Kick
                # watermark bar (y 1420) and the subtitle zone.
                box_y = 800
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

    @retry(max_attempts=2, retryable_exceptions=(RuntimeError,))
    def edit_clip(self, raw_clip_path: str, output_path: str, streamer: str = "adinross",
                  layout: str = None, ass_subtitle_path: str = None,
                  hook_banner: str = None, mirror: bool = True, hdr_sharpness: bool = True,
                  auto_captions: bool = False, caption_position: str = "top",
                  words: list = None, silence_trim: bool = True,
                  generate_thumbnail: bool = True, bgm: bool = False,
                  bgm_mood: str = None, category: str = None,
                  broll: bool = False, broll_query: str = None,
                  speaker_segments: list = None, trajectories: dict = None,
                  split_speakers: list = None, active_speaker: str = None,
                  encoder: str = None) -> str:
        """
        Converts horizontal stream clip to 9:16 vertical TikTok clip (1080x1920)
        Adheres 100% to Official Kick Clipping Program Rules & prxno standard:
        - Full-width side-to-side Kick watermark bar (KICK.COM/{STREAMER})
        - Compliant overlay placement (>= 25% above bottom, safe from TikTok UI)
        - Anti-Saturation Mirroring (bypasses unoriginal content flag)
        - HDR Sharpness & Contrast Boost
        - Clean On-Screen Text (Top Hook Banner): the banner shows for
          the first 4 seconds only (BANNER_DURATION_SECONDS), fades out
          over 0.5 s and stays hidden for the rest of the clip.
        - Auto-captions OFF by default (clean cinematic look): subtitles
          are only burned in when auto_captions=True is passed (the
          pipeline enables it via subtitles=True / ENABLE_SUBTITLES).
        - Dead-space removal: when Whisper word timings are passed via
          ``words`` (and silence_trim is True), long silent gaps are cut
          before rendering. Behaviour is identical when words is None.
        - Auto-thumbnail: a JPG (same basename + _thumb.jpg) is saved next
          to the render unless generate_thumbnail is False. Thumbnail
          failures never fail the render.
        - Per-streamer profiles: when layout is None, the layout comes
          from the streamer's profile (streamer_profiles.json), falling
          back to blurred_backdrop.
        - Hardware encoding (Tier 5): the encoder is auto-detected
          (NVENC > AMF > VAAPI > libx264) with a runtime test-encode;
          override with the VIDEO_ENCODER env var or encoder=....
        - Background music (Tier 5): bgm=True mixes a track from
          assets/bgm/<mood>/ under the speech with sidechain ducking;
          the mood comes from bgm_mood or the clip category. A failed
          BGM render retries once without music.
        - B-roll (Tier 5): broll=True cuts a Pexels stock clip in
          mid-clip (needs PEXELS_API_KEY). Skipped silently without one.
        - Camera-switch / split-screen layouts (Tier 5): layout=
          "camera_switch" (needs speaker_segments + trajectories) or
          "split_screen" (needs split_speakers); both fall back to
          speaker_tracking when the speaker data is missing.
        - Retried once on ffmpeg failure (@retry); the pipeline's
          resume_clip covers longer outages.
        """
        if not os.path.exists(raw_clip_path):
            raise FileNotFoundError(f"Source clip not found: {raw_clip_path}")

        # 0a. Per-streamer profile layout when none was requested.
        if layout is None:
            try:
                from streamer_profiles import get_profile
                layout = get_profile(streamer).get("layout") or "blurred_backdrop"
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Profile lookup failed, using blurred_backdrop: {e}")
                layout = "blurred_backdrop"

        # 0b. Dead-space removal: pre-cut long silences using Whisper word
        # timings when provided. Behaviour is identical when words is None.
        source_path = raw_clip_path
        temp_silence_path = None
        if silence_trim and words:
            try:
                import time
                from silence_cutter import cut_silence
                temp_silence_path = str(EDITED_DIR / f"temp_nosilence_{streamer}_{int(time.time()*1000)}.mp4")
                _, seconds_removed = cut_silence(raw_clip_path, words, temp_silence_path)
                if seconds_removed > 0:
                    source_path = temp_silence_path
                    log_event("INFO", "ClipEditor", f"Silence trim removed {seconds_removed}s of dead space")
                else:
                    # Nothing to cut: drop the untouched copy, use the original.
                    try:
                        os.remove(temp_silence_path)
                    except OSError:
                        pass
                    temp_silence_path = None
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Silence trim failed, using original clip: {e}")
                source_path = raw_clip_path
                if temp_silence_path and os.path.exists(temp_silence_path):
                    try:
                        os.remove(temp_silence_path)
                    except OSError:
                        pass
                temp_silence_path = None

        # 0c. Encoder selection (Tier 5): VIDEO_ENCODER env override or
        # the edit_clip(encoder=...) arg wins; otherwise auto-detect with
        # a runtime test-encode, so a listed-but-broken hardware encoder
        # (e.g. NVENC with no GPU) falls through to a working one.
        encoder_choice = (encoder or VIDEO_ENCODER or "auto").strip().lower()
        vaapi_input = False
        codec_args = []
        try:
            from encoder import ENCODER_ARGS, detect_encoder
            if encoder_choice not in ("auto", ""):
                codec = encoder_choice
                codec_args = list(ENCODER_ARGS.get(codec, []))
                vaapi_input = codec == "h264_vaapi"
            else:
                codec, codec_args = detect_encoder(verify=True)
                vaapi_input = codec == "h264_vaapi"
        except Exception as e:
            log_event("WARNING", "ClipEditor", f"Encoder detection failed, using libx264: {e}")
            codec, codec_args = "libx264", ["-preset", "veryfast", "-crf", "21"]
        log_event("INFO", "ClipEditor", f"Encoder: {codec} {' '.join(codec_args)}")

        # 0d. Background music (Tier 5): pick a track for the clip's
        # mood. No MP3s in assets/bgm/<mood>/ means no music — the
        # render continues unchanged.
        bgm_path = None
        if bgm:
            try:
                from bgm import pick_bgm, mood_for_category
                mood = bgm_mood or mood_for_category(category)
                bgm_path = pick_bgm(mood)
                if bgm_path:
                    log_event("INFO", "ClipEditor", f"BGM selected ({mood}): {bgm_path}")
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"BGM selection failed, continuing without music: {e}")

        # 0e. B-roll (Tier 5): fetch a stock cutaway and plan its window
        # (mid-clip, ~4 s). Needs PEXELS_API_KEY; without a key or a
        # result the render continues unchanged.
        broll_path = None
        temp_broll_path = None
        broll_start = broll_end = None
        if broll:
            try:
                import time
                from broll import search_broll, download_broll
                clip_duration = _probe_duration(source_path)
                query = broll_query or category or "crowd cheering"
                results = search_broll(query)
                if not results:
                    log_event("INFO", "ClipEditor", "No B-roll found; skipping B-roll")
                elif not clip_duration or clip_duration < 10:
                    log_event("INFO", "ClipEditor", "Clip too short for a B-roll cutaway; skipping B-roll")
                else:
                    temp_broll_path = str(EDITED_DIR / f"temp_broll_{streamer}_{int(time.time()*1000)}.mp4")
                    downloaded = download_broll(results[0]["url"], temp_broll_path)
                    if downloaded:
                        stock_len = results[0].get("duration") or 4.0
                        window = min(4.0, float(stock_len))
                        broll_start = round(clip_duration * 0.35, 2)
                        broll_end = round(min(broll_start + window, clip_duration - 1.0), 2)
                        if broll_end - broll_start >= 1.5:
                            broll_path = downloaded
                        else:
                            broll_path = None
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"B-roll setup failed, continuing without B-roll: {e}")
                broll_path = None

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
        # Tier 5 modes need speaker data the pipeline does not produce
        # yet (no diarization); without it they fall back to
        # speaker_tracking with a logged warning instead of failing.
        if layout == "camera_switch" and not (speaker_segments and trajectories):
            log_event("WARNING", "ClipEditor", "camera_switch needs speaker_segments + trajectories; falling back to speaker_tracking")
            layout = "speaker_tracking"
        if layout == "split_screen" and not split_speakers:
            log_event("WARNING", "ClipEditor", "split_screen needs split_speakers; falling back to speaker_tracking")
            layout = "speaker_tracking"
        split_divider_y = 960  # Clean 50/50 split
        base_label = "[base]"
        if layout == "speaker_tracking" or layout == "smart_crop":
            # Active Speaker Tracking & Dynamic Face Auto-Framing (Follows the talking person)
            from speaker_tracker import SpeakerTracker
            tracker = SpeakerTracker()
            try:
                smoothed_x, rw, rh, cw = tracker.analyze_video_trajectory(source_path)
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
        elif layout == "camera_switch":
            # Full-screen cuts to whoever is speaking (Tier 5). Segments
            # must cover the whole clip on the (silence-trimmed) source
            # timeline or the video ends up shorter than the audio.
            try:
                from render_modes import build_camera_switch_filter
                cam_filter, cam_out = build_camera_switch_filter(
                    speaker_segments, trajectories,
                    width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
                    source_label=source_label,
                )
                filter_complex.append(cam_filter)
                base_label = cam_out
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Camera-switch failed ({e}); falling back to smart crop")
                filter_complex.append(
                    f"{source_label}scale=-1:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}[base]"
                )
                base_label = "[base]"
            wm_y = 1420
        elif layout == "split_screen":
            # Two speakers stacked vertically, active speaker highlighted
            # (Tier 5). The highlight is static per render.
            try:
                from render_modes import build_split_screen_filter
                split_filter, split_out = build_split_screen_filter(
                    split_speakers, active_speaker=active_speaker,
                    width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
                    source_label=source_label,
                )
                filter_complex.append(split_filter)
                base_label = split_out
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Split-screen failed ({e}); falling back to smart crop")
                filter_complex.append(
                    f"{source_label}scale=-1:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}[base]"
                )
                base_label = "[base]"
            wm_y = 1420
        else:
            filter_complex.append(
                f"{source_label}scale=-1:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}[base]"
            )
            wm_y = 1420

        current_label = base_label

        # 2b. B-roll cutaway (Tier 5): full-screen stock overlay mid-clip,
        # faded in/out. Duration-preserving (overlay, not xfade), so the
        # audio stays in sync; the B-roll input is looped so it always
        # covers the window.
        if broll_path and broll_start is not None:
            try:
                from broll import build_kenburns_filter, build_broll_overlay_filter
                kb_filter = build_kenburns_filter(
                    duration=broll_end - broll_start,
                    input_label="[1:v]",
                    output_label="[broll_kb]",
                )
                overlay_filter, broll_out = build_broll_overlay_filter(
                    base_label=base_label, broll_label="[broll_kb]",
                    start=broll_start, end=broll_end,
                    output_label="[broll_out]",
                )
                filter_complex.append(kb_filter)
                filter_complex.append(overlay_filter)
                current_label = broll_out
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"B-roll overlay failed, continuing without B-roll: {e}")
                current_label = base_label

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
        # Banner timing (Sara's rule): visible for the first 4 s only,
        # 0.5 s fade-out, then hidden. The fade runs on the looped card
        # stream (a still image has no timeline of its own) and the
        # overlay enable is belt-and-braces so nothing lingers past 4.5 s.
        # The card stream is trimmed to the banner window: ffmpeg keeps
        # the output running until the longest filter input ends, so an
        # untrimmed loop would stretch the clip to the loop length.
        card_png = None
        if hook_banner and caption_position != "none":
            import time
            card_png = str(EDITED_DIR / f"temp_card_{streamer}_{int(time.time()*1000)}.png")
            self.generate_screen_caption_card(hook_banner, card_png, position=caption_position, layout=layout)
            escaped_card = self.escape_path(card_png)
            banner_end = BANNER_DURATION_SECONDS + BANNER_FADE_SECONDS
            clip_len = _probe_duration(source_path)
            card_end = banner_end if not clip_len else min(banner_end, clip_len)
            filter_complex.append(
                f"movie='{escaped_card}',loop=loop={BANNER_LOOP_FRAMES}:size=1:start=0,"
                f"trim=0:{card_end:.2f},setpts=PTS-STARTPTS,"
                f"format=rgba,fade=t=out:st={BANNER_DURATION_SECONDS:.2f}:d={BANNER_FADE_SECONDS:.2f}:alpha=1[card_faded];"
                f"{current_label}[card_faded]overlay=0:0:enable='between(t,0,{banner_end:.2f})'[with_card]"
            )
            current_label = "[with_card]"

        # 6. Dynamic Subtitles (OFF by default as requested by user)
        if auto_captions and ass_subtitle_path and os.path.exists(ass_subtitle_path):
            escaped_ass = self.escape_path(ass_subtitle_path)
            fonts_dir = resolve_fonts_dir()
            if fonts_dir:
                escaped_fonts = self.escape_path(fonts_dir)
                filter_complex.append(f"{current_label}subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'[v_final]")
            else:
                filter_complex.append(f"{current_label}subtitles='{escaped_ass}'[v_final]")
            final_v_out = "[v_final]"
        else:
            final_v_out = current_label

        # VAAPI encode needs the frames uploaded to the GPU at the end of
        # the (software) filter chain.
        if vaapi_input:
            filter_complex.append(f"{final_v_out}format=nv12,hwupload[v_vaapi]")
            final_v_out = "[v_vaapi]"

        def build_cmd(use_bgm):
            """Assemble the ffmpeg command; BGM is a separate variant so
            a BGM failure can retry without music (see below)."""
            cmd = ["ffmpeg", "-y"]
            if vaapi_input:
                from encoder import VAAPI_INPUT_ARGS
                cmd += VAAPI_INPUT_ARGS
            cmd += ["-i", source_path]
            if broll_path:
                # Input 1: B-roll, looped so it always covers its window.
                cmd += ["-stream_loop", "-1", "-i", broll_path]
            filters = list(filter_complex)
            audio_map = "0:a?"
            if use_bgm and bgm_path:
                from bgm import bgm_ffmpeg_input_args, build_ducking_filter
                cmd += bgm_ffmpeg_input_args(bgm_path)
                bgm_index = 2 if broll_path else 1
                duck_filter, _ = build_ducking_filter(bgm_label=f"[{bgm_index}:a]")
                filters.append(duck_filter)
                audio_map = "[a_out]"
            cmd += [
                "-filter_complex", ";".join(filters),
                "-map", final_v_out,
                "-map", audio_map,
                "-c:v", codec,
                *codec_args,
                "-c:a", "aac",
                "-b:a", "192k",
                "-movflags", "+faststart",
                output_path,
            ]
            return cmd

        def run_ffmpeg(cmd):
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                timeout=180,
            )

        try:
            log_event("INFO", "ClipEditor", f"Rendering compliant clip for {streamer}: {output_path}")
            res = run_ffmpeg(build_cmd(use_bgm=True))
            if res.returncode != 0 and bgm_path:
                # BGM is the usual suspect (bad MP3, or a source with no
                # audio stream for the ducking filter) — retry once
                # without music before giving up.
                log_event("WARNING", "ClipEditor", f"BGM render failed ({res.stderr[-200:]}); retrying without BGM")
                res = run_ffmpeg(build_cmd(use_bgm=False))
            if res.returncode != 0:
                log_event("ERROR", "ClipEditor", f"FFmpeg render error: {res.stderr[-300:]}")
                raise RuntimeError(f"FFmpeg failed: {res.stderr[-200:]}")

            # 7. Auto-thumbnail: best frame + hook text next to the render.
            # Failures here must never fail the render. The frame is
            # grabbed from the pre-overlay source clip (source_path),
            # not the rendered clip: the render already has the white
            # hook banner burned in, and drawing the thumbnail's own
            # text box on top of it would put two text boxes on one
            # frame. (source_path still exists here; temp-file cleanup
            # runs in the finally block below.)
            if generate_thumbnail:
                try:
                    from thumbnail import generate_thumbnail as _make_thumbnail
                    thumb_path = str(Path(output_path).with_name(Path(output_path).stem + "_thumb.jpg"))
                    _make_thumbnail(output_path, hook_banner or "", thumb_path,
                                    words=words, frame_source_path=source_path)
                    log_event("INFO", "ClipEditor", f"Thumbnail saved: {thumb_path}")
                except Exception as e:
                    log_event("WARNING", "ClipEditor", f"Thumbnail generation failed (render unaffected): {e}")

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
            if temp_silence_path and os.path.exists(temp_silence_path):
                try:
                    os.remove(temp_silence_path)
                except Exception:
                    pass
            if temp_broll_path and os.path.exists(temp_broll_path):
                try:
                    os.remove(temp_broll_path)
                except Exception:
                    pass
