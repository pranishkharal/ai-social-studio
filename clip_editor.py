import os
import sys
import subprocess
from pathlib import Path
from config import (BANNER_TIMING, EDITED_DIR, QUOTE_LINE_ENABLED,
                VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_ENCODER, TRACKING_TARGET)
from database import log_event
from face_avoidance import (choose_banner_y, choose_quote_y,
                            face_region_for_clip, mirror_face_boxes,
                            mirror_trajectory)
from kick_program import generate_kick_watermark
from retry_utils import retry
from themes import get_theme, hook_colors, progress_bar as theme_progress_bar

# Hook banner timing: "4s" is Sara's rule — the white hook banner shows
# for the first BANNER_DURATION_SECONDS of a clip, then fades out over
# BANNER_FADE_SECONDS and stays hidden for the rest of the clip. The
# "whole_clip" mode (FEATURE 7, config BANNER_TIMING) keeps the banner
# on screen for the whole clip, like the TikTok clipping accounts
# (@zysno / @godsclipper / @kicksclipping). The Kick watermark bar is
# branding, not the banner, and stays visible throughout in both modes.
# The banner card is a single still image, so the overlay input is
# looped to give the fade filter a running timeline — the loop count is
# finite on purpose (loop=-1 stalls the filter graph) and far more
# frames than the 4.5 s window needs at any sane image fps. In
# whole-clip mode the loop count scales with the clip duration, and
# when the duration cannot be probed the card is trimmed to
# WHOLE_CLIP_FALLBACK_SECONDS (far longer than any clip).
BANNER_DURATION_SECONDS = 4.0
BANNER_FADE_SECONDS = 0.5
BANNER_LOOP_FRAMES = 600
WHOLE_CLIP_FALLBACK_SECONDS = 120.0


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


def render_timeout_seconds(duration):
    """ffmpeg render timeout scaled to the source clip duration: roughly
    20x realtime, at least 180 s (the old fixed value — a 1080p60 render
    of a 30 s clip failed twice on it), capped at 1200 s. A missing
    duration (probe failed) keeps the 180 s floor."""
    if not duration:
        return 180
    return max(180, min(1200, int(duration * 20)))


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

    def _measure_caption_box(self, text: str):
        """(lines, box_w, box_h) the hook banner card would have for
        this text, without drawing anything. The wrap uses the same
        fonts and max width as generate_screen_caption_card, so
        face-aware placement in edit_clip() and the drawn card agree
        exactly on the card's size."""
        from PIL import Image, ImageDraw

        canvas = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
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
        return lines, box_w, box_h

    def generate_screen_caption_card(self, text: str, output_path: str, position: str = "top", layout: str = "blurred_backdrop", banner_y: int = None, theme: str = None) -> str:
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
        - banner_y (FEATURE 1): explicit top-edge y for the "top"
          position, chosen by face-aware placement in edit_clip() so
          the card dodges the face as rendered. None (default) keeps
          the fixed layout behaviour above; it is ignored for the
          "bottom" position.
        - theme (feature 21): a themes.json look name — the hook-box
          fill and text colours come from the theme. None (default)
          is the "clean" theme: white box, black text, exactly as
          before.
        """
        from PIL import Image, ImageDraw, ImageFont

        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        font_main = load_font(38, bold=True)
        font_emoji = load_font(36, emoji=True)

        # Theme hook-box colours (default theme = white box, black text).
        box_fill, text_fill = hook_colors(theme)

        lines, box_w, box_h = self._measure_caption_box(text)
        line_height = 48
        pad_x = 34
        pad_y = 26

        # Positioning: Flexible Top / Bottom placement
        if position == "top":
            # Top Banner placement (as circled in user screenshot)
            if banner_y is not None:
                # Face-aware placement (FEATURE 1): explicit y chosen
                # in edit_clip() so the card dodges the rendered face.
                box_y = int(banner_y)
            elif layout == "blurred_backdrop":
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
            elif layout == "split":
                # Gaming Split Layout: Streamer occupies top 35% (y 0-672).
                # The face is centered in the lower-middle of the facecam.
                # Place the viral hook banner cleanly in the upper safe space (y 110),
                # safely above the streamer's head and never covering their face or gameplay.
                box_y = 110
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
        
        # Theme hook-box card (default theme: pure white rounded card)
        draw.rounded_rectangle(
            [box_x, box_y, box_x + box_w, box_y + box_h],
            radius=20,
            fill=box_fill
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
                    draw.text((cur_x, cur_y), char, font=font_main, fill=text_fill)
                    cur_x += draw.textlength(char, font=font_main)
            cur_y += line_height
            
        canvas.save(output_path, "PNG")
        return output_path

    def _measure_quote_block(self, text: str):
        """(lines, block_h) the quote card would have for this text,
        without drawing anything. The wrap/truncation matches
        generate_quote_card exactly, so face-aware placement in
        edit_clip() and the drawn card agree on the block's size."""
        from PIL import Image, ImageDraw

        canvas = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        font = load_font(44, bold=True)
        max_text_width = 900

        def _wrap(words):
            lines = []
            cur_line = []
            for word in words:
                trial = " ".join(cur_line + [word])
                if draw.textlength(trial, font=font) <= max_text_width:
                    cur_line.append(word)
                else:
                    if cur_line:
                        lines.append(" ".join(cur_line))
                    cur_line = [word]
            if cur_line:
                lines.append(" ".join(cur_line))
            return lines

        words = text.split()
        lines = _wrap(words)
        # At most 2 lines: drop words from the end until it fits.
        truncated = False
        while len(lines) > 2 and len(words) > 1:
            words = words[:-1]
            truncated = True
            lines = _wrap(words)
        if truncated and lines:
            marked = lines[-1] + "…"
            if draw.textlength(marked, font=font) <= max_text_width:
                lines[-1] = marked

        line_height = 56
        return lines, len(lines) * line_height

    def generate_quote_card(self, text: str, output_path: str, quote_y: int = None) -> str:
        """
        Renders a quote line (FEATURE 2) as a transparent full-frame
        PNG: white bold text with a black outline — no white box, the
        box is the headline's job — word-wrapped to at most 2 lines
        and centred in the lower third (around y 1180-1300, above the
        Kick watermark bar at y 1420). One static line, never
        word-by-word captions: subtitles stay off by default.
        quote_y (FEATURE 1): explicit centre y, chosen by
        face-aware placement in edit_clip() so the line dodges the
        face as rendered and the headline. None (default) centres
        the block on y 1240.
        """
        from PIL import Image, ImageDraw

        canvas = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        font = load_font(44, bold=True)
        stroke_width = 3

        lines, block_h = self._measure_quote_block(text)

        # Centre the text block on y 1240 (lower third, above the
        # Kick bar): 1 line spans ~1212-1256, 2 lines ~1184-1296.
        # quote_y overrides the centre when face-aware placement
        # chose a different spot.
        centre_y = quote_y if quote_y is not None else 1240
        line_height = 56
        top = centre_y - block_h // 2
        for i, line in enumerate(lines):
            line_w = draw.textlength(line, font=font)
            x = int((VIDEO_WIDTH - line_w) // 2)
            y = top + i * line_height
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255),
                      stroke_width=stroke_width, stroke_fill=(0, 0, 0, 255))

        canvas.save(output_path, "PNG")
        return output_path

    @retry(max_attempts=2, retryable_exceptions=(RuntimeError,))
    def edit_clip(self, raw_clip_path: str, output_path: str, streamer: str = "adinross",
                  layout: str = None, ass_subtitle_path: str = None,
                  hook_banner: str = None, banner_timing: str = None,
                  quote_line: str = None, quote_line_enabled: bool = None,
                  mirror: bool = True, hdr_sharpness: bool = True,
                  auto_captions: bool = False, caption_position: str = "top",
                  words: list = None, silence_trim: bool = True,
                  generate_thumbnail: bool = True, bgm: bool = False,
                  bgm_mood: str = None, category: str = None,
                  broll: bool = False, broll_query: str = None,
                  speaker_segments: list = None, trajectories: dict = None,
                  split_speakers: list = None, active_speaker: str = None,
                  encoder: str = None, tracking_target: str = None,
                  voiceover_path: str = None, voiceover_at: float = 0.0,
                  theme: str = None, meme_mood: str = None,
                  meme_at: float = None,
                  custom_crop: dict = None, trim_start: float = None,
                  trim_end: float = None) -> str:
        """
        Converts horizontal stream clip to 9:16 vertical TikTok clip (1080x1920)
        Adheres 100% to Official Kick Clipping Program Rules & prxno standard:
        - Full-width side-to-side Kick watermark bar (KICK.COM/{STREAMER})
        - Compliant overlay placement (>= 25% above bottom, safe from TikTok UI)
        - Anti-Saturation Mirroring (bypasses unoriginal content flag)
        - HDR Sharpness & Contrast Boost
        - Clean On-Screen Text (Top Hook Banner): by default the banner
          shows for the first 4 seconds only (BANNER_DURATION_SECONDS),
          fades out over 0.5 s and stays hidden for the rest of the
          clip. banner_timing="whole_clip" keeps it on screen for the
          whole clip instead (FEATURE 7; None reads config
          BANNER_TIMING, invalid values fall back to "4s"). The quote
          line (quote_line=..., FEATURE 2) follows the same timing.
        - Quote line (FEATURE 2): one static spoken line under the
          headline, white bold text with a black outline in the lower
          third — a single caption, NOT word-by-word subtitles (those
          stay off by default). OFF by default since v1.4.1: the line
          is drawn only when quote_line_enabled=True is passed or
          config QUOTE_LINE_ENABLED is true (None means the config
          setting). Disabled when quote_line is empty.
        - Face-aware banner placement (FEATURE 1): for face-tracking
          layouts the banner and quote line dodge the streamer's
          face as rendered (sampled face boxes mapped through the
          crop trajectory); without detected faces both keep their
          fixed positions (banner y 800, quote centred on y 1240).
          The crop trajectory is mirrored when mirror=True so the
          crop lands on the face as rendered, not its mirror image.
        - Tracking target (FEATURE 3): tracking_target="body" makes
          the speaker_tracking layout follow the whole body
          (MediaPipe Pose) instead of the face; it falls back to
          face tracking when no body is found. None means the
          TRACKING_TARGET setting (default "face").
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
          BGM render retries once without music. The built-in track
          assets/bgm/chill/soft_pad.mp3 (generated on first use if
          missing) is the fallback for all moods.
        - Render timeout: the ffmpeg timeout scales with the source
          clip duration (~20x realtime, 180 s floor, 1200 s cap) so a
          1080p60 render no longer dies on the old fixed 180 s limit.
        - Loudness normalisation (gta6-clipper feature 10): every
          render is normalised to -14 LUFS with two-pass linear
          loudnorm (the source audio is measured first, then the
          measured values are applied in the render), and a safety
          pass re-normalises only the audio (video copied) if the
          finished clip lands more than 1.5 LU off.
        - B-roll (Tier 5): broll=True cuts a Pexels stock clip in
          mid-clip (needs PEXELS_API_KEY). Skipped silently without one.
        - Camera-switch / split-screen layouts (Tier 5): layout=
          "camera_switch" (needs speaker_segments + trajectories) or
          "split_screen" (needs split_speakers); both fall back to
          speaker_tracking when the speaker data is missing.
        - Voice-over (gta6-clipper feature 7): voiceover_path= a
          recorded line (Sara's own commentary) mixed in at
          voiceover_at= seconds as a post-pass after the render —
          the game audio ducks ~15 dB under it while she talks and
          the video is copied untouched. A line longer than the rest
          of the clip is trimmed, not dropped. A failed voice-over
          pass logs a warning and keeps the render without it.
        - Theme (gta6-clipper feature 21): theme= a themes.json look
          name (or the CLIP_THEME env var). It sets the hook-box
          colours and, when the theme enables it, a progress bar
          along the bottom of the frame. None is the "clean" theme:
          white box, black text, no progress bar — the default look
          is unchanged.
        - Meme sounds (gta6-clipper feature 6): meme_mood= one of the
          9 moods in meme_sounds.py applies a synthesised sound +
          visual gag at the payoff as a post-pass after the render.
          OPT-IN ONLY, default OFF: pass meme_mood= explicitly, or
          set MEME_SOUNDS_ENABLED=1 with MEME_MOOD. At most one per
          clip, never in the first 3 s. A failed meme pass logs a
          warning and keeps the render without it.
        - Speaker layout (gta6-clipper feature 16): when
          speaker_segments carry "split" entries (fast line trades,
          as produced by speaker_framing), camera_switch renders
          those spans as a split screen instead of cutting.
        - Sentence snapping (gta6-clipper feature 21): the silence
          trim snaps the clip bounds to sentence boundaries, so a
          trimmed clip never stops mid-thought.
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
                # Sentence snapping (feature 21): the trim snaps the clip
                # bounds to sentence boundaries, so a trimmed clip never
                # stops mid-thought.
                _, seconds_removed = cut_silence(raw_clip_path, words, temp_silence_path,
                                                 snap=True)
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

        # 0f. Loudness (gta6-clipper feature 10): measure the source
        # audio now (after the silence trim, so the measurement matches
        # the audio actually rendered) so the render can apply two-pass
        # linear loudnorm (measured values + linear=true). A failed
        # measure (no audio stream, silent audio) means no loudnorm
        # filter — the render is unchanged and the post-render safety
        # pass still runs.
        loudnorm_stats = None
        if not custom_crop:
            try:
                from loudness import measure_loudnorm_stats
                loudnorm_stats = measure_loudnorm_stats(source_path)
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Loudness measure failed, rendering without loudnorm: {e}")

        # 0. Get official full-width Kick Watermark overlay for this streamer
        watermark_png = generate_kick_watermark(streamer)
        escaped_wm = self.escape_path(watermark_png)

        filter_complex = []
        
        # 1. Anti-Saturation Mirror Filter
        source_label = "[0:v]"
        if mirror and not custom_crop:
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
        split_divider_y = 672  # 35% Streamer on Top, 65% Gameplay on Bottom (Pro Gaming Layout)
        base_label = "[base]"
        # Face-aware placement (FEATURE 1): sampled face boxes + the
        # trajectory actually used for the crop, both in rendered
        # (post-hflip) coordinates. They stay None for non-face-
        # tracking layouts and on any analysis failure, in which
        # case the banner and quote keep their fixed positions.
        face_boxes = None
        face_traj = None
        face_crop_w = None
        face_raw_h = None
        if layout == "speaker_tracking" or layout == "smart_crop":
            # Active Speaker Tracking & Dynamic Face Auto-Framing (Follows the talking person)
            from speaker_tracker import SpeakerTracker
            # FEATURE 3: what speaker_tracking locks onto — "face"
            # (default) or "body" (MediaPipe Pose full-body tracking).
            # None means the TRACKING_TARGET setting (default "face").
            _target = tracking_target if tracking_target is not None else TRACKING_TARGET
            _target = str(_target).strip().lower()
            if _target not in ("face", "body"):
                _target = "face"
            tracker = SpeakerTracker()
            used_body = False
            try:
                if _target == "body":
                    # Full-body tracking (MediaPipe Pose). Falls back to
                    # face tracking when no body is found in the clip or
                    # the body tracker fails — never fails the render.
                    body_tracker = None
                    try:
                        from body_tracker import BodyTracker
                        body_tracker = BodyTracker()
                        smoothed_x, rw, rh, cw = body_tracker.analyze_video_trajectory(source_path)
                        if not body_tracker.body_found:
                            raise RuntimeError("no body detected in clip")
                        used_body = True
                    except Exception as e:
                        log_event("WARNING", "ClipEditor", f"Body tracking unavailable ({e}); falling back to face tracking")
                    finally:
                        if body_tracker is not None:
                            body_tracker.close()
                if not used_body:
                    smoothed_x, rw, rh, cw = tracker.analyze_video_trajectory(source_path)
                # Mirror fix: the tracker analyses the UNMIRRORED
                # source, but the crop is applied AFTER hflip when
                # mirror=True (the default). An unmirrored crop_x
                # frames the mirror image of the face — for an
                # off-centre facecam the face is not even in the
                # rendered frame. Mirror the trajectory so the crop
                # lands on the face as rendered.
                if mirror:
                    smoothed_x = mirror_trajectory(smoothed_x, rw - cw)
                crop_filter = tracker.get_ffmpeg_crop_filter(smoothed_x, 30.0, cw, rh)
                filter_complex.append(f"{source_label}{crop_filter}[base]")
                # Sampled face boxes for banner/quote placement
                # (FEATURE 1) — face path only. On the body path
                # face_boxes stays None and the banner/quote keep
                # their fixed positions (BodyTracker has no
                # analyze_face_boxes; a body box spans most of the
                # frame, so dodging it is meaningless).
                if not used_body:
                    try:
                        face_boxes = tracker.analyze_face_boxes(source_path)
                        if mirror:
                            face_boxes = mirror_face_boxes(face_boxes, rw)
                        face_traj = smoothed_x
                        face_crop_w, face_raw_h = cw, rh
                    except Exception as e:
                        log_event("WARNING", "ClipEditor", f"Face-box sampling failed, banner placement stays fixed: {e}")
                        face_boxes = None
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Speaker tracking fallback to smart crop: {e}")
                filter_complex.append(
                    f"{source_label}scale=-1:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}[base]"
                )
            finally:
                # Release the MediaPipe detector (no-op on Haar/YOLO);
                # otherwise it logs an ignored exception at shutdown.
                tracker.close()
            # Full-width Kick Watermark bar placed flush right above TikTok username (y: 1530)
            wm_y = 1420
        elif layout == "split":
            # Auto-detect streamer facecam corner if no custom crop was supplied
            if not custom_crop:
                try:
                    from speaker_tracker import detect_facecam_box
                    detected = detect_facecam_box(raw_clip_path)
                    if detected and "cam" in detected:
                        custom_crop = detected
                        log_event("INFO", "ClipEditor", f"Auto-detected streamer facecam in {detected.get('corner', 'corner')}: {detected['cam']}")
                except Exception as e:
                    log_event("WARNING", "ClipEditor", f"Facecam auto-detect skipped: {e}")

            if custom_crop and ("cam" in custom_crop or "game" in custom_crop):
                cc = custom_crop.get("cam", {"x": 0.65, "y": 0.05, "w": 0.3, "h": 0.35})
                cg = custom_crop.get("game", {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8})
                cx = max(0.0, min(0.95, float(cc.get("x", 0.65))))
                cy = max(0.0, min(0.95, float(cc.get("y", 0.05))))
                cw = max(0.05, min(1.0 - cx, float(cc.get("w", 0.3))))
                ch = max(0.05, min(1.0 - cy, float(cc.get("h", 0.35))))

                gx = max(0.0, min(0.95, float(cg.get("x", 0.1))))
                gy = max(0.0, min(0.95, float(cg.get("y", 0.1))))
                gw = max(0.05, min(1.0 - gx, float(cg.get("w", 0.8))))
                gh = max(0.05, min(1.0 - gy, float(cg.get("h", 0.8))))

                top_flip = ",hflip" if mirror else ""
                bot_flip = ",hflip" if mirror else ""

                filter_complex.append(
                    f"[0:v]split=2[cam_in][game_in];"
                    f"[cam_in]crop=iw*{cw:.4f}:ih*{ch:.4f}:iw*{cx:.4f}:ih*{cy:.4f},scale={VIDEO_WIDTH}:{split_divider_y}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{split_divider_y}{top_flip}[top];"
                    f"[game_in]crop=iw*{gw:.4f}:ih*{gh:.4f}:iw*{gx:.4f}:ih*{gy:.4f},scale={VIDEO_WIDTH}:{VIDEO_HEIGHT - split_divider_y}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT - split_divider_y}{bot_flip}[bottom];"
                    f"[top][bottom]vstack[base]"
                )
            else:
                top_flip = ",hflip" if mirror else ""
                bot_flip = ",hflip" if mirror else ""
                filter_complex.append(
                    f"[0:v]split=2[cam][game];"
                    f"[cam]crop=iw*0.5:ih*0.6:0:0,scale={VIDEO_WIDTH}:{split_divider_y}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{split_divider_y}{top_flip}[top];"
                    f"[game]crop=iw:ih*0.7:0:ih*0.3,scale={VIDEO_WIDTH}:{VIDEO_HEIGHT - split_divider_y}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT - split_divider_y}{bot_flip}[bottom];"
                    f"[top][bottom]vstack[base]"
                )
            # Watermark flush across the middle divider seam
            wm_y = split_divider_y - 28
        elif layout in ("full", "custom_full"):
            if custom_crop and ("game" in custom_crop or "cam" in custom_crop):
                cg = custom_crop.get("game") or custom_crop.get("cam")
                gx = max(0.0, min(0.95, float(cg.get("x", 0.2))))
                gy = max(0.0, min(0.95, float(cg.get("y", 0.0))))
                gw = max(0.05, min(1.0 - gx, float(cg.get("w", 0.5625))))
                gh = max(0.05, min(1.0 - gy, float(cg.get("h", 1.0))))
                full_flip = ",hflip" if mirror else ""
                filter_complex.append(
                    f"[0:v]crop=iw*{gw:.4f}:ih*{gh:.4f}:iw*{gx:.4f}:ih*{gy:.4f},scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}{full_flip}[base]"
                )
            else:
                filter_complex.append(
                    f"{source_label}scale=-1:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}[base]"
                )
            wm_y = 1420
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
            # Feature 16: when the segments carry "split" entries (fast
            # line trades, as produced by speaker_framing), those spans
            # render as a split screen instead of cutting.
            try:
                if any(seg.get("split") for seg in (speaker_segments or [])):
                    from render_modes import build_speaker_layout_filter
                    cam_filter, cam_out = build_speaker_layout_filter(
                        speaker_segments, trajectories,
                        width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
                        source_label=source_label,
                    )
                else:
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

        # Face-aware placement for camera_switch / split_screen
        # (FEATURE 1): those layouts crop per speaker, so there is no
        # single crop trajectory to reuse — sample the faces and map
        # them through the speaker_tracking trajectory as an
        # approximation of where faces appear in the rendered frame.
        # Best effort, like the speaker_tracking branch above.
        if layout in ("camera_switch", "split_screen") and face_boxes is None:
            _tracker = None
            try:
                from speaker_tracker import SpeakerTracker
                _tracker = SpeakerTracker()
                _sx, _rw, _rh, _cw = _tracker.analyze_video_trajectory(source_path)
                if mirror:
                    _sx = mirror_trajectory(_sx, _rw - _cw)
                face_boxes = _tracker.analyze_face_boxes(source_path)
                if mirror:
                    face_boxes = mirror_face_boxes(face_boxes, _rw)
                face_traj = _sx
                face_crop_w, face_raw_h = _cw, _rh
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Face-box sampling failed, banner placement stays fixed: {e}")
                face_boxes = None
            finally:
                if _tracker is not None:
                    _tracker.close()

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

        # 3. HDR Contrast & Vibrancy Boost (Fast hardware-friendly filter)
        if hdr_sharpness:
            if custom_crop:
                filter_complex.append(
                    f"{current_label}eq=contrast=1.06:saturation=1.10[hdr]"
                )
            else:
                filter_complex.append(
                    f"{current_label}unsharp=3:3:0.8:3:3:0.0,eq=contrast=1.06:saturation=1.12[hdr]"
                )
            current_label = "[hdr]"

        # 4. Burn Full-Width Kick Watermark Bar (Spans 100% across the screen side-to-side)
        filter_complex.append(
            f"movie='{escaped_wm}',scale={VIDEO_WIDTH}:-1[wm];"
            f"{current_label}[wm]overlay=0:{wm_y}[with_wm]"
        )
        current_label = "[with_wm]"

        # 5. Add Clean On-Screen Story Caption Card (Prestige White Sticker Card in Safe Zone)
        # Banner timing (FEATURE 7): "4s" (Sara's rule — visible for
        # the first 4 s, 0.5 s fade-out, then hidden) or "whole_clip"
        # (the headline stays on screen for the whole clip, like the
        # TikTok clipping accounts). None reads config BANNER_TIMING;
        # invalid values fall back to "4s". The quote line (5b) follows
        # the same timing as the banner. The fade runs on the looped
        # card stream (a still image has no timeline of its own) and
        # the overlay enable is belt-and-braces so nothing lingers past
        # 4.5 s in 4s mode. The card stream is trimmed to the banner
        # window: ffmpeg keeps the output running until the longest
        # filter input ends, so an untrimmed loop would stretch the
        # clip to the loop length. No fade-out is ever applied to the
        # main video or audio streams — the clip ends abruptly and
        # loops, like the TikTok accounts' clips.
        timing = str(banner_timing or BANNER_TIMING or "4s").strip().lower()
        if timing not in ("4s", "whole_clip"):
            timing = "4s"
        # Quote line: an explicit argument wins; otherwise the
        # QUOTE_LINE_ENABLED setting decides (default off since v1.4.1).
        show_quote = (QUOTE_LINE_ENABLED if quote_line_enabled is None
                      else bool(quote_line_enabled))
        clip_len = _probe_duration(source_path)

        def _card_filter(escaped_png, card_label, out_label, base_label):
            """Filter for one timed overlay card (headline or quote).

            4s mode: the card is trimmed to 4.5 s, fades out over the
            last 0.5 s, and the overlay is enabled only for 0-4.5 s.
            whole_clip mode: no fade and no enable window — the card
            is looped (loop count scales with the clip duration) and
            trimmed to the clip duration, or to the fallback cap when
            probing failed, and stays visible throughout. The loop is
            finite in both modes (loop=-1 stalls the filter graph).
            """
            if timing == "whole_clip":
                card_len = clip_len if clip_len else WHOLE_CLIP_FALLBACK_SECONDS
                loop_frames = max(BANNER_LOOP_FRAMES, int(card_len * 30) + 60)
                return (
                    f"movie='{escaped_png}',loop=loop={loop_frames}:size=1:start=0,"
                    f"trim=0:{card_len:.2f},setpts=PTS-STARTPTS,"
                    f"format=rgba[{card_label}];"
                    f"{base_label}[{card_label}]overlay=0:0[{out_label}]"
                )
            banner_end = BANNER_DURATION_SECONDS + BANNER_FADE_SECONDS
            card_end = banner_end if not clip_len else min(banner_end, clip_len)
            return (
                f"movie='{escaped_png}',loop=loop={BANNER_LOOP_FRAMES}:size=1:start=0,"
                f"trim=0:{card_end:.2f},setpts=PTS-STARTPTS,"
                f"format=rgba,fade=t=out:st={BANNER_DURATION_SECONDS:.2f}:d={BANNER_FADE_SECONDS:.2f}:alpha=1[{card_label}];"
                f"{base_label}[{card_label}]overlay=0:0:enable='between(t,0,{banner_end:.2f})'[{out_label}]"
            )

        # 5a. Face-aware placement (FEATURE 1): when face data was
        # captured for a face-tracking layout, place the banner (and
        # the quote line) where they do not cover any face in the
        # rendered clip. Without face data the overrides stay None
        # and both cards use their fixed positions, exactly as
        # before. Placement is baked into the card PNGs; banner
        # timing (4s vs whole_clip) is the overlay and is
        # independent of placement, so this works with both.
        banner_y = None
        quote_y = None
        banner_box = None
        if face_boxes is not None and face_traj is not None:
            try:
                face_region = face_region_for_clip(
                    face_boxes, face_traj, face_crop_w, face_raw_h)
                if face_region is not None:
                    if hook_banner and caption_position == "top":
                        _lines, box_w, box_h = self._measure_caption_box(hook_banner)
                        banner_y = choose_banner_y(face_region, box_h, layout, box_w)
                        banner_box = ((VIDEO_WIDTH - box_w) // 2, banner_y,
                                      (VIDEO_WIDTH + box_w) // 2, banner_y + box_h)
                    elif hook_banner and caption_position == "bottom":
                        _lines, box_w, box_h = self._measure_caption_box(hook_banner)
                        banner_box = (75, 1410 - box_h, 75 + box_w, 1410)
                    if quote_line and show_quote:
                        _qlines, quote_block_h = self._measure_quote_block(quote_line)
                        quote_y = choose_quote_y(face_region, banner_box, quote_block_h)
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Face-aware placement failed, using fixed positions: {e}")
                banner_y = quote_y = banner_box = None

        card_png = None
        quote_png = None
        if hook_banner and caption_position != "none":
            import time
            card_png = str(EDITED_DIR / f"temp_card_{streamer}_{int(time.time()*1000)}.png")
            self.generate_screen_caption_card(hook_banner, card_png, position=caption_position, layout=layout,
                                              banner_y=banner_y, theme=theme)
            card_label = "card_faded" if timing == "4s" else "card"
            filter_complex.append(_card_filter(self.escape_path(card_png),
                                               card_label, "with_card", current_label))
            current_label = "[with_card]"

        # 5b. Quote line (FEATURE 2): one static spoken line under the
        # headline, same timing as the banner. Never word-by-word
        # captions — subtitles stay off by default. Off by default
        # since v1.4.1: drawn only when show_quote is true (explicit
        # quote_line_enabled=True, or QUOTE_LINE_ENABLED=true).
        if quote_line and show_quote:
            import time
            quote_png = str(EDITED_DIR / f"temp_quote_{streamer}_{int(time.time()*1000)}.png")
            self.generate_quote_card(quote_line, quote_png, quote_y=quote_y)
            quote_label = "quote_faded" if timing == "4s" else "quote"
            filter_complex.append(_card_filter(self.escape_path(quote_png),
                                               quote_label, "with_quote", current_label))
            current_label = "[with_quote]"

        # 5c. Theme progress bar (feature 21): a thin bar along the
        # bottom of the frame that fills with the clip's progress.
        # Only when the theme enables it — the default "clean" theme
        # does not, so the default look is unchanged. Skipped when the
        # duration could not be probed (no progress to show).
        _theme_dict = get_theme(theme)
        _bar_on, _bar_color = theme_progress_bar(_theme_dict)
        if _bar_on and clip_len:
            try:
                from render_modes import build_progress_bar_filter
                filter_complex.append(
                    f"{current_label}{build_progress_bar_filter(clip_len, _bar_color)}[with_bar]"
                )
                current_label = "[with_bar]"
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Progress bar failed, skipping it: {e}")

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
            if trim_start is not None and trim_start > 0:
                cmd += ["-ss", f"{trim_start:.3f}"]
            if trim_end is not None and trim_end > (trim_start or 0):
                cmd += ["-to", f"{trim_end:.3f}"]
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
            if loudnorm_stats:
                # Two-pass loudnorm on the final audio: the ducked mix
                # when BGM is on, the source audio otherwise. The
                # measure only succeeds when the source has audio, so
                # [0:a] (no "?") is safe here.
                from loudness import loudnorm_filter
                audio_label = audio_map if use_bgm and bgm_path else "[0:a]"
                filters.append(f"{audio_label}{loudnorm_filter(loudnorm_stats)}[a_norm]")
                audio_map = "[a_norm]"
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
            # Render timeout scales with the source duration (see
            # render_timeout_seconds): the old fixed 180 s killed
            # 1080p60 renders of longer clips (clip #12, twice).
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                timeout=render_timeout_seconds(clip_len),
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

            # 6b. Voice-over (gta6-clipper feature 7): mix Sara's
            # recorded line in at voiceover_at seconds. The video is
            # copied untouched; a failed pass keeps the render as-is.
            if voiceover_path:
                try:
                    import time as _time
                    from voiceover import add_voiceover as _add_voiceover
                    # Temp file next to the output, not in EDITED_DIR:
                    # os.replace cannot move a file across filesystems,
                    # and output_path may live on another mount (/tmp).
                    _vo_tmp = str(Path(output_path).parent / f"temp_vo_{streamer}_{int(_time.time() * 1000)}.mp4")
                    _add_voiceover(output_path, voiceover_path, voiceover_at, _vo_tmp)
                    os.replace(_vo_tmp, output_path)
                    log_event("INFO", "ClipEditor", f"Voice-over mixed in at {voiceover_at}s")
                except Exception as e:
                    log_event("WARNING", "ClipEditor", f"Voice-over failed, keeping render without it: {e}")

            # 6c. Meme sounds (gta6-clipper feature 6): one synthesised
            # sound + visual gag at the payoff. OPT-IN ONLY — an
            # explicit meme_mood=, or MEME_SOUNDS_ENABLED=1 with
            # MEME_MOOD. At most one per clip, never in the first 3 s
            # (enforced in meme_sounds.apply_meme). A failed pass
            # keeps the render as-is.
            _meme = meme_mood
            if not _meme:
                try:
                    from meme_sounds import default_mood as _default_meme_mood
                    _meme = _default_meme_mood()
                except Exception:
                    _meme = None
            if _meme:
                try:
                    import time as _time2
                    from meme_sounds import apply_meme as _apply_meme
                    # Temp file next to the output, not in EDITED_DIR:
                    # os.replace cannot move a file across filesystems,
                    # and output_path may live on another mount (/tmp).
                    _out_dir = Path(output_path).parent
                    _meme_tmp = str(_out_dir / f"temp_meme_{streamer}_{int(_time2.time() * 1000)}.mp4")
                    _meme_res = _apply_meme(output_path, _meme, _meme_tmp, at=meme_at)
                    os.replace(_meme_tmp, output_path)
                    log_event("INFO", "ClipEditor",
                              f"Meme layer applied ({_meme} at {_meme_res['at']}s, "
                              f"{_meme_res['sound']})")
                except Exception as e:
                    log_event("WARNING", "ClipEditor", f"Meme layer failed, keeping render without it: {e}")

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

            # 8. Loudness safety pass (gta6-clipper feature 10): if the
            # finished clip lands more than 1.5 LU from -14 LUFS (BGM
            # mixing, ducking or a voice-over shift the level a
            # little), only the audio is re-normalised — the video is
            # stream-copied untouched. Runs last so it measures the
            # final audio. Never fails the render.
            try:
                from loudness import fix_loudness
                fix_loudness(output_path)
            except Exception as e:
                log_event("WARNING", "ClipEditor", f"Loudness safety pass failed (render unaffected): {e}")

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
            if quote_png and os.path.exists(quote_png):
                try:
                    os.remove(quote_png)
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
