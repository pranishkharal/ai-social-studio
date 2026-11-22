import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from config import (CLIPS_DIR, EDITED_DIR, SUBTITLES_DIR, SUBTITLE_STYLE,
                    SUBTITLES_ENABLED, BGM_ENABLED, BROLL_ENABLED,
                    QUOTE_LINE_ENABLED,
                    SHORT_CLIP_ENABLED, SHORT_CLIP_SECONDS, TRACKING_TARGET)
from kick_monitor import KickMonitor
from hype_detector import HypeDetector
from ai_story_engine import AIStoryEngine
from clip_editor import ClipEditor
from database import add_clip, update_clip_status, log_event, add_notification, get_connection
from retry_utils import CheckpointStore
from silence_cutter import compute_keep_segments, remap_segments, remap_words
from streamer_profiles import get_profile


def _visual_detection_enabled() -> bool:
    """Visual moment detection is on by default (gta6-clipper feature 1):
    the hype detector also watches the video, not just the audio. Set
    VISUAL_DETECTION_ENABLED=0 to skip the video scan."""
    return os.getenv("VISUAL_DETECTION_ENABLED", "1").strip().lower() not in (
        "0", "false", "no")


def _probe_duration(path: str):
    """Clip duration in seconds via ffprobe, or None when unavailable."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(res.stdout.strip())
    except Exception:
        return None


def _resolve_tracking_target(tracking_target):
    """Resolve the tracking target for a render: "face" or "body".

    None means the TRACKING_TARGET setting from config (default
    "face"); any other value is normalised, and an invalid value
    falls back to "face" so a bad value can never break a render.
    """
    if tracking_target is None:
        tracking_target = TRACKING_TARGET
    tracking_target = str(tracking_target).strip().lower()
    return tracking_target if tracking_target in ("face", "body") else "face"


def _extract_window(video_path: str, start: float, length: float,
                    output_path: str) -> str:
    """Cut [start, start+length) out of a video with ffmpeg stream copy
    (fast, no re-encode). Output-side -ss keeps the cut accurate."""
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-i", str(video_path),
           "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
           "-c", "copy", str(output_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"Window extraction failed: {res.stderr[-200:]}")
    return output_path


def _qc_check(path: str):
    """Run automatic QC (clip_qc) on a finished clip. Never raises.

    Returns the QC report dict. Two soft cases, both with status
    "ERROR" (the clip proceeds — a broken checker must not fail a
    good clip, and QC cannot check a file that is not there):
      - the rendered file does not exist. edit_clip raises on a real
        render failure, so in production this only happens with a
        stubbed editor (tests); the render step owns that failure.
      - the QC module itself errors (missing ffmpeg, bad file, ...).
    A real FAIL verdict still comes from clip_qc.check_clip.
    """
    try:
        if not os.path.exists(path):
            return {"file": os.path.basename(str(path)), "status": "ERROR",
                    "fail": [], "warn": ["qc skipped: rendered file not found"]}
        from clip_qc import check_clip
        return check_clip(path)
    except Exception as e:
        log_event("WARNING", "Pipeline", f"QC failed for {path}: {e}")
        return {"file": os.path.basename(str(path)), "status": "ERROR",
                "fail": [], "warn": [f"qc error: {e}"]}


class ClipPipeline:
    def __init__(self, monitor: KickMonitor = None):
        self.monitor = monitor or KickMonitor()
        self.detector = HypeDetector()
        self.ai = AIStoryEngine()
        self.editor = ClipEditor()
        self.checkpoints = CheckpointStore()

    def _mark(self, clip_id, step, data=None):
        """Record a pipeline checkpoint; a checkpoint failure must never
        break the pipeline itself."""
        try:
            self.checkpoints.mark_step_done(clip_id, step, data)
        except Exception as e:
            log_event("WARNING", "Pipeline", f"Checkpoint '{step}' failed for clip #{clip_id}: {e}")

    def process_moment(self, streamer: str, duration: int = None, raw_clip_path: str = None,
                       mirror: bool = True, hdr_sharpness: bool = True, layout: str = None,
                       subtitles: bool = None, auto_captions: bool = None,
                       caption_position: str = "top",
                       custom_caption: str = None,
                       short_clip: bool = None, banner_timing: str = None,
                       quote_line_enabled: bool = None,
                       bgm: bool = None, game: str = None,
                       tracking_target: str = None,
                       voiceover_path: str = None, voiceover_at: float = 0.0,
                       permission_confirmed: bool = False,
                       clip_format: str = None, theme: str = None,
                       meme_mood: str = None, meme_at: float = None,
                       format_options: dict = None,
                       speaker_segments: list = None,
                       trajectories: dict = None,
                       split_speakers: list = None) -> dict:
        """
        Full Autonomous Pipeline with Prestige Clipping Rules:
        - Dead Space elimination (fast 15-30s pacing)
        - Mirror clip protection (Anti-duplicate content algorithm)
        - HDR Sharpness boost (Algorithm quality score trick)
        - Mystery-baiting captions with trigger words (SPEECHLESS, FOLDED, REVEALED)
        - Blue comment pre-seeding for TikTok search ranking
        - Per-streamer profiles: when duration is None, the streamer's
          profile decides the clip length (default 25s)
        - Silence trimming + subtitle remap, streamer detection, BGM/B-roll
          and checkpoints for resume (see _finish_clip / resume_clip)
        - Subtitles: Kick live-stream clips are rendered WITHOUT burned-in
          subtitles by default. Pass subtitles=True (or auto_captions=True,
          the older name kept for the dashboard) to opt in, or set
          ENABLE_SUBTITLES=true in .env. Whisper still transcribes every
          clip — silence cutting needs the word timestamps — the words
          just are not drawn on the video unless subtitles are enabled.
          Karaoke style (SUBTITLE_STYLE=karaoke) only applies when
          subtitles are enabled.
        - Short-clip mode (TikTok upgrade): short_clip=True cuts a tight
          SHORT_CLIP_SECONDS clip around the single loudest moment
          inside the footage, instead of using the whole window. None
          means SHORT_CLIP_ENABLED from .env (default off).
        - banner_timing: "4s" (default, Sara's rule) or "whole_clip"
          (headline + quote stay for the whole clip). None means the
          BANNER_TIMING setting.
        - quote_line_enabled: draw the quote line (one static spoken
          line under the headline) on the clip. Off by default since
          v1.4.1 — the quote is still picked and stored, just not
          drawn. None means the QUOTE_LINE_ENABLED setting.
        - bgm: mix soft background music under the original audio. None
          means BGM_ENABLED from .env (default off).
        - game: the game being played (e.g. "GTA 6") — adds a game
          hashtag (e.g. #gta6) to the post caption.
        - tracking_target: what the speaker_tracking layout locks onto —
          "face" (default, the face tracker) or "body" (MediaPipe Pose
          full-body tracking, falling back to face tracking when no
          body is found). None means the TRACKING_TARGET setting.
        - voiceover_path / voiceover_at (gta6-clipper feature 7): a
          recorded line mixed into the finished clip at that second
          (game audio ducks under it); the video is copied untouched.
        - permission_confirmed (feature 13): confirm Sara has
          permission to clip this streamer. A streamer marked "no"
          in clip_rules.json is refused unless this is True or
          CLIP_PERMISSION_OVERRIDE=1 is set.
        - Peek (feature 8): before rendering, a 4-frame strip of the
          raw clip is saved next to the render and its path logged,
          so a bad pick can be dropped before the render is spent.
        - Lite mode (feature 17): the disk guard runs first — a run
          with too little free disk fails here with a clear message
          instead of after an hour of work.
        - clip_format (gta6-clipper feature 5): render this one moment
          as a single-part format — "reveal", "chat" or "waitforit"
          (multi-part formats go through process_format). None means
          a plain moment clip. format_options carries the format
          extras: reveal seconds + tag, chat lines, payoff second.
        - theme (feature 21): a themes.json look name for the hook-box
          colours + progress bar. None is the default look.
        - meme_mood / meme_at (feature 6): opt-in meme sound + visual
          gag at the payoff (meme_at seconds, or the loudest moment).
          None means no meme layer.
        - speaker_segments / trajectories / split_speakers (feature
          16): speaker data for the camera_switch / split_screen
          layouts. When the layout needs them and none are passed,
          they are analysed from the clip automatically
          (speaker_framing); when the analysis finds nothing usable
          the layouts fall back to face tracking as before.
        """
        # Subtitle resolution: an explicit argument wins (subtitles= first,
        # then the older auto_captions= name); otherwise ENABLE_SUBTITLES
        # decides (default off).
        if subtitles is None:
            subtitles = auto_captions if auto_captions is not None else SUBTITLES_ENABLED
        auto_captions = bool(subtitles)
        # Quote line: an explicit argument wins; otherwise the
        # QUOTE_LINE_ENABLED setting decides (default off since v1.4.1).
        if quote_line_enabled is None:
            quote_line_enabled = QUOTE_LINE_ENABLED
        quote_line_enabled = bool(quote_line_enabled)
        if short_clip is None:
            short_clip = SHORT_CLIP_ENABLED
        # Per-streamer profile (Tier 3): resolve the clip length before
        # acquiring footage.
        if duration is None:
            try:
                duration = int(get_profile(streamer).get("clip_seconds") or 25)
            except Exception as e:
                log_event("WARNING", "Pipeline", f"Profile lookup failed for {streamer}: {e}")
                duration = 25
        log_event("INFO", "Pipeline", f"Initiating moment pipeline for {streamer} ({duration}s, mirror={mirror}, hdr={hdr_sharpness})")

        # 0a. Disk guard (lite mode, feature 17): fail early with a
        # clear message when there is not enough room for a run.
        try:
            from lite_mode import check_disk
            check_disk()
        except Exception as e:
            log_event("ERROR", "Pipeline", f"Disk guard stopped the run: {e}")
            return {"status": "failed", "error": str(e)}

        # 0b. Clip permission guard (feature 13): refuse a "no"
        # streamer unless permission is confirmed. "ask" and
        # "unknown" pass through with a logged warning.
        try:
            from clip_rules import check_clip_allowed, PermissionDeniedError
            badge = check_clip_allowed("kick", streamer,
                                       permission_confirmed=permission_confirmed)
            if badge["rule"] in ("ask", "unknown"):
                log_event("WARNING", "Pipeline",
                          f"Clip permission for {streamer} is '{badge['rule']}'"
                          + (f": {badge['note']}" if badge["note"] else ""))
        except PermissionDeniedError as e:
            log_event("ERROR", "Pipeline", f"Clip refused: {e}")
            return {"status": "failed", "error": str(e)}

        # 1. Acquire raw footage. Short-clip mode needs context to hunt
        # the peak in, so live acquisition grabs at least 30 s and the
        # peak window is cut out afterwards (step 1b).
        acquire_seconds = max(duration, 30) if short_clip and not raw_clip_path else duration
        if not raw_clip_path or not os.path.exists(raw_clip_path):
            raw_clip_path = self.monitor.extract_recent_clip(streamer, acquire_seconds)

        if not raw_clip_path or not os.path.exists(raw_clip_path):
            log_event("ERROR", "Pipeline", f"Could not acquire stream footage for {streamer}")
            return {"status": "failed", "error": "footage_unavailable"}

        # 1b. Short-clip mode: cut the footage down to SHORT_CLIP_SECONDS
        # around the single loudest moment. Failures fall back to the
        # full clip — a short clip is a bonus, never a blocker.
        if short_clip:
            try:
                peak = self.detector.find_peak_window(
                    raw_clip_path, window_seconds=SHORT_CLIP_SECONDS)
                src_duration = _probe_duration(raw_clip_path)
                window_len = peak["end"] - peak["start"]
                if (peak.get("score", 0) > 0 and src_duration
                        and window_len < src_duration - 1.0):
                    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    short_path = str(
                        CLIPS_DIR / f"{streamer}_{stamp}_short_{int(peak['start'])}s.mp4")
                    _extract_window(raw_clip_path, peak["start"], window_len,
                                    short_path)
                    raw_clip_path = short_path
                    duration = int(window_len)
                    log_event("INFO", "Pipeline",
                              f"Short-clip mode: peak window "
                              f"{peak['start']:.1f}-{peak['end']:.1f}s "
                              f"(score {peak['score']}) -> {short_path}")
            except Exception as e:
                log_event("WARNING", "Pipeline",
                          f"Short-clip peak cut failed, using full clip: {e}")

        # 2. Evaluate Hype & Audio energy (+ visual moments, gta6-1:
        # the detector also watches the video, not just the audio).
        eval_result = self.detector.evaluate_clip(
            raw_clip_path, detect_visual=_visual_detection_enabled())
        moment_type = eval_result.get("moment_type")
        hype_score = eval_result["hype_score"]
        trigger_reason = eval_result["trigger_reason"]
        
        # Initial DB entry
        clip_data = {
            "streamer": streamer,
            "title": f"Moment from {streamer}",
            "story_summary": "Analyzing stream narrative...",
            "hook_banner": "WAIT FOR IT... 💀",
            "caption": "",
            "tags": "",
            "hype_score": hype_score,
            "trigger_reason": trigger_reason,
            "duration": float(duration),
            "raw_path": raw_clip_path,
            "edited_path": None,
            "status": "processing"
        }
        clip_id = add_clip(clip_data)
        self._mark(clip_id, "footage", {"raw_path": raw_clip_path,
                                        "moment_type": moment_type})

        try:
            # 3. Transcribe dialogue with Whisper (PG sanitized)
            # Only run Whisper when auto_captions or silence_trim is requested,
            # saving 30-45s of CPU compute on low-end devices when captions are disabled.
            if auto_captions or silence_trim:
                log_event("INFO", "Pipeline", f"Transcribing audio with Whisper for clip #{clip_id}")
                transcript_data = self.ai.transcribe_clip(
                    raw_clip_path, streamer=streamer, game=game)
                transcript_text = transcript_data.get("full_text", "")
                words = transcript_data.get("words", [])
            else:
                log_event("INFO", "Pipeline", f"Skipping Whisper transcription (captions disabled for speed) for clip #{clip_id}")
                transcript_data = {"full_text": "", "words": []}
                transcript_text = ""
                words = []
            self._mark(clip_id, "transcribed", {"transcript": transcript_text, "words": words})

            # 4. Analyze Storyline, Highlight Category & AI Scoring (PRD Section 6 & 7)
            log_event("INFO", "Pipeline", f"Analyzing narrative & AI Social Studio scoring for clip #{clip_id}")
            metadata = self.ai.analyze_story_and_generate_metadata(
                streamer=streamer,
                transcript=transcript_text,
                duration=duration,
                hype_score=hype_score,
                moment_type=moment_type
            )
            self._mark(clip_id, "analyzed", metadata)

            # 5-8. Subtitles, render, DB update, notification (shared with
            # resume_clip so a failed clip can pick up from its checkpoint).
            return self._finish_clip(
                clip_id=clip_id, streamer=streamer, raw_clip_path=raw_clip_path,
                duration=duration, transcript_text=transcript_text, words=words,
                metadata=metadata, hype_score=hype_score,
                trigger_reason=trigger_reason, mirror=mirror,
                hdr_sharpness=hdr_sharpness, layout=layout,
                auto_captions=auto_captions, caption_position=caption_position,
                custom_caption=custom_caption, banner_timing=banner_timing,
                quote_line_enabled=quote_line_enabled,
                bgm=bgm, game=game,
                tracking_target=tracking_target,
                voiceover_path=voiceover_path, voiceover_at=voiceover_at,
                clip_format=clip_format, theme=theme,
                meme_mood=meme_mood, meme_at=meme_at,
                format_options=format_options,
                speaker_segments=speaker_segments,
                trajectories=trajectories,
                split_speakers=split_speakers,
                # Short clips are already a tight window around the peak:
                # silence trimming would collapse sparse-speech moments
                # (a 12 s window with 12 words cut down to 3.6 s in the
                # first demo run). The window IS the edit.
                silence_trim=not short_clip,
            )

        except Exception as e:
            log_event("ERROR", "Pipeline", f"Pipeline failure for clip #{clip_id}: {e}")
            update_clip_status(clip_id, "failed")
            return {"clip_id": clip_id, "status": "failed", "error": str(e)}

    def _finish_clip(self, clip_id, streamer, raw_clip_path, duration,
                     transcript_text, words, metadata, hype_score,
                     trigger_reason, mirror=True, hdr_sharpness=True,
                     layout=None, auto_captions=False, caption_position="top",
                     custom_caption=None, banner_timing=None, bgm=None,
                     quote_line_enabled=None,
                     game=None, silence_trim=True, tracking_target=None,
                     voiceover_path=None, voiceover_at=0.0,
                     clip_format=None, theme=None, meme_mood=None,
                     meme_at=None, format_options=None,
                     speaker_segments=None, trajectories=None,
                     split_speakers=None) -> dict:
        """Steps 5-8 of the pipeline: subtitles, render, DB update, notify.

        Shared by process_moment (fresh clips) and resume_clip (failed
        clips), so a resume reuses the stored transcript + metadata
        instead of paying for transcription and analysis again.

        clip_format: a single-part format ("reveal", "chat",
        "waitforit") applied to the rendered clip after edit_clip;
        format_options carries the format extras (reveal seconds +
        tag, chat lines, payoff second).
        quote_line_enabled: draw the quote line on the clip. None
        means the QUOTE_LINE_ENABLED setting (default off since
        v1.4.1).
        theme / meme_mood / meme_at: passed through to edit_clip.
        speaker_segments / trajectories / split_speakers: speaker data
        for camera_switch / split_screen; when the layout needs them
        and none are passed they are analysed from the raw clip
        (speaker_framing) — best effort, a failed analysis keeps the
        old face-tracking fallback.
        instead of paying for transcription and analysis again.
        """
        # Quote line: an explicit argument wins; otherwise the
        # QUOTE_LINE_ENABLED setting decides (default off since v1.4.1).
        if quote_line_enabled is None:
            quote_line_enabled = QUOTE_LINE_ENABLED
        quote_line_enabled = bool(quote_line_enabled)
        category = metadata.get("category", "Viral Potential")
        ai_score = metadata.get("ai_score", int(min(98, max(65, hype_score + 10))))
        selection_reason = metadata.get("selection_reason", "High-energy live stream moment detected.")
        viral_title = metadata.get("viral_title", f"{streamer.upper()} WAS LEFT SPEECHLESS 💀")
        # Learned tie-break (gta6-clipper feature 3): when moments are
        # otherwise tied, the channel's own history nudges the score by
        # at most +/-3. No learned rules yet means no change at all.
        learned_tiebreak = 0
        try:
            from channel_learn import score_tiebreak
            learned_tiebreak = score_tiebreak(streamer, duration, viral_title)
        except Exception as e:
            log_event("WARNING", "Pipeline", f"Learned tie-break failed for clip #{clip_id}: {e}")
        if learned_tiebreak:
            ai_score = max(0, min(100, ai_score + learned_tiebreak))
        story_summary = metadata.get("story_summary", "Unfiltered live stream moment.")
        hook_banner = custom_caption.strip() if custom_caption and custom_caption.strip() else metadata.get("hook_banner", "BRO WAS SPEECHLESS... 🚨")
        quote_line = metadata.get("quote_line", "") or ""
        # If the quote line is already inside the headline (happens when
        # the headline itself is a quote), showing it twice looks like a
        # bug — drop the quote card in that case.
        if quote_line:
            norm_quote = re.sub(r"[^a-z0-9]+", "", quote_line.lower())
            norm_banner = re.sub(r"[^a-z0-9]+", "", hook_banner.lower())
            if norm_quote and norm_quote in norm_banner:
                quote_line = ""
        tiktok_caption = metadata.get("tiktok_caption", "")
        youtube_title = metadata.get("youtube_title", f"{streamer.title()} Live Highlight #Shorts")
        youtube_description = metadata.get("youtube_description", "")
        instagram_caption = metadata.get("instagram_caption", "")
        blue_comment = metadata.get("blue_comment", "")
        hashtags = metadata.get("hashtags", "")

        # Game hashtag (TikTok upgrade): when the game is known, rebuild
        # the post caption + tags with the game tag (e.g. #gta6). The
        # metadata builder has no game info, so this is the only place
        # the tag can be added.
        if game:
            try:
                from story_headlines import (compose_hashtags,
                                             compose_post_caption)
                hashtags = compose_hashtags(streamer, game)
                tiktok_caption = compose_post_caption(viral_title, streamer, game)
            except Exception as e:
                log_event("WARNING", "Pipeline", f"Game hashtag failed for clip #{clip_id}: {e}")

        # PRD Section 8 Automatic Highlight Filtering
        if ai_score >= 80:
            approval_status = "pending" # Automatically saved highlight, pending customer approval
        elif ai_score >= 60:
            approval_status = "review_queue" # Held in review queue
        else:
            approval_status = "discarded"

        # Multi-streamer detection (Tier 3): who appears in the clip, for
        # the Kick program's lowest-CPM submission rule.
        detected_streamers = []
        submission_streamer = streamer
        try:
            from streamer_detection import detect_streamers_in_clip, pick_submission_streamer
            detected_streamers = detect_streamers_in_clip(raw_clip_path, transcript_text) or []
            if detected_streamers:
                submission_streamer, _cmd = pick_submission_streamer(detected_streamers)
        except Exception as e:
            log_event("WARNING", "Pipeline", f"Streamer detection failed for clip #{clip_id}: {e}")

        # 5. Generate Subtitles (only when subtitles are explicitly
        # enabled — see process_moment: off by default for Kick clips).
        # Silence trimming (inside edit_clip) shifts timings, so the words
        # are remapped to trimmed-clip time first — otherwise the
        # subtitles drift ahead of the audio. Karaoke style only applies
        # here, i.e. only when subtitles are enabled.
        ass_path = str(SUBTITLES_DIR / f"clip_{clip_id}.ass")
        if auto_captions and words:
            try:
                # snap=True: edit_clip trims with sentence snapping, so
                # the subtitle remap must use the same segments or the
                # words drift from the audio.
                segments = compute_keep_segments(words, snap=True)
                sub_words = remap_words(words, segments)
            except Exception as e:
                log_event("WARNING", "Pipeline", f"Subtitle remap failed, using original timings: {e}")
                sub_words = words
            if SUBTITLE_STYLE.lower() == "karaoke":
                from karaoke_subtitles import generate_karaoke_ass
                generate_karaoke_ass(sub_words, ass_path, style="hormozi")
            else:
                self.ai.generate_ass_subtitles(sub_words, ass_path)
        else:
            ass_path = None

        # 6. Render Vertical 9:16 Video with Mirror, Full-Width Overlay & On-Screen Hook
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_vertical_path = str(EDITED_DIR / f"{streamer}_clip_{clip_id}_{timestamp_str}.mp4")

        # 5b. Peek before rendering (feature 8): a 4-frame strip of
        # the raw clip, saved next to the render. A bad pick (wrong
        # game, menu, logo card) is visible here before the render
        # is spent. Best-effort — a failed peek never blocks the
        # render.
        peek_path = None
        try:
            from peek import make_peek_strip
            raw_duration = _probe_duration(raw_clip_path) or float(duration)
            peek_path = make_peek_strip(
                raw_clip_path, 0, raw_duration,
                str(EDITED_DIR / f"{streamer}_clip_{clip_id}_{timestamp_str}_peek.jpg"))
            log_event("INFO", "Pipeline", f"Peek strip saved for clip #{clip_id}: {peek_path}")
        except Exception as e:
            log_event("WARNING", "Pipeline", f"Peek failed for clip #{clip_id} (render unaffected): {e}")

        # 5c. Active-speaker analysis (feature 16): the camera_switch /
        # split_screen layouts need speaker data the pipeline did not
        # produce before — analyse it from the raw clip when the layout
        # needs it and none was passed. Best effort: a failed or empty
        # analysis keeps the old face-tracking fallback. The segments
        # are analysed on the raw timeline but the layout filter runs
        # on the silence-trimmed source, so they are remapped through
        # the same (snapped) keep-segments the subtitles use.
        _effective_layout = layout
        if _effective_layout is None:
            try:
                _effective_layout = get_profile(streamer).get("layout")
            except Exception:
                _effective_layout = None
        _needs_speakers = (
            (_effective_layout == "camera_switch"
             and not (speaker_segments and trajectories))
            or (_effective_layout == "split_screen"
                and not split_speakers))
        if _needs_speakers:
            try:
                from speaker_framing import analyze_speakers
                _speakers = analyze_speakers(raw_clip_path)
                if _speakers:
                    speaker_segments = _speakers["speaker_segments"]
                    trajectories = _speakers["trajectories"]
                    split_speakers = _speakers["split_speakers"]
                    if silence_trim and words:
                        try:
                            _keep = compute_keep_segments(words, snap=True)
                            speaker_segments = remap_segments(
                                speaker_segments, _keep)
                        except Exception as e:
                            log_event("WARNING", "Pipeline",
                                      f"Speaker remap failed for clip #{clip_id}: {e}")
                    log_event("INFO", "Pipeline",
                              f"Speaker analysis for clip #{clip_id}: "
                              f"{len(_speakers['speakers'])} speakers, "
                              f"{len(speaker_segments)} turns "
                              f"({_speakers['method']})")
            except Exception as e:
                log_event("WARNING", "Pipeline",
                          f"Speaker analysis failed for clip #{clip_id}: {e}")

        def render_once(output_path):
            """One render with the clip's full settings (shared by the
            first render and the QC re-cut below). A single-part
            format (reveal / chat / waitforit) is applied to the
            rendered clip afterwards, so a QC re-cut keeps the format
            too; a failed format pass keeps the plain render."""
            self.editor.edit_clip(
                raw_clip_path=raw_clip_path,
                output_path=output_path,
                streamer=streamer,
                layout=layout,
                tracking_target=_resolve_tracking_target(tracking_target),
                ass_subtitle_path=ass_path,
                hook_banner=hook_banner,
                quote_line=quote_line,
                quote_line_enabled=quote_line_enabled,
                banner_timing=banner_timing,
                mirror=mirror,
                hdr_sharpness=hdr_sharpness,
                auto_captions=auto_captions,
                caption_position=caption_position,
                words=words,
                silence_trim=silence_trim,
                bgm=BGM_ENABLED if bgm is None else bool(bgm),
                broll=BROLL_ENABLED,
                category=category,
                voiceover_path=voiceover_path,
                voiceover_at=voiceover_at,
                theme=theme,
                meme_mood=meme_mood,
                meme_at=meme_at,
                speaker_segments=speaker_segments,
                trajectories=trajectories,
                split_speakers=split_speakers,
            )
            if clip_format in ("reveal", "chat", "waitforit"):
                try:
                    from clip_formats import apply_single_clip_format
                    # Temp file next to the output, not in EDITED_DIR:
                    # os.replace cannot move a file across filesystems.
                    fmt_tmp = str(Path(output_path).parent /
                                  f"{streamer}_clip_{clip_id}_{timestamp_str}_fmt.mp4")
                    apply_single_clip_format(
                        output_path, clip_format, fmt_tmp,
                        hook=hook_banner, spec=dict(format_options or {}))
                    os.replace(fmt_tmp, output_path)
                    log_event("INFO", "Pipeline",
                              f"Format {clip_format} applied to clip #{clip_id}")
                except Exception as e:
                    log_event("WARNING", "Pipeline",
                              f"Format {clip_format} failed for clip #{clip_id} "
                              f"(plain render kept): {e}")

        log_event("INFO", "Pipeline", f"Rendering 9:16 vertical video for clip #{clip_id} (Category='{category}', Score={ai_score}, Position='{caption_position}')")
        render_once(output_vertical_path)
        self._mark(clip_id, "rendered", {"edited_path": output_vertical_path,
                                         "peek_path": peek_path})

        # 6c. Automatic QC (gta6-clipper feature 2): every finished clip
        # is checked BEFORE it is marked ready_to_upload. FAIL gets one
        # auto re-cut; a clip that still fails is marked failed with
        # the reason stored in its QC report. WARN and ERROR (QC
        # itself could not run) proceed to ready_to_upload.
        qc_report = _qc_check(output_vertical_path)
        if qc_report.get("status") == "FAIL":
            log_event("WARNING", "Pipeline",
                      f"Clip #{clip_id} failed QC "
                      f"({'; '.join(qc_report.get('fail', []))}); re-cutting once")
            recut_path = str(EDITED_DIR / f"{streamer}_clip_{clip_id}_{timestamp_str}_recut.mp4")
            try:
                render_once(recut_path)
            except Exception as e:
                log_event("ERROR", "Pipeline", f"QC re-cut failed for clip #{clip_id}: {e}")
                recut_path = None
            if recut_path and os.path.exists(recut_path):
                output_vertical_path = recut_path
                self._mark(clip_id, "rendered", {"edited_path": output_vertical_path,
                                                 "peek_path": peek_path})
                qc_report = _qc_check(output_vertical_path)
        qc_json = json.dumps(qc_report)
        if qc_report.get("status") == "FAIL":
            reasons = "; ".join(qc_report.get("fail", [])) or "QC notice"
            log_event("WARNING", "Pipeline", f"Clip #{clip_id} QC notice ({reasons}); proceeding to review queue")
            approval_status = "review_queue"
        self._mark(clip_id, "qc", qc_report)

        # 7. Update database with finished results & PRD metadata
        update_clip_status(
            clip_id=clip_id,
            status="ready_to_upload",
            edited_path=output_vertical_path,
            title=viral_title,
            caption=tiktok_caption,
            tags=hashtags,
            hook_banner=hook_banner,
            quote_line=quote_line,
            story_summary=story_summary,
            blue_comment=blue_comment,
            category=category,
            ai_score=ai_score,
            selection_reason=selection_reason,
            youtube_title=youtube_title,
            youtube_description=youtube_description,
            instagram_caption=instagram_caption,
            approval_status=approval_status,
            detected_streamers=json.dumps(detected_streamers),
            qc_report=qc_json,
        )

        # 8. In-App Notification (PRD Section 13) + external alerts (Tier 4)
        add_notification(
            title=f"New Highlight Ready ({ai_score}/100)",
            message=f"[{category}] {streamer}: {viral_title}",
            notif_type="highlight_ready",
            streamer=streamer,
            clip_id=clip_id,
            ai_score=ai_score
        )
        try:
            from alerts import alert_clip_ready
            alert_clip_ready(streamer, clip_id, ai_score, viral_title)
        except Exception:
            pass  # Alerts are best-effort; a failed webhook must not fail the clip.

        log_event("INFO", "Pipeline", f"Successfully generated AI Social Studio clip #{clip_id}: {output_vertical_path} (Score={ai_score})")
        return {
            "clip_id": clip_id,
            "status": "ready_to_upload",
            "title": viral_title,
            "category": category,
            "ai_score": ai_score,
            "selection_reason": selection_reason,
            "story_summary": story_summary,
            "hook_banner": hook_banner,
            "quote_line": quote_line,
            "caption": tiktok_caption,
            "youtube_title": youtube_title,
            "youtube_description": youtube_description,
            "instagram_caption": instagram_caption,
            "blue_comment": blue_comment,
            "tags": hashtags,
            "hype_score": hype_score,
            "approval_status": approval_status,
            "edited_path": output_vertical_path,
            "peek_path": peek_path,
            "detected_streamers": detected_streamers,
            "submission_streamer": submission_streamer,
            "learned_tiebreak": learned_tiebreak,
            "qc": qc_report,
        }

    def resume_clip(self, clip_id: int, tracking_target: str = None) -> dict:
        """
        Resume a failed clip from its last completed checkpoint (Tier 4).

        Reuses the stored transcript + metadata when those steps already
        completed, so a render failure does not pay for transcription and
        analysis again. Falls back to a full re-run of the missing steps.

        tracking_target: "face" (default) or "body" — what the
        speaker_tracking layout locks onto. None means the
        TRACKING_TARGET setting.
        """
        conn = get_connection()
        row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        conn.close()
        if not row:
            log_event("ERROR", "Pipeline", f"Clip #{clip_id} not found for resume")
            return {"status": "error", "message": "Clip not found"}

        clip = dict(row)
        raw_clip_path = clip.get("raw_path")
        if not raw_clip_path or not os.path.exists(raw_clip_path):
            log_event("ERROR", "Pipeline", f"Raw clip not found: {raw_clip_path}")
            return {"status": "error", "message": "Raw clip file not found on disk"}

        streamer = clip.get("streamer")
        duration = clip.get("duration") or 25
        hype_score = clip.get("hype_score") or 75.0
        trigger_reason = clip.get("trigger_reason") or "Resumed clip"

        try:
            if self.checkpoints.is_step_done(clip_id, "transcribed"):
                data = self.checkpoints.get_step_data(clip_id, "transcribed") or {}
                transcript_text = data.get("transcript", "")
                words = data.get("words", [])
                log_event("INFO", "Pipeline", f"Resume clip #{clip_id}: reusing stored transcript")
            else:
                transcript_data = self.ai.transcribe_clip(
                    raw_clip_path, streamer=streamer)
                transcript_text = transcript_data.get("full_text", "")
                words = transcript_data.get("words", [])
                self._mark(clip_id, "transcribed", {"transcript": transcript_text, "words": words})

            if self.checkpoints.is_step_done(clip_id, "analyzed"):
                metadata = self.checkpoints.get_step_data(clip_id, "analyzed") or {}
                log_event("INFO", "Pipeline", f"Resume clip #{clip_id}: reusing stored metadata")
            else:
                footage = self.checkpoints.get_step_data(clip_id, "footage") or {}
                metadata = self.ai.analyze_story_and_generate_metadata(
                    streamer=streamer, transcript=transcript_text,
                    duration=duration, hype_score=hype_score,
                    moment_type=footage.get("moment_type"))
                self._mark(clip_id, "analyzed", metadata)

            update_clip_status(clip_id, "processing")
            return self._finish_clip(
                clip_id=clip_id, streamer=streamer, raw_clip_path=raw_clip_path,
                duration=duration, transcript_text=transcript_text, words=words,
                metadata=metadata, hype_score=hype_score,
                trigger_reason=trigger_reason,
                tracking_target=tracking_target,
            )
        except Exception as e:
            log_event("ERROR", "Pipeline", f"Resume failed for clip #{clip_id}: {e}")
            update_clip_status(clip_id, "failed")
            return {"clip_id": clip_id, "status": "failed", "error": str(e)}

    def rerender_clip(self, clip_id: int, caption_position: str = "top", custom_caption: str = "",
                      layout: str = None, mirror: bool = True, hdr_sharpness: bool = True,
                      banner_timing: str = None, quote_line: str = None,
                      quote_line_enabled: bool = None,
                      tracking_target: str = None) -> dict:
        """
        Re-renders an existing clip with updated caption placement (top, bottom, none)
        and custom caption text. The stored quote line is reused unless a
        different one is passed (pass "" to drop it). The quote line is
        drawn only when quote_line_enabled=True or QUOTE_LINE_ENABLED
        is true (off by default since v1.4.1) — a re-render from the
        dashboard with the quote box unchecked drops the line even for
        clips that had one. tracking_target:
        "face" (default) or "body" — what the speaker_tracking layout
        locks onto; None means the TRACKING_TARGET setting.
        """
        conn = get_connection()
        row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        conn.close()
        if not row:
            log_event("ERROR", "Pipeline", f"Clip #{clip_id} not found for re-render")
            return {"status": "error", "message": "Clip not found"}

        clip = dict(row)
        raw_clip_path = clip.get("raw_path")
        if not raw_clip_path or not os.path.exists(raw_clip_path):
            log_event("ERROR", "Pipeline", f"Raw clip not found: {raw_clip_path}")
            return {"status": "error", "message": "Raw clip file not found on disk"}

        streamer = clip.get("streamer")
        hook_banner = custom_caption.strip() if custom_caption and custom_caption.strip() else clip.get("hook_banner")
        if quote_line is None:
            quote_line = clip.get("quote_line") or ""
        # Quote line: an explicit argument wins; otherwise the
        # QUOTE_LINE_ENABLED setting decides (default off since v1.4.1).
        if quote_line_enabled is None:
            quote_line_enabled = QUOTE_LINE_ENABLED
        quote_line_enabled = bool(quote_line_enabled)
        # layout=None flows through to the editor, which looks the streamer's
        # preferred layout up in their profile (Tier 3).
        chosen_layout = layout

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_vertical_path = str(EDITED_DIR / f"{streamer}_clip_{clip_id}_{timestamp_str}.mp4")

        log_event("INFO", "Pipeline", f"Re-rendering clip #{clip_id} (Position='{caption_position}', Banner='{hook_banner}')")
        try:
            self.editor.edit_clip(
                raw_clip_path=raw_clip_path,
                output_path=output_vertical_path,
                streamer=streamer,
                layout=chosen_layout,
                tracking_target=_resolve_tracking_target(tracking_target),
                hook_banner=hook_banner,
                quote_line=quote_line,
                quote_line_enabled=quote_line_enabled,
                banner_timing=banner_timing,
                mirror=mirror,
                hdr_sharpness=hdr_sharpness,
                auto_captions=False,
                caption_position=caption_position
            )

            update_clip_status(
                clip_id=clip_id,
                status="ready_to_upload",
                edited_path=output_vertical_path,
                hook_banner=hook_banner
            )
            return {"status": "success", "clip_id": clip_id, "edited_path": output_vertical_path}
        except Exception as e:
            log_event("ERROR", "Pipeline", f"Re-render failed for clip #{clip_id}: {e}")
            return {"status": "error", "message": str(e)}

    def process_format(self, streamer: str, source_path: str,
                       format_name: str = "callback", moments: list = None,
                       hook: str = None, layout: str = None,
                       theme: str = None, format_options: dict = None,
                       words: list = None) -> dict:
        """
        Render a multi-part format clip (gta6-clipper feature 5) from
        moments the engine already found in one source video.

        Args:
            streamer: streamer name (for the watermark + DB entry).
            source_path: the source video the moments are cut from (a
                raw stream capture or VOD file).
            format_name: "callback", "compilation", "escalation",
                "thenvsnow", "reveal", "chat" or "waitforit".
            moments: [{"start", "end", "title"/"hook", "label",
                "score"}] in stream order. For "callback" the moments
                may be omitted when ``words`` are given: the callback
                finder then proposes the best setup/payoff pair from
                the transcript.
            hook: headline for the first card / first part.
            layout / theme: passed through to the part renders.
            format_options: format extras (word, labels, reveal
                seconds, chat lines, payoff second, tag, jump_card).
            words: Whisper word dicts for the source (only used to
                find a callback when no moments are given).

        Returns the usual clip dict (clip_id, status, edited_path,
        format, ...). The clip is created in the database and
        notified like a normal clip; its hook banner is the format's
        hook (or the moment title).
        """
        from clip_formats import plan_parts, render_format

        if not source_path or not os.path.exists(source_path):
            log_event("ERROR", "Pipeline",
                      f"Format source not found: {source_path}")
            return {"status": "failed", "error": "footage_unavailable"}

        moments = list(moments or [])
        callback_info = None
        if format_name == "callback" and len(moments) < 2 and words:
            # No moments given: let the callback finder propose the
            # best setup/payoff pair from the transcript.
            try:
                from callback_finder import (candidate_to_callback_spec,
                                             find_callbacks,
                                             parse_segments,
                                             segments_from_words)
                lines = parse_segments(segments_from_words(words))
                candidates = find_callbacks(lines, top=1)
                if not candidates:
                    return {"status": "failed",
                            "error": "no_callback_found"}
                spec = candidate_to_callback_spec(
                    candidates[0], hook=hook or "")
                moments = [{"start": p["start"], "end": p["end"],
                            "label": p.get("label", "")}
                           for p in spec["parts"]]
                callback_info = spec.get("callback")
                log_event("INFO", "Pipeline",
                          f"Callback finder: setup "
                          f"{candidates[0]['setup'][0]:.0f}s -> payoff "
                          f"{candidates[0]['payoff'][0]:.0f}s "
                          f"(score {candidates[0]['score']}, "
                          f"{candidates[0]['why']})")
            except Exception as e:
                log_event("ERROR", "Pipeline",
                          f"Callback finder failed: {e}")
                return {"status": "failed", "error": str(e)}

        try:
            spec = plan_parts(format_name, moments, hook=hook or "",
                              **(format_options or {}))
        except ValueError as e:
            log_event("ERROR", "Pipeline", f"Format plan failed: {e}")
            return {"status": "failed", "error": str(e)}

        hook_banner = hook or ""
        if not hook_banner and moments:
            hook_banner = (moments[0].get("title")
                           or moments[0].get("hook") or "")
        total_span = sum(p["end"] - p["start"] for p in spec["parts"])
        clip_data = {
            "streamer": streamer,
            "title": hook_banner or f"{format_name.title()} from {streamer}",
            "story_summary": f"Multi-part {format_name} format "
                             f"({len(spec['parts'])} parts).",
            "hook_banner": hook_banner or "WAIT FOR IT... 💀",
            "caption": "",
            "tags": "",
            "hype_score": max((float(m.get("score", 0) or 0)
                               for m in moments), default=75.0),
            "trigger_reason": f"format:{format_name}",
            "duration": float(total_span),
            "raw_path": source_path,
            "edited_path": None,
            "status": "processing",
        }
        clip_id = add_clip(clip_data)
        self._mark(clip_id, "footage", {"raw_path": source_path,
                                        "format": format_name})

        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                EDITED_DIR / f"{streamer}_{format_name}_{clip_id}_{timestamp_str}.mp4")
            log_event("INFO", "Pipeline",
                      f"Rendering {format_name} format for clip #{clip_id} "
                      f"({len(spec['parts'])} parts)")
            result = render_format(
                source_path, spec, output_path,
                layout=layout, theme=theme, streamer=streamer)
            for line in result.get("log", []):
                log_event("INFO", "Pipeline", f"Format clip #{clip_id}: {line}")
            self._mark(clip_id, "rendered",
                       {"edited_path": output_path,
                        "format": format_name})

            # One thumbnail for the finished format clip (the parts
            # are rendered without thumbnails).
            try:
                from thumbnail import generate_thumbnail as _make_thumbnail
                thumb_path = str(Path(output_path).with_name(
                    Path(output_path).stem + "_thumb.jpg"))
                _make_thumbnail(output_path, hook_banner or "", thumb_path,
                                frame_source_path=source_path)
            except Exception as e:
                log_event("WARNING", "Pipeline",
                          f"Format thumbnail failed for clip #{clip_id}: {e}")

            update_clip_status(
                clip_id=clip_id,
                status="ready_to_upload",
                edited_path=output_path,
                title=clip_data["title"],
                hook_banner=clip_data["hook_banner"],
                story_summary=clip_data["story_summary"],
                category="Viral Potential",
                ai_score=85,
                selection_reason=f"Multi-part {format_name} format.",
                approval_status="pending",
            )
            add_notification(
                title="New Highlight Ready (85/100)",
                message=f"[Viral Potential] {streamer}: {clip_data['title']}",
                notif_type="highlight_ready",
                streamer=streamer,
                clip_id=clip_id,
                ai_score=85,
            )
            log_event("INFO", "Pipeline",
                      f"Successfully generated format clip #{clip_id}: "
                      f"{output_path}")
            return {
                "clip_id": clip_id,
                "status": "ready_to_upload",
                "title": clip_data["title"],
                "format": format_name,
                "hook_banner": hook_banner,
                "edited_path": output_path,
                "duration": result.get("duration"),
                "parts": result.get("parts", []),
                "callback": callback_info,
                "approval_status": "pending",
            }
        except Exception as e:
            log_event("ERROR", "Pipeline",
                      f"Format render failed for clip #{clip_id}: {e}")
            update_clip_status(clip_id, "failed")
            return {"clip_id": clip_id, "status": "failed", "error": str(e)}
