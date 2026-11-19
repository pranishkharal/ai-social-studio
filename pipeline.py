import json
import os
import time
from datetime import datetime
from pathlib import Path

from config import (EDITED_DIR, SUBTITLES_DIR, SUBTITLE_STYLE,
                    SUBTITLES_ENABLED, BGM_ENABLED, BROLL_ENABLED)
from kick_monitor import KickMonitor
from hype_detector import HypeDetector
from ai_story_engine import AIStoryEngine
from clip_editor import ClipEditor
from database import add_clip, update_clip_status, log_event, add_notification, get_connection
from retry_utils import CheckpointStore
from silence_cutter import compute_keep_segments, remap_words
from streamer_profiles import get_profile

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
                       custom_caption: str = None) -> dict:
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
        """
        # Subtitle resolution: an explicit argument wins (subtitles= first,
        # then the older auto_captions= name); otherwise ENABLE_SUBTITLES
        # decides (default off).
        if subtitles is None:
            subtitles = auto_captions if auto_captions is not None else SUBTITLES_ENABLED
        auto_captions = bool(subtitles)
        # Per-streamer profile (Tier 3): resolve the clip length before
        # acquiring footage.
        if duration is None:
            try:
                duration = int(get_profile(streamer).get("clip_seconds") or 25)
            except Exception as e:
                log_event("WARNING", "Pipeline", f"Profile lookup failed for {streamer}: {e}")
                duration = 25
        log_event("INFO", "Pipeline", f"Initiating moment pipeline for {streamer} ({duration}s, mirror={mirror}, hdr={hdr_sharpness})")
        
        # 1. Acquire raw footage
        if not raw_clip_path or not os.path.exists(raw_clip_path):
            raw_clip_path = self.monitor.extract_recent_clip(streamer, duration)
            
        if not raw_clip_path or not os.path.exists(raw_clip_path):
            log_event("ERROR", "Pipeline", f"Could not acquire stream footage for {streamer}")
            return {"status": "failed", "error": "footage_unavailable"}

        # 2. Evaluate Hype & Audio energy
        eval_result = self.detector.evaluate_clip(raw_clip_path)
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
        self._mark(clip_id, "footage", {"raw_path": raw_clip_path})

        try:
            # 3. Transcribe dialogue with Whisper (PG sanitized)
            log_event("INFO", "Pipeline", f"Transcribing audio for clip #{clip_id}")
            transcript_data = self.ai.transcribe_clip(raw_clip_path)
            transcript_text = transcript_data.get("full_text", "")
            words = transcript_data.get("words", [])
            self._mark(clip_id, "transcribed", {"transcript": transcript_text, "words": words})

            # 4. Analyze Storyline, Highlight Category & AI Scoring (PRD Section 6 & 7)
            log_event("INFO", "Pipeline", f"Analyzing narrative & AI Social Studio scoring for clip #{clip_id}")
            metadata = self.ai.analyze_story_and_generate_metadata(
                streamer=streamer,
                transcript=transcript_text,
                duration=duration,
                hype_score=hype_score
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
                custom_caption=custom_caption,
            )

        except Exception as e:
            log_event("ERROR", "Pipeline", f"Pipeline failure for clip #{clip_id}: {e}")
            update_clip_status(clip_id, "failed")
            return {"clip_id": clip_id, "status": "failed", "error": str(e)}

    def _finish_clip(self, clip_id, streamer, raw_clip_path, duration,
                     transcript_text, words, metadata, hype_score,
                     trigger_reason, mirror=True, hdr_sharpness=True,
                     layout=None, auto_captions=False, caption_position="top",
                     custom_caption=None) -> dict:
        """Steps 5-8 of the pipeline: subtitles, render, DB update, notify.

        Shared by process_moment (fresh clips) and resume_clip (failed
        clips), so a resume reuses the stored transcript + metadata
        instead of paying for transcription and analysis again.
        """
        category = metadata.get("category", "Viral Potential")
        ai_score = metadata.get("ai_score", int(min(98, max(65, hype_score + 10))))
        selection_reason = metadata.get("selection_reason", "High-energy live stream moment detected.")
        viral_title = metadata.get("viral_title", f"{streamer.upper()} WAS LEFT SPEECHLESS 💀")
        story_summary = metadata.get("story_summary", "Unfiltered live stream moment.")
        hook_banner = custom_caption.strip() if custom_caption and custom_caption.strip() else metadata.get("hook_banner", "BRO WAS SPEECHLESS... 🚨")
        tiktok_caption = metadata.get("tiktok_caption", "")
        youtube_title = metadata.get("youtube_title", f"{streamer.title()} Live Highlight #Shorts")
        youtube_description = metadata.get("youtube_description", "")
        instagram_caption = metadata.get("instagram_caption", "")
        blue_comment = metadata.get("blue_comment", "")
        hashtags = metadata.get("hashtags", "")

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
                segments = compute_keep_segments(words)
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

        log_event("INFO", "Pipeline", f"Rendering 9:16 vertical video for clip #{clip_id} (Category='{category}', Score={ai_score}, Position='{caption_position}')")
        self.editor.edit_clip(
            raw_clip_path=raw_clip_path,
            output_path=output_vertical_path,
            streamer=streamer,
            layout=layout,
            ass_subtitle_path=ass_path,
            hook_banner=hook_banner,
            mirror=mirror,
            hdr_sharpness=hdr_sharpness,
            auto_captions=auto_captions,
            caption_position=caption_position,
            words=words,
            bgm=BGM_ENABLED,
            broll=BROLL_ENABLED,
            category=category,
        )
        self._mark(clip_id, "rendered", {"edited_path": output_vertical_path})

        # 7. Update database with finished results & PRD metadata
        update_clip_status(
            clip_id=clip_id,
            status="ready_to_upload",
            edited_path=output_vertical_path,
            title=viral_title,
            caption=tiktok_caption,
            tags=hashtags,
            hook_banner=hook_banner,
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
            "caption": tiktok_caption,
            "youtube_title": youtube_title,
            "youtube_description": youtube_description,
            "instagram_caption": instagram_caption,
            "blue_comment": blue_comment,
            "tags": hashtags,
            "hype_score": hype_score,
            "approval_status": approval_status,
            "edited_path": output_vertical_path,
            "detected_streamers": detected_streamers,
            "submission_streamer": submission_streamer,
        }

    def resume_clip(self, clip_id: int) -> dict:
        """
        Resume a failed clip from its last completed checkpoint (Tier 4).

        Reuses the stored transcript + metadata when those steps already
        completed, so a render failure does not pay for transcription and
        analysis again. Falls back to a full re-run of the missing steps.
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
                transcript_data = self.ai.transcribe_clip(raw_clip_path)
                transcript_text = transcript_data.get("full_text", "")
                words = transcript_data.get("words", [])
                self._mark(clip_id, "transcribed", {"transcript": transcript_text, "words": words})

            if self.checkpoints.is_step_done(clip_id, "analyzed"):
                metadata = self.checkpoints.get_step_data(clip_id, "analyzed") or {}
                log_event("INFO", "Pipeline", f"Resume clip #{clip_id}: reusing stored metadata")
            else:
                metadata = self.ai.analyze_story_and_generate_metadata(
                    streamer=streamer, transcript=transcript_text,
                    duration=duration, hype_score=hype_score)
                self._mark(clip_id, "analyzed", metadata)

            update_clip_status(clip_id, "processing")
            return self._finish_clip(
                clip_id=clip_id, streamer=streamer, raw_clip_path=raw_clip_path,
                duration=duration, transcript_text=transcript_text, words=words,
                metadata=metadata, hype_score=hype_score,
                trigger_reason=trigger_reason,
            )
        except Exception as e:
            log_event("ERROR", "Pipeline", f"Resume failed for clip #{clip_id}: {e}")
            update_clip_status(clip_id, "failed")
            return {"clip_id": clip_id, "status": "failed", "error": str(e)}

    def rerender_clip(self, clip_id: int, caption_position: str = "top", custom_caption: str = "",
                      layout: str = None, mirror: bool = True, hdr_sharpness: bool = True) -> dict:
        """
        Re-renders an existing clip with updated caption placement (top, bottom, none)
        and custom caption text.
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
                hook_banner=hook_banner,
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
