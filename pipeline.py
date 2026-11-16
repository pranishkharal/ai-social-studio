import os
import time
from datetime import datetime
from pathlib import Path

from config import EDITED_DIR, SUBTITLES_DIR, VIDEO_LAYOUT
from kick_monitor import KickMonitor
from hype_detector import HypeDetector
from ai_story_engine import AIStoryEngine
from clip_editor import ClipEditor
from database import add_clip, update_clip_status, log_event, add_notification, get_connection

class ClipPipeline:
    def __init__(self, monitor: KickMonitor = None):
        self.monitor = monitor or KickMonitor()
        self.detector = HypeDetector()
        self.ai = AIStoryEngine()
        self.editor = ClipEditor()

    def process_moment(self, streamer: str, duration: int = 25, raw_clip_path: str = None, 
                       mirror: bool = True, hdr_sharpness: bool = True, layout: str = None,
                       auto_captions: bool = False, caption_position: str = "top",
                       custom_caption: str = None) -> dict:
        """
        Full Autonomous Pipeline with Prestige Clipping Rules:
        - Dead Space elimination (fast 15-30s pacing)
        - Mirror clip protection (Anti-duplicate content algorithm)
        - HDR Sharpness boost (Algorithm quality score trick)
        - Mystery-baiting captions with trigger words (SPEECHLESS, FOLDED, REVEALED)
        - Blue comment pre-seeding for TikTok search ranking
        """
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

        try:
            # 3. Transcribe dialogue with Whisper (PG sanitized)
            log_event("INFO", "Pipeline", f"Transcribing audio for clip #{clip_id}")
            transcript_data = self.ai.transcribe_clip(raw_clip_path)
            transcript_text = transcript_data.get("full_text", "")
            words = transcript_data.get("words", [])

            # 4. Analyze Storyline, Highlight Category & AI Scoring (PRD Section 6 & 7)
            log_event("INFO", "Pipeline", f"Analyzing narrative & AI Social Studio scoring for clip #{clip_id}")
            metadata = self.ai.analyze_story_and_generate_metadata(
                streamer=streamer,
                transcript=transcript_text,
                duration=duration,
                hype_score=hype_score
            )

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

            # 5. Generate Subtitles (Only if enabled, default is OFF)
            ass_path = str(SUBTITLES_DIR / f"clip_{clip_id}.ass")
            if auto_captions and words:
                self.ai.generate_ass_subtitles(words, ass_path)
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
                layout=layout or VIDEO_LAYOUT,
                ass_subtitle_path=ass_path,
                hook_banner=hook_banner,
                mirror=mirror,
                hdr_sharpness=hdr_sharpness,
                auto_captions=auto_captions,
                caption_position=caption_position
            )

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
                approval_status=approval_status
            )

            # 8. In-App Notification (PRD Section 13)
            add_notification(
                title=f"New Highlight Ready ({ai_score}/100)",
                message=f"[{category}] {streamer}: {viral_title}",
                notif_type="highlight_ready",
                streamer=streamer,
                clip_id=clip_id,
                ai_score=ai_score
            )

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
                "edited_path": output_vertical_path
            }

        except Exception as e:
            log_event("ERROR", "Pipeline", f"Pipeline failure for clip #{clip_id}: {e}")
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
        chosen_layout = layout or VIDEO_LAYOUT

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
