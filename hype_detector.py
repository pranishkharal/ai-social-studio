import re
import os
import subprocess
from pathlib import Path
from config import HYPE_VOLUME_SPIKE_DB, HYPE_VOLUME_JUMP_THRESHOLD
from database import log_event

IMPORTANT_WORDS = {
    "what", "why", "how", "wait", "bro", "nah", "no", "nope", "crazy", 
    "insane", "actually", "seriously", "never", "finally", "damn", "wow", 
    "omg", "wtf", "look", "listen", "guys", "chat", "literally", "impossible", "really"
}

REACTION_WORDS = {
    "laugh", "laughing", "lol", "lmao", "haha", "hahaha", "scream", 
    "screaming", "shout", "shouting", "cry", "crying", "angry", "mad", 
    "shock", "shocked", "surprised", "folded", "fumbled"
}

class HypeDetector:
    def __init__(self):
        # ffmpeg 8 prints mean_volume BEFORE max_volume; older ffmpeg
        # printed max first. Two independent searches work with either
        # order — a single max-then-mean regex silently never matches
        # on ffmpeg 8 and every clip falls back to score 10.0.
        self.max_volume_regex = re.compile(r"max_volume:\s*(-?[\d.]+)\s*dB")
        self.mean_volume_regex = re.compile(r"mean_volume:\s*(-?[\d.]+)\s*dB")

    def analyze_audio_energy(self, video_path: str) -> dict:
        """Runs fast FFmpeg volume detection on a video clip.

        Audio-only (``-vn``): volumedetect needs no video, and decoding
        a 1080p60 video stream to null is slow enough that the analysis
        timed out and fell back to 10.0 on a loaded machine. Audio-only
        typically takes well under a second even for a 30 s clip.
        """
        if not os.path.exists(video_path):
            return {"score": 0.0, "is_hype": False, "reason": "file_not_found"}

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn",  # audio only — see docstring
            "-af", "volumedetect",
            "-f", "null", "-"
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = res.stderr
            max_match = self.max_volume_regex.search(output)
            mean_match = self.mean_volume_regex.search(output)

            if max_match and mean_match:
                max_vol = float(max_match.group(1))
                mean_vol = float(mean_match.group(1))
                vol_jump = max_vol - mean_vol
                
                score = 0.0
                if max_vol > -20.0:
                    score += min(50.0, (20.0 + max_vol) * 2.5)
                if vol_jump > 10.0:
                    score += min(50.0, (vol_jump - 10.0) * 2.5)
                
                score = round(min(100.0, max(0.0, score)), 1)
                is_hype = (max_vol >= HYPE_VOLUME_SPIKE_DB) and (vol_jump >= HYPE_VOLUME_JUMP_THRESHOLD)
                
                return {
                    "score": score,
                    "max_volume": max_vol,
                    "mean_volume": mean_vol,
                    "vol_jump": vol_jump,
                    "is_hype": is_hype,
                    "reason": "Sudden scream / laughter / audio volume spike" if is_hype else "Normal talking volume"
                }
        except Exception as e:
            log_event("WARNING", "HypeDetector", f"Audio analysis error: {e}")
            
        return {"score": 10.0, "is_hype": False, "reason": "analysis_fallback"}

    def score_transcript_intelligence(self, transcript_text: str) -> dict:
        """
        Calculates lexical intelligence score from prxno:
        - Hook words in first 12 words
        - Reaction keywords
        - Dialogue density
        """
        if not transcript_text:
            return {"score": 0.0, "hook_score": 0.0}

        words = re.findall(r"[A-Za-z0-9']+", transcript_text.lower())
        if not words:
            return {"score": 0.0, "hook_score": 0.0}

        word_count = len(words)
        important_hits = sum(1 for w in words if w in IMPORTANT_WORDS)
        reaction_hits = sum(1 for w in words if w in REACTION_WORDS)

        keyword_score = min(100.0, important_hits * 12.0)
        reaction_score = min(100.0, reaction_hits * 18.0)
        density_score = min(100.0, word_count * 2.5)

        first_words = words[:12]
        hook_hits = sum(1 for w in first_words if w in IMPORTANT_WORDS)
        hook_score = min(100.0, hook_hits * 22.0)

        lexical_score = round(
            hook_score * 0.35 + reaction_score * 0.30 + keyword_score * 0.20 + density_score * 0.15,
            1
        )
        return {
            "score": lexical_score,
            "hook_score": hook_score,
            "reaction_hits": reaction_hits,
            "important_hits": important_hits
        }

    def evaluate_clip(self, video_path: str, transcript_text: str = "") -> dict:
        """Combines Audio Energy + prxno Lexical Intelligence for maximum viral accuracy."""
        audio_metrics = self.analyze_audio_energy(video_path)
        lexical_metrics = self.score_transcript_intelligence(transcript_text)
        
        # 60% Audio Energy + 40% Lexical Intelligence
        if transcript_text:
            total_score = round(audio_metrics["score"] * 0.6 + lexical_metrics["score"] * 0.4, 1)
        else:
            total_score = audio_metrics["score"]
            
        is_viral = total_score >= 55.0 or audio_metrics.get("is_hype", False)
        
        return {
            "hype_score": total_score,
            "is_viral_candidate": is_viral,
            "trigger_reason": audio_metrics.get("reason", "Hype Threshold Crossed"),
            "audio_metrics": audio_metrics,
            "lexical_metrics": lexical_metrics
        }
