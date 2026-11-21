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


def _probe_duration(path):
    """Clip duration in seconds via ffprobe, or None when unavailable."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(res.stdout.strip())
    except Exception:
        return None

class HypeDetector:
    def __init__(self):
        # ffmpeg 8 prints mean_volume BEFORE max_volume; older ffmpeg
        # printed max first. Two independent searches work with either
        # order — a single max-then-mean regex silently never matches
        # on ffmpeg 8 and every clip falls back to score 10.0.
        self.max_volume_regex = re.compile(r"max_volume:\s*(-?[\d.]+)\s*dB")
        self.mean_volume_regex = re.compile(r"mean_volume:\s*(-?[\d.]+)\s*dB")

    def _score_from_volumes(self, max_vol: float, mean_vol: float) -> float:
        """Shared hype score (0-100) from volumedetect max/mean levels:
        loud peaks and sudden jumps both count, capped at 50 each."""
        vol_jump = max_vol - mean_vol
        score = 0.0
        if max_vol > -20.0:
            score += min(50.0, (20.0 + max_vol) * 2.5)
        if vol_jump > 10.0:
            score += min(50.0, (vol_jump - 10.0) * 2.5)
        return round(min(100.0, max(0.0, score)), 1)

    def _window_score(self, video_path: str, start: float, duration: float) -> float:
        """Hype score of one audio window, via audio-only volumedetect
        (``-ss``/``-t`` input seeking, ``-vn`` — no video decoding).
        Returns 0.0 when the window cannot be analysed."""
        cmd = [
            "ffmpeg", "-ss", f"{start:.2f}", "-t", f"{duration:.2f}",
            "-i", video_path,
            "-vn",  # audio only — see analyze_audio_energy
            "-af", "volumedetect",
            "-f", "null", "-"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except Exception as e:
            log_event("WARNING", "HypeDetector", f"Peak-window analysis error: {e}")
            return 0.0
        max_match = self.max_volume_regex.search(res.stderr)
        mean_match = self.mean_volume_regex.search(res.stderr)
        if not (max_match and mean_match):
            return 0.0
        return self._score_from_volumes(float(max_match.group(1)),
                                        float(mean_match.group(1)))

    def find_peak_window(self, video_path: str, window_seconds: float = 15,
                         step_seconds: float = 5) -> dict:
        """Find the loudest ``window_seconds`` stretch of a video, for
        short-clip mode (cut a tight 10-15 s clip around the single
        peak moment instead of a fixed 30 s window).

        Scans windows of ``window_seconds`` across the video in
        ``step_seconds`` steps, scoring each with audio-only
        volumedetect, and returns the highest-scoring window as
        ``{"start": float, "end": float, "score": float}`` (seconds,
        rounded to 2 decimals). A final window flush with the end of
        the clip is scanned too, so the tail is never missed when the
        stepping does not land on it.

        A missing file returns all zeros; a video shorter than the
        window returns the whole clip.
        """
        if not os.path.exists(video_path):
            return {"start": 0.0, "end": 0.0, "score": 0.0}
        if window_seconds <= 0:
            return {"start": 0.0, "end": 0.0, "score": 0.0}
        if step_seconds <= 0:
            step_seconds = 5

        duration = _probe_duration(video_path)
        if not duration or duration <= 0:
            return {"start": 0.0, "end": 0.0, "score": 0.0}
        if duration <= window_seconds:
            return {"start": 0.0, "end": round(duration, 2),
                    "score": self._window_score(video_path, 0, duration)}

        best = {"start": 0.0, "end": float(window_seconds), "score": -1.0}
        start = 0.0
        last_start = None
        while start + window_seconds <= duration + 1e-9:
            score = self._window_score(video_path, start, window_seconds)
            if score > best["score"]:
                best = {"start": round(start, 2),
                        "end": round(start + window_seconds, 2),
                        "score": score}
            last_start = start
            start += step_seconds
        # Final window flush with the end of the clip, if the stepping
        # did not land on it (e.g. a 32 s clip with 15 s windows).
        tail_start = duration - window_seconds
        if last_start is None or abs(tail_start - last_start) > 1e-9:
            score = self._window_score(video_path, tail_start, window_seconds)
            if score > best["score"]:
                best = {"start": round(tail_start, 2),
                        "end": round(duration, 2), "score": score}
        return best

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
                score = self._score_from_volumes(max_vol, mean_vol)
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

    def evaluate_clip(self, video_path: str, transcript_text: str = "",
                      visual_moments=None, ocr_events=None,
                      detect_visual=False) -> dict:
        """Combines Audio Energy + prxno Lexical Intelligence + (when
        given) Visual Moments and game-banner OCR for viral accuracy.

        * ``visual_moments``: moments from
          ``visual_moments.detect_visual_moments`` (or ``detect_visual``
          runs the detection here -- lazy import, failures just mean
          audio-only scoring). The best moment's confidence becomes a
          visual score (0-100) blended into the total, and its type is
          reported as ``moment_type`` for headlines.
        * ``ocr_events``: merged events from
          ``game_ocr.scan_game_events``; a WASTED / MISSION PASSED
          banner adds up to 15 points on top.

        With neither signal passed, the scoring is exactly the old
        60% audio + 40% lexical blend.
        """
        audio_metrics = self.analyze_audio_energy(video_path)
        lexical_metrics = self.score_transcript_intelligence(transcript_text)

        moments = list(visual_moments) if visual_moments else []
        if detect_visual and not moments:
            try:
                from visual_moments import detect_visual_moments
                moments = detect_visual_moments(video_path).get("moments", [])
            except Exception as e:
                log_event("WARNING", "HypeDetector",
                          f"Visual moment detection failed: {e}")
                moments = []
        best_moment = None
        visual_score = 0.0
        if moments:
            best_moment = max(moments, key=lambda m: m.get("score", 0.0))
            visual_score = round(
                float(best_moment.get("score", 0.0)) * 100, 1)
        moment_type = best_moment.get("type") if best_moment else None

        ocr_bonus = 0.0
        ocr_labels = []
        if ocr_events:
            try:
                from game_ocr import ocr_score_bonus
                ocr_bonus = ocr_score_bonus(ocr_events)
            except Exception as e:
                log_event("WARNING", "HypeDetector",
                          f"OCR bonus failed: {e}")
                ocr_bonus = 0.0
            ocr_labels = [e.get("label") for e in ocr_events
                          if e.get("kind") == "word" and e.get("label")]

        # 60% Audio Energy + 40% Lexical Intelligence, re-weighted when
        # a visual signal exists (visual 20%, audio/lexical share the
        # rest); without a transcript the visual signal shares with
        # audio 70/30.
        if transcript_text:
            if moments:
                total_score = round(
                    audio_metrics["score"] * 0.5
                    + lexical_metrics["score"] * 0.3
                    + visual_score * 0.2, 1)
            else:
                total_score = round(
                    audio_metrics["score"] * 0.6
                    + lexical_metrics["score"] * 0.4, 1)
        elif moments:
            total_score = round(
                audio_metrics["score"] * 0.7 + visual_score * 0.3, 1)
        else:
            total_score = audio_metrics["score"]
        total_score = round(min(100.0, total_score + ocr_bonus), 1)

        is_viral = total_score >= 55.0 or audio_metrics.get("is_hype", False)

        trigger_reason = audio_metrics.get("reason", "Hype Threshold Crossed")
        if ocr_labels:
            trigger_reason += f" + game banner: {', '.join(ocr_labels)}"
        if moment_type:
            trigger_reason += f" + visual moment: {moment_type}"

        return {
            "hype_score": total_score,
            "is_viral_candidate": is_viral,
            "trigger_reason": trigger_reason,
            "audio_metrics": audio_metrics,
            "lexical_metrics": lexical_metrics,
            "visual_score": visual_score,
            "moment_type": moment_type,
            "visual_moments": moments,
            "ocr_bonus": ocr_bonus,
            "ocr_events": list(ocr_events) if ocr_events else [],
        }
