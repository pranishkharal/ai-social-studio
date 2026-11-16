import os
import json
import re
from pathlib import Path
from config import BASE_DIR, GEMINI_API_KEY, GEMINI_MODEL, SUBTITLES_DIR
from database import log_event

# Local speech-to-text fallback
try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
except Exception as e:
    whisper_model = None

# Gemini client
gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        log_event("WARNING", "AIStoryEngine", f"Gemini init error: {e}")

# Strict Shadowban Prevention: Prestige-style Leet-Speak Sanitizer
SHADOWBAN_REPLACEMENTS = {
    r"\bfight\b": "F1GHT",
    r"\bfighting\b": "F1GHTING",
    r"\bgun\b": "G*N",
    r"\bdead\b": "D*AD",
    r"\bkill\b": "K*LL",
    r"\bfuck\b": "F*CK",
    r"\bbully\b": "BULLY",
    r"\bhate\b": "HATE",
    r"\bsuicide\b": "S*ICIDE"
}

# Official 8 Highlight Categories from AI Social Studio PRD (Section 6)
PRD_CATEGORIES = [
    "Funny Moments",
    "Exciting Gameplay",
    "Streamer Reactions",
    "Unexpected Moments",
    "Big Wins",
    "Angry or Emotional Reactions",
    "Viral Potential",
    "Interesting Conversations"
]

class AIStoryEngine:
    def __init__(self):
        SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)

    def prestige_sanitize(self, text: str) -> str:
        """Applies Prestige Leet-Speak tricks (e.g. F1GHT) to prevent 0-view shadowbans."""
        sanitized = text
        for pat, rep in SHADOWBAN_REPLACEMENTS.items():
            sanitized = re.sub(pat, rep, sanitized, flags=re.IGNORECASE)
        return sanitized

    def transcribe_clip(self, video_path: str) -> dict:
        """Transcribes dialogue and extracts word-level timestamps using faster-whisper."""
        if not whisper_model or not os.path.exists(video_path):
            return {"full_text": "", "words": []}
            
        try:
            segments, info = whisper_model.transcribe(video_path, beam_size=1, word_timestamps=True)
            words = []
            full_text_pieces = []
            
            for seg in segments:
                full_text_pieces.append(seg.text)
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        cleaned_word = self.prestige_sanitize(w.word.strip())
                        words.append({
                            "word": cleaned_word,
                            "start": round(w.start, 2),
                            "end": round(w.end, 2)
                        })
                else:
                    words.append({
                        "word": self.prestige_sanitize(seg.text.strip()),
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2)
                    })
                    
            return {
                "full_text": " ".join(full_text_pieces).strip(),
                "words": words,
                "language": info.language
            }
        except Exception as e:
            log_event("ERROR", "AIStoryEngine", f"Whisper transcription failed: {e}")
            return {"full_text": "", "words": []}

    def analyze_story_and_generate_metadata(self, streamer: str, transcript: str, duration: float, category: str = "Gaming", hype_score: float = 75.0) -> dict:
        """
        AI Social Studio v1.0 Story Intelligence:
        - Evaluates event into one of the 8 PRD Highlight Categories
        - Computes 0-100 AI Score & Selection Reason
        - Multi-Platform Metadata: TikTok, YouTube Shorts, Instagram Reels
        - Prestige Formula Captions & Capitalized Trigger Words
        """
        sanitized_transcript = self.prestige_sanitize(transcript)

        if gemini_client and sanitized_transcript.strip():
            categories_str = ", ".join([f'"{c}"' for c in PRD_CATEGORIES])
            prompt = f"""
You are the lead editor and AI content analyst for AI Social Studio v1.0 and @prestigeclipn.
Analyze this Kick live stream moment:

Streamer: {streamer}
Category Hint: {category}
Duration: {duration} seconds
Raw Audio Hype Score: {hype_score} / 100
Dialogue Transcript:
"{sanitized_transcript}"

Strict AI Social Studio PRD Requirements:
1. HIGHLIGHT CATEGORY: Choose exactly ONE from this list:
   [{categories_str}]
2. AI SCORE (0 to 100):
   - 80-100: Exceptional, viral, hilarious or intense gameplay (Auto-save)
   - 60-79: Good moment, worth reviewing
   - 0-59: Mediocre or filler
3. SELECTION REASON: 1 concise sentence explaining why this moment was selected.
4. TIKTOK CAPTION FORMULA:
   - Must strictly follow: "{streamer.title()} [ACTION_VERB_IN_CAPS] [context with selective CAPS words]…[2 EMOJIS]#{streamer.lower()} #[secondary_topic] #fyp"
5. ON-SCREEN STORY CAPTION CARD (Prestige White Sticker Card):
   - Format: "{streamer.title()} was [SHOCKED/SPEECHLESS/HEARTBROKEN/EXPOSED] after [specific context from transcript with CAPITALIZED trigger words] [1-2 EMOJIS]"
   - Example: "N3on was SHOCKED after Ryan Garcia just ANNOUNCED that he's F1GHTING Boxing Champ Floyd Mayweather 😳🥊"
   - Example: "Adin Ross was COMPLETELY SPEECHLESS after LOSING $50,000 on a single hand of Blackjack 💀💸"
   - Must be 12 to 22 words, engaging narrative hook that explains what happened.
6. MULTI-PLATFORM METADATA:
   - youtube_title: Catchy title under 65 chars ending with #Shorts
   - youtube_description: 2 sentence description with search tags
   - instagram_caption: Aesthetic viral caption with emojis

Return ONLY a valid JSON object matching this schema:
{{
  "category": "One of the 8 PRD categories",
  "ai_score": 88,
  "selection_reason": "High emotional intensity combined with sudden streamer shock reaction.",
  "story_summary": "1 sentence description of what occurred",
  "viral_title": "{streamer.title()} [TRIGGER WORD IN CAPS]... 💀",
  "hook_banner": "{streamer.title()} was SHOCKED after [shocking context] 😳🥊",
  "screen_caption": "{streamer.title()} was SHOCKED after [shocking context] 😳🥊",
  "tiktok_caption": "{streamer.title()} finally REVEALS what happened on stream…😳👀#{streamer.lower()} #clips #fyp",
  "youtube_title": "{streamer.title()} WAS LEFT SPEECHLESS! #Shorts",
  "youtube_description": "Watch what happened live on Kick. Subscribe for daily stream moments! #{streamer.lower()} #kickclips #shorts",
  "instagram_caption": "He really did not expect this to happen live… 😳 Drop your thoughts below! 👇 #{streamer.lower()} #reels #explore",
  "blue_comment": "Who else noticed what happened at the end? 😳👇",
  "hashtags": "#{streamer.lower()} #clips #fyp"
}}
"""
            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                data = json.loads(raw_text.strip())
                data["viral_title"] = self.prestige_sanitize(data.get("viral_title", ""))
                screen_cap = data.get("screen_caption") or data.get("hook_banner", "")
                data["hook_banner"] = self.prestige_sanitize(screen_cap)
                data["screen_caption"] = data["hook_banner"]
                data["tiktok_caption"] = self.prestige_sanitize(data.get("tiktok_caption", ""))
                data["blue_comment"] = self.prestige_sanitize(data.get("blue_comment", ""))
                if data.get("category") not in PRD_CATEGORIES:
                    data["category"] = "Streamer Reactions"
                return data
            except Exception as e:
                log_event("WARNING", "AIStoryEngine", f"Gemini story analysis error: {e}")

        # Prestige Live Data Heuristic Fallback
        lower_t = sanitized_transcript.lower()
        if any(w in lower_t for w in ["cry", "sad", "break up", "heartbroken", "anniversary"]):
            cat = "Angry or Emotional Reactions"
            caption = f"{streamer.title()} was HEARTBROKEN after realizing what happened live on stream…😭💔#{streamer.lower()} #clips #fyp"
            banner = f"{streamer.title()} was HEARTBROKEN after realizing what happened live on stream 😭💔"
            title = f"{streamer.title()} was HEARTBROKEN After What Happened 😭💔"
            score = 88
            reason = "High emotional intensity and dramatic streamer breakdown."
        elif any(w in lower_t for w in ["win", "won", "clutch", "victory", "champion", "jackpot"]):
            cat = "Big Wins"
            caption = f"{streamer.title()} just HIT THE BIGGEST WIN live on stream…🤑🏆#{streamer.lower()} #clips #fyp"
            banner = f"{streamer.title()} was SHOCKED after hitting the BIGGEST WIN live on stream 🤑🏆"
            title = f"{streamer.title()} Hits INSANE VICTORY Live 🏆"
            score = 92
            reason = "Massive victory moment with intense celebratory energy."
        elif any(w in lower_t for w in ["game", "kill", "headshot", "shot", "ace", "clutch"]):
            cat = "Exciting Gameplay"
            caption = f"{streamer.title()} had the MOST INSANE GAMEPLAY sequence live on stream…🎯🔥#{streamer.lower()} #clips #fyp"
            banner = f"{streamer.title()} went ABSOLUTELY CRAZY during this unreal gameplay sequence 🎯🔥"
            title = f"{streamer.title()} with an UNREAL Clutch 🎯🔥"
            score = 85
            reason = "High-velocity competitive gaming sequence."
        elif any(w in lower_t for w in ["fight", "drama", "argue", "hate"]):
            cat = "Streamer Reactions"
            caption = f"{streamer.title()} finally REVEALS the truth behind the DRAMA after what happened…😳👀#{streamer.lower()} #clips #fyp"
            banner = f"{streamer.title()} was SHOCKED after this wild DRAMA broke out live on stream 😳👀"
            title = f"{streamer.title()} finally REVEALS The Truth 😳👀"
            score = 86
            reason = "Unfiltered dramatic conversation with strong audience engagement potential."
        elif any(w in lower_t for w in ["lol", "lmao", "haha", "joke", "funny"]):
            cat = "Funny Moments"
            caption = f"{streamer.title()} could NOT stop laughing after this happened live on stream…😂💀#{streamer.lower()} #clips #fyp"
            banner = f"{streamer.title()} could NOT stop laughing after what just happened live on stream 😂💀"
            title = f"{streamer.title()} COULD NOT STOP LAUGHING 😂💀"
            score = 84
            reason = "Spontaneous comedic situation with genuine laughter."
        else:
            cat = "Viral Potential"
            caption = f"{streamer.title()} was completely SPEECHLESS after this happened live on stream…😳👀#{streamer.lower()} #clips #fyp"
            banner = f"{streamer.title()} was COMPLETELY SPEECHLESS after what happened live on stream 😳🚨"
            title = f"{streamer.title()} was COMPLETELY SPEECHLESS 💀"
            score = int(min(95, max(65, hype_score + 10)))
            reason = f"Notable live stream reaction detected with peak decibel and hype rating of {hype_score}."

        return {
            "category": cat,
            "ai_score": score,
            "selection_reason": reason,
            "story_summary": f"Dramatic unfiltered moment from {streamer}'s stream.",
            "viral_title": title,
            "hook_banner": banner,
            "screen_caption": banner,
            "tiktok_caption": caption,
            "youtube_title": f"{streamer.title()} Left SPEECHLESS on Stream! #Shorts",
            "youtube_description": f"Watch what happened with {streamer} on Kick! Like and subscribe for more viral stream moments. #{streamer.lower()} #kickclips #shorts",
            "instagram_caption": f"Bro really could not believe this happened live… 💀 #{streamer.lower()} #reels #explore",
            "blue_comment": f"Did {streamer} actually mean this or was it just for the stream? Let's settle this in comments 👇",
            "hashtags": f"#{streamer.lower()} #clips #fyp"
        }

    def generate_ass_subtitles(self, words: list, output_ass_path: str, video_width: int = 1080, video_height: int = 1920) -> str:
        """
        Generates Advanced SubStation Alpha (.ass) subtitles with rapid 2-3 word pacing
        to maximize viewer retention and prevent scroll-aways.
        """
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,Impact,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,1,0,0,0,100,100,2,0,1,6,4,2,60,60,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        chunk_size = 3
        
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i + chunk_size]
            if not chunk:
                continue
            
            start_sec = chunk[0]["start"]
            end_sec = chunk[-1]["end"]
            
            def fmt_time(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int(round((t - int(t)) * 100))
                return f"{h:d}:{m:02d}:{s:02d}.{ms:02d}"

            start_str = fmt_time(start_sec)
            end_str = fmt_time(max(start_sec + 0.35, end_sec + 0.08))
            
            text_str = " ".join([w["word"] for w in chunk]).upper()
            line = f"Dialogue: 0,{start_str},{end_str},TikTok,,0,0,0,,{{\\c&H00FFFF&}}{text_str}"
            events.append(line)

        full_ass_content = header + "\n".join(events) + "\n"
        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(full_ass_content)
            
        return output_ass_path
