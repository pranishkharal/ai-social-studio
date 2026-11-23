import os
import json
import re
from pathlib import Path
from config import BASE_DIR, GEMINI_API_KEY, GEMINI_MODEL, SUBTITLES_DIR
from database import log_event
from retry_utils import retry
from llm_providers import generate_metadata_with_fallback
from story_headlines import (
    build_story_headline,
    compose_hashtags,
    compose_post_caption,
    pick_quote_line,
)

# Local speech-to-text (faster-whisper), loaded LAZILY on first use.
# Importing this module must never download or load a model -- the model
# is created on the first transcribe_clip() call and then cached.
# Configured via env vars (see _whisper_config); defaults keep the
# historical behaviour: tiny model, CPU, int8, beam size 1.
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# Backwards-compatible handle: None until the first lazy load, then the
# loaded model. Prefer get_whisper_model() -- this is only kept so code
# that referenced the old module-level variable keeps working.
whisper_model = None
_whisper_model_key = None

KNOWN_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large",
                        "large-v2", "large-v3", "large-v3-turbo", "turbo"}


def _whisper_config():
    """
    Whisper settings from env vars:
        WHISPER_MODEL         Model size: tiny/small/medium/... (default "tiny")
        WHISPER_DEVICE        "cpu" (default) or "cuda"
        WHISPER_BEAM_SIZE     Beam size for decoding (default 1)
        WHISPER_COMPUTE_TYPE  ctranslate2 compute type (default "int8")
    """
    model = os.getenv("WHISPER_MODEL", "tiny").strip().lower() or "tiny"
    if model not in KNOWN_WHISPER_MODELS:
        log_event("WARNING", "AIStoryEngine",
                  f"Unknown WHISPER_MODEL '{model}'; falling back to 'tiny'")
        model = "tiny"
    device = os.getenv("WHISPER_DEVICE", "cpu").strip().lower() or "cpu"
    try:
        beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
    except ValueError:
        beam_size = 1
    beam_size = max(1, beam_size)
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
    return {"model": model, "device": device,
            "beam_size": beam_size, "compute_type": compute_type}


# Whisper vocabulary prompt (gta6-clipper feature 9): without a prompt,
# Whisper hears what it expects — "Chad" for "chat", game and streamer
# names mangled — and those mistranscriptions leak into headlines (the
# TikTok build caught "Ethan wants to be" in a demo headline). Passing
# the game's and streamer's names plus chat slang as initial_prompt
# steers the tiny model to the right words, and also makes it punctuate,
# which the sentence snapping needs. Set WHISPER_VOCAB_PROMPT to replace
# the built prompt entirely.
WHISPER_VOCAB_GAMES = [
    "GTA 6", "Grand Theft Auto VI", "GTA 5", "San Andreas", "Vice City",
    "Leonida", "Lucia", "Jason", "Rockstar",
]
WHISPER_VOCAB_SLANG = ["Chat", "KEKW", "LUL", "W", "bro"]


def build_vocab_prompt(streamer=None, game=None):
    """Build the Whisper initial_prompt for one clip.

    The game's name goes first (when known), then the built-in game
    vocabulary, then the streamer's name, then chat slang — two short
    sentences, deduplicated case-insensitively. An explicit
    WHISPER_VOCAB_PROMPT env var replaces the whole prompt.
    """
    override = os.getenv("WHISPER_VOCAB_PROMPT", "").strip()
    if override:
        return override
    names = []
    seen = set()

    def _add(name):
        name = (name or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            names.append(name)

    _add(game)
    for name in WHISPER_VOCAB_GAMES:
        _add(name)
    _add(streamer)
    slang = ", ".join(WHISPER_VOCAB_SLANG)
    return f"{', '.join(names)}. {slang}."


def get_whisper_model():
    """
    Return the cached Whisper model, loading it on first call.

    WHISPER_DEVICE=cuda attempts the GPU and falls back to CPU with a
    logged warning when CUDA is unavailable (no GPU, no CUDA build, or a
    compute type the GPU build rejects). Returns None when faster-whisper
    is not installed or the model cannot be loaded at all.
    """
    global whisper_model, _whisper_model_key
    cfg = _whisper_config()
    key = (cfg["model"], cfg["device"], cfg["compute_type"])
    if whisper_model is not None and _whisper_model_key == key:
        return whisper_model

    if WhisperModel is None:
        return None

    device = cfg["device"]
    try:
        model = WhisperModel(cfg["model"], device=device,
                             compute_type=cfg["compute_type"])
    except Exception as e:
        if device == "cuda":
            log_event("WARNING", "AIStoryEngine",
                      f"CUDA Whisper load failed ({e}); falling back to CPU")
            try:
                model = WhisperModel(cfg["model"], device="cpu",
                                     compute_type=cfg["compute_type"])
                device = "cpu"
            except Exception as e2:
                log_event("ERROR", "AIStoryEngine",
                          f"Whisper model load failed on CPU too: {e2}")
                return None
        else:
            log_event("ERROR", "AIStoryEngine",
                      f"Whisper model load failed: {e}")
            return None

    whisper_model = model
    _whisper_model_key = (cfg["model"], device, cfg["compute_type"])
    return model

# Gemini key manager (bulk keys + rotation -- see gemini_keys.py).
# Import-safe: when the manager cannot be built, gemini_keys is None
# and every Gemini call raises, so the heuristic fallback runs exactly
# as before. gemini_client is a rotating stand-in for genai.Client:
# existing callers (captions.py's optional LLM path) keep working
# unchanged and get rotation for free.
try:
    from gemini_keys import (
        GeminiUnavailableError, NoHealthyKeyError, RotatingGeminiClient,
        get_key_manager,
    )
    gemini_keys = get_key_manager()
    gemini_client = RotatingGeminiClient(gemini_keys)
except Exception as e:
    log_event("WARNING", "AIStoryEngine",
              f"Gemini key manager unavailable: {e}")
    gemini_keys = None
    gemini_client = None

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

    def transcribe_clip(self, video_path: str, streamer: str = None,
                        game: str = None) -> dict:
        """Transcribes dialogue and extracts word-level timestamps using faster-whisper.

        streamer / game feed the vocabulary prompt (build_vocab_prompt):
        the names Whisper should expect to hear. Both are optional —
        without them the built-in game + slang vocabulary is used.
        """
        model = get_whisper_model()
        if model is None or not os.path.exists(video_path):
            return {"full_text": "", "words": []}

        beam_size = _whisper_config()["beam_size"]
        try:
            return self._transcribe_attempt(model, video_path, beam_size,
                                            streamer=streamer, game=game)
        except Exception as e:
            log_event("ERROR", "AIStoryEngine", f"Whisper transcription failed: {e}")
            return {"full_text": "", "words": []}

    @retry(max_attempts=2)
    def _transcribe_attempt(self, model, video_path, beam_size,
                            streamer=None, game=None):
        """One Whisper transcription attempt.

        Retried once (via @retry) on transient failure — e.g. a flaky
        model load or a locked file — before transcribe_clip falls back
        to an empty transcript. The vocabulary prompt (game + streamer
        names + chat slang) is passed as initial_prompt, and the VAD
        filter skips non-speech so silence is not hallucinated as words.
        """
        segments, info = model.transcribe(
            video_path, beam_size=beam_size, word_timestamps=True,
            vad_filter=True,
            initial_prompt=build_vocab_prompt(streamer, game))
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

    def analyze_story_and_generate_metadata(self, streamer: str, transcript: str, duration: float, category: str = "Gaming", hype_score: float = 75.0, moment_type: str = None) -> dict:
        """
        AI Social Studio v1.0 Story Intelligence:
        - Evaluates event into one of the 8 PRD Highlight Categories
        - Computes 0-100 AI Score & Selection Reason
        - Multi-Platform Metadata: TikTok, YouTube Shorts, Instagram Reels
        - Prestige Formula Captions & Capitalized Trigger Words
        - Provider chain (Tier 5): Gemini -> OpenAI-compatible local
          model (LM Studio / vLLM / NVIDIA NIM) -> heuristic fallback.
          The winning provider is recorded as "llm_provider".
        """
        sanitized_transcript = self.prestige_sanitize(transcript)

        if not sanitized_transcript.strip():
            # No dialogue to analyze — heuristic fallback only.
            return self._heuristic_metadata(streamer, sanitized_transcript, hype_score, moment_type)

        prompt = self._build_analysis_prompt(
            streamer, category, duration, hype_score, sanitized_transcript)

        def gemini_fn(_prompt):
            if gemini_keys is None or not gemini_keys.has_keys():
                raise RuntimeError("Gemini client not configured")
            last_error = None
            # One attempt per configured key at most: each attempt uses
            # the next healthy key (round-robin), and report_failure
            # cools down / invalidates the failed key, so the loop
            # always terminates. The last error is raised only when no
            # healthy key remains -- then the fallback chain proceeds
            # to the local model and heuristics exactly as before.
            for _ in range(max(gemini_keys.key_count(), 1)):
                try:
                    client, key_id = gemini_keys.get_client()
                except GeminiUnavailableError:
                    raise  # no SDK installed -- more attempts can't help
                except NoHealthyKeyError:
                    break  # every key is down; raise the last error
                try:
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=_prompt,
                    )
                    raw_text = response.text.strip()
                except Exception as e:
                    gemini_keys.report_failure(key_id, e)
                    last_error = e
                    continue
                gemini_keys.report_success(key_id)
                # Parsing happens AFTER the success report: a bad
                # response is not the key's fault, so it must not
                # count against the key. A parse failure propagates
                # as before (the fallback chain catches it).
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                data = json.loads(raw_text.strip())
                return self._sanitize_metadata(data)
            if last_error is not None:
                raise RuntimeError(
                    f"Gemini failed on all keys: {last_error}")
            raise RuntimeError("Gemini client not configured")

        def heuristic_fn(_prompt):
            return self._heuristic_metadata(streamer, sanitized_transcript, hype_score, moment_type)

        return generate_metadata_with_fallback(prompt, gemini_fn, heuristic_fn)

    def _build_analysis_prompt(self, streamer, category, duration, hype_score, sanitized_transcript):
        """The story-analysis prompt, shared by every LLM provider."""
        categories_str = ", ".join([f'"{c}"' for c in PRD_CATEGORIES])
        return f"""
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
4. VIRAL CAPTION GUIDE STRATEGY (Watch Time & Comment Baiting):
   - Match the clip to the best viral strategy from the Caption Guide:
     * Celebrity Encounter: "the moment {streamer} had NO IDEA who he just met ❤️🩹" (Leave celebrity's name anonymous to drive comment debates)
     * Song Reaction: "{streamer} was SHOCKED after this rapper dropped a GRAMMY WINNING SONG 😭❤️🩹" (Keep rapper anonymous for mystery)
     * Emotional / Heartbreak: "the emotional moment {streamer} realized what happened ❤️🩹" or "{streamer} was HEARTBROKEN after [context] ❤️🩹"
     * Apology / Maturity: "the wholesome moment {streamer} showed MATURITY and apologized after what happened ❤️🩹"
     * Career Milestone: "{streamer} had his \"I MADE IT MOMENT\" and was emotional after what happened ❤️🩹"
     * Viral / High Stakes: Provide specific details, exact dollar amounts, cars/items, or consequences (e.g. "{streamer} was EMOTIONAL after realizing he had to pay $25,000 to fix Ron's spoiler on his $400,000 Porsche GT3 RS")
   - Boost with trigger words in ALL CAPS where appropriate: PROVED, SPEECHLESS, REGRETTED, REVEALED
   - Target watch-time: leave out just enough info so viewers must finish the clip or check comments.
   - End with emojis (❤️🩹, 😭, 😳, 💀) and hashtags: "#{streamer.lower()} #fyp #kick"
5. ON-SCREEN STORY CAPTION CARD (Prestige White Sticker Card):
   - Format: "{streamer.title()} was [SHOCKED/SPEECHLESS/HEARTBROKEN/EXPOSED] after [specific context from transcript with CAPITALIZED trigger words] [1-2 EMOJIS]"
   - Example: "N3on was SHOCKED after Ryan Garcia just ANNOUNCED that he's F1GHTING Boxing Champ Floyd Mayweather 😳🥊"
   - Example: "Adin Ross was COMPLETELY SPEECHLESS after LOSING $50,000 on a single hand of Blackjack 💀💸"
   - Must be 12 to 22 words, engaging narrative hook that explains what happened.
6. STORY HEADLINE RULE (most important): viral_title and hook_banner must
   name the SPECIFIC moment from the transcript -- a verb + object taken
   from what was actually said, plus a name or quoted phrase when one
   appears (e.g. "Adinross finds his calling after Chaz calls him out",
   "Zavala breaks the guitar Amanda got him"). Do NOT write a generic
   "SPEECHLESS" / "SHOCKED" line unless the transcript genuinely shows
   shock -- a headline that could describe any clip is a bad headline.
7. QUOTE LINE: pick the single punchiest spoken line from the transcript
   (exact words, 4-15 words) and return it as "quote_line".
8. MULTI-PLATFORM METADATA:
   - youtube_title: Catchy title under 65 chars ending with #Shorts
   - youtube_description: 2 sentence description with search tags
   - instagram_caption: Aesthetic viral caption with emojis

Return ONLY a valid JSON object matching this schema:
{{
  "category": "One of the 8 PRD categories",
  "ai_score": 88,
  "selection_reason": "High emotional intensity combined with sudden streamer shock reaction.",
  "story_summary": "1 sentence description of what occurred",
  "viral_title": "{streamer.title()} finds his calling after Chaz calls him out",
  "hook_banner": "{streamer.title()} finds his calling after Chaz calls him out 😳",
  "screen_caption": "{streamer.title()} finds his calling after Chaz calls him out 😳",
  "quote_line": "Best spoken line, exact words from the transcript, 4-15 words",
  "tiktok_caption": "{streamer.title()} finally REVEALS what happened on stream…😳👀#{streamer.lower()} #clips #fyp",
  "youtube_title": "{streamer.title()} WAS LEFT SPEECHLESS! #Shorts",
  "youtube_description": "Watch what happened live on Kick. Subscribe for daily stream moments! #{streamer.lower()} #kickclips #shorts",
  "instagram_caption": "He really did not expect this to happen live… 😳 Drop your thoughts below! 👇 #{streamer.lower()} #reels #explore",
  "blue_comment": "Who else noticed what happened at the end? 😳👇",
  "hashtags": "#{streamer.lower()} #clips #fyp"
}}
"""

    def _sanitize_metadata(self, data):
        """Prestige sanitizer + category validation for LLM metadata.

        Shared by every provider in the fallback chain so Gemini and the
        local model get identical post-processing.
        """
        data["viral_title"] = self.prestige_sanitize(data.get("viral_title", ""))
        screen_cap = data.get("screen_caption") or data.get("hook_banner", "")
        data["hook_banner"] = self.prestige_sanitize(screen_cap)
        data["screen_caption"] = data["hook_banner"]
        data["tiktok_caption"] = self.prestige_sanitize(data.get("tiktok_caption", ""))
        data["blue_comment"] = self.prestige_sanitize(data.get("blue_comment", ""))
        quote = self.prestige_sanitize(str(data.get("quote_line") or ""))
        # A quote line longer than 120 chars is not a punchy one-liner.
        data["quote_line"] = quote if len(quote) <= 120 else ""
        if data.get("category") not in PRD_CATEGORIES:
            data["category"] = "Streamer Reactions"
        return data

    def _heuristic_metadata(self, streamer, sanitized_transcript, hype_score, moment_type=None):
        """Prestige Live Data Heuristic Fallback — keyword-driven metadata.

        Always available locally; used when no LLM provider answers (or
        when the transcript is empty).
        """
        lower_t = sanitized_transcript.lower()
        if any(w in lower_t for w in ["cry", "sad", "break up", "heartbroken", "anniversary"]):
            cat = "Angry or Emotional Reactions"
            emoji = "😭💔"
            score = 88
            reason = "High emotional intensity and dramatic streamer breakdown."
        elif any(w in lower_t for w in ["win", "won", "clutch", "victory", "champion", "jackpot"]):
            cat = "Big Wins"
            emoji = "🤑🏆"
            score = 92
            reason = "Massive victory moment with intense celebratory energy."
        elif any(w in lower_t for w in ["game", "kill", "headshot", "shot", "ace", "clutch"]):
            cat = "Exciting Gameplay"
            emoji = "🎯🔥"
            score = 85
            reason = "High-velocity competitive gaming sequence."
        elif any(w in lower_t for w in ["fight", "drama", "argue", "hate"]):
            cat = "Streamer Reactions"
            emoji = "😳👀"
            score = 86
            reason = "Unfiltered dramatic conversation with strong audience engagement potential."
        elif any(w in lower_t for w in ["lol", "lmao", "haha", "joke", "funny"]):
            cat = "Funny Moments"
            emoji = "😂💀"
            score = 84
            reason = "Spontaneous comedic situation with genuine laughter."
        else:
            cat = "Viral Potential"
            emoji = "😳🚨"
            score = int(min(95, max(65, hype_score + 10)))
            reason = f"Notable live stream reaction detected with peak decibel and hype rating of {hype_score}."

        # Story headline (Feature 1), quote line (Feature 2) and post
        # caption with hashtags (Feature 4) -- all built from the actual
        # transcript instead of the old generic SPEECHLESS lines.
        headline = build_story_headline(
            streamer, sanitized_transcript, category=cat,
            moment_type=moment_type)
        quote = pick_quote_line(sanitized_transcript)
        banner = f"{headline} {emoji}".strip()
        caption = compose_post_caption(headline, streamer)

        return {
            "category": cat,
            "ai_score": score,
            "selection_reason": reason,
            "story_summary": f"Dramatic unfiltered moment from {streamer}'s stream.",
            "viral_title": headline,
            "hook_banner": banner,
            "screen_caption": banner,
            "quote_line": quote,
            "tiktok_caption": caption,
            "youtube_title": f"{streamer.title()} Left SPEECHLESS on Stream! #Shorts",
            "youtube_description": f"Watch what happened with {streamer} on Kick! Like and subscribe for more viral stream moments. #{streamer.lower()} #kickclips #shorts",
            "instagram_caption": f"Bro really could not believe this happened live… 💀 #{streamer.lower()} #reels #explore",
            "blue_comment": f"Did {streamer} actually mean this or was it just for the stream? Let's settle this in comments 👇",
            "hashtags": compose_hashtags(streamer),
            "llm_provider": "heuristic",
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
