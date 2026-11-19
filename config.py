import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables if present
load_dotenv(BASE_DIR / ".env")

# Monitored Streamers (Kick usernames)
STREAMERS_TO_MONITOR = ["xqc", "adinross", "westcol", "tarik"]

# System settings
CHECK_INTERVAL_SECONDS = 15  # Polling interval to check if streamers are live (uses < 0.01% CPU)
STREAM_BUFFER_MINUTES = 10   # Rolling disk buffer of latest live stream footage
DEFAULT_CLIP_SECONDS = 45    # Default length of an extracted hype clip

# Hype Detection Sensitivities
HYPE_VOLUME_SPIKE_DB = -15.0     # Audio peak volume to trigger hype detection (screams, laughter)
HYPE_VOLUME_JUMP_THRESHOLD = 8.0 # Sudden decibel jump compared to average
CHAT_VELOCITY_SPIKE_MULTIPLIER = 3.0 # Spike multiplier in chat msgs/sec

# AI Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"  # Fast, highly accurate multimodal & storytelling model

# Video Editing Configuration
VIDEO_LAYOUT = "speaker_tracking"  # Options: 'speaker_tracking' (dynamic speaker auto-framing), 'split', 'blurred_backdrop', 'center_crop'
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
SUBTITLE_FONT_SIZE = 48
SUBTITLE_HIGHLIGHT_COLOR = "&H0000FFFF"  # Yellow highlight in ASS format
SUBTITLE_PRIMARY_COLOR = "&H00FFFFFF"    # White text

# Paths
BUFFER_DIR = BASE_DIR / "buffer"
CLIPS_DIR = BASE_DIR / "clips"
EDITED_DIR = BASE_DIR / "edited"
SUBTITLES_DIR = BASE_DIR / "subtitles"
DATABASE_FILE = BASE_DIR / "kick_engine.db"

# ---------------------------------------------------------------------------
# Live-clipping upgrades (features 1-17 + Tier 5) — central settings.
# Every value is read from the environment with a safe default that
# preserves the pre-upgrade behaviour. The feature modules also read the
# same variables directly, so these constants are the documented source of
# truth for wiring code (dashboard, pipeline) — keep defaults in sync.
# See .env.example for the full list with explanations.
# ---------------------------------------------------------------------------
def _env_bool(name, default=False):
    return os.getenv(name, "true" if default else "false").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# --- Tier 1: auto-clip watcher ---
AUTO_CLIP_ENABLED = _env_bool("AUTO_CLIP_ENABLED", False)
AUTO_CLIP_MIN_SCORE = _env_int("AUTO_CLIP_MIN_SCORE", 70)
AUTO_CLIP_CHECK_SECONDS = _env_int("AUTO_CLIP_CHECK_SECONDS", 5)
AUTO_CLIP_DURATION = _env_int("AUTO_CLIP_DURATION", 25)
AUTO_CLIP_SAMPLE_SECONDS = _env_int("AUTO_CLIP_SAMPLE_SECONDS", 5)
AUTO_CLIP_COOLDOWN_SECONDS = _env_int("AUTO_CLIP_COOLDOWN_SECONDS", 90)
AUTO_CLIP_DUP_SIMILARITY = _env_float("AUTO_CLIP_DUP_SIMILARITY", 0.7)
AUTO_CLIP_DUP_WINDOW_MINUTES = _env_int("AUTO_CLIP_DUP_WINDOW_MINUTES", 30)

# --- Tier 1: chat hype (Kick chat via Pusher) ---
CHAT_HYPE_ENABLED = _env_bool("CHAT_HYPE_ENABLED", False)
KICK_PUSHER_APP_KEY = os.getenv("KICK_PUSHER_APP_KEY", "32cbd69e4b950bf97679")
CHAT_HYPE_WINDOW_SECONDS = _env_int("CHAT_HYPE_WINDOW_SECONDS", 10)
CHAT_HYPE_THRESHOLD = _env_int("CHAT_HYPE_THRESHOLD", 70)
CHAT_HYPE_CALLBACK_COOLDOWN = _env_int("CHAT_HYPE_CALLBACK_COOLDOWN", 30)

# --- Tier 1: Kick webhooks / OAuth ---
KICK_CLIENT_ID = os.getenv("KICK_CLIENT_ID", "")
KICK_CLIENT_SECRET = os.getenv("KICK_CLIENT_SECRET", "")
KICK_ACCESS_TOKEN = os.getenv("KICK_ACCESS_TOKEN", "")
KICK_WEBHOOK_URL = os.getenv("KICK_WEBHOOK_URL", "")
KICK_WEBHOOK_SECRET = os.getenv("KICK_WEBHOOK_SECRET", "")

# --- Tier 2: YouTube + Instagram uploaders ---
YOUTUBE_CLIENT_SECRETS_FILE = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", str(BASE_DIR / "client_secrets.json"))
YOUTUBE_TOKEN_FILE = os.getenv("YOUTUBE_TOKEN_FILE", str(BASE_DIR / "token.json"))
META_IG_USER_ID = os.getenv("META_IG_USER_ID", "")
META_IG_ACCESS_TOKEN = os.getenv("META_IG_ACCESS_TOKEN", "")
META_GRAPH_API_VERSION = os.getenv("META_GRAPH_API_VERSION", "v21.0")
INSTAGRAM_POLL_INTERVAL_SECONDS = _env_int("INSTAGRAM_POLL_INTERVAL_SECONDS", 5)
INSTAGRAM_POLL_TIMEOUT_SECONDS = _env_int("INSTAGRAM_POLL_TIMEOUT_SECONDS", 300)

# --- Tier 2: upload scheduler ---
TIKTOK_MIN_GAP_MINUTES = _env_int("TIKTOK_MIN_GAP_MINUTES", 180)
YOUTUBE_MIN_GAP_MINUTES = _env_int("YOUTUBE_MIN_GAP_MINUTES", 120)
INSTAGRAM_MIN_GAP_MINUTES = _env_int("INSTAGRAM_MIN_GAP_MINUTES", 180)
TIKTOK_MAX_PER_DAY = _env_int("TIKTOK_MAX_PER_DAY", 5)
YOUTUBE_MAX_PER_DAY = _env_int("YOUTUBE_MAX_PER_DAY", 5)
INSTAGRAM_MAX_PER_DAY = _env_int("INSTAGRAM_MAX_PER_DAY", 5)
UPLOAD_MAX_ATTEMPTS = _env_int("UPLOAD_MAX_ATTEMPTS", 3)
UPLOAD_SCHEDULER_POLL_SECONDS = _env_int("UPLOAD_SCHEDULER_POLL_SECONDS", 60)

# --- Tier 3: transcription / silence / profiles ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_BEAM_SIZE = _env_int("WHISPER_BEAM_SIZE", 1)
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SILENCE_MAX_GAP_SECONDS = _env_float("SILENCE_MAX_GAP_SECONDS", 1.5)
SILENCE_PAD_SECONDS = _env_float("SILENCE_PAD_SECONDS", 0.15)
STREAMER_PROFILES_PATH = os.getenv("STREAMER_PROFILES_PATH", str(BASE_DIR / "streamer_profiles.json"))
FACE_REFERENCE_DIR = os.getenv("FACE_REFERENCE_DIR", str(BASE_DIR / "face_references"))

# --- Tier 4: dashboard auth / alerts ---
# Bind address for the dashboard server (main.py). 127.0.0.1 = this
# computer only; 0.0.0.0 = reachable on the local network (e.g. from an
# Android phone on the same Wi-Fi — set a DASHBOARD_PASSWORD first).
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = _env_int("DASHBOARD_PORT", 8500)
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")
DASHBOARD_PASSWORD_HASH = os.getenv("DASHBOARD_PASSWORD_HASH", "")
DASHBOARD_SECRET_KEY = os.getenv("DASHBOARD_SECRET_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# --- Tier 5: rendering / LLM providers ---
FACE_TRACKER_BACKEND = os.getenv("FACE_TRACKER_BACKEND", "auto")
YOLO_FACE_MODEL = os.getenv("YOLO_FACE_MODEL", "yolov8n-face.pt")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
BGM_DIR = os.getenv("BGM_DIR", str(BASE_DIR / "assets" / "bgm"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "local-model")
VIDEO_ENCODER = os.getenv("VIDEO_ENCODER", "auto")
SUBTITLE_STYLE = os.getenv("SUBTITLE_STYLE", "classic")
BROLL_ENABLED = _env_bool("BROLL_ENABLED", False)
BGM_ENABLED = _env_bool("BGM_ENABLED", False)
