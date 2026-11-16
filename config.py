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
