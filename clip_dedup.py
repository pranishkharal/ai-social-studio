"""
Clip cooldown + duplicate protection for the auto-clip watcher.

Tier 1, feature 4 (see ai-social-studio-feature-suggestions.md).

Two guards, both consulted by AutoClipWatcher before it fires a clip:

1. Cooldown -- at least ``AUTO_CLIP_COOLDOWN_SECONDS`` (default 90) between
   auto-clips for the same streamer. Enforced with an in-memory timestamp
   (fast path, survives nothing) *plus* a read-only check of the most recent
   non-failed clip for that streamer in the ``clips`` table (survives
   restarts and also blocks while a manual clip is still processing).

2. Duplicate detection -- ``is_duplicate(streamer, transcript)`` compares a
   transcript against recent clips for the same streamer (last 30 minutes by
   default) using Jaccard word-overlap similarity. NOTE: the ``clips`` table
   has no transcript column, so recent clips are represented by their
   title + story_summary + caption text. The watcher cannot use this check
   pre-clip (no transcript exists yet); it is for the post-transcription
   stage of the pipeline / review tooling.

Everything here is read-only with respect to the database (SELECT only) and
fails open: if the DB is unavailable, the in-memory cooldown still applies
and the duplicate check reports "not a duplicate" rather than crashing the
watcher.

Environment variables (read at construction time, so tests can pass
explicit arguments instead):
    AUTO_CLIP_COOLDOWN_SECONDS      default 90
    AUTO_CLIP_DUP_SIMILARITY        default 0.7  (Jaccard threshold)
    AUTO_CLIP_DUP_WINDOW_MINUTES    default 30
"""

import os
import re
import threading
import time
from datetime import datetime, timezone

from database import get_connection, log_event


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


_WORD_RE = re.compile(r"[a-z0-9']+")


def _word_set(text):
    """Lowercased set of alphanumeric words in *text*."""
    if not text:
        return set()
    return set(_WORD_RE.findall(text.lower()))


def jaccard_similarity(text_a, text_b):
    """Jaccard word-overlap similarity between two texts, 0.0 - 1.0."""
    words_a = _word_set(text_a)
    words_b = _word_set(text_b)
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _parse_db_timestamp(value):
    """Parse a SQLite CURRENT_TIMESTAMP value ("YYYY-MM-DD HH:MM:SS", UTC).

    Tolerates ISO-8601 variants ("T" separator, trailing "Z", fractional
    seconds). Returns an aware UTC datetime, or None if unparseable.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


class ClipCooldown:
    """Per-streamer auto-clip cooldown + recent-clip duplicate detection."""

    def __init__(self, cooldown_seconds=None, similarity_threshold=None,
                 recent_window_minutes=None):
        self.cooldown_seconds = (
            _env_int("AUTO_CLIP_COOLDOWN_SECONDS", 90)
            if cooldown_seconds is None else int(cooldown_seconds)
        )
        self.similarity_threshold = (
            _env_float("AUTO_CLIP_DUP_SIMILARITY", 0.7)
            if similarity_threshold is None else float(similarity_threshold)
        )
        self.recent_window_minutes = (
            _env_int("AUTO_CLIP_DUP_WINDOW_MINUTES", 30)
            if recent_window_minutes is None else int(recent_window_minutes)
        )
        self._last_clip_at = {}  # {streamer: time.monotonic() of last auto-clip}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------
    def can_clip(self, streamer):
        """Return (allowed: bool, reason: str) for an auto-clip now.

        Checks the in-memory timestamp first, then the most recent
        non-failed clip for this streamer in the database.
        """
        streamer = (streamer or "").strip().lower()
        if not streamer:
            return False, "empty streamer name"

        now_mono = time.monotonic()
        with self._lock:
            last = self._last_clip_at.get(streamer)
        if last is not None:
            elapsed = now_mono - last
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                return False, f"cooldown: {remaining:.0f}s remaining"

        # DB check: a recent clip (manual or auto, still processing or done)
        # also blocks a new auto-clip. Failed clips do not block.
        try:
            conn = get_connection()
            try:
                row = conn.execute(
                    "SELECT created_at FROM clips "
                    "WHERE streamer = ? AND status != 'failed' "
                    "ORDER BY created_at DESC LIMIT 1",
                    (streamer,),
                ).fetchone()
            finally:
                conn.close()
        except Exception as e:  # DB unavailable: fail open, in-memory check stands
            log_event("WARNING", "ClipCooldown",
                      f"DB cooldown check failed for {streamer}: {e}")
            return True, "ok (db unavailable)"

        if row is not None:
            created_at = _parse_db_timestamp(row["created_at"])
            if created_at is not None:
                age = (datetime.now(timezone.utc) - created_at).total_seconds()
                if age < self.cooldown_seconds:
                    remaining = self.cooldown_seconds - age
                    return False, (
                        f"cooldown: clip in DB {age:.0f}s ago "
                        f"({remaining:.0f}s remaining)"
                    )
        return True, "ok"

    def record_clip(self, streamer):
        """Mark that an auto-clip just fired for *streamer* (starts cooldown)."""
        streamer = (streamer or "").strip().lower()
        if not streamer:
            return
        with self._lock:
            self._last_clip_at[streamer] = time.monotonic()

    def seconds_until_ready(self, streamer):
        """Seconds until *streamer* is off in-memory cooldown (0 if ready)."""
        streamer = (streamer or "").strip().lower()
        with self._lock:
            last = self._last_clip_at.get(streamer)
        if last is None:
            return 0.0
        return max(0.0, self.cooldown_seconds - (time.monotonic() - last))

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------
    def is_duplicate(self, streamer, transcript):
        """Return (is_duplicate: bool, reason: str).

        Compares *transcript* against recent clips for *streamer* (last
        ``recent_window_minutes``) using Jaccard word overlap. A transcript
        with fewer than 5 distinct words is too short to judge and is never
        flagged as a duplicate.
        """
        streamer = (streamer or "").strip().lower()
        words = _word_set(transcript)
        if len(words) < 5:
            return False, "transcript too short to compare"
        if not streamer:
            return False, "empty streamer name"

        try:
            conn = get_connection()
            try:
                rows = conn.execute(
                    "SELECT title, story_summary, caption FROM clips "
                    "WHERE streamer = ? "
                    "AND created_at >= datetime('now', ?) "
                    "ORDER BY created_at DESC",
                    (streamer, f"-{self.recent_window_minutes} minutes"),
                ).fetchall()
            finally:
                conn.close()
        except Exception as e:  # DB unavailable: fail open
            log_event("WARNING", "ClipCooldown",
                      f"DB duplicate check failed for {streamer}: {e}")
            return False, "db unavailable"

        best = 0.0
        for row in rows:
            # The clips table stores no transcript; compare against the text
            # the pipeline wrote for the clip (title + summary + caption).
            clip_text = " ".join(
                part for part in
                (row["title"], row["story_summary"], row["caption"])
                if part
            )
            best = max(best, jaccard_similarity(transcript, clip_text))

        if best >= self.similarity_threshold:
            return True, (
                f"similarity {best:.2f} >= threshold "
                f"{self.similarity_threshold:.2f}"
            )
        return False, f"max similarity {best:.2f}"
