"""
Auto-clip watcher: clip hype moments without a human pressing the button.

Tier 1, feature 1 (see ai-social-studio-feature-suggestions.md). This is the
piece that turns the tool from a "clip button" into a live clipping engine.

How it works:
* A background thread wakes every ``AUTO_CLIP_CHECK_SECONDS`` (default 5 s)
  and, for each streamer with an active rolling buffer, samples the newest
  buffer audio: it cuts a short slice (default 5 s) off the end of the
  latest chunk with ffmpeg and runs ``HypeDetector.analyze_audio_energy``
  on it (fast volumedetect, no re-encode).
* When chat hype is available (``CHAT_HYPE_ENABLED=true`` and
  ``websocket-client`` installed), the chat score for that streamer is
  blended in: combined = 0.6 * audio + 0.4 * chat. Without chat, the audio
  score alone decides. The chat score includes the distinct-chatter
  spike bonus (many DIFFERENT people chatting at once -- see
  chat_hype.py), so a real crowd reaction can trigger a clip even when
  the audio is quiet, while bots and one-person floods cannot.
* When the combined score reaches ``AUTO_CLIP_MIN_SCORE`` (default 70) and
  the cooldown/duplicate guard (``clip_dedup.ClipCooldown``) allows it, the
  watcher fires ``pipeline.process_moment(streamer, duration)`` in a worker
  thread -- never on the watcher thread, so one slow clip can never stall
  sampling. At most one clip per streamer runs at a time.

The watcher is OFF by default (``AUTO_CLIP_ENABLED`` defaults to "false"),
so existing behaviour is unchanged unless it is explicitly turned on. The
chat client is imported lazily and only when chat hype is enabled, so a
missing ``websocket-client`` package can never break the audio-only path.

Environment variables (read at construction time; constructor arguments
override them, which is how tests configure the watcher):
    AUTO_CLIP_ENABLED         default "false"
    AUTO_CLIP_MIN_SCORE       default 70
    AUTO_CLIP_CHECK_SECONDS   default 5
    AUTO_CLIP_DURATION        default 25 (seconds of footage per auto-clip)
    AUTO_CLIP_SAMPLE_SECONDS  default 5 (audio slice cut for each sample)
    CHAT_HYPE_ENABLED         default "false" (see chat_hype.py)
"""

import os
import subprocess
import sys
import tempfile
import threading
import time

from clip_dedup import ClipCooldown
from database import log_event
from hype_detector import HypeDetector


def _env_bool(name, default=False):
    return os.getenv(name, "true" if default else "false").strip().lower() in (
        "1", "true", "yes", "on",
    )


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


class AutoClipWatcher:
    """Background thread that auto-clips hype moments from live buffers.

    *monitor* is a KickMonitor (owns the rolling buffers), *pipeline* is a
    ClipPipeline (``process_moment`` does the real work). *detector*,
    *cooldown* and *chat_client* are injectable for tests; when omitted they
    are built from the real classes / env flags.
    """

    def __init__(self, monitor, pipeline, detector=None, cooldown=None,
                 chat_client=None, enabled=None, min_score=None,
                 check_seconds=None, duration=None, sample_seconds=None):
        self.monitor = monitor
        self.pipeline = pipeline
        self.detector = detector if detector is not None else HypeDetector()
        self.cooldown = cooldown if cooldown is not None else ClipCooldown()
        self.chat_client = chat_client  # None: created lazily if chat enabled
        self.enabled = (
            _env_bool("AUTO_CLIP_ENABLED", False)
            if enabled is None else bool(enabled)
        )
        self.min_score = (
            _env_float("AUTO_CLIP_MIN_SCORE", 70.0)
            if min_score is None else float(min_score)
        )
        self.check_seconds = (
            _env_int("AUTO_CLIP_CHECK_SECONDS", 5)
            if check_seconds is None else int(check_seconds)
        )
        self.duration = (
            _env_int("AUTO_CLIP_DURATION", 25)
            if duration is None else int(duration)
        )
        self.sample_seconds = (
            _env_int("AUTO_CLIP_SAMPLE_SECONDS", 5)
            if sample_seconds is None else int(sample_seconds)
        )
        self._stop_event = threading.Event()
        self._thread = None
        self._running = False
        self._in_progress = set()  # streamers with a clip worker running
        self._in_progress_lock = threading.Lock()
        self._chat_client_attempted = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start the watcher thread. Returns False when auto-clip is
        disabled (the default) -- nothing starts, nothing changes."""
        if not self.enabled:
            log_event("INFO", "AutoClipWatcher",
                      "Auto-clip disabled (AUTO_CLIP_ENABLED=false); "
                      "watcher not started")
            return False
        if self._running:
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="AutoClipWatcher")
        self._thread.start()
        self._running = True
        log_event("INFO", "AutoClipWatcher",
                  f"Auto-clip watcher started (min_score={self.min_score}, "
                  f"every {self.check_seconds}s, duration={self.duration}s)")
        return True

    def stop(self):
        """Stop the watcher thread (in-flight clip workers finish on their
        own; they are daemon threads and never block shutdown for long)."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.check_seconds + 5)
        self._running = False
        log_event("INFO", "AutoClipWatcher", "Auto-clip watcher stopped")

    @property
    def running(self):
        return (self._running and self._thread is not None
                and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _watch_loop(self):
        while not self._stop_event.is_set():
            try:
                self._ensure_chat_client()
                for username in self._live_buffered_streamers():
                    if self._stop_event.is_set():
                        break
                    try:
                        self.check_streamer(username)
                    except Exception as e:
                        log_event("WARNING", "AutoClipWatcher",
                                  f"Check failed for {username}: {e}")
            except Exception as e:
                # Never let one bad cycle kill the watcher.
                log_event("ERROR", "AutoClipWatcher",
                          f"Watcher loop error: {e}")
            self._stop_event.wait(self.check_seconds)

    def _live_buffered_streamers(self):
        """Streamers whose rolling-buffer recorder process is alive."""
        result = []
        recorders = getattr(self.monitor, "active_recorders", {}) or {}
        for username, proc in list(recorders.items()):
            try:
                if proc is None or proc.poll() is None:
                    result.append(username)
            except Exception:
                result.append(username)
        return result

    def _ensure_chat_client(self):
        """Lazily build the chat client (once) when chat hype is enabled."""
        if self.chat_client is not None or self._chat_client_attempted:
            return
        self._chat_client_attempted = True
        if not _env_bool("CHAT_HYPE_ENABLED", False):
            return
        try:
            # Lazy import: chat_hype is optional, and importing it must
            # never break the audio-only path.
            from chat_hype import create_chat_client_from_env
            client = create_chat_client_from_env()
            if client is not None and client.start():
                self.chat_client = client
                log_event("INFO", "AutoClipWatcher",
                          "Chat hype client attached to watcher")
        except Exception as e:
            log_event("WARNING", "AutoClipWatcher",
                      f"Could not start chat hype client: {e}")

    # ------------------------------------------------------------------
    # Sampling + scoring
    # ------------------------------------------------------------------
    def check_streamer(self, username):
        """Sample one streamer and fire a clip if the score warrants it.

        Returns the sample dict ``{streamer, score, audio_score,
        chat_score, source}``. Safe to call directly (tests / debugging).
        """
        audio_score = self._sample_audio_score(username)
        chat_score = self._chat_score_for(username)
        if chat_score is not None:
            combined = round(audio_score * 0.6 + chat_score * 0.4, 1)
            source = "audio+chat"
        else:
            combined = audio_score
            source = "audio"
        sample = {
            "streamer": username,
            "score": combined,
            "audio_score": audio_score,
            "chat_score": chat_score,
            "source": source,
        }
        if combined >= self.min_score:
            self._maybe_clip(username, combined, sample)
        return sample

    def _sample_audio_score(self, username):
        """HypeDetector score for the newest audio in a streamer's buffer."""
        try:
            chunks = self.monitor.list_buffer_chunks(username)
        except Exception as e:
            log_event("WARNING", "AutoClipWatcher",
                      f"Could not list buffer chunks for {username}: {e}")
            return 0.0
        if not chunks:
            return 0.0
        newest = chunks[0]
        sample_path = self._cut_sample_slice(str(newest))
        target = sample_path or str(newest)
        try:
            metrics = self.detector.analyze_audio_energy(target)
            return float(metrics.get("score", 0.0))
        except Exception as e:
            log_event("WARNING", "AutoClipWatcher",
                      f"Audio sampling failed for {username}: {e}")
            return 0.0
        finally:
            if sample_path:
                try:
                    os.remove(sample_path)
                except OSError:
                    pass

    def _cut_sample_slice(self, chunk_path):
        """Cut the last ``sample_seconds`` of a chunk to a temp file.

        Returns the temp path, or None when ffmpeg fails (the caller then
        analyses the whole chunk instead). The temp file lives in the OS
        temp dir and is always removed by the caller.
        """
        tmp = tempfile.NamedTemporaryFile(
            suffix=".ts", prefix="autoclip_sample_", delete=False)
        tmp_path = tmp.name
        tmp.close()
        cmd = [
            "ffmpeg", "-y",
            "-sseof", f"-{self.sample_seconds}",
            "-i", chunk_path,
            "-c", "copy",
            tmp_path,
        ]
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                timeout=15,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32"
                    else 0
                ),
            )
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                return tmp_path
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None

    def _chat_score_for(self, username):
        """Chat hype score for a streamer, or None when chat is unavailable
        (no client, no chatroom id, or a lookup error)."""
        if self.chat_client is None:
            return None
        try:
            cache = getattr(self.monitor, "streamer_cache", {}) or {}
            info = cache.get(username) or {}
            chatroom_id = info.get("chatroom_id")
            if chatroom_id is None:
                return None
            # Hand the Kick channel id to the chat client: with it, the
            # client negotiates the current Pusher app key + cluster
            # from Kick on every (re)connect instead of trusting the
            # hard-coded key (which Kick has rotated before).
            channel_id = info.get("channel_id")
            if channel_id is not None:
                setter = getattr(self.chat_client, "set_channel_id", None)
                if callable(setter):
                    setter(channel_id)
            try:
                self.chat_client.subscribe(chatroom_id)
            except Exception:
                pass
            return float(self.chat_client.get_hype_score(chatroom_id))
        except Exception as e:
            log_event("WARNING", "AutoClipWatcher",
                      f"Chat score lookup failed for {username}: {e}")
            return None

    # ------------------------------------------------------------------
    # Clipping (worker thread, never blocks the watcher)
    # ------------------------------------------------------------------
    def _maybe_clip(self, username, score, sample):
        """Fire a clip worker for *username* if no clip is already running
        for them and the cooldown allows it. Returns True when fired."""
        with self._in_progress_lock:
            if username in self._in_progress:
                return False
        allowed, reason = self.cooldown.can_clip(username)
        if not allowed:
            log_event("INFO", "AutoClipWatcher",
                      f"Skipping auto-clip for {username} "
                      f"(score {score}): {reason}")
            return False
        with self._in_progress_lock:
            if username in self._in_progress:
                return False
            self._in_progress.add(username)
        log_event("INFO", "AutoClipWatcher",
                  f"Hype {score} for {username} ({sample['source']}); "
                  f"firing auto-clip ({self.duration}s)")
        worker = threading.Thread(
            target=self._run_clip, args=(username,),
            daemon=True, name=f"AutoClip-{username}")
        worker.start()
        return True

    def _run_clip(self, username):
        try:
            result = self.pipeline.process_moment(
                username, duration=self.duration)
            status = (result or {}).get("status")
            if status in ("ready_to_upload", "success"):
                # Only a finished clip starts the cooldown; a failed
                # attempt leaves the door open for the next real moment.
                self.cooldown.record_clip(username)
                log_event("INFO", "AutoClipWatcher",
                          f"Auto-clip done for {username}: "
                          f"clip #{(result or {}).get('clip_id')} ({status})")
            else:
                log_event("WARNING", "AutoClipWatcher",
                          f"Auto-clip for {username} ended with "
                          f"status={status}: {(result or {}).get('error', '')}")
        except Exception as e:
            log_event("ERROR", "AutoClipWatcher",
                      f"Auto-clip worker failed for {username}: {e}")
        finally:
            with self._in_progress_lock:
                self._in_progress.discard(username)
