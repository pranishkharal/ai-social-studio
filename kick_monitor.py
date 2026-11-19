import os
import sys
import time
import json
import ssl
import subprocess
import threading
import urllib.request
from pathlib import Path
from datetime import datetime

from config import (
    BASE_DIR, BUFFER_DIR, CLIPS_DIR, CHECK_INTERVAL_SECONDS, 
    STREAM_BUFFER_MINUTES, STREAMERS_TO_MONITOR
)
from database import update_streamer_status, log_event, add_notification

class KickMonitor:
    def __init__(self):
        # Verified TLS: kick.com serves a valid public certificate, so there
        # is no reason to disable verification (previously CERT_NONE, which
        # left API calls open to interception). See Tier 1 feature 15 notes.
        self.ssl_ctx = ssl.create_default_context()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        self.active_recorders = {}  # {username: subprocess.Popen}
        self.streamer_cache = {}
        self.known_live = set()
        self.running = False

    def fetch_channel_info(self, username: str) -> dict:
        """Fetches channel status, metadata, and AWS IVS playback URL directly with 0% CPU strain."""
        url = f"https://kick.com/api/v2/channels/{username.strip().lower()}"
        try:
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, context=self.ssl_ctx, timeout=6) as resp:
                data = json.loads(resp.read().decode())
                
                is_live = data.get("livestream") is not None
                livestream = data.get("livestream") or {}
                
                result = {
                    "username": username.lower(),
                    "is_live": 1 if is_live else 0,
                    "channel_id": data.get("id"),
                    "chatroom_id": data.get("chatroom", {}).get("id"),
                    "title": livestream.get("session_title", data.get("user", {}).get("bio", "")),
                    "category": livestream.get("categories", [{}])[0].get("name") if livestream.get("categories") else "Gaming",
                    "viewer_count": livestream.get("viewer_count", 0),
                    "playback_url": data.get("playback_url"),
                    "last_checked": datetime.now().isoformat()
                }
                update_streamer_status(result)
                self.streamer_cache[username.lower()] = result
                return result
        except Exception as e:
            # Handle rate-limit or offline errors gracefully
            log_event("WARNING", "KickMonitor", f"Failed fetching {username}: {e}")
            return {
                "username": username.lower(),
                "is_live": 0,
                "channel_id": None,
                "chatroom_id": None,
                "title": "",
                "category": "",
                "viewer_count": 0,
                "playback_url": None,
                "last_checked": datetime.now().isoformat()
            }

    def start_rolling_buffer(self, username: str, playback_url: str):
        """
        Starts a zero-reencoding stream-copy recorder.
        Writes rolling 60-second chunks keeping only the last 10 minutes on disk.
        Takes ~0.1% CPU.
        """
        if username in self.active_recorders and self.active_recorders[username].poll() is None:
            return  # Already recording
            
        streamer_buf_dir = BUFFER_DIR / username
        streamer_buf_dir.mkdir(parents=True, exist_ok=True)
        
        segment_pattern = str(streamer_buf_dir / "chunk_%03d.ts")
        
        # FFmpeg stream-copy command with rolling segment wrap
        # -segment_wrap 10 keeps exactly 10 rolling segments (10 minutes)
        cmd = [
            "ffmpeg", "-y",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", playback_url,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", "60",
            "-segment_wrap", str(STREAM_BUFFER_MINUTES),
            "-reset_timestamps", "1",
            segment_pattern
        ]
        
        try:
            log_event("INFO", "KickMonitor", f"Starting rolling buffer for {username}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            self.active_recorders[username] = proc
        except Exception as e:
            log_event("ERROR", "KickMonitor", f"Error launching recorder for {username}: {e}")

    def stop_recorder(self, username: str):
        if username in self.active_recorders:
            proc = self.active_recorders.pop(username)
            if proc.poll() is None:
                proc.terminate()
                log_event("INFO", "KickMonitor", f"Stopped recorder for {username}")

    def list_buffer_chunks(self, username: str):
        """
        Returns the rolling-buffer chunk files for a streamer, newest first
        (sorted by modification time, descending). Used by the auto-clip
        watcher to sample the latest audio. Returns [] when there is no
        buffer directory for the streamer.
        """
        streamer_buf_dir = BUFFER_DIR / username
        if not streamer_buf_dir.exists():
            return []
        return sorted(
            streamer_buf_dir.glob("chunk_*.ts"),
            key=os.path.getmtime,
            reverse=True,
        )

    def handle_stream_started(self, username: str):
        """
        Webhook hook (kick_webhooks.py): a livestream.status.updated event
        said this streamer went live. Re-fetch channel info (to confirm and
        get a fresh playback URL) and start the rolling buffer. Idempotent:
        safe to call when the stream is already being recorded, and a no-op
        when the API disagrees that the stream is live. Returns True when
        recording is (now) active.
        """
        username = username.strip().lower()
        info = self.fetch_channel_info(username)
        if not info.get("is_live"):
            log_event("INFO", "KickMonitor",
                      f"Webhook said {username} is live but the API disagrees; ignoring")
            return False
        if username not in self.known_live:
            self.known_live.add(username)
            add_notification(
                title=f"🔴 {username} is now LIVE on Kick!",
                message=f"Broadcasting '{info.get('title', '')}' in {info.get('category', 'Gaming')} with {info.get('viewer_count', 0):,} viewers.",
                notif_type="channel_live",
                streamer=username
            )
            # External alert (Tier 4): Discord/Telegram, best-effort —
            # an alert failure must never break live detection.
            try:
                from alerts import alert_streamer_live
                alert_streamer_live(username, info.get("title", ""),
                                    info.get("viewer_count", 0))
            except Exception:
                pass
        self.start_rolling_buffer(username, info.get("playback_url"))
        return True

    def handle_stream_ended(self, username: str):
        """
        Webhook hook (kick_webhooks.py): a livestream.status.updated event
        said this streamer went offline. Stops the rolling buffer and clears
        live state. Idempotent. Returns True.
        """
        username = username.strip().lower()
        if username in self.known_live:
            self.known_live.remove(username)
        self.stop_recorder(username)
        log_event("INFO", "KickMonitor",
                  f"Webhook: {username} went offline, recorder stopped")
        return True

    def extract_recent_clip(self, username: str, duration_seconds: int = 45) -> str:
        """
        Extracts the most recent footage directly from live stream or rolling buffer.
        Returns the path to the saved raw MP4 clip.
        """
        CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_clip_path = str(CLIPS_DIR / f"{username}_{timestamp}_raw.mp4")
        
        # If live stream playback URL is known, we can fetch the latest slice directly via FFmpeg
        info = self.streamer_cache.get(username) or self.fetch_channel_info(username)
        playback_url = info.get("playback_url")
        
        if playback_url:
            cmd = [
                "ffmpeg", "-y",
                "-i", playback_url,
                "-t", str(duration_seconds),
                "-c", "copy",
                output_clip_path
            ]
            try:
                subprocess.run(
                    cmd, 
                    check=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                    timeout=duration_seconds + 15
                )
                if os.path.exists(output_clip_path) and os.path.getsize(output_clip_path) > 10000:
                    log_event("INFO", "KickMonitor", f"Successfully extracted {duration_seconds}s clip for {username}")
                    return output_clip_path
            except Exception as e:
                log_event("ERROR", "KickMonitor", f"Direct clip extraction failed for {username}: {e}")

        # Fallback to existing buffer chunks if available
        streamer_buf_dir = BUFFER_DIR / username
        chunks = sorted(streamer_buf_dir.glob("chunk_*.ts"), key=os.path.getmtime, reverse=True)
        if chunks:
            latest_chunk = str(chunks[0])
            fallback_cmd = [
                "ffmpeg", "-y",
                "-i", latest_chunk,
                "-t", str(duration_seconds),
                "-c", "copy",
                output_clip_path
            ]
            subprocess.run(fallback_cmd, capture_output=True)
            if os.path.exists(output_clip_path):
                return output_clip_path

        return None

    def monitor_loop(self):
        """Continuous background monitoring thread."""
        self.running = True
        log_event("INFO", "KickMonitor", "Monitoring thread started.")
        while self.running:
            for username in list(STREAMERS_TO_MONITOR):
                info = self.fetch_channel_info(username)
                is_live = bool(info.get("is_live"))
                if is_live:
                    if username not in self.known_live:
                        self.known_live.add(username)
                        add_notification(
                            title=f"🔴 {username} is now LIVE on Kick!",
                            message=f"Broadcasting '{info.get('title', '')}' in {info.get('category', 'Gaming')} with {info.get('viewer_count', 0):,} viewers.",
                            notif_type="channel_live",
                            streamer=username
                        )
                        # External alert (Tier 4): Discord/Telegram,
                        # best-effort — never breaks live detection.
                        try:
                            from alerts import alert_streamer_live
                            alert_streamer_live(username, info.get("title", ""),
                                                info.get("viewer_count", 0))
                        except Exception:
                            pass
                    # Streamer is live! Ensure rolling buffer is active
                    self.start_rolling_buffer(username, info.get("playback_url"))
                else:
                    if username in self.known_live:
                        self.known_live.remove(username)
                    self.stop_recorder(username)
                time.sleep(2)
            time.sleep(CHECK_INTERVAL_SECONDS)

    def start_background(self):
        t = threading.Thread(target=self.monitor_loop, daemon=True)
        t.start()
        return t
