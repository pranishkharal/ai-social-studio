"""
Live chat hype detection for Kick, via Kick's Pusher websocket.

Tier 1, feature 2 (see ai-social-studio-feature-suggestions.md). This finally
uses ``CHAT_VELOCITY_SPIKE_MULTIPLIER`` from config.py, which was defined but
never referenced anywhere.

Kick chat setup, verified 2026-11-19 against kick-js (retconned/kick-js),
kickforge, kick-chat-wrapper and the kick-api Rust crate:

* Kick chat runs on Pusher. Endpoint (no auth needed to *read* chat)::

      wss://ws-us2.pusher.com/app/<APP_KEY>?protocol=7&client=js&version=8.4.0&flash=false

  Current app key ``32cbd69e4b950bf97679``. Kick has rotated this key before
  (an older key, ``eb1d5f283081a78b932c``, appears in older libraries), so it
  is overridable via ``KICK_PUSHER_APP_KEY``. If chat stops flowing, check the
  network tab on kick.com for the current ``pusher`` URL.

* One websocket connection carries many chatrooms. Subscribe per chatroom
  with a ``pusher:subscribe`` event for channel ``chatrooms.{id}.v2``.
  The chatroom id comes from ``kick_monitor.fetch_channel_info`` (already
  fetched and stored in the ``streamers`` table / ``streamer_cache``).

* Chat messages arrive as Pusher envelopes::

      {"event": "App\\Events\\ChatMessageEvent",
       "data": "{\\"id\\": ..., \\"chatroom_id\\": ..., \\"content\\": ...,
                \\"sender\\": {\\"username\\": ...}}",
       "channel": "chatrooms.{id}.v2"}

  Note ``data`` is a JSON *string*, not an object. Emotes appear inline in
  ``content`` as ``[emote:ID:NAME]`` (e.g. ``[emote:123:KEKW]``) as well as
  plain typed words (``KEKW``). Both forms are counted.

What is tracked per chatroom (rolling window, default 10 s):
* messages/second vs. a slow-moving baseline (spike detection),
* emote storms (configurable emote list),
* keyword bursts ("CLIP IT", "WTF", "LOL", ...).

``get_hype_score(chatroom_id)`` combines those into a 0-100 score that the
auto-clip watcher blends with the audio score.

Dependency: ``websocket-client`` (``pip install websocket-client``). It is an
OPTIONAL import -- if it is missing, the client logs a clear error and
disables itself instead of crashing the app.

Environment variables:
    CHAT_HYPE_ENABLED           default "false" -- chat hype is off unless set
    KICK_PUSHER_APP_KEY         default "32cbd69e4b950bf97679" (verified key)
    CHAT_HYPE_EMOTES            comma list, default "KEKW,Pog,OMEGALUL,LUL,Kappa,PepeLaugh,5Head,Clueless,POGGERS,EZ"
    CHAT_HYPE_KEYWORDS          comma list, default "CLIP IT,WTF,LOL,CLIP,INSANE,NO WAY"
    CHAT_HYPE_WINDOW_SECONDS    default 10
    CHAT_HYPE_THRESHOLD         default 70 (score at/above which on_hype fires)
    CHAT_HYPE_CALLBACK_COOLDOWN default 30 (seconds between on_hype calls per chatroom)
"""

import json
import os
import re
import threading
import time
from collections import deque

from config import CHAT_VELOCITY_SPIKE_MULTIPLIER
from database import log_event

try:
    import websocket  # websocket-client (optional dependency)
    _WEBSOCKET_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by tests via monkeypatch
    websocket = None
    _WEBSOCKET_AVAILABLE = False

# Verified current Pusher app key for kick.com chat (see module docstring).
DEFAULT_PUSHER_APP_KEY = "32cbd69e4b950bf97679"
PUSHER_CLUSTER_HOST = "ws-us2.pusher.com"
PUSHER_CLIENT_QUERY = "protocol=7&client=js&version=8.4.0&flash=false"

DEFAULT_EMOTES = [
    "KEKW", "Pog", "OMEGALUL", "LUL", "Kappa",
    "PepeLaugh", "5Head", "Clueless", "POGGERS", "EZ",
]
DEFAULT_KEYWORDS = ["CLIP IT", "WTF", "LOL", "CLIP", "INSANE", "NO WAY"]

# Pusher event names that carry a chat message. Kick sends the Laravel-style
# class name; accept the short form too for robustness.
CHAT_MESSAGE_EVENTS = {"App\\Events\\ChatMessageEvent", "ChatMessage"}

_EMOTE_TAG_RE = re.compile(r"\[emote:\d+:([^\]]+)\]")
_CHANNEL_RE = re.compile(r"chatrooms\.(\d+)\.v2")


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


def _env_list(name, default):
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def get_pusher_url(app_key=None):
    """Full Pusher websocket URL for Kick chat."""
    key = app_key or os.getenv("KICK_PUSHER_APP_KEY", DEFAULT_PUSHER_APP_KEY)
    return (
        f"wss://{PUSHER_CLUSTER_HOST}/app/{key}?{PUSHER_CLIENT_QUERY}"
    )


def chatroom_channel(chatroom_id):
    """Pusher channel name for a Kick chatroom."""
    return f"chatrooms.{int(chatroom_id)}.v2"


def extract_emote_names(content):
    """Emote names in a chat message: ``[emote:ID:NAME]`` tags + plain words.

    Returns a set of uppercased emote names found in *content*.
    """
    found = {name.upper() for name in _EMOTE_TAG_RE.findall(content or "")}
    for word in re.findall(r"[A-Za-z0-9]+", content or ""):
        found.add(word.upper())
    return found


class ChatHypeTracker:
    """Rolling-window hype statistics for one chatroom."""

    def __init__(self, chatroom_id, window_seconds=None, emotes=None,
                 keywords=None):
        self.chatroom_id = int(chatroom_id)
        self.window_seconds = (
            _env_int("CHAT_HYPE_WINDOW_SECONDS", 10)
            if window_seconds is None else int(window_seconds)
        )
        self.emotes = {e.upper() for e in
                       (_env_list("CHAT_HYPE_EMOTES", DEFAULT_EMOTES)
                        if emotes is None else emotes)}
        self.keywords = [k.upper() for k in
                         (_env_list("CHAT_HYPE_KEYWORDS", DEFAULT_KEYWORDS)
                          if keywords is None else keywords)]
        self._messages = deque()  # (monotonic_ts, had_emote, keyword_hits)
        self._lock = threading.Lock()
        self._baseline_mps = 0.0  # slow EMA of messages/second
        self.total_messages = 0

    # -- ingestion ------------------------------------------------------
    def record_message(self, content, timestamp=None):
        """Record one chat message. *timestamp* is time.monotonic() (tests)."""
        now = time.monotonic() if timestamp is None else timestamp
        upper = (content or "").upper()
        emote_names = extract_emote_names(content)
        had_emote = bool(emote_names & self.emotes)
        keyword_hits = sum(1 for kw in self.keywords if kw in upper)
        with self._lock:
            self._messages.append((now, had_emote, keyword_hits))
            self.total_messages += 1
            self._prune_locked(now)
            # Slow baseline update, but never let a spike drag the baseline
            # up with it (otherwise every spike would normalise itself away).
            rate = len(self._messages) / float(self.window_seconds)
            if self._baseline_mps == 0.0 or rate < self._baseline_mps * 2.0:
                alpha = 0.02
                self._baseline_mps = (
                    (1.0 - alpha) * self._baseline_mps + alpha * rate
                )

    def _prune_locked(self, now):
        cutoff = now - self.window_seconds
        while self._messages and self._messages[0][0] < cutoff:
            self._messages.popleft()

    # -- scoring --------------------------------------------------------
    def get_stats(self):
        """Current window statistics (also prunes stale messages)."""
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            count = len(self._messages)
            emote_count = sum(1 for _, e, _ in self._messages if e)
            keyword_total = sum(k for _, _, k in self._messages)
            baseline = self._baseline_mps
        mps = count / float(self.window_seconds) if self.window_seconds else 0.0
        return {
            "chatroom_id": self.chatroom_id,
            "messages_per_second": round(mps, 2),
            "baseline_mps": round(baseline, 2),
            "window_messages": count,
            "emote_messages": emote_count,
            "emote_share": round(emote_count / count, 2) if count else 0.0,
            "keyword_hits": keyword_total,
            "total_messages": self.total_messages,
        }

    def get_hype_score(self):
        """Hype score 0-100 for this chatroom right now.

        * Velocity (up to 60): messages/second vs. baseline. A ratio of 1x
          scores 0; a ratio at/above CHAT_VELOCITY_SPIKE_MULTIPLIER (3.0)
          scores the full 60.
        * Emote storm (up to 25): share of window messages carrying a
          tracked emote; 50%+ emote share scores the full 25.
        * Keyword bursts (up to 15): +3 per keyword hit in the window.
        """
        stats = self.get_stats()
        count = stats["window_messages"]
        if count == 0:
            return 0.0
        baseline = max(stats["baseline_mps"], 0.2)  # floor: quiet chats
        ratio = stats["messages_per_second"] / baseline
        multiplier = max(CHAT_VELOCITY_SPIKE_MULTIPLIER, 1.01)
        velocity = min(60.0, max(0.0, (ratio - 1.0) / (multiplier - 1.0)) * 60.0)
        emote = min(25.0, stats["emote_share"] * 50.0)
        keywords = min(15.0, stats["keyword_hits"] * 3.0)
        return round(min(100.0, velocity + emote + keywords), 1)


class KickChatClient:
    """One Pusher websocket connection subscribed to many Kick chatrooms.

    ``on_hype`` (optional) is called as ``on_hype(chatroom_id, score, stats)``
    when a chatroom's hype score crosses the threshold (at most once per
    ``CHAT_HYPE_CALLBACK_COOLDOWN`` seconds per chatroom). ``on_message``
    (optional) is called as ``on_message(chatroom_id, message_dict)`` for
    every chat message received.
    """

    def __init__(self, on_hype=None, on_message=None, emotes=None,
                 keywords=None, window_seconds=None, pusher_app_key=None,
                 hype_threshold=None, callback_cooldown=None,
                 sleep_fn=None):
        self.on_hype = on_hype
        self.on_message = on_message
        self._emotes = emotes
        self._keywords = keywords
        self._window_seconds = window_seconds
        self._pusher_app_key = pusher_app_key
        self.hype_threshold = (
            _env_float("CHAT_HYPE_THRESHOLD", 70.0)
            if hype_threshold is None else float(hype_threshold)
        )
        self.callback_cooldown = (
            _env_float("CHAT_HYPE_CALLBACK_COOLDOWN", 30.0)
            if callback_cooldown is None else float(callback_cooldown)
        )
        self._sleep = sleep_fn or time.sleep
        self._trackers = {}            # chatroom_id -> ChatHypeTracker
        self._subscribed = set()       # chatroom_ids we want (re-sub on reconnect)
        self._last_hype_callback = {}  # chatroom_id -> monotonic ts
        self._lock = threading.Lock()
        self._ws = None
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False

    # -- availability / lifecycle ---------------------------------------
    @property
    def available(self):
        """True when the websocket-client package is importable."""
        return _WEBSOCKET_AVAILABLE

    @property
    def running(self):
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def subscribed_chatrooms(self):
        with self._lock:
            return sorted(self._subscribed)

    def start(self):
        """Start the background connection thread. Returns False if the
        websocket-client package is missing (logs a clear error instead of
        crashing)."""
        if not self.available:
            log_event(
                "ERROR", "KickChatClient",
                "websocket-client is not installed; chat hype disabled. "
                "Install it with: pip install websocket-client",
            )
            return False
        if self._running:
            return True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="KickChatClient")
        self._thread.start()
        self._running = True
        log_event("INFO", "KickChatClient", "Chat client started")
        return True

    def stop(self):
        self._stop_event.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._running = False
        log_event("INFO", "KickChatClient", "Chat client stopped")

    # -- subscriptions ----------------------------------------------------
    def subscribe(self, chatroom_id):
        """Subscribe to a chatroom (idempotent; re-sent on reconnect)."""
        chatroom_id = int(chatroom_id)
        with self._lock:
            if chatroom_id not in self._trackers:
                self._trackers[chatroom_id] = ChatHypeTracker(
                    chatroom_id,
                    window_seconds=self._window_seconds,
                    emotes=self._emotes,
                    keywords=self._keywords,
                )
            self._subscribed.add(chatroom_id)
        ws = self._ws
        if ws is not None:
            self._send_subscribe(ws, chatroom_id)

    def unsubscribe(self, chatroom_id):
        chatroom_id = int(chatroom_id)
        with self._lock:
            self._subscribed.discard(chatroom_id)
            self._trackers.pop(chatroom_id, None)
        ws = self._ws
        if ws is not None:
            try:
                ws.send(json.dumps({
                    "event": "pusher:unsubscribe",
                    "data": {"channel": chatroom_channel(chatroom_id)},
                }))
            except Exception:
                pass

    def _send_subscribe(self, ws, chatroom_id):
        try:
            ws.send(json.dumps({
                "event": "pusher:subscribe",
                "data": {"channel": chatroom_channel(chatroom_id)},
            }))
        except Exception as e:
            log_event("WARNING", "KickChatClient",
                      f"Subscribe failed for chatroom {chatroom_id}: {e}")

    # -- scoring API (used by the watcher) --------------------------------
    def get_hype_score(self, chatroom_id):
        """Hype score 0-100 for a chatroom, or 0.0 if not subscribed."""
        tracker = self._trackers.get(int(chatroom_id))
        return tracker.get_hype_score() if tracker else 0.0

    def get_stats(self, chatroom_id):
        tracker = self._trackers.get(int(chatroom_id))
        return tracker.get_stats() if tracker else None

    # -- connection loop --------------------------------------------------
    def _run_loop(self):
        """Connect, run until dropped, reconnect with exponential backoff."""
        backoff = 1.0
        max_backoff = 60.0
        while not self._stop_event.is_set():
            connected_at = time.monotonic()
            try:
                ws = websocket.WebSocketApp(
                    get_pusher_url(self._pusher_app_key),
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws = ws
                ws.run_forever(ping_interval=30, ping_timeout=10)
                # A connection that stayed up a while earns a backoff
                # reset; quick failures keep backing off exponentially.
                if time.monotonic() - connected_at >= 10.0:
                    backoff = 1.0
            except Exception as e:
                log_event("WARNING", "KickChatClient",
                          f"Chat connection error: {e}")
            finally:
                self._ws = None
            if self._stop_event.is_set():
                break
            log_event("INFO", "KickChatClient",
                      f"Reconnecting chat in {backoff:.0f}s")
            self._sleep(backoff)
            backoff = min(max_backoff, backoff * 2.0)

    def _on_open(self, ws):
        # (Re-)subscribe to every chatroom we track.
        for chatroom_id in self.subscribed_chatrooms:
            self._send_subscribe(ws, chatroom_id)

    def _on_error(self, ws, error):
        log_event("WARNING", "KickChatClient", f"Chat websocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        log_event("INFO", "KickChatClient",
                  f"Chat websocket closed ({close_status_code}): {close_msg}")

    def _on_message(self, ws, raw_message):
        try:
            self.handle_raw_message(raw_message)
        except Exception as e:
            log_event("WARNING", "KickChatClient",
                      f"Failed to handle chat message: {e}")

    # -- message handling ---------------------------------------------------
    def handle_raw_message(self, raw_message):
        """Parse one raw Pusher envelope and feed the matching tracker.

        Public (not just a callback) so it can be unit-tested with canned
        payloads and so other transports can feed messages through it.
        Returns the parsed chat message dict, or None for non-chat events.
        """
        try:
            envelope = json.loads(raw_message)
        except (TypeError, ValueError):
            return None
        event = envelope.get("event", "")
        if event == "pusher:connection_established":
            return None
        if event == "pusher:error":
            log_event("WARNING", "KickChatClient",
                      f"Pusher error: {envelope.get('data')}")
            return None
        if event not in CHAT_MESSAGE_EVENTS:
            return None  # Subscription / ban / poll / etc. -- ignored for now

        data = envelope.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                return None
        if not isinstance(data, dict):
            return None

        chatroom_id = data.get("chatroom_id")
        if chatroom_id is None:
            match = _CHANNEL_RE.search(envelope.get("channel", "") or "")
            if match:
                chatroom_id = int(match.group(1))
        if chatroom_id is None:
            return None
        chatroom_id = int(chatroom_id)

        with self._lock:
            tracker = self._trackers.get(chatroom_id)
        if tracker is None:
            # Message for a chatroom we are not tracking; ignore it.
            return None

        content = data.get("content", "") or ""
        tracker.record_message(content)

        if self.on_message is not None:
            try:
                self.on_message(chatroom_id, data)
            except Exception as e:
                log_event("WARNING", "KickChatClient",
                          f"on_message callback failed: {e}")

        score = tracker.get_hype_score()
        if score >= self.hype_threshold and self.on_hype is not None:
            now = time.monotonic()
            with self._lock:
                last = self._last_hype_callback.get(chatroom_id, 0.0)
                due = (now - last) >= self.callback_cooldown
                if due:
                    self._last_hype_callback[chatroom_id] = now
            if due:
                try:
                    self.on_hype(chatroom_id, score, tracker.get_stats())
                except Exception as e:
                    log_event("WARNING", "KickChatClient",
                              f"on_hype callback failed: {e}")
        return data


def create_chat_client_from_env(on_hype=None, on_message=None):
    """Build a KickChatClient from env flags, or None when disabled.

    Returns None (with a log line) when CHAT_HYPE_ENABLED is not true or
    when websocket-client is missing. The caller must still call .start().
    """
    if not _env_bool("CHAT_HYPE_ENABLED", False):
        return None
    if not _WEBSOCKET_AVAILABLE:
        log_event(
            "ERROR", "KickChatClient",
            "CHAT_HYPE_ENABLED=true but websocket-client is not installed; "
            "chat hype disabled. Install it with: pip install websocket-client",
        )
        return None
    return KickChatClient(on_hype=on_hype, on_message=on_message)
