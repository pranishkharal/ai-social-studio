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
  is overridable via ``KICK_PUSHER_APP_KEY`` -- and, when the Kick channel
  id is known, the client negotiates the CURRENT key + cluster from Kick's
  own realtime endpoint on every (re)connect (``get_pusher_url``), falling
  back to this hard-coded key when negotiation fails. If chat stops flowing
  and no channel id is set, check the network tab on kick.com for the
  current ``pusher`` URL.

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
* keyword bursts ("CLIP IT", "WTF", "LOL", ...),
* distinct-chatter spikes: different PEOPLE chatting in the last 10 s
  vs. the median of the previous 5 minutes (>= 3x and >= 15 people,
  60 s cooldown; bots, ad links and !commands filtered out). One
  spammer flooding 200 messages is 1 chatter: no spike.

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
import statistics
import threading
import time
import urllib.error
import urllib.request
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
# This is only the FALLBACK now: get_pusher_url() negotiates the current
# key + cluster from Kick's own realtime endpoint whenever it knows the
# channel id (Kick has rotated the key before -- see negotiate below).
DEFAULT_PUSHER_APP_KEY = "32cbd69e4b950bf97679"
PUSHER_CLUSTER_HOST = "ws-us2.pusher.com"
PUSHER_CLIENT_QUERY = "protocol=7&client=js&version=8.4.0&flash=false"

# Kick hands out the current Pusher app key + cluster per request, no
# login (web.kick.com realtime negotiation). The {channel_id} is the
# Kick *channel* id (kick_monitor stores it as "channel_id"), NOT the
# chatroom id used for subscriptions.
KICK_REALTIME_URL = (
    "https://web.kick.com/api/v1/realtime/channels/{channel_id}"
    "/chat/connection"
)
KICK_REALTIME_TIMEOUT = float(
    os.getenv("KICK_REALTIME_TIMEOUT", "10") or 10)

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


def negotiate_pusher_credentials(channel_id, opener=None,
                               timeout=KICK_REALTIME_TIMEOUT):
    """Ask Kick for the current Pusher app key + cluster (no login).

    POSTs to Kick's realtime negotiation endpoint for *channel_id* and
    returns ``(app_key, cluster)``. Raises on any failure (network,
    unexpected shape) -- the caller falls back to the hard-coded key.
    ``opener`` is an injectable ``(url, body_bytes, headers) -> dict``
    for tests; the default uses urllib with a browser User-Agent.
    """
    body = json.dumps({
        "client": {"id": "ai-social-studio", "type": "web"},
        "capabilities": {"accepted_providers": [{"provider": "pusher"}]},
    }).encode()
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    url = KICK_REALTIME_URL.format(channel_id=int(channel_id))
    if opener is not None:
        data = opener(url, body, headers)
    else:
        request = urllib.request.Request(url, body, headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    credentials = (data["data"]["connections"][0]["credentials"])
    return credentials["app_key"], credentials["cluster"]


def get_pusher_url(app_key=None, channel_id=None, cluster=None):
    """Full Pusher websocket URL for Kick chat.

    Priority: an explicit ``app_key`` argument, then the
    ``KICK_PUSHER_APP_KEY`` env override, then -- when ``channel_id``
    is known -- the key + cluster negotiated from Kick's realtime
    endpoint on this very call (Kick rotates the key, so negotiating
    per connection means a rotation can never silently kill chat).
    Any negotiation failure falls back to the hard-coded default key
    and cluster, so chat keeps working exactly as before.
    """
    key = app_key or os.getenv("KICK_PUSHER_APP_KEY")
    host = PUSHER_CLUSTER_HOST
    if cluster:
        host = f"ws-{cluster}.pusher.com"
    if not key and channel_id is not None:
        try:
            key, negotiated_cluster = negotiate_pusher_credentials(
                channel_id)
            host = f"ws-{negotiated_cluster}.pusher.com"
        except Exception as e:
            log_event("WARNING", "KickChatClient",
                      f"Pusher key negotiation failed ({e}); using the "
                      f"built-in key")
    key = key or DEFAULT_PUSHER_APP_KEY
    return f"wss://{host}/app/{key}?{PUSHER_CLIENT_QUERY}"


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


# Well-known chat bots (lowercase). A bot's messages never count toward
# the distinct-chatter spike -- one bot flooding chat must not look
# like a crowd.
BOT_USERNAMES = {
    "nightbot", "streamelements", "streamlabs", "moobot", "fossabot",
    "wizebot", "botrixoficial", "sery_bot", "kofistreambot",
    "pokemoncommunitygame", "kickbot", "bottrix",
}

# Distinct-chatter spike rule (see ChatHypeTracker.check_chat_spike):
# chatters in the last 10 s >= 3x the median of the previous 5 minutes'
# 10 s bins, AND >= 15 people, with a 60 s cooldown between spikes.
SPIKE_WINDOW_SECONDS = 10
SPIKE_HISTORY_SECONDS = 300   # 5 minutes of history bins
SPIKE_MIN_HISTORY_BINS = 12   # need 2 minutes of history before firing
SPIKE_RATIO = 3.0
SPIKE_MIN_CHATTERS = 15
SPIKE_COOLDOWN_SECONDS = 60
# Huge chats (median >= 20 chatters per 10 s bin) are busy all the
# time: a bin 1.25x busier than its own 5-minute neighbourhood (and
# beyond Poisson noise) counts as a spike instead.
SPIKE_HUGE_CHAT_MEDIAN = 20
SPIKE_LOCAL_RATIO = 1.25
# Hype-score points a live distinct-chatter spike adds (capped at 100
# with the rest of the score).
SPIKE_SCORE_BONUS = 20.0


def is_bot_message(sender, content):
    """True when a message should not count as a human chatter: a known
    bot name, an ad link, or a ``!command``."""
    if sender and str(sender).lower() in BOT_USERNAMES:
        return True
    text = content or ""
    return "http" in text or ".com/" in text or text.startswith("!")


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
        # Distinct-chatter spike tracking: (monotonic_ts, sender) for
        # human messages only, kept for the spike window + 5 min of
        # history (330 s). One spammer flooding 200 messages is still
        # 1 chatter, so bots and floods cannot fake a spike.
        self._chatters = deque()
        self._last_spike_at = float("-inf")
        self._last_spike = None  # last fired spike dict (for stats)

    # -- ingestion ------------------------------------------------------
    def record_message(self, content, timestamp=None, sender=None):
        """Record one chat message. *timestamp* is time.monotonic() (tests).

        *sender* is the chatter's username (None when unknown): it is
        only used for the distinct-chatter spike, never for the
        velocity/emote/keyword stats, so callers that do not know the
        sender keep the old behaviour exactly.
        """
        now = time.monotonic() if timestamp is None else timestamp
        upper = (content or "").upper()
        emote_names = extract_emote_names(content)
        had_emote = bool(emote_names & self.emotes)
        keyword_hits = sum(1 for kw in self.keywords if kw in upper)
        with self._lock:
            self._messages.append((now, had_emote, keyword_hits))
            self.total_messages += 1
            self._prune_locked(now)
            if sender and not is_bot_message(sender, content):
                self._chatters.append((now, str(sender).lower()))
                cutoff = now - (SPIKE_WINDOW_SECONDS
                                + SPIKE_HISTORY_SECONDS)
                while self._chatters and self._chatters[0][0] < cutoff:
                    self._chatters.popleft()
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

    # -- distinct-chatter spikes ----------------------------------------
    def _chatter_bins(self, now):
        """Distinct-chatter counts per 10 s bin: (current, [history]).

        History bins cover the previous 5 minutes, newest first, and
        only bins after the first recorded message are returned (a
        chat that just started has no baseline yet).
        """
        window = SPIKE_WINDOW_SECONDS
        first_ts = self._chatters[0][0] if self._chatters else None
        bins = [set() for _ in range(1 + SPIKE_HISTORY_SECONDS // window)]
        for ts, sender in self._chatters:
            age = now - ts
            if age < 0 or age >= window * len(bins):
                continue
            bins[int(age // window)].add(sender)
        current = len(bins[0])
        history = []
        for i in range(1, len(bins)):
            bin_start = now - window * (i + 1)
            if first_ts is not None and bin_start < first_ts:
                break
            history.append(len(bins[i]))
        return current, history

    def distinct_chatters(self, window_seconds=SPIKE_WINDOW_SECONDS,
                          now=None):
        """Distinct human chatters in the last *window_seconds*."""
        now = time.monotonic() if now is None else now
        with self._lock:
            return len({sender for ts, sender in self._chatters
                        if now - window_seconds < ts <= now})

    def check_chat_spike(self, now=None):
        """Distinct-chatter spike check (people, not messages).

        Fires when the distinct chatters in the last 10 s are >= 3x the
        median of the previous 5 minutes' 10 s bins AND >= 15 people,
        with a 60 s cooldown. Needs 2 minutes of history first.
        For huge chats (median >= 20 chatters per bin) the global
        baseline is always high, so a bin 1.25x busier than its own
        5-minute neighbourhood (beyond Poisson noise) fires instead.
        Returns the spike dict when one fires, else None. The cooldown
        means repeated calls during one spike return None after the
        first fire.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            if now - self._last_spike_at <= SPIKE_COOLDOWN_SECONDS:
                return None
            if not self._chatters:
                return None
            current, history = self._chatter_bins(now)
            if len(history) < SPIKE_MIN_HISTORY_BINS:
                return None
            baseline = statistics.median(history)
            fired = None
            if baseline >= SPIKE_HUGE_CHAT_MEDIAN:
                # Local-median fallback for huge chats.
                local = statistics.median(history)
                noise = 3 * (local ** 0.5)
                if (current >= SPIKE_LOCAL_RATIO * local
                        and current >= local + noise):
                    fired = {"distinct_chatters": current,
                             "baseline_median": local,
                             "ratio": round(current / max(local, 1), 2),
                             "local_fallback": True}
            elif (current >= SPIKE_MIN_CHATTERS
                  and current >= SPIKE_RATIO * max(baseline, 1)):
                fired = {"distinct_chatters": current,
                         "baseline_median": baseline,
                         "ratio": round(current / max(baseline, 1), 2),
                         "local_fallback": False}
            if fired is not None:
                self._last_spike_at = now
                self._last_spike = dict(fired, fired_at=now)
            return fired

    @property
    def spike_active(self):
        """True when a distinct-chatter spike fired within the cooldown."""
        return (self._last_spike is not None
                and time.monotonic() - self._last_spike_at
                <= SPIKE_COOLDOWN_SECONDS)

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
            "distinct_chatters": self.distinct_chatters(now=now),
            "spike_active": self.spike_active,
            "last_spike": self._last_spike,
        }

    def get_hype_score(self):
        """Hype score 0-100 for this chatroom right now.

        * Velocity (up to 60): messages/second vs. baseline. A ratio of 1x
          scores 0; a ratio at/above CHAT_VELOCITY_SPIKE_MULTIPLIER (3.0)
          scores the full 60.
        * Emote storm (up to 25): share of window messages carrying a
          tracked emote; 50%+ emote share scores the full 25.
        * Keyword bursts (up to 15): +3 per keyword hit in the window.
        * Distinct-chatter spike (+20): a live spike -- many DIFFERENT
          people chatting at once -- adds a flat bonus on top (the
          total still caps at 100). Bots and one-person floods cannot
          earn this: it counts people, not messages.
        """
        # Evaluating the score also evaluates the spike, so the watcher
        # (which only calls get_hype_score) sees spikes without any
        # extra wiring. The 60 s cooldown keeps this cheap and
        # idempotent once a spike has fired.
        self.check_chat_spike()
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
        spike = SPIKE_SCORE_BONUS if stats["spike_active"] else 0.0
        return round(min(100.0, velocity + emote + keywords + spike), 1)


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
                 sleep_fn=None, channel_id=None):
        self.on_hype = on_hype
        self.on_message = on_message
        self._emotes = emotes
        self._keywords = keywords
        self._window_seconds = window_seconds
        self._pusher_app_key = pusher_app_key
        # Kick channel id (NOT chatroom id): when known, the Pusher app
        # key + cluster are negotiated fresh from Kick on every
        # (re)connect instead of trusting the hard-coded key. The app
        # key is global to Kick, so any one channel id will do -- the
        # watcher sets this from the streamer cache when it can.
        self._channel_id = channel_id
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

    def set_channel_id(self, channel_id):
        """Set the Kick channel id used for Pusher key negotiation.

        Takes effect on the next (re)connect -- the current connection
        keeps its key. Safe to call repeatedly with the same id.
        """
        if channel_id is not None:
            self._channel_id = int(channel_id)

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
                    get_pusher_url(self._pusher_app_key,
                                   channel_id=self._channel_id),
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
        sender = (data.get("sender") or {}).get("username")
        tracker.record_message(content, sender=sender)

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


def create_chat_client_from_env(on_hype=None, on_message=None,
                                channel_id=None):
    """Build a KickChatClient from env flags, or None when disabled.

    Returns None (with a log line) when CHAT_HYPE_ENABLED is not true or
    when websocket-client is missing. The caller must still call .start().
    *channel_id* is the Kick channel id used to negotiate the Pusher
    app key on every connect (see get_pusher_url); it can also be set
    later with ``client.set_channel_id(...)``.
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
    return KickChatClient(on_hype=on_hype, on_message=on_message,
                          channel_id=channel_id)
