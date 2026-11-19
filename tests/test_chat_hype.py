"""Tests for chat_hype.py (Tier 1, feature 2: live chat hype detection).

Self-contained: this VM lacks python-dotenv and websocket-client, and
importing the real database.py would create the repo's real kick_engine.db,
so dotenv/database are stubbed in sys.modules before importing the module
under test -- but only if tests/conftest.py has not already provided them
(it stubs dotenv and points the real database at a throwaway file; reusing
that keeps this file a good citizen in combined pytest runs). websocket-client
is faked per-test via monkeypatch. Stubs this file installs itself are
removed again in teardown_module.
"""

import json
import sys
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- test-only stubs (see module docstring) ---
_INSTALLED_STUBS = []


def _install_stub(name, module):
    """Install *module* under *name* only if nothing is there yet."""
    if name not in sys.modules:
        sys.modules[name] = module
        _INSTALLED_STUBS.append(name)


_dotenv_stub = types.ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **k: False
_install_stub("dotenv", _dotenv_stub)

_database_stub = types.ModuleType("database")
_database_stub.log_event = lambda *a, **k: None
_database_stub.update_streamer_status = lambda *a, **k: None
_database_stub.add_notification = lambda *a, **k: 0
_database_stub.add_clip = lambda *a, **k: 0
_database_stub.get_connection = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("database stubbed in tests"))
_database_stub.get_all_clips = lambda: []
_install_stub("database", _database_stub)


def teardown_module(module):
    for name in _INSTALLED_STUBS:
        sys.modules.pop(name, None)


import chat_hype  # noqa: E402
from chat_hype import (  # noqa: E402
    ChatHypeTracker,
    KickChatClient,
    chatroom_channel,
    create_chat_client_from_env,
    extract_emote_names,
    get_pusher_url,
)


# ----------------------------------------------------------------------
# Helpers / constants
# ----------------------------------------------------------------------
def _chat_envelope(chatroom_id, content, sender="someuser"):
    """A realistic raw Pusher envelope for a Kick chat message."""
    return json.dumps({
        "event": "App\\Events\\ChatMessageEvent",
        "data": json.dumps({
            "id": "msg-1",
            "chatroom_id": chatroom_id,
            "content": content,
            "type": "message",
            "created_at": "2026-11-19T10:00:00Z",
            "sender": {"id": 99, "username": sender, "slug": sender},
        }),
        "channel": f"chatrooms.{chatroom_id}.v2",
    })


class FakeWebSocketApp:
    """Stand-in for websocket.WebSocketApp; records what it is sent."""

    instances = []

    def __init__(self, url, on_open=None, on_message=None, on_error=None,
                 on_close=None):
        self.url = url
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.sent = []
        self.closed = False
        self.run_forever_calls = 0
        FakeWebSocketApp.instances.append(self)

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        self.closed = True

    def run_forever(self, *a, **k):
        self.run_forever_calls += 1
        if self.on_open:
            self.on_open(self)
        # Return immediately (simulates an instant disconnect).


@pytest.fixture(autouse=True)
def _reset_fake_ws():
    FakeWebSocketApp.instances.clear()
    yield
    FakeWebSocketApp.instances.clear()


def _install_fake_websocket(monkeypatch):
    fake_module = types.ModuleType("websocket")
    fake_module.WebSocketApp = FakeWebSocketApp
    monkeypatch.setattr(chat_hype, "websocket", fake_module)
    monkeypatch.setattr(chat_hype, "_WEBSOCKET_AVAILABLE", True)
    return fake_module


# ----------------------------------------------------------------------
# Pusher URL / channel naming (verified Kick shapes)
# ----------------------------------------------------------------------
def test_pusher_url_uses_verified_app_key():
    url = get_pusher_url()
    assert url.startswith("wss://ws-us2.pusher.com/app/32cbd69e4b950bf97679?")
    assert "protocol=7" in url
    assert "client=js" in url


def test_pusher_url_env_override(monkeypatch):
    monkeypatch.setenv("KICK_PUSHER_APP_KEY", "deadbeef1234")
    assert "deadbeef1234" in get_pusher_url()


def test_chatroom_channel_name():
    assert chatroom_channel(12345) == "chatrooms.12345.v2"


# ----------------------------------------------------------------------
# Emote extraction
# ----------------------------------------------------------------------
def test_extract_emote_tag_syntax():
    names = extract_emote_names("that was [emote:123:KEKW] so funny [emote:9:Pog]")
    assert "KEKW" in names
    assert "POG" in names


def test_extract_plain_emote_words():
    names = extract_emote_names("KEKW KEKW OMEGALUL")
    assert "KEKW" in names
    assert "OMEGALUL" in names


def test_extract_emote_names_empty():
    assert extract_emote_names("") == set()
    assert extract_emote_names(None) == set()


# ----------------------------------------------------------------------
# ChatHypeTracker scoring
# ----------------------------------------------------------------------
def _tracker_with_baseline(baseline_mps, window_seconds=10):
    tracker = ChatHypeTracker(111, window_seconds=window_seconds)
    tracker._baseline_mps = baseline_mps
    return tracker


def test_tracker_no_messages_scores_zero():
    tracker = ChatHypeTracker(111)
    assert tracker.get_hype_score() == 0.0


def test_tracker_quiet_chat_scores_low():
    # 5 messages in a 10 s window = 0.5 mps vs a 1.0 baseline: below the
    # baseline, no emotes, no keywords -> 0.
    tracker = _tracker_with_baseline(1.0)
    import time
    now = time.monotonic()
    for i in range(5):
        tracker.record_message("hello there", timestamp=now)
    assert tracker.get_hype_score() == 0.0


def test_tracker_velocity_spike_scores_high():
    # 30 messages in a 10 s window = 3 mps vs a 0.5 baseline = 6x ratio,
    # well above the 3x spike multiplier -> full 60 velocity points.
    tracker = _tracker_with_baseline(0.5)
    import time
    now = time.monotonic()
    for _ in range(30):
        tracker.record_message("hello there", timestamp=now)
    score = tracker.get_hype_score()
    assert score == 60.0


def test_tracker_emote_storm_adds_points():
    # Same message rate as baseline (no velocity points) but every message
    # carries a tracked emote -> 25 emote points.
    tracker = _tracker_with_baseline(1.0)
    import time
    now = time.monotonic()
    for _ in range(5):
        tracker.record_message("KEKW KEKW", timestamp=now)
    assert tracker.get_hype_score() == 25.0


def test_tracker_keyword_burst_adds_points():
    tracker = _tracker_with_baseline(1.0)
    import time
    now = time.monotonic()
    for _ in range(5):
        tracker.record_message("CLIP IT bro", timestamp=now)
    # 5 messages, each with "CLIP IT" (+ also matches "CLIP"): 2 hits each
    # -> 10 hits * 3 = 30, capped at 15.
    assert tracker.get_hype_score() == 15.0


def test_tracker_score_caps_at_100():
    tracker = _tracker_with_baseline(0.2)
    import time
    now = time.monotonic()
    for _ in range(60):
        tracker.record_message("KEKW CLIP IT WTF", timestamp=now)
    assert tracker.get_hype_score() == 100.0


def test_tracker_window_prunes_old_messages():
    tracker = ChatHypeTracker(111, window_seconds=10)
    import time
    now = time.monotonic()
    for _ in range(10):
        tracker.record_message("hello", timestamp=now - 30)  # stale
    assert tracker.get_stats()["window_messages"] == 0
    assert tracker.get_hype_score() == 0.0


def test_tracker_stats_shape():
    tracker = _tracker_with_baseline(1.0)
    import time
    now = time.monotonic()
    tracker.record_message("KEKW hello", timestamp=now)
    tracker.record_message("just chatting", timestamp=now)
    stats = tracker.get_stats()
    assert stats["chatroom_id"] == 111
    assert stats["window_messages"] == 2
    assert stats["emote_messages"] == 1
    assert stats["emote_share"] == 0.5
    assert stats["total_messages"] == 2
    assert stats["messages_per_second"] == 0.2


def test_tracker_baseline_does_not_chase_spikes():
    # A burst must not drag the baseline up with it (else spikes
    # normalise themselves away).
    tracker = _tracker_with_baseline(0.5)
    import time
    now = time.monotonic()
    for _ in range(50):
        tracker.record_message("hello", timestamp=now)
    assert tracker._baseline_mps < 1.0


# ----------------------------------------------------------------------
# KickChatClient message handling
# ----------------------------------------------------------------------
def test_client_ignores_unknown_chatroom():
    client = KickChatClient()
    data = client.handle_raw_message(_chat_envelope(999, "hello"))
    assert data is None  # not subscribed


def test_client_handles_chat_message_and_tracks_it():
    client = KickChatClient()
    client.subscribe(123)
    data = client.handle_raw_message(_chat_envelope(123, "KEKW that was great"))
    assert data is not None
    assert data["content"] == "KEKW that was great"
    stats = client.get_stats(123)
    assert stats["window_messages"] == 1
    assert stats["emote_messages"] == 1


def test_client_get_hype_score_unknown_chatroom_is_zero():
    client = KickChatClient()
    assert client.get_hype_score(4242) == 0.0


def test_client_ignores_non_chat_events():
    client = KickChatClient()
    client.subscribe(123)
    payload = json.dumps({
        "event": "App\\Events\\SubscriptionEvent",
        "data": json.dumps({"chatroom_id": 123}),
        "channel": "chatrooms.123.v2",
    })
    assert client.handle_raw_message(payload) is None
    assert client.get_stats(123)["window_messages"] == 0


def test_client_ignores_malformed_payloads():
    client = KickChatClient()
    client.subscribe(123)
    assert client.handle_raw_message("not json at all") is None
    assert client.handle_raw_message(json.dumps({"event": "pusher:ping"})) is None
    assert client.handle_raw_message(
        json.dumps({"event": "pusher:connection_established",
                    "data": "{}"})) is None


def test_client_accepts_short_chat_event_name():
    client = KickChatClient()
    client.subscribe(123)
    payload = json.dumps({
        "event": "ChatMessage",
        "data": {"chatroom_id": 123, "content": "hi"},
        "channel": "chatrooms.123.v2",
    })
    assert client.handle_raw_message(payload) is not None
    assert client.get_stats(123)["window_messages"] == 1


def test_client_resolves_chatroom_from_channel_when_missing():
    client = KickChatClient()
    client.subscribe(777)
    payload = json.dumps({
        "event": "App\\Events\\ChatMessageEvent",
        "data": json.dumps({"content": "hi there"}),  # no chatroom_id
        "channel": "chatrooms.777.v2",
    })
    assert client.handle_raw_message(payload) is not None
    assert client.get_stats(777)["window_messages"] == 1


def test_client_on_message_callback():
    received = []
    client = KickChatClient(on_message=lambda cid, msg: received.append((cid, msg)))
    client.subscribe(123)
    client.handle_raw_message(_chat_envelope(123, "hello chat"))
    assert len(received) == 1
    assert received[0][0] == 123
    assert received[0][1]["sender"]["username"] == "someuser"


def test_client_on_hype_callback_fires_once_per_cooldown():
    fired = []
    client = KickChatClient(
        on_hype=lambda cid, score, stats: fired.append((cid, score)),
        hype_threshold=50.0,
        callback_cooldown=30.0,
    )
    client.subscribe(123)
    tracker = client._trackers[123]
    tracker._baseline_mps = 0.2
    # Flood: 40 messages with emotes -> score hits 100.
    for _ in range(40):
        client.handle_raw_message(_chat_envelope(123, "KEKW CLIP IT"))
    assert len(fired) == 1
    assert fired[0][0] == 123
    assert fired[0][1] >= 50.0


def test_client_subscribe_unsubscribe():
    client = KickChatClient()
    client.subscribe(111)
    client.subscribe(222)
    assert client.subscribed_chatrooms == [111, 222]
    client.subscribe(111)  # idempotent
    assert client.subscribed_chatrooms == [111, 222]
    client.unsubscribe(111)
    assert client.subscribed_chatrooms == [222]
    assert client.get_stats(111) is None


# ----------------------------------------------------------------------
# KickChatClient connection lifecycle (fake websocket)
# ----------------------------------------------------------------------
def test_client_start_sends_subscribe_on_open(monkeypatch):
    _install_fake_websocket(monkeypatch)
    client = KickChatClient(sleep_fn=lambda s: None)
    client.subscribe(123)
    client.subscribe(456)
    assert client.start() is True
    # The fake's run_forever returns immediately, so the loop reconnects in
    # a tight loop; stop it and check at least one instance subscribed.
    client.stop()
    assert FakeWebSocketApp.instances
    first = FakeWebSocketApp.instances[0]
    assert first.url == get_pusher_url()
    sent_events = [json.loads(p) for p in first.sent]
    assert {"event": "pusher:subscribe",
            "data": {"channel": "chatrooms.123.v2"}} in sent_events
    assert {"event": "pusher:subscribe",
            "data": {"channel": "chatrooms.456.v2"}} in sent_events


def test_client_start_without_websocket_package_fails_cleanly(monkeypatch):
    monkeypatch.setattr(chat_hype, "websocket", None)
    monkeypatch.setattr(chat_hype, "_WEBSOCKET_AVAILABLE", False)
    client = KickChatClient()
    assert client.available is False
    assert client.start() is False


def test_client_reconnects_with_backoff(monkeypatch):
    _install_fake_websocket(monkeypatch)
    sleeps = []
    client = KickChatClient(sleep_fn=lambda s: sleeps.append(s))
    stop_after = threading.Event()

    # Stop the loop once 3 connection attempts have happened.
    orig_run = FakeWebSocketApp.run_forever

    def _counting_run(self, *a, **k):
        orig_run(self, *a, **k)
        if len(FakeWebSocketApp.instances) >= 3:
            client._stop_event.set()

    monkeypatch.setattr(FakeWebSocketApp, "run_forever", _counting_run)
    client.start()
    client._thread.join(timeout=5)
    assert len(FakeWebSocketApp.instances) >= 3
    # Exponential backoff: 1s, then 2s, ...
    assert sleeps[:2] == [1.0, 2.0]


# ----------------------------------------------------------------------
# Env factory
# ----------------------------------------------------------------------
def test_create_client_from_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CHAT_HYPE_ENABLED", raising=False)
    assert create_chat_client_from_env() is None


def test_create_client_from_env_enabled(monkeypatch):
    _install_fake_websocket(monkeypatch)
    monkeypatch.setenv("CHAT_HYPE_ENABLED", "true")
    client = create_chat_client_from_env()
    assert isinstance(client, KickChatClient)


def test_create_client_from_env_enabled_but_no_package(monkeypatch):
    monkeypatch.setenv("CHAT_HYPE_ENABLED", "true")
    monkeypatch.setattr(chat_hype, "websocket", None)
    monkeypatch.setattr(chat_hype, "_WEBSOCKET_AVAILABLE", False)
    assert create_chat_client_from_env() is None
