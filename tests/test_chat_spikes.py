"""Tests for chat_hype.py: Kick Pusher key negotiation (gta6-clipper
feature 11) and smarter distinct-chatter spikes (feature 12).

No network anywhere: negotiation is tested with an injected opener,
and spikes with synthetic timestamps. The existing test_chat_hype.py
keeps covering the velocity/emote/keyword scoring; this file only
adds the new behaviour.
"""

import json
import sys
import time
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse the conftest stubs when present (combined runs); install the
# same minimal stubs when run standalone.
if "dotenv" not in sys.modules:
    _dotenv = types.ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = _dotenv
if "database" not in sys.modules:
    _database = types.ModuleType("database")
    _database.log_event = lambda *a, **k: None
    sys.modules["database"] = _database

import chat_hype  # noqa: E402
from chat_hype import (  # noqa: E402
    DEFAULT_PUSHER_APP_KEY,
    ChatHypeTracker,
    KickChatClient,
    get_pusher_url,
    is_bot_message,
    negotiate_pusher_credentials,
)


def _negotiation_payload(key="negotiated-key-123", cluster="eu"):
    return {"data": {"connections": [
        {"credentials": {"app_key": key, "cluster": cluster}}]}}


# ---------------------------------------------------------------------------
# Key negotiation (feature 11)
# ---------------------------------------------------------------------------

def test_negotiate_returns_key_and_cluster():
    seen = {}

    def opener(url, body, headers):
        seen["url"] = url
        seen["body"] = json.loads(body)
        return _negotiation_payload()

    key, cluster = negotiate_pusher_credentials(98765, opener=opener)
    assert (key, cluster) == ("negotiated-key-123", "eu")
    assert "98765" in seen["url"]
    assert "web.kick.com" in seen["url"]
    assert seen["body"]["capabilities"]["accepted_providers"] == [
        {"provider": "pusher"}]


def test_negotiate_bad_shape_raises():
    def opener(url, body, headers):
        return {"data": {"connections": []}}

    with pytest.raises((KeyError, IndexError)):
        negotiate_pusher_credentials(1, opener=opener)


def test_get_pusher_url_negotiates_with_channel_id(monkeypatch):
    monkeypatch.setattr(
        chat_hype, "negotiate_pusher_credentials",
        lambda channel_id: ("fresh-key-999", "us2"))
    url = get_pusher_url(channel_id=12345)
    assert url.startswith("wss://ws-us2.pusher.com/app/fresh-key-999?")
    assert "protocol=7" in url


def test_get_pusher_url_negotiation_failure_falls_back(monkeypatch):
    def _boom(channel_id):
        raise RuntimeError("kick is down")

    monkeypatch.setattr(chat_hype, "negotiate_pusher_credentials", _boom)
    url = get_pusher_url(channel_id=12345)
    assert DEFAULT_PUSHER_APP_KEY in url
    assert url.startswith("wss://ws-us2.pusher.com/app/")


def test_get_pusher_url_explicit_key_skips_negotiation(monkeypatch):
    def _boom(channel_id):
        raise AssertionError("negotiation must not run")

    monkeypatch.setattr(chat_hype, "negotiate_pusher_credentials", _boom)
    assert "my-key" in get_pusher_url(app_key="my-key", channel_id=1)


def test_get_pusher_url_env_override_skips_negotiation(monkeypatch):
    def _boom(channel_id):
        raise AssertionError("negotiation must not run")

    monkeypatch.setattr(chat_hype, "negotiate_pusher_credentials", _boom)
    monkeypatch.setenv("KICK_PUSHER_APP_KEY", "env-key-1")
    assert "env-key-1" in get_pusher_url(channel_id=1)


def test_get_pusher_url_no_channel_id_uses_default():
    assert DEFAULT_PUSHER_APP_KEY in get_pusher_url()


def test_client_channel_id_setter():
    client = KickChatClient(channel_id=777)
    assert client._channel_id == 777
    client.set_channel_id(888)
    assert client._channel_id == 888
    client.set_channel_id(None)  # None never clears a known id
    assert client._channel_id == 888


def test_client_connects_with_negotiated_url(monkeypatch):
    """The run loop asks get_pusher_url for the negotiated key."""
    seen = {}
    holder = {}

    class FakeWS:
        def __init__(self, url, **kwargs):
            seen["url"] = url

        def run_forever(self, *a, **k):
            pass  # instant disconnect, like the real drop path

    fake_module = types.ModuleType("websocket")
    fake_module.WebSocketApp = FakeWS
    monkeypatch.setattr(chat_hype, "websocket", fake_module)
    monkeypatch.setattr(chat_hype, "_WEBSOCKET_AVAILABLE", True)
    monkeypatch.setattr(
        chat_hype, "negotiate_pusher_credentials",
        lambda channel_id: ("live-key", "ap1"))
    # Stop the loop after the first (failed) connection attempt.
    client = KickChatClient(
        channel_id=42,
        sleep_fn=lambda s: holder["client"]._stop_event.set())
    holder["client"] = client
    client._run_loop()
    assert seen["url"].startswith("wss://ws-ap1.pusher.com/app/live-key?")


# ---------------------------------------------------------------------------
# Bot filtering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sender,content", [
    ("Nightbot", "hello"),
    ("streamelements", "hello"),
    ("someone", "win big at https://bet.example"),
    ("someone", "check out mysite.com/free"),
    ("someone", "!uptime"),
])
def test_is_bot_message(sender, content):
    assert is_bot_message(sender, content)


@pytest.mark.parametrize("sender,content", [
    ("someone", "LETS GO"),
    ("nightbot_fan", "hello"),  # contains "nightbot" but is not the bot
    (None, "hello"),
    ("someone", ""),
])
def test_is_bot_message_human(sender, content):
    assert not is_bot_message(sender, content)


# ---------------------------------------------------------------------------
# Distinct-chatter spikes (feature 12)
# ---------------------------------------------------------------------------

def _tracker_with_history(base, msgs_per_bin=10, n_bins=30, burst=None):
    """A tracker with 5 min of steady chat, then an optional burst.

    History: ``msgs_per_bin`` distinct chatters per 10 s bin, spread
    evenly. ``burst`` is a list of senders all chatting at
    ``base + 300`` (the current bin at check time ``base + 310``).
    """
    tracker = ChatHypeTracker(111, window_seconds=10)
    for bin_index in range(n_bins):
        for i in range(msgs_per_bin):
            t = base + bin_index * 10 + (i % 10)
            tracker.record_message(
                "hello", timestamp=t,
                sender=f"user_{bin_index}_{i}")
    if burst:
        for i, sender in enumerate(burst):
            tracker.record_message(
                "WOW", timestamp=base + 301 + (i % 10) / 10, sender=sender)
    return tracker


def test_spike_fires_on_crowd_burst():
    base = time.monotonic() - 400
    burst = [f"fan_{i}" for i in range(40)]
    tracker = _tracker_with_history(base, burst=burst)
    spike = tracker.check_chat_spike(now=base + 310)
    assert spike is not None
    assert spike["distinct_chatters"] == 40
    assert spike["baseline_median"] == 10
    assert spike["ratio"] == 4.0
    assert spike["local_fallback"] is False


def test_spike_cooldown_holds_second_fire():
    base = time.monotonic() - 400
    burst = [f"fan_{i}" for i in range(40)]
    tracker = _tracker_with_history(base, burst=burst)
    assert tracker.check_chat_spike(now=base + 310) is not None
    assert tracker.check_chat_spike(now=base + 320) is None  # cooldown
    # After the 60 s cooldown a continued crowd fires again.
    for i, sender in enumerate(burst):
        tracker.record_message("WOW", timestamp=base + 371 + (i % 10) / 10,
                               sender=sender)
    assert tracker.check_chat_spike(now=base + 380) is not None


def test_spike_needs_fifteen_people():
    base = time.monotonic() - 400
    # 14 chatters vs a baseline of 1: 14x the median, but under the
    # 15-person floor -- no spike.
    tracker = _tracker_with_history(
        base, msgs_per_bin=1, burst=[f"fan_{i}" for i in range(14)])
    assert tracker.check_chat_spike(now=base + 310) is None


def test_spike_needs_two_minutes_of_history():
    base = time.monotonic() - 400
    tracker = ChatHypeTracker(111, window_seconds=10)
    for i in range(60):  # only 60 s of history
        tracker.record_message("hi", timestamp=base + i,
                               sender=f"user_{i}")
    for i in range(40):
        tracker.record_message("WOW", timestamp=base + 60 + i / 10,
                               sender=f"fan_{i}")
    assert tracker.check_chat_spike(now=base + 70) is None


def test_one_spammer_flooding_is_one_chatter():
    base = time.monotonic() - 400
    tracker = _tracker_with_history(base)
    for i in range(200):  # 200 messages, one sender
        tracker.record_message("spam", timestamp=base + 300 + i / 20,
                               sender="spammer")
    assert tracker.check_chat_spike(now=base + 310) is None
    assert tracker.distinct_chatters(now=base + 310) == 1


def test_bots_and_commands_do_not_count_as_chatters():
    base = time.monotonic() - 400
    tracker = _tracker_with_history(base)
    for i in range(40):
        tracker.record_message("!uptime", timestamp=base + 300 + i / 10,
                               sender=f"fan_{i}")
    for i in range(40):
        tracker.record_message("hello", timestamp=base + 300 + i / 10,
                               sender="nightbot")
    assert tracker.distinct_chatters(now=base + 310) == 0
    assert tracker.check_chat_spike(now=base + 310) is None


def test_huge_chat_local_median_fallback():
    base = time.monotonic() - 400
    # Median 25 chatters/bin: the global 3x rule (75) is unreachable,
    # but 45 is 1.8x the local median and beyond Poisson noise.
    burst = [f"fan_{i}" for i in range(45)]
    tracker = _tracker_with_history(base, msgs_per_bin=25, burst=burst)
    spike = tracker.check_chat_spike(now=base + 310)
    assert spike is not None
    assert spike["local_fallback"] is True
    assert spike["distinct_chatters"] == 45


def test_huge_chat_small_bump_does_not_fire():
    base = time.monotonic() - 400
    burst = [f"fan_{i}" for i in range(28)]  # 28 vs median 25: noise
    tracker = _tracker_with_history(base, msgs_per_bin=25, burst=burst)
    assert tracker.check_chat_spike(now=base + 310) is None


def test_spike_bonus_in_live_score():
    """A live spike (messages arriving now) lifts get_hype_score."""
    now = time.monotonic()
    base = now - 310
    tracker = _tracker_with_history(base, msgs_per_bin=10,
                                    burst=[f"fan_{i}" for i in range(40)])
    plain = ChatHypeTracker(111, window_seconds=10)
    for bin_index in range(30):
        for i in range(10):
            plain.record_message(
                "hello", timestamp=base + bin_index * 10 + (i % 10))
    for i in range(40):
        plain.record_message("WOW", timestamp=base + 301 + (i % 10) / 10)
    assert tracker.get_hype_score() == plain.get_hype_score() + 20.0
    assert tracker.get_stats()["spike_active"] is True
    assert tracker.get_stats()["distinct_chatters"] == 40


def test_stats_report_distinct_chatters():
    tracker = ChatHypeTracker(111, window_seconds=10)
    now = time.monotonic()
    tracker.record_message("hi", timestamp=now, sender="alice")
    tracker.record_message("hi", timestamp=now, sender="alice")
    tracker.record_message("hi", timestamp=now, sender="bob")
    tracker.record_message("hi", timestamp=now)  # unknown sender
    stats = tracker.get_stats()
    assert stats["distinct_chatters"] == 2
    assert stats["window_messages"] == 4


def test_client_passes_sender_to_tracker():
    client = KickChatClient()
    client.subscribe(123)
    envelope = json.dumps({
        "event": "App\\Events\\ChatMessageEvent",
        "data": json.dumps({
            "id": "m1", "chatroom_id": 123, "content": "KEKW great",
            "sender": {"id": 9, "username": "alice", "slug": "alice"},
        }),
        "channel": "chatrooms.123.v2",
    })
    assert client.handle_raw_message(envelope) is not None
    assert client.get_stats(123)["distinct_chatters"] == 1


def test_record_message_without_sender_keeps_old_behaviour():
    tracker = ChatHypeTracker(111)
    now = time.monotonic()
    tracker.record_message("KEKW hello", timestamp=now)
    assert tracker.get_stats()["window_messages"] == 1
    assert tracker.get_stats()["distinct_chatters"] == 0
