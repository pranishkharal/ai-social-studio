"""Tests for alerts.py (feature 16) — Discord / Telegram alerting.

Network calls are mocked (urllib.request.urlopen); the async dispatcher is
monkeypatched to run synchronously so assertions are deterministic.
"""

import json
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import alerts  # noqa: E402


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("DISCORD_WEBHOOK_URL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def run_sync(monkeypatch):
    """Make notify_event() dispatch synchronously instead of in a thread."""
    monkeypatch.setattr(alerts, "_run_async", lambda target, *a, **k: target(*a, **k))


@pytest.fixture
def capture_urlopen(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def _payload(request):
    return json.loads(request.data.decode("utf-8"))


# --- Discord ---------------------------------------------------------------

def test_discord_embed_payload_shape(capture_urlopen):
    ok = alerts.send_discord_webhook(
        "https://discord.com/api/webhooks/123/abc",
        "Streamer is LIVE",
        "xQc is playing Valorant",
        fields={"Viewers": 1234},
        color=0x53FC18,
    )
    assert ok is True
    assert len(capture_urlopen) == 1
    req = capture_urlopen[0]
    assert req.full_url == "https://discord.com/api/webhooks/123/abc"
    assert req.get_header("Content-type") == "application/json"
    payload = _payload(req)
    embed = payload["embeds"][0]
    assert embed["title"] == "Streamer is LIVE"
    assert embed["description"] == "xQc is playing Valorant"
    assert embed["color"] == 0x53FC18
    assert embed["fields"] == [{"name": "Viewers", "value": "1234", "inline": True}]
    assert "timestamp" in embed


def test_discord_default_color_and_no_fields(capture_urlopen):
    alerts.send_discord_webhook("https://discord.com/api/webhooks/1/2", "T", "M")
    embed = _payload(capture_urlopen[0])["embeds"][0]
    assert embed["color"] == alerts.COLOR_BLUE
    assert "fields" not in embed


def test_discord_failure_returns_false_and_never_raises(monkeypatch):
    def boom(request, timeout=None):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert alerts.send_discord_webhook("https://discord.com/api/webhooks/1/2", "T", "M") is False


def test_discord_empty_url_returns_false(capture_urlopen):
    assert alerts.send_discord_webhook("", "T", "M") is False
    assert capture_urlopen == []


# --- Telegram ---------------------------------------------------------------

def test_telegram_url_and_payload(capture_urlopen):
    ok = alerts.send_telegram_message("BOTTOKEN123", "98765", "hello world")
    assert ok is True
    req = capture_urlopen[0]
    assert req.full_url == "https://api.telegram.org/botBOTTOKEN123/sendMessage"
    payload = _payload(req)
    assert payload["chat_id"] == "98765"
    assert payload["text"] == "hello world"


def test_telegram_failure_returns_false_and_never_raises(monkeypatch):
    def boom(request, timeout=None):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert alerts.send_telegram_message("tok", "123", "hi") is False


# --- Dispatcher -------------------------------------------------------------

def test_notify_event_noop_when_unconfigured(monkeypatch, capture_urlopen):
    assert alerts.notify_event("streamer_live", "T", "M") is False
    assert capture_urlopen == []


def test_notify_event_sends_to_both_channels(monkeypatch, run_sync, capture_urlopen):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/2")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    assert alerts.notify_event("clip_ready", "Clip ready", "Funny moment", Clip="#7") is True
    assert len(capture_urlopen) == 2
    urls = [r.full_url for r in capture_urlopen]
    assert "https://discord.com/api/webhooks/1/2" in urls
    assert "https://api.telegram.org/bottok/sendMessage" in urls
    tg_payload = _payload([r for r in capture_urlopen if "telegram" in r.full_url][0])
    assert "Clip ready" in tg_payload["text"]
    assert "Clip: #7" in tg_payload["text"]


def test_notify_event_discord_only(monkeypatch, run_sync, capture_urlopen):
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1/2")
    assert alerts.notify_event("clip_ready", "T", "M") is True
    assert len(capture_urlopen) == 1
    assert "discord.com" in capture_urlopen[0].full_url


def test_notify_event_telegram_needs_both_token_and_chat_id(monkeypatch, run_sync, capture_urlopen):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")  # no chat id
    assert alerts.notify_event("clip_ready", "T", "M") is False
    assert capture_urlopen == []


# --- Convenience helpers ----------------------------------------------------

def test_convenience_helpers_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(alerts, "notify_event", lambda *a, **k: calls.append((a, k)) or True)

    alerts.alert_streamer_live("xqc", "Just Chatting", 5000)
    alerts.alert_clip_ready("xqc", 7, 88, "Funny moment")
    alerts.alert_upload_done(7, "TikTok", "https://tiktok.com/@x/1")
    alerts.alert_upload_failed(7, "TikTok", "login expired")

    assert [c[0][0] for c in calls] == [
        "streamer_live", "clip_ready", "upload_done", "upload_failed",
    ]
    assert "xqc" in calls[0][0][1]  # title mentions the streamer
    assert calls[1][1]["AI_score"] == 88
    assert calls[2][1]["URL"] == "https://tiktok.com/@x/1"
    assert calls[3][1]["Error"] == "login expired"
