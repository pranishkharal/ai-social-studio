"""Tests for kick_monitor.py — livestream playback URL + IVS headers.

Regression cover for the stale live-capture path: the v2 channel
endpoint no longer reliably returns ``playback_url`` — the working
source is GET /api/v2/channels/{slug}/livestream — and Kick's AWS IVS
playback URLs return 400 to bare ffmpeg requests, so ffmpeg needs a
Referer header plus a browser User-Agent.

Network and ffmpeg are fully mocked; no live Kick API is touched.
Kick's API (and the proxies in front of it) sometimes drops a response
mid-read (``IncompleteRead``), so the monitor retries transient fetch
failures a few times before giving up — covered below too.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "dotenv" not in sys.modules:
    _dotenv = types.ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = _dotenv

import kick_monitor  # noqa: E402
from kick_monitor import KickMonitor  # noqa: E402

CHANNEL_LIVE = {
    "id": 123,
    "chatroom": {"id": 456},
    "user": {"bio": "bio text"},
    "livestream": {
        "session_title": "PLAYING GTA 6",
        "categories": [{"name": "Grand Theft Auto VI"}],
        "viewer_count": 36000,
    },
    # NOTE: no top-level playback_url — the stale field the old code read.
}

CHANNEL_LIVE_WITH_PLAYBACK = dict(CHANNEL_LIVE)
CHANNEL_LIVE_WITH_PLAYBACK["playback_url"] = "https://ivs.example/channel.m3u8"

CHANNEL_OFFLINE = {
    "id": 123,
    "chatroom": {"id": 456},
    "user": {"bio": "bio text"},
    "livestream": None,
}

LIVESTREAM_BODY = {
    "data": {
        "id": 789,
        "slug": "adinross",
        "session_title": "PLAYING GTA 6",
        "playback_url": "https://ivs.example/livestream.m3u8?token=abc",
        "viewers": 36000,
    }
}


class _FakeResponse:
    """Minimal requests response: .json() + .raise_for_status()."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


@pytest.fixture
def monitor(monkeypatch):
    """KickMonitor with DB writes stubbed out."""
    monkeypatch.setattr(kick_monitor, "update_streamer_status", lambda *a, **k: None)
    monkeypatch.setattr(kick_monitor, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(kick_monitor, "add_notification", lambda *a, **k: 0)
    # Retries sleep 1s between attempts; skip the wait in tests.
    monkeypatch.setattr(kick_monitor.time, "sleep", lambda *a, **k: None)
    return KickMonitor()


def _fake_get_factory(requested, channel_body, livestream_body=None,
                      livestream_error=None):
    def _fake_get(url, headers=None, timeout=None):
        requested.append(url)
        if url.endswith("/livestream"):
            if livestream_error is not None:
                raise livestream_error
            return _FakeResponse(livestream_body)
        return _FakeResponse(channel_body)

    return _fake_get


def test_fetch_channel_info_uses_livestream_playback_url(monitor, monkeypatch):
    requested = []
    monkeypatch.setattr(
        kick_monitor.requests, "get",
        _fake_get_factory(requested, CHANNEL_LIVE, LIVESTREAM_BODY))
    info = monitor.fetch_channel_info("adinross")
    assert info["is_live"] == 1
    assert info["playback_url"] == "https://ivs.example/livestream.m3u8?token=abc"
    assert info["viewer_count"] == 36000
    assert any(u.endswith("/livestream") for u in requested)


def test_fetch_channel_info_falls_back_to_channel_playback_url(monitor, monkeypatch):
    requested = []
    monkeypatch.setattr(
        kick_monitor.requests, "get",
        _fake_get_factory(requested, CHANNEL_LIVE_WITH_PLAYBACK, None,
                              livestream_error=RuntimeError("livestream 500")))
    info = monitor.fetch_channel_info("adinross")
    assert info["is_live"] == 1
    # Livestream endpoint failed; the channel payload's copy still works.
    assert info["playback_url"] == "https://ivs.example/channel.m3u8"


def test_fetch_channel_info_offline_skips_livestream(monitor, monkeypatch):
    requested = []
    monkeypatch.setattr(
        kick_monitor.requests, "get",
        _fake_get_factory(requested, CHANNEL_OFFLINE, LIVESTREAM_BODY))
    info = monitor.fetch_channel_info("xqc")
    assert info["is_live"] == 0
    assert info["playback_url"] is None
    assert not any(u.endswith("/livestream") for u in requested)


def test_fetch_playback_url_returns_none_on_error(monitor, monkeypatch):
    monkeypatch.setattr(
        kick_monitor.requests, "get",
        _fake_get_factory([], CHANNEL_LIVE, None,
                              livestream_error=RuntimeError("boom")))
    assert monitor.fetch_playback_url("adinross") is None


def test_fetch_channel_info_retries_transient_incomplete_read(monitor, monkeypatch):
    """A dropped mid-read response must not mark a live streamer offline.

    Kick's API (and the proxies in front of it) sometimes drops a
    response mid-read; the monitor retries and the next attempt
    succeeds. (Under requests the truncation surfaces as a
    ConnectionError.)
    """
    calls = {"channel": 0}

    def flaky_get(url, headers=None, timeout=None):
        if url.endswith("/livestream"):
            return _FakeResponse(LIVESTREAM_BODY)
        calls["channel"] += 1
        if calls["channel"] < 3:
            raise requests.exceptions.ConnectionError("dropped mid-read")
        return _FakeResponse(CHANNEL_LIVE)

    monkeypatch.setattr(kick_monitor.requests, "get", flaky_get)
    info = monitor.fetch_channel_info("adinross")
    assert calls["channel"] == 3
    assert info["is_live"] == 1
    assert info["viewer_count"] == 36000
    assert info["playback_url"] == "https://ivs.example/livestream.m3u8?token=abc"


def test_get_json_raises_after_all_attempts_fail(monitor, monkeypatch):
    calls = {"n": 0}

    def always_fails(url, headers=None, timeout=None):
        calls["n"] += 1
        raise ConnectionError("dropped")

    monkeypatch.setattr(kick_monitor.requests, "get", always_fails)
    with pytest.raises(ConnectionError):
        monitor._get_json("https://kick.com/api/v2/channels/adinross")
    assert calls["n"] == 3


def test_start_rolling_buffer_passes_ivs_headers(monitor, monkeypatch, tmp_path):
    monkeypatch.setattr(kick_monitor, "BUFFER_DIR", tmp_path / "buffer")
    captured = {}

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)

        def poll(self):
            return None

    monkeypatch.setattr(kick_monitor.subprocess, "Popen", FakePopen)
    monitor.start_rolling_buffer("adinross", "https://ivs.example/live.m3u8")
    cmd = captured["cmd"]
    assert "-headers" in cmd
    assert cmd[cmd.index("-headers") + 1] == "Referer: https://kick.com/"
    assert "-user_agent" in cmd
    assert "Mozilla/5.0" in cmd[cmd.index("-user_agent") + 1]
    # Input options must come before -i.
    assert cmd.index("-headers") < cmd.index("-i")
    assert cmd.index("-user_agent") < cmd.index("-i")


def test_extract_recent_clip_passes_ivs_headers(monitor, monkeypatch, tmp_path):
    monkeypatch.setattr(kick_monitor, "CLIPS_DIR", tmp_path / "clips")
    monitor.streamer_cache["adinross"] = {"playback_url": "https://ivs.example/live.m3u8"}
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * 20000)  # > 10_000 byte success threshold
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(kick_monitor.subprocess, "run", fake_run)
    result = monitor.extract_recent_clip("adinross", duration_seconds=5)
    assert result is not None and Path(result).exists()
    cmd = captured["cmd"]
    assert cmd[cmd.index("-headers") + 1] == "Referer: https://kick.com/"
    assert "Mozilla/5.0" in cmd[cmd.index("-user_agent") + 1]
    assert cmd.index("-headers") < cmd.index("-i")
