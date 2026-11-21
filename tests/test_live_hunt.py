"""Tests for live_hunt.py (gta6-clipper feature 14).

All network access is mocked: lookup functions are replaced, and the
past-stream fetchers are injected via ``fns=``.
"""

import pytest

import live_hunt


@pytest.fixture(autouse=True)
def clear_caches():
    live_hunt._CACHE.clear()
    live_hunt._PCACHE.clear()
    yield
    live_hunt._CACHE.clear()
    live_hunt._PCACHE.clear()


def test_lookup_caches_for_60s(monkeypatch):
    calls = []

    def fake_look(handle):
        calls.append(handle)
        return {"live": True, "title": "GTA 6", "viewers": 100, "started": ""}

    monkeypatch.setitem(live_hunt.LOOK, "kick", fake_look)
    first = live_hunt.lookup("kick", "adinross", now=1000.0)
    second = live_hunt.lookup("kick", "adinross", now=1050.0)
    assert first is second  # cached
    assert len(calls) == 1
    live_hunt.lookup("kick", "adinross", now=1100.0)
    assert len(calls) == 2  # expired after 60 s


def test_lookup_failure_returns_error_row(monkeypatch):
    def boom(handle):
        raise ConnectionError("blocked")

    monkeypatch.setitem(live_hunt.LOOK, "kick", boom)
    result = live_hunt.lookup("kick", "adinross", now=1000.0)
    assert result["live"] is False
    assert "blocked" in result["error"]


def test_hunt_sorts_live_first_then_viewers(tmp_path):
    def fake_lookup(platform, handle):
        return {
            ("kick", "a"): {"live": False, "title": "", "viewers": 0, "started": ""},
            ("kick", "b"): {"live": True, "title": "B", "viewers": 50, "started": ""},
            ("twitch", "c"): {"live": True, "title": "C", "viewers": 5000, "started": ""},
        }[(platform, handle)]

    rows = live_hunt.hunt(tmp_path, [
        {"platform": "kick", "handle": "a"},
        {"platform": "kick", "handle": "b"},
        {"platform": "twitch", "handle": "c"},
    ], lookup_fn=fake_lookup)
    assert [r["handle"] for r in rows] == ["c", "b", "a"]
    assert rows[0]["viewers"] == 5000
    assert all(r["watching"] is False for r in rows)


def test_stream_url():
    assert live_hunt.stream_url("kick", "adinross") == "https://kick.com/adinross"
    assert live_hunt.stream_url("twitch", "xqc") == "https://www.twitch.tv/xqc"
    assert live_hunt.stream_url("youtube", "ishowspeed") == "https://www.youtube.com/@ishowspeed/live"
    assert live_hunt.stream_url("youtube", "@ishowspeed") == "https://www.youtube.com/@ishowspeed/live"
    with pytest.raises(ValueError):
        live_hunt.stream_url("rumble", "x")


def test_start_rejects_bad_platform_or_handle(tmp_path):
    with pytest.raises(ValueError):
        live_hunt.start(tmp_path, "rumble", "x")
    with pytest.raises(ValueError):
        live_hunt.start(tmp_path, "kick", "bad handle!")


def test_stop_when_nothing_running(tmp_path):
    result = live_hunt.stop("kick", "adinross", workspace=tmp_path)
    assert result == {"ok": True, "was_running": False}


def test_past_uses_injected_fetchers_and_caches(tmp_path):
    calls = []

    def fake_kick(handle):
        calls.append(handle)
        return [{"title": "VOD 1", "url": "https://kick.com/x/videos/1",
                 "date": "2026-11-20", "duration_s": 3600, "views": 10}]

    fns = {"kick": fake_kick}
    rows = live_hunt.past("kick", "x", now=1000.0, fns=fns)
    assert rows[0]["title"] == "VOD 1"
    live_hunt.past("kick", "x", now=1500.0, fns=fns)
    assert len(calls) == 1  # cached for 10 min


def test_past_kick_failure_becomes_error_row():
    def blocked(handle):
        raise Exception("403 Cloudflare")

    rows = live_hunt.past("kick", "x", now=1000.0, fns={"kick": blocked})
    assert len(rows) == 1
    assert "Kick blocks this lookup" in rows[0]["error"]


def test_past_rejects_bad_input():
    with pytest.raises(ValueError):
        live_hunt.past("rumble", "x")
    with pytest.raises(ValueError):
        live_hunt.past("kick", "bad handle!")


def test_hunt_endpoint_uses_lookup(monkeypatch):
    """GET /api/hunt parses the streamers list and returns live rows."""
    import gta6_routers

    def fake_lookup(platform, handle):
        return {"live": True, "title": "Live!", "viewers": 42, "started": ""}

    monkeypatch.setattr(live_hunt, "lookup", fake_lookup)
    rows = gta6_routers.hunt_endpoint(streamers="kick:adinross,twitch:xqc")
    assert len(rows) == 2
    assert all(r["live"] for r in rows)
    assert {r["handle"] for r in rows} == {"adinross", "xqc"}


def test_hunt_past_endpoint_error_row(monkeypatch):
    import gta6_routers

    def blocked(platform, handle):
        raise ValueError("Bad platform or handle")

    monkeypatch.setattr(live_hunt, "past", blocked)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        gta6_routers.hunt_past_endpoint(platform="rumble", handle="x")
    assert exc.value.status_code == 400
