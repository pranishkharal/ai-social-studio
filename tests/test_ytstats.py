"""Tests for ytstats.py (gta6-clipper feature 20).

All YouTube calls go through the ``_get`` test hook — no network, no
real API key. Live behaviour needs Sara's API key / OAuth client.
"""

import json
import time
import urllib.parse
from datetime import datetime, timezone

import pytest

import ytstats


@pytest.fixture()
def fake_api(monkeypatch):
    now = time.time()
    iso = lambda h: datetime.fromtimestamp(now - h * 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mk = lambda i, t, views, h: {
        "id": i,
        "snippet": {"title": t, "publishedAt": iso(h)},
        "statistics": {"viewCount": str(views), "likeCount": str(views // 100),
                       "commentCount": "1"},
        "contentDetails": {"duration": "PT30S"}}
    items = [mk("abcdefghij" + str(i), "Watch Rockstar 6 trailer", 1000 + i * 10, 100 + i * 24)
             for i in range(8)]
    items.append(mk("zzzzzzzzzzz",
                    "gta 6 some very long title about nothing in particular here, honestly",
                    20, 30))

    def fake(url, headers):
        if "playlistItems" in url:
            return {"items": [{"contentDetails": {"videoId": x["id"]}} for x in items]}
        if "/channels" in url:
            return {"items": [{"id": "UC" + "a" * 22}]}
        if "youtubeanalytics" in url:
            if "elapsedVideoTimeRatio" in url:
                return {"columnHeaders": [{"name": "elapsedVideoTimeRatio"},
                                          {"name": "audienceWatchRatio"}],
                        "rows": [[0.1, 0.5], [0.5, 0.4]]}
            if "dimensions=video" in url:
                return {"columnHeaders": [{"name": n} for n in
                                          "video views averageViewDuration averageViewPercentage subscribersGained".split()],
                        "rows": [["x", 100, 15, 70, 2]]}
            return {"columnHeaders": [{"name": n} for n in
                                      "views averageViewDuration averageViewPercentage subscribersGained".split()],
                    "rows": [[100, 8, 40, 1]]}
        ids = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("id", [""])[0].split(",")
        return {"items": [x for x in items if x["id"] in ids]}

    monkeypatch.setattr(ytstats, "_get", fake)
    return items


def test_vid_id():
    assert ytstats.vid_id("https://youtube.com/shorts/zzzzzzzzzzz?x=1") == "zzzzzzzzzzz"
    assert ytstats.vid_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert ytstats.vid_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert ytstats.vid_id("not a link") is None


def test_parse_duration():
    assert ytstats.parse_duration("PT1M5S") == 65
    assert ytstats.parse_duration("PT30S") == 30
    assert ytstats.parse_duration("PT1H2M3S") == 3723
    assert ytstats.parse_duration("garbage") == 0


def test_videos_and_latest_shorts(fake_api):
    videos = ytstats.videos("KEY", ["zzzzzzzzzzz"])
    assert len(videos) == 1
    assert videos[0]["views"] == 20
    assert videos[0]["seconds"] == 30
    shorts = ytstats.latest_shorts("KEY", "@x", 20)
    assert len(shorts) == 9


def test_channel_id(fake_api):
    assert ytstats.channel_id("KEY", "UC" + "a" * 22) == "UC" + "a" * 22
    assert ytstats.channel_id("KEY", "@x") == "UC" + "a" * 22


def test_diagnose_key_level(fake_api):
    videos = ytstats.videos("KEY", ["zzzzzzzzzzz"])
    others = ytstats.latest_shorts("KEY", "@x", 20)
    result = ytstats.diagnose(videos[0], others, deep_stats={})
    keys = {r["key"] for r in result["reasons"]}
    assert {"views", "title_len", "verb"} <= keys
    assert result["level"] == "key"


def test_diagnose_analytics_level(tmp_path, fake_api):
    """With a saved OAuth token, deep() adds retention and the
    diagnosis flags the 3-second drop-off and low average view %."""
    ytstats.client_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    ytstats.client_path(tmp_path).write_text(json.dumps({
        "installed": {"client_id": "i", "client_secret": "s",
                      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                      "token_uri": "https://oauth2.googleapis.com/token"}}))
    ytstats.token_path(tmp_path).write_text(json.dumps({
        "access_token": "t", "at": time.time(), "expires_in": 3600}))
    video = ytstats.videos("KEY", ["zzzzzzzzzzz"])[0]
    deep_stats = ytstats.deep(tmp_path, video)
    assert deep_stats["averageViewPercentage"] == 40
    assert deep_stats["kept_at_3s"] == 0.5
    assert deep_stats["median"]["averageViewPercentage"] == 70
    others = ytstats.latest_shorts("KEY", "@x", 20)
    result = ytstats.diagnose(video, others, deep_stats=deep_stats)
    keys = {r["key"] for r in result["reasons"]}
    assert {"avp", "hook3"} <= keys
    assert result["level"] == "analytics"


def test_deep_without_signin_returns_empty(tmp_path):
    video = {"id": "zzzzzzzzzzz", "published": "2026-11-20T00:00:00Z",
             "seconds": 30}
    assert ytstats.deep(tmp_path, video) == {}


def test_check_client_rejects_non_google_urls():
    good = {"installed": {"client_id": "i", "client_secret": "s",
                          "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                          "token_uri": "https://oauth2.googleapis.com/token"}}
    assert ytstats.check_client(good)["client_id"] == "i"
    for bad in ("https://evil.example/token",
                "http://oauth2.googleapis.com/token",
                "https://oauth2.googleapis.com.evil.io/t",
                "https://x@evil.io/t"):
        with pytest.raises(ValueError, match="rejected"):
            ytstats.check_client({"installed": dict(good["installed"], token_uri=bad)})


def test_ai_prompt():
    video = {"title": "gta 6 stuff", "views": 20, "likes": 0, "comments": 0,
             "seconds": 30, "published": "2026-11-20T00:00:00Z"}
    prompt = ytstats.ai_prompt(video, [{"text": "Low views."}], hook="WATCH THIS")
    assert "5 titles" in prompt
    assert "gta 6 stuff" in prompt


def test_diagnose_young_video_gets_info_not_bad():
    now = datetime.now(timezone.utc)
    video = {"id": "a", "title": "Watch Rockstar 6", "views": 5, "likes": 0,
             "comments": 0, "seconds": 30,
             "published": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
    result = ytstats.diagnose(video, [])
    views = [r for r in result["reasons"] if r["key"] == "views"]
    assert views and views[0]["level"] == "info"
