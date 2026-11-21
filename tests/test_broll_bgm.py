"""Tests for broll.py and bgm.py — HTTP is mocked, no API key needed."""
import io
import json
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bgm  # noqa: E402
import broll  # noqa: E402

PEXELS_RESPONSE = {
    "videos": [
        {
            "id": 111,
            "duration": 12,
            "video_files": [
                {"file_type": "video/mp4", "quality": "sd", "width": 640,
                 "height": 360, "link": "https://cdn.example.com/sd.mp4"},
                {"file_type": "video/mp4", "quality": "hd", "width": 1920,
                 "height": 1080, "link": "https://cdn.example.com/hd.mp4"},
            ],
        },
        {
            "id": 222,
            "duration": 8,
            "video_files": [
                {"file_type": "video/mp4", "quality": "hd", "width": 1080,
                 "height": 1920, "link": "https://cdn.example.com/portrait.mp4"},
            ],
        },
    ]
}


class FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def read(self, *args, **kwargs):
        return self._buf.read(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _reset_used_ids():
    broll.USED_PEXELS_IDS.clear()
    yield
    broll.USED_PEXELS_IDS.clear()


# ---------------------------------------------------------------- broll search

def test_search_broll_no_api_key_is_graceful_noop(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    assert broll.search_broll("crowd cheering") == []


def test_search_broll_prefers_hd_mp4(monkeypatch):
    def fake_urlopen(req, timeout=None):
        assert "api.pexels.com/videos/search" in req.full_url
        assert "query=crowd+cheering" in req.full_url
        assert "orientation=portrait" in req.full_url
        assert req.get_header("Authorization") == "test-key"
        return FakeHTTPResponse(json.dumps(PEXELS_RESPONSE).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    results = broll.search_broll("crowd cheering", api_key="test-key")
    assert len(results) == 2
    assert results[0] == {"id": 111, "url": "https://cdn.example.com/hd.mp4",
                          "width": 1920, "height": 1080, "duration": 12}
    assert results[1]["url"] == "https://cdn.example.com/portrait.mp4"


def test_search_broll_deduplicates_within_a_run(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return FakeHTTPResponse(json.dumps(PEXELS_RESPONSE).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    first = broll.search_broll("crowd", api_key="k")
    assert len(first) == 2
    # Second identical search: pool exhausted -> resets and returns them again.
    second = broll.search_broll("crowd", api_key="k")
    assert len(second) == 2


def test_search_broll_api_error_returns_empty_list(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise IOError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert broll.search_broll("crowd", api_key="k") == []


def test_search_broll_no_videos_returns_empty_list(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return FakeHTTPResponse(json.dumps({"videos": []}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert broll.search_broll("zzz-no-match", api_key="k") == []


def test_search_broll_skips_videos_without_mp4(monkeypatch):
    payload = {"videos": [{"id": 9, "duration": 5,
                           "video_files": [{"file_type": "video/webm",
                                            "link": "https://x/y.webm"}]}]}

    def fake_urlopen(req, timeout=None):
        return FakeHTTPResponse(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert broll.search_broll("x", api_key="k") == []


# ---------------------------------------------------------------- broll download

def test_download_broll_writes_file(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        return FakeHTTPResponse(b"FAKE-MP4-BYTES")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "broll" / "clip.mp4"
    assert broll.download_broll("https://cdn.example.com/hd.mp4", dest) == str(dest)
    assert dest.read_bytes() == b"FAKE-MP4-BYTES"
    assert not Path(str(dest) + ".part").exists()


def test_download_broll_failure_returns_none_and_no_partial_file(monkeypatch, tmp_path):
    def fake_urlopen(req, timeout=None):
        raise IOError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "clip.mp4"
    assert broll.download_broll("https://cdn.example.com/hd.mp4", dest) is None
    assert not dest.exists()


# ---------------------------------------------------------------- broll filters

def test_kenburns_filter_contains_zoompan():
    filt = broll.build_kenburns_filter(duration=5.0)
    assert "zoompan=" in filt
    assert "s=1080x1920" in filt
    assert filt.startswith("[broll]scale=2160:3840")
    assert filt.endswith("[broll_kb]")
    assert "min(1.0000+" in filt  # zoom ramps up from 1.0


def test_broll_xfade_filter_structure():
    filt, label = broll.build_broll_xfade_filter(
        start=10.0, end=15.0, duration=30.0, fade=0.3)
    assert label == "[broll_out]"
    assert "trim=0:10.00" in filt          # pre segment
    assert "trim=0:5.00" in filt           # b-roll segment (5 s)
    assert "trim=15.00:30.00" in filt      # post segment
    assert filt.count("xfade=transition=fade:duration=0.30") == 2
    assert "offset=9.70" in filt           # first fade starts 0.3 s early
    assert filt.endswith("[broll_out]")


def test_broll_xfade_at_clip_start_has_no_pre_segment():
    filt, _ = broll.build_broll_xfade_filter(start=0.0, end=5.0, duration=30.0)
    assert "trim=0:0.00" not in filt
    assert filt.count("xfade=") == 1


def test_broll_xfade_covering_whole_clip_has_no_xfade():
    filt, label = broll.build_broll_xfade_filter(start=0.0, end=30.0, duration=30.0)
    assert "xfade=" not in filt
    assert filt.endswith("[broll_out]")


def test_broll_xfade_invalid_window_raises():
    with pytest.raises(ValueError, match="Invalid B-roll window"):
        broll.build_broll_xfade_filter(start=10.0, end=5.0, duration=30.0)
    with pytest.raises(ValueError, match="fade"):
        broll.build_broll_xfade_filter(start=10.0, end=10.2, duration=30.0, fade=0.3)


def test_broll_overlay_filter_structure():
    filt, label = broll.build_broll_overlay_filter(start=10.0, end=15.0, fade=0.3)
    assert label == "[broll_out]"
    assert "fade=t=in:st=10.00:d=0.30" in filt
    assert "fade=t=out:st=14.70:d=0.30" in filt
    assert "overlay=0:0:enable='between(t,10.00,15.00)'" in filt


# ---------------------------------------------------------------- bgm

@pytest.fixture
def bgm_pool(tmp_path):
    pool = tmp_path / "bgm"
    (pool / "chill").mkdir(parents=True)
    (pool / "epic").mkdir(parents=True)
    (pool / "upbeat").mkdir(parents=True)
    (pool / "chill" / "calm.mp3").write_bytes(b"x")
    (pool / "chill" / "notes.txt").write_text("not music")
    (pool / "epic" / "battle.mp3").write_bytes(b"x")
    (pool / "epic" / "rise.MP3").write_bytes(b"x")
    return pool


def test_pick_bgm_returns_file_from_mood_dir(bgm_pool):
    picked = bgm.pick_bgm("epic", bgm_dir=bgm_pool)
    assert picked in {str(bgm_pool / "epic" / "battle.mp3"),
                      str(bgm_pool / "epic" / "rise.MP3")}


def test_pick_bgm_falls_back_to_chill_when_mood_empty(bgm_pool):
    assert bgm.pick_bgm("upbeat", bgm_dir=bgm_pool) == str(bgm_pool / "chill" / "calm.mp3")


def test_pick_bgm_falls_back_to_chill_when_mood_missing(bgm_pool):
    assert bgm.pick_bgm("suspense", bgm_dir=bgm_pool) == str(bgm_pool / "chill" / "calm.mp3")


def test_pick_bgm_none_when_pool_empty(tmp_path):
    assert bgm.pick_bgm("epic", bgm_dir=tmp_path / "empty") is None


def test_list_bgm_only_mp3s(bgm_pool):
    assert bgm.list_bgm("chill", bgm_dir=bgm_pool) == [str(bgm_pool / "chill" / "calm.mp3")]


def test_mood_for_category_covers_all_prd_categories():
    expected = {
        "Funny Moments": "upbeat",
        "Exciting Gameplay": "epic",
        "Streamer Reactions": "upbeat",
        "Unexpected Moments": "suspense",
        "Big Wins": "epic",
        "Angry or Emotional Reactions": "sad",
        "Viral Potential": "upbeat",
        "Interesting Conversations": "chill",
    }
    for category, mood in expected.items():
        assert bgm.mood_for_category(category) == mood
    assert bgm.mood_for_category("Something Else") == "chill"


def test_bgm_ffmpeg_input_args_loop_forever():
    assert bgm.bgm_ffmpeg_input_args("/music/epic.mp3") == [
        "-stream_loop", "-1", "-i", "/music/epic.mp3"]


def test_ducking_filter_uses_sidechaincompress():
    filt, label = bgm.build_ducking_filter()
    assert label == "[a_out]"
    assert "sidechaincompress=threshold=0.08:ratio=5.0:attack=100:release=1000" in filt
    assert "[0:a]aformat=" in filt
    assert "[1:a]aformat=" in filt
    assert f"volume={bgm.BGM_BASE_VOLUME}[bgm]" in filt  # soft level (0.12): music stays well under the speech
    assert "amix=inputs=2:duration=first:normalize=0[a_out]" in filt


def test_background_filter_has_no_sidechain():
    filt, _ = bgm.build_background_filter()
    assert "sidechaincompress" not in filt
    assert "amix=inputs=2:duration=first:normalize=0[a_out]" in filt
