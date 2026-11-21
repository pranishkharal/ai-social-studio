"""Tests for vod_signals.py (gta6-clipper feature 18) and its
integration into vod_clipper.find_best_moment.

No network anywhere: the Twitch GQL calls go through an injected
opener, the yt-dlp metadata through an injected metadata function,
and find_best_moment runs on small synthetic videos made with ffmpeg.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vod_signals  # noqa: E402
from vod_signals import (  # noqa: E402
    clipped_signal,
    fetch_signals_for_source,
    fetch_twitch_clips,
    fetch_youtube_heatmap,
    replayed_signal,
    window_signal_bonus,
)
from vod_clipper import find_best_moment  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


# ---------------------------------------------------------------------------
# Twitch clips ([CLIPPED])
# ---------------------------------------------------------------------------

def _gql_opener(responses):
    """An opener returning canned GQL responses by query substring."""
    calls = []

    def opener(query):
        calls.append(query)
        for needle, response in responses:
            if needle in query:
                return response
        raise AssertionError(f"unexpected query: {query[:80]}")

    opener.calls = calls
    return opener


def _clip_edge(slug, offset, duration, views, video_id, title="clip"):
    return {"node": {"slug": slug, "title": title, "viewCount": views,
                     "videoOffsetSeconds": offset,
                     "durationSeconds": duration,
                     "video": {"id": video_id}}}


def test_fetch_twitch_clips_filters_to_this_vod():
    owner = {"video": {"owner": {"login": "xqc"}}}
    edges = [
        _clip_edge("a", 120, 30, 5600, "123456"),
        _clip_edge("b", 300, 25, 900, "999999"),  # another VOD: dropped
        _clip_edge("c", 95, 20, 400, "123456"),
    ]
    opener = _gql_opener([
        ("owner", owner),
        ("clips", {"user": {"clips": {"edges": edges}}}),
    ])
    clips = fetch_twitch_clips("123456", opener=opener)
    assert [c["offset"] for c in clips] == [95, 120]
    assert clips[0]["views"] == 400
    assert clips[1]["slug"] == "a"
    # One owner lookup + one query per period.
    assert len(opener.calls) == 5


def test_fetch_twitch_clips_dedupes_across_periods():
    owner = {"video": {"owner": {"login": "xqc"}}}
    edge = _clip_edge("a", 120, 30, 5600, "123456")
    opener = _gql_opener([
        ("owner", owner),
        ("clips", {"user": {"clips": {"edges": [edge]}}}),
    ])
    assert len(fetch_twitch_clips("123456", opener=opener)) == 1


def test_fetch_twitch_clips_period_failure_tolerated():
    owner = {"video": {"owner": {"login": "xqc"}}}
    calls = {"n": 0}

    def opener(query):
        if "owner" in query:
            return owner
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("server error")  # one period fails
        return {"user": {"clips": {"edges": [
            _clip_edge("a", 120, 30, 100, "123456")]}}}

    assert len(fetch_twitch_clips("123456", opener=opener)) == 1


def test_fetch_twitch_clips_no_owner_returns_empty():
    opener = _gql_opener([("owner", {"video": None})])
    assert fetch_twitch_clips("123456", opener=opener) == []


def test_clipped_signal_tags_and_counts():
    clips = [
        {"offset": 120, "duration": 30, "views": 5600, "title": "a",
         "slug": "a"},
        {"offset": 122, "duration": 20, "views": 400, "title": "b",
         "slug": "b"},
    ]
    assert clipped_signal(clips, 120, 125) == {
        "count": 2, "views": 6000, "tag": " [CLIPPED x2 6.0k views]"}
    assert clipped_signal(clips, 116, 117) == {
        "count": 1, "views": 5600, "tag": " [CLIPPED x1 5.6k views]"}
    # 5 s lead: a line just before the clip still counts.
    assert clipped_signal(clips, 115, 116)["count"] == 1
    assert clipped_signal(clips, 200, 210)["count"] == 0
    assert clipped_signal([], 1, 2)["tag"] == ""
    small = [{"offset": 10, "duration": 10, "views": 42, "title": "s",
              "slug": "s"}]
    assert clipped_signal(small, 10, 15)["tag"] == " [CLIPPED x1 42 views]"


# ---------------------------------------------------------------------------
# YouTube most-replayed ([REPLAYED])
# ---------------------------------------------------------------------------

def _heatmap():
    # 100 bins of 10 s; median value 0.1, one big peak at 510-520 s.
    heatmap = [{"start_time": i * 10, "end_time": (i + 1) * 10,
                "value": 0.1} for i in range(100)]
    heatmap[51]["value"] = 0.8
    return heatmap


def test_replayed_signal_peak_and_quiet():
    heatmap = _heatmap()
    assert replayed_signal(heatmap, 512, 518) == {
        "ratio": 8.0, "tag": " [REPLAYED x8]"}
    assert replayed_signal(heatmap, 2, 8)["tag"] == ""
    assert replayed_signal(heatmap, 100, 200)["tag"] == ""
    assert replayed_signal([], 1, 2) == {"ratio": 0.0, "tag": ""}


def test_replayed_signal_ignores_intro_peak():
    heatmap = _heatmap()
    heatmap[0]["value"] = 0.9  # intro is always replayed: not a moment
    assert replayed_signal(heatmap, 0, 10)["tag"] == ""


def test_fetch_youtube_heatmap_uses_metadata_fn():
    heatmap = _heatmap()
    info = fetch_youtube_heatmap(
        "https://youtube.com/watch?v=abc",
        metadata_fn=lambda url: {"heatmap": heatmap})
    assert info == heatmap


def test_fetch_youtube_heatmap_failures_return_empty():
    assert fetch_youtube_heatmap(
        "https://youtube.com/watch?v=abc",
        metadata_fn=lambda url: {}) == []
    assert fetch_youtube_heatmap(
        "https://youtube.com/watch?v=abc",
        metadata_fn=lambda url: (_ for _ in ()).throw(
            RuntimeError("yt-dlp down"))) == []


# ---------------------------------------------------------------------------
# Combined window bonus
# ---------------------------------------------------------------------------

def test_window_signal_bonus_clipped_scales_with_views():
    small = [{"offset": 10, "duration": 10, "views": 100, "title": "s",
              "slug": "s"}]
    big = [{"offset": 10, "duration": 10, "views": 5000, "title": "b",
            "slug": "b"}]
    small_bonus = window_signal_bonus(small, [], 10, 20)["bonus"]
    big_bonus = window_signal_bonus(big, [], 10, 20)["bonus"]
    assert 0 < small_bonus < big_bonus <= 20.0
    assert window_signal_bonus([], [], 10, 20)["bonus"] == 0.0


def test_window_signal_bonus_replayed_scales_with_ratio():
    heatmap = _heatmap()
    result = window_signal_bonus([], heatmap, 512, 518)
    assert result["bonus"] == 15.0  # 8x ratio = full replay budget
    assert result["replayed"]["tag"] == " [REPLAYED x8]"


def test_window_signal_bonus_combined_caps_at_35():
    clips = [{"offset": 512, "duration": 30, "views": 100000,
              "title": "b", "slug": f"b{i}"} for i in range(5)]
    result = window_signal_bonus(clips, _heatmap(), 512, 518)
    assert result["bonus"] == 35.0


# ---------------------------------------------------------------------------
# Source routing
# ---------------------------------------------------------------------------

def test_fetch_signals_for_source_twitch_url():
    owner = {"video": {"owner": {"login": "xqc"}}}
    opener = _gql_opener([
        ("owner", owner),
        ("clips", {"user": {"clips": {"edges": [
            _clip_edge("a", 120, 30, 100, "123456")]}}}),
    ])
    signals = fetch_signals_for_source(
        "https://www.twitch.tv/videos/123456", opener=opener)
    assert len(signals["clips"]) == 1
    assert signals["heatmap"] == []


def test_fetch_signals_for_source_youtube_url():
    signals = fetch_signals_for_source(
        "https://www.youtube.com/watch?v=abc",
        metadata_fn=lambda url: {"heatmap": _heatmap()})
    assert len(signals["heatmap"]) == 100
    assert signals["clips"] == []


def test_fetch_signals_for_source_kick_and_local_are_empty():
    assert fetch_signals_for_source("https://kick.com/adinross") == {
        "clips": [], "heatmap": []}
    assert fetch_signals_for_source("/tmp/some-video.mp4") == {
        "clips": [], "heatmap": []}


# ---------------------------------------------------------------------------
# find_best_moment integration (real ffmpeg, synthetic video)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uniform_video(tmp_path_factory):
    """12 s of flat quiet audio: every window scores the same, so the
    viewer signals alone decide the best moment."""
    video = tmp_path_factory.mktemp("signals") / "flat.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=12:size=320x180:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12,volume=0.05",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(video)],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-500:]
    return str(video)


@needs_ffmpeg
def test_find_best_moment_without_signals_picks_first_window(uniform_video):
    best = find_best_moment(uniform_video, window_seconds=3)
    assert best["start"] == 0.0  # ties go to the first window
    assert "signal_bonus" not in best


@needs_ffmpeg
def test_find_best_moment_signals_pick_the_clipped_window(uniform_video):
    # A 6 s scan gives windows 0-6 and 6-12; a clip at 11-12 s only
    # overlaps the second window (the 5 s lead stops at 6 s exactly).
    signals = {"clips": [{"offset": 11, "duration": 1, "views": 5000,
                          "title": "c", "slug": "c"}],
               "heatmap": []}
    best = find_best_moment(uniform_video, window_seconds=6,
                            signals=signals)
    assert best["start"] == 6.0  # the window the viewers clipped
    assert best["signal_bonus"] > 0
    assert "[CLIPPED" in best["signal_tags"]
    assert best["score"] > 0


@needs_ffmpeg
def test_find_best_moment_signals_pick_the_replayed_window(uniform_video):
    heatmap = [{"start_time": i * 3, "end_time": (i + 1) * 3,
                "value": 0.1} for i in range(4)]
    heatmap[3]["value"] = 0.8  # peak in the 9-12 s window
    best = find_best_moment(uniform_video, window_seconds=3,
                            signals={"clips": [], "heatmap": heatmap})
    assert best["start"] == 9.0
    assert "[REPLAYED" in best["signal_tags"]


@needs_ffmpeg
def test_clip_vod_passes_signals_for_twitch_url(monkeypatch, tmp_path):
    """clip_vod fetches viewer signals for Twitch URLs and hands them
    to find_best_moment (both mocked -- no network, no ffmpeg scan)."""
    import vod_clipper
    from pipeline import ClipPipeline

    video = tmp_path / "downloaded.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(vod_clipper, "download_vod",
                        lambda url, out_dir=None: str(video))
    captured = {}
    monkeypatch.setattr(
        vod_clipper, "find_best_moment",
        lambda path, window_seconds=30, max_windows=40, signals=None: (
            captured.update(signals=signals),
            {"start": 0.0, "end": 25.0, "score": 80.0})[1])
    monkeypatch.setattr(
        vod_clipper, "_extract_window",
        lambda src, start, length, out: (
            Path(out).write_bytes(b"fake"), out)[1])
    monkeypatch.setattr(
        vod_signals, "fetch_signals_for_source",
        lambda source, opener=None, metadata_fn=None: {
            "clips": [{"offset": 5, "duration": 10, "views": 100,
                       "title": "c", "slug": "c"}],
            "heatmap": []})
    monkeypatch.setattr(
        ClipPipeline, "process_moment",
        lambda self, **kw: {"status": "ready_to_upload"})
    result = vod_clipper.clip_vod(
        "https://www.twitch.tv/videos/123456", duration=25)
    assert result["status"] == "ready_to_upload"
    assert captured["signals"]["clips"][0]["slug"] == "c"


@needs_ffmpeg
def test_clip_vod_skips_signals_when_disabled(monkeypatch, tmp_path):
    import vod_clipper
    from pipeline import ClipPipeline

    video = tmp_path / "downloaded.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(vod_clipper, "download_vod",
                        lambda url, out_dir=None: str(video))
    captured = {}
    monkeypatch.setattr(
        vod_clipper, "find_best_moment",
        lambda path, window_seconds=30, max_windows=40, signals=None: (
            captured.update(signals=signals),
            {"start": 0.0, "end": 25.0, "score": 80.0})[1])
    monkeypatch.setattr(
        vod_clipper, "_extract_window",
        lambda src, start, length, out: (
            Path(out).write_bytes(b"fake"), out)[1])
    monkeypatch.setattr(
        vod_signals, "fetch_signals_for_source",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not fetch")))
    monkeypatch.setattr(
        ClipPipeline, "process_moment",
        lambda self, **kw: {"status": "ready_to_upload"})
    vod_clipper.clip_vod("https://www.twitch.tv/videos/123456",
                         duration=25, use_vod_signals=False)
    assert captured["signals"] is None
