"""Tests for vod_clipper.py — VOD / multi-source clipping (Feature 6).

No network and no real downloads anywhere in this file: downloads are
tested with a fake yt_dlp module, the missing-yt-dlp path with a patched
import, and the pipeline with a stubbed process_moment. The only real
work is ffmpeg on small synthetic videos made with lavfi.
"""

import builtins
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vod_clipper  # noqa: E402
from vod_clipper import (  # noqa: E402
    VodError,
    clip_vod,
    detect_source_kind,
    download_vod,
    find_best_moment,
    resolve_vod_source,
    _streamer_from_url,
)
from hype_detector import HypeDetector  # noqa: E402
from pipeline import ClipPipeline  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


# ---------------------------------------------------------------------------
# Synthetic video: 12 s, 10 s of quiet tone then a 2 s loud burst at 10 s.
# ffmpeg's sine source is quiet (amplitude ~0.13), so the burst is
# amplified x8 to land near 0 dB — a clear hype contrast with the quiet
# part (max volume about -44 dB).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loud_burst_video(tmp_path_factory):
    video = tmp_path_factory.mktemp("vod") / "burst.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=12:size=320x180:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=10,volume=0.05",
         "-f", "lavfi", "-i", "sine=frequency=880:duration=2,volume=8",
         "-filter_complex", "[1:a][2:a]concat=n=2:v=0:a=1[a]",
         "-map", "0:v", "-map", "[a]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(video)],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-500:]
    return str(video)


# ---------------------------------------------------------------------------
# detect_source_kind
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,kind", [
    ("https://kick.com/adinross", "kick"),
    ("https://kick.com/video/abc123", "kick"),
    ("https://www.twitch.tv/xqc", "twitch"),
    ("https://twitch.tv/videos/123456", "twitch"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
    ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
    ("https://example.com/some-video.mp4", "unknown_url"),
    ("not-a-url-at-all", "unknown_url"),
])
def test_detect_source_kind_urls(url, kind):
    assert detect_source_kind(url) == kind


def test_detect_source_kind_local_file(tmp_path):
    video = tmp_path / "local.mp4"
    video.write_bytes(b"fake")
    assert detect_source_kind(str(video)) == "local_file"


def test_detect_source_kind_missing_path_is_unknown():
    assert detect_source_kind("/definitely/not/here.mp4") == "unknown_url"


# ---------------------------------------------------------------------------
# resolve_vod_source
# ---------------------------------------------------------------------------

def test_resolve_vod_source_local_file(tmp_path):
    video = tmp_path / "local.mp4"
    video.write_bytes(b"fake")
    assert resolve_vod_source(str(video)) == str(video)


def test_resolve_vod_source_missing_path_raises():
    with pytest.raises(VodError):
        resolve_vod_source("/definitely/not/here.mp4")


def test_resolve_vod_source_unknown_url_raises():
    with pytest.raises(VodError):
        resolve_vod_source("https://example.com/some-video.mp4")


def test_resolve_vod_source_empty_raises():
    with pytest.raises(VodError):
        resolve_vod_source("")


def test_resolve_vod_source_delegates_url_to_download(monkeypatch, tmp_path):
    """A recognised URL goes through download_vod (stubbed: no network)."""
    monkeypatch.setattr(
        vod_clipper, "download_vod",
        lambda url, out_dir=None: str(tmp_path / "downloaded.mp4"))
    assert resolve_vod_source("https://kick.com/adinross") == \
        str(tmp_path / "downloaded.mp4")


# ---------------------------------------------------------------------------
# download_vod — yt-dlp missing
# ---------------------------------------------------------------------------

def test_download_vod_without_yt_dlp(monkeypatch, tmp_path):
    """yt-dlp absent -> VodError telling the user how to install it."""
    monkeypatch.delitem(sys.modules, "yt_dlp", raising=False)
    real_import = builtins.__import__

    def _no_yt_dlp(name, *args, **kwargs):
        if name == "yt_dlp" or name.startswith("yt_dlp."):
            raise ImportError("No module named 'yt_dlp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_yt_dlp)
    with pytest.raises(VodError) as excinfo:
        download_vod("https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                     out_dir=str(tmp_path))
    assert "pip install yt-dlp" in str(excinfo.value)


# ---------------------------------------------------------------------------
# download_vod — fake yt_dlp module (no network)
# ---------------------------------------------------------------------------

def _install_fake_yt_dlp(monkeypatch, tmp_path, fail_with=None):
    """Install a fake yt_dlp module; returns the recorded YoutubeDL opts."""
    created = tmp_path / "some_video.mp4"
    recorded = {}

    class FakeYDL:
        def __init__(self, opts):
            recorded.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            if fail_with is not None:
                raise fail_with
            created.write_bytes(b"fake video")
            return {"title": "some_video", "ext": "mp4",
                    "requested_downloads": [{"filepath": str(created)}]}

        def prepare_filename(self, info):
            return str(created)

    fake_module = types.ModuleType("yt_dlp")
    fake_module.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
    return recorded, created


def test_download_vod_uses_yt_dlp_and_returns_path(monkeypatch, tmp_path):
    recorded, created = _install_fake_yt_dlp(monkeypatch, tmp_path)
    result = download_vod("https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                          out_dir=str(tmp_path))
    assert result == str(created)
    # 720p cap, quiet, sanitized filename template in the target dir.
    assert recorded["format"] == "bv*[height<=720]+ba/b[height<=720]"
    assert recorded["quiet"] is True
    assert recorded["no_warnings"] is True
    assert recorded["restrictfilenames"] is True
    assert recorded["outtmpl"] == str(tmp_path / "%(title)s.%(ext)s")


def test_download_vod_default_out_dir_is_config_vod_dir(monkeypatch, tmp_path):
    recorded, _ = _install_fake_yt_dlp(monkeypatch, tmp_path)
    monkeypatch.setattr(vod_clipper, "VOD_DOWNLOAD_DIR", str(tmp_path))
    download_vod("https://kick.com/adinross")
    assert recorded["outtmpl"] == str(tmp_path / "%(title)s.%(ext)s")


def test_download_vod_failure_becomes_vod_error(monkeypatch, tmp_path):
    _install_fake_yt_dlp(monkeypatch, tmp_path,
                         fail_with=RuntimeError("network down"))
    with pytest.raises(VodError) as excinfo:
        download_vod("https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                     out_dir=str(tmp_path))
    assert "Could not download" in str(excinfo.value)


def test_download_vod_missing_file_after_download(monkeypatch, tmp_path):
    """yt-dlp reports success but no file exists -> VodError, not a crash."""

    class FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=True):
            return {"title": "ghost", "ext": "mp4",
                    "requested_downloads": []}

        def prepare_filename(self, info):
            return str(tmp_path / "ghost.mp4")

    fake_module = types.ModuleType("yt_dlp")
    fake_module.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
    with pytest.raises(VodError) as excinfo:
        download_vod("https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                     out_dir=str(tmp_path))
    assert "could not be found" in str(excinfo.value)


# ---------------------------------------------------------------------------
# _streamer_from_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://kick.com/adinross", "adinross"),
    ("https://www.twitch.tv/xqc", "xqc"),
    ("https://twitch.tv/videos/123456", None),   # /videos/ is a route
    ("https://kick.com/video/abc123", None),     # /video/ is a route
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", None),
])
def test_streamer_from_url(url, expected):
    assert _streamer_from_url(url) == expected


# ---------------------------------------------------------------------------
# find_best_moment (real ffmpeg on the synthetic video)
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_find_best_moment_finds_loud_burst(loud_burst_video):
    best = find_best_moment(loud_burst_video, window_seconds=3)
    # The loud burst starts at 10 s, inside the [9, 12] window.
    assert best["start"] == pytest.approx(9.0)
    assert best["end"] == pytest.approx(12.0)
    assert best["score"] > 40  # quiet windows score 0


@needs_ffmpeg
def test_find_best_moment_short_video_is_one_window(loud_burst_video):
    best = find_best_moment(loud_burst_video, window_seconds=30)
    assert best["start"] == pytest.approx(0.0)
    assert best["end"] == pytest.approx(12.0)


@needs_ffmpeg
def test_find_best_moment_respects_max_windows(loud_burst_video, monkeypatch):
    calls = []

    def counting(self, path):
        calls.append(path)
        return {"score": float(len(calls))}

    monkeypatch.setattr(HypeDetector, "analyze_audio_energy", counting)
    best = find_best_moment(loud_burst_video, window_seconds=3, max_windows=2)
    assert len(calls) == 2  # 12 s video strided into 2 windows, not 4
    assert best["score"] == 2.0


def test_find_best_moment_missing_file_raises():
    with pytest.raises(VodError):
        find_best_moment("/definitely/not/here.mp4")


# ---------------------------------------------------------------------------
# clip_vod end to end with the pipeline stubbed
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_clip_vod_end_to_end_stubbed_pipeline(loud_burst_video, monkeypatch):
    calls = {}

    def fake_process_moment(self, streamer, duration=None, raw_clip_path=None,
                            layout=None, **kwargs):
        calls.update(streamer=streamer, duration=duration,
                     raw_clip_path=raw_clip_path, layout=layout)
        # The extracted window must exist while the pipeline runs.
        assert raw_clip_path and os.path.exists(raw_clip_path)
        return {"status": "ready_to_upload", "clip_id": 1}

    monkeypatch.setattr(ClipPipeline, "process_moment", fake_process_moment)
    result = clip_vod(loud_burst_video, streamer="teststreamer", duration=6,
                      layout="blurred_backdrop", window_seconds=3)

    assert calls["streamer"] == "teststreamer"
    assert calls["duration"] == 6
    assert calls["layout"] == "blurred_backdrop"
    assert calls["raw_clip_path"].endswith(".mp4")
    assert result["status"] == "ready_to_upload"
    assert result["vod_source"] == loud_burst_video
    assert result["vod_window"]["start"] == pytest.approx(9.0)
    assert result["vod_window"]["score"] > 40


@needs_ffmpeg
def test_clip_vod_defaults_streamer_to_vod(loud_burst_video, monkeypatch):
    calls = {}

    def fake_process_moment(self, streamer, duration=None, raw_clip_path=None,
                            layout=None, **kwargs):
        calls["streamer"] = streamer
        return {"status": "ready_to_upload"}

    monkeypatch.setattr(ClipPipeline, "process_moment", fake_process_moment)
    clip_vod(loud_burst_video, duration=6, window_seconds=3)
    # A local file has no streamer slug in its path.
    assert calls["streamer"] == "vod"


@needs_ffmpeg
def test_clip_vod_derives_streamer_from_kick_url(monkeypatch, tmp_path):
    """A Kick URL's slug becomes the streamer (download stubbed)."""
    video = tmp_path / "downloaded.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(vod_clipper, "download_vod",
                        lambda url, out_dir=None: str(video))
    monkeypatch.setattr(
        vod_clipper, "find_best_moment",
        lambda path, window_seconds=30, max_windows=40: {
            "start": 0.0, "end": 25.0, "score": 80.0})
    monkeypatch.setattr(
        vod_clipper, "_extract_window",
        lambda src, start, length, out: (Path(out).write_bytes(b"fake"), out)[1])
    calls = {}

    def fake_process_moment(self, streamer, duration=None, raw_clip_path=None,
                            layout=None, **kwargs):
        calls["streamer"] = streamer
        return {"status": "ready_to_upload"}

    monkeypatch.setattr(ClipPipeline, "process_moment", fake_process_moment)
    result = clip_vod("https://kick.com/adinross", duration=25)
    assert calls["streamer"] == "adinross"
    assert result["vod_source"] == "https://kick.com/adinross"
    assert result["vod_window"] == {"start": 0.0, "end": 25.0, "score": 80.0}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(vod_clipper, "clip_vod",
                        lambda *a, **k: {"status": "ready_to_upload"})
    rc = vod_clipper.main(["some.mp4", "--streamer", "bob", "--duration", "10",
                           "--layout", "blurred_backdrop"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"status": "ready_to_upload"}


def test_cli_vod_error_exits_1(monkeypatch, capsys):
    def _boom(*a, **k):
        raise VodError("no such video")

    monkeypatch.setattr(vod_clipper, "clip_vod", _boom)
    rc = vod_clipper.main(["/nope.mp4"])
    assert rc == 1
    assert "no such video" in capsys.readouterr().err
