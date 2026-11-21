"""Tests for peek.py (gta6-clipper feature 8).

Local strips use a real ffmpeg-generated video; the remote download
path is mocked (no network in tests).
"""

import subprocess

import pytest

import peek


def _make_video(path, seconds=12):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x180:rate=10",
         "-c:v", "libx264", str(path)],
        check=True, timeout=60)
    return path


def test_peek_times():
    assert [round(x) for x in peek.peek_times(100, 130)] == [101, 110, 120, 129]
    times = peek.peek_times(0, 30)
    assert times[0] == 1 and times[-1] == 29
    assert all(0 < t < 30 for t in times)


def test_make_peek_strip_local(tmp_path):
    video = _make_video(tmp_path / "video.mp4")
    out = tmp_path / "strip.jpg"
    result = peek.make_peek_strip(video, 0, 12, out)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 1000


def test_make_peek_strip_remote_uses_download_fn(tmp_path):
    video = _make_video(tmp_path / "video.mp4")

    def fake_download(url, start, end, dest):
        assert url == "https://example.com/vod"
        assert (start, end) == (10.0, 40.0)
        import shutil
        shutil.copy(video, dest)
        return dest

    out = tmp_path / "remote_strip.jpg"
    result = peek.make_peek_strip("https://example.com/vod", 10, 40, out,
                                  download_fn=fake_download)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 1000


def test_peek_moment_failure_returns_none(tmp_path):
    # A missing file fails the peek — and the caller gets None, not an
    # exception, so a failed peek never blocks a render.
    assert peek.peek_moment("bad", tmp_path / "missing.mp4", 0, 10,
                            tmp_path) is None


def test_peek_picks(tmp_path):
    import json
    video = _make_video(tmp_path / "video.mp4")
    picks_file = tmp_path / "picks.json"
    picks_file.write_text(json.dumps([
        {"name": "good", "url": str(video), "start": 0, "end": 12},
        {"name": "bad", "url": str(tmp_path / "missing.mp4"), "start": 0, "end": 12},
    ]))
    results = peek.peek_picks(picks_file, tmp_path / "peek")
    assert results["good"] is not None
    assert results["bad"] is None
    assert (tmp_path / "peek" / "good_peek.jpg").exists()
