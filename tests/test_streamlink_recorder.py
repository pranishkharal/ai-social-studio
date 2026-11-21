"""Tests for streamlink_recorder.py (gta6-clipper feature 15).

No network: segment handling is tested on real ffmpeg-generated .ts
files, and the record loop is tested through its pure helpers plus a
disk-full short-circuit. The live test (a real stream) is a manual
step — see the build report.
"""

import datetime as dt
import subprocess
import time

import pytest

import streamlink_recorder as rec


def _make_segment(path, seconds=4):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x180:rate=10",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)],
        check=True, timeout=60)


def _seg_name(when):
    return "s_" + when.strftime("%Y%m%d_%H%M%S") + ".ts"


def test_backoff_seconds():
    assert rec.backoff_seconds(0) == 3
    assert rec.backoff_seconds(9) == 3
    assert rec.backoff_seconds(10) == 6
    assert rec.backoff_seconds(11) == 12
    assert rec.backoff_seconds(100) == 300  # capped at 5 min


def test_streamlink_command():
    cmd = rec.streamlink_command("https://kick.com/adinross", python="python3")
    assert cmd[:4] == ["python3", "-m", "streamlink", "--stdout"]
    assert cmd[-2:] == ["https://kick.com/adinross", "720p60,720p,best"]


def test_streamlink_command_quality_override(monkeypatch):
    # Explicit arg wins, then the env var, then the default ladder.
    cmd = rec.streamlink_command("https://kick.com/x", python="python3",
                                 quality="360p")
    assert cmd[-1] == "360p"
    monkeypatch.setenv("STREAMLINK_QUALITY", "480p")
    cmd = rec.streamlink_command("https://kick.com/x", python="python3")
    assert cmd[-1] == "480p"
    monkeypatch.delenv("STREAMLINK_QUALITY")
    cmd = rec.streamlink_command("https://kick.com/x", python="python3")
    assert cmd[-1] == "720p60,720p,best"


def test_ffmpeg_segment_command(tmp_path):
    cmd = rec.ffmpeg_segment_command(tmp_path)
    assert cmd[:5] == ["ffmpeg", "-loglevel", "error", "-i", "-"]
    assert "-segment_time" in cmd and "4" in cmd
    assert str(tmp_path / "s_%Y%m%d_%H%M%S.ts") in cmd


def test_list_segments_sorted_and_prune(tmp_path):
    now = time.time()
    for seconds_ago, name in ((500, "old"), (100, "mid"), (5, "new")):
        when = dt.datetime.fromtimestamp(now - seconds_ago)
        (tmp_path / _seg_name(when)).write_bytes(b"x")
    (tmp_path / "not-a-segment.txt").write_bytes(b"x")
    segments = rec.list_segments(tmp_path)
    assert len(segments) == 3
    assert [s[0] for s in segments] == sorted(s[0] for s in segments)
    removed = rec.prune_segments(tmp_path, keep_seconds=200, now=now)
    assert removed == 1  # only the 500 s old one
    assert len(rec.list_segments(tmp_path)) == 2


def test_latest_window_concatenates(tmp_path):
    now = time.time()
    for seconds_ago in (30, 20, 10):
        when = dt.datetime.fromtimestamp(now - seconds_ago)
        _make_segment(tmp_path / _seg_name(when))
    # One more segment as the "newest, still being written" one.
    _make_segment(tmp_path / _seg_name(dt.datetime.fromtimestamp(now)))
    out = tmp_path / "window.mp4"
    result = rec.latest_window(tmp_path, 60, out)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 1000


def test_latest_window_no_segments(tmp_path):
    assert rec.latest_window(tmp_path, 60, tmp_path / "out.mp4") is None


def test_cut_window(tmp_path):
    now = time.time()
    for seconds_ago in (40, 30, 20, 10):
        when = dt.datetime.fromtimestamp(now - seconds_ago)
        _make_segment(tmp_path / _seg_name(when))
    _make_segment(tmp_path / _seg_name(dt.datetime.fromtimestamp(now)))
    out = tmp_path / "cut.mp4"
    result = rec.cut_window(tmp_path, out, now - 15,
                            seconds_before=10, seconds_after=5)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 1000


def test_cut_window_no_coverage(tmp_path):
    assert rec.cut_window(tmp_path, tmp_path / "cut.mp4", time.time()) is None


def test_run_once_disk_full_short_circuits(monkeypatch, tmp_path):
    monkeypatch.setattr(rec, "disk_full", lambda path, minimum=None: True)
    recorder = rec.StreamlinkRecorder("https://kick.com/x", tmp_path)
    assert recorder.run_once() == "disk_full"


def test_recorder_stop_flag(tmp_path):
    recorder = rec.StreamlinkRecorder("https://kick.com/x", tmp_path)
    assert not recorder.stop_requested
    recorder.stop()
    assert recorder.stop_requested


def test_disk_full_real_disk(tmp_path):
    # /tmp on this VM is a small tmpfs, so test with explicit minimums
    # instead of assuming 2 GB free.
    assert rec.disk_full(tmp_path, minimum=1) is False
    assert rec.disk_full(tmp_path, minimum=10 ** 15) is True
