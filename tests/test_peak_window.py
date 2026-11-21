"""Tests for HypeDetector.find_peak_window — the helper short-clip
mode uses to cut a tight 10-15 s clip around the single loudest
moment instead of a fixed 30 s window.

The burst test builds a real synthetic video with ffmpeg: a quiet tone
with a loud burst at a known time. Everything else is small and fast.
"""

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "dotenv" not in sys.modules:
    _dotenv = types.ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = _dotenv

from hype_detector import HypeDetector  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _make_clip(path, duration=20, burst=None):
    """Synthetic clip: quiet 440 Hz tone, optionally with a loud burst.

    ``burst`` is (start, end) seconds — inside it the tone is boosted
    to full volume; outside it the tone sits at 0.05 (barely audible).
    Video is a tiny low-fps test pattern; the analysis is audio-only.
    """
    audio_filter = "volume=0.05"
    if burst:
        audio_filter += (f",volume=20:enable='between(t,{burst[0]},{burst[1]})'")
    res = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i",
         f"testsrc=duration={duration}:size=320x180:rate=10",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
         "-af", audio_filter,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-300:]


@needs_ffmpeg
def test_find_peak_window_covers_loud_burst(tmp_path):
    video = tmp_path / "burst.mp4"
    _make_clip(video, duration=20, burst=(8, 12))
    window = HypeDetector().find_peak_window(str(video),
                                             window_seconds=5, step_seconds=2.5)
    assert set(window) == {"start", "end", "score"}
    # The winning 5 s window covers the middle of the burst (t=10).
    assert window["start"] <= 10 <= window["end"]
    assert window["end"] - window["start"] == pytest.approx(5.0)
    assert window["score"] > 0


@needs_ffmpeg
def test_find_peak_window_short_video_returns_whole_clip(tmp_path):
    video = tmp_path / "short.mp4"
    _make_clip(video, duration=6)
    window = HypeDetector().find_peak_window(str(video),
                                             window_seconds=15, step_seconds=5)
    assert window["start"] == 0.0
    assert window["end"] == pytest.approx(6.0, abs=0.1)
    assert window["score"] >= 0


def test_find_peak_window_missing_file():
    assert HypeDetector().find_peak_window("/definitely/not/here.mp4") == {
        "start": 0.0, "end": 0.0, "score": 0.0}


@needs_ffmpeg
def test_find_peak_window_scans_tail(tmp_path):
    """A burst at the very end is found even when the stepping does
    not land on it (the tail window flush with the clip end)."""
    video = tmp_path / "tail.mp4"
    _make_clip(video, duration=22, burst=(19, 22))
    window = HypeDetector().find_peak_window(str(video),
                                             window_seconds=5, step_seconds=5)
    assert window["start"] <= 20 <= window["end"]
    assert window["score"] > 0
