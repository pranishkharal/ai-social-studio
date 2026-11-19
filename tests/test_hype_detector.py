"""Tests for hype_detector.py — audio-energy parsing (ffmpeg volumedetect).

Regression cover for the ffmpeg 8 ordering bug: ffmpeg 8 prints
``mean_volume`` BEFORE ``max_volume``, so the old single
max-then-mean regex never matched and every clip fell back to score
10.0. The detector now searches for each value independently, so both
orderings parse. The analysis is also audio-only (``-vn``): decoding
the 1080p60 video stream to null was slow enough to hit the timeout
and fall back to 10.0 on a loaded machine.
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

# Real ffmpeg 8.1.2 volumedetect stderr (mean printed first).
FFMPEG8_STDERR = """\
[Parsed_volumedetect_0 @ 0x7360a4001f40] n_samples: 2881536
[Parsed_volumedetect_0 @ 0x7360a4001f40] mean_volume: -29.2 dB
[Parsed_volumedetect_0 @ 0x7360a4001f40] max_volume: -8.6 dB
[Parsed_volumedetect_0 @ 0x7360a4001f40] histogram_8db: 19
"""

# Older ffmpeg order (max printed first) — must keep working.
FFMPEG_OLD_STDERR = """\
[Parsed_volumedetect_0 @ 0x7360a4001f40] n_samples: 2881536
[Parsed_volumedetect_0 @ 0x7360a4001f40] max_volume: -8.6 dB
[Parsed_volumedetect_0 @ 0x7360a4001f40] mean_volume: -29.2 dB
"""

# max -8.6, mean -29.2 -> jump 20.6:
#   max part: (20 - 8.6) * 2.5 = 28.5
#   jump part: (20.6 - 10) * 2.5 = 26.5
#   total 55.0, and is_hype (max >= -15, jump >= 8).
EXPECTED_SCORE = 55.0


def _fake_run_factory(stderr):
    def _fake_run(cmd, capture_output=True, text=True, timeout=30):
        assert cmd[0] == "ffmpeg"
        assert "volumedetect" in cmd
        # Audio-only: decoding the video stream to null is what made
        # the analysis slow enough to time out on a loaded machine.
        assert "-vn" in cmd
        return subprocess.CompletedProcess(cmd, 0, "", stderr)

    return _fake_run


def test_analyze_audio_energy_ffmpeg8_order(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(FFMPEG8_STDERR))
    result = HypeDetector().analyze_audio_energy(str(video))
    assert result["score"] == EXPECTED_SCORE
    assert result["max_volume"] == -8.6
    assert result["mean_volume"] == -29.2
    assert result["vol_jump"] == pytest.approx(20.6)
    assert result["is_hype"] is True
    assert result["reason"] != "analysis_fallback"


def test_analyze_audio_energy_old_order_still_works(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(FFMPEG_OLD_STDERR))
    result = HypeDetector().analyze_audio_energy(str(video))
    assert result["score"] == EXPECTED_SCORE
    assert result["max_volume"] == -8.6
    assert result["mean_volume"] == -29.2


def test_analyze_audio_energy_no_volumes_falls_back(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr(
        subprocess, "run",
        _fake_run_factory("ffmpeg version 8.1.2\nno volume lines here\n"))
    result = HypeDetector().analyze_audio_energy(str(video))
    assert result == {"score": 10.0, "is_hype": False, "reason": "analysis_fallback"}


def test_analyze_audio_energy_missing_file():
    result = HypeDetector().analyze_audio_energy("/definitely/not/here.mp4")
    assert result == {"score": 0.0, "is_hype": False, "reason": "file_not_found"}


@needs_ffmpeg
def test_analyze_audio_energy_real_ffmpeg(tmp_path):
    """End-to-end against the installed ffmpeg (8.x in this VM/CI)."""
    video = tmp_path / "tone.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=320x180:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(video)],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-300:]
    result = HypeDetector().analyze_audio_energy(str(video))
    assert result["reason"] != "analysis_fallback"
    assert result["max_volume"] > result["mean_volume"]
