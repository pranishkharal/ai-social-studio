"""Tests for encoder.py — ffmpeg output is faked, nothing is executed."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import encoder  # noqa: E402
from encoder import detect_encoder, select_encoder  # noqa: E402

HEADER = """Encoders:
 V..... = Video
 ------  ------
"""

NVENC_OUTPUT = HEADER + """ V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
"""

AMF_OUTPUT = HEADER + """ V....D h264_amf             AMD AMF H.264 Encoder (codec h264)
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
"""

VAAPI_OUTPUT = HEADER + """ V....D h264_vaapi           H.264/AVC (VAAPI) (codec h264)
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
"""

CPU_ONLY_OUTPUT = HEADER + """ V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
"""

RGB_ONLY_OUTPUT = HEADER + """ V....D libx264rgb           libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 RGB (codec h264)
"""


@pytest.fixture(autouse=True)
def _fresh_cache():
    encoder.reset_cache()
    yield
    encoder.reset_cache()


def test_nvenc_wins_over_cpu():
    codec, args = select_encoder(NVENC_OUTPUT)
    assert codec == "h264_nvenc"
    assert args == ["-preset", "p4", "-rc", "vbr", "-cq", "23"]


def test_amf_when_no_nvenc():
    codec, args = select_encoder(AMF_OUTPUT)
    assert codec == "h264_amf"
    assert "-quality" in args and "balanced" in args


def test_vaapi_when_no_nvenc_or_amf():
    codec, args = select_encoder(VAAPI_OUTPUT)
    assert codec == "h264_vaapi"
    assert args == ["-b:v", "8M"]


def test_libx264_fallback_matches_current_clipeditor_behaviour():
    codec, args = select_encoder(CPU_ONLY_OUTPUT)
    assert codec == "libx264"
    assert args == ["-preset", "veryfast", "-crf", "21"]


def test_empty_output_falls_back_to_libx264():
    assert select_encoder("") == ("libx264", ["-preset", "veryfast", "-crf", "21"])
    assert select_encoder(None) == ("libx264", ["-preset", "veryfast", "-crf", "21"])


def test_libx264rgb_does_not_count_as_libx264():
    # Token matching: "libx264rgb" must not satisfy the libx264 check, so
    # the result still falls back to libx264 defaults via the fallback path.
    codec, args = select_encoder(RGB_ONLY_OUTPUT)
    assert codec == "libx264"
    assert args == ["-preset", "veryfast", "-crf", "21"]


def test_priority_order_nvenc_over_amf_over_vaapi():
    all_three = HEADER + """ V....D h264_vaapi           H.264/AVC (VAAPI) (codec h264)
 V....D h264_amf             AMD AMF H.264 Encoder (codec h264)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
"""
    assert select_encoder(all_three)[0] == "h264_nvenc"
    no_nvenc = all_three.replace(" V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)\n", "")
    assert select_encoder(no_nvenc)[0] == "h264_amf"


def test_detect_encoder_runs_ffmpeg_once_and_caches(monkeypatch):
    calls = []

    class FakeResult:
        stdout = NVENC_OUTPUT

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert detect_encoder()[0] == "h264_nvenc"
    assert detect_encoder()[0] == "h264_nvenc"
    assert len(calls) == 1
    assert calls[0] == ["ffmpeg", "-hide_banner", "-encoders"]


def test_detect_encoder_missing_ffmpeg_returns_libx264(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("no ffmpeg here")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert detect_encoder() == ("libx264", ["-preset", "veryfast", "-crf", "21"])


def test_detect_encoder_accepts_injected_output():
    assert detect_encoder(encoders_output=VAAPI_OUTPUT)[0] == "h264_vaapi"
