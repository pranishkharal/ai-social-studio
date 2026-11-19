"""Tests for silence_cutter.py (feature 9: dead-space removal)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import silence_cutter
from silence_cutter import compute_keep_segments, cut_silence, remap_words

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def words(*spans):
    """Build word dicts from (word, start, end) triples."""
    return [{"word": w, "start": s, "end": e} for w, s, e in spans]


# --- compute_keep_segments ---

def test_no_words_returns_empty():
    assert compute_keep_segments([]) == []
    assert compute_keep_segments(None) == []


def test_single_word_gets_pad():
    segs = compute_keep_segments(words(("hello", 1.0, 1.5)), max_gap=1.5, pad=0.15)
    assert segs == [(0.85, 1.65)]


def test_pad_clamped_at_zero():
    segs = compute_keep_segments(words(("hi", 0.05, 0.4)), max_gap=1.5, pad=0.15)
    assert segs[0][0] == 0.0


def test_short_gap_kept_long_gap_cut():
    # gap 1.0s kept, gap 2.0s cut (threshold 1.5)
    ws = words(("one", 0.0, 0.5), ("two", 1.5, 2.0), ("three", 4.0, 4.5))
    segs = compute_keep_segments(ws, max_gap=1.5, pad=0.15)
    assert segs == [(0.0, 2.15), (3.85, 4.65)]


def test_gap_exactly_at_threshold_is_kept():
    # Only gaps LONGER than the threshold are dropped.
    ws = words(("one", 0.0, 0.5), ("two", 2.0, 2.5))
    segs = compute_keep_segments(ws, max_gap=1.5, pad=0.15)
    assert len(segs) == 1


def test_unsorted_words_are_sorted():
    ws = words(("two", 5.0, 5.5), ("one", 0.0, 0.5))
    segs = compute_keep_segments(ws, max_gap=1.5, pad=0.1)
    assert segs == [(0.0, 0.6), (4.9, 5.6)]


def test_overlapping_pads_merge():
    # Gap 1.6s > 1.5 threshold, but pad 1.0 on both sides overlaps segments.
    ws = words(("one", 0.0, 0.5), ("two", 2.1, 2.6))
    segs = compute_keep_segments(ws, max_gap=1.5, pad=1.0)
    assert segs == [(0.0, 3.6)]


def test_malformed_words_skipped():
    ws = [{"word": "ok", "start": 0.0, "end": 0.5},
          {"word": "bad"},  # no timestamps
          {"word": "alsobad", "start": "x", "end": 1.0}]
    assert compute_keep_segments(ws, max_gap=1.5, pad=0.0) == [(0.0, 0.5)]


def test_env_vars_drive_defaults(monkeypatch):
    monkeypatch.setenv("SILENCE_MAX_GAP_SECONDS", "3.0")
    monkeypatch.setenv("SILENCE_PAD_SECONDS", "0.0")
    ws = words(("one", 0.0, 0.5), ("two", 3.0, 3.5))
    # 2.5s gap is under the 3.0s env threshold -> single segment
    assert compute_keep_segments(ws) == [(0.0, 3.5)]


def test_bad_env_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("SILENCE_MAX_GAP_SECONDS", "not-a-number")
    monkeypatch.setenv("SILENCE_PAD_SECONDS", "also-bad")
    assert silence_cutter.max_gap_seconds() == 1.5
    assert silence_cutter.pad_seconds() == 0.15


# --- remap_words ---

def test_remap_words_shifts_timestamps():
    ws = words(("one", 0.5, 1.0), ("two", 5.0, 5.5))
    segs = compute_keep_segments(ws, max_gap=1.5, pad=0.0)
    assert segs == [(0.5, 1.0), (5.0, 5.5)]
    remapped = remap_words(ws, segs)
    # Trimmed clip = [0.5-1.0] + [5.0-5.5], so the first word starts at 0.0
    # and the second word follows immediately after it (4.5s removed total).
    assert remapped[0]["start"] == 0.0
    assert remapped[0]["end"] == 0.5
    assert remapped[1]["start"] == 0.5
    assert remapped[1]["end"] == 1.0


def test_remap_words_empty_segments():
    assert remap_words(words(("a", 0.0, 1.0)), []) == []


# --- cut_silence (mocked ffmpeg) ---

def _fake_run_factory(calls, fail_first_copy_concat=False):
    state = {"concat_copy_failed": False}

    def fake_run(cmd, timeout=300):
        calls.append(list(cmd))
        is_concat = "-f" in cmd and "concat" in cmd
        is_copy = "-c" in cmd and "copy" in cmd
        if is_concat and is_copy and fail_first_copy_concat and not state["concat_copy_failed"]:
            state["concat_copy_failed"] = True
            return subprocess.CompletedProcess(cmd, 1, "", "concat copy failed")
        # Pretend ffmpeg wrote its output file (last arg).
        Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


def test_cut_silence_no_words_copies_input(tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake-video-bytes")
    out = tmp_path / "out.mp4"
    result_path, removed = cut_silence(str(src), [], str(out))
    assert result_path == str(out)
    assert removed == 0.0
    assert out.read_bytes() == b"fake-video-bytes"


def test_cut_silence_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cut_silence(str(tmp_path / "nope.mp4"), words(("a", 0, 1)), str(tmp_path / "o.mp4"))


def test_cut_silence_no_gap_copies_input(tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake-video-bytes")
    out = tmp_path / "out.mp4"
    monkeypatch.setattr(silence_cutter, "_probe_duration", lambda p: 2.0)
    result_path, removed = cut_silence(
        str(src), words(("a", 0.0, 1.0), ("b", 1.2, 2.0)), str(out))
    assert removed == 0.0
    assert out.read_bytes() == b"fake-video-bytes"


def test_cut_silence_mocked_ffmpeg_commands(tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake-video-bytes")
    out = tmp_path / "out.mp4"
    calls = []
    monkeypatch.setattr(silence_cutter, "_probe_duration", lambda p: 10.0)
    monkeypatch.setattr(silence_cutter, "_run_ffmpeg", _fake_run_factory(calls))

    result_path, removed = cut_silence(
        str(src),
        words(("one", 0.5, 1.0), ("two", 6.0, 6.5)),
        str(out), max_gap=1.5, pad=0.15)

    assert result_path == str(out)
    # kept 0.35-1.15 and 5.85-6.65 = 1.6s of 10s -> 8.4s removed
    assert removed == pytest.approx(8.4)
    assert out.exists()

    segment_calls = [c for c in calls if "-ss" in c]
    assert len(segment_calls) == 2
    for c in segment_calls:
        assert "-c" in c and "copy" in c  # stream copy for segments
    concat_calls = [c for c in calls if "-f" in c and "concat" in c]
    assert len(concat_calls) == 1
    assert "-c" in concat_calls[0] and "copy" in concat_calls[0]
    assert concat_calls[0][-1] == str(out)


def test_cut_silence_falls_back_to_reencode(tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake-video-bytes")
    out = tmp_path / "out.mp4"
    calls = []
    monkeypatch.setattr(silence_cutter, "_probe_duration", lambda p: 10.0)
    monkeypatch.setattr(
        silence_cutter, "_run_ffmpeg",
        _fake_run_factory(calls, fail_first_copy_concat=True))

    result_path, removed = cut_silence(
        str(src), words(("one", 0.5, 1.0), ("two", 6.0, 6.5)),
        str(out), max_gap=1.5, pad=0.15)

    assert result_path == str(out)
    assert removed == pytest.approx(8.4)
    concat_calls = [c for c in calls if "-f" in c and "concat" in c]
    assert len(concat_calls) == 2
    assert "libx264" in concat_calls[1]  # re-encode fallback


# --- cut_silence (real ffmpeg, tiny generated media) ---

@needs_ffmpeg
def test_cut_silence_real_ffmpeg(tmp_path):
    src = tmp_path / "in.mp4"
    # 6s clip, frequent keyframes so stream-copy cuts land near the targets.
    res = subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "5", "-keyint_min", "5",
        "-c:a", "aac", "-shortest", str(src),
    ], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-300:]

    out = tmp_path / "out.mp4"
    result_path, removed = cut_silence(
        str(src),
        words(("one", 0.5, 1.0), ("two", 4.5, 5.0)),
        str(out), max_gap=1.5, pad=0.15)

    assert result_path == str(out)
    assert out.exists()
    # kept ~1.6s of 6s -> ~4.4s removed (reported from word timings)
    assert removed == pytest.approx(4.4, abs=0.2)
    # Actual output is shorter than the input (keyframe rounding adds a bit).
    duration = silence_cutter._probe_duration(str(out))
    assert duration is not None
    assert 0.5 < duration < 4.0
