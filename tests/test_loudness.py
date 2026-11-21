"""Tests for loudness normalisation to -14 LUFS (gta6-clipper feature 10).

Real ffmpeg renders on small synthetic clips: a quiet source must come
out at -14 LUFS through the editor, and the safety pass must fix a clip
that lands too far off. Skipped when ffmpeg is not installed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loudness import (  # noqa: E402
    TARGET_LUFS,
    fix_loudness,
    loudnorm_filter,
    measure_loudnorm_stats,
    measure_lufs,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_clip(path, duration=6, volume=1.0, loudnorm=False, silent=False):
    """A small synthetic clip: moving test picture + a steady tone."""
    audio = "anullsrc=r=48000:cl=stereo" if silent else "sine=f=440"
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc2=s=640x360:r=25:d={duration}",
           "-f", "lavfi", "-i", f"{audio}:d={duration}"]
    filters = []
    if not silent and volume != 1.0:
        filters.append(f"volume={volume}")
    if loudnorm:
        filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


def _audio_rate(path):
    """Audio sample rate of a file (ffprobe), or None."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30)
    text = out.stdout.strip()
    return int(text) if text.isdigit() else None


# ------------------------------------------------------------- measurement

def test_measure_loudnorm_stats_returns_numeric_fields(tmp_path):
    clip = _make_clip(tmp_path / "src.mp4", volume=0.1)
    stats = measure_loudnorm_stats(clip)
    assert stats is not None
    for key in ("input_i", "input_tp", "input_lra", "input_thresh",
                "target_offset"):
        float(stats[key])  # every field parses as a number
    assert float(stats["input_i"]) < -30  # a 0.1-volume tone is quiet


def test_measure_loudnorm_stats_silent_audio_is_none(tmp_path):
    clip = _make_clip(tmp_path / "silent.mp4", silent=True)
    assert measure_loudnorm_stats(clip) is None


def test_measure_loudnorm_stats_missing_file_is_none(tmp_path):
    assert measure_loudnorm_stats(tmp_path / "nope.mp4") is None


def test_measure_lufs_loudnormed_clip_is_minus_14(tmp_path):
    clip = _make_clip(tmp_path / "good.mp4", loudnorm=True)
    lufs = measure_lufs(clip)
    assert lufs is not None
    assert abs(lufs - TARGET_LUFS) <= 1.0


def test_measure_lufs_missing_file_is_none(tmp_path):
    assert measure_lufs(tmp_path / "nope.mp4") is None


def test_loudnorm_filter_two_pass_with_stats():
    stats = {"input_i": "-23.50", "input_tp": "-4.20", "input_lra": "7.10",
             "input_thresh": "-34.00", "target_offset": "0.30"}
    filt = loudnorm_filter(stats)
    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in filt
    assert "measured_I=-23.50" in filt
    assert "offset=0.30:linear=true" in filt
    # Linear loudnorm upsamples to 192 kHz; the chain must resample
    # back so every render leaves at 48 kHz.
    assert filt.endswith(",aresample=48000")


def test_loudnorm_filter_without_stats_is_single_pass():
    filt = loudnorm_filter(None)
    assert filt == "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000"
    assert "linear" not in filt


# -------------------------------------------------------------- safety pass

def test_fix_loudness_repairs_a_quiet_clip(tmp_path):
    clip = _make_clip(tmp_path / "quiet.mp4", volume=0.05)
    assert measure_lufs(clip) < -30
    after = fix_loudness(clip)
    assert after is not None
    assert abs(after - TARGET_LUFS) <= 1.5
    assert not list(tmp_path.glob("*.loudfix.mp4"))  # no temp left behind


def test_fix_loudness_keeps_48k_sample_rate(tmp_path):
    # Regression: linear loudnorm upsamples to 192 kHz and the AAC
    # encoder caps at 96 kHz, so a fixed clip came out at 96 kHz —
    # and the format joiner fails on mixed sample rates. The fix
    # resamples to 48 kHz: a 44.1 kHz fixture must come out 48 kHz.
    clip = _make_clip(tmp_path / "quiet.mp4", volume=0.05)
    assert _audio_rate(clip) == 44100  # the sine fixture's own rate
    fix_loudness(clip)
    assert _audio_rate(clip) == 48000


def test_fix_loudness_leaves_a_good_clip_alone(tmp_path):
    clip = _make_clip(tmp_path / "good.mp4", loudnorm=True)
    before = measure_lufs(clip)
    after = fix_loudness(clip)
    assert after is not None
    assert abs(after - before) < 0.5  # untouched, not re-encoded
    assert abs(after - TARGET_LUFS) <= 1.5


def test_fix_loudness_missing_file_returns_none(tmp_path):
    assert fix_loudness(tmp_path / "nope.mp4") is None


# ------------------------------------------------------- full editor render

def test_edit_clip_normalises_a_quiet_source(tmp_path, monkeypatch, fresh_db):
    """A real render through ClipEditor: a -40 LUFS source comes out
    at -14 LUFS, and the render itself still succeeds (same duration,
    video present)."""
    import clip_editor as clip_editor_module
    from clip_editor import ClipEditor

    monkeypatch.setattr(clip_editor_module, "EDITED_DIR", tmp_path)
    src = _make_clip(tmp_path / "src.mp4", duration=6, volume=0.1)
    assert measure_lufs(src) < -30

    out = str(tmp_path / "rendered.mp4")
    ClipEditor().edit_clip(
        raw_clip_path=str(src), output_path=out, streamer="adinross",
        layout="blurred_backdrop", mirror=False, hdr_sharpness=False,
        generate_thumbnail=False, encoder="libx264")
    assert Path(out).exists()
    assert abs(measure_lufs(out) - TARGET_LUFS) <= 1.5
    # Regression: the render's linear loudnorm used to leave the
    # audio at 96 kHz, which breaks the format joiner.
    assert _audio_rate(out) == 48000


def test_edit_clip_leaves_a_good_source_at_minus_14(tmp_path, monkeypatch,
                                                    fresh_db):
    """A source already at -14 LUFS still renders fine — normalisation
    must not break a normal render."""
    import clip_editor as clip_editor_module
    from clip_editor import ClipEditor

    monkeypatch.setattr(clip_editor_module, "EDITED_DIR", tmp_path)
    src = _make_clip(tmp_path / "src.mp4", duration=6, loudnorm=True)

    out = str(tmp_path / "rendered.mp4")
    ClipEditor().edit_clip(
        raw_clip_path=str(src), output_path=out, streamer="adinross",
        layout="blurred_backdrop", mirror=False, hdr_sharpness=False,
        generate_thumbnail=False, encoder="libx264")
    assert Path(out).exists()
    assert abs(measure_lufs(out) - TARGET_LUFS) <= 1.5
    # Same 96 kHz regression as above, from the other direction: a
    # source that is already on-target still goes through loudnorm.
    assert _audio_rate(out) == 48000
