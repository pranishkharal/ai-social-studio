"""Tests for FEATURE 5: soft background music with a built-in track.

- The built-in track assets/bgm/chill/soft_pad.mp3 is found by the
  picker (and is the fallback for every mood). The MP3 is not stored
  in the repo — it is synthesised with ffmpeg on first use when the
  file is missing, so a fresh clone still gets BGM out of the box.
- The mix is soft: BGM_BASE_VOLUME is 0.12 (~-18 dB under the speech),
  read from the BGM_BASE_VOLUME env var.
- A real ffmpeg render with bgm=True succeeds, has an audio stream,
  and the music is audible but much quieter than the speech in a
  speech-free segment (volumedetect on a 4 s synthetic clip: tone for
  the first 2 s, silence for the last 2 s).
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bgm  # noqa: E402
import clip_editor  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")

MEAN_RE = re.compile(r"mean_volume:\s*(-inf|-?[\d.]+)\s*dB")


def _segment_mean_db(path, start, end):
    """Mean volume (dB) of one segment, via atrim + volumedetect.

    atrim (not -ss) is used on purpose: input seeking pulls in the
    frame straddling the cut, which leaks loud samples into a silent
    segment. Returns float("-inf") for digital silence.
    """
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path),
         "-af", f"atrim={start}:{end},volumedetect", "-vn", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60)
    match = MEAN_RE.search(res.stderr)
    assert match, f"no mean_volume in volumedetect output: {res.stderr[-200:]}"
    return float(match.group(1))


def _make_speech_then_silence(path):
    """4 s synthetic clip: 440 Hz tone for 0-2 s (speech proxy),
    silence for 2-4 s (speech-free segment for the mix check).

    The tone is stereo (pan duplicates the mono sine to both
    channels at full level): real stream audio is stereo, and the
    mixer's mono->stereo upmix would otherwise cost -3 dB and make
    level comparisons against the plain render meaningless.
    """
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=duration=4:size=640x360:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-af", "pan=stereo|c0=c0|c1=c0,volume=0:enable='between(t,2,4)'",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-300:]


# ------------------------------------------------------------- built-in track

def test_pick_bgm_finds_builtin_soft_pad(monkeypatch):
    monkeypatch.delenv("BGM_DIR", raising=False)
    picked = bgm.pick_bgm("chill")
    assert picked is not None
    assert picked.endswith(str(Path("assets") / "bgm" / "chill" / "soft_pad.mp3"))
    assert Path(picked).is_file()


def test_pick_bgm_falls_back_to_builtin_for_empty_moods(monkeypatch):
    monkeypatch.delenv("BGM_DIR", raising=False)
    # epic/ ships empty, so the picker falls back to chill/soft_pad.mp3.
    # pick_bgm runs first: on a fresh clone it generates the track, and
    # only then does list_bgm see it.
    picked = bgm.pick_bgm("epic")
    assert picked.endswith("soft_pad.mp3")
    assert bgm.list_bgm("chill") == [bgm.pick_bgm("chill")]


def test_builtin_track_is_a_real_60s_mp3(monkeypatch):
    monkeypatch.delenv("BGM_DIR", raising=False)
    track = Path(bgm.pick_bgm("chill"))
    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "error",
         "-show_entries", "stream=codec_type,codec_name",
         "-show_entries", "format=duration", "-of", "json", str(track)],
        capture_output=True, text=True, check=True, timeout=60)
    info = json.loads(probe.stdout)
    assert [s["codec_name"] for s in info["streams"]] == ["mp3"]
    assert abs(float(info["format"]["duration"]) - 60.0) < 0.5


# --------------------------------------------------- generation on first use

@needs_ffmpeg
def test_generate_builtin_track_creates_valid_60s_mp3(tmp_path):
    """The generator synthesises a real 60 s stereo MP3."""
    track = bgm.generate_builtin_track(bgm_dir=tmp_path)
    assert track is not None
    track_path = Path(track)
    assert track_path == tmp_path / "chill" / "soft_pad.mp3"
    assert track_path.is_file()
    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "error",
         "-show_entries", "stream=codec_name,channels",
         "-show_entries", "format=duration", "-of", "json", str(track_path)],
        capture_output=True, text=True, check=True, timeout=60)
    info = json.loads(probe.stdout)
    assert [s["codec_name"] for s in info["streams"]] == ["mp3"]
    assert info["streams"][0]["channels"] == 2
    assert abs(float(info["format"]["duration"]) - 60.0) < 0.5


@needs_ffmpeg
def test_generate_builtin_track_leaves_existing_file_alone(tmp_path):
    first = bgm.generate_builtin_track(bgm_dir=tmp_path)
    mtime = Path(first).stat().st_mtime
    second = bgm.generate_builtin_track(bgm_dir=tmp_path)
    assert second == first
    assert Path(second).stat().st_mtime == mtime


@needs_ffmpeg
def test_pick_bgm_generates_builtin_when_missing(tmp_path, monkeypatch):
    """With an empty default pool, pick_bgm synthesises the pad."""
    monkeypatch.delenv("BGM_DIR", raising=False)
    monkeypatch.setattr(bgm, "_bgm_base_dir", lambda bgm_dir=None: tmp_path)
    picked = bgm.pick_bgm("chill")
    assert picked is not None
    assert Path(picked) == tmp_path / "chill" / "soft_pad.mp3"
    assert Path(picked).is_file()


def test_pick_bgm_does_not_generate_for_custom_pool(tmp_path):
    """A custom pool location is never written to: no MP3s means None."""
    assert bgm.pick_bgm("chill", bgm_dir=tmp_path) is None
    assert not (tmp_path / "chill" / "soft_pad.mp3").exists()


# ---------------------------------------------------------------- soft volume

def test_bgm_base_volume_default_is_soft():
    assert bgm.BGM_BASE_VOLUME == 0.12


def test_ducking_filter_default_volume_is_soft():
    filt, _ = bgm.build_ducking_filter()
    assert "volume=0.12[bgm]" in filt
    # Sidechain ducking itself is unchanged.
    assert "sidechaincompress=threshold=0.08:ratio=5.0:attack=100:release=1000" in filt


def test_bgm_base_volume_env_override(monkeypatch):
    monkeypatch.setenv("BGM_BASE_VOLUME", "0.2")
    assert bgm._env_volume("BGM_BASE_VOLUME", 0.12) == 0.2
    monkeypatch.setenv("BGM_BASE_VOLUME", "not-a-number")
    assert bgm._env_volume("BGM_BASE_VOLUME", 0.12) == 0.12
    monkeypatch.setenv("BGM_BASE_VOLUME", "0")
    assert bgm._env_volume("BGM_BASE_VOLUME", 0.12) == 0.12


# -------------------------------------------------------------- real rendering

@needs_ffmpeg
def test_real_render_with_bgm_has_audio_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)
    monkeypatch.delenv("BGM_DIR", raising=False)
    src = tmp_path / "src.mp4"
    _make_speech_then_silence(src)

    editor = clip_editor.ClipEditor()
    out = tmp_path / "out_bgm.mp4"
    result = editor.edit_clip(
        str(src), str(out), streamer="xqc", layout="center_crop",
        hook_banner="TEST BANNER", bgm=True, generate_thumbnail=False)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "error",
         "-show_entries", "stream=codec_type,codec_name,width,height",
         "-show_entries", "format=duration", "-of", "json", str(out)],
        capture_output=True, text=True, check=True, timeout=60)
    info = json.loads(probe.stdout)
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(video) == 1
    assert video[0]["width"] == 1080 and video[0]["height"] == 1920
    assert len(audio) == 1  # ducked speech + music mix
    assert abs(float(info["format"]["duration"]) - 4.0) < 0.5


@needs_ffmpeg
def test_bgm_mix_is_quiet_under_speech(tmp_path, monkeypatch):
    """The music is audible in the speech-free tail, but far quieter
    than the speech — and the tail is silent without BGM."""
    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)
    monkeypatch.delenv("BGM_DIR", raising=False)
    src = tmp_path / "src.mp4"
    _make_speech_then_silence(src)

    editor = clip_editor.ClipEditor()
    out_bgm = tmp_path / "out_bgm.mp4"
    out_plain = tmp_path / "out_plain.mp4"
    editor.edit_clip(str(src), str(out_bgm), streamer="xqc",
                     layout="center_crop", hook_banner="TEST BANNER",
                     bgm=True, generate_thumbnail=False)
    editor.edit_clip(str(src), str(out_plain), streamer="xqc",
                     layout="center_crop", hook_banner="TEST BANNER",
                     bgm=False, generate_thumbnail=False)

    # The tail starts at 2.5 s: the frame straddling the 2 s cut still
    # carries a few loud samples, so 2.0-2.5 s is not clean silence.
    speech_db = _segment_mean_db(out_bgm, 0, 2)
    music_db = _segment_mean_db(out_bgm, 2.5, 4)
    plain_tail_db = _segment_mean_db(out_plain, 2.5, 4)

    # Without BGM the tail is (near-)silence.
    assert plain_tail_db < -60, f"plain tail not silent: {plain_tail_db} dB"
    # With BGM the music is present in the tail...
    assert music_db > -60, f"music inaudible in tail: {music_db} dB"
    # ...but far quieter than the speech (at least 10 dB under it).
    assert music_db < speech_db - 10, (
        f"music too loud: speech {speech_db} dB, music {music_db} dB")


@needs_ffmpeg
def test_bgm_does_not_quiet_original_audio(tmp_path, monkeypatch):
    """BGM on must not make the original audio quieter: amix divides
    by the input count by default, which dropped the speech ~6 dB in
    the first demo run (full-clip mean -29.7 dB without BGM vs -34.0 dB
    with it). normalize=0 keeps the speech at its level."""
    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)
    monkeypatch.delenv("BGM_DIR", raising=False)
    src = tmp_path / "src.mp4"
    _make_speech_then_silence(src)

    editor = clip_editor.ClipEditor()
    out_bgm = tmp_path / "out_bgm.mp4"
    out_plain = tmp_path / "out_plain.mp4"
    editor.edit_clip(str(src), str(out_bgm), streamer="xqc",
                     layout="center_crop", hook_banner="TEST BANNER",
                     bgm=True, generate_thumbnail=False)
    editor.edit_clip(str(src), str(out_plain), streamer="xqc",
                     layout="center_crop", hook_banner="TEST BANNER",
                     bgm=False, generate_thumbnail=False)

    speech_bgm = _segment_mean_db(out_bgm, 0, 2)
    speech_plain = _segment_mean_db(out_plain, 0, 2)
    # The voice is mixed at 1.2 (+1.6 dB) plus a little ducked music,
    # so the BGM speech segment is slightly louder, never quieter.
    assert speech_bgm >= speech_plain - 1.0, (
        f"BGM quieted the speech: {speech_plain} dB plain vs "
        f"{speech_bgm} dB with BGM")
    assert speech_bgm <= speech_plain + 3.0, (
        f"BGM boosted the speech too much: {speech_plain} dB plain vs "
        f"{speech_bgm} dB with BGM")
