"""Tests for voiceover.py (gta6-clipper feature 7) + the dashboard
upload endpoint in gta6_routers.py.

Media is generated locally with ffmpeg lavfi (no network, no fixtures
from disk): a 6 s test clip with a sine-wave soundtrack and a 2 s
voice line.
"""

import base64
import inspect
import subprocess

import pytest

import voiceover


def _make_clip(path, seconds=6):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x180:rate=10",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)],
        check=True, timeout=60)
    return path


def _make_voice(path, seconds=2, freq=880):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-c:a", "pcm_s16le", str(path)],
        check=True, timeout=60)
    return path


def _make_silent_video(path, seconds=3):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x180:rate=10",
         "-c:v", "libx264", str(path)],
        check=True, timeout=60)
    return path


def test_probe_duration_and_has_audio(tmp_path):
    clip = _make_clip(tmp_path / "clip.mp4")
    assert 5.5 < voiceover.probe_duration(clip) < 6.5
    assert voiceover.has_audio(clip)
    silent = _make_silent_video(tmp_path / "silent.mp4")
    assert not voiceover.has_audio(silent)
    assert voiceover.probe_duration(tmp_path / "missing.mp4") == 0.0


def test_build_voiceover_filter_ducks_and_trims():
    graph = voiceover.build_voiceover_filter(30.0, 5.0, True)
    assert "sidechaincompress" in graph
    assert "adelay=5000|5000" in graph
    assert "atrim=end=25.00" in graph  # too-long line trimmed at the end
    assert "loudnorm" in graph
    # A clip with no audio gets a silence source instead of [0:a].
    graph_silent = voiceover.build_voiceover_filter(30.0, 5.0, False)
    assert "anullsrc" in graph_silent


def test_add_voiceover_real_mix(tmp_path):
    clip = _make_clip(tmp_path / "clip.mp4")
    line = _make_voice(tmp_path / "line.wav")
    out = tmp_path / "voiced.mp4"
    result = voiceover.add_voiceover(clip, line, 2.0, out)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 1000
    # Same duration as the source (the line is mixed in, not appended).
    assert abs(voiceover.probe_duration(out) - 6.0) < 0.5
    assert voiceover.has_audio(out)


def test_add_voiceover_trims_long_line(tmp_path):
    clip = _make_clip(tmp_path / "clip.mp4", seconds=6)
    long_line = _make_voice(tmp_path / "long.wav", seconds=10)
    out = tmp_path / "voiced.mp4"
    voiceover.add_voiceover(clip, long_line, 4.0, out)
    assert out.exists()
    # The clip is NOT stretched by the 10 s line starting at 4 s.
    assert abs(voiceover.probe_duration(out) - 6.0) < 0.5


def test_add_voiceover_rejects_silent_recording(tmp_path):
    clip = _make_clip(tmp_path / "clip.mp4")
    silent = _make_silent_video(tmp_path / "silent.mp4")
    with pytest.raises(ValueError, match="no sound"):
        voiceover.add_voiceover(clip, silent, 1.0, tmp_path / "out.mp4")


def test_add_voiceover_rejects_start_past_end(tmp_path):
    clip = _make_clip(tmp_path / "clip.mp4", seconds=6)
    line = _make_voice(tmp_path / "line.wav")
    with pytest.raises(ValueError, match="only 6.0 s long"):
        voiceover.add_voiceover(clip, line, 10.0, tmp_path / "out.mp4")


def test_add_voiceover_missing_files(tmp_path):
    line = _make_voice(tmp_path / "line.wav")
    with pytest.raises(FileNotFoundError):
        voiceover.add_voiceover(tmp_path / "nope.mp4", line, 1.0, tmp_path / "o.mp4")
    clip = _make_clip(tmp_path / "clip.mp4")
    with pytest.raises(FileNotFoundError):
        voiceover.add_voiceover(clip, tmp_path / "nope.wav", 1.0, tmp_path / "o.mp4")


def test_edit_clip_accepts_voiceover_params():
    """clip_editor.edit_clip gained voiceover_path / voiceover_at."""
    import clip_editor
    params = inspect.signature(clip_editor.ClipEditor.edit_clip).parameters
    assert "voiceover_path" in params
    assert "voiceover_at" in params
    assert params["voiceover_at"].default == 0.0


def test_voiceover_endpoint(tmp_path, fresh_db, monkeypatch):
    """POST /api/voiceover mixes the line into the clip's edited file
    and updates the DB row."""
    import gta6_routers
    clip = _make_clip(tmp_path / "edited.mp4")
    clip_id = fresh_db["seed_clip"](edited_path=str(clip))
    line = _make_voice(tmp_path / "line.wav")
    req = gta6_routers.VoiceoverUpload(
        clip_id=clip_id, voiceover_at=2.0,
        audio_base64=base64.b64encode(line.read_bytes()).decode(),
        filename="line.wav")
    result = gta6_routers.upload_voiceover(req)
    assert result["status"] == "ok"
    assert result["edited_path"].endswith("_vo.mp4")
    assert voiceover.has_audio(result["edited_path"])
    import database
    conn = database.get_connection()
    row = conn.execute("SELECT edited_path FROM clips WHERE id = ?",
                       (clip_id,)).fetchone()
    conn.close()
    assert row["edited_path"] == result["edited_path"]


def test_voiceover_endpoint_rejects_bad_audio(tmp_path, fresh_db):
    import gta6_routers
    from fastapi import HTTPException
    clip = _make_clip(tmp_path / "edited.mp4")
    clip_id = fresh_db["seed_clip"](edited_path=str(clip))
    req = gta6_routers.VoiceoverUpload(
        clip_id=clip_id, voiceover_at=1.0,
        audio_base64=base64.b64encode(b"not audio").decode(),
        filename="line.wav")
    with pytest.raises(HTTPException) as exc:
        gta6_routers.upload_voiceover(req)
    assert exc.value.status_code == 400
