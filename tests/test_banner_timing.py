"""Tests for the 4-second hook banner rule for Kick live-stream clips.

Sara's rule: the white hook banner text shows for the first 4 seconds
of a clip only, then fades out over 0.5 s and stays hidden for the
rest of the clip. The Kick watermark bar at the bottom is branding,
not the banner, and stays visible throughout. The rule applies to
every layout (classic, face-view, camera-switch) because the banner
is composited in one place, after the layout transform.
"""

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

import clip_editor  # noqa: E402


@pytest.fixture
def editor_setup(tmp_path, monkeypatch):
    """ClipEditor with ffmpeg, logging and watermark stubbed out
    (same shape as tests/test_subtitles.py::editor_setup)."""
    monkeypatch.setattr(clip_editor, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)

    watermark = tmp_path / "wm.png"
    from PIL import Image
    Image.new("RGBA", (1082, 108), (0, 0, 0, 255)).save(watermark)
    monkeypatch.setattr(clip_editor, "generate_kick_watermark",
                        lambda streamer: str(watermark))

    ffmpeg_calls = []

    def fake_run(cmd, capture_output=True, text=True, creationflags=0, timeout=180):
        ffmpeg_calls.append(list(cmd))
        if str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(clip_editor.subprocess, "run", fake_run)

    editor = clip_editor.ClipEditor()
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    return editor, raw, tmp_path, ffmpeg_calls


def _render_filter(ffmpeg_calls):
    renders = [c for c in ffmpeg_calls if "-filter_complex" in c]
    assert len(renders) == 1
    return " ".join(renders[0])


def test_banner_duration_constants():
    assert clip_editor.BANNER_DURATION_SECONDS == 4.0
    assert clip_editor.BANNER_FADE_SECONDS == 0.5


def test_banner_fades_out_after_four_seconds(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    # Fade-out starts at 4 s, lasts 0.5 s, on the alpha channel.
    assert "fade=t=out:st=4.00:d=0.50:alpha=1" in filter_str
    # Overlay is enabled only for the 0-4.5 s window.
    assert "overlay=0:0:enable='between(t,0,4.50)'" in filter_str
    # The still card image is looped so the fade has a running timeline.
    assert "loop=loop=600:size=1:start=0" in filter_str
    # ... and the loop is finite: loop=-1 stalls the filter graph.
    assert "loop=loop=-1" not in filter_str
    # The card stream is trimmed to the banner window: an untrimmed
    # loop would stretch the output to the loop length.
    assert "trim=0:4.50,setpts=PTS-STARTPTS" in filter_str


def test_kick_bar_stays_visible_throughout(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    # The watermark overlay has no enable window: visible all clip long.
    assert "overlay=0:1420[with_wm]" in filter_str
    # The 4.5 s enable window appears exactly once: on the banner only.
    assert filter_str.count("enable='between(t,0,4.50)'") == 1


def test_banner_timing_applies_in_other_layouts(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="center_crop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "fade=t=out:st=4.00:d=0.50:alpha=1" in filter_str
    assert "overlay=0:0:enable='between(t,0,4.50)'" in filter_str


def test_no_banner_means_no_banner_filter(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "card_faded" not in filter_str
    assert "fade=t=out" not in filter_str
    # The Kick bar is still there.
    assert "[with_wm]" in filter_str


# ---------------------------------------------------------------------------
# FEATURE 7: headline timing setting ("4s" vs "whole_clip")
# ---------------------------------------------------------------------------

def test_whole_clip_mode_has_no_fade_and_no_enable_window(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     banner_timing="whole_clip", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    # No fade-out on the banner at all.
    assert "fade=t=out" not in filter_str
    # No 4.5 s enable window: the overlay has no enable clause.
    assert "between(t,0,4.50)" not in filter_str
    assert "overlay=0:0[with_card]" in filter_str
    # The stubbed ffprobe returns no duration, so the card falls back
    # to the generous 120 s cap (loop count scales with it).
    assert "trim=0:120.00,setpts=PTS-STARTPTS" in filter_str
    assert "loop=loop=3660:size=1:start=0" in filter_str
    assert "loop=loop=-1" not in filter_str


def test_whole_clip_mode_trims_card_to_probed_duration(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    monkeypatch.setattr(clip_editor, "_probe_duration", lambda path: 30.0)
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     banner_timing="whole_clip", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "trim=0:30.00,setpts=PTS-STARTPTS" in filter_str
    assert "loop=loop=960:size=1:start=0" in filter_str
    assert "fade=t=out" not in filter_str


def test_4s_mode_explicit_param_unchanged(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     banner_timing="4s", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "fade=t=out:st=4.00:d=0.50:alpha=1" in filter_str
    assert "overlay=0:0:enable='between(t,0,4.50)'" in filter_str
    assert "trim=0:4.50,setpts=PTS-STARTPTS" in filter_str


def test_banner_timing_invalid_value_falls_back_to_4s(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     banner_timing="forever", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "fade=t=out:st=4.00:d=0.50:alpha=1" in filter_str
    assert "overlay=0:0:enable='between(t,0,4.50)'" in filter_str


def test_banner_timing_defaults_to_config(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    monkeypatch.setattr(clip_editor, "BANNER_TIMING", "whole_clip")
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "fade=t=out" not in filter_str
    assert "overlay=0:0[with_card]" in filter_str


def test_banner_timing_param_beats_config(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    monkeypatch.setattr(clip_editor, "BANNER_TIMING", "whole_clip")
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     banner_timing="4s", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "fade=t=out:st=4.00:d=0.50:alpha=1" in filter_str


# ---------------------------------------------------------------------------
# FEATURE 2: the quote line follows the banner timing
# ---------------------------------------------------------------------------

def test_quote_line_follows_4s_timing(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     quote_line_enabled=True,
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    # Second card present, with its own fade and the same 4.5 s window.
    assert "[quote_faded]" in filter_str
    assert "[with_quote]" in filter_str
    assert filter_str.count("fade=t=out:st=4.00:d=0.50:alpha=1") == 2
    assert filter_str.count("enable='between(t,0,4.50)'") == 2


def test_quote_line_follows_whole_clip_timing(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     quote_line_enabled=True,
                     banner_timing="whole_clip", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "[quote]" in filter_str
    assert "[with_quote]" in filter_str
    assert "overlay=0:0[with_quote]" in filter_str
    assert "fade=t=out" not in filter_str
    assert "between(t,0,4.50)" not in filter_str


def test_quote_line_empty_means_no_quote_filter(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "[with_quote]" not in filter_str
    assert "temp_quote_" not in filter_str
    # The headline is unaffected.
    assert "[with_card]" in filter_str


def test_quote_line_disabled_by_config(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    monkeypatch.setattr(clip_editor, "QUOTE_LINE_ENABLED", False)
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "[with_quote]" not in filter_str
    assert "temp_quote_" not in filter_str


# ---------------------------------------------------------------------------
# Loop ending: no fade-out on the main video/audio streams
# ---------------------------------------------------------------------------

def test_no_fade_out_on_main_streams(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     quote_line_enabled=True,
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    # No audio fade-out anywhere: the clip ends abruptly and loops.
    assert "afade" not in filter_str
    # Every video fade-out belongs to a still-image card input
    # (headline / quote movie=... stream), never to the base chain.
    for segment in filter_str.split(";"):
        if "fade=t=out" in segment:
            assert "movie=" in segment


def test_no_fade_out_in_whole_clip_mode(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     quote_line_enabled=True,
                     banner_timing="whole_clip", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "fade=t=out" not in filter_str
    assert "afade" not in filter_str


# ---------------------------------------------------------------------------
# Render timeout scales with clip duration (clip #12 fix)
# ---------------------------------------------------------------------------

def test_render_timeout_seconds():
    assert clip_editor.render_timeout_seconds(None) == 180
    assert clip_editor.render_timeout_seconds(0) == 180
    assert clip_editor.render_timeout_seconds(5.0) == 180   # floor
    assert clip_editor.render_timeout_seconds(9.0) == 180   # 9*20 = 180
    assert clip_editor.render_timeout_seconds(30.0) == 600  # 30*20
    assert clip_editor.render_timeout_seconds(45.0) == 900  # 45*20
    assert clip_editor.render_timeout_seconds(100.0) == 1200  # capped


def test_render_timeout_scales_with_clip_duration(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    calls = []

    def recording_run(cmd, capture_output=True, text=True, creationflags=0,
                      timeout=180):
        calls.append((list(cmd), timeout))
        if str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(clip_editor.subprocess, "run", recording_run)
    monkeypatch.setattr(clip_editor, "_probe_duration", lambda path: 30.0)
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Test banner", generate_thumbnail=False)
    renders = [t for cmd, t in calls if "-filter_complex" in cmd]
    assert len(renders) == 1
    assert renders[0] == 600  # 30 s clip -> max(180, 30*20)


def test_render_timeout_falls_back_to_180_without_duration(editor_setup, monkeypatch):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    calls = []

    def recording_run(cmd, capture_output=True, text=True, creationflags=0,
                      timeout=180):
        calls.append((list(cmd), timeout))
        if str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(clip_editor.subprocess, "run", recording_run)
    monkeypatch.setattr(clip_editor, "_probe_duration", lambda path: None)
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Test banner", generate_thumbnail=False)
    renders = [t for cmd, t in calls if "-filter_complex" in cmd]
    assert len(renders) == 1
    assert renders[0] == 180
