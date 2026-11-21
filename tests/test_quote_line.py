"""Tests for the quote line (FEATURE 2): one static spoken line under
the headline, white bold text with a black outline in the lower third.

The quote is a single caption, NOT word-by-word subtitles — subtitles
stay off by default. It follows the same timing as the headline (see
tests/test_banner_timing.py for the timing assertions); these tests
cover the card itself and its wiring into edit_clip.
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
    (same shape as tests/test_banner_timing.py::editor_setup)."""
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


def _alpha_bbox(path):
    """Bounding box of the non-transparent pixels of a PNG."""
    from PIL import Image
    img = Image.open(path)
    assert img.mode == "RGBA"
    assert img.size == (1080, 1920)
    return img.split()[3].getbbox()


# ---------------------------------------------------------------- card itself

def test_generate_quote_card_transparent_png_lower_third(tmp_path):
    editor = clip_editor.ClipEditor()
    out = tmp_path / "quote.png"
    assert editor.generate_quote_card("Bro, I cannot believe this", str(out)) == str(out)
    assert out.exists()

    bbox = _alpha_bbox(out)
    assert bbox is not None, "quote card has no visible pixels"
    _left, top, _right, bottom = bbox
    # Visible text sits in the lower third (y 1180-1300), above the
    # Kick bar at y 1420 — with a small margin for font metrics.
    assert top >= 1150, f"text starts too high: y={top}"
    assert bottom <= 1350, f"text ends too low: y={bottom}"

    # The text is white bold with a black outline: plenty of near-white
    # pixels (letter interiors), and no white box behind it — the card
    # is transparent everywhere except the lettering.
    from PIL import Image
    img = Image.open(out)
    pixels = img.load()
    white = 0
    visible = 0
    for y in range(bbox[1], bbox[3]):
        for x in range(bbox[0], bbox[2]):
            r, g, b, a = pixels[x, y]
            if a > 128:
                visible += 1
                if r > 200 and g > 200 and b > 200:
                    white += 1
    assert visible > 100
    assert white > 50, "no white text pixels found"


def test_generate_quote_card_nothing_above_lower_third(tmp_path):
    editor = clip_editor.ClipEditor()
    out = tmp_path / "quote.png"
    editor.generate_quote_card("Short quote", str(out))
    from PIL import Image
    alpha = Image.open(out).split()[3]
    # Everything above y 1000 is fully transparent.
    assert alpha.crop((0, 0, 1080, 1000)).getbbox() is None


def test_generate_quote_card_long_text_wraps_to_two_lines(tmp_path):
    editor = clip_editor.ClipEditor()
    out = tmp_path / "quote.png"
    long_text = ("This is a very long spoken line that would never fit on "
                 "one line and keeps going until it has to be cut short")
    editor.generate_quote_card(long_text, str(out))
    bbox = _alpha_bbox(out)
    assert bbox is not None
    _left, top, _right, bottom = bbox
    # At most 2 lines of 56 px: the text block stays compact and inside
    # the lower third instead of spilling up or down the frame.
    assert bottom - top <= 140, f"text block too tall: {bottom - top}px"
    assert top >= 1150
    assert bottom <= 1350


# ---------------------------------------------------------------- edit_clip wiring

def test_edit_clip_with_quote_line_adds_second_overlay(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     quote_line_enabled=True,
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "temp_quote_" in filter_str
    assert "[with_quote]" in filter_str
    # The quote overlay is applied after the headline overlay.
    assert filter_str.index("[with_card]") < filter_str.index("[with_quote]")
    # The Kick bar and the headline are still there.
    assert "[with_wm]" in filter_str
    assert "[with_card]" in filter_str


def test_edit_clip_quote_without_headline_still_renders(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     quote_line="Bro, I literally cannot believe this",
                     quote_line_enabled=True,
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "[with_quote]" in filter_str
    assert "[with_card]" not in filter_str


def test_quote_temp_png_cleaned_up(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     quote_line_enabled=True,
                     generate_thumbnail=False)
    assert list(tmp_path.glob("temp_quote_*.png")) == []
    assert list(tmp_path.glob("temp_card_*.png")) == []


# ---------------------------------------------------------------- off by default (v1.4.1)

def test_quote_line_off_by_default(editor_setup, monkeypatch):
    """A quote line is passed, but nothing opts in: no quote overlay
    is drawn, while the headline and the Kick bar are untouched."""
    monkeypatch.setattr(clip_editor, "QUOTE_LINE_ENABLED", False)
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "[with_quote]" not in filter_str
    assert "temp_quote_" not in filter_str
    assert "[with_card]" in filter_str
    assert "[with_wm]" in filter_str


def test_quote_line_enabled_by_config(editor_setup, monkeypatch):
    """QUOTE_LINE_ENABLED=true opts in without a per-call argument."""
    monkeypatch.setattr(clip_editor, "QUOTE_LINE_ENABLED", True)
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "[with_quote]" in filter_str


def test_quote_line_explicit_false_beats_config(editor_setup, monkeypatch):
    """An explicit quote_line_enabled=False wins over config True."""
    monkeypatch.setattr(clip_editor, "QUOTE_LINE_ENABLED", True)
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="blurred_backdrop",
                     hook_banner="Adinross was COMPLETELY SPEECHLESS",
                     quote_line="Bro, I literally cannot believe this",
                     quote_line_enabled=False,
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "[with_quote]" not in filter_str
    assert "temp_quote_" not in filter_str


def test_quote_line_config_default_is_off(monkeypatch):
    """The shipped config default is off: with no env var set, a
    fresh config read gives QUOTE_LINE_ENABLED == False."""
    import importlib

    import config

    monkeypatch.delenv("QUOTE_LINE_ENABLED", raising=False)
    importlib.reload(config)
    try:
        assert config.QUOTE_LINE_ENABLED is False
    finally:
        importlib.reload(config)
