"""Tests for karaoke_subtitles.py (no ffmpeg/MediaPipe needed)."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karaoke_subtitles import generate_karaoke_ass, STYLES  # noqa: E402

WORDS = [
    {"word": "hello", "start": 0.0, "end": 0.4},
    {"word": "world", "start": 0.4, "end": 0.9},
    {"word": "this", "start": 1.0, "end": 1.3},
    {"word": "is", "start": 1.3, "end": 1.5},
    {"word": "great", "start": 1.6, "end": 2.1},
]


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def test_header_targets_1080x1920_canvas(tmp_path):
    out = generate_karaoke_ass(WORDS, str(tmp_path / "clip.ass"))
    text = _read(out)
    assert "PlayResX: 1080" in text
    assert "PlayResY: 1920" in text
    assert "ScriptType: v4.00+" in text
    assert "[V4+ Styles]" in text
    assert "[Events]" in text


def test_karaoke_tags_highlight_each_word_as_spoken(tmp_path):
    out = generate_karaoke_ass(WORDS, str(tmp_path / "clip.ass"))
    text = _read(out)
    # First line: hello (0.0->0.4) = 40cs, world (0.4->1.0) = 60cs,
    # this (1.0->1.3) = 30cs.
    assert r"{\k40}HELLO" in text
    assert r"{\k60}WORLD" in text
    assert r"{\k30}THIS" in text
    # Second line: is (1.3->1.6) = 30cs, great (1.6->2.1) = 50cs.
    assert r"{\k30}IS" in text
    assert r"{\k50}GREAT" in text


def test_words_chunked_three_per_line(tmp_path):
    out = generate_karaoke_ass(WORDS, str(tmp_path / "clip.ass"))
    text = _read(out)
    dialogues = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogues) == 2
    assert dialogues[0].startswith("Dialogue: 0,0:00:00.00,0:00:01.38,")
    assert dialogues[1].startswith("Dialogue: 0,0:00:01.30,0:00:02.18,")


def test_long_pause_breaks_the_line(tmp_path):
    words = [
        {"word": "one", "start": 0.0, "end": 0.3},
        {"word": "two", "start": 2.0, "end": 2.3},  # 1.7 s gap
    ]
    out = generate_karaoke_ass(words, str(tmp_path / "clip.ass"))
    text = _read(out)
    dialogues = [ln for ln in text.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogues) == 2


def test_hormozi_style_yellow_highlight_and_bounce(tmp_path):
    out = generate_karaoke_ass(WORDS, str(tmp_path / "clip.ass"), style="hormozi")
    text = _read(out)
    assert "Style: Karaoke,Impact,64,&H0000FFFF,&H00FFFFFF" in text
    assert r"\t(0,90,\fscx112\fscy112)" in text  # bounce pop on first word
    assert "HELLO" in text  # uppercased


def test_minimal_style_white_no_bounce(tmp_path):
    out = generate_karaoke_ass(WORDS, str(tmp_path / "clip.ass"), style="minimal")
    text = _read(out)
    assert "Style: Karaoke,Arial,56,&H00FFFFFF,&H00D8D8D8" in text
    assert "\\t(" not in text
    assert "hello" in text  # original case kept
    assert r"{\k40}hello" in text  # karaoke sweep still present


def test_unknown_style_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown karaoke style"):
        generate_karaoke_ass(WORDS, str(tmp_path / "clip.ass"), style="neon")


def test_empty_words_writes_header_only(tmp_path):
    out = generate_karaoke_ass([], str(tmp_path / "clip.ass"))
    text = _read(out)
    assert "[Events]" in text
    assert "Dialogue:" not in text


def test_special_characters_are_escaped(tmp_path):
    words = [{"word": "a{b}c\\d", "start": 0.0, "end": 0.5}]
    out = generate_karaoke_ass(words, str(tmp_path / "clip.ass"))
    text = _read(out)
    assert r"A\{B\}C\\D" in text


def test_creates_parent_directories(tmp_path):
    out = generate_karaoke_ass(WORDS, str(tmp_path / "sub" / "deep" / "clip.ass"))
    assert Path(out).exists()


def test_all_styles_advertised_are_valid(tmp_path):
    for style in STYLES:
        out = generate_karaoke_ass(WORDS, str(tmp_path / f"{style}.ass"), style=style)
        assert Path(out).exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_libass_accepts_the_file(tmp_path):
    """Real check: ffmpeg's subtitles filter must burn the .ass without error."""
    ass = generate_karaoke_ass(WORDS, str(tmp_path / "clip.ass"))
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30:d=3",
         "-vf", f"subtitles='{ass}'",
         "-frames:v", "30", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stderr
