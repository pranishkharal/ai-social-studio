"""Tests for the no-subtitles-by-default rule for Kick live-stream clips.

Kick clips are rendered WITHOUT burned-in subtitles unless subtitles are
explicitly enabled (pipeline subtitles=True, the dashboard's
auto-captions checkbox, or ENABLE_SUBTITLES=true). Whisper still
transcribes every clip — silence cutting needs the word timestamps —
the words just are not drawn on the video. Karaoke style only applies
when subtitles are enabled.
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
import pipeline as pipeline_module  # noqa: E402
import thumbnail  # noqa: E402
from pipeline import ClipPipeline  # noqa: E402

WORDS = [
    {"word": "what", "start": 0.0, "end": 0.4},
    {"word": "just", "start": 0.4, "end": 0.7},
    {"word": "happened", "start": 0.7, "end": 1.2},
]

MINIMAL_ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,Impact,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,1,0,0,0,100,100,2,0,1,6,4,2,60,60,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,TikTok,,0,0,0,,WHAT JUST HAPPENED
"""


# ---------------------------------------------------------------------------
# Editor level: the ffmpeg filter is the source of truth.
# ---------------------------------------------------------------------------

@pytest.fixture
def editor_setup(tmp_path, monkeypatch):
    """ClipEditor with ffmpeg, logging and watermark stubbed out
    (same shape as tests/test_thumbnail.py::editor_setup)."""
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


def test_edit_clip_default_has_no_subtitle_filter(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="center_crop", generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "subtitles=" not in filter_str
    assert ".ass" not in filter_str


def test_edit_clip_opt_in_burns_subtitles(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    ass_path = tmp_path / "clip.ass"
    ass_path.write_text(MINIMAL_ASS, encoding="utf-8")
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="center_crop", ass_subtitle_path=str(ass_path),
                     auto_captions=True, generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "subtitles=" in filter_str
    assert "clip.ass" in filter_str


def test_edit_clip_opt_in_without_ass_file_burns_nothing(editor_setup):
    editor, raw, tmp_path, ffmpeg_calls = editor_setup
    out = tmp_path / "edited.mp4"
    editor.edit_clip(str(raw), str(out), streamer="adinross",
                     layout="center_crop", auto_captions=True,
                     generate_thumbnail=False)
    filter_str = _render_filter(ffmpeg_calls)
    assert "subtitles=" not in filter_str


# ---------------------------------------------------------------------------
# Pipeline level: subtitles= / auto_captions= / ENABLE_SUBTITLES wiring.
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_setup(tmp_path, monkeypatch, fresh_db):
    """ClipPipeline with the editor captured and side-effect dirs isolated."""
    monkeypatch.setattr(pipeline_module, "SUBTITLES_DIR", tmp_path / "subtitles")
    monkeypatch.setattr(pipeline_module, "EDITED_DIR", tmp_path / "edited")
    monkeypatch.setattr(pipeline_module, "SUBTITLES_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "SUBTITLE_STYLE", "classic")
    (tmp_path / "subtitles").mkdir()
    (tmp_path / "edited").mkdir()

    pipe = ClipPipeline()
    monkeypatch.setattr(pipe, "_mark", lambda *a, **k: None)
    monkeypatch.setattr(
        pipe.ai, "transcribe_clip",
        lambda path: {"full_text": "what just happened", "words": list(WORDS)})

    captured = {}

    def fake_edit_clip(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "edited" / "out.mp4")

    monkeypatch.setattr(pipe.editor, "edit_clip", fake_edit_clip)

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    return pipe, raw, tmp_path, captured


def test_process_moment_default_has_no_subtitles(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw))
    assert result["status"] == "ready_to_upload"
    assert captured["auto_captions"] is False
    assert captured["ass_subtitle_path"] is None
    # No .ass file was generated either.
    assert list((tmp_path / "subtitles").glob("*.ass")) == []


def test_process_moment_subtitles_opt_in(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw),
                                 subtitles=True)
    assert result["status"] == "ready_to_upload"
    assert captured["auto_captions"] is True
    ass_path = captured["ass_subtitle_path"]
    assert ass_path is not None
    assert Path(ass_path).exists()
    assert "WHAT JUST HAPPENED" in Path(ass_path).read_text(encoding="utf-8")


def test_process_moment_auto_captions_alias_still_works(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw),
                        auto_captions=True)
    assert captured["auto_captions"] is True
    assert captured["ass_subtitle_path"] is not None


def test_process_moment_env_opt_in(pipeline_setup, monkeypatch):
    pipe, raw, tmp_path, captured = pipeline_setup
    monkeypatch.setattr(pipeline_module, "SUBTITLES_ENABLED", True)
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw))
    assert captured["auto_captions"] is True
    assert captured["ass_subtitle_path"] is not None


def test_process_moment_karaoke_only_when_subtitles_enabled(pipeline_setup,
                                                            monkeypatch):
    pipe, raw, tmp_path, captured = pipeline_setup
    monkeypatch.setattr(pipeline_module, "SUBTITLE_STYLE", "karaoke")

    # Karaoke style + subtitles OFF: no ASS file at all.
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw))
    assert captured["ass_subtitle_path"] is None
    assert list((tmp_path / "subtitles").glob("*.ass")) == []

    # Karaoke style + subtitles ON: karaoke ASS with \k timing tags.
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw),
                        subtitles=True)
    ass_path = captured["ass_subtitle_path"]
    assert ass_path is not None
    assert r"\k" in Path(ass_path).read_text(encoding="utf-8")
