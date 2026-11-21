"""Integration tests for the TikTok clipping upgrades (v1.2.0) wiring.

These cover the seams the feature workers did not own: the pipeline
passing the quote line / banner timing / BGM / game through to the
editor, the short-clip peak cut, the quote line stored in the database
and reused on re-render, and the game hashtag on the post caption.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline as pipeline_module  # noqa: E402
from pipeline import ClipPipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# The real adinross transcript from the 2026-11-19 live run (clip #11):
# the story-headline builder turns it into a specific headline and the
# quote picker finds a punchy line in it.
CLIP_11_TRANSCRIPT = json.loads(
    (REPO_ROOT / "checkpoints" / "clip_11.json").read_text()
)["steps"]["transcribed"]["data"]["transcript"]


@pytest.fixture
def pipeline_setup(tmp_path, monkeypatch, fresh_db):
    """ClipPipeline with the editor captured and side-effect dirs isolated
    (same shape as tests/test_subtitles.py::pipeline_setup)."""
    monkeypatch.setattr(pipeline_module, "SUBTITLES_DIR", tmp_path / "subtitles")
    monkeypatch.setattr(pipeline_module, "EDITED_DIR", tmp_path / "edited")
    monkeypatch.setattr(pipeline_module, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(pipeline_module, "SUBTITLES_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "SUBTITLE_STYLE", "classic")
    monkeypatch.setattr(pipeline_module, "BGM_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "BROLL_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "QUOTE_LINE_ENABLED", False)
    (tmp_path / "subtitles").mkdir()
    (tmp_path / "edited").mkdir()
    (tmp_path / "clips").mkdir()

    pipe = ClipPipeline()
    monkeypatch.setattr(pipe, "_mark", lambda *a, **k: None)
    monkeypatch.setattr(
        pipe.ai, "transcribe_clip",
        lambda path, **kwargs: {"full_text": CLIP_11_TRANSCRIPT, "words": []})

    captured = {}

    def fake_edit_clip(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "edited" / "out.mp4")

    monkeypatch.setattr(pipe.editor, "edit_clip", fake_edit_clip)

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    return pipe, raw, tmp_path, captured


def _clip_row(clip_id):
    import database
    conn = database.get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?",
                       (clip_id,)).fetchone()
    conn.close()
    return dict(row)


def test_process_moment_passes_specific_headline_and_quote(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw))
    assert result["status"] == "ready_to_upload"
    # Story headline names the moment — not the generic SPEECHLESS line.
    assert result["title"] == "Adinross finds his calling — \"Chaz is not clean\""
    assert "COMPLETELY SPEECHLESS" not in result["title"]
    # The quote line reaches the editor and the result.
    assert captured["quote_line"]
    assert "badass" in captured["quote_line"]
    assert result["quote_line"] == captured["quote_line"]
    # ... and is stored in the database for re-render.
    assert _clip_row(result["clip_id"])["quote_line"] == captured["quote_line"]


def test_process_moment_banner_timing_passthrough(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw),
                        banner_timing="whole_clip")
    assert captured["banner_timing"] == "whole_clip"

    captured.clear()
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw))
    # None means "the BANNER_TIMING setting decides" in the editor.
    assert captured["banner_timing"] is None


def test_process_moment_bgm_passthrough(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw),
                        bgm=True)
    assert captured["bgm"] is True

    captured.clear()
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw))
    assert captured["bgm"] is False  # BGM_ENABLED=False in the fixture


def test_process_moment_game_adds_game_hashtag(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw), game="GTA 6")
    assert "#gta6" in result["tags"]
    assert "#adinross" in result["tags"]
    assert "#fyp" in result["tags"] and "#kick" in result["tags"]
    assert "#gta6" in result["caption"]

    result2 = pipe.process_moment("adinross", duration=10,
                                  raw_clip_path=str(raw))
    assert "#gta6" not in result2["tags"]
    assert result2["tags"] == "#adinross #fyp #kick"


def test_process_moment_short_clip_cuts_peak_window(pipeline_setup,
                                                    monkeypatch):
    pipe, raw, tmp_path, captured = pipeline_setup
    monkeypatch.setattr(
        pipe.detector, "find_peak_window",
        lambda path, window_seconds=15, step_seconds=5:
            {"start": 5.0, "end": 17.0, "score": 88.0})
    monkeypatch.setattr(pipeline_module, "_probe_duration",
                        lambda path: 30.0)

    def fake_extract(video_path, start, length, output_path):
        Path(output_path).write_bytes(b"fake-short")
        return output_path

    monkeypatch.setattr(pipeline_module, "_extract_window", fake_extract)

    result = pipe.process_moment("adinross", duration=30,
                                 raw_clip_path=str(raw), short_clip=True)
    assert result["status"] == "ready_to_upload"
    # The editor rendered the cut window, not the full 30 s file.
    assert captured["raw_clip_path"] != str(raw)
    assert "short_5s" in captured["raw_clip_path"]
    assert Path(captured["raw_clip_path"]).exists()
    # The stored clip length is the window length (12 s), not 30 s.
    assert _clip_row(result["clip_id"])["duration"] == 12.0


def test_process_moment_short_clip_failure_falls_back_to_full_clip(
        pipeline_setup, monkeypatch):
    pipe, raw, tmp_path, captured = pipeline_setup

    def boom(*a, **k):
        raise RuntimeError("no ffmpeg here")

    monkeypatch.setattr(pipe.detector, "find_peak_window", boom)
    result = pipe.process_moment("adinross", duration=30,
                                 raw_clip_path=str(raw), short_clip=True)
    assert result["status"] == "ready_to_upload"
    assert captured["raw_clip_path"] == str(raw)


def test_short_clip_mode_skips_silence_trim(pipeline_setup, monkeypatch):
    """A short clip is already a tight peak window — silence trimming
    would collapse sparse-speech moments (12 s -> 3.6 s in the first
    demo run). Normal clips still trim."""
    pipe, raw, tmp_path, captured = pipeline_setup
    monkeypatch.setattr(
        pipe.detector, "find_peak_window",
        lambda path, window_seconds=15, step_seconds=5:
            {"start": 5.0, "end": 17.0, "score": 88.0})
    monkeypatch.setattr(pipeline_module, "_probe_duration",
                        lambda path: 30.0)
    monkeypatch.setattr(
        pipeline_module, "_extract_window",
        lambda v, s, l, o: (Path(o).write_bytes(b"fake-short"), o)[1])

    pipe.process_moment("adinross", duration=30, raw_clip_path=str(raw),
                        short_clip=True)
    assert captured["silence_trim"] is False

    captured.clear()
    pipe.process_moment("adinross", duration=30, raw_clip_path=str(raw),
                        short_clip=False)
    assert captured["silence_trim"] is True


def test_rerender_reuses_stored_quote_line(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw))
    clip_id = result["clip_id"]
    stored_quote = result["quote_line"]
    assert stored_quote

    captured.clear()
    rerun = pipe.rerender_clip(clip_id)
    assert rerun["status"] == "success"
    assert captured["quote_line"] == stored_quote
    assert captured["banner_timing"] is None

    captured.clear()
    rerun = pipe.rerender_clip(clip_id, quote_line="",
                               banner_timing="whole_clip")
    assert rerun["status"] == "success"
    assert captured["quote_line"] == ""
    assert captured["banner_timing"] == "whole_clip"


def test_process_moment_quote_line_off_by_default(pipeline_setup):
    """v1.4.1: the quote is still picked and stored, but the editor is
    told NOT to draw it unless the line is opted in."""
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw))
    assert result["status"] == "ready_to_upload"
    assert captured["quote_line_enabled"] is False
    # The quote text still flows through and is stored for the opt-in.
    assert captured["quote_line"]
    assert result["quote_line"] == captured["quote_line"]
    assert _clip_row(result["clip_id"])["quote_line"] == captured["quote_line"]


def test_process_moment_quote_line_opt_in(pipeline_setup):
    """quote_line_enabled=True reaches the editor."""
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw),
                                 quote_line_enabled=True)
    assert result["status"] == "ready_to_upload"
    assert captured["quote_line_enabled"] is True


def test_process_moment_quote_line_enabled_by_config(pipeline_setup,
                                                     monkeypatch):
    """QUOTE_LINE_ENABLED=true opts in without a per-call argument."""
    monkeypatch.setattr(pipeline_module, "QUOTE_LINE_ENABLED", True)
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw))
    assert result["status"] == "ready_to_upload"
    assert captured["quote_line_enabled"] is True


def test_rerender_quote_line_off_by_default(pipeline_setup):
    """A re-render with no opt-in drops the quote line even when the
    clip has a stored quote (the editor gets False)."""
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw))
    clip_id = result["clip_id"]
    assert result["quote_line"]

    captured.clear()
    rerun = pipe.rerender_clip(clip_id)
    assert rerun["status"] == "success"
    assert captured["quote_line"] == result["quote_line"]
    assert captured["quote_line_enabled"] is False

    captured.clear()
    rerun = pipe.rerender_clip(clip_id, quote_line_enabled=True)
    assert rerun["status"] == "success"
    assert captured["quote_line_enabled"] is True


def test_quote_line_dropped_when_it_duplicates_headline(pipeline_setup):
    """When the headline itself is a quote, the quote picker returns the
    same line — showing it twice looks like a bug, so the pipeline drops
    the quote card (seen in the first VOD demo render)."""
    pipe, raw, tmp_path, captured = pipeline_setup
    metadata = {
        "category": "Viral Potential",
        "ai_score": 65,
        "selection_reason": "test",
        "viral_title": "Adinross: \"Oh, hey God damn I've been so annoying, bro.\"",
        "hook_banner": "Adinross: \"Oh, hey God damn I've been so annoying, bro.\"",
        "quote_line": "Oh, hey God damn I've been so annoying, bro.",
        "tiktok_caption": "x",
        "hashtags": "#adinross #fyp #kick",
    }
    result = pipe._finish_clip(
        clip_id=1, streamer="adinross", raw_clip_path=str(raw),
        duration=15, transcript_text="x", words=[],
        metadata=metadata, hype_score=50.0, trigger_reason="test")
    assert result["status"] == "ready_to_upload"
    assert captured["quote_line"] == ""
    assert result["quote_line"] == ""


def test_quote_line_column_migrates_onto_existing_db(tmp_path, monkeypatch):
    """init_db adds quote_line to a clips table created without it."""
    import sqlite3
    import database

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE clips (id INTEGER PRIMARY KEY, streamer TEXT)")
    conn.commit()
    conn.close()

    monkeypatch.setattr(database, "DATABASE_FILE", db_path)
    database.init_db()

    conn = sqlite3.connect(str(db_path))
    cols = [c[1] for c in conn.execute("PRAGMA table_info(clips)")]
    conn.close()
    assert "quote_line" in cols
