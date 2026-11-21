"""Coordinator-owned integration tests for the gta6-clipper upgrade.

These cover the seams the feature workers did not own:
- the pipeline running visual detection and passing moment_type through
  to the story-headline builder (worker 1 left pipeline.py untouched);
- the _ed_to_present "happened" -> "happenes" bug found during the
  detection build.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline as pipeline_module  # noqa: E402
from pipeline import ClipPipeline  # noqa: E402


# ------------------------------------------------------- _ed_to_present fix

def test_ed_to_present_en_verbs():
    """-en verbs drop "ed", not just "d" (the "happenes" bug)."""
    from story_headlines import _ed_to_present

    assert _ed_to_present("happened") == "happens"
    assert _ed_to_present("opened") == "opens"
    assert _ed_to_present("listened") == "listens"
    assert _ed_to_present("sharpened") == "sharpens"


def test_ed_to_present_silent_e_verbs_unchanged():
    """Silent-e verbs still drop only the "d" — the fix must not
    break them (deleted -> deletes, intervened -> intervenes)."""
    from story_headlines import _ed_to_present

    assert _ed_to_present("deleted") == "deletes"
    assert _ed_to_present("intervened") == "intervenes"
    assert _ed_to_present("liked") == "likes"
    assert _ed_to_present("baked") == "bakes"
    assert _ed_to_present("stopped") == "stops"
    assert _ed_to_present("kissed") == "kisses"
    assert _ed_to_present("tried") == "tries"


def test_ed_to_present_genuine_double_consonants():
    """Stems that genuinely end in a double consonant must NOT be
    stripped: "killed" -> "kills", not "kils" (the old dedup broke
    every one of these)."""
    from story_headlines import _ed_to_present

    assert _ed_to_present("kissed") == "kisses"
    assert _ed_to_present("missed") == "misses"
    assert _ed_to_present("passed") == "passes"
    assert _ed_to_present("called") == "calls"
    assert _ed_to_present("killed") == "kills"
    assert _ed_to_present("added") == "adds"
    # ...while true -ed doubling still dedups.
    assert _ed_to_present("grabbed") == "grabs"
    assert _ed_to_present("dropped") == "drops"
    assert _ed_to_present("slammed") == "slams"
    assert _ed_to_present("chatted") == "chats"


def test_headline_never_says_happenes():
    from story_headlines import build_story_headline

    headline = build_story_headline(
        "adinross", "Bro something insane happened right there man.")
    assert "happenes" not in headline


# ------------------------------------------- pipeline visual-detection wiring

@pytest.fixture
def wired_pipe(tmp_path, monkeypatch, fresh_db):
    """ClipPipeline with stubbed detector/transcribe/editor, as in the
    TikTok pipeline tests, plus capture of the headline builder args."""
    monkeypatch.setattr(pipeline_module, "SUBTITLES_DIR", tmp_path / "subtitles")
    monkeypatch.setattr(pipeline_module, "EDITED_DIR", tmp_path / "edited")
    monkeypatch.setattr(pipeline_module, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(pipeline_module, "SUBTITLES_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "BGM_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "BROLL_ENABLED", False)
    (tmp_path / "subtitles").mkdir()
    (tmp_path / "edited").mkdir()
    (tmp_path / "clips").mkdir()

    pipe = ClipPipeline()
    monkeypatch.setattr(pipe, "_mark", lambda *a, **k: None)
    monkeypatch.setattr(
        pipe.ai, "transcribe_clip",
        lambda path, **kwargs: {"full_text": "Whoa he just crashed the car bro.",
                                "words": []})
    monkeypatch.setattr(
        pipe.editor, "edit_clip",
        lambda **kwargs: str(tmp_path / "edited" / "out.mp4"))

    captured = {}

    def fake_evaluate(video_path, transcript_text="", **kwargs):
        captured["evaluate_kwargs"] = kwargs
        return {"hype_score": 80, "trigger_reason": "test loud",
                "moment_type": "crash"}

    monkeypatch.setattr(pipe.detector, "evaluate_clip", fake_evaluate)

    import ai_story_engine
    real_headline = ai_story_engine.build_story_headline

    def recording_headline(*args, **kwargs):
        captured["headline_kwargs"] = kwargs
        return real_headline(*args, **kwargs)

    monkeypatch.setattr(ai_story_engine, "build_story_headline",
                        recording_headline)

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    return pipe, raw, captured


def test_process_moment_runs_visual_detection(wired_pipe, monkeypatch):
    """process_moment asks the detector to watch the video, and the
    moment type reaches the headline builder."""
    monkeypatch.delenv("VISUAL_DETECTION_ENABLED", raising=False)
    pipe, raw, captured = wired_pipe
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw))
    assert result["status"] == "ready_to_upload"
    assert captured["evaluate_kwargs"].get("detect_visual") is True
    assert captured["headline_kwargs"].get("moment_type") == "crash"


def test_visual_detection_can_be_disabled(wired_pipe, monkeypatch):
    """VISUAL_DETECTION_ENABLED=0 skips the video scan."""
    monkeypatch.setenv("VISUAL_DETECTION_ENABLED", "0")
    pipe, raw, captured = wired_pipe
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw))
    assert result["status"] == "ready_to_upload"
    assert captured["evaluate_kwargs"].get("detect_visual") is False
