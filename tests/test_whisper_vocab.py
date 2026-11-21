"""Tests for the Whisper vocabulary prompt (gta6-clipper feature 9).

The prompt is built from the game + streamer names and chat slang, is
deduplicated, and honours the WHISPER_VOCAB_PROMPT override. A mocked
Whisper model asserts the prompt and the VAD filter actually reach
model.transcribe() — no real model is loaded. A pipeline test asserts
process_moment passes the streamer and game through.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai_story_engine as ai_module  # noqa: E402
from ai_story_engine import AIStoryEngine, build_vocab_prompt  # noqa: E402


# ------------------------------------------------------------ prompt build

def test_prompt_defaults_cover_game_and_slang():
    prompt = build_vocab_prompt()
    assert "GTA 6" in prompt
    assert "Leonida" in prompt
    assert "Lucia" in prompt
    assert "Chat" in prompt
    assert "KEKW" in prompt
    assert "bro" in prompt


def test_prompt_puts_game_first_and_adds_streamer():
    prompt = build_vocab_prompt(streamer="adinross", game="GTA 6")
    assert prompt.startswith("GTA 6,")
    assert "adinross" in prompt


def test_prompt_deduplicates_case_insensitively():
    # "GTA 6" is both the game arg and in the built-in list: once only.
    prompt = build_vocab_prompt(streamer="adinross", game="gta 6")
    assert prompt.lower().count("gta 6") == 1


def test_prompt_env_override_replaces_everything(monkeypatch):
    monkeypatch.setenv("WHISPER_VOCAB_PROMPT", "Custom words only.")
    assert build_vocab_prompt(streamer="xqc", game="GTA 6") == \
        "Custom words only."


# ------------------------------------------------------- mocked transcribe

class _FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _FakeSegment:
    def __init__(self, text, words):
        self.text = text
        self.words = words
        self.start = words[0].start if words else 0.0
        self.end = words[-1].end if words else 0.0


class _FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append({"path": path, **kwargs})
        seg = _FakeSegment("hello chat", [
            _FakeWord("hello", 0.0, 0.5),
            _FakeWord("chat", 0.5, 1.0),
        ])
        info = types.SimpleNamespace(language="en")
        return [seg], info


@pytest.fixture
def fake_whisper(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(ai_module, "get_whisper_model", lambda: model)
    return model


def test_transcribe_passes_prompt_and_vad(tmp_path, fake_whisper):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video")
    result = AIStoryEngine().transcribe_clip(
        str(clip), streamer="adinross", game="GTA 6")
    assert result["full_text"] == "hello chat"
    assert len(fake_whisper.calls) == 1
    call = fake_whisper.calls[0]
    assert call["vad_filter"] is True
    assert call["word_timestamps"] is True
    assert "adinross" in call["initial_prompt"]
    assert call["initial_prompt"].startswith("GTA 6,")
    assert "KEKW" in call["initial_prompt"]


def test_transcribe_without_streamer_still_prompts(tmp_path, fake_whisper):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video")
    AIStoryEngine().transcribe_clip(str(clip))
    call = fake_whisper.calls[0]
    assert call["vad_filter"] is True
    assert "GTA 6" in call["initial_prompt"]
    assert "Chat" in call["initial_prompt"]


def test_transcribe_no_model_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_module, "get_whisper_model", lambda: None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video")
    assert AIStoryEngine().transcribe_clip(str(clip)) == {
        "full_text": "", "words": []}


def test_transcribe_missing_file_returns_empty(fake_whisper, tmp_path):
    assert AIStoryEngine().transcribe_clip(
        str(tmp_path / "nope.mp4")) == {"full_text": "", "words": []}
    assert fake_whisper.calls == []  # the model is never called


# ------------------------------------------------------------ pipeline pass

def test_pipeline_passes_streamer_and_game_to_transcribe(
        tmp_path, monkeypatch, fresh_db):
    """process_moment forwards streamer + game so the vocabulary prompt
    is built from the right names."""
    import pipeline as pipeline_module
    from pipeline import ClipPipeline

    monkeypatch.setattr(pipeline_module, "SUBTITLES_DIR", tmp_path / "subtitles")
    monkeypatch.setattr(pipeline_module, "EDITED_DIR", tmp_path / "edited")
    monkeypatch.setattr(pipeline_module, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(pipeline_module, "SUBTITLES_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "SUBTITLE_STYLE", "classic")
    monkeypatch.setattr(pipeline_module, "BGM_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "BROLL_ENABLED", False)
    (tmp_path / "subtitles").mkdir()
    (tmp_path / "edited").mkdir()
    (tmp_path / "clips").mkdir()

    pipe = ClipPipeline()
    monkeypatch.setattr(pipe, "_mark", lambda *a, **k: None)
    seen = {}

    def capture_transcribe(path, streamer=None, game=None):
        seen.update(streamer=streamer, game=game)
        return {"full_text": "", "words": []}

    monkeypatch.setattr(pipe.ai, "transcribe_clip", capture_transcribe)
    monkeypatch.setattr(pipe.detector, "evaluate_clip",
                        lambda path, **kwargs: {"hype_score": 80.0,
                                      "trigger_reason": "test"})
    monkeypatch.setattr(pipe.editor, "edit_clip",
                        lambda **kwargs: kwargs["output_path"])

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    result = pipe.process_moment("adinross", duration=25,
                                 raw_clip_path=str(raw), game="GTA 6")
    assert seen == {"streamer": "adinross", "game": "GTA 6"}
    # The stubbed editor writes no file, so QC is skipped (ERROR) and
    # the clip still completes — the existing-test compatibility path.
    assert result["status"] == "ready_to_upload"
    assert result["qc"]["status"] == "ERROR"
