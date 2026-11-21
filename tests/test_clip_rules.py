"""Tests for clip_rules.py (gta6-clipper feature 13)."""

import pytest

import clip_rules
from clip_rules import PermissionDeniedError


def test_adinross_is_ask_with_kick_program_note():
    badge = clip_rules.get_rule("kick", "adinross")
    assert badge["rule"] == "ask"
    assert "Kick" in badge["note"] and "Clipping Program" in badge["note"]
    assert badge["source"].startswith("https://")


def test_unknown_streamer_defaults_to_unknown():
    badge = clip_rules.get_rule("kick", "someone-nobody-knows")
    assert badge == {"rule": "unknown", "note": "", "source": ""}


def test_lookup_is_case_insensitive_and_strips_at():
    assert clip_rules.get_rule("KICK", "AdinRoss")["rule"] == "ask"
    assert clip_rules.get_rule("youtube", "@ishowspeed")["rule"] == "allowed"


def test_check_allows_ask_and_unknown():
    assert clip_rules.check_clip_allowed("kick", "adinross")["rule"] == "ask"
    assert clip_rules.check_clip_allowed("kick", "nobody")["rule"] == "unknown"


def test_check_refuses_no_streamer():
    rules = {"kick:banned": {"rule": "no", "note": "Asked clippers to stop.",
                             "source": "https://example.com"}}
    with pytest.raises(PermissionDeniedError, match="banned"):
        clip_rules.check_clip_allowed("kick", "banned", rules=rules)


def test_check_allows_no_streamer_with_explicit_confirmation():
    rules = {"kick:banned": {"rule": "no", "note": "", "source": ""}}
    badge = clip_rules.check_clip_allowed("kick", "banned",
                                          permission_confirmed=True, rules=rules)
    assert badge["rule"] == "no"


def test_check_allows_no_streamer_with_env_override(monkeypatch):
    rules = {"kick:banned": {"rule": "no", "note": "", "source": ""}}
    monkeypatch.setenv("CLIP_PERMISSION_OVERRIDE", "1")
    badge = clip_rules.check_clip_allowed("kick", "banned", rules=rules)
    assert badge["rule"] == "no"


def test_invalid_rule_value_becomes_unknown():
    rules = {"kick:weird": {"rule": "maybe", "note": "", "source": ""}}
    assert clip_rules.get_rule("kick", "weird", rules=rules)["rule"] == "unknown"


def test_load_rules_missing_file_returns_empty(tmp_path):
    assert clip_rules.load_rules(tmp_path / "nope.json") == {}


def test_list_rules_sorted():
    rules = clip_rules.list_rules()
    keys = [(r["platform"], r["streamer"]) for r in rules]
    assert keys == sorted(keys)
    assert any(r["streamer"] == "adinross" and r["rule"] == "ask" for r in rules)


def test_pipeline_refuses_no_streamer(monkeypatch):
    """The pipeline guard: a "no" streamer is refused before any
    footage is acquired."""
    import pipeline as pipeline_module
    rules = {"kick:banned": {"rule": "no", "note": "", "source": ""}}
    monkeypatch.setattr(clip_rules, "load_rules", lambda path=None: rules)
    # Build the pipeline without running its real __init__ (which
    # needs the monitor/detector/AI singletons).
    pipe = pipeline_module.ClipPipeline.__new__(pipeline_module.ClipPipeline)
    result = pipe.process_moment("banned", duration=15)
    assert result["status"] == "failed"
    assert "no" in result["error"]
