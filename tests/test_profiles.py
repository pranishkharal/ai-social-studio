"""Tests for streamer_profiles.py (feature 11: per-streamer profiles)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamer_profiles
from streamer_profiles import DEFAULT_PROFILE, get_profile, load_profiles

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_JSON = REPO_ROOT / "streamer_profiles.json"


def test_real_json_exists_and_has_monitored_streamers():
    assert REAL_JSON.exists()
    data = json.loads(REAL_JSON.read_text())
    assert "default" in data
    for streamer in ["xqc", "adinross", "westcol", "tarik"]:
        assert streamer in data, f"{streamer} missing from streamer_profiles.json"


def test_real_json_profiles_vary():
    data = json.loads(REAL_JSON.read_text())
    clip_lengths = {data[s]["clip_seconds"] for s in ["xqc", "adinross", "westcol", "tarik"]}
    layouts = {data[s]["layout"] for s in ["xqc", "adinross", "westcol", "tarik"]}
    assert len(clip_lengths) >= 2, "profiles should vary clip lengths"
    assert len(layouts) >= 2, "profiles should vary layouts"


def test_real_json_has_all_keys():
    data = json.loads(REAL_JSON.read_text())
    required = {"hype_db_threshold", "clip_seconds", "layout",
                "caption_style", "platforms", "auto_clip_enabled"}
    for name, profile in data.items():
        assert required <= set(profile), f"{name} profile missing keys"


def test_get_profile_returns_streamer_overrides():
    profile = get_profile("xqc")
    assert profile["clip_seconds"] == 30
    assert profile["layout"] == "speaker_tracking"
    assert profile["hype_db_threshold"] == -18.0
    assert profile["platforms"] == ["tiktok", "youtube"]


def test_get_profile_case_insensitive_and_at_sign():
    assert get_profile("XQC") == get_profile("xqc")
    assert get_profile("AdinRoss") == get_profile("adinross")
    assert get_profile("@westcol") == get_profile("westcol")


def test_get_profile_unknown_streamer_gets_default():
    profile = get_profile("somebody_new")
    assert profile == get_profile("default")
    assert profile["clip_seconds"] == 45
    assert profile["layout"] == "blurred_backdrop"
    assert profile["auto_clip_enabled"] is True


def test_get_profile_none_and_empty():
    assert get_profile(None) == get_profile("default")
    assert get_profile("") == get_profile("default")


def test_partial_profile_merges_over_defaults(tmp_path):
    custom = tmp_path / "profiles.json"
    custom.write_text(json.dumps({
        "default": {"clip_seconds": 20, "layout": "split"},
        "someone": {"clip_seconds": 99},
    }))
    profile = get_profile("someone", path=str(custom))
    assert profile["clip_seconds"] == 99  # streamer override wins
    assert profile["layout"] == "split"  # from the file's default
    assert profile["hype_db_threshold"] == DEFAULT_PROFILE["hype_db_threshold"]
    assert profile["caption_style"] == DEFAULT_PROFILE["caption_style"]


def test_missing_file_falls_back_to_builtin_defaults(tmp_path):
    profiles = load_profiles(path=str(tmp_path / "does-not-exist.json"))
    assert profiles == {"default": dict(DEFAULT_PROFILE)}
    assert get_profile("xqc", path=str(tmp_path / "does-not-exist.json")) == DEFAULT_PROFILE


def test_invalid_json_falls_back_to_builtin_defaults(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    assert load_profiles(path=str(bad)) == {"default": dict(DEFAULT_PROFILE)}


def test_non_dict_json_falls_back(tmp_path):
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]")
    assert load_profiles(path=str(bad)) == {"default": dict(DEFAULT_PROFILE)}


def test_env_var_overrides_profiles_path(tmp_path, monkeypatch):
    custom = tmp_path / "env_profiles.json"
    custom.write_text(json.dumps({"default": {"clip_seconds": 12}}))
    monkeypatch.setenv("STREAMER_PROFILES_PATH", str(custom))
    assert get_profile("anyone")["clip_seconds"] == 12


def test_returned_profile_is_a_copy():
    profile = get_profile("xqc")
    profile["clip_seconds"] = 12345
    profile["platforms"].append("nowhere")
    fresh = get_profile("xqc")
    assert fresh["clip_seconds"] == 30
    assert "nowhere" not in fresh["platforms"]
