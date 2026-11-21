"""Tests for lite_mode.py (gta6-clipper feature 17)."""

import os

import pytest

import lite_mode
from lite_mode import DiskFullError


def test_ram_gb_positive():
    assert lite_mode.ram_gb() > 0


def test_lite_enabled_env_override(monkeypatch):
    monkeypatch.setenv("LITE_MODE", "1")
    assert lite_mode.lite_enabled() is True
    monkeypatch.setenv("LITE_MODE", "0")
    assert lite_mode.lite_enabled() is False


def test_lite_enabled_auto_detect(monkeypatch):
    monkeypatch.delenv("LITE_MODE", raising=False)
    monkeypatch.setattr(lite_mode, "ram_gb", lambda: 8.0)
    assert lite_mode.lite_enabled() is True
    monkeypatch.setattr(lite_mode, "ram_gb", lambda: 16.0)
    assert lite_mode.lite_enabled() is False
    monkeypatch.setattr(lite_mode, "ram_gb", lambda: 12.0)
    assert lite_mode.lite_enabled() is False  # 12 GB is the threshold, not under it


def test_settings_lite(monkeypatch):
    monkeypatch.setenv("LITE_MODE", "1")
    settings = lite_mode.settings()
    assert settings["lite"] is True
    assert settings["max_concurrent_renders"] == 1
    assert settings["render_fps"] == 30
    assert settings["max_stream_hours"] == 1
    assert settings["delete_audio_after_transcribe"] is True


def test_settings_full(monkeypatch):
    monkeypatch.setenv("LITE_MODE", "0")
    settings = lite_mode.settings()
    assert settings["lite"] is False
    assert settings["max_concurrent_renders"] == 3
    assert settings["render_fps"] == 60
    assert settings["max_stream_hours"] is None
    assert settings["delete_audio_after_transcribe"] is False


def test_check_disk_ok(tmp_path):
    assert lite_mode.check_disk(tmp_path, need_gb=0.001) > 0


def test_check_disk_raises_with_clear_message(tmp_path):
    with pytest.raises(DiskFullError, match="Only .* GB free"):
        lite_mode.check_disk(tmp_path, need_gb=10 ** 6)


def test_note(monkeypatch):
    monkeypatch.setenv("LITE_MODE", "0")
    assert lite_mode.note() == ""
    monkeypatch.setenv("LITE_MODE", "1")
    assert "Lite mode on" in lite_mode.note()
    assert "1 render at a time" in lite_mode.note()
