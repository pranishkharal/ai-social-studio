"""Tests for meme_sounds.py — synthesised meme sounds + visual gags.

Sound synthesis and planning are pure (numpy only). The ffmpeg render
tests use small synthetic fixtures. No network.
"""

import subprocess
import wave

import numpy as np
import pytest

import meme_sounds
from meme_sounds import (HOOK_SAFE, MAX_EXT, MOODS, SOUNDS, apply_meme,
                         find_payoff, make_all_sounds, meme_enabled,
                         meme_filter_complex, plan_meme, probe_video,
                         sound_path, write_wav)


# ---------------------------------------------------------------------------
# synthesis
# ---------------------------------------------------------------------------


def test_all_sounds_are_audible_not_clipping_sane_length():
    for name, fn in SOUNDS.items():
        samples = fn()
        duration = len(samples) / meme_sounds.SR
        assert 0.3 < duration < 4, name
        assert 0.3 < np.abs(samples).max() <= 1, name
        assert not np.isnan(samples).any(), name


def test_vine_boom_is_sub_bass():
    samples = meme_sounds.vine_boom()
    freqs = np.fft.rfftfreq(len(samples), 1 / meme_sounds.SR)
    peak = freqs[np.abs(np.fft.rfft(samples)).argmax()]
    assert peak < 150, f"boom peak {peak} Hz should be sub-bass"


def test_crickets_are_high():
    samples = meme_sounds.crickets()
    freqs = np.fft.rfftfreq(len(samples), 1 / meme_sounds.SR)
    peak = freqs[np.abs(np.fft.rfft(samples)).argmax()]
    assert 3500 < peak < 5200, f"crickets peak {peak} Hz"


def test_write_wav_roundtrip(tmp_path):
    path = tmp_path / "boom.wav"
    write_wav(path, meme_sounds.vine_boom())
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == meme_sounds.SR
        assert w.getnframes() > meme_sounds.SR  # longer than 1 s


def test_sound_path_generates_once_then_reuses(tmp_path):
    first = sound_path("ding", tmp_path)
    assert first.exists()
    mtime = first.stat().st_mtime_ns
    second = sound_path("ding", tmp_path)
    assert second == first
    assert second.stat().st_mtime_ns == mtime  # not regenerated


def test_sound_path_unknown_name():
    with pytest.raises(ValueError):
        sound_path("not_a_sound", "/tmp")


def test_make_all_sounds(tmp_path):
    names = make_all_sounds(tmp_path)
    assert names == sorted(SOUNDS)
    assert all((tmp_path / f"{n}.wav").exists() for n in names)


# ---------------------------------------------------------------------------
# moods + opt-in
# ---------------------------------------------------------------------------


def test_moods_cover_nine_moods_with_real_sounds():
    assert set(MOODS) == {"embarrassing", "fail", "shock", "hype", "win",
                          "awkward", "cringe", "funny", "suspense"}
    for mood, spec in MOODS.items():
        assert spec["sound"] in SOUNDS, mood
        assert spec["effect"], mood


def test_meme_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv("MEME_SOUNDS_ENABLED", raising=False)
    assert meme_enabled() is False
    monkeypatch.setenv("MEME_SOUNDS_ENABLED", "1")
    assert meme_enabled() is True
    monkeypatch.setenv("MEME_SOUNDS_ENABLED", "yes")
    assert meme_enabled() is True


def test_default_mood_needs_enabled_and_valid_name(monkeypatch):
    monkeypatch.delenv("MEME_SOUNDS_ENABLED", raising=False)
    monkeypatch.setenv("MEME_MOOD", "shock")
    assert meme_sounds.default_mood() is None  # off wins
    monkeypatch.setenv("MEME_SOUNDS_ENABLED", "1")
    assert meme_sounds.default_mood() == "shock"
    monkeypatch.setenv("MEME_MOOD", "nope")
    assert meme_sounds.default_mood() is None


# ---------------------------------------------------------------------------
# planning (pure)
# ---------------------------------------------------------------------------


def test_plan_meme_clamps_payoff_out_of_hook_zone():
    plan = plan_meme("shock", duration=10.0, at=0.5)
    assert plan["at"] >= HOOK_SAFE
    plan = plan_meme("shock", duration=10.0, at=9.9)
    assert plan["at"] <= 9.6  # duration - 0.4


def test_plan_meme_total_never_exceeds_max_ext():
    for mood in MOODS:
        plan = plan_meme(mood, duration=10.0, at=5.0, sound_len=3.4)
        assert plan["total"] <= 10.0 + MAX_EXT + 1e-9, mood
        assert plan["total"] >= 10.0, mood


def test_plan_meme_auto_tail_fits_the_sound():
    # awkward: crickets are 3.4 s, hold 0.8 s, sound starts at 5 s.
    # The sound fits inside the clip + hold, so the tail is just the
    # 0.3 s auto breath after the sound.
    plan = plan_meme("awkward", duration=10.0, at=5.0, sound_len=3.4)
    assert plan["hold"] == 0.8
    assert plan["tail"] == 0.3
    assert plan["total"] == 11.1
    # In a 6 s clip the sound runs 1.6 s past the end: the tail grows
    # to fit it (1.6 s overrun + 0.3 s breath).
    plan = plan_meme("awkward", duration=6.0, at=5.0, sound_len=3.4)
    assert plan["tail"] == pytest.approx(1.9)
    assert plan["total"] == pytest.approx(8.7)


def test_plan_meme_unknown_mood():
    with pytest.raises(ValueError):
        plan_meme("nope", duration=10.0)


def test_meme_filter_complex_hold_vs_no_hold():
    no_hold = meme_filter_complex(
        360, 640, 10.0,
        plan_meme("shock", duration=10.0, at=5.0, sound_len=1.8),
        sound_len=1.8)
    assert "concat=n=2:v=1:a=0[vo]" in no_hold
    assert "loudnorm=I=-14" in no_hold
    assert "adelay=" in no_hold
    assert "[hold]" not in no_hold

    with_hold = meme_filter_complex(
        360, 640, 10.0,
        plan_meme("embarrassing", duration=10.0, at=5.0, sound_len=3.4),
        sound_len=3.4)
    assert "concat=n=3:v=1:a=0[vo]" in with_hold
    assert "tpad=stop_mode=clone" in with_hold  # freeze-frame
    assert "vignette" in with_hold


# ---------------------------------------------------------------------------
# ffmpeg renders (small fixtures)
# ---------------------------------------------------------------------------


def _make_fixture(path, duration=10.0, loud_at=None):
    """360x640 test clip with a sine tone; optionally loud at loud_at."""
    if loud_at is None:
        audio = ["-f", "lavfi", "-i", f"sine=f=300:d={duration}"]
    else:
        audio = ["-f", "lavfi", "-i", f"sine=f=300:d={duration}",
                 "-af", f"volume='if(between(t,{loud_at},{loud_at + 0.3}),4,0.3)':eval=frame"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=s=360x640:r=30:d={duration}",
         *audio, "-shortest", "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=120)
    return path


def test_find_payoff_finds_the_loud_burst(tmp_path):
    src = _make_fixture(tmp_path / "src.mp4", loud_at=7.0)
    payoff = find_payoff(src, 10.0)
    assert 6.5 <= payoff <= 7.6, payoff


def test_find_payoff_never_in_hook_zone(tmp_path):
    src = _make_fixture(tmp_path / "src.mp4", loud_at=1.0)
    assert find_payoff(src, 10.0) >= HOOK_SAFE


def test_apply_meme_shock_renders(tmp_path):
    src = _make_fixture(tmp_path / "src.mp4")
    out = tmp_path / "shock.mp4"
    result = apply_meme(src, "shock", out, at=5.0,
                        sounds_directory=tmp_path / "sounds")
    assert result["at"] == 5.0
    assert result["sound"] == "vine_boom.wav"
    assert out.exists()
    w, h, d = probe_video(out)
    assert (w, h) == (360, 640)
    assert abs(d - result["duration"]) < 0.15
    assert abs(d - 10.0) < 0.2  # shock has no hold/tail: same length


def test_apply_meme_freeze_frame_extends_clip(tmp_path):
    src = _make_fixture(tmp_path / "src.mp4")
    out = tmp_path / "emb.mp4"
    result = apply_meme(src, "embarrassing", out, at=5.0,
                        sounds_directory=tmp_path / "sounds")
    assert result["hold"] == 1.2
    # 10 s clip + 1.2 s freeze + 0.3 s auto tail breath.
    assert result["duration"] == pytest.approx(11.5, abs=0.05)
    w, h, d = probe_video(out)
    assert abs(d - 11.5) < 0.2


def test_apply_meme_hook_zone_protected(tmp_path):
    src = _make_fixture(tmp_path / "src.mp4")
    result = apply_meme(src, "shock", tmp_path / "hook.mp4", at=0.5,
                        sounds_directory=tmp_path / "sounds")
    assert result["at"] >= HOOK_SAFE


def test_apply_meme_unknown_mood(tmp_path):
    src = _make_fixture(tmp_path / "src.mp4")
    with pytest.raises(ValueError):
        apply_meme(src, "nope", tmp_path / "x.mp4")
