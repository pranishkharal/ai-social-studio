"""Tests for visual_moments.py (gta6-clipper feature 1).

No network anywhere. The scoring tests are pure numpy on synthetic
feature arrays (mirroring the reference repo's selftest scenarios);
the end-to-end test runs real ffmpeg on a small synthetic video, and
works with or without OpenCV (without it the video pass is skipped
and moments are scored from audio only).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import visual_moments  # noqa: E402
from visual_moments import (  # noqa: E402
    audio_band_levels,
    best_moment,
    detect_visual_moments,
    laughter_modulation,
    moment_tag,
    percentile_scale,
    previous_median,
    score_moment_types,
    segments_from_scores,
    smooth,
    text_features,
)

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def test_percentile_scale_median_zero_p98_one():
    values = np.arange(100, dtype=float)
    scaled = percentile_scale(values)
    assert scaled[50] == pytest.approx(0.0, abs=0.02)
    assert scaled[98] == pytest.approx(1.0, abs=0.02)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0


def test_smooth_centred_average():
    values = np.array([0.0, 0.0, 9.0, 0.0, 0.0])
    assert smooth(values, 3)[2] == pytest.approx(3.0)


def test_previous_median_is_causal():
    values = np.array([1.0, 2.0, 3.0, 100.0])
    # The median before index 3 only sees 1, 2, 3 -- never the spike.
    assert previous_median(values, 3)[3] == pytest.approx(2.0)


def test_audio_band_levels_separate_frequencies():
    rate = 16000
    t = np.arange(rate * 2) / rate
    low_tone = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    voice_tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    low_bands = audio_band_levels(low_tone)
    voice_bands = audio_band_levels(voice_tone)
    assert low_bands["low"].mean() > low_bands["voice"].mean() + 10
    assert voice_bands["voice"].mean() > voice_bands["low"].mean() + 10


def test_laughter_modulation_laugh_beats_steady_speech():
    rng = np.random.default_rng(1)
    rate = 16000
    t = np.arange(rate * 20) / rate
    noise = rng.normal(0, 0.02, len(t)).astype(np.float32)
    steady = noise + 0.2 * np.sin(2 * np.pi * 800 * t)
    # 5.5 Hz on/off gating = "ha-ha-ha" bursts.
    laugh = noise + 0.4 * np.sin(2 * np.pi * 800 * t) * (
        np.sin(2 * np.pi * 5.5 * t) > 0)

    def modulation(signal):
        voice_env = 10 ** (audio_band_levels(signal)["voice"] / 20)
        return laughter_modulation(voice_env)[100:300].max()

    assert modulation(laugh) > 0.6 > modulation(steady) + 0.1


# ---------------------------------------------------------------------------
# Scoring on synthetic features (one planted moment per type)
# ---------------------------------------------------------------------------

def _synthetic_features():
    """5 minutes of bins with one planted moment per type."""
    rng = np.random.default_rng(1)
    n = 600  # 5 min at 2 fps
    features = {
        k: np.abs(rng.normal(0, 0.1, n))
        for k in ("m", "out", "rot", "cam", "mod")
    }
    for k in ("low", "voice", "high", "all"):
        features[k] = -50 + rng.normal(0, 0.4, n)
    features["mod"] *= 0.3
    for k in ("txt_laugh", "txt_hype", "chat_lol"):
        features[k] = np.zeros(n)

    def plant(a, b, **kw):
        for k, v in kw.items():
            features[k][a:b] += v

    plant(100, 160, m=1.5, out=1.0, low=14, all=10)  # chase: 30 s
    plant(250, 253, m=3.5, all=22, high=18, voice=10)  # crash: spike
    for a in range(330, 370, 4):  # fight: repeated hits
        plant(a, a + 1, all=18, high=14, voice=14, m=2)
    plant(330, 370, m=1.2)
    plant(440, 456, rot=3.0, m=2.0)  # stunt: rotation
    plant(520, 540, voice=14, all=8, cam=3.0)  # reaction
    plant(570, 590, voice=8, mod=0.9, txt_laugh=1, chat_lol=1)  # funny
    return features


def test_each_moment_type_found_where_planted():
    moments = segments_from_scores(
        score_moment_types(_synthetic_features(), True, True))
    expected = [("chase", 100, 160), ("crash", 250, 253),
                ("fight", 330, 370), ("stunt", 440, 456),
                ("reaction", 520, 540), ("funny", 570, 590)]
    for moment_type, a, b in expected:
        assert any(m["type"] == moment_type
                   and m["t0"] < b / 2 and m["t1"] > a / 2
                   for m in moments), (moment_type, moments)


def test_no_false_hits_in_quiet_start():
    moments = segments_from_scores(
        score_moment_types(_synthetic_features(), True, True))
    assert not any(m["t1"] < 45 for m in moments)


def test_audio_only_finds_crash_but_never_stunt():
    moments = segments_from_scores(
        score_moment_types(_synthetic_features(), False, False))
    assert any(m["type"] == "crash" for m in moments)
    assert not any(m["type"] == "stunt" for m in moments)
    # Chase/crash/fight are capped without video (motion is unknown);
    # reaction/funny are voice-driven, so they are not capped.
    assert all(m["score"] <= 0.8 for m in moments
               if m["type"] in ("chase", "crash", "fight"))


def test_segments_merge_short_gaps_and_drop_short_ones():
    scores = {"chase": np.zeros(40), "crash": np.zeros(40),
              "fight": np.zeros(40), "stunt": np.zeros(40),
              "reaction": np.zeros(40), "funny": np.zeros(40)}
    scores["chase"][10:16] = 0.9   # 3 s -- under the 5 s chase minimum
    scores["crash"][20:24] = 0.9   # 2 s crash, fine (>= 1 s)
    moments = segments_from_scores(scores)
    assert not any(m["type"] == "chase" for m in moments)
    assert any(m["type"] == "crash" and m["t0"] == 10.0
               and m["t1"] == 12.0 for m in moments)


def test_text_features_mark_laugh_hype_and_chat():
    lines = [
        (0.0, 4.0, "haha that was amazing [CHAT x2: \"KEKW\" x9]"),
        (10.0, 12.0, "oh my god no way!"),
        (20.0, 22.0, "just talking normally"),
    ]
    feats = text_features(lines, 60)
    assert feats["txt_laugh"][0:8].max() == 1
    assert feats["chat_lol"][0:8].max() == 1
    assert feats["txt_hype"][20:24].max() == 1
    assert feats["txt_laugh"][40:44].max() == 0


def test_best_moment_and_tag():
    moments = [
        {"t0": 0.0, "t1": 5.0, "type": "chase", "score": 0.6},
        {"t0": 10.0, "t1": 12.0, "type": "crash", "score": 0.9},
    ]
    assert best_moment(moments)["type"] == "crash"
    assert moment_tag(best_moment(moments)) == "[MOMENT crash 0.9]"
    assert best_moment([]) is None


# ---------------------------------------------------------------------------
# End to end on a synthetic video (ffmpeg; OpenCV optional)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    video = tmp_path_factory.mktemp("visual") / "clip.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=12:size=320x180:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(video)],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-500:]
    return str(video)


@needs_ffmpeg
def test_detect_visual_moments_runs_on_synthetic_video(synthetic_video):
    result = detect_visual_moments(synthetic_video)
    assert result["dur"] == pytest.approx(12.0, abs=0.5)
    assert isinstance(result["moments"], list)
    assert result["scan_s"] >= 0
    assert result["x_realtime"] > 0
    for moment in result["moments"]:
        assert moment["type"] in visual_moments.MOMENT_TYPES
        assert 0.0 <= moment["score"] <= 1.0
        assert moment["t1"] > moment["t0"]


@needs_ffmpeg
def test_detect_visual_moments_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        detect_visual_moments("/definitely/not/here.mp4")


# ---------------------------------------------------------------------------
# Hype detector integration
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_evaluate_clip_blends_visual_moments(synthetic_video):
    from hype_detector import HypeDetector
    detector = HypeDetector()
    base = detector.evaluate_clip(synthetic_video, "wow that was crazy")
    moments = [{"t0": 1.0, "t1": 8.0, "type": "chase", "score": 0.9},
               {"t0": 2.0, "t1": 4.0, "type": "crash", "score": 0.6}]
    with_visual = detector.evaluate_clip(
        synthetic_video, "wow that was crazy", visual_moments=moments)
    assert with_visual["moment_type"] == "chase"
    assert with_visual["visual_score"] == 90.0
    assert with_visual["visual_moments"] == moments
    # 50% audio + 30% lexical + 20% visual(90) vs 60/40 without visual.
    expected = round(base["audio_metrics"]["score"] * 0.5
                     + base["lexical_metrics"]["score"] * 0.3
                     + 90.0 * 0.2, 1)
    assert with_visual["hype_score"] == expected
    assert "visual moment: chase" in with_visual["trigger_reason"]


@needs_ffmpeg
def test_evaluate_clip_ocr_bonus_and_labels(synthetic_video):
    from hype_detector import HypeDetector
    detector = HypeDetector()
    events = [{"t": 5.0, "t_end": 5.0, "kind": "word", "label": "WASTED",
               "hits": 2, "weight": 0.8}]
    result = detector.evaluate_clip(synthetic_video, ocr_events=events)
    assert result["ocr_bonus"] == 12.0  # 0.8 weight * 15 max
    assert "game banner: WASTED" in result["trigger_reason"]
    base = detector.evaluate_clip(synthetic_video)
    assert result["hype_score"] == round(
        min(100.0, base["hype_score"] + 12.0), 1)


@needs_ffmpeg
def test_evaluate_clip_without_signals_is_unchanged(synthetic_video):
    from hype_detector import HypeDetector
    detector = HypeDetector()
    result = detector.evaluate_clip(synthetic_video, "wow that was crazy")
    expected = round(result["audio_metrics"]["score"] * 0.6
                     + result["lexical_metrics"]["score"] * 0.4, 1)
    assert result["hype_score"] == expected
    assert result["moment_type"] is None
    assert result["visual_score"] == 0.0
    assert result["ocr_bonus"] == 0.0


# ---------------------------------------------------------------------------
# Headline integration (moment type names the moment)
# ---------------------------------------------------------------------------

def test_headline_uses_moment_type_fallback():
    from story_headlines import build_story_headline
    headline = build_story_headline(
        "adinross", "um uh yeah", moment_type="crash")
    assert "crashed" in headline
    assert "Adinross" in headline


def test_headline_specific_transcript_beats_moment_type():
    from story_headlines import build_story_headline
    headline = build_story_headline(
        "adinross", "He smashed the guitar Amanda gave him.",
        moment_type="crash")
    assert "smashes" in headline.lower()
