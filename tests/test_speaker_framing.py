"""Tests for speaker_framing.py — active-speaker framing.

The labelling / clustering / trajectory math is pure (numpy only).
The model download is monkeypatched (no network). The loudness gate
uses ffmpeg on a synthetic fixture. The full analyze_speakers path is
tested with a stubbed face tracker + stubbed lip-motion measurement,
so it runs without OpenCV/MediaPipe.
"""

import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import speaker_framing as sf


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


def _samples(boxes_per_frame):
    return [(i, boxes) for i, boxes in enumerate(boxes_per_frame)]


def test_cluster_people_two_people_sorted_left_to_right():
    frames = [
        [(100, 100, 80, 80), (800, 100, 80, 80)],
        [(105, 102, 78, 80), (795, 98, 82, 80)],
        [(98, 99, 80, 80), (805, 101, 80, 80)],
        [(102, 100, 80, 80), (798, 100, 80, 80)],
    ]
    people = sf.cluster_people(_samples(frames))
    assert len(people) == 2
    assert people[0][0] < people[1][0]  # left first
    assert people[0][0] == pytest.approx(101, abs=3)
    assert people[1][0] == pytest.approx(799, abs=4)


def test_cluster_people_drops_stray_detections():
    frames = [
        [(100, 100, 80, 80), (800, 100, 80, 80)],
        [(100, 100, 80, 80), (800, 100, 80, 80)],
        [(100, 100, 80, 80), (800, 100, 80, 80)],
        [(100, 100, 80, 80), (400, 400, 30, 30)],  # stray, one frame only
    ]
    people = sf.cluster_people(_samples(frames))
    assert len(people) == 2


def test_cluster_people_caps_at_five():
    frames = [[(i * 100, 100, 60, 60) for i in range(7)] for _ in range(4)]
    assert len(sf.cluster_people(_samples(frames))) == 5


def test_cluster_people_empty():
    assert sf.cluster_people([]) == []
    assert sf.cluster_people([(0, []), (1, [])]) == []


# ---------------------------------------------------------------------------
# runs / hold / trades
# ---------------------------------------------------------------------------


def test_runs():
    assert sf.runs([0, 0, 1, 1, 1, 0]) == [[0, 0, 2], [1, 2, 5], [0, 5, 6]]
    assert sf.runs([]) == []


def test_hold_merges_short_runs_into_longer_neighbour():
    labels = [0] * 20 + [1] * 2 + [0] * 20
    held = sf.hold(labels, 5)
    assert held == [0] * 42  # the 2-sample blip is absorbed


def test_hold_keeps_runs_over_the_minimum():
    labels = [0] * 20 + [1] * 10 + [0] * 20
    assert sf.hold(labels, 5) == labels


def test_trades_finds_fast_back_and_forth():
    # 4 short runs, 2 people -> one trade span.
    labels = [0] * 30 + [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5 + [1] * 30
    spans = sf.trades(labels, 10)
    assert len(spans) == 1
    start, end, who = spans[0]
    assert who == (0, 1)
    assert start == 30 and end == 50


def test_trades_ignores_slow_turns():
    labels = [0] * 30 + [1] * 30 + [0] * 30
    assert sf.trades(labels, 10) == []


# ---------------------------------------------------------------------------
# labelling
# ---------------------------------------------------------------------------


def test_label_speakers_switches_only_on_clear_margin():
    # Person 0 talks (0-2 s), then person 1 (2-4 s). 10 Hz samples.
    scores = np.zeros((2, 40))
    scores[0, :20] = 0.05
    scores[1, 20:] = 0.05
    labels, splits = sf.label_speakers(scores)
    assert set(labels[:20]) == {0}
    assert set(labels[20:]) == {1}
    assert splits == []


def test_label_speakers_keeps_speaker_through_silence():
    scores = np.zeros((2, 40))
    scores[0, :10] = 0.05
    # Silence 1-3 s, then a tie: the speaker must stay 0.
    scores[0, 30:] = 0.02
    scores[1, 30:] = 0.02
    labels, _ = sf.label_speakers(scores)
    assert set(labels) == {0}


def test_label_speakers_enforces_min_hold():
    # Person 1 is active for only 0.5 s mid-clip: too short to cut to.
    scores = np.zeros((2, 60))
    scores[0, :] = 0.05
    scores[1, 30:35] = 0.09  # brief, louder interjection
    labels, _ = sf.label_speakers(scores)
    assert set(labels) == {0}


def test_label_speakers_fast_trades_become_splits():
    # Rapid back-and-forth for 6 s in the middle of a 9 s clip: 1 s
    # turns (10 samples each) — fast enough to trade, long enough to
    # survive the 0.5 s hold.
    scores = np.zeros((2, 90))
    scores[0, :20] = 0.05
    for turn in range(6):
        speaker = 1 if turn % 2 == 0 else 0
        scores[speaker, 20 + turn * 10:30 + turn * 10] = 0.08
    scores[1, 80:] = 0.05
    labels, splits = sf.label_speakers(scores)
    assert splits  # at least one fast-trade span
    assert any(isinstance(l, tuple) for l in labels)


def test_label_speakers_empty():
    assert sf.label_speakers(np.zeros((2, 0))) == ([], [])


# ---------------------------------------------------------------------------
# segments + trajectories
# ---------------------------------------------------------------------------


def test_segments_from_labels_round_trips():
    labels = [0] * 15 + [1] * 20 + [(0, 1)] * 10 + [1] * 15
    segments = sf.segments_from_labels(labels, names=["alice", "bob"])
    assert len(segments) == 4
    assert segments[0] == {"speaker": "alice", "start": 0.0, "end": 1.5}
    assert segments[1]["speaker"] == "bob"
    assert segments[2]["speaker"] == "alice+bob"
    assert segments[2]["split"] == ["alice", "bob"]
    assert segments[2]["start"] == 3.5
    assert segments[2]["end"] == 4.5


def test_trajectory_for_box_centres_on_face_and_clamps():
    traj = sf.trajectory_for_box((100, 100, 80, 80), raw_w=1920, raw_h=1080)
    assert traj["crop_w"] == 607  # 1080 * 9/16
    assert traj["raw_h"] == 1080
    # Face centre x=140, crop centred on it, clamped to >= 0.
    assert traj["x"] == 0
    traj = sf.trajectory_for_box((1600, 100, 80, 80), raw_w=1920, raw_h=1080)
    assert traj["x"] == 1920 - 607  # clamped to the right edge


def test_smooth_scores_averages():
    scores = np.array([[0.0, 1.0, 0.0, 1.0, 0.0]])
    smoothed = sf.smooth_scores(scores, 0.3)
    assert smoothed.shape == scores.shape
    assert smoothed[0, 1] < 1.0  # the spike is spread out
    assert smoothed[0, 1] > 0.0


# ---------------------------------------------------------------------------
# loudness gate (ffmpeg fixture)
# ---------------------------------------------------------------------------


def _make_audio_fixture(path):
    """10 s clip: loud sine 0-5 s, silence 5-10 s."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=10",
         "-f", "lavfi", "-i", "sine=f=440:d=10",
         "-af", "volume='if(lt(t,5),1,0)':eval=frame",
         "-shortest", "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=120)
    return path


def test_speech_loudness_mask_marks_loud_samples(tmp_path):
    src = _make_audio_fixture(tmp_path / "src.mp4")
    mask = sf.speech_loudness_mask(src, 100)
    assert mask.shape == (100,)
    assert mask[:40].mean() > 0.8   # loud half: mostly talking
    assert mask[60:].mean() < 0.2   # silent half: mostly quiet


def test_speech_loudness_mask_missing_file_is_all_true():
    mask = sf.speech_loudness_mask("/nonexistent/video.mp4", 10)
    assert mask.shape == (10,)
    assert mask.all()  # ungated fallback


# ---------------------------------------------------------------------------
# model download (monkeypatched — no network)
# ---------------------------------------------------------------------------


def test_ensure_landmarker_model_uses_env_override(tmp_path):
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"x" * 3_500_000)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv(sf.ENV_MODEL_PATH, str(model))
    try:
        assert sf.ensure_landmarker_model(tmp_path) == model
    finally:
        monkeypatch.undo()


def test_ensure_landmarker_model_rejects_bad_override(tmp_path):
    model = tmp_path / "tiny.task"
    model.write_bytes(b"x" * 100)  # too small to be the real model
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv(sf.ENV_MODEL_PATH, str(model))
    try:
        assert sf.ensure_landmarker_model(tmp_path) is None
    finally:
        monkeypatch.undo()


def test_ensure_landmarker_model_downloads_once(monkeypatch, tmp_path):
    calls = []

    def fake_download(url, dest, timeout=60):
        calls.append(url)
        Path(dest).write_bytes(b"y" * 3_700_000)

    monkeypatch.setattr(sf, "_download_file", fake_download)
    monkeypatch.delenv(sf.ENV_MODEL_PATH, raising=False)
    model = sf.ensure_landmarker_model(tmp_path)
    assert model == tmp_path / sf.LANDMARKER_FILENAME
    assert model.exists()
    assert "face_landmarker" in calls[0]
    # Second call reuses the downloaded file (no second download).
    assert sf.ensure_landmarker_model(tmp_path) == model
    assert len(calls) == 1


def test_ensure_landmarker_model_download_failure_returns_none(monkeypatch, tmp_path):
    def failing_download(url, dest, timeout=60):
        raise ConnectionError("no network")

    monkeypatch.setattr(sf, "_download_file", failing_download)
    monkeypatch.delenv(sf.ENV_MODEL_PATH, raising=False)
    assert sf.ensure_landmarker_model(tmp_path) is None


def test_landmarker_enabled_env(monkeypatch):
    monkeypatch.delenv(sf.ENV_LANDMARKER, raising=False)
    assert sf.landmarker_enabled() is True
    monkeypatch.setenv(sf.ENV_LANDMARKER, "0")
    assert sf.landmarker_enabled() is False


# ---------------------------------------------------------------------------
# analyze_speakers (stubbed tracker + measurements — no cv2 needed)
# ---------------------------------------------------------------------------


def _install_stub_tracker(monkeypatch, boxes_per_frame):
    """Stub speaker_tracker.SpeakerTracker with canned face boxes."""
    stub = types.ModuleType("speaker_tracker")

    class StubTracker:
        def analyze_face_boxes(self, path, sample_fps=6):
            return [(i, boxes) for i, boxes in enumerate(boxes_per_frame)]

        def close(self):
            pass

    stub.SpeakerTracker = StubTracker
    monkeypatch.setitem(sys.modules, "speaker_tracker", stub)


def _install_stub_cv2(monkeypatch, width=1920, height=1080, frames=300, fps=30.0):
    """Stub cv2.VideoCapture with canned video properties."""
    stub = types.ModuleType("cv2")
    stub.CAP_PROP_FRAME_WIDTH = 3
    stub.CAP_PROP_FRAME_HEIGHT = 4
    stub.CAP_PROP_FRAME_COUNT = 7
    stub.CAP_PROP_FPS = 5

    class StubCapture:
        def __init__(self, path):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            return {3: width, 4: height, 7: frames, 5: fps}[prop]

        def release(self):
            pass

    stub.VideoCapture = StubCapture
    monkeypatch.setitem(sys.modules, "cv2", stub)


def test_analyze_speakers_missing_file_returns_none():
    assert sf.analyze_speakers("/nonexistent/video.mp4") is None


def test_analyze_speakers_one_person_returns_none(monkeypatch, tmp_path):
    video = tmp_path / "one.mp4"
    video.write_bytes(b"fake")
    _install_stub_tracker(monkeypatch, [[(100, 100, 80, 80)]] * 4)
    assert sf.analyze_speakers(str(video)) is None


def test_analyze_speakers_two_people_full_path(monkeypatch, tmp_path):
    video = tmp_path / "two.mp4"
    video.write_bytes(b"fake")
    frames = [[(100, 100, 80, 80), (1500, 100, 80, 80)]] * 6
    _install_stub_tracker(monkeypatch, frames)
    _install_stub_cv2(monkeypatch, frames=300, fps=30.0)
    monkeypatch.setenv(sf.ENV_LANDMARKER, "0")  # force the fallback path

    # Person 0 talks 0-5 s, person 1 talks 5-10 s (10 Hz samples).
    scores = np.zeros((2, 100))
    scores[0, :50] = 0.08
    scores[1, 50:] = 0.08
    monkeypatch.setattr(sf, "_mouth_motion_scores",
                        lambda *a, **k: scores)
    monkeypatch.setattr(sf, "speech_loudness_mask",
                        lambda *a, **k: np.ones(100, dtype=bool))

    result = sf.analyze_speakers(str(video))
    assert result is not None
    assert result["method"] == "fallback"
    assert result["speakers"] == ["speaker_1", "speaker_2"]
    assert result["speaker_segments"][0]["speaker"] == "speaker_1"
    assert result["speaker_segments"][-1]["speaker"] == "speaker_2"
    assert result["trajectories"]["speaker_1"]["crop_w"] == 607
    assert len(result["split_speakers"]) == 2
    # Segments cover the whole 10 s clip.
    assert result["speaker_segments"][0]["start"] == 0.0
    assert result["speaker_segments"][-1]["end"] == 10.0


def test_analyze_speakers_quiet_lips_returns_none(monkeypatch, tmp_path):
    video = tmp_path / "quiet.mp4"
    video.write_bytes(b"fake")
    frames = [[(100, 100, 80, 80), (1500, 100, 80, 80)]] * 6
    _install_stub_tracker(monkeypatch, frames)
    _install_stub_cv2(monkeypatch, frames=300, fps=30.0)
    monkeypatch.setenv(sf.ENV_LANDMARKER, "0")
    monkeypatch.setattr(sf, "_mouth_motion_scores",
                        lambda *a, **k: np.zeros((2, 100)))
    monkeypatch.setattr(sf, "speech_loudness_mask",
                        lambda *a, **k: np.ones(100, dtype=bool))
    assert sf.analyze_speakers(str(video)) is None
