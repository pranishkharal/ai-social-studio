"""Tests for game_ocr.py (gta6-clipper feature 4).

No network anywhere. OCR is tested with an injected fake OCR function
plus the real RapidOCR engine when it happens to be installed (the
synthetic-frame test is skipped without OpenCV, and the real-engine
test is skipped without RapidOCR -- the module must work either way).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import game_ocr  # noqa: E402
from game_ocr import (  # noqa: E402
    colour_cues,
    detect_frame,
    match_banner,
    merge_events,
    normalize_banner_text,
    ocr_available,
    ocr_score_bonus,
    run_ocr,
    scan_game_events,
)

try:
    import cv2
except ImportError:
    cv2 = None

# The shared fake cv2 (installed by test_face_tracker_mp) is not real
# OpenCV: it has no randn / putText / VideoWriter, so these tests skip
# when it is the cv2 in play.
_REAL_CV2 = cv2 is not None and "fake" not in getattr(cv2, "__version__", "")
needs_cv2 = pytest.mark.skipif(not _REAL_CV2, reason="real OpenCV not installed")

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")


def _fake_ocr_box(text, conf=0.95, big=True, centred=True):
    """A fake OCR result: one box in the centre band of a 640x360 crop."""
    # The crop is 640x360-ish (centre band of the frame); a big box is
    # ~40 px tall, centred horizontally.
    if big:
        box = [[200, 150], [440, 150], [440, 190], [200, 190]]
    else:
        box = [[200, 150], [260, 150], [260, 162], [200, 162]]
    if not centred:
        box = [[p[0] - 180, p[1]] for p in box]
    return [(box, text, conf)]


def _card_frame(text, colour, grey=False):
    """Synthetic 640x360 frame with big centred text (like the reference
    repo's vision selftest cards)."""
    frame = np.full((360, 640, 3), 90, np.uint8)
    cv2.randn(frame, 90, 20)
    if grey:
        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                             cv2.COLOR_GRAY2BGR)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 2.2, 5)[0]
    cv2.putText(frame, text, ((640 - size[0]) // 2, 180),
                cv2.FONT_HERSHEY_DUPLEX, 2.2, colour, 5)
    return frame


# ---------------------------------------------------------------------------
# Fuzzy banner matching (no OCR engine needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,label", [
    ("WASTED", "WASTED"),
    ("BUSTEO", "BUSTED"),            # OCR slip still counts
    ("MISSION PASSEO!", "MISSION PASSED"),
    ("mission failed", "MISSION FAILED"),
    ("VICTORY ROYALE", "VICTORY ROYALE"),
    ("YOU DIED", "YOU DIED"),
    ("GAME OVER", "GAME OVER"),
    ("the player got BUSTED today", "BUSTED"),  # token window
])
def test_match_banner(text, label):
    assert match_banner(text) == label


@pytest.mark.parametrize("text", [
    "hello world",
    "WAS",          # too short
    "",
    "JUST CHATTING",
])
def test_match_banner_no_match(text):
    assert match_banner(text) is None


def test_normalize_banner_text():
    assert normalize_banner_text("Mission Passed!") == "MISSION PASSED"
    assert normalize_banner_text("wasted123") == "WASTED"


# ---------------------------------------------------------------------------
# Colour cues (OpenCV only)
# ---------------------------------------------------------------------------

@needs_cv2
def test_colour_cues_grey_frame_has_low_saturation():
    frame = _card_frame("X", (40, 190, 230), grey=True)
    assert colour_cues(frame)["sat"] < 0.05


@needs_cv2
def test_colour_cues_gold_and_red():
    gold = np.full((360, 640, 3), (40, 190, 230), np.uint8)  # BGR gold
    assert colour_cues(gold)["gold"] > 0.5
    red = np.full((360, 640, 3), (30, 30, 220), np.uint8)  # BGR red
    assert colour_cues(red)["red"] > 0.5


# ---------------------------------------------------------------------------
# Per-frame detection with injected OCR
# ---------------------------------------------------------------------------

@needs_cv2
def test_detect_frame_known_banner_with_fake_ocr():
    frame = _card_frame("WASTED", (30, 30, 220), grey=True)
    events = detect_frame(
        frame, ocr_fn=lambda img: _fake_ocr_box("WASTED"))
    assert [e[1] for e in events if e[0] == "word"] == ["WASTED"]


@needs_cv2
def test_detect_frame_low_confidence_banner_ignored():
    frame = _card_frame("WASTED", (30, 30, 220), grey=True)
    events = detect_frame(
        frame, ocr_fn=lambda img: _fake_ocr_box("WASTED", conf=0.5))
    assert not [e for e in events if e[0] == "word"]


@needs_cv2
def test_detect_frame_unknown_big_caps_text_reported():
    frame = _card_frame("LEGENDARY", (255, 255, 255))
    events = detect_frame(
        frame, ocr_fn=lambda img: _fake_ocr_box("LEGENDARY"))
    assert [e[1] for e in events if e[0] == "text"] == ["TEXT LEGENDARY"]


@needs_cv2
def test_detect_frame_small_text_not_a_banner():
    frame = _card_frame("LEGENDARY", (255, 255, 255))
    events = detect_frame(
        frame, ocr_fn=lambda img: _fake_ocr_box("LEGENDARY", big=False))
    assert events == []


# ---------------------------------------------------------------------------
# Merge + bonus (pure logic)
# ---------------------------------------------------------------------------

def test_merge_events_groups_same_label_within_gap():
    raw = [(10.0, "word", "WASTED", {"conf": 0.95}),
           (20.0, "word", "WASTED", {"conf": 0.95}),
           (100.0, "word", "WASTED", {"conf": 0.95})]
    events = merge_events(raw)
    assert len(events) == 2
    assert events[0]["hits"] == 2
    assert events[0]["t"] == 10.0 and events[0]["t_end"] == 20.0
    assert events[0]["weight"] == 0.8


def test_merge_events_drops_single_low_conf_hit_without_cue():
    raw = [(10.0, "word", "WASTED", {"conf": 0.7, "cues": {"sat": 0.9}})]
    assert merge_events(raw) == []


def test_merge_events_keeps_low_conf_hit_when_grey_cue_agrees():
    raw = [(10.0, "word", "WASTED",
            {"conf": 0.7, "cues": {"sat": 0.1, "gold": 0.0, "red": 0.0}})]
    assert len(merge_events(raw)) == 1


def test_ocr_score_bonus():
    assert ocr_score_bonus([]) == 0.0
    assert ocr_score_bonus([{"weight": 0.8}]) == 12.0
    assert ocr_score_bonus([{"weight": 0.5}]) == 7.5
    # Two banners add a little, capped at 15.
    assert ocr_score_bonus([{"weight": 0.8}, {"weight": 0.8},
                            {"weight": 0.8}]) == 15.0


def test_run_ocr_without_engine_returns_empty(monkeypatch):
    monkeypatch.setattr(game_ocr, "_get_ocr_engine", lambda: None)
    assert run_ocr(np.zeros((100, 100, 3), np.uint8)) == []


def test_ocr_available_returns_bool():
    assert isinstance(ocr_available(), bool)


# ---------------------------------------------------------------------------
# Whole-video scan with injected OCR
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def banner_video(tmp_path_factory):
    video = tmp_path_factory.mktemp("ocr") / "banner.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=8:size=320x180:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(video)],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-500:]
    return str(video)


@needs_ffmpeg
@needs_cv2
def test_scan_game_events_with_fake_ocr(banner_video):
    result = scan_game_events(
        banner_video, ocr_fn=lambda img: _fake_ocr_box("BUSTED"))
    assert result["ocr_available"] is True
    labels = [e["label"] for e in result["events"]]
    assert "BUSTED" in labels
    assert all(e["kind"] == "word" for e in result["events"]
               if e["label"] == "BUSTED")


@needs_ffmpeg
def test_scan_game_events_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        scan_game_events("/definitely/not/here.mp4")


@needs_ffmpeg
@needs_cv2
def test_scan_game_events_real_ocr_on_synthetic_card(tmp_path):
    """End to end with the real RapidOCR engine, when installed."""
    pytest.importorskip("rapidocr_onnxruntime",
                        reason="rapidocr-onnxruntime not installed")
    video = tmp_path / "wasted.mp4"
    # 6 s of a static grey card with big red WASTED text.
    card = _card_frame("WASTED", (30, 30, 220), grey=True)
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    for _ in range(60):
        writer.write(card)
    writer.release()
    result = scan_game_events(str(video), fps=1.0)
    assert result["ocr_available"] is True
    assert "WASTED" in [e["label"] for e in result["events"]]
