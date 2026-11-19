"""Tests for streamer_detection.py (feature 13: multi-streamer detection)."""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# dotenv is not installed in this VM; stub it before streamer_detection
# lazily imports config / kick_program.
if "dotenv" not in sys.modules:
    _dotenv = types.ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = _dotenv

import streamer_detection
from streamer_detection import (
    detect_streamers_in_clip,
    detect_streamers_with_evidence,
    find_mentioned_streamers,
    pick_submission_streamer,
)


# --- signal (a): transcript mentions ---

def test_single_mention():
    assert find_mentioned_streamers("xqc was playing Valorant") == ["xqc"]


def test_multiple_mentions_in_order():
    result = find_mentioned_streamers(
        "Adin Ross called xQc and then Tarik joined the call")
    assert result == ["adinross", "xqc", "tarik"]


def test_case_insensitive_and_at_mentions():
    assert find_mentioned_streamers("I saw @XQC yesterday") == ["xqc"]
    assert find_mentioned_streamers("ADINROSS IS LIVE") == ["adinross"]


def test_aliases_match():
    assert find_mentioned_streamers("adin ross is here") == ["adinross"]
    assert find_mentioned_streamers("trainwreck just won big") == ["trainwreckstv"]
    assert find_mentioned_streamers("xqcow is streaming") == ["xqc"]
    assert find_mentioned_streamers("ryan garcia showed up") == ["ryangarcia"]


def test_word_boundaries_prevent_false_positives():
    # "adin" inside "trading", "miz" inside "mizuno" must not match.
    assert find_mentioned_streamers("he was trading stocks all day") == []
    assert find_mentioned_streamers("wearing my mizuno shoes") == []
    assert find_mentioned_streamers("the xqcowfan account") == []


def test_unknown_names_not_detected():
    assert find_mentioned_streamers("some random guy named bob") == []


def test_empty_transcript():
    assert find_mentioned_streamers("") == []
    assert find_mentioned_streamers(None) == []


def test_known_streamers_override():
    # Restrict the search to a custom list.
    assert find_mentioned_streamers("xqc and adinross", ["xqc"]) == ["xqc"]
    assert find_mentioned_streamers("xqc and adinross", []) == []


def test_creator_rates_names_are_known():
    # Names from kick_program.CREATOR_RATES (not just monitored streamers).
    assert find_mentioned_streamers("n3on and sneako were there") == ["n3on", "sneako"]


def test_duplicate_mentions_deduplicated():
    assert find_mentioned_streamers("xqc xqc XQC") == ["xqc"]


# --- combined detection ---

def test_detect_transcript_only():
    result = detect_streamers_in_clip(None, "xqc and adinross played together")
    assert result == ["xqc", "adinross"]


def test_detect_nothing():
    assert detect_streamers_in_clip(None, "nothing to see here") == []


def test_detect_with_evidence_structure():
    detected, evidence = detect_streamers_with_evidence(
        None, "tarik clutched the round")
    assert detected == ["tarik"]
    assert evidence["mentioned"] == ["tarik"]
    assert evidence["recognized"] == []
    assert evidence["face_evidence"] is None  # no video path given
    assert evidence["multi_person_suspected"] is False
    assert evidence["face_recognition_available"] is False  # not installed here
    assert evidence["reference_streamers"] == []


def test_detect_with_evidence_honest_note_when_nothing_found():
    detected, evidence = detect_streamers_with_evidence(None, "just gameplay")
    assert detected == []
    assert "transcript" in evidence["note"].lower()
    assert "face_recognition" in evidence["note"]


def test_face_evidence_skipped_gracefully_without_cv2(tmp_path, monkeypatch):
    # Simulate a missing cv2 explicitly: another test module installs a
    # fake cv2 in this pytest process, so absence must not be assumed.
    # (sys.modules["cv2"] = None makes `import cv2` raise ImportError.)
    monkeypatch.setitem(sys.modules, "cv2", None)
    fake_video = tmp_path / "v.mp4"
    fake_video.write_bytes(b"x")
    assert streamer_detection._face_evidence(str(fake_video)) is None
    assert streamer_detection._face_evidence(None) is None
    assert streamer_detection._face_evidence(str(tmp_path / "missing.mp4")) is None


def test_reference_encodings_empty_without_library(tmp_path):
    assert streamer_detection._load_reference_encodings(tmp_path) == {}
    assert streamer_detection._face_recognition_available() is False


# --- POV submission picker ---

def test_pick_lowest_cpm_streamer():
    # xqc $15 beats adinross $30
    assert pick_submission_streamer(["adinross", "xqc"]) == ("xqc", "/xqc")
    assert pick_submission_streamer(["xqc", "adinross"]) == ("xqc", "/xqc")


def test_pick_single_streamer():
    assert pick_submission_streamer(["adinross"]) == ("adinross", "/adinross")
    assert pick_submission_streamer(["N3ON"]) == ("n3on", "/n3on")


def test_pick_empty_falls_back_to_other():
    assert pick_submission_streamer([]) == ("other", "/other")
    assert pick_submission_streamer(None) == ("other", "/other")


def test_pick_unknown_name_falls_back_to_other():
    assert pick_submission_streamer(["not_a_creator"]) == ("other", "/other")


def test_pick_prefers_program_streamer_over_unknown():
    # xqc ($15) beats the /other fallback ($40) of an unknown name.
    assert pick_submission_streamer(["not_a_creator", "xqc"]) == ("xqc", "/xqc")


def test_pick_monitored_but_unlisted_streamer_submits_as_other():
    # tarik is monitored but not in CREATOR_RATES, so per kick_program
    # rules the clip submits under /other.
    assert pick_submission_streamer(["tarik"]) == ("other", "/other")
