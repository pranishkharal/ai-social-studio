"""Tests for callback_finder.py — setup/payoff pair finding.

Pure tests, no network, no ffmpeg.
"""

import pytest

from callback_finder import (candidate_to_callback_spec, find_callbacks,
                             parse_segments, segments_from_words)


def _seg(start, end, text, **signals):
    seg = {"start": start, "end": end, "text": text}
    seg.update(signals)
    return seg


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def test_parse_segments_reads_embedded_tags():
    lines = parse_segments([
        _seg(10, 12, "hello chat [LOUD +20dB] [CHAT x5 LOL] [HEAT 80]"),
        _seg(20, 21, "plain line"),
    ])
    assert lines[0]["loud"] == 20
    assert lines[0]["chat"] == 5
    assert lines[0]["heat"] == 80
    assert lines[0]["plain"] == "hello chat"
    assert lines[1]["loud"] == 0
    assert lines[1]["plain"] == "plain line"


def test_parse_segments_explicit_signals_win_over_tags():
    lines = parse_segments([
        _seg(10, 12, "loud line [LOUD +5dB]", loud=30, chat=2, heat=50),
    ])
    assert lines[0]["loud"] == 30
    assert lines[0]["chat"] == 2
    assert lines[0]["heat"] == 50


def test_parse_segments_skips_bad_rows():
    lines = parse_segments([
        _seg(10, 12, "ok"),
        {"start": "nope", "end": 5, "text": "bad timing"},
        {"start": 30, "end": 31, "text": "   "},
        _seg(40, 41, "also ok"),
    ])
    assert [l["plain"] for l in lines] == ["ok", "also ok"]


def test_segments_from_words_splits_on_punctuation():
    words = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "will", "start": 0.2, "end": 0.4},
        {"word": "win.", "start": 0.4, "end": 0.7},
        {"word": "Watch", "start": 1.0, "end": 1.2},
        {"word": "me!", "start": 1.2, "end": 1.5},
    ]
    segs = segments_from_words(words)
    assert len(segs) == 2
    assert segs[0]["text"] == "I will win."
    assert segs[0]["start"] == 0.0
    assert segs[0]["end"] == 0.7
    assert segs[1]["text"] == "Watch me!"


def test_segments_from_words_splits_on_long_pause_without_punctuation():
    words = [
        {"word": "hello", "start": 0.0, "end": 0.3},
        {"word": "chat", "start": 0.3, "end": 0.6},
        {"word": "later", "start": 5.0, "end": 5.3},
        {"word": "guys", "start": 5.3, "end": 5.6},
    ]
    segs = segments_from_words(words)
    assert len(segs) == 2
    assert segs[0]["text"] == "hello chat"
    assert segs[1]["text"] == "later guys"


# ---------------------------------------------------------------------------
# finding
# ---------------------------------------------------------------------------


def _claim_payoff_lines():
    return parse_segments([
        _seg(100, 105, "I will never die to the cops, trust me."),
        _seg(110, 112, "hello chat"),
        _seg(2600, 2610, "The cops got me again, I told you! "
                         "[LOUD +20dB] [CHAT x5 LOL]"),
        _seg(2700, 2701, "ok [LOUD +30dB]"),
        _seg(2800, 2801, "bet on the platypus"),
        _seg(5000, 5002, "platypus wins [LOUD +20dB]"),
    ])


def test_find_callbacks_claim_then_payoff():
    result = find_callbacks(_claim_payoff_lines(), min_gap=300)
    assert result
    top = result[0]
    assert top["setup"][0] == 100.0
    assert top["payoff"][0] == 2600.0
    assert 40 < top["gap_min"] < 45
    assert "claim" in top["why"]
    assert top["score"] > 0


def test_find_callbacks_respects_min_gap():
    lines = _claim_payoff_lines()
    # min_gap 3000 s (50 min) rules out the 2600 s payoff (~41 min gap).
    result = find_callbacks(lines, min_gap=3000)
    assert all(c["gap_min"] >= 50 for c in result)


def test_find_callbacks_respects_max_gap():
    lines = _claim_payoff_lines()
    # max_gap 600 s rules out everything (smallest gap is ~41 min).
    assert find_callbacks(lines, min_gap=60, max_gap=600) == []


def test_find_callbacks_empty_input():
    assert find_callbacks([]) == []
    assert find_callbacks(None) == []


def test_find_callbacks_same_claim_again_needs_no_spike():
    lines = parse_segments([
        _seg(0, 3, "I will never lose this game."),
        _seg(400, 403, "I will never lose this game, chat."),
    ])
    result = find_callbacks(lines, min_gap=300)
    assert result
    assert "same claim again" in result[0]["why"]


def test_find_callbacks_question_answered_later():
    lines = parse_segments([
        _seg(0, 4, "Will the platypus win the race?"),
        _seg(100, 102, "filler talk here"),
        _seg(400, 404, "The platypus won the race! [LOUD +20dB]"),
    ])
    result = find_callbacks(lines, min_gap=300)
    assert result
    assert "question answered later" in result[0]["why"]


def test_find_callbacks_bare_again_is_noise():
    # A claim + a bare "again" with no shared word and no cue must not
    # match: "again" after any claim is noise.
    lines = parse_segments([
        _seg(0, 3, "I will never die."),
        _seg(400, 403, "again [LOUD +20dB]"),
    ])
    assert find_callbacks(lines, min_gap=300) == []


def test_find_callbacks_limits_pairs_per_region():
    # Three identical-ish setups near each other, one payoff: at most
    # 2 pairs may share a setup region.
    lines = parse_segments([
        _seg(0, 3, "I will never die to the cops."),
        _seg(30, 33, "I will never die to the cops, trust me."),
        _seg(60, 63, "I will never die to the cops again."),
        _seg(400, 404, "The cops got me again, I told you! [LOUD +20dB]"),
    ])
    result = find_callbacks(lines, min_gap=300, top=25)
    assert len(result) <= 2


def test_find_callbacks_rare_word_returns():
    # IDF needs a real transcript: 40 filler lines make three rare
    # shared words ("platypus quokka wombat") weigh enough to fire.
    segments = [_seg(0, 3, "bet on the platypus quokka wombat")]
    for k in range(40):
        segments.append(_seg(10 + k * 9, 11 + k * 9, f"filler line {k}"))
    segments.append(_seg(400, 402, "platypus quokka wombat wins [LOUD +20dB]"))
    result = find_callbacks(parse_segments(segments), min_gap=300)
    assert result
    assert "rare word returns" in result[0]["why"]


# ---------------------------------------------------------------------------
# spec conversion
# ---------------------------------------------------------------------------


def test_candidate_to_callback_spec_windows():
    candidate = {
        "score": 8.5,
        "setup": (100.0, 105.0, "I will never die."),
        "payoff": (2600.0, 2610.0, "The cops got me!"),
        "gap_min": 41.7,
        "why": "claim then cue",
        "tags": "payoff LOUD+20",
    }
    spec = candidate_to_callback_spec(candidate, hook="HE CALLED IT")
    assert spec["format"] == "callback"
    assert spec["hook"] == "HE CALLED IT"
    claim, payoff = spec["parts"]
    assert 3.0 <= claim["end"] - claim["start"] <= 6.0
    assert 8.0 <= payoff["end"] - payoff["start"] <= 15.0
    # The payoff window starts 2 s before the payoff line.
    assert payoff["start"] == 2598.0
    assert spec["callback"]["score"] == 8.5
    assert spec["callback"]["gap_min"] == 41.7
