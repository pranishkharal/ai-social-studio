"""Tests for learn-from-your-own-channel (gta6-clipper feature 3).

No network anywhere: the yt-dlp call is replaced by an injected runner
returning a fixed payload. Covers the fetch + cache, row building and
matching, the 8-Short minimum, rule learning, YouTube auto-fill into
analytics (with no duplicate snapshots), the pipeline tie-break, and
the rules showing up in analytics_summary().
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import channel_learn  # noqa: E402
from channel_learn import (  # noqa: E402
    build_rows,
    fetch_shorts,
    learn_rules,
    load_learned_rules,
    local_clip_index,
    save_learned_rules,
    score_tiebreak,
    summarize_learning,
    sync_channel_performance,
)

# Ten Shorts: Speed's clips dominate, Kai's trail, questions flop.
# Timestamps are 3 days apart so post-time blocks differ.
ENTRIES = [
    {"title": "Speed screams at GTA 6 map", "view_count": 900000,
     "timestamp": 1763000000, "duration": 22},
    {"title": "Speed finds KFC", "view_count": 700000,
     "timestamp": 1763259200, "duration": 22},
    {"title": "Speed breaks car 3 times", "view_count": 800000,
     "timestamp": 1763518400, "duration": 25},
    {"title": "Kai laughs at trailer", "view_count": 120000,
     "timestamp": 1763777600, "duration": 28},
    {"title": "Kai reacts?", "view_count": 100000,
     "timestamp": 1764036800, "duration": 24},
    {"title": "why gta 6 is late", "view_count": 40000,
     "timestamp": 1764296000, "duration": 31},
    {"title": "why this matters?", "view_count": 30000,
     "timestamp": 1764555200, "duration": 33},
    {"title": "trailer talk", "view_count": 35000,
     "timestamp": 1764814400, "duration": 18},
    {"title": "Kai calm take", "view_count": 110000,
     "timestamp": 1765073600, "duration": 26},
    {"title": "Speed again", "view_count": 600000,
     "timestamp": 1765332800, "duration": 21},
]
PAYLOAD = {"channel": "Fixture", "entries": ENTRIES}


class FakeRun:
    """Stand-in for subprocess.run: returns the payload, counts calls."""

    def __init__(self, payload=PAYLOAD, returncode=0, stderr=""):
        self.payload = payload
        self.returncode = returncode
        self.stderr = stderr
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(args[0])
        import types
        return types.SimpleNamespace(
            returncode=self.returncode, stderr=self.stderr,
            stdout=json.dumps(self.payload))


@pytest.fixture
def learn_dirs(tmp_path, monkeypatch):
    """Point the learn cache + rules files at a temp dir."""
    monkeypatch.setattr(channel_learn, "CLIPS_DIR", tmp_path)
    monkeypatch.setattr(channel_learn, "CHANNEL_URL",
                        "https://www.youtube.com/@fixture")
    return tmp_path


# ------------------------------------------------------------------- fetch

def test_fetch_shorts_parses_payload(learn_dirs):
    runner = FakeRun()
    data = fetch_shorts("https://www.youtube.com/@fixture", runner=runner)
    assert data["channel"] == "Fixture"
    assert len(data["entries"]) == 10
    assert "yt_dlp" in runner.calls[0]  # went through yt-dlp, not an API


def test_fetch_shorts_uses_cache_within_a_day(learn_dirs):
    runner = FakeRun()
    fetch_shorts("https://www.youtube.com/@fixture", runner=runner)
    fetch_shorts("https://www.youtube.com/@fixture", runner=runner)
    assert len(runner.calls) == 1  # second call served from cache


def test_fetch_shorts_refresh_ignores_cache(learn_dirs):
    runner = FakeRun()
    fetch_shorts("https://www.youtube.com/@fixture", runner=runner)
    fetch_shorts("https://www.youtube.com/@fixture", refresh=True,
                 runner=runner)
    assert len(runner.calls) == 2


def test_fetch_shorts_refetches_after_cache_expiry(learn_dirs, tmp_path):
    runner = FakeRun()
    fetch_shorts("https://www.youtube.com/@fixture", runner=runner)
    cache = next(tmp_path.glob("learn_cache_*.json"))
    old = time.time() - 90000  # older than the 24 h TTL
    import os
    os.utime(cache, (old, old))
    fetch_shorts("https://www.youtube.com/@fixture", runner=runner)
    assert len(runner.calls) == 2


def test_fetch_shorts_failure_raises(learn_dirs):
    runner = FakeRun(returncode=1, stderr="boom")
    with pytest.raises(RuntimeError, match="yt-dlp failed"):
        fetch_shorts("https://www.youtube.com/@fixture", runner=runner)


# -------------------------------------------------------------------- rows

def test_build_rows_matches_local_clips():
    index = [{"clip_id": 7, "titles": ["Speed finds KFC"],
              "source": "speed", "length": 22}]
    rows = build_rows(ENTRIES, index)
    assert len(rows) == 10
    matched = [r for r in rows if r["matched"]]
    assert len(matched) == 1
    assert matched[0]["clip_id"] == 7
    assert matched[0]["source"] == "speed"
    assert matched[0]["length"] == "20-30 s"


def test_build_rows_skips_entries_without_views():
    entries = ENTRIES + [{"title": "private short", "view_count": None}]
    assert len(build_rows(entries, [])) == 10


def test_build_rows_hook_features():
    rows = build_rows(ENTRIES, [])
    by_title = {r["title"]: r for r in rows}
    assert by_title["Speed breaks car 3 times"]["number"] is True
    assert by_title["Kai reacts?"]["question"] is True
    assert by_title["Speed screams at GTA 6 map"]["name_first"] is True
    assert by_title["why gta 6 is late"]["name_first"] is False


# ------------------------------------------------------------------- rules

def test_learn_rules_finds_the_winning_streamer():
    rows = build_rows(ENTRIES, [])
    rules = learn_rules(rows)
    assert rules, "Speed's 600k-900k views should beat the median clearly"
    texts = [r["text"] for r in rules]
    assert any("Speed" in t and "median views" in t for t in texts)
    speed_rule = next(r for r in rules if r["kind"] == "source"
                      and r["value"] == "Speed")
    assert speed_rule["ratio"] >= 1.5


def test_learn_rules_needs_eight_shorts():
    rows = build_rows(ENTRIES[:5], [])
    assert learn_rules(rows) == []
    summary = summarize_learning(rows)
    assert summary["too_few"] is True
    assert summary["shorts"] == 5


def test_learn_rules_no_pattern_is_empty():
    flat = [{"title": f"clip {i}", "view_count": 1000 + i,
             "timestamp": 1763000000 + i * 86400, "duration": 25}
            for i in range(10)]
    assert learn_rules(build_rows(flat, [])) == []


def test_save_and_load_rules_roundtrip(learn_dirs):
    summary = summarize_learning(build_rows(ENTRIES, []), "Fixture")
    save_learned_rules(summary)
    loaded = load_learned_rules()
    assert loaded is not None
    assert loaded["shorts"] == 10
    assert loaded["rules"] == summary["rules"]


def test_load_rules_missing_is_none(learn_dirs):
    assert load_learned_rules() is None


# -------------------------------------------------------------------- sync

def test_sync_autofills_youtube_views(learn_dirs, fresh_db):
    import analytics

    clip_id = fresh_db["seed_clip"](
        streamer="speed", title="Speed finds KFC",
        youtube_title="Speed finds KFC")
    runner = FakeRun()
    result = sync_channel_performance(runner=runner)
    assert result["status"] == "ok"
    assert result["matched"] == 1
    assert result["recorded"] == 1
    rows = analytics.list_performance()
    assert len(rows) == 1
    assert rows[0]["clip_id"] == clip_id
    assert rows[0]["platform"] == "youtube"
    assert rows[0]["views"] == 700000


def test_sync_does_not_duplicate_unchanged_views(learn_dirs, fresh_db):
    fresh_db["seed_clip"](streamer="speed", title="Speed finds KFC",
                          youtube_title="Speed finds KFC")
    first = sync_channel_performance(runner=FakeRun())
    assert first["recorded"] == 1
    second = sync_channel_performance(runner=FakeRun(), refresh=True)
    assert second["recorded"] == 0  # same views: no new snapshot


def test_sync_records_changed_views(learn_dirs, fresh_db):
    fresh_db["seed_clip"](streamer="speed", title="Speed finds KFC",
                          youtube_title="Speed finds KFC")
    sync_channel_performance(runner=FakeRun())
    grown = json.loads(json.dumps(PAYLOAD))
    grown["entries"][1]["view_count"] = 750000
    result = sync_channel_performance(runner=FakeRun(payload=grown),
                                      refresh=True)
    assert result["recorded"] == 1


def test_sync_not_configured_without_url(learn_dirs, monkeypatch):
    monkeypatch.setattr(channel_learn, "CHANNEL_URL", "")
    result = sync_channel_performance(runner=FakeRun())
    assert result["status"] == "not_configured"


def test_sync_fetch_failure_is_an_error_dict(learn_dirs):
    result = sync_channel_performance(
        runner=FakeRun(returncode=1, stderr="no network"))
    assert result["status"] == "error"


def test_sync_skips_newest_when_undated(learn_dirs, fresh_db):
    undated = {"channel": "Fixture",
               "entries": [{**e, "timestamp": None} for e in ENTRIES]}
    result = sync_channel_performance(runner=FakeRun(payload=undated))
    # 10 entries minus the 3 newest (still gaining views, no dates).
    assert result["shorts"] == 7


def test_local_clip_index_reads_clips_table(fresh_db):
    clip_id = fresh_db["seed_clip"](streamer="xqc", title="My clip",
                                    youtube_title="My clip YT")
    index = local_clip_index()
    entry = next(e for e in index if e["clip_id"] == clip_id)
    assert entry["titles"] == ["My clip YT", "My clip"]
    assert entry["source"] == "xqc"


# --------------------------------------------------------------- tie-break

RULES = [
    {"kind": "source", "value": "speed", "ratio": 2.0, "samples": 4,
     "text": "Sources: speed gets 2.0x the median views (4 Shorts)."},
    {"kind": "length", "value": "20-30 s", "ratio": 1.6, "samples": 5,
     "text": "Length: 20-30 s gets 1.6x the median views (5 Shorts)."},
    {"kind": "name_first", "value": True, "ratio": 1.5, "samples": 6,
     "text": "Hooks with a name first: 1.5x the views of those without."},
]


def test_tiebreak_favours_what_the_channel_likes():
    delta = score_tiebreak("speed", 25, "Speed does a thing", rules=RULES)
    assert delta == 3  # +2 source, +1 length, +1 name-first, clamped at 3


def test_tiebreak_penalises_what_the_channel_dislikes():
    delta = score_tiebreak("speed", 45, "why this happened", rules=RULES)
    assert delta == 1  # +2 source, -1 missing the name-first hook


def test_tiebreak_no_rules_is_zero():
    assert score_tiebreak("speed", 25, "Speed does a thing", rules=[]) == 0


def test_tiebreak_loads_saved_rules(learn_dirs):
    save_learned_rules({"rules": RULES, "shorts": 10,
                        "updated_at": "2026-11-21T00:00:00+00:00"})
    assert score_tiebreak("speed", 25, "Speed does a thing") == 3


# ------------------------------------------------------- analytics exposure

def test_analytics_summary_exposes_learned_rules(learn_dirs, fresh_db):
    import analytics

    assert analytics.analytics_summary()["learned_rules"] == []
    save_learned_rules({"rules": RULES, "shorts": 10,
                        "updated_at": "2026-11-21T00:00:00+00:00"})
    summary = analytics.analytics_summary()
    assert summary["learned_rules"] == RULES
    assert summary["learned_rules_updated_at"] == "2026-11-21T00:00:00+00:00"


def test_refresh_youtube_performance_wrapper(learn_dirs, fresh_db,
                                             monkeypatch):
    import analytics

    # The wrapper has no runner hook, so only the not-configured path
    # is exercised here (no network in tests); the real sync is
    # covered by the sync tests above.
    monkeypatch.setattr(channel_learn, "CHANNEL_URL", "")
    result = analytics.refresh_youtube_performance(channel_url="")
    assert result["status"] == "not_configured"
