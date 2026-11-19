"""Tests for clip_dedup.py (Tier 1, feature 4: cooldown + duplicates).

Self-contained: this VM lacks python-dotenv, and importing the real
database.py would create the repo's real kick_engine.db, so both are
stubbed in sys.modules before importing the module under test -- but only
if tests/conftest.py has not already provided them (it stubs dotenv and
points the real database at a throwaway file; reusing that keeps this file
a good citizen in combined pytest runs). Stubs this file installs itself
are removed again in teardown_module. All DB work here runs against a real
SQLite file under tmp_path.
"""

import sqlite3
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- test-only stubs (see module docstring) ---
_INSTALLED_STUBS = []


def _install_stub(name, module):
    """Install *module* under *name* only if nothing is there yet."""
    if name not in sys.modules:
        sys.modules[name] = module
        _INSTALLED_STUBS.append(name)


_dotenv_stub = types.ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **k: False
_install_stub("dotenv", _dotenv_stub)

_database_stub = types.ModuleType("database")
_database_stub.log_event = lambda *a, **k: None
_database_stub.update_streamer_status = lambda *a, **k: None
_database_stub.add_notification = lambda *a, **k: 0
_database_stub.add_clip = lambda *a, **k: 0


def _no_db(*a, **k):
    raise RuntimeError("database stubbed in tests; monkeypatch get_connection")


_database_stub.get_connection = _no_db
_database_stub.get_all_clips = lambda: []
_install_stub("database", _database_stub)


def teardown_module(module):
    for name in _INSTALLED_STUBS:
        sys.modules.pop(name, None)


import clip_dedup  # noqa: E402
from clip_dedup import ClipCooldown, jaccard_similarity  # noqa: E402

# Minimal copy of the real clips-table columns the dedup queries touch.
CLIPS_SCHEMA = """
CREATE TABLE clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streamer TEXT NOT NULL,
    title TEXT,
    story_summary TEXT,
    caption TEXT,
    status TEXT DEFAULT 'detected',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""


def _utcnow_sql(offset_seconds=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
            ).strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """A real SQLite DB file under tmp_path, wired in as get_connection."""
    db_file = tmp_path / "test_clips.db"
    conn = sqlite3.connect(db_file)
    conn.execute(CLIPS_SCHEMA)
    conn.commit()
    conn.close()

    def _get_connection():
        c = sqlite3.connect(db_file)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(clip_dedup, "get_connection", _get_connection)
    return db_file


def _insert_clip(db_file, streamer, title="", story="", caption="",
                 status="ready_to_upload", age_seconds=0):
    conn = sqlite3.connect(db_file)
    conn.execute(
        "INSERT INTO clips (streamer, title, story_summary, caption, status,"
        " created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (streamer, title, story, caption, status, _utcnow_sql(-age_seconds)),
    )
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# jaccard_similarity
# ----------------------------------------------------------------------
def test_jaccard_identical_is_one():
    assert jaccard_similarity("hello brave new world", "hello brave new world") == 1.0


def test_jaccard_disjoint_is_zero():
    assert jaccard_similarity("alpha beta gamma", "one two three") == 0.0


def test_jaccard_partial_overlap():
    # {a,b,c} vs {b,c,d} -> 2/4 = 0.5
    assert jaccard_similarity("a b c", "b c d") == 0.5


def test_jaccard_empty_is_zero():
    assert jaccard_similarity("", "something here") == 0.0
    assert jaccard_similarity("", "") == 0.0


def test_jaccard_case_insensitive():
    assert jaccard_similarity("Hello World", "hello world") == 1.0


# ----------------------------------------------------------------------
# can_clip -- in-memory cooldown
# ----------------------------------------------------------------------
def test_can_clip_fresh_streamer_allowed(tmp_db):
    cd = ClipCooldown(cooldown_seconds=90)
    allowed, _ = cd.can_clip("xqc")
    assert allowed is True


def test_can_clip_blocked_after_record(tmp_db):
    cd = ClipCooldown(cooldown_seconds=90)
    cd.record_clip("xqc")
    allowed, reason = cd.can_clip("xqc")
    assert allowed is False
    assert "cooldown" in reason


def test_can_clip_other_streamer_unaffected(tmp_db):
    cd = ClipCooldown(cooldown_seconds=90)
    cd.record_clip("xqc")
    allowed, _ = cd.can_clip("adinross")
    assert allowed is True


def test_can_clip_case_insensitive_streamer(tmp_db):
    cd = ClipCooldown(cooldown_seconds=90)
    cd.record_clip("XQC")
    allowed, _ = cd.can_clip("xqc")
    assert allowed is False


def test_can_clip_expires(monkeypatch, tmp_db):
    cd = ClipCooldown(cooldown_seconds=10)
    cd.record_clip("xqc")
    # Pretend 11 seconds passed by moving the stored timestamp back.
    import time
    with cd._lock:
        cd._last_clip_at["xqc"] = time.monotonic() - 11
    allowed, _ = cd.can_clip("xqc")
    assert allowed is True


def test_seconds_until_ready(tmp_db):
    cd = ClipCooldown(cooldown_seconds=90)
    assert cd.seconds_until_ready("xqc") == 0.0
    cd.record_clip("xqc")
    assert 0 < cd.seconds_until_ready("xqc") <= 90


def test_can_clip_empty_streamer_denied(tmp_db):
    cd = ClipCooldown()
    allowed, _ = cd.can_clip("")
    assert allowed is False


# ----------------------------------------------------------------------
# can_clip -- DB check
# ----------------------------------------------------------------------
def test_can_clip_blocked_by_recent_db_clip(tmp_db):
    _insert_clip(tmp_db, "xqc", title="some clip", age_seconds=30)
    cd = ClipCooldown(cooldown_seconds=90)
    allowed, reason = cd.can_clip("xqc")
    assert allowed is False
    assert "cooldown" in reason


def test_can_clip_allowed_when_db_clip_is_old(tmp_db):
    _insert_clip(tmp_db, "xqc", title="old clip", age_seconds=3600)
    cd = ClipCooldown(cooldown_seconds=90)
    allowed, _ = cd.can_clip("xqc")
    assert allowed is True


def test_can_clip_failed_db_clip_does_not_block(tmp_db):
    _insert_clip(tmp_db, "xqc", title="failed clip", status="failed",
                 age_seconds=10)
    cd = ClipCooldown(cooldown_seconds=90)
    allowed, _ = cd.can_clip("xqc")
    assert allowed is True


def test_can_clip_processing_db_clip_blocks(tmp_db):
    _insert_clip(tmp_db, "xqc", title="in progress", status="processing",
                 age_seconds=10)
    cd = ClipCooldown(cooldown_seconds=90)
    allowed, _ = cd.can_clip("xqc")
    assert allowed is False


def test_can_clip_db_error_fails_open(monkeypatch):
    def _boom():
        raise sqlite3.OperationalError("no such table: clips")

    monkeypatch.setattr(clip_dedup, "get_connection", _boom)
    cd = ClipCooldown(cooldown_seconds=90)
    allowed, reason = cd.can_clip("xqc")
    assert allowed is True
    assert "db unavailable" in reason


# ----------------------------------------------------------------------
# is_duplicate
# ----------------------------------------------------------------------
def test_is_duplicate_same_words(tmp_db):
    transcript = "bro that was absolutely insane what just happened chat"
    _insert_clip(tmp_db, "xqc",
                 title="BRO THAT WAS ABSOLUTELY INSANE",
                 story="what just happened chat",
                 age_seconds=60)
    cd = ClipCooldown(similarity_threshold=0.7)
    dup, reason = cd.is_duplicate("xqc", transcript)
    assert dup is True
    assert "similarity" in reason


def test_is_duplicate_different_content(tmp_db):
    _insert_clip(tmp_db, "xqc",
                 title="cooking stream pasta recipe",
                 story="making dinner with chat tonight",
                 age_seconds=60)
    cd = ClipCooldown(similarity_threshold=0.7)
    dup, _ = cd.is_duplicate(
        "xqc", "bro that was absolutely insane what just happened chat")
    assert dup is False


def test_is_duplicate_old_clip_ignored(tmp_db):
    transcript = "bro that was absolutely insane what just happened chat"
    _insert_clip(tmp_db, "xqc",
                 title="BRO THAT WAS ABSOLUTELY INSANE",
                 story="what just happened chat",
                 age_seconds=3600)  # outside the 30-min window
    cd = ClipCooldown(similarity_threshold=0.7, recent_window_minutes=30)
    dup, _ = cd.is_duplicate("xqc", transcript)
    assert dup is False


def test_is_duplicate_other_streamer_ignored(tmp_db):
    transcript = "bro that was absolutely insane what just happened chat"
    _insert_clip(tmp_db, "adinross",
                 title="BRO THAT WAS ABSOLUTELY INSANE",
                 story="what just happened chat",
                 age_seconds=60)
    cd = ClipCooldown(similarity_threshold=0.7)
    dup, _ = cd.is_duplicate("xqc", transcript)
    assert dup is False


def test_is_duplicate_short_transcript_never_matches(tmp_db):
    _insert_clip(tmp_db, "xqc", title="wow bro", age_seconds=60)
    cd = ClipCooldown(similarity_threshold=0.7)
    dup, reason = cd.is_duplicate("xqc", "wow bro")
    assert dup is False
    assert "too short" in reason


def test_is_duplicate_empty_transcript(tmp_db):
    cd = ClipCooldown()
    dup, _ = cd.is_duplicate("xqc", "")
    assert dup is False


def test_is_duplicate_db_error_fails_open(monkeypatch):
    def _boom():
        raise sqlite3.OperationalError("no such table: clips")

    monkeypatch.setattr(clip_dedup, "get_connection", _boom)
    cd = ClipCooldown()
    dup, reason = cd.is_duplicate(
        "xqc", "bro that was absolutely insane what just happened chat")
    assert dup is False
    assert reason == "db unavailable"


def test_env_var_overrides(monkeypatch, tmp_db):
    monkeypatch.setenv("AUTO_CLIP_COOLDOWN_SECONDS", "45")
    monkeypatch.setenv("AUTO_CLIP_DUP_SIMILARITY", "0.5")
    monkeypatch.setenv("AUTO_CLIP_DUP_WINDOW_MINUTES", "10")
    cd = ClipCooldown()
    assert cd.cooldown_seconds == 45
    assert cd.similarity_threshold == 0.5
    assert cd.recent_window_minutes == 10


def test_env_var_invalid_values_fall_back_to_defaults(monkeypatch, tmp_db):
    monkeypatch.setenv("AUTO_CLIP_COOLDOWN_SECONDS", "not-a-number")
    monkeypatch.setenv("AUTO_CLIP_DUP_SIMILARITY", "nah")
    cd = ClipCooldown()
    assert cd.cooldown_seconds == 90
    assert cd.similarity_threshold == 0.7
