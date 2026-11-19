"""Tests for auto_clip_watcher.py (Tier 1, feature 1: auto-clip watcher)
and the KickMonitor.list_buffer_chunks helper it relies on.

Self-contained: this VM lacks python-dotenv, and importing the real
database.py would create the repo's real kick_engine.db, so both are
stubbed in sys.modules before importing the modules under test -- but only
if tests/conftest.py has not already provided them (it stubs dotenv and
points the real database at a throwaway file; reusing that keeps this file
a good citizen in combined pytest runs). Stubs this file installs itself
are removed again in teardown_module. ffmpeg/subprocess is mocked; buffer
chunks are fake files under tmp_path; the pipeline is a fake.
"""

import os
import subprocess
import sys
import threading
import time
import types
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
_database_stub.get_connection = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("database stubbed in tests"))
_database_stub.get_all_clips = lambda: []
_install_stub("database", _database_stub)


def teardown_module(module):
    for name in _INSTALLED_STUBS:
        sys.modules.pop(name, None)


import auto_clip_watcher  # noqa: E402
import kick_monitor  # noqa: E402
from auto_clip_watcher import AutoClipWatcher  # noqa: E402
from clip_dedup import ClipCooldown  # noqa: E402


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeProc:
    """Stand-in for a recorder subprocess.Popen."""

    def __init__(self, alive=True):
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


class FakeMonitor:
    def __init__(self):
        self.active_recorders = {}
        self.streamer_cache = {}
        self._chunks = {}  # username -> [Path, ...] newest first

    def list_buffer_chunks(self, username):
        return list(self._chunks.get(username, []))

    def set_chunks(self, username, paths):
        self._chunks[username] = list(paths)


class FakeDetector:
    def __init__(self, score=0.0):
        self.score = score
        self.calls = []

    def analyze_audio_energy(self, video_path):
        self.calls.append(video_path)
        return {"score": self.score, "is_hype": self.score >= 70}


class FakePipeline:
    def __init__(self, result=None, block_event=None):
        self.calls = []
        self.result = result or {"status": "ready_to_upload", "clip_id": 7}
        self.block_event = block_event  # if set, process_moment waits on it
        self.call_started = threading.Event()

    def process_moment(self, streamer, duration=25, **kwargs):
        self.calls.append((streamer, duration))
        self.call_started.set()
        if self.block_event is not None:
            self.block_event.wait(timeout=10)
        return dict(self.result)


class FakeChatClient:
    def __init__(self, scores=None):
        self.scores = scores or {}
        self.subscribed = []

    def subscribe(self, chatroom_id):
        self.subscribed.append(chatroom_id)

    def get_hype_score(self, chatroom_id):
        return self.scores.get(chatroom_id, 0.0)


class FakeCooldown(ClipCooldown):
    """ClipCooldown with the DB check disabled (in-memory only)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)

    def can_clip(self, streamer):
        streamer = (streamer or "").strip().lower()
        with self._lock:
            last = self._last_clip_at.get(streamer)
        if last is not None and (time.monotonic() - last) < self.cooldown_seconds:
            return False, "cooldown (fake)"
        return True, "ok"


def _make_chunk(tmp_path, name, size=1024):
    path = tmp_path / name
    path.write_bytes(b"\x47" * size)  # fake .ts bytes
    return path


def _make_watcher(monitor=None, pipeline=None, detector=None,
                  cooldown=None, chat_client=None, **kwargs):
    monitor = monitor if monitor is not None else FakeMonitor()
    pipeline = pipeline if pipeline is not None else FakePipeline()
    detector = detector if detector is not None else FakeDetector(score=0.0)
    cooldown = cooldown if cooldown is not None else FakeCooldown(
        cooldown_seconds=90)
    defaults = dict(enabled=True, min_score=70.0, check_seconds=5,
                    duration=25)
    defaults.update(kwargs)
    return AutoClipWatcher(
        monitor, pipeline, detector=detector, cooldown=cooldown,
        chat_client=chat_client, **defaults)


def _wait_for(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ----------------------------------------------------------------------
# Sampling / scoring
# ----------------------------------------------------------------------
def test_check_streamer_no_chunks_scores_zero(tmp_path):
    monitor = FakeMonitor()
    monitor.active_recorders["xqc"] = FakeProc()
    watcher = _make_watcher(monitor=monitor, detector=FakeDetector(score=95.0))
    sample = watcher.check_streamer("xqc")
    assert sample["score"] == 0.0
    assert sample["audio_score"] == 0.0


def test_check_streamer_below_threshold_no_clip(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    pipeline = FakePipeline()
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            detector=FakeDetector(score=40.0))
    # ffmpeg slice fails in tests (no real media) -> falls back to the chunk.
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    sample = watcher.check_streamer("xqc")
    assert sample["score"] == 40.0
    assert sample["source"] == "audio"
    assert pipeline.calls == []


def test_check_streamer_above_threshold_fires_clip(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    pipeline = FakePipeline()
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            detector=FakeDetector(score=95.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    sample = watcher.check_streamer("xqc")
    assert sample["score"] == 95.0
    assert _wait_for(lambda: len(pipeline.calls) == 1)
    assert pipeline.calls[0] == ("xqc", 25)


def test_check_streamer_uses_newest_chunk_first(tmp_path, monkeypatch):
    old = _make_chunk(tmp_path, "chunk_001.ts")
    new = _make_chunk(tmp_path, "chunk_002.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [new, old])  # newest first
    detector = FakeDetector(score=10.0)
    watcher = _make_watcher(monitor=monitor, detector=detector)
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    watcher.check_streamer("xqc")
    # Slice failed (mocked ffmpeg wrote nothing), so the whole newest chunk
    # was analysed -- never the older one.
    assert detector.calls == [str(new)]


def test_check_streamer_chat_blending(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    monitor.streamer_cache["xqc"] = {"chatroom_id": 2001}
    chat = FakeChatClient(scores={2001: 90.0})
    watcher = _make_watcher(monitor=monitor, chat_client=chat,
                            detector=FakeDetector(score=50.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    sample = watcher.check_streamer("xqc")
    # 0.6 * 50 + 0.4 * 90 = 66
    assert sample["score"] == 66.0
    assert sample["audio_score"] == 50.0
    assert sample["chat_score"] == 90.0
    assert sample["source"] == "audio+chat"
    assert 2001 in chat.subscribed


def test_check_streamer_chat_can_push_over_threshold(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    monitor.streamer_cache["xqc"] = {"chatroom_id": 2001}
    chat = FakeChatClient(scores={2001: 100.0})
    pipeline = FakePipeline()
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            chat_client=chat, detector=FakeDetector(score=60.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    sample = watcher.check_streamer("xqc")
    # 0.6 * 60 + 0.4 * 100 = 76 >= 70 -> clip fires.
    assert sample["score"] == 76.0
    assert _wait_for(lambda: len(pipeline.calls) == 1)


def test_check_streamer_no_chatroom_id_is_audio_only(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    # No streamer_cache entry -> no chatroom id -> chat skipped.
    chat = FakeChatClient(scores={2001: 100.0})
    watcher = _make_watcher(monitor=monitor, chat_client=chat,
                            detector=FakeDetector(score=55.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    sample = watcher.check_streamer("xqc")
    assert sample["score"] == 55.0
    assert sample["chat_score"] is None
    assert sample["source"] == "audio"


# ----------------------------------------------------------------------
# Cooldown / overlap guards
# ----------------------------------------------------------------------
def test_cooldown_blocks_second_clip(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    pipeline = FakePipeline()
    cooldown = FakeCooldown(cooldown_seconds=90)
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            cooldown=cooldown,
                            detector=FakeDetector(score=95.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    watcher.check_streamer("xqc")
    assert _wait_for(lambda: len(pipeline.calls) == 1)
    # Cooldown is recorded only after the worker finishes successfully.
    assert _wait_for(lambda: cooldown.seconds_until_ready("xqc") > 0)
    watcher.check_streamer("xqc")
    time.sleep(0.1)
    assert len(pipeline.calls) == 1  # no second clip


def test_failed_clip_does_not_start_cooldown(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    pipeline = FakePipeline(result={"status": "failed",
                                    "error": "footage_unavailable"})
    cooldown = FakeCooldown(cooldown_seconds=90)
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            cooldown=cooldown,
                            detector=FakeDetector(score=95.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    watcher.check_streamer("xqc")
    assert _wait_for(lambda: len(pipeline.calls) == 1)
    time.sleep(0.1)  # let the worker finish
    assert cooldown.seconds_until_ready("xqc") == 0.0


def test_overlapping_clips_for_same_streamer_blocked(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    release = threading.Event()
    pipeline = FakePipeline(block_event=release)
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            detector=FakeDetector(score=95.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    watcher.check_streamer("xqc")
    assert pipeline.call_started.wait(timeout=5)
    # A second high score while the first clip is still running: blocked.
    watcher.check_streamer("xqc")
    watcher.check_streamer("xqc")
    time.sleep(0.1)
    assert len(pipeline.calls) == 1
    release.set()
    assert _wait_for(lambda: "xqc" not in watcher._in_progress)
    # After the worker finishes, a new clip may fire (cooldown was recorded
    # on success, so this one is blocked by cooldown instead -- use a fresh
    # watcher with a fresh cooldown to prove the in-progress guard cleared).
    watcher2 = _make_watcher(monitor=monitor, pipeline=pipeline,
                             detector=FakeDetector(score=95.0))
    watcher2.check_streamer("xqc")
    assert _wait_for(lambda: len(pipeline.calls) == 2)


def test_watcher_never_blocks_on_slow_pipeline(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    release = threading.Event()
    pipeline = FakePipeline(block_event=release)
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            detector=FakeDetector(score=95.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    start = time.monotonic()
    watcher.check_streamer("xqc")
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # returned immediately; clip runs in a worker
    release.set()


def test_cooldown_consulted_before_clipping(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    pipeline = FakePipeline()
    cooldown = FakeCooldown(cooldown_seconds=90)
    cooldown.record_clip("xqc")  # already on cooldown
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            cooldown=cooldown,
                            detector=FakeDetector(score=99.0))
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    watcher.check_streamer("xqc")
    time.sleep(0.1)
    assert pipeline.calls == []


# ----------------------------------------------------------------------
# Sample slicing
# ----------------------------------------------------------------------
def test_cut_sample_slice_success(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")

    def _fake_run(cmd, capture_output=None, timeout=None, creationflags=0):
        assert cmd[0] == "ffmpeg"
        assert "-sseof" in cmd
        out = cmd[-1]
        Path(out).write_bytes(b"\x47" * 512)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(auto_clip_watcher.subprocess, "run", _fake_run)
    watcher = _make_watcher(sample_seconds=5)
    sliced = watcher._cut_sample_slice(str(chunk))
    assert sliced is not None
    assert sliced.endswith(".ts")
    assert os.path.getsize(sliced) == 512
    os.remove(sliced)


def test_cut_sample_slice_failure_returns_none(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")

    def _boom(*a, **k):
        raise OSError("ffmpeg not found")

    monkeypatch.setattr(auto_clip_watcher.subprocess, "run", _boom)
    watcher = _make_watcher()
    assert watcher._cut_sample_slice(str(chunk)) is None


def test_sample_temp_file_is_cleaned_up(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    created = []

    def _fake_run(cmd, capture_output=None, timeout=None, creationflags=0):
        out = cmd[-1]
        Path(out).write_bytes(b"\x47" * 512)
        created.append(out)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(auto_clip_watcher.subprocess, "run", _fake_run)
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    watcher = _make_watcher(monitor=monitor, detector=FakeDetector(score=10.0))
    watcher.check_streamer("xqc")
    assert created
    assert not os.path.exists(created[0])


# ----------------------------------------------------------------------
# Lifecycle / env flags
# ----------------------------------------------------------------------
def test_start_returns_false_when_disabled():
    watcher = _make_watcher(enabled=False)
    assert watcher.start() is False
    assert watcher.running is False


def test_start_stop_lifecycle():
    monitor = FakeMonitor()
    watcher = _make_watcher(monitor=monitor, enabled=True, check_seconds=1)
    try:
        assert watcher.start() is True
        assert watcher.running is True
        assert watcher.start() is True  # idempotent
    finally:
        watcher.stop()
    assert watcher.running is False


def test_watch_loop_samples_live_streamers(tmp_path, monkeypatch):
    chunk = _make_chunk(tmp_path, "chunk_001.ts")
    monitor = FakeMonitor()
    monitor.set_chunks("xqc", [chunk])
    monitor.active_recorders["xqc"] = FakeProc(alive=True)
    monitor.active_recorders["dead"] = FakeProc(alive=False)
    pipeline = FakePipeline()
    detector = FakeDetector(score=95.0)
    watcher = _make_watcher(monitor=monitor, pipeline=pipeline,
                            detector=detector, enabled=True, check_seconds=1)
    monkeypatch.setattr(auto_clip_watcher.subprocess, "run",
                        lambda *a, **k: None)
    try:
        watcher.start()
        assert _wait_for(lambda: len(pipeline.calls) == 1, timeout=5)
        assert pipeline.calls[0][0] == "xqc"  # only the live recorder
    finally:
        watcher.stop()


def test_env_flags(monkeypatch):
    monkeypatch.setenv("AUTO_CLIP_ENABLED", "true")
    monkeypatch.setenv("AUTO_CLIP_MIN_SCORE", "80")
    monkeypatch.setenv("AUTO_CLIP_CHECK_SECONDS", "3")
    monkeypatch.setenv("AUTO_CLIP_DURATION", "45")
    monkeypatch.setenv("AUTO_CLIP_SAMPLE_SECONDS", "8")
    watcher = AutoClipWatcher(FakeMonitor(), FakePipeline())
    assert watcher.enabled is True
    assert watcher.min_score == 80.0
    assert watcher.check_seconds == 3
    assert watcher.duration == 45
    assert watcher.sample_seconds == 8


def test_env_flags_defaults(monkeypatch):
    for var in ("AUTO_CLIP_ENABLED", "AUTO_CLIP_MIN_SCORE",
                "AUTO_CLIP_CHECK_SECONDS", "AUTO_CLIP_DURATION",
                "AUTO_CLIP_SAMPLE_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    watcher = AutoClipWatcher(FakeMonitor(), FakePipeline())
    assert watcher.enabled is False  # off by default: behaviour unchanged
    assert watcher.min_score == 70.0
    assert watcher.check_seconds == 5
    assert watcher.duration == 25
    assert watcher.sample_seconds == 5


def test_live_buffered_streamers_only_counts_alive_recorders():
    monitor = FakeMonitor()
    monitor.active_recorders["live1"] = FakeProc(alive=True)
    monitor.active_recorders["live2"] = None  # tolerate a None proc
    monitor.active_recorders["dead"] = FakeProc(alive=False)
    watcher = _make_watcher(monitor=monitor)
    assert sorted(watcher._live_buffered_streamers()) == ["live1", "live2"]


# ----------------------------------------------------------------------
# KickMonitor.list_buffer_chunks (used by the watcher)
# ----------------------------------------------------------------------
def test_list_buffer_chunks_newest_first(tmp_path, monkeypatch):
    buf_dir = tmp_path / "buffer"
    streamer_dir = buf_dir / "xqc"
    streamer_dir.mkdir(parents=True)
    oldest = streamer_dir / "chunk_001.ts"
    newest = streamer_dir / "chunk_002.ts"
    oldest.write_bytes(b"x")
    newest.write_bytes(b"x")
    old_time = time.time() - 120
    os.utime(oldest, (old_time, old_time))
    monkeypatch.setattr(kick_monitor, "BUFFER_DIR", buf_dir)
    monitor = kick_monitor.KickMonitor()
    chunks = monitor.list_buffer_chunks("xqc")
    assert chunks == [newest, oldest]


def test_list_buffer_chunks_missing_streamer(tmp_path, monkeypatch):
    monkeypatch.setattr(kick_monitor, "BUFFER_DIR", tmp_path / "buffer")
    monitor = kick_monitor.KickMonitor()
    assert monitor.list_buffer_chunks("nobody") == []


def test_list_buffer_chunks_ignores_non_chunks(tmp_path, monkeypatch):
    buf_dir = tmp_path / "buffer"
    streamer_dir = buf_dir / "xqc"
    streamer_dir.mkdir(parents=True)
    (streamer_dir / "chunk_001.ts").write_bytes(b"x")
    (streamer_dir / "notes.txt").write_bytes(b"x")
    (streamer_dir / "other.mp4").write_bytes(b"x")
    monkeypatch.setattr(kick_monitor, "BUFFER_DIR", buf_dir)
    monitor = kick_monitor.KickMonitor()
    chunks = monitor.list_buffer_chunks("xqc")
    assert len(chunks) == 1
    assert chunks[0].name == "chunk_001.ts"
