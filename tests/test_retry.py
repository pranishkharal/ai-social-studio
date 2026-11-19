"""Tests for retry_utils.py (feature 17) — @retry and CheckpointStore."""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retry_utils import CheckpointStore, retry  # noqa: E402


# --- @retry -----------------------------------------------------------------

def test_retry_succeeds_on_third_try():
    attempts = []

    @retry(max_attempts=3, base_delay=0)
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("transient")
        return "done"

    assert flaky() == "done"
    assert len(attempts) == 3


def test_retry_gives_up_after_max_attempts():
    attempts = []

    @retry(max_attempts=3, base_delay=0)
    def always_fails():
        attempts.append(1)
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        always_fails()
    assert len(attempts) == 3


def test_retry_ignores_non_retryable_exception():
    attempts = []

    @retry(max_attempts=5, base_delay=0, retryable_exceptions=(ValueError,))
    def wrong_kind():
        attempts.append(1)
        raise TypeError("not retryable")

    with pytest.raises(TypeError):
        wrong_kind()
    assert len(attempts) == 1  # no retries for a non-retryable error


def test_retry_on_retry_callback():
    calls = []

    @retry(max_attempts=3, base_delay=0.5, backoff=1.0,
           on_retry=lambda attempt, exc, delay: calls.append((attempt, str(exc), delay)))
    def flaky():
        if len(calls) < 2:
            raise RuntimeError("flaky")
        return "ok"

    assert flaky() == "ok"
    assert calls == [(1, "flaky", 0.5), (2, "flaky", 0.5)]


def test_retry_usable_bare_without_arguments():
    @retry
    def fine():
        return 42

    assert fine() == 42


def test_retry_rejects_zero_max_attempts():
    with pytest.raises(ValueError):
        retry(max_attempts=0)


# --- CheckpointStore --------------------------------------------------------

def test_checkpoint_roundtrip(tmp_path):
    store = CheckpointStore(tmp_path)
    store.mark_step_done(7, "extract")
    store.mark_step_done(7, "transcribe", data={"words": 120})

    assert store.get_completed_steps(7) == ["extract", "transcribe"]
    assert store.is_step_done(7, "extract") is True
    assert store.is_step_done(7, "render") is False
    assert store.get_step_data(7, "transcribe") == {"words": 120}
    assert store.get_step_data(7, "render") is None

    # The checkpoint is a real JSON file on disk, named per clip.
    checkpoint_file = tmp_path / "clip_7.json"
    assert checkpoint_file.exists()
    assert json.loads(checkpoint_file.read_text())["steps"]["extract"]["done"] is True


def test_checkpoint_unknown_clip_is_empty(tmp_path):
    store = CheckpointStore(tmp_path)
    assert store.get_completed_steps(999) == []
    assert store.is_step_done(999, "extract") is False


def test_checkpoint_persists_across_instances(tmp_path):
    CheckpointStore(tmp_path).mark_step_done(3, "extract")
    # A fresh instance on the same directory sees the same steps.
    assert CheckpointStore(tmp_path).get_completed_steps(3) == ["extract"]


def test_checkpoint_clear(tmp_path):
    store = CheckpointStore(tmp_path)
    store.mark_step_done(5, "extract")
    assert (tmp_path / "clip_5.json").exists()
    store.clear(5)
    assert not (tmp_path / "clip_5.json").exists()
    assert store.get_completed_steps(5) == []
    store.clear(5)  # clearing a missing checkpoint is a no-op, not an error


def test_checkpoint_corrupt_file_starts_fresh(tmp_path):
    (tmp_path / "clip_9.json").write_text("{not valid json")
    store = CheckpointStore(tmp_path)
    assert store.get_completed_steps(9) == []
    store.mark_step_done(9, "extract")
    assert store.get_completed_steps(9) == ["extract"]
