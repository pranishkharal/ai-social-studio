"""
Retry + resume helpers (feature 17).

Two pieces that make the clip pipeline survive transient failures instead
of marking a clip "failed" and stopping:

1. @retry — retry a function with exponential backoff.
2. CheckpointStore — record which pipeline steps already finished for a
   clip, so a failed run can resume from the step that broke instead of
   starting over.

Stdlib only. Importable standalone (config is imported lazily with a
fallback to this file's directory, so importing this module never fails
when python-dotenv is missing).

How the coordinator applies these to pipeline.py:

    from retry_utils import retry, CheckpointStore

    checkpoints = CheckpointStore()  # JSON files under BASE_DIR/checkpoints

    class ClipPipeline:
        @retry(max_attempts=3, base_delay=2.0,
               retryable_exceptions=(TimeoutError, ConnectionError))
        def _call_gemini(self, ...):
            ...  # transient Gemini timeouts are retried automatically

        def process_moment(self, streamer, ...):
            clip_id = ...
            done = checkpoints.get_completed_steps(clip_id)
            if "extract" not in done:
                self._extract(...)
                checkpoints.mark_step_done(clip_id, "extract")
            if "transcribe" not in done:
                self._transcribe(...)
                checkpoints.mark_step_done(clip_id, "transcribe")
            ...
            checkpoints.clear(clip_id)  # all steps done; drop the file
"""

import functools
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from config import BASE_DIR
except Exception:  # config needs python-dotenv; fall back to this file's dir
    BASE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS_DIR = BASE_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# @retry
# ---------------------------------------------------------------------------

def retry(_func=None, *, max_attempts=3, base_delay=1.0, backoff=2.0,
          max_delay=60.0, retryable_exceptions=(Exception,), on_retry=None):
    """Retry a function on transient failures, with exponential backoff.

    Usable bare (@retry) or with arguments (@retry(max_attempts=5, ...)).

    Args:
        max_attempts: total attempts including the first (must be >= 1).
        base_delay: seconds to wait before the first retry.
        backoff: delay multiplier after each failed attempt.
        max_delay: upper bound (seconds) for any single wait.
        retryable_exceptions: tuple of exception types worth retrying;
            anything else propagates immediately.
        on_retry: optional callback(attempt, exception, delay_seconds)
            invoked before each retry (attempt is 1-based, counting the
            attempt that just failed).

    The last exception is re-raised once attempts are exhausted.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        break
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        func.__name__, attempt, max_attempts, exc, delay,
                    )
                    if on_retry is not None:
                        try:
                            on_retry(attempt, exc, delay)
                        except Exception:
                            logger.exception("on_retry callback failed; ignoring")
                    time.sleep(delay)
                    delay = min(delay * backoff, max_delay)
            raise last_exc

        return wrapper

    if _func is not None:  # used bare, as @retry
        return decorator(_func)
    return decorator


# ---------------------------------------------------------------------------
# CheckpointStore
# ---------------------------------------------------------------------------

class CheckpointStore:
    """Per-clip JSON checkpoints so a failed run can resume mid-pipeline.

    One file per clip: <checkpoints_dir>/clip_<id>.json, shaped like:
        {"clip_id": 12, "steps": {"extract": {"done": true, "data": {...},
                                              "completed_at": "..."}},
         "updated_at": "..."}

    Writes are atomic (temp file + os.replace) so a crash mid-write never
    leaves a half-written checkpoint behind.
    """

    def __init__(self, checkpoints_dir=None):
        self.dir = Path(checkpoints_dir) if checkpoints_dir else DEFAULT_CHECKPOINTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, clip_id) -> Path:
        return self.dir / f"clip_{clip_id}.json"

    def _read(self, clip_id) -> dict:
        path = self._path(clip_id)
        if not path.exists():
            return {"clip_id": clip_id, "steps": {}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("steps"), dict):
                return data
        except (OSError, ValueError):
            logger.warning("Checkpoint for clip %s is corrupt; starting fresh", clip_id)
        return {"clip_id": clip_id, "steps": {}}

    def _write(self, clip_id, data: dict):
        path = self._path(clip_id)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def mark_step_done(self, clip_id, step, data=None):
        """Record that `step` finished for `clip_id` (with optional `data`).

        `data` must be JSON-serialisable — e.g. the path of a file the
        step produced, so the resume path can reuse it.
        """
        record = self._read(clip_id)
        record["steps"][step] = {
            "done": True,
            "data": data,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(clip_id, record)

    def get_completed_steps(self, clip_id) -> list:
        """Step names (in completion order) already done for `clip_id`."""
        record = self._read(clip_id)
        return [name for name, info in record["steps"].items() if info.get("done")]

    def is_step_done(self, clip_id, step) -> bool:
        """True if `step` already completed for `clip_id`."""
        record = self._read(clip_id)
        info = record["steps"].get(step)
        return bool(info and info.get("done"))

    def get_step_data(self, clip_id, step):
        """The `data` recorded with mark_step_done(), or None."""
        record = self._read(clip_id)
        info = record["steps"].get(step)
        return info.get("data") if info else None

    def clear(self, clip_id):
        """Delete the checkpoint for `clip_id` (call when the run finishes)."""
        try:
            self._path(clip_id).unlink()
        except FileNotFoundError:
            pass
