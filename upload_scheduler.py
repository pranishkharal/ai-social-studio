"""
Upload scheduler / queue (Feature 6).

Clips approved during a long stream should not all be dumped at once — they
get spread across the day, per platform, with minimum gaps and daily caps.

How it works:
    - :func:`enqueue_clip` adds a clip to the ``upload_queue`` table. If the
      requested time is too close to the last queued post for that platform,
      it is pushed later so posts stay spaced out (see the *_MIN_GAP_MINUTES
      env vars).
    - :class:`UploadScheduler` is a background thread that picks up due jobs,
      checks the per-platform daily cap, and dispatches to the right uploader:
      TikTok via screen_controller (Playwright, manual confirm in the browser
      window), YouTube via youtube_uploader, Instagram via instagram_uploader.
    - Failed jobs retry with exponential backoff up to UPLOAD_MAX_ATTEMPTS.
    - Posted URLs are recorded back on the queue row, and successful
      YouTube/Instagram posts are also recorded in the earnings tracker
      (best-effort — a failure there never fails the upload job).

Env vars:
    TIKTOK_MIN_GAP_MINUTES     Min minutes between TikTok posts. Default: 180
    YOUTUBE_MIN_GAP_MINUTES    Min minutes between YouTube posts. Default: 120
    INSTAGRAM_MIN_GAP_MINUTES  Min minutes between Instagram posts. Default: 180
    TIKTOK_MAX_PER_DAY         Max TikTok posts per day. Default: 5
    YOUTUBE_MAX_PER_DAY        Max YouTube posts per day. Default: 5
    INSTAGRAM_MAX_PER_DAY      Max Instagram posts per day. Default: 5
    UPLOAD_MAX_ATTEMPTS        Max tries per job before it is marked failed.
                               Default: 3
    UPLOAD_SCHEDULER_POLL_SECONDS  How often the background thread checks
                               for due jobs. Default: 60

    Instagram jobs need a public video URL (the Graph API cannot read local
    files — see instagram_uploader.py). Pass it to enqueue_clip as
    ``video_url``; a job without one fails with a clear error message.
"""

import os
import threading
import time
from datetime import datetime, timedelta, timezone

from database import get_connection, log_event

PLATFORMS = ("tiktok", "youtube", "instagram")

DEFAULT_GAP_MINUTES = {"tiktok": 180, "youtube": 120, "instagram": 180}
DEFAULT_MAX_PER_DAY = {"tiktok": 5, "youtube": 5, "instagram": 5}

# Statuses: pending -> processing -> done | failed ; pending -> cancelled
STATUSES = ("pending", "processing", "done", "failed", "cancelled")


def init_upload_queue_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS upload_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id INTEGER NOT NULL,
            platform TEXT NOT NULL,
            scheduled_for TIMESTAMP NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            video_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            posted_url TEXT
        )
    """)
    # Safe migration for queues created before video_url existed.
    cols = [c[1] for c in conn.execute("PRAGMA table_info(upload_queue)").fetchall()]
    if "video_url" not in cols:
        conn.execute("ALTER TABLE upload_queue ADD COLUMN video_url TEXT")
    conn.commit()
    conn.close()


def _ensure_tables():
    init_upload_queue_table()


def _min_gap_minutes(platform: str) -> int:
    return int(os.getenv(f"{platform.upper()}_MIN_GAP_MINUTES",
                         DEFAULT_GAP_MINUTES[platform]))


def _max_per_day(platform: str) -> int:
    return int(os.getenv(f"{platform.upper()}_MAX_PER_DAY",
                         DEFAULT_MAX_PER_DAY[platform]))


def _max_attempts() -> int:
    return int(os.getenv("UPLOAD_MAX_ATTEMPTS", "3"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_dt(value) -> datetime:
    """Parses an ISO string / datetime into an aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _apply_spacing(platform: str, requested: datetime) -> datetime:
    """Pushes ``requested`` later if it is too close to the last queued post
    for this platform (minimum-gap rule)."""
    conn = get_connection()
    row = conn.execute("""
        SELECT MAX(scheduled_for) AS last_time FROM upload_queue
        WHERE platform = ? AND status IN ('pending', 'processing')
    """, (platform,)).fetchone()
    conn.close()
    if not row or not row["last_time"]:
        return requested
    earliest = _to_dt(row["last_time"]) + timedelta(minutes=_min_gap_minutes(platform))
    return max(requested, earliest)


def enqueue_clip(clip_id: int, platform: str, scheduled_for=None,
                 video_url: str = None) -> dict:
    """Queues a clip for upload. Returns the created queue row.

    Args:
        clip_id: ID of a rendered clip.
        platform: "tiktok", "youtube" or "instagram".
        scheduled_for: datetime / ISO string, or None for "as soon as
            possible" (now, subject to the spacing rule).
        video_url: Required for Instagram — the clip's public video URL.
    """
    _ensure_tables()
    platform = (platform or "").strip().lower()
    if platform not in PLATFORMS:
        raise ValueError(f"Unknown platform '{platform}': use tiktok, youtube or instagram")

    conn = get_connection()
    clip = conn.execute("SELECT id FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not clip:
        raise ValueError(f"Clip #{clip_id} not found")

    requested = _to_dt(scheduled_for) if scheduled_for else _now()
    scheduled = _apply_spacing(platform, requested)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO upload_queue (clip_id, platform, scheduled_for, status,
                                  attempts, video_url, created_at)
        VALUES (?, ?, ?, 'pending', 0, ?, ?)
    """, (clip_id, platform, _to_iso(scheduled), video_url,
          _now().strftime("%Y-%m-%d %H:%M:%S")))
    job_id = cursor.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM upload_queue WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if scheduled > requested:
        log_event("INFO", "UploadScheduler",
                  f"Clip #{clip_id} -> {platform} spaced to {scheduled.isoformat()} "
                  f"(min gap {_min_gap_minutes(platform)} min)")
    else:
        log_event("INFO", "UploadScheduler",
                  f"Clip #{clip_id} queued for {platform} at {scheduled.isoformat()}")
    return dict(row)


def get_queue_job(job_id: int):
    _ensure_tables()
    conn = get_connection()
    row = conn.execute("SELECT * FROM upload_queue WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_queue(status: str = None, platform: str = None, limit: int = 50) -> list:
    _ensure_tables()
    query = "SELECT * FROM upload_queue"
    clauses, params = [], []
    if status:
        clauses.append("status = ?")
        params.append(status.lower())
    if platform:
        clauses.append("platform = ?")
        params.append(platform.lower())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY scheduled_for ASC LIMIT ?"
    params.append(limit)
    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_due_jobs(limit: int = 10) -> list:
    """Pending jobs whose scheduled time has arrived, oldest first."""
    _ensure_tables()
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM upload_queue
        WHERE status = 'pending' AND scheduled_for <= ?
        ORDER BY scheduled_for ASC LIMIT ?
    """, (_to_iso(_now()), limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def cancel_job(job_id: int) -> dict:
    """Cancels a pending job. Returns the updated row (or None)."""
    _ensure_tables()
    conn = get_connection()
    conn.execute(
        "UPDATE upload_queue SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
        (job_id,))
    conn.commit()
    row = conn.execute("SELECT * FROM upload_queue WHERE id = ?", (job_id,)).fetchone()
    conn.close()
    if row and row["status"] == "cancelled":
        log_event("INFO", "UploadScheduler", f"Cancelled queue job #{job_id}")
    return dict(row) if row else None


def _posts_today(platform: str) -> int:
    """Jobs for this platform completed (or in progress) today, UTC."""
    today = _now().strftime("%Y-%m-%d")
    conn = get_connection()
    row = conn.execute("""
        SELECT COUNT(*) AS n FROM upload_queue
        WHERE platform = ? AND status IN ('done', 'processing')
          AND strftime('%Y-%m-%d', scheduled_for) = ?
    """, (platform, today)).fetchone()
    conn.close()
    return row["n"]


def _set_job(job_id: int, **fields):
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn = get_connection()
    conn.execute(f"UPDATE upload_queue SET {assignments} WHERE id = ?",
                 (*fields.values(), job_id))
    conn.commit()
    conn.close()


def _record_earnings_submission(clip_id: int, platform: str, posted_url: str):
    """Best-effort: log a successful post in the earnings tracker."""
    try:
        from earnings import record_submission
        record_submission(clip_id, platform, posted_url)
    except Exception as e:
        log_event("WARNING", "UploadScheduler",
                  f"Post succeeded but earnings recording failed: {e}")


def _dispatch_job(job: dict) -> str:
    """Uploads one job. Returns the posted URL (or None for TikTok, where the
    user clicks Post manually in the browser window). Raises on failure."""
    clip_id = job["clip_id"]
    platform = job["platform"]

    conn = get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Clip #{clip_id} no longer exists")
    clip = dict(row)
    video_path = clip.get("edited_path") or clip.get("raw_path")
    if not video_path:
        raise ValueError(f"Clip #{clip_id} has no rendered video file yet")

    if platform == "tiktok":
        # Lazy import: keeps Playwright out of the import path and lets tests
        # substitute the controller.
        from screen_controller import KickScreenController
        controller = KickScreenController()
        try:
            ok = controller.upload_to_tiktok(
                video_path, clip.get("caption") or clip.get("title") or "",
                clip.get("tags") or "", headless=False)
        finally:
            controller.close()
        if not ok:
            raise RuntimeError("TikTok staging failed (see ScreenController logs)")
        # The user clicks Post in the browser window, so there is no URL yet.
        return None

    if platform == "youtube":
        from youtube_uploader import upload_clip
        return upload_clip(clip_id)

    if platform == "instagram":
        video_url = job.get("video_url")
        if not video_url:
            raise ValueError(
                "Instagram job has no public video_url — the Graph API cannot "
                "read local files. Re-enqueue with the clip's public video URL."
            )
        from instagram_uploader import upload_clip_reel
        return upload_clip_reel(clip_id, video_url)

    raise ValueError(f"Unknown platform '{platform}'")


def _process_job(job: dict) -> dict:
    """Runs one job end to end: daily-cap check, dispatch, retry/backoff,
    status updates. Returns the final queue row."""
    job_id = job["id"]
    platform = job["platform"]

    if _posts_today(platform) >= _max_per_day(platform):
        # Push to tomorrow instead of failing: the cap is a pacing rule.
        new_time = (_now() + timedelta(days=1)).replace(
            hour=9, minute=0, second=0, microsecond=0)
        _set_job(job_id, scheduled_for=_to_iso(new_time),
                 last_error=f"Daily cap reached ({_max_per_day(platform)}/day); rescheduled")
        log_event("INFO", "UploadScheduler",
                  f"Job #{job_id} ({platform}) hit the daily cap; moved to {new_time.date()}")
        return get_queue_job(job_id)

    _set_job(job_id, status="processing")
    try:
        posted_url = _dispatch_job(job)
        _set_job(job_id, status="done", posted_url=posted_url, last_error=None)
        log_event("INFO", "UploadScheduler",
                  f"Job #{job_id} done: clip #{job['clip_id']} -> {platform} "
                  f"({posted_url or 'manual confirm in browser'})")
        if posted_url:
            _record_earnings_submission(job["clip_id"], platform, posted_url)
        return get_queue_job(job_id)
    except Exception as e:
        attempts = (job.get("attempts") or 0) + 1
        if attempts >= _max_attempts():
            _set_job(job_id, status="failed", attempts=attempts, last_error=str(e))
            log_event("ERROR", "UploadScheduler",
                      f"Job #{job_id} failed permanently after {attempts} attempts: {e}")
        else:
            # Exponential backoff: 5, 10, 20, ... minutes.
            delay = 5 * (2 ** (attempts - 1))
            retry_at = _now() + timedelta(minutes=delay)
            _set_job(job_id, status="pending", attempts=attempts,
                     last_error=str(e), scheduled_for=_to_iso(retry_at))
            log_event("WARNING", "UploadScheduler",
                      f"Job #{job_id} failed (attempt {attempts}/{_max_attempts()}): "
                      f"{e}. Retrying at {retry_at.isoformat()}")
        return get_queue_job(job_id)


class UploadScheduler:
    """Background thread that picks up due upload jobs and dispatches them.

    Usage:
        scheduler = UploadScheduler()
        scheduler.start()
        ...
        scheduler.stop()
    """

    def __init__(self, poll_seconds: float = None):
        self.poll_seconds = poll_seconds if poll_seconds is not None else float(
            os.getenv("UPLOAD_SCHEDULER_POLL_SECONDS", "60"))
        self._stop_event = threading.Event()
        self._thread = None

    def process_due_jobs_once(self) -> list:
        """Processes every due job once. Also used by the thread loop and
        directly by tests / manual runs."""
        results = []
        for job in get_due_jobs():
            results.append(_process_job(job))
        return results

    def _run(self):
        log_event("INFO", "UploadScheduler", "Upload scheduler started")
        while not self._stop_event.is_set():
            try:
                self.process_due_jobs_once()
            except Exception as e:
                log_event("ERROR", "UploadScheduler", f"Scheduler loop error: {e}")
            self._stop_event.wait(self.poll_seconds)
        log_event("INFO", "UploadScheduler", "Upload scheduler stopped")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="UploadScheduler")
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
