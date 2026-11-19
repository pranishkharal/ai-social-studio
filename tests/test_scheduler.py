"""Tests for upload_scheduler.py.

Uploaders are mocked; the queue runs against a throwaway sqlite DB.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

import database
import upload_scheduler
import youtube_uploader
import instagram_uploader
from upload_scheduler import (
    UploadScheduler, cancel_job, enqueue_clip, get_due_jobs, get_queue_job,
    list_queue,
)


def _dt(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _seed_ready_clip(fresh_db, tmp_path, streamer="xqc"):
    video = tmp_path / f"{streamer}_clip.mp4"
    video.write_bytes(b"fake video")
    return fresh_db["seed_clip"](streamer=streamer, edited_path=str(video))


# ---------------------------------------------------------------------------
# Enqueue + spacing rules
# ---------------------------------------------------------------------------

def test_enqueue_defaults_to_now_and_pending(fresh_db, tmp_path):
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    job = enqueue_clip(clip_id, "youtube")
    assert job["status"] == "pending"
    assert job["platform"] == "youtube"
    assert job["attempts"] == 0
    assert _dt(job["scheduled_for"]) <= datetime.now(timezone.utc) + timedelta(seconds=5)


def test_enqueue_rejects_unknown_platform(fresh_db, tmp_path):
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    with pytest.raises(ValueError, match="Unknown platform"):
        enqueue_clip(clip_id, "twitch")


def test_enqueue_rejects_missing_clip(fresh_db):
    with pytest.raises(ValueError, match="not found"):
        enqueue_clip(999, "youtube")


def test_spacing_pushes_second_job_by_min_gap(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_MIN_GAP_MINUTES", "120")
    c1 = _seed_ready_clip(fresh_db, tmp_path)
    c2 = fresh_db["seed_clip"](streamer="adinross",
                               edited_path=str(tmp_path / "adin_clip.mp4"))
    t0 = datetime.now(timezone.utc)
    j1 = enqueue_clip(c1, "youtube", scheduled_for=t0)
    j2 = enqueue_clip(c2, "youtube", scheduled_for=t0)
    gap = (_dt(j2["scheduled_for"]) - _dt(j1["scheduled_for"])).total_seconds() / 60
    assert gap >= 120


def test_spacing_is_per_platform(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_MIN_GAP_MINUTES", "120")
    monkeypatch.setenv("TIKTOK_MIN_GAP_MINUTES", "180")
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    t0 = datetime.now(timezone.utc)
    j1 = enqueue_clip(clip_id, "youtube", scheduled_for=t0)
    j2 = enqueue_clip(clip_id, "tiktok", scheduled_for=t0)
    # Different platform: no push.
    assert _dt(j2["scheduled_for"]) == _dt(j1["scheduled_for"])


def test_spacing_respects_explicit_future_time(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_MIN_GAP_MINUTES", "120")
    c1 = _seed_ready_clip(fresh_db, tmp_path)
    c2 = fresh_db["seed_clip"](streamer="adinross",
                               edited_path=str(tmp_path / "a2.mp4"))
    t0 = datetime.now(timezone.utc)
    enqueue_clip(c1, "youtube", scheduled_for=t0)
    far_future = t0 + timedelta(hours=5)
    j2 = enqueue_clip(c2, "youtube", scheduled_for=far_future)
    assert _dt(j2["scheduled_for"]) == far_future


# ---------------------------------------------------------------------------
# Queue listing / due jobs / cancel
# ---------------------------------------------------------------------------

def test_get_due_jobs_only_returns_due_pending(fresh_db, tmp_path):
    c1 = _seed_ready_clip(fresh_db, tmp_path)
    c2 = fresh_db["seed_clip"](streamer="adinross",
                               edited_path=str(tmp_path / "a3.mp4"))
    now = datetime.now(timezone.utc)
    enqueue_clip(c1, "youtube", scheduled_for=now - timedelta(minutes=1))
    enqueue_clip(c2, "youtube", scheduled_for=now + timedelta(hours=2))
    due = get_due_jobs()
    assert [d["clip_id"] for d in due] == [c1]


def test_cancel_job(fresh_db, tmp_path):
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    job = enqueue_clip(clip_id, "tiktok")
    cancelled = cancel_job(job["id"])
    assert cancelled["status"] == "cancelled"
    assert get_due_jobs() == []


def test_cancel_missing_job_returns_none(fresh_db):
    assert cancel_job(999) is None


def test_list_queue_filters(fresh_db, tmp_path):
    c1 = _seed_ready_clip(fresh_db, tmp_path)
    j1 = enqueue_clip(c1, "youtube")
    enqueue_clip(c1, "tiktok")
    cancel_job(j1["id"])
    assert len(list_queue(status="cancelled")) == 1
    assert len(list_queue(platform="tiktok")) == 1
    assert len(list_queue()) == 2


# ---------------------------------------------------------------------------
# Dispatch: YouTube success, retry/backoff, permanent failure
# ---------------------------------------------------------------------------

def test_process_due_youtube_job_success(fresh_db, tmp_path, monkeypatch):
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    enqueue_clip(clip_id, "youtube",
                 scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1))
    monkeypatch.setattr(youtube_uploader, "upload_clip",
                        lambda cid: "https://www.youtube.com/watch?v=ABC123")

    results = UploadScheduler().process_due_jobs_once()
    assert len(results) == 1
    assert results[0]["status"] == "done"
    assert results[0]["posted_url"] == "https://www.youtube.com/watch?v=ABC123"

    # Success also records an earnings submission (best-effort integration).
    conn = database.get_connection()
    subs = conn.execute("SELECT * FROM submissions").fetchall()
    conn.close()
    assert len(subs) == 1
    assert subs[0]["platform"] == "youtube"
    assert subs[0]["cpm"] == 15  # xqc rate


def test_process_due_job_retries_with_backoff_then_fails(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_ATTEMPTS", "2")
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    enqueue_clip(clip_id, "youtube",
                 scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1))

    def boom(cid):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(youtube_uploader, "upload_clip", boom)
    scheduler = UploadScheduler()

    first = scheduler.process_due_jobs_once()[0]
    assert first["status"] == "pending"  # scheduled for retry
    assert first["attempts"] == 1
    assert "quota exceeded" in first["last_error"]
    # Retry is in the future (5 min backoff) — not due yet.
    assert get_due_jobs() == []

    # Force it due again and run: second failure exhausts attempts.
    conn = database.get_connection()
    conn.execute("UPDATE upload_queue SET scheduled_for = ? WHERE id = ?",
                 (datetime.now(timezone.utc).isoformat(), first["id"]))
    conn.commit()
    conn.close()
    second = scheduler.process_due_jobs_once()[0]
    assert second["status"] == "failed"
    assert second["attempts"] == 2


def test_daily_cap_reschedules_to_tomorrow(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_MAX_PER_DAY", "0")
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    enqueue_clip(clip_id, "youtube",
                 scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1))
    upload_mock = MagicMock(return_value="https://www.youtube.com/watch?v=X")
    monkeypatch.setattr(youtube_uploader, "upload_clip", upload_mock)

    result = UploadScheduler().process_due_jobs_once()[0]
    assert result["status"] == "pending"
    assert "Daily cap" in (result["last_error"] or "")
    assert _dt(result["scheduled_for"]).date() > datetime.now(timezone.utc).date()
    upload_mock.assert_not_called()


def test_instagram_job_without_video_url_fails_clearly(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_MAX_ATTEMPTS", "1")
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    enqueue_clip(clip_id, "instagram",
                 scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1))
    result = UploadScheduler().process_due_jobs_once()[0]
    assert result["status"] == "failed"
    assert "public video_url" in result["last_error"]


def test_instagram_job_with_video_url_dispatches(fresh_db, tmp_path, monkeypatch):
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    enqueue_clip(clip_id, "instagram",
                 scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1),
                 video_url="https://example.com/clip.mp4")
    monkeypatch.setattr(instagram_uploader, "upload_clip_reel",
                        lambda cid, url: "https://www.instagram.com/reel/XYZ/")
    result = UploadScheduler().process_due_jobs_once()[0]
    assert result["status"] == "done"
    assert result["posted_url"] == "https://www.instagram.com/reel/XYZ/"


def test_tiktok_dispatch_uses_screen_controller(fresh_db, tmp_path, monkeypatch):
    import screen_controller

    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    enqueue_clip(clip_id, "tiktok",
                 scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1))

    calls = {}

    class FakeController:
        def upload_to_tiktok(self, video_path, caption, tags, headless=False):
            calls["video_path"] = video_path
            calls["caption"] = caption
            return True

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(screen_controller, "KickScreenController", FakeController)
    result = UploadScheduler().process_due_jobs_once()[0]
    assert result["status"] == "done"
    assert result["posted_url"] is None  # manual Post click in browser
    assert calls["video_path"].endswith("_clip.mp4")
    assert calls["closed"] is True


def test_tiktok_dispatch_failure_retries(fresh_db, tmp_path, monkeypatch):
    import screen_controller

    monkeypatch.setenv("UPLOAD_MAX_ATTEMPTS", "1")
    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    enqueue_clip(clip_id, "tiktok",
                 scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1))

    class FailingController:
        def upload_to_tiktok(self, *a, **k):
            return False

        def close(self):
            pass

    monkeypatch.setattr(screen_controller, "KickScreenController", FailingController)
    result = UploadScheduler().process_due_jobs_once()[0]
    assert result["status"] == "failed"
    assert "TikTok staging failed" in result["last_error"]


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------

def test_scheduler_thread_processes_due_jobs(fresh_db, tmp_path, monkeypatch):
    import time

    clip_id = _seed_ready_clip(fresh_db, tmp_path)
    job = enqueue_clip(clip_id, "youtube",
                       scheduled_for=datetime.now(timezone.utc) - timedelta(minutes=1))
    monkeypatch.setattr(youtube_uploader, "upload_clip",
                        lambda cid: "https://www.youtube.com/watch?v=THREAD")

    scheduler = UploadScheduler(poll_seconds=0.05)
    scheduler.start()
    assert scheduler.is_running
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if get_queue_job(job["id"])["status"] == "done":
                break
            time.sleep(0.05)
        assert get_queue_job(job["id"])["status"] == "done"
    finally:
        scheduler.stop()
    assert not scheduler.is_running
