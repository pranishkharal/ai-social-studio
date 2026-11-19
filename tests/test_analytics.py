"""Tests for analytics.py (feedback loop) and dashboard_routers.py endpoints."""

import asyncio
from datetime import datetime

import pytest

import analytics
import database
import dashboard_routers
from analytics import analytics_summary, monthly_report, record_performance
from dashboard_routers import (
    PerformanceCreate, QueueEnqueue, SubmissionCreate, SubmissionViewsUpdate,
    analytics_monthly_endpoint, analytics_summary_endpoint,
    cancel_queue_endpoint, earnings_summary_endpoint, enqueue_endpoint,
    list_queue_endpoint, list_submissions_endpoint,
    record_performance_endpoint, record_submission_endpoint,
    update_submission_views_endpoint,
)
from fastapi import HTTPException


def run(coro):
    return asyncio.run(coro)


def _seed_scored_clip(fresh_db, ai_score, category="Funny Moments",
                      streamer="xqc"):
    return fresh_db["seed_clip"](streamer=streamer, ai_score=ai_score,
                                 category=category)


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def test_record_performance_rejects_bad_platform(fresh_db):
    clip_id = _seed_scored_clip(fresh_db, 85)
    with pytest.raises(ValueError, match="Unknown platform"):
        record_performance(clip_id, "twitch", 100)


def test_analytics_summary_buckets_and_recommendation(fresh_db):
    # 3 clips in 90+ averaging 100k, 3 in 80-89 averaging 10k, 3 in 70-79 avg 1k.
    for score, views in [(95, 100_000), (92, 100_000), (91, 100_000),
                         (85, 10_000), (82, 10_000), (80, 10_000),
                         (75, 1_000), (72, 1_000), (70, 1_000)]:
        clip_id = _seed_scored_clip(fresh_db, score)
        record_performance(clip_id, "tiktok", views)

    summary = analytics_summary()
    assert summary["tracked_clips"] == 9
    assert summary["total_views"] == 333_000
    buckets = summary["by_score_bucket"]
    assert buckets["90+"]["avg_views"] == 100_000
    assert buckets["80-89"]["avg_views"] == 10_000
    assert buckets["70-79"]["avg_views"] == 1_000
    rec = summary["threshold_recommendation"]
    assert rec["best_bucket"] == "90+"
    assert rec["suggested_auto_clip_min_score"] == 90
    assert rec["confidence"] == "high"


def test_analytics_summary_low_confidence_with_small_sample(fresh_db):
    clip_id = _seed_scored_clip(fresh_db, 65)
    record_performance(clip_id, "youtube", 50_000)
    rec = analytics_summary()["threshold_recommendation"]
    assert rec["best_bucket"] == "60-69"
    assert rec["confidence"] == "low"
    assert rec["suggested_auto_clip_min_score"] == 60


def test_analytics_summary_empty(fresh_db):
    summary = analytics_summary()
    assert summary["tracked_clips"] == 0
    assert summary["by_score_bucket"] == {}
    assert summary["threshold_recommendation"]["confidence"] == "none"
    assert summary["threshold_recommendation"]["suggested_auto_clip_min_score"] == 80


def test_analytics_summary_uses_latest_snapshot(fresh_db):
    clip_id = _seed_scored_clip(fresh_db, 88)
    record_performance(clip_id, "tiktok", 1_000)
    record_performance(clip_id, "tiktok", 25_000)  # newer snapshot wins
    summary = analytics_summary()
    assert summary["total_views"] == 25_000
    assert summary["tracked_posts"] == 1


def test_analytics_summary_by_category_and_streamer(fresh_db):
    c1 = _seed_scored_clip(fresh_db, 85, category="Funny Moments", streamer="xqc")
    c2 = _seed_scored_clip(fresh_db, 85, category="Big Wins", streamer="adinross")
    record_performance(c1, "tiktok", 10_000)
    record_performance(c2, "tiktok", 40_000)
    summary = analytics_summary()
    assert summary["by_category"]["Big Wins"]["avg_views"] == 40_000
    assert summary["by_streamer"]["adinross"]["total_views"] == 40_000
    assert summary["by_streamer"]["xqc"]["total_views"] == 10_000


def test_monthly_report_current_month(fresh_db):
    now = datetime.now()
    c1 = _seed_scored_clip(fresh_db, 90, streamer="xqc")
    c2 = _seed_scored_clip(fresh_db, 80, streamer="adinross")
    record_performance(c1, "tiktok", 30_000, likes=500, comments=40)
    record_performance(c2, "youtube", 5_000, likes=100, comments=5)

    report = monthly_report(now.year, now.month)
    assert report["total_views"] == 35_000
    assert report["total_likes"] == 600
    assert report["total_comments"] == 45
    assert report["tracked_clips"] == 2
    assert report["top_clips"][0]["clip_id"] == c1  # highest views first
    assert report["top_clips"][0]["views"] == 30_000
    assert report["by_platform"]["tiktok"]["total_views"] == 30_000


def test_monthly_report_empty_month(fresh_db):
    report = monthly_report(2030, 1)
    assert report["total_views"] == 0
    assert report["top_clips"] == []


# ---------------------------------------------------------------------------
# Dashboard router endpoints (fastapi stubbed; handlers called directly)
# ---------------------------------------------------------------------------

def test_router_registers_expected_routes():
    paths = {(m, p) for m, p, _ in dashboard_routers.router.routes}
    assert ("GET", "/api/earnings/summary") in paths
    assert ("GET", "/api/earnings/submissions") in paths
    assert ("POST", "/api/earnings/submissions") in paths
    assert ("PATCH", "/api/earnings/submissions/{submission_id}") in paths
    assert ("GET", "/api/analytics/summary") in paths
    assert ("GET", "/api/analytics/monthly") in paths
    assert ("POST", "/api/analytics/performance") in paths
    assert ("GET", "/api/queue") in paths
    assert ("POST", "/api/queue") in paths
    assert ("POST", "/api/queue/{job_id}/cancel") in paths


def test_router_earnings_flow(fresh_db):
    clip_id = _seed_scored_clip(fresh_db, 88)
    sub = run(record_submission_endpoint(SubmissionCreate(
        clip_id=clip_id, platform="tiktok",
        video_url="https://tiktok.com/@x/1")))
    assert sub["cpm"] == 15

    updated = run(update_submission_views_endpoint(
        sub["id"], SubmissionViewsUpdate(views=100_000, likes=2_000)))
    assert updated["estimated_earnings"] == 15.0

    summary = run(earnings_summary_endpoint())
    assert summary["total_earnings"] == 15.0
    subs = run(list_submissions_endpoint())
    assert len(subs) == 1


def test_router_earnings_bad_platform_is_400(fresh_db):
    with pytest.raises(HTTPException) as exc:
        run(record_submission_endpoint(SubmissionCreate(
            clip_id=None, platform="twitch", video_url="https://x")))
    assert exc.value.status_code == 400


def test_router_update_missing_submission_is_404(fresh_db):
    with pytest.raises(HTTPException) as exc:
        run(update_submission_views_endpoint(
            999, SubmissionViewsUpdate(views=10)))
    assert exc.value.status_code == 404


def test_router_analytics_flow(fresh_db):
    clip_id = _seed_scored_clip(fresh_db, 92)
    perf = run(record_performance_endpoint(PerformanceCreate(
        clip_id=clip_id, platform="youtube", views=77_000, likes=300)))
    assert perf["views"] == 77_000

    summary = run(analytics_summary_endpoint())
    assert summary["total_views"] == 77_000

    now = datetime.now()
    report = run(analytics_monthly_endpoint(year=now.year, month=now.month))
    assert report["total_views"] == 77_000


def test_router_monthly_bad_month_is_400(fresh_db):
    with pytest.raises(HTTPException) as exc:
        run(analytics_monthly_endpoint(year=2026, month=13))
    assert exc.value.status_code == 400


def test_router_queue_flow(fresh_db, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    clip_id = fresh_db["seed_clip"](edited_path=str(video))

    job = run(enqueue_endpoint(QueueEnqueue(clip_id=clip_id, platform="youtube")))
    assert job["status"] == "pending"

    jobs = run(list_queue_endpoint())
    assert len(jobs) == 1

    cancelled = run(cancel_queue_endpoint(job["id"]))
    assert cancelled["status"] == "cancelled"


def test_router_queue_missing_clip_is_400(fresh_db):
    with pytest.raises(HTTPException) as exc:
        run(enqueue_endpoint(QueueEnqueue(clip_id=999, platform="youtube")))
    assert exc.value.status_code == 400


def test_router_cancel_missing_job_is_404(fresh_db):
    with pytest.raises(HTTPException) as exc:
        run(cancel_queue_endpoint(999))
    assert exc.value.status_code == 404


def test_router_cancel_done_job_is_400(fresh_db, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    clip_id = fresh_db["seed_clip"](edited_path=str(video))
    job = run(enqueue_endpoint(QueueEnqueue(clip_id=clip_id, platform="youtube")))
    # Simulate a finished job, then try to cancel it.
    conn = database.get_connection()
    conn.execute("UPDATE upload_queue SET status = 'done' WHERE id = ?",
                 (job["id"],))
    conn.commit()
    conn.close()
    with pytest.raises(HTTPException) as exc:
        run(cancel_queue_endpoint(job["id"]))
    assert exc.value.status_code == 400
