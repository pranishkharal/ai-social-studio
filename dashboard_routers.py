"""
Dashboard API router for Tier 2 features (earnings, analytics, upload queue).

The coordinator mounts this router into web_dashboard.py:

    from dashboard_routers import router
    app.include_router(router)

Endpoints:
    GET   /api/earnings/summary?streamer=&month=
    GET   /api/earnings/submissions?streamer=&platform=
    POST  /api/earnings/submissions          {clip_id, platform, video_url, streamer?, streamers?}
    PATCH /api/earnings/submissions/{id}     {views?, likes?}
    GET   /api/analytics/summary
    GET   /api/analytics/monthly?year=&month=
    POST  /api/analytics/performance         {clip_id, platform, views, likes?, comments?}
    GET   /api/queue?status=&platform=
    POST  /api/queue                         {clip_id, platform, scheduled_for?, video_url?}
    POST  /api/queue/{id}/cancel
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import analytics
import earnings
import upload_scheduler

router = APIRouter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SubmissionCreate(BaseModel):
    clip_id: int
    platform: str
    video_url: str
    streamer: str = None
    streamers: list = None


class SubmissionViewsUpdate(BaseModel):
    views: int = None
    likes: int = None


class PerformanceCreate(BaseModel):
    clip_id: int
    platform: str
    views: int
    likes: int = 0
    comments: int = 0


class QueueEnqueue(BaseModel):
    clip_id: int
    platform: str
    scheduled_for: str = None
    video_url: str = None


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

@router.get("/api/earnings/summary")
async def earnings_summary_endpoint(streamer: str = None, month: str = None):
    return earnings.earnings_summary(streamer=streamer, month=month)


@router.get("/api/earnings/submissions")
async def list_submissions_endpoint(streamer: str = None, platform: str = None,
                                    limit: int = 50):
    return earnings.list_submissions(streamer=streamer, platform=platform, limit=limit)


@router.post("/api/earnings/submissions")
async def record_submission_endpoint(req: SubmissionCreate):
    try:
        return earnings.record_submission(
            clip_id=req.clip_id, platform=req.platform, video_url=req.video_url,
            streamer=req.streamer, streamers=req.streamers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/api/earnings/submissions/{submission_id}")
async def update_submission_views_endpoint(submission_id: int,
                                           req: SubmissionViewsUpdate):
    try:
        return earnings.update_views(submission_id, views=req.views, likes=req.likes)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@router.get("/api/analytics/summary")
async def analytics_summary_endpoint():
    return analytics.analytics_summary()


@router.get("/api/analytics/monthly")
async def analytics_monthly_endpoint(year: int = None, month: int = None):
    now = datetime.now()
    year = year if year is not None else now.year
    month = month if month is not None else now.month
    if not 1 <= month <= 12:
        raise HTTPException(status_code=400, detail="month must be 1-12")
    return analytics.monthly_report(year, month)


@router.post("/api/analytics/performance")
async def record_performance_endpoint(req: PerformanceCreate):
    try:
        return analytics.record_performance(
            clip_id=req.clip_id, platform=req.platform, views=req.views,
            likes=req.likes, comments=req.comments)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Upload queue
# ---------------------------------------------------------------------------

@router.get("/api/queue")
async def list_queue_endpoint(status: str = None, platform: str = None,
                              limit: int = 50):
    return upload_scheduler.list_queue(status=status, platform=platform, limit=limit)


@router.post("/api/queue")
async def enqueue_endpoint(req: QueueEnqueue):
    try:
        return upload_scheduler.enqueue_clip(
            clip_id=req.clip_id, platform=req.platform,
            scheduled_for=req.scheduled_for, video_url=req.video_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/queue/{job_id}/cancel")
async def cancel_queue_endpoint(job_id: int):
    job = upload_scheduler.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Queue job #{job_id} not found")
    if job["status"] != "cancelled":
        raise HTTPException(
            status_code=400,
            detail=f"Queue job #{job_id} is {job['status']} and cannot be cancelled")
    return job
