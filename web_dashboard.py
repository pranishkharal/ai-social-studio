import os
import psutil
import threading
from pathlib import Path
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import BASE_DIR, EDITED_DIR, STREAMERS_TO_MONITOR
from database import (
    get_all_streamers, get_all_clips, get_recent_logs, 
    get_connection, log_event, update_streamer_status,
    get_notifications, mark_notification_read, mark_all_notifications_read,
    update_clip_approval, toggle_streamer_monitoring, delete_clip
)
from ai_story_engine import PRD_CATEGORIES
from kick_monitor import KickMonitor
from pipeline import ClipPipeline
from screen_controller import KickScreenController

app = FastAPI(title="Kick AI Clip Engine")

# Mount templates and static
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
static_dir = BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Shared singletons
monitor = KickMonitor()
pipeline = ClipPipeline(monitor)
screen_agent = KickScreenController()

# Start background monitor thread
monitor_thread = monitor.start_background()

# Request Models
class StreamerReq(BaseModel):
    username: str

from kick_program import CREATOR_RATES, get_creator_info, format_bot_submission_command

class DiscordCmdReq(BaseModel):
    streamer: str
    urls: list

@app.get("/api/creators/rates")
async def get_rates():
    return CREATOR_RATES

@app.post("/api/discord-command")
async def generate_discord_command(req: DiscordCmdReq):
    cmd_str = format_bot_submission_command(req.streamer, req.urls)
    info = get_creator_info(req.streamer)
    return {
        "command": cmd_str,
        "rate_per_100k": info["cpm"],
        "rate_per_1m": info["cpm"] * 10,
        "streamer": req.streamer
    }
class ClipTriggerReq(BaseModel):
    streamer: str
    duration: int = 25
    mirror: bool = True
    hdr_sharpness: bool = True
    layout: str = "speaker_tracking"
    auto_captions: bool = False
    caption_position: str = "top"
    custom_caption: str = ""

class RerenderReq(BaseModel):
    caption_position: str = "top"
    custom_caption: str = ""
    layout: str = "speaker_tracking"
    mirror: bool = True
    hdr_sharpness: bool = True

class ScrubReq(BaseModel):
    streamer: str
    offset: int = -30

class UploadReq(BaseModel):
    caption: str = ""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    """Live resource telemetry demonstrating the ultra-light footprint."""
    process = psutil.Process()
    ram_mb = round(process.memory_info().rss / (1024 * 1024), 1)
    cpu_percent = round(psutil.cpu_percent(interval=None), 1)
    return {
        "cpu_percent": cpu_percent,
        "ram_mb": ram_mb,
        "active_recorders": len(monitor.active_recorders)
    }

@app.get("/api/streamers")
async def list_streamers():
    return get_all_streamers()

@app.post("/api/streamers")
async def add_streamer(req: StreamerReq):
    uname = req.username.strip().lower()
    if uname not in STREAMERS_TO_MONITOR:
        STREAMERS_TO_MONITOR.append(uname)
    info = monitor.fetch_channel_info(uname)
    log_event("INFO", "WebAPI", f"Added streamer to watchlist: {uname}")
    return {"status": "added", "streamer": info}

@app.delete("/api/streamers/{username}")
async def remove_streamer(username: str):
    uname = username.strip().lower()
    if uname in STREAMERS_TO_MONITOR:
        STREAMERS_TO_MONITOR.remove(uname)
    monitor.stop_recorder(uname)
    conn = get_connection()
    conn.execute("DELETE FROM streamers WHERE username = ?", (uname,))
    conn.commit()
    conn.close()
    log_event("INFO", "WebAPI", f"Removed streamer: {uname}")
    return {"status": "removed"}

@app.get("/api/clips")
async def list_clips():
    return get_all_clips()

@app.get("/api/clips/{clip_id}")
async def get_clip(clip_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Clip not found")
    return dict(row)

@app.post("/api/clips/trigger")
async def trigger_clip(req: ClipTriggerReq, background_tasks: BackgroundTasks):
    """Triggers instant clip capture, hype analysis, storytelling, and vertical rendering with Prestige rules."""
    def run_pipeline():
        pipeline.process_moment(
            streamer=req.streamer, 
            duration=req.duration,
            mirror=req.mirror,
            hdr_sharpness=req.hdr_sharpness,
            layout=req.layout,
            auto_captions=req.auto_captions,
            caption_position=req.caption_position,
            custom_caption=req.custom_caption
        )

    background_tasks.add_task(run_pipeline)
    return {"status": "started", "message": f"Processing {req.duration}s clip for {req.streamer} (Position='{req.caption_position}')."}

@app.post("/api/clips/{clip_id}/rerender")
async def rerender_clip_endpoint(clip_id: int, req: RerenderReq, background_tasks: BackgroundTasks):
    def run_rerender():
        pipeline.rerender_clip(
            clip_id=clip_id,
            caption_position=req.caption_position,
            custom_caption=req.custom_caption,
            layout=req.layout,
            mirror=req.mirror,
            hdr_sharpness=req.hdr_sharpness
        )
    background_tasks.add_task(run_rerender)
    return {"status": "started", "message": f"Re-rendering clip #{clip_id} with caption '{req.caption_position}'"}

@app.post("/api/agent/open_browser")
async def open_browser_viewer(req: StreamerReq, background_tasks: BackgroundTasks):
    def run_browser():
        screen_agent.open_stream_in_browser(req.username, headless=False)
    background_tasks.add_task(run_browser)
    return {"status": "launching_browser"}

@app.post("/api/agent/scrub")
async def scrub_stream(req: ScrubReq):
    success = screen_agent.scrub_player(req.offset)
    return {"status": "success" if success else "failed"}

@app.api_route("/video/{clip_id}", methods=["GET", "HEAD"])
async def stream_video(clip_id: int):
    conn = get_connection()
    row = conn.execute("SELECT edited_path, raw_path FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Video not found")
    path = row["edited_path"] or row["raw_path"]
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(path, media_type="video/mp4")

@app.post("/api/clips/{clip_id}/upload")
async def upload_clip(clip_id: int, req: UploadReq, background_tasks: BackgroundTasks):
    conn = get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not row or not row["edited_path"]:
        raise HTTPException(status_code=400, detail="Clip not rendered yet")

    def run_upload():
        caption = req.caption or row["caption"] or row["title"]
        screen_agent.upload_to_tiktok(row["edited_path"], caption, row["tags"] or "", headless=False)

    background_tasks.add_task(run_upload)
    return {"status": "upload_dispatched", "message": "TikTok uploader launched! Check browser window to confirm post."}

@app.get("/api/logs")
async def list_logs():
    return get_recent_logs(40)

@app.get("/api/categories")
async def list_categories():
    return PRD_CATEGORIES

@app.get("/api/notifications")
async def list_notifications(unread_only: bool = False):
    notifs = get_notifications(unread_only=unread_only, limit=40)
    unread_count = sum(1 for n in notifs if n.get("is_read") == 0)
    return {"notifications": notifs, "unread_count": unread_count}

@app.post("/api/notifications/read_all")
async def read_all_notifs():
    mark_all_notifications_read()
    return {"status": "ok"}

@app.post("/api/notifications/{notif_id}/read")
async def read_notif(notif_id: int):
    mark_notification_read(notif_id)
    return {"status": "ok"}

@app.post("/api/clips/{clip_id}/approve")
async def approve_clip(clip_id: int):
    update_clip_approval(clip_id, "approved")
    return {"status": "approved", "clip_id": clip_id}

@app.post("/api/clips/{clip_id}/reject")
async def reject_clip(clip_id: int):
    update_clip_approval(clip_id, "rejected")
    return {"status": "rejected", "clip_id": clip_id}

@app.post("/api/clips/{clip_id}/restore")
async def restore_clip(clip_id: int):
    update_clip_approval(clip_id, "pending")
    return {"status": "restored", "clip_id": clip_id}

@app.delete("/api/clips/{clip_id}")
async def delete_clip_endpoint(clip_id: int):
    delete_clip(clip_id)
    log_event("INFO", "WebAPI", f"Deleted clip #{clip_id} and purged files from disk")
    return {"status": "deleted", "clip_id": clip_id}

class MonitoringReq(BaseModel):
    enabled: bool

@app.post("/api/streamers/{username}/toggle")
async def toggle_streamer(username: str, req: MonitoringReq):
    toggle_streamer_monitoring(username, req.enabled)
    return {"status": "updated", "username": username, "enabled": req.enabled}
