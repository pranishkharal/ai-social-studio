import os
import psutil
import threading
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
import requests
from pydantic import BaseModel

from config import BASE_DIR, CLIPS_DIR, EDITED_DIR, STREAMERS_TO_MONITOR, AUTO_CLIP_ENABLED
from database import (
    get_all_streamers, get_all_clips, get_recent_logs, 
    get_connection, log_event, update_streamer_status,
    get_notifications, mark_notification_read, mark_all_notifications_read,
    update_clip_approval, toggle_streamer_monitoring, delete_clip, add_notification
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

# --- Live-clipping upgrades: routers, auth middleware, services ---
# Each router mounts in its own try/except so one broken module never
# takes the whole dashboard down.
try:
    import kick_webhooks
    kick_webhooks.set_monitor(monitor)
    app.include_router(kick_webhooks.router)
    log_event("INFO", "WebAPI", "Kick webhooks router mounted at /api/webhooks/kick")
except Exception as e:
    log_event("WARNING", "WebAPI", f"Kick webhooks router not mounted: {e}")

try:
    import dashboard_routers
    app.include_router(dashboard_routers.router)
    log_event("INFO", "WebAPI", "Earnings/analytics/queue routers mounted")
except Exception as e:
    log_event("WARNING", "WebAPI", f"Earnings/analytics/queue routers not mounted: {e}")

try:
    import gta6_routers
    app.include_router(gta6_routers.router)
    log_event("INFO", "WebAPI", "gta6-clipper routers mounted (voice-over, live hunt, clip rules, YouTube stats, self-update, schedule dry-run)")
except Exception as e:
    log_event("WARNING", "WebAPI", f"gta6-clipper routers not mounted: {e}")

try:
    import auth
    app.include_router(auth.router)
    log_event("INFO", "WebAPI", "Dashboard login router mounted at /login")
except Exception as e:
    log_event("WARNING", "WebAPI", f"Dashboard login router not mounted: {e}")
    auth = None


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Dashboard login gate (Tier 4).

    When no password is configured, auth.is_auth_enabled() is False and
    every request passes through. When a password IS configured, every
    route except the login/logout pages, the Kick webhook endpoint
    (Kick cannot log in) and static assets requires a valid session
    cookie: API callers get 401 JSON, browsers get redirected to /login.
    """
    if auth is not None and auth.is_auth_enabled():
        path = request.url.path
        public = path in ("/login", "/logout") or path.startswith("/static") or path == "/api/webhooks/kick"
        if not public:
            token = request.cookies.get(auth.SESSION_COOKIE_NAME)
            if not auth.verify_session_token(token):
                if path.startswith("/api/"):
                    return JSONResponse({"detail": "Login required"}, status_code=401)
                return RedirectResponse(url="/login", status_code=303)
    return await call_next(request)


# Upload scheduler (Tier 2): drains the upload queue in the background.
upload_scheduler = None
try:
    from upload_scheduler import UploadScheduler
    upload_scheduler = UploadScheduler()
    upload_scheduler.start()
    log_event("INFO", "WebAPI", "Upload scheduler started")
except Exception as e:
    log_event("WARNING", "WebAPI", f"Upload scheduler not started: {e}")

# Auto-clip watcher (Tier 1): only when explicitly enabled — it clips
# automatically whenever a monitored stream gets hypey.
auto_clip_watcher = None
if AUTO_CLIP_ENABLED:
    try:
        from auto_clip_watcher import AutoClipWatcher
        auto_clip_watcher = AutoClipWatcher(monitor, pipeline)
        auto_clip_watcher.start()
        log_event("INFO", "WebAPI", "Auto-clip watcher started (AUTO_CLIP_ENABLED)")
    except Exception as e:
        log_event("WARNING", "WebAPI", f"Auto-clip watcher not started: {e}")

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
    duration: Optional[int] = None  # None = the streamer's profile decides
    mirror: bool = True
    hdr_sharpness: bool = True
    layout: str = "speaker_tracking"
    tracking_target: str = "face"  # "face" (default) or "body" — what speaker_tracking locks onto
    auto_captions: bool = False
    caption_position: str = "top"
    custom_caption: str = ""
    # TikTok upgrades (v1.2.0): None = the matching .env setting decides.
    short_clip: Optional[bool] = None  # tight clip around the peak moment
    banner_timing: Optional[str] = None  # "4s" (default) or "whole_clip"
    bgm: Optional[bool] = None  # soft background music under the audio
    game: Optional[str] = None  # e.g. "GTA 6" -> adds a #gta6 hashtag
    # Quote line (v1.4.1): one static spoken line under the headline.
    # Off by default — the dashboard checkbox opts in per clip.
    quote_line_enabled: bool = False

class RerenderReq(BaseModel):
    caption_position: str = "top"
    custom_caption: str = ""
    layout: str = "speaker_tracking"
    tracking_target: str = "face"  # "face" (default) or "body"
    mirror: bool = True
    hdr_sharpness: bool = True
    banner_timing: Optional[str] = None  # "4s" (default) or "whole_clip"
    quote_line: Optional[str] = None  # None = reuse the stored quote line
    # Quote line (v1.4.1): off by default — the modal checkbox opts in.
    quote_line_enabled: bool = False

class VodClipReq(BaseModel):
    source: str  # Kick/Twitch/YouTube URL or a local video file path
    streamer: Optional[str] = None  # default: slug from the URL, else "vod"
    duration: Optional[int] = 25
    layout: Optional[str] = None

class ScrubReq(BaseModel):
    streamer: str
    offset: int = -30

class UploadReq(BaseModel):
    caption: str = ""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # New Starlette signature: TemplateResponse(request, name, context).
    # The legacy TemplateResponse(name, {"request": ...}) form was removed
    # in Starlette 1.x and raises TypeError there.
    return templates.TemplateResponse(request, "index.html")

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
            tracking_target=req.tracking_target,
            auto_captions=req.auto_captions,
            caption_position=req.caption_position,
            custom_caption=req.custom_caption,
            short_clip=req.short_clip,
            banner_timing=req.banner_timing,
            quote_line_enabled=req.quote_line_enabled,
            bgm=req.bgm,
            game=req.game
        )

    background_tasks.add_task(run_pipeline)
    duration_label = f"{req.duration}s" if req.duration else "profile-length"
    return {"status": "started", "message": f"Processing {duration_label} clip for {req.streamer} (Position='{req.caption_position}')."}

@app.post("/api/vod/clip")
async def vod_clip(req: VodClipReq, background_tasks: BackgroundTasks):
    """Clip the best moment from a past video (TikTok upgrade, v1.2.0).

    Source can be a Kick VOD, Twitch VOD/clip, YouTube link (podcasts
    included) or a local video file. Downloads with yt-dlp when the
    source is a URL, finds the loudest moment, and runs it through the
    normal pipeline. Runs in the background; the finished clip appears
    in the clips list.
    """
    def run_vod():
        try:
            from vod_clipper import clip_vod
            clip_vod(source=req.source, streamer=req.streamer,
                     duration=req.duration or 25, layout=req.layout)
        except Exception as e:
            log_event("ERROR", "WebAPI", f"VOD clip failed for {req.source}: {e}")

    background_tasks.add_task(run_vod)
    return {"status": "started", "message": f"Clipping best moment from {req.source}"}

@app.post("/api/clips/{clip_id}/rerender")
async def rerender_clip_endpoint(clip_id: int, req: RerenderReq, background_tasks: BackgroundTasks):
    def run_rerender():
        pipeline.rerender_clip(
            clip_id=clip_id,
            caption_position=req.caption_position,
            custom_caption=req.custom_caption,
            layout=req.layout,
            tracking_target=req.tracking_target,
            mirror=req.mirror,
            hdr_sharpness=req.hdr_sharpness,
            banner_timing=req.banner_timing,
            quote_line=req.quote_line,
            quote_line_enabled=req.quote_line_enabled
        )
    background_tasks.add_task(run_rerender)
    return {"status": "started", "message": f"Re-rendering clip #{clip_id} with caption '{req.caption_position}'"}

@app.post("/api/clips/{clip_id}/retry")
async def retry_clip_endpoint(clip_id: int, background_tasks: BackgroundTasks):
    """Resume a failed clip from its last completed checkpoint (Tier 4)."""
    def run_resume():
        pipeline.resume_clip(clip_id)
    background_tasks.add_task(run_resume)
    return {"status": "started", "message": f"Resuming clip #{clip_id} from its last checkpoint"}


class WebhookSubscribeReq(BaseModel):
    broadcaster_user_id: Optional[int] = None


@app.get("/api/webhooks/status")
async def webhooks_status():
    """Kick webhook configuration status + setup checklist (Tier 1)."""
    try:
        import kick_webhooks
        return {
            "configured": kick_webhooks.webhooks_configured(),
            "checklist": kick_webhooks.get_setup_checklist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webhook module unavailable: {e}")


@app.post("/api/webhooks/subscribe")
async def webhooks_subscribe(req: WebhookSubscribeReq):
    """Subscribe to livestream.status.updated events (Tier 1).

    With no broadcaster_user_id, subscribes to the token owner's own
    channel. Subscribing to other broadcasters needs a token authorised
    for them — unverified whether a user token can do that, so the
    endpoint reports Kick's own answer.
    """
    try:
        import kick_webhooks
        token = kick_webhooks.get_access_token()
        result = kick_webhooks.subscribe_livestream_status(token, req.broadcaster_user_id)
        return {"status": "subscribed", "subscription": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# ---------------------------------------------------------------------------
# Kick Community Clips (Browse viewer-published clips & 1-click AI Convert)
# ---------------------------------------------------------------------------

SEEN_COMMUNITY_CLIP_IDS = set()
INITIALIZED_COMMUNITY_STREAMERS = set()

@app.get("/api/kick/community-clips")
async def get_kick_community_clips(streamer: str = "westcol", sort: str = "date", time_range: str = "day"):
    """Fetch viewer/community clips from Kick for a streamer."""
    try:
        clean_streamer = streamer.strip().lower()
        url = f"https://kick.com/api/v2/channels/{clean_streamer}/clips?sort={sort}&time={time_range}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=6)
        if resp.status_code != 200:
            return {"clips": [], "error": f"Kick API returned {resp.status_code}"}
        data = resp.json()
        raw_clips = data.get("clips", [])
        
        conn = get_connection()
        existing_titles = set(row[0] for row in conn.execute("SELECT title FROM clips").fetchall())
        conn.close()

        is_first_load = clean_streamer not in INITIALIZED_COMMUNITY_STREAMERS
        formatted = []
        new_clip_count = 0

        for c in raw_clips:
            clip_id = c.get("id")
            title = c.get("title", "")
            clip_url = c.get("clip_url") or c.get("video_url")
            thumb = c.get("thumbnail_url")
            creator = c.get("creator", {}).get("username", "Viewer")
            views = c.get("views", 0)
            duration = c.get("duration", 30)
            created_at = c.get("created_at", "")
            
            is_new = False
            if clip_id:
                if is_first_load:
                    SEEN_COMMUNITY_CLIP_IDS.add(clip_id)
                elif clip_id not in SEEN_COMMUNITY_CLIP_IDS:
                    SEEN_COMMUNITY_CLIP_IDS.add(clip_id)
                    is_new = True
                    new_clip_count += 1
                    try:
                        add_notification(
                            title=f"🎬 New Clip: {title[:40]}",
                            message=f"Viewer '{creator}' just clipped {clean_streamer} on Kick!",
                            notif_type="kick_community_clip",
                            streamer=clean_streamer
                        )
                    except Exception:
                        pass

            formatted.append({
                "id": clip_id,
                "title": title,
                "clip_url": clip_url,
                "thumbnail_url": thumb,
                "creator": creator,
                "views": views,
                "duration": duration,
                "created_at": created_at,
                "is_new": is_new,
                "already_edited": any(title.lower() in et.lower() for et in existing_titles if et) if title else False
            })

        INITIALIZED_COMMUNITY_STREAMERS.add(clean_streamer)
        return {"clips": formatted, "streamer": clean_streamer, "new_count": new_clip_count}
    except Exception as e:
        log_event("WARNING", "WebAPI", f"Failed fetching Kick community clips: {e}")
        return {"clips": [], "error": str(e)}


class ConvertKickClipReq(BaseModel):
    streamer: str
    clip_url: str
    title: Optional[str] = None
    duration: Optional[int] = 30
    clip_id: Optional[str] = None


@app.post("/api/kick/convert-clip")
async def convert_kick_clip(req: ConvertKickClipReq, background_tasks: BackgroundTasks):
    """Download a viewer-made Kick clip, run Whisper STT + Caption Guide, and render 9:16 vertical TikTok video."""
    def run_conversion():
        try:
            import subprocess, re
            safe_streamer = req.streamer.strip().lower()
            safe_id = re.sub(r'[^a-zA-Z0-9_-]', '', req.clip_id or 'clip')
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = str(CLIPS_DIR / f"{safe_streamer}_kick_{safe_id}_{stamp}_raw.mp4")
            
            log_event("INFO", "KickCommunity", f"Downloading Kick community clip: {req.title or req.clip_url}")
            # Stream-copy directly from Kick m3u8 playlist to local mp4
            target_dur = min(int(req.duration or 30), 60)
            cmd = ["ffmpeg", "-y", "-i", req.clip_url, "-t", str(target_dur), "-c", "copy", raw_path]
            p = subprocess.run(cmd, capture_output=True, timeout=120)
            if p.returncode != 0 or not os.path.exists(raw_path) or os.path.getsize(raw_path) < 1000:
                log_event("ERROR", "KickCommunity", f"Failed to download Kick clip: {p.stderr.decode('utf-8', 'replace')}")
                return
            
            # Process moment through pipeline (Whisper STT + Caption Guide + 9:16 Crop)
            log_event("INFO", "KickCommunity", f"Processing Kick clip #{req.clip_id} ({target_dur}s) through AI Social Studio...")
            pipeline.process_moment(
                streamer=safe_streamer,
                raw_clip_path=raw_path,
                duration=target_dur
            )
            add_notification(
                title=f"🎬 Kick Clip Converted: {req.title or 'Viewer Clip'}",
                message=f"Successfully edited into 9:16 vertical video with Caption Guide hook!",
                notif_type="highlight_ready",
                streamer=safe_streamer
            )
        except Exception as e:
            log_event("ERROR", "KickCommunity", f"Error converting Kick community clip: {e}")

    background_tasks.add_task(run_conversion)
    return {"status": "started", "message": f"Downloading and converting Kick clip '{req.title or ''}' into 9:16 vertical format"}

