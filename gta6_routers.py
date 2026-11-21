"""
Dashboard API router for the gta6-clipper upgrades (voice-over, live
hunt, clip permission rules, YouTube stats, self-update, schedule
dry-run).

The coordinator mounts this router into web_dashboard.py:

    import gta6_routers
    app.include_router(gta6_routers.router)

Endpoints:
    POST  /api/voiceover                 {clip_id, voiceover_at, audio_base64, filename?}
    GET   /api/clip-rules/{platform}/{streamer}
    GET   /api/clip-rules
    GET   /api/hunt?streamers=kick:adinross,twitch:xqc
    GET   /api/hunt/past?platform=kick&handle=adinross
    POST  /api/hunt/start                {platform, handle}
    POST  /api/hunt/stop                 {platform, handle}
    GET   /api/yt/status
    POST  /api/yt/key                    {api_key, channel?}
    GET   /api/yt/channel?channel=@handle
    GET   /api/yt/diagnose?video_url=&hook=
    GET   /api/update/check
    POST  /api/update/apply              {dry?}
    POST  /api/schedule/dry-run          {items: [{clip_id, day, time, platform}]}
"""

import base64
import os
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import clip_rules
import live_hunt
import self_update
import voiceover
import ytstats
from config import BASE_DIR, EDITED_DIR, STREAMERS_TO_MONITOR
from database import get_connection, log_event, update_clip_status

router = APIRouter()

# Voice-over recordings Sara uploads (phone or mic): these formats
# only, so a random file cannot be smuggled in as audio.
VOICEOVER_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class VoiceoverUpload(BaseModel):
    clip_id: int
    voiceover_at: float = 0.0
    audio_base64: str = ""
    filename: str = "voiceover.wav"


class HuntControl(BaseModel):
    platform: str = ""
    handle: str = ""


class YtKeySave(BaseModel):
    api_key: str = ""
    channel: str = ""


class UpdateApply(BaseModel):
    dry: bool = False


class ScheduleItem(BaseModel):
    clip_id: int
    day: str = ""       # "2026-11-23" or a weekday name ("Mon", "Monday")
    time: str = ""      # "18:00"
    platform: str = "tiktok"


class ScheduleDryRun(BaseModel):
    items: list = None


# ---------------------------------------------------------------------------
# Voice-over
# ---------------------------------------------------------------------------

@router.post("/api/voiceover")
def upload_voiceover(req: VoiceoverUpload):
    """Add Sara's recorded line to an already-rendered clip.

    The line is saved under edited/voiceovers/, mixed into the clip
    at ``voiceover_at`` seconds (game audio ducks under it), and the
    clip's edited file is replaced with the voiced version. The
    video is copied untouched, so this takes seconds.
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (req.clip_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Clip #{req.clip_id} not found")
    clip = dict(row)
    edited_path = clip.get("edited_path")
    if not edited_path or not os.path.exists(edited_path):
        raise HTTPException(status_code=400, detail="Clip not rendered yet")

    ext = Path(req.filename or "voiceover.wav").suffix.lower()
    if ext not in VOICEOVER_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Voice-over must be one of: {', '.join(sorted(VOICEOVER_EXTENSIONS))}")
    try:
        audio = base64.b64decode(req.audio_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="audio_base64 is not valid base64")
    if not audio:
        raise HTTPException(status_code=400, detail="Empty recording")

    voice_dir = EDITED_DIR / "voiceovers"
    voice_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    voice_path = voice_dir / f"clip_{req.clip_id}_{stamp}{ext}"
    voice_path.write_bytes(audio)

    voiced_path = str(Path(edited_path).with_name(
        Path(edited_path).stem + "_vo.mp4"))
    try:
        voiceover.add_voiceover(edited_path, str(voice_path),
                                req.voiceover_at, voiced_path)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    update_clip_status(req.clip_id, "ready_to_upload", edited_path=voiced_path)
    log_event("INFO", "WebAPI",
              f"Voice-over added to clip #{req.clip_id} at {req.voiceover_at}s")
    return {"status": "ok", "clip_id": req.clip_id,
            "edited_path": voiced_path,
            "voiceover_at": req.voiceover_at}


# ---------------------------------------------------------------------------
# Clip permission rules
# ---------------------------------------------------------------------------

@router.get("/api/clip-rules/{platform}/{streamer}")
def clip_rule_endpoint(platform: str, streamer: str):
    """The clipping badge for one streamer: allowed / ask / no /
    unknown, with a note and a source link."""
    return clip_rules.get_rule(platform, streamer)


@router.get("/api/clip-rules")
def clip_rules_list_endpoint():
    return clip_rules.list_rules()


# ---------------------------------------------------------------------------
# Multi-platform live hunt
# ---------------------------------------------------------------------------

def _hunt_streamers(streamers_param):
    """Parse "kick:adinross,twitch:xqc" into [{"platform", "handle"}].
    None means the monitored Kick streamers from config."""
    if not streamers_param:
        return [{"platform": "kick", "handle": name}
                for name in STREAMERS_TO_MONITOR]
    out = []
    for item in streamers_param.split(","):
        item = item.strip()
        if not item:
            continue
        platform, _, handle = item.partition(":")
        platform, handle = platform.strip().lower(), handle.strip()
        if platform in live_hunt.LOOK and handle:
            out.append({"platform": platform, "handle": handle})
    return out


@router.get("/api/hunt")
def hunt_endpoint(streamers: str = None):
    """Who of your streamers is live right now, on Kick, Twitch and
    YouTube — with viewers, title and whether a hunt recorder is
    running for them."""
    return live_hunt.hunt(BASE_DIR, _hunt_streamers(streamers))


@router.get("/api/hunt/past")
def hunt_past_endpoint(platform: str = "kick", handle: str = ""):
    """A streamer's last 5 past streams, for VOD clipping. Kick's
    lookup is sometimes Cloudflare-blocked — then the row carries an
    "error" instead of failing the request."""
    try:
        return live_hunt.past(platform, handle)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/hunt/start")
def hunt_start_endpoint(req: HuntControl):
    try:
        return live_hunt.start(BASE_DIR, req.platform.strip().lower(),
                               req.handle.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/hunt/stop")
def hunt_stop_endpoint(req: HuntControl):
    return live_hunt.stop(req.platform.strip().lower(), req.handle.strip(),
                          workspace=BASE_DIR)


# ---------------------------------------------------------------------------
# YouTube stats (needs Sara's API key / OAuth client to run live)
# ---------------------------------------------------------------------------

@router.get("/api/yt/status")
def yt_status_endpoint():
    key = os.environ.get("YT_API_KEY", "")
    channel = os.environ.get("YT_CHANNEL", "")
    return {
        "has_key": bool(key),
        "channel": channel,
        "has_oauth": ytstats.client_path(BASE_DIR).exists(),
        "signed_in": ytstats.token_path(BASE_DIR).exists(),
        "network": "Calls go to Google APIs (googleapis.com) only when a key "
                   "is saved or a linked Short is opened; nothing else leaves "
                   "this PC.",
    }


@router.post("/api/yt/key")
def yt_key_endpoint(req: YtKeySave):
    """Save a YouTube Data API key for this run (put YT_API_KEY in
    .env to keep it across restarts). The key is checked with one
    cheap read-only call before it is accepted."""
    key = (req.api_key or "").strip()
    if key:
        if not re.fullmatch(r"[\w-]{20,60}", key):
            raise HTTPException(status_code=400,
                                detail="That does not look like an API key.")
        try:
            ytstats.data_api("videos", key, part="id", id="dQw4w9WgXcQ")
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail=str(e))
        os.environ["YT_API_KEY"] = key
    if req.channel is not None:
        os.environ["YT_CHANNEL"] = req.channel.strip()
    return {"ok": True, "note": "Key saved for this run only — add "
                                "YT_API_KEY to .env to keep it."}


@router.get("/api/yt/channel")
def yt_channel_endpoint(channel: str = None):
    key = os.environ.get("YT_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400,
                            detail="Add your YouTube API key first (POST /api/yt/key).")
    channel = channel or os.environ.get("YT_CHANNEL", "")
    if not channel:
        raise HTTPException(status_code=400,
                            detail="Add your channel (@handle or UC id).")
    try:
        return {"shorts": ytstats.latest_shorts(key, channel, 20)}
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/api/yt/diagnose")
def yt_diagnose_endpoint(video_url: str = "", hook: str = ""):
    """Diagnose why one Short underperformed: Level 1 stats from the
    API key, plus Level 2 retention when signed in with OAuth."""
    key = os.environ.get("YT_API_KEY", "")
    if not key:
        raise HTTPException(status_code=400,
                            detail="Add your YouTube API key first (POST /api/yt/key).")
    video_id = ytstats.vid_id(video_url)
    if not video_id:
        raise HTTPException(status_code=400,
                            detail="Paste a YouTube link or the 11-character video id.")
    try:
        found = ytstats.videos(key, [video_id])
        if not found:
            raise HTTPException(status_code=404,
                                detail="Video not found (private or deleted?).")
        video = found[0]
        channel = os.environ.get("YT_CHANNEL", "")
        others = ytstats.latest_shorts(key, channel, 20) if channel else []
        deep_stats = ytstats.deep(BASE_DIR, video)
        out = {"video": video, "analytics": {k: v for k, v in deep_stats.items()
                                             if k != "median"}}
        out.update(ytstats.diagnose(video, others, deep_stats, hook=hook))
        return out
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---------------------------------------------------------------------------
# Self-update
# ---------------------------------------------------------------------------

@router.get("/api/update/check")
def update_check_endpoint():
    return self_update.check(root=BASE_DIR)


@router.post("/api/update/apply")
def update_apply_endpoint(req: UpdateApply):
    """Apply an update from the manifest (UPDATE_MANIFEST_URL), or
    dry-run it to list what would be replaced. Backs up first and
    rolls back on failure."""
    return self_update.apply(BASE_DIR, root=BASE_DIR, dry=req.dry)


# ---------------------------------------------------------------------------
# Schedule dry-run (review UX: what WOULD be sent, before it is sent)
# ---------------------------------------------------------------------------

WEEKDAYS = {"mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1,
            "wed": 2, "wednesday": 2, "thu": 3, "thur": 3, "thurs": 3,
            "thursday": 3, "fri": 4, "friday": 4, "sat": 5, "saturday": 5,
            "sun": 6, "sunday": 6}


def resolve_schedule_day(day, today=None):
    """A schedule day — "2026-11-23" or a weekday name ("Mon") — to a
    date. A weekday name means the next occurrence of that day
    (today when it is that weekday). Returns a date, or None."""
    from datetime import date, timedelta
    today = today or date.today()
    text = (day or "").strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        pass
    weekday = WEEKDAYS.get(text.lower())
    if weekday is None:
        return None
    delta = (weekday - today.weekday()) % 7
    return today + timedelta(days=delta)


def dry_run_schedule(items, clips_by_id, queued_clip_ids, today=None):
    """What a week schedule WOULD send, without sending anything.

    ``items``: [{"clip_id", "day", "time", "platform"}] (day is a date
    or a weekday name, time is "HH:MM"). ``clips_by_id``: {clip_id:
    {"title", "approval_status", "edited_path", ...}}.
    ``queued_clip_ids``: clip ids already in the upload queue.
    Returns {"planned": [...], "warnings": [...]} — planned entries
    carry the resolved datetime and the clip title; warnings explain
    every item that would NOT be sent and why.
    """
    from datetime import datetime as dt
    planned, warnings = [], []
    seen = set()
    for item in items or []:
        clip_id = item.get("clip_id")
        label = f"Clip #{clip_id}"
        clip = clips_by_id.get(clip_id)
        if clip is None:
            warnings.append(f"{label}: not found — skipped.")
            continue
        if not clip.get("edited_path"):
            warnings.append(f"{label} ({clip.get('title', '')}): not rendered yet — skipped.")
            continue
        if clip.get("approval_status") not in ("approved", "pending"):
            warnings.append(f"{label} ({clip.get('title', '')}): approval status is "
                            f"'{clip.get('approval_status')}' — approve it first.")
            continue
        if clip_id in queued_clip_ids:
            warnings.append(f"{label} ({clip.get('title', '')}): already in the upload queue.")
            continue
        day = resolve_schedule_day(item.get("day", ""), today=today)
        if day is None:
            warnings.append(f"{label}: day '{item.get('day')}' is not a date or weekday — skipped.")
            continue
        try:
            hour, minute = (int(x) for x in str(item.get("time", "")).split(":"))
            when = dt(day.year, day.month, day.day, hour, minute)
        except (ValueError, TypeError):
            warnings.append(f"{label}: time '{item.get('time')}' is not HH:MM — skipped.")
            continue
        key = (when, str(item.get("platform", "tiktok")).lower())
        if key in seen:
            warnings.append(f"{label}: two uploads at {when:%Y-%m-%d %H:%M} on "
                            f"{item.get('platform')} — move one.")
            continue
        seen.add(key)
        planned.append({
            "clip_id": clip_id,
            "title": clip.get("title", ""),
            "platform": str(item.get("platform", "tiktok")).lower(),
            "scheduled_for": when.isoformat(timespec="minutes"),
        })
    planned.sort(key=lambda p: p["scheduled_for"])
    return {"planned": planned, "warnings": warnings}


@router.post("/api/schedule/dry-run")
def schedule_dry_run_endpoint(req: ScheduleDryRun):
    """Preview what a dragged-onto-the-week schedule would send, with
    warnings for anything that would be skipped. Nothing is queued."""
    items = req.items or []
    clip_ids = [item.get("clip_id") for item in items
                if isinstance(item, dict) and item.get("clip_id") is not None]
    clips_by_id = {}
    if clip_ids:
        conn = get_connection()
        placeholders = ",".join("?" for _ in clip_ids)
        rows = conn.execute(
            f"SELECT id, title, approval_status, edited_path FROM clips "
            f"WHERE id IN ({placeholders})", clip_ids).fetchall()
        conn.close()
        clips_by_id = {row["id"]: dict(row) for row in rows}
    try:
        from upload_scheduler import list_queue
        queued = {job["clip_id"] for job in list_queue(status="pending", limit=500)}
    except Exception:
        queued = set()
    return dry_run_schedule(items, clips_by_id, queued)
