"""
Discord / Telegram alerts (feature 16).

Sends a Discord webhook message and/or a Telegram bot message when
something worth knowing happens: a streamer goes live, a high-score clip
is ready, or an upload finishes / fails. Notifications previously lived
only inside the dashboard; these reach Sara wherever she is.

Stdlib only (urllib) — no new dependencies. This module is importable
standalone: it never imports the app's config/database at module level,
and falls back to the stdlib logging module if database.log_event is
unavailable.

Configuration (environment variables):
    DISCORD_WEBHOOK_URL  Full Discord channel webhook URL
                         (Server Settings -> Integrations -> Webhooks).
    TELEGRAM_BOT_TOKEN   Telegram bot token from @BotFather.
    TELEGRAM_CHAT_ID     Chat ID to message (from @userinfobot, or the
                         bot's getUpdates response after messaging it).

Sending rules:
    * notify_event() dispatches to whichever channels are configured
      (both, if both are set). If none are configured it is a silent
      no-op that only logs at debug level.
    * Every send runs in a daemon thread so alerting never blocks the
      pipeline, and no send ever raises — failures are caught and logged
      via database.log_event (with a logging fallback).

Wiring (done by the coordinator — NOT here): call notify_event() (or the
alert_* helpers) from database.add_notification / pipeline.py at the
points where notifications are created.
"""

import json
import logging
import os
import threading
import urllib.error  # noqa: F401  (re-exported for tests that patch urlopen)
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 10

# Discord embed colours
COLOR_GREEN = 0x53FC18  # Kick green — good news
COLOR_BLUE = 0x3498DB   # info
COLOR_RED = 0xE74C3C    # failure
COLOR_GOLD = 0xF1C40F   # highlight


# ---------------------------------------------------------------------------
# Logging helper (database.log_event when available, stdlib logging otherwise)
# ---------------------------------------------------------------------------

_log_event_fn = None
_log_event_tried = False


def _log_event(level: str, message: str):
    """Log via database.log_event when importable, else stdlib logging."""
    global _log_event_fn, _log_event_tried
    if not _log_event_tried:
        _log_event_tried = True
        try:
            from database import log_event as _db_log_event
            _log_event_fn = _db_log_event
        except Exception:
            _log_event_fn = None
    if _log_event_fn is not None:
        try:
            _log_event_fn(level, "Alerts", message)
            return
        except Exception:
            pass
    getattr(logger, level.lower(), logger.info)(message)


# ---------------------------------------------------------------------------
# Low-level senders (synchronous; never raise)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict) -> int:
    """POST a JSON payload; returns the HTTP status code. May raise."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ai-social-studio/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.status


def send_discord_webhook(webhook_url, title, message, fields=None, color=None) -> bool:
    """Send one Discord embed via a webhook URL. Returns False on failure.

    `fields` is an optional dict of name -> value shown as inline embed
    fields; `color` is an optional integer embed colour (default blue).
    """
    if not webhook_url:
        return False
    embed = {
        "title": title,
        "description": message,
        "color": COLOR_BLUE if color is None else color,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if fields:
        embed["fields"] = [
            {"name": str(name), "value": str(value), "inline": True}
            for name, value in fields.items()
        ]
    try:
        _post_json(webhook_url, {"embeds": [embed]})
        return True
    except Exception as exc:
        _log_event("WARNING", f"Discord webhook failed: {exc}")
        return False


def send_telegram_message(bot_token, chat_id, text) -> bool:
    """Send one Telegram message via the Bot API. Returns False on failure."""
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    try:
        _post_json(url, payload)
        return True
    except Exception as exc:
        _log_event("WARNING", f"Telegram message failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Dispatcher (non-blocking; never raises)
# ---------------------------------------------------------------------------

def _run_async(target, *args, **kwargs):
    """Run `target` in a daemon thread so alerting never blocks the caller.

    Kept as a module-level function so tests can monkeypatch it to run
    synchronously.
    """
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread


def notify_event(event_type, title, message, color=None, **fields) -> bool:
    """Send an alert to every configured channel (Discord and/or Telegram).

    Reads DISCORD_WEBHOOK_URL, TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from
    the environment. `fields` become Discord embed fields and are appended
    to the Telegram text. Returns True if at least one channel was
    dispatched, False when nothing is configured. Never raises.
    """
    try:
        discord_url = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        if not discord_url and not (telegram_token and telegram_chat_id):
            logger.debug(
                "notify_event(%s): no alert channel configured; skipping",
                event_type,
            )
            return False

        if discord_url:
            _run_async(
                send_discord_webhook,
                discord_url,
                title,
                message,
                fields or None,
                color,
            )
        if telegram_token and telegram_chat_id:
            text = f"{title}\n{message}"
            if fields:
                text += "\n" + "\n".join(f"{name}: {value}" for name, value in fields.items())
            _run_async(send_telegram_message, telegram_token, telegram_chat_id, text)
        return True
    except Exception as exc:  # never let alerting break the caller
        _log_event("WARNING", f"notify_event({event_type}) failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def alert_streamer_live(streamer, title, viewers) -> bool:
    """Alert that `streamer` just went live (title + viewer count)."""
    return notify_event(
        "streamer_live",
        f"🔴 {streamer} is LIVE",
        title or "",
        color=COLOR_GREEN,
        Streamer=streamer,
        Viewers=viewers,
    )


def alert_clip_ready(streamer, clip_id, ai_score, title) -> bool:
    """Alert that a clip is rendered and waiting in the review queue."""
    return notify_event(
        "clip_ready",
        f"✂️ Clip ready — {streamer}",
        title or "",
        color=COLOR_BLUE,
        Clip=f"#{clip_id}",
        AI_score=ai_score,
        Streamer=streamer,
    )


def alert_upload_done(clip_id, platform, url) -> bool:
    """Alert that a clip finished uploading to `platform`."""
    return notify_event(
        "upload_done",
        f"✅ Upload complete — {platform}",
        url or "",
        color=COLOR_GREEN,
        Clip=f"#{clip_id}",
        Platform=platform,
        URL=url or "",
    )


def alert_upload_failed(clip_id, platform, error) -> bool:
    """Alert that an upload to `platform` failed."""
    return notify_event(
        "upload_failed",
        f"❌ Upload failed — {platform}",
        str(error),
        color=COLOR_RED,
        Clip=f"#{clip_id}",
        Platform=platform,
        Error=str(error),
    )
