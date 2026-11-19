"""
Instagram Reels uploader for AI Social Studio (Feature 5, Instagram half).

Publishes rendered 9:16 clips as Instagram Reels via the Meta Graph API,
using the AI-generated ``instagram_caption`` already stored on each clip row.

Uses ONLY the Python standard library (urllib) — no new dependency.

Setup:
    1. Connect an Instagram Business/Creator account to a Facebook Page.
    2. Create a Meta app with the ``instagram_content_publish`` permission
       and generate a long-lived Page access token.
    3. Set META_IG_USER_ID (the Instagram Business account id) and
       META_IG_ACCESS_TOKEN in the environment.

IMPORTANT HONEST LIMITATION — PUBLIC VIDEO URL REQUIRED:
    The Graph API does NOT accept file uploads. The ``video_url`` you pass to
    :func:`upload_reel` must be a publicly reachable HTTPS URL (a direct link
    to an .mp4 file). A local file path will NOT work. Before calling this
    uploader you must host the clip somewhere public first — e.g. upload the
    MP4 to your own static hosting, an S3 bucket, or any public file host —
    then pass that URL here. A future static-file host inside this app could
    automate that step; until then the URL is supplied by the caller (the
    dashboard enqueue form and the scheduler both accept it).

Flow (Meta Graph API):
    1. POST /{ig-user-id}/media  (media_type=REELS, video_url, caption)
       -> creation container id
    2. Poll GET /{container-id}?fields=status_code until FINISHED
    3. POST /{ig-user-id}/media_publish (creation_id) -> published media id
    4. GET /{media-id}?fields=permalink -> public Reels URL

Env vars:
    META_IG_USER_ID              Instagram Business account id (required).
    META_IG_ACCESS_TOKEN         Long-lived Page access token (required).
    META_GRAPH_API_VERSION       Graph API version. Default: v21.0
    INSTAGRAM_POLL_INTERVAL_SECONDS  Seconds between container status polls.
                                     Default: 5
    INSTAGRAM_POLL_TIMEOUT_SECONDS   Max seconds to wait for processing.
                                     Default: 300
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from database import get_connection, log_event

GRAPH_API_BASE = "https://graph.facebook.com"

MAX_CAPTION_CHARS = 2200  # Instagram caption limit


class InstagramUploadError(Exception):
    """Raised when a Reels publish cannot be completed."""


def _api_version() -> str:
    return os.getenv("META_GRAPH_API_VERSION", "v21.0")


def _credentials(user_id: str = None, access_token: str = None):
    ig_user_id = user_id or os.getenv("META_IG_USER_ID", "")
    token = access_token or os.getenv("META_IG_ACCESS_TOKEN", "")
    if not ig_user_id:
        raise InstagramUploadError(
            "META_IG_USER_ID is not set. Set it to your Instagram Business "
            "account id (find it in Meta's Graph API Explorer)."
        )
    if not token:
        raise InstagramUploadError(
            "META_IG_ACCESS_TOKEN is not set. Set it to a long-lived Page "
            "access token with the instagram_content_publish permission."
        )
    return ig_user_id, token


def _api_request(method: str, path: str, params: dict = None, timeout: int = 60):
    """One Graph API call. Returns the decoded JSON body.

    Raises InstagramUploadError with Meta's error message on failure.
    """
    url = f"{GRAPH_API_BASE}/{_api_version()}{path}"
    data = None
    if method.upper() == "GET":
        if params:
            url += "?" + urllib.parse.urlencode(params)
    else:
        data = urllib.parse.urlencode(params or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
            message = body.get("error", {}).get("message", str(body))
        except Exception:
            message = f"HTTP {e.code}"
        raise InstagramUploadError(f"Meta Graph API error ({method} {path}): {message}") from e
    except urllib.error.URLError as e:
        raise InstagramUploadError(f"Could not reach the Meta Graph API: {e}") from e


def _create_reel_container(ig_user_id: str, token: str, video_url: str, caption: str) -> str:
    body = _api_request("POST", f"/{ig_user_id}/media", {
        "media_type": "REELS",
        "video_url": video_url,
        "caption": caption,
        "access_token": token,
    })
    container_id = body.get("id")
    if not container_id:
        raise InstagramUploadError(f"Container creation returned no id: {body}")
    return container_id


def _wait_for_container(container_id: str, token: str,
                        poll_interval: float = None, timeout: float = None) -> None:
    poll_interval = poll_interval if poll_interval is not None else float(
        os.getenv("INSTAGRAM_POLL_INTERVAL_SECONDS", "5"))
    timeout = timeout if timeout is not None else float(
        os.getenv("INSTAGRAM_POLL_TIMEOUT_SECONDS", "300"))
    deadline = time.monotonic() + timeout
    while True:
        body = _api_request("GET", f"/{container_id}", {
            "fields": "status_code,status",
            "access_token": token,
        })
        status = (body.get("status_code") or "").upper()
        if status == "FINISHED":
            return
        if status in ("ERROR", "EXPIRED"):
            raise InstagramUploadError(
                f"Instagram rejected the Reels container ({status}): "
                f"{body.get('status', '')}. Check that the video URL is a "
                "publicly reachable MP4 (9:16, under 15 minutes)."
            )
        if time.monotonic() >= deadline:
            raise InstagramUploadError(
                f"Instagram container {container_id} did not finish processing "
                f"within {timeout:.0f}s (last status: {status or 'unknown'})."
            )
        time.sleep(poll_interval)


def _publish_container(ig_user_id: str, token: str, container_id: str) -> str:
    body = _api_request("POST", f"/{ig_user_id}/media_publish", {
        "creation_id": container_id,
        "access_token": token,
    })
    media_id = body.get("id")
    if not media_id:
        raise InstagramUploadError(f"Publish returned no media id: {body}")
    return media_id


def _fetch_permalink(media_id: str, token: str):
    """Returns the public Reels URL, or None if it cannot be fetched."""
    try:
        body = _api_request("GET", f"/{media_id}", {
            "fields": "permalink",
            "access_token": token,
        })
        return body.get("permalink")
    except InstagramUploadError as e:
        log_event("WARNING", "InstagramUploader",
                  f"Reel published (id {media_id}) but permalink lookup failed: {e}")
        return None


def upload_reel(video_url: str, caption: str = "", dry_run: bool = False,
                ig_user_id: str = None, access_token: str = None,
                poll_interval: float = None, timeout: float = None) -> str:
    """Publishes a Reel from a PUBLIC video URL. Returns the public Reels URL.

    Args:
        video_url: Publicly reachable HTTPS URL of the .mp4 file. Local file
            paths are NOT accepted by the Graph API — see the module docstring.
        caption: Reels caption (truncated to 2200 chars).
        dry_run: When True, validates inputs/credentials and returns a fake
            URL without calling the Graph API (safe for testing).

    Returns:
        The public permalink (e.g. https://www.instagram.com/reel/...). If the
        permalink lookup fails after a successful publish, returns the media
        id as ``instagram_media:<id>`` instead so the post is still traceable.
    """
    if not video_url or not str(video_url).lower().startswith(("http://", "https://")):
        raise InstagramUploadError(
            "upload_reel needs a publicly reachable video URL (http/https). "
            f"Got: {video_url!r}. Local files must be uploaded to public "
            "hosting first — the Graph API cannot read files from this machine."
        )
    caption = (caption or "")[:MAX_CAPTION_CHARS]
    ig_id, token = _credentials(ig_user_id, access_token)

    if dry_run:
        log_event("INFO", "InstagramUploader",
                  f"[dry-run] Would publish Reel from {video_url} to IG user {ig_id}")
        return "https://www.instagram.com/reel/DRY_RUN"

    log_event("INFO", "InstagramUploader", f"Creating Reels container for {video_url}")
    container_id = _create_reel_container(ig_id, token, video_url, caption)
    log_event("INFO", "InstagramUploader",
              f"Waiting for Instagram to process container {container_id}")
    _wait_for_container(container_id, token, poll_interval, timeout)
    media_id = _publish_container(ig_id, token, container_id)
    permalink = _fetch_permalink(media_id, token)
    url = permalink or f"instagram_media:{media_id}"
    log_event("INFO", "InstagramUploader", f"Instagram Reel published: {url}")
    return url


def upload_clip_reel(clip_id: int, video_url: str, dry_run: bool = False,
                     ig_user_id: str = None, access_token: str = None,
                     poll_interval: float = None, timeout: float = None) -> str:
    """Publishes an existing clip (by DB id) as an Instagram Reel.

    Uses the clip's AI-generated ``instagram_caption`` plus its ``tags``.
    ``video_url`` must be the clip's PUBLIC URL — see the module docstring.
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not row:
        raise InstagramUploadError(f"Clip #{clip_id} not found")
    clip = dict(row)

    caption = (clip.get("instagram_caption") or "").strip()
    tags = (clip.get("tags") or "").strip()
    if tags and tags not in caption:
        caption = f"{caption}\n\n{tags}".strip() if caption else tags
    if not caption:
        caption = clip.get("title") or f"Kick highlight #{clip_id}"

    return upload_reel(
        video_url=video_url,
        caption=caption,
        dry_run=dry_run,
        ig_user_id=ig_user_id,
        access_token=access_token,
        poll_interval=poll_interval,
        timeout=timeout,
    )
