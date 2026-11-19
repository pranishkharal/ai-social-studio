"""
YouTube Shorts uploader for AI Social Studio (Feature 5, YouTube half).

Uploads rendered 9:16 clips to YouTube as Shorts via the YouTube Data API v3,
using the AI-generated ``youtube_title`` / ``youtube_description`` / ``tags``
already stored on each clip row.

Setup (one time, on the machine that runs the engine):
    1. pip install google-api-python-client google-auth-oauthlib
    2. In Google Cloud Console, create an OAuth 2.0 "Desktop app" client and
       download the JSON as ``client_secrets.json`` next to this file
       (or point YOUTUBE_CLIENT_SECRETS_FILE at it).
    3. Run ``python -m youtube_uploader`` once: a browser window opens for the
       one-time Google sign-in. The refresh token is saved to ``token.json``
       next to this file and reused afterwards.

Security: ``client_secrets.json`` and ``token.json`` are credentials.
NEVER commit them — add both to .gitignore. They are read from disk only.

Env vars:
    YOUTUBE_CLIENT_SECRETS_FILE  Path to the OAuth client-secrets JSON.
                                 Default: <BASE_DIR>/client_secrets.json
    YOUTUBE_TOKEN_FILE           Path to the stored OAuth token JSON.
                                 Default: <BASE_DIR>/token.json

The Google client libraries are OPTIONAL imports: if they are not installed,
upload functions raise a clear error telling the user what to install,
instead of crashing the whole engine at import time.
"""

import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from config import BASE_DIR
from database import get_connection, log_event

# ---------------------------------------------------------------------------
# Optional Google API imports (never crash the engine if missing)
# ---------------------------------------------------------------------------

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    _GOOGLE_LIBS_AVAILABLE = True
    _GOOGLE_IMPORT_ERROR = None
except Exception as e:  # ImportError or anything else at import time
    build = None
    HttpError = None
    MediaFileUpload = None
    InstalledAppFlow = None
    Request = None
    Credentials = None
    _GOOGLE_LIBS_AVAILABLE = False
    _GOOGLE_IMPORT_ERROR = e

try:
    import httplib2
    _RETRIABLE_EXCEPTIONS = (httplib2.HttpLib2Error, IOError, OSError)
except Exception:
    httplib2 = None
    _RETRIABLE_EXCEPTIONS = (IOError, OSError)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

CHUNK_SIZE_BYTES = 8 * 1024 * 1024  # 8 MiB resumable-upload chunks
MAX_UPLOAD_RETRIES = 10
RETRIABLE_STATUS_CODES = (500, 502, 503, 504)

MAX_TITLE_CHARS = 100
MAX_DESCRIPTION_CHARS = 5000
MAX_TAGS_CHARS = 500  # YouTube limit on the combined tag string


class YouTubeUploadError(Exception):
    """Raised when a YouTube upload cannot be completed."""


def _require_google_libs():
    if not _GOOGLE_LIBS_AVAILABLE:
        raise YouTubeUploadError(
            "YouTube upload needs the Google API client libraries, which are "
            "not installed. Run: pip install google-api-python-client "
            f"google-auth-oauthlib (import error was: {_GOOGLE_IMPORT_ERROR})"
        )


def _default_client_secrets_file() -> str:
    return os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", str(BASE_DIR / "client_secrets.json"))


def _default_token_file() -> str:
    return os.getenv("YOUTUBE_TOKEN_FILE", str(BASE_DIR / "token.json"))


def _to_rfc3339(value) -> str:
    """Accepts a datetime or ISO-8601 string, returns RFC 3339 UTC ('...Z')."""
    if isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_tags(tags) -> list:
    """Normalises the clip's stored tags into a clean list of YouTube tags.

    Accepts a comma/space separated string (with or without '#') or a list.
    Drops empties, strips '#', dedupes, and truncates to YouTube's 500-char
    combined limit.
    """
    if not tags:
        return []
    if isinstance(tags, str):
        raw = [t for part in tags.split(",") for t in part.split()]
    else:
        raw = list(tags)
    cleaned = []
    seen = set()
    for tag in raw:
        tag = str(tag).strip().lstrip("#").strip()
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            cleaned.append(tag)
    # Enforce YouTube's 500-char combined tag limit.
    total = 0
    kept = []
    for tag in cleaned:
        total += len(tag) + 1
        if total > MAX_TAGS_CHARS:
            break
        kept.append(tag)
    return kept


def get_authenticated_service(client_secrets_file: str = None, token_file: str = None):
    """Builds an authenticated YouTube Data API client.

    Reuses the refresh token in ``token_file`` when present; otherwise runs
    the installed-app OAuth flow once (opens a browser) and saves the token.
    """
    _require_google_libs()
    secrets_path = client_secrets_file or _default_client_secrets_file()
    token_path = token_file or _default_token_file()

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(secrets_path):
                raise YouTubeUploadError(
                    f"YouTube OAuth client secrets not found at {secrets_path}. "
                    "Download the Desktop-app OAuth JSON from Google Cloud Console "
                    "and save it there, or set YOUTUBE_CLIENT_SECRETS_FILE."
                )
            flow = InstalledAppFlow.from_client_secrets_file(secrets_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
        log_event("INFO", "YouTubeUploader", f"Saved YouTube OAuth token to {token_path}")

    return build(API_SERVICE_NAME, API_VERSION, credentials=creds)


def _resumable_upload(insert_request):
    """Executes a resumable upload request with retries (Google's pattern).

    Returns the API response dict, which contains the new video's ``id``.
    """
    response = None
    error = None
    retry = 0
    while response is None:
        try:
            _status, response = insert_request.next_chunk()
            if response is not None:
                if "id" in response:
                    return response
                raise YouTubeUploadError(
                    f"YouTube upload finished but the response has no video id: {response}"
                )
        except Exception as e:
            # HttpError with a retriable status, or a retriable transport error.
            is_http = HttpError is not None and isinstance(e, HttpError)
            if is_http and getattr(e, "resp", None) is not None and e.resp.status in RETRIABLE_STATUS_CODES:
                error = f"YouTube API returned retriable HTTP {e.resp.status}: {e}"
            elif isinstance(e, _RETRIABLE_EXCEPTIONS):
                error = f"Retriable upload error: {e}"
            else:
                raise YouTubeUploadError(f"YouTube upload failed: {e}") from e

        if error is not None:
            log_event("WARNING", "YouTubeUploader", error)
            retry += 1
            if retry > MAX_UPLOAD_RETRIES:
                raise YouTubeUploadError(
                    f"YouTube upload failed after {MAX_UPLOAD_RETRIES} retries: {error}"
                )
            sleep_seconds = random.random() * (2 ** retry)
            log_event("INFO", "YouTubeUploader",
                      f"Retrying YouTube upload in {sleep_seconds:.1f}s (attempt {retry})")
            time.sleep(sleep_seconds)
            error = None


def upload_short(video_path: str, title: str, description: str = "",
                 tags=None, privacy: str = "public", schedule_time=None,
                 dry_run: bool = False, client_secrets_file: str = None,
                 token_file: str = None, service=None) -> str:
    """Uploads a video file as a YouTube Short and returns its watch URL.

    Args:
        video_path: Path to the rendered 9:16 MP4.
        title: Short title (truncated to 100 chars).
        description: Description (truncated to 5000 chars).
        tags: Tag string or list (see :func:`parse_tags`).
        privacy: "public", "unlisted" or "private".
        schedule_time: Optional datetime / ISO string to publish later. When
            set, the video is uploaded as private with a publishAt time.
        dry_run: When True, validates inputs and returns a fake URL without
            touching the YouTube API (safe for testing).
        service: Optional pre-built YouTube client (used by tests to inject
            a mock instead of doing OAuth).

    Returns:
        The public watch URL, e.g. https://www.youtube.com/watch?v=VIDEO_ID
    """
    if not video_path or not os.path.exists(video_path):
        raise YouTubeUploadError(f"Video file not found: {video_path}")
    if not title or not str(title).strip():
        raise YouTubeUploadError("A title is required to upload a YouTube Short")
    if privacy not in ("public", "unlisted", "private"):
        raise YouTubeUploadError(f"Invalid privacy '{privacy}': use public, unlisted or private")

    title = str(title).strip()[:MAX_TITLE_CHARS]
    description = (description or "")[:MAX_DESCRIPTION_CHARS]
    tag_list = parse_tags(tags)

    if dry_run:
        log_event("INFO", "YouTubeUploader",
                  f"[dry-run] Would upload {video_path} as '{title}' ({privacy})")
        return "https://www.youtube.com/watch?v=DRY_RUN"

    # A caller-supplied service (tests, or a pre-built client) skips the
    # lib check; building one here requires the Google libraries.
    yt = service or get_authenticated_service(client_secrets_file, token_file)
    if MediaFileUpload is None:
        _require_google_libs()  # raises the clear pip-install error

    status = {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}
    if schedule_time:
        # publishAt only works on private videos; YouTube publishes it at the
        # scheduled time automatically.
        status["privacyStatus"] = "private"
        status["publishAt"] = _to_rfc3339(schedule_time)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tag_list,
            "categoryId": "22",  # People & Blogs — closest fit for streamer clips
        },
        "status": status,
    }

    media = MediaFileUpload(video_path, chunksize=CHUNK_SIZE_BYTES, resumable=True)
    insert_request = yt.videos().insert(
        part="snippet,status", body=body, media_body=media
    )
    log_event("INFO", "YouTubeUploader", f"Uploading {video_path} to YouTube as '{title}'")
    response = _resumable_upload(insert_request)
    video_id = response["id"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    log_event("INFO", "YouTubeUploader", f"YouTube Short published: {url}")
    return url


def upload_clip(clip_id: int, privacy: str = "public", schedule_time=None,
                dry_run: bool = False, service=None) -> str:
    """Uploads an existing clip (by DB id) to YouTube Shorts.

    Uses the clip's AI-generated ``youtube_title`` / ``youtube_description``
    / ``tags`` and its rendered ``edited_path``. Falls back to the clip title
    and caption when the YouTube-specific fields are empty.
    """
    conn = get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    if not row:
        raise YouTubeUploadError(f"Clip #{clip_id} not found")
    clip = dict(row)

    video_path = clip.get("edited_path") or clip.get("raw_path")
    if not video_path:
        raise YouTubeUploadError(f"Clip #{clip_id} has no rendered video file yet")

    title = clip.get("youtube_title") or clip.get("title") or f"Kick highlight #{clip_id}"
    description = clip.get("youtube_description") or clip.get("caption") or ""
    return upload_short(
        video_path=video_path,
        title=title,
        description=description,
        tags=clip.get("tags"),
        privacy=privacy,
        schedule_time=schedule_time,
        dry_run=dry_run,
        service=service,
    )


def fetch_video_statistics(video_url: str, service=None):
    """Best-effort view/like counts for a YouTube URL.

    Returns ``{"views": int, "likes": int}`` or ``None`` when the URL is not
    a YouTube link or the API call fails. Used by the earnings tracker.
    """
    import re

    match = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", video_url or "")
    if not match:
        return None
    video_id = match.group(1)
    try:
        _require_google_libs()
        yt = service or get_authenticated_service()
        resp = yt.videos().list(part="statistics", id=video_id).execute()
        items = resp.get("items") or []
        if not items:
            return None
        stats = items[0].get("statistics", {})
        return {
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
        }
    except Exception as e:
        log_event("WARNING", "YouTubeUploader", f"Could not fetch stats for {video_url}: {e}")
        return None


if __name__ == "__main__":
    # One-time OAuth setup: run `python -m youtube_uploader` and sign in
    # in the browser window that opens. The token is saved for later runs.
    get_authenticated_service()
    print("YouTube OAuth is set up. Token saved — uploads can now run headless.")
