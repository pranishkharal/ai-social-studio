"""Tests for youtube_uploader.py and instagram_uploader.py.

Google API clients and Meta Graph API calls are mocked — no network, no keys.
"""

import json
import urllib.error
from unittest.mock import MagicMock

import pytest

import instagram_uploader
import youtube_uploader
from youtube_uploader import YouTubeUploadError, parse_tags, upload_clip, upload_short
from instagram_uploader import InstagramUploadError, upload_clip_reel, upload_reel


# ---------------------------------------------------------------------------
# YouTube: tag parsing
# ---------------------------------------------------------------------------

def test_parse_tags_string():
    assert parse_tags("#kick #funny, xqc") == ["kick", "funny", "xqc"]


def test_parse_tags_list_and_dedupe():
    assert parse_tags(["Kick", "kick", "#KICK", "clip"]) == ["Kick", "clip"]


def test_parse_tags_empty():
    assert parse_tags("") == []
    assert parse_tags(None) == []


def test_parse_tags_truncates_to_500_chars():
    tags = parse_tags(["x" * 300, "y" * 300])
    assert sum(len(t) + 1 for t in tags) <= 500


# ---------------------------------------------------------------------------
# YouTube: dry run + missing libraries + mocked upload
# ---------------------------------------------------------------------------

def test_upload_short_dry_run_returns_fake_url(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video")
    url = upload_short(str(video), "Title", "Desc", "#a #b", dry_run=True)
    assert url == "https://www.youtube.com/watch?v=DRY_RUN"


def test_upload_short_missing_file():
    with pytest.raises(YouTubeUploadError, match="not found"):
        upload_short("/no/such/file.mp4", "Title", dry_run=True)


def test_upload_short_missing_title(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(YouTubeUploadError, match="title is required"):
        upload_short(str(video), "   ", dry_run=True)


def test_upload_short_bad_privacy(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(YouTubeUploadError, match="Invalid privacy"):
        upload_short(str(video), "T", privacy="secret", dry_run=True)


def test_upload_short_without_google_libs_raises_clear_error(tmp_path, monkeypatch):
    # google libs are not installed in this VM: the error must say how to fix it.
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(youtube_uploader, "_GOOGLE_LIBS_AVAILABLE", False)
    monkeypatch.setattr(youtube_uploader, "_GOOGLE_IMPORT_ERROR", ImportError("nope"))
    with pytest.raises(YouTubeUploadError, match="pip install google-api-python-client"):
        upload_short(str(video), "Title")


def test_upload_short_mocked_service(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    insert_request = MagicMock()
    insert_request.next_chunk.return_value = (None, {"id": "VID123"})
    service = MagicMock()
    service.videos.return_value.insert.return_value = insert_request
    monkeypatch.setattr(youtube_uploader, "MediaFileUpload", MagicMock(return_value="media"))

    url = upload_short(str(video), "My Short", "Desc here", "#kick #funny",
                       service=service)
    assert url == "https://www.youtube.com/watch?v=VID123"
    body = service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["snippet"]["title"] == "My Short"
    assert body["snippet"]["tags"] == ["kick", "funny"]
    assert body["status"]["privacyStatus"] == "public"


def test_upload_short_schedule_sets_private_and_publish_at(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    insert_request = MagicMock()
    insert_request.next_chunk.return_value = (None, {"id": "VID999"})
    service = MagicMock()
    service.videos.return_value.insert.return_value = insert_request
    monkeypatch.setattr(youtube_uploader, "MediaFileUpload", MagicMock(return_value="media"))

    upload_short(str(video), "T", schedule_time="2026-12-01T15:00:00Z",
                 service=service)
    body = service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["publishAt"] == "2026-12-01T15:00:00Z"


def test_upload_short_retries_transient_error_then_succeeds(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    insert_request = MagicMock()
    insert_request.next_chunk.side_effect = [
        IOError("connection reset"),
        (None, {"id": "VID777"}),
    ]
    service = MagicMock()
    service.videos.return_value.insert.return_value = insert_request
    monkeypatch.setattr(youtube_uploader, "MediaFileUpload", MagicMock(return_value="media"))
    monkeypatch.setattr(youtube_uploader.time, "sleep", lambda s: None)

    url = upload_short(str(video), "T", service=service)
    assert url == "https://www.youtube.com/watch?v=VID777"
    assert insert_request.next_chunk.call_count == 2


def test_upload_clip_dry_run_uses_db_metadata(fresh_db, tmp_path):
    video = tmp_path / "edited.mp4"
    video.write_bytes(b"x")
    clip_id = fresh_db["seed_clip"](
        youtube_title="YT Title", youtube_description="YT Desc",
        edited_path=str(video))
    url = upload_clip(clip_id, dry_run=True)
    assert url.endswith("DRY_RUN")


def test_upload_clip_missing_clip(fresh_db):
    with pytest.raises(YouTubeUploadError, match="not found"):
        upload_clip(999, dry_run=True)


def test_upload_clip_no_video_file(fresh_db):
    clip_id = fresh_db["seed_clip"](edited_path=None)
    with pytest.raises(YouTubeUploadError, match="no rendered video"):
        upload_clip(clip_id, dry_run=True)


# ---------------------------------------------------------------------------
# Instagram: mocked Graph API
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def ig_env(monkeypatch):
    monkeypatch.setenv("META_IG_USER_ID", "17841400008460056")
    monkeypatch.setenv("META_IG_ACCESS_TOKEN", "test-token")
    return "17841400008460056"


def test_upload_reel_full_flow(ig_env, monkeypatch):
    calls = []
    poll_count = {"n": 0}

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        if req.get_method() == "POST" and url.endswith("/media"):
            return FakeResponse({"id": "container-1"})
        if req.get_method() == "GET" and "container-1" in url:
            poll_count["n"] += 1
            status = "IN_PROGRESS" if poll_count["n"] == 1 else "FINISHED"
            return FakeResponse({"status_code": status, "id": "container-1"})
        if req.get_method() == "POST" and url.endswith("/media_publish"):
            return FakeResponse({"id": "media-42"})
        if req.get_method() == "GET" and "media-42" in url:
            return FakeResponse({"permalink": "https://www.instagram.com/reel/ABC123/"})
        raise AssertionError(f"unexpected request: {req.get_method()} {url}")

    monkeypatch.setattr(instagram_uploader.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(instagram_uploader.time, "sleep", lambda s: None)

    url = upload_reel("https://example.com/clip.mp4", "Funny moment #kick")
    assert url == "https://www.instagram.com/reel/ABC123/"
    assert poll_count["n"] == 2
    # Container creation posted the right fields.
    container_call = [c for c in calls if c.endswith("/media")][0]
    assert "graph.facebook.com/v21.0" in container_call


def test_upload_reel_dry_run(ig_env):
    url = upload_reel("https://example.com/clip.mp4", "caption", dry_run=True)
    assert url == "https://www.instagram.com/reel/DRY_RUN"


def test_upload_reel_rejects_local_path(ig_env):
    with pytest.raises(InstagramUploadError, match="publicly reachable video URL"):
        upload_reel("C:/clips/edited.mp4", "caption")


def test_upload_reel_missing_credentials(monkeypatch):
    monkeypatch.delenv("META_IG_USER_ID", raising=False)
    monkeypatch.delenv("META_IG_ACCESS_TOKEN", raising=False)
    with pytest.raises(InstagramUploadError, match="META_IG_USER_ID"):
        upload_reel("https://example.com/clip.mp4", "caption")


def test_upload_reel_container_error(ig_env, monkeypatch):
    def fake_urlopen(req, timeout=None):
        if req.get_method() == "POST":
            return FakeResponse({"id": "container-9"})
        return FakeResponse({"status_code": "ERROR", "status": "media invalid"})

    monkeypatch.setattr(instagram_uploader.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(InstagramUploadError, match="rejected the Reels container"):
        upload_reel("https://example.com/bad.mp4", "caption")


def test_upload_reel_poll_timeout(ig_env, monkeypatch):
    def fake_urlopen(req, timeout=None):
        if req.get_method() == "POST":
            return FakeResponse({"id": "container-9"})
        return FakeResponse({"status_code": "IN_PROGRESS"})

    monkeypatch.setattr(instagram_uploader.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(instagram_uploader.time, "sleep", lambda s: None)
    with pytest.raises(InstagramUploadError, match="did not finish processing"):
        upload_reel("https://example.com/slow.mp4", "caption", timeout=0.01)


def test_upload_reel_api_http_error(ig_env, monkeypatch):
    def fake_urlopen(req, timeout=None):
        body = json.dumps({"error": {"message": "Invalid OAuth token"}}).encode()
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, _BodyIO(body))

    class _BodyIO:
        def __init__(self, b):
            self._b = b

        def read(self):
            return self._b

        def close(self):
            pass

    monkeypatch.setattr(instagram_uploader.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(InstagramUploadError, match="Invalid OAuth token"):
        upload_reel("https://example.com/clip.mp4", "caption")


def test_upload_clip_reel_dry_run(fresh_db, ig_env):
    clip_id = fresh_db["seed_clip"](instagram_caption="IG caption here")
    url = upload_clip_reel(clip_id, "https://example.com/clip.mp4", dry_run=True)
    assert url == "https://www.instagram.com/reel/DRY_RUN"


def test_upload_clip_reel_missing_clip(fresh_db, ig_env):
    with pytest.raises(InstagramUploadError, match="not found"):
        upload_clip_reel(999, "https://example.com/clip.mp4", dry_run=True)
