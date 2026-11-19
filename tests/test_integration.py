"""Integration tests for the coordinator's wiring (all tiers).

These tests cover the seams between the workers' modules — the parts no
single worker owned: the encoder verification in the editor, the subtitle
remap in the pipeline, the LLM fallback chain, checkpoint resume, the
Tier 5 render helpers wired into ClipEditor, and the dashboard routers
mounting under the FastAPI stub.

Real ffmpeg is used where a test says so (marked with needs_ffmpeg);
everything else is offline.
"""

import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _render_calls(ffmpeg_calls):
    """Render invocations only (excludes `ffmpeg -encoders` probes)."""
    return [c for c in ffmpeg_calls if "-filter_complex" in c]


# ---------------------------------------------------------------------------
# Router modules mount under the FastAPI stub (conftest)
# ---------------------------------------------------------------------------

def test_router_modules_import_and_register_routes():
    import auth
    import dashboard_routers
    import kick_webhooks

    assert hasattr(auth, "router")
    assert hasattr(dashboard_routers, "router")
    assert hasattr(kick_webhooks, "router")

    def routes(mod):
        return {(m, p) for (m, p, _f) in mod.router.routes}

    assert ("POST", "/api/webhooks/kick") in routes(kick_webhooks)
    assert ("GET", "/api/earnings/summary") in routes(dashboard_routers)
    assert ("GET", "/api/analytics/summary") in routes(dashboard_routers)
    assert ("GET", "/api/queue") in routes(dashboard_routers)
    assert ("GET", "/login") in routes(auth)
    assert ("POST", "/login") in routes(auth)
    assert ("POST", "/logout") in routes(auth)


def test_webhook_helpers_available():
    import kick_webhooks
    assert callable(kick_webhooks.set_monitor)
    assert callable(kick_webhooks.webhooks_configured)
    assert callable(kick_webhooks.get_setup_checklist)
    assert callable(kick_webhooks.get_access_token)
    assert callable(kick_webhooks.subscribe_livestream_status)
    # Nothing configured in the test env: the checklist must say so.
    assert kick_webhooks.webhooks_configured() is False
    checklist = kick_webhooks.get_setup_checklist()
    assert isinstance(checklist, list) and len(checklist) > 0
    assert any("CLIENT_ID" in step for step in checklist)


# ---------------------------------------------------------------------------
# Encoder verification (Tier 5): a listed-but-broken encoder must fail
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_test_encode_accepts_libx264_rejects_bogus_codec():
    from encoder import _test_encode
    assert _test_encode("libx264", ["-preset", "veryfast", "-crf", "21"]) is True
    assert _test_encode("h264_bogus_codec_xyz", []) is False


@needs_ffmpeg
def test_detect_encoder_verify_never_picks_broken_hardware():
    # On a machine with no GPU this must NOT pick h264_nvenc even when
    # ffmpeg lists it; the verified result must actually encode.
    from encoder import _test_encode, detect_encoder
    codec, args = detect_encoder(verify=True)
    assert _test_encode(codec, args) is True


def test_select_encoder_prefers_hardware_when_listed():
    from encoder import select_encoder
    fake = (
        " V..... h264_nvenc  NVIDIA NVENC H.264 encoder\n"
        " V..... libx264     libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10\n"
    )
    codec, args = select_encoder(fake)
    assert codec == "h264_nvenc"
    assert args


# ---------------------------------------------------------------------------
# Subtitle remap (Tier 3): words move to trimmed-clip time
# ---------------------------------------------------------------------------

def test_subtitle_remap_round_trip():
    from silence_cutter import compute_keep_segments, remap_words
    words = [
        {"word": "hello", "start": 0.0, "end": 0.8},
        {"word": "world", "start": 5.0, "end": 5.9},
    ]
    segments = compute_keep_segments(words)
    remapped = remap_words(words, segments)
    assert len(remapped) == 2
    # First word untouched (it is in the first kept segment).
    assert remapped[0]["start"] == pytest.approx(0.0)
    assert remapped[0]["end"] == pytest.approx(0.8)
    # Second word shifts earlier by (about) the removed 4.2 s gap.
    assert remapped[1]["start"] < 5.0
    assert remapped[1]["start"] == pytest.approx(0.8, abs=0.4)
    # Word durations are preserved by the remap.
    assert (remapped[1]["end"] - remapped[1]["start"]) == pytest.approx(0.9, abs=0.05)


# ---------------------------------------------------------------------------
# Config defaults (Tier 1-5): new behaviour is off unless opted into
# ---------------------------------------------------------------------------

def test_config_env_parsers():
    import config
    assert config._env_bool("AI_SOCIAL_STUDIO_NO_SUCH_VAR", True) is True
    assert config._env_bool("AI_SOCIAL_STUDIO_NO_SUCH_VAR", False) is False
    assert config._env_int("AI_SOCIAL_STUDIO_NO_SUCH_VAR", 42) == 42
    assert config._env_float("AI_SOCIAL_STUDIO_NO_SUCH_VAR", 1.5) == 1.5


def test_config_safe_defaults(monkeypatch):
    # With the new env vars unset, every new behaviour defaults to off /
    # to the pre-upgrade behaviour. (Re-read through the parsers so the
    # test does not depend on import-time environment.)
    import os
    for var in ("AUTO_CLIP_ENABLED", "CHAT_HYPE_ENABLED", "BGM_ENABLED",
                "BROLL_ENABLED"):
        monkeypatch.delenv(var, raising=False)
        assert os.getenv(var) is None
    import config
    assert config._env_bool("AUTO_CLIP_ENABLED", False) is False
    assert config._env_bool("BGM_ENABLED", False) is False
    assert config._env_bool("BROLL_ENABLED", False) is False
    assert config._env_int("AUTO_CLIP_MIN_SCORE", 70) == 70
    assert config.SUBTITLE_STYLE in ("classic", "karaoke")


# ---------------------------------------------------------------------------
# Per-streamer profiles (Tier 3)
# ---------------------------------------------------------------------------

def test_profile_lookup_defaults():
    from streamer_profiles import get_profile
    assert get_profile("xqc").get("clip_seconds") == 30
    assert get_profile("westcol").get("clip_seconds") == 60
    default = get_profile("somebody_nobody_has_seen")
    assert default.get("clip_seconds") == 45
    assert default.get("layout") == "blurred_backdrop"


# ---------------------------------------------------------------------------
# Karaoke subtitles + BGM ducking filter (Tier 5 string builders)
# ---------------------------------------------------------------------------

def test_karaoke_ass_contains_karaoke_tags(tmp_path):
    from karaoke_subtitles import generate_karaoke_ass
    words = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
    ]
    out = tmp_path / "karaoke.ass"
    result = generate_karaoke_ass(words, str(out), style="hormozi")
    assert result == str(out)
    content = out.read_text(encoding="utf-8")
    assert "{\\k" in content
    assert "Dialogue:" in content


def test_bgm_ducking_filter_uses_sidechaincompress():
    from bgm import build_ducking_filter, mood_for_category
    filt, label = build_ducking_filter()
    assert "sidechaincompress" in filt
    assert label == "[a_out]"
    assert "[0:a]" in filt and "[1:a]" in filt
    filt2, _ = build_ducking_filter(bgm_label="[2:a]")
    assert "[2:a]" in filt2
    assert mood_for_category("Big Wins") == "epic"
    assert mood_for_category("Not A Category") == "chill"


# ---------------------------------------------------------------------------
# Broken cv2 must degrade gracefully (production hardening)
# ---------------------------------------------------------------------------

def test_broken_cv2_skips_face_evidence(monkeypatch, tmp_path):
    """A cv2 module missing .empty()/.imread() must behave like no cv2."""
    import streamer_detection
    import thumbnail

    broken = types.ModuleType("cv2")

    class BrokenCascade:
        def __init__(self, *a, **k):
            pass

    broken.CascadeClassifier = BrokenCascade
    monkeypatch.setitem(sys.modules, "cv2", broken)

    fake_video = tmp_path / "v.mp4"
    fake_video.write_bytes(b"x")
    assert streamer_detection._face_evidence(str(fake_video)) is None
    assert thumbnail._largest_face_time(str(fake_video), 10.0) is None


# ---------------------------------------------------------------------------
# LLM fallback chain (Tier 5): Gemini -> local -> heuristic
# ---------------------------------------------------------------------------

def test_llm_fallback_lands_on_heuristic(monkeypatch):
    import ai_story_engine
    import llm_providers

    monkeypatch.setattr(ai_story_engine, "gemini_client", None)
    monkeypatch.setattr(
        llm_providers.OpenAICompatibleClient, "generate_json",
        lambda self, prompt, **kw: (_ for _ in ()).throw(
            RuntimeError("no local server")))
    engine = ai_story_engine.AIStoryEngine()
    meta = engine.analyze_story_and_generate_metadata(
        "xqc", "what a win clutch victory", 30)
    assert meta["category"] == "Big Wins"
    assert meta["ai_score"] == 92
    assert meta["llm_provider"] == "heuristic"


def test_llm_fallback_uses_local_model_when_available(monkeypatch):
    import ai_story_engine
    import llm_providers

    monkeypatch.setattr(ai_story_engine, "gemini_client", None)

    def fake_local(self, prompt, **kw):
        return {
            "category": "Funny Moments",
            "ai_score": 90,
            "selection_reason": "Local model test.",
            "story_summary": "Test summary.",
            "viral_title": "XQC DID SOMETHING FUNNY",
            "hook_banner": "Xqc was SHOCKED",
            "screen_caption": "Xqc was SHOCKED",
            "tiktok_caption": "funny caption",
            "youtube_title": "Funny #Shorts",
            "youtube_description": "desc",
            "instagram_caption": "insta",
            "blue_comment": "comment",
            "hashtags": "#xqc",
        }

    monkeypatch.setattr(
        llm_providers.OpenAICompatibleClient, "generate_json", fake_local)
    engine = ai_story_engine.AIStoryEngine()
    meta = engine.analyze_story_and_generate_metadata(
        "xqc", "haha that was funny lol", 30)
    assert meta["llm_provider"] == "openai_compatible"
    assert meta["category"] == "Funny Moments"
    assert meta["ai_score"] == 90


# ---------------------------------------------------------------------------
# Camera-switch fallback (Tier 5): no speaker data -> speaker_tracking
# ---------------------------------------------------------------------------

def test_camera_switch_falls_back_without_speaker_data(tmp_path, monkeypatch):
    import clip_editor
    from PIL import Image

    logs = []
    monkeypatch.setattr(clip_editor, "log_event",
                        lambda level, src, msg: logs.append((level, src, msg)))
    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)
    watermark = tmp_path / "wm.png"
    Image.new("RGBA", (1082, 108), (0, 0, 0, 255)).save(watermark)
    monkeypatch.setattr(clip_editor, "generate_kick_watermark",
                        lambda streamer: str(watermark))

    ffmpeg_calls = []

    def fake_run(cmd, *args, **kwargs):
        ffmpeg_calls.append(list(cmd))
        if str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(clip_editor.subprocess, "run", fake_run)

    # Stub speaker_tracker: trajectory analysis fails, so the
    # speaker_tracking branch itself falls back to a plain smart crop.
    tracker_mod = types.ModuleType("speaker_tracker")

    class BoomTracker:
        def analyze_video_trajectory(self, *a, **k):
            raise RuntimeError("no faces here")

        def get_ffmpeg_crop_filter(self, *a, **k):
            raise AssertionError("must not be called")

    tracker_mod.SpeakerTracker = BoomTracker
    monkeypatch.setitem(sys.modules, "speaker_tracker", tracker_mod)

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    out = tmp_path / "edited.mp4"
    editor = clip_editor.ClipEditor()
    result = editor.edit_clip(str(raw), str(out), streamer="xqc",
                              layout="camera_switch", generate_thumbnail=False)
    assert result == str(out)
    assert any("camera_switch needs" in msg for _, _, msg in logs)
    renders = _render_calls(ffmpeg_calls)
    assert len(renders) == 1
    filt = " ".join(renders[0])
    assert "crop=1080:1920" in filt  # smart-crop fallback, not a crash


def test_split_screen_falls_back_without_speaker_data(tmp_path, monkeypatch):
    import clip_editor
    from PIL import Image

    logs = []
    monkeypatch.setattr(clip_editor, "log_event",
                        lambda level, src, msg: logs.append((level, src, msg)))
    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)
    watermark = tmp_path / "wm.png"
    Image.new("RGBA", (1082, 108), (0, 0, 0, 255)).save(watermark)
    monkeypatch.setattr(clip_editor, "generate_kick_watermark",
                        lambda streamer: str(watermark))

    ffmpeg_calls = []

    def fake_run(cmd, *args, **kwargs):
        ffmpeg_calls.append(list(cmd))
        if str(cmd[-1]).endswith(".mp4"):
            Path(cmd[-1]).touch()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(clip_editor.subprocess, "run", fake_run)

    tracker_mod = types.ModuleType("speaker_tracker")

    class BoomTracker:
        def analyze_video_trajectory(self, *a, **k):
            raise RuntimeError("no faces here")

    tracker_mod.SpeakerTracker = BoomTracker
    monkeypatch.setitem(sys.modules, "speaker_tracker", tracker_mod)

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    out = tmp_path / "edited.mp4"
    editor = clip_editor.ClipEditor()
    result = editor.edit_clip(str(raw), str(out), streamer="xqc",
                              layout="split_screen", generate_thumbnail=False)
    assert result == str(out)
    assert any("split_screen needs" in msg for _, _, msg in logs)
    assert len(_render_calls(ffmpeg_calls)) == 1


# ---------------------------------------------------------------------------
# Pipeline: profile duration + checkpoint resume (Tiers 3 + 4)
# ---------------------------------------------------------------------------

def _make_pipeline(tmp_path, monkeypatch):
    """ClipPipeline with checkpoints redirected to tmp_path."""
    import retry_utils
    from pipeline import ClipPipeline
    monkeypatch.setattr(retry_utils, "DEFAULT_CHECKPOINTS_DIR",
                        tmp_path / "checkpoints")
    return ClipPipeline()


def test_process_moment_uses_profile_duration(fresh_db, tmp_path, monkeypatch):
    import database
    from pipeline import ClipPipeline  # noqa: F401  (import check)

    pipeline = _make_pipeline(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    extract_calls = []
    monkeypatch.setattr(
        pipeline.monitor, "extract_recent_clip",
        lambda streamer, duration: extract_calls.append(duration) or str(raw))
    monkeypatch.setattr(
        pipeline.detector, "evaluate_clip",
        lambda path: {"hype_score": 80.0, "trigger_reason": "test"})
    monkeypatch.setattr(
        pipeline.ai, "transcribe_clip",
        lambda path: {"full_text": "hello world", "words": []})
    monkeypatch.setattr(
        pipeline.ai, "analyze_story_and_generate_metadata",
        lambda **kw: {"category": "Funny Moments", "ai_score": 85,
                      "selection_reason": "test", "story_summary": "s",
                      "viral_title": "T", "hook_banner": "H",
                      "tiktok_caption": "c", "youtube_title": "Y",
                      "youtube_description": "D", "instagram_caption": "I",
                      "blue_comment": "B", "hashtags": "#x"})
    edit_calls = []
    monkeypatch.setattr(
        pipeline.editor, "edit_clip",
        lambda **kw: edit_calls.append(kw) or str(tmp_path / "edited.mp4"))

    result = pipeline.process_moment("xqc", duration=None)
    assert result["status"] == "ready_to_upload"
    # xqc's profile says 30 s clips — the extractor must have been asked
    # for 30 s, not the old hardcoded 25.
    assert extract_calls == [30]
    # Words flow through to the editor (silence trimming input).
    assert edit_calls and edit_calls[0]["words"] == []
    # The clip row records the detected streamers (none here).
    conn = database.get_connection()
    row = conn.execute("SELECT detected_streamers FROM clips WHERE id = ?",
                       (result["clip_id"],)).fetchone()
    conn.close()
    assert json.loads(row["detected_streamers"]) == []


def test_resume_clip_reuses_checkpoints(fresh_db, tmp_path, monkeypatch):
    import database
    from database import add_clip, update_clip_status
    from retry_utils import CheckpointStore

    pipeline = _make_pipeline(tmp_path, monkeypatch)
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")

    clip_id = add_clip({
        "streamer": "xqc", "title": "Moment from xqc",
        "story_summary": "Analyzing...", "hook_banner": "WAIT FOR IT...",
        "caption": "", "tags": "", "hype_score": 80.0,
        "trigger_reason": "test", "duration": 25.0,
        "raw_path": str(raw), "edited_path": None, "status": "failed",
    })
    update_clip_status(clip_id, "failed")

    # The failed run got through transcription + analysis before dying.
    store = CheckpointStore(tmp_path / "checkpoints")
    store.mark_step_done(clip_id, "transcribed", {
        "transcript": "hello world",
        "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
    })
    store.mark_step_done(clip_id, "analyzed", {
        "category": "Big Wins", "ai_score": 92,
        "selection_reason": "test", "story_summary": "s",
        "viral_title": "T", "hook_banner": "H", "tiktok_caption": "c",
        "youtube_title": "Y", "youtube_description": "D",
        "instagram_caption": "I", "blue_comment": "B", "hashtags": "#x",
    })
    pipeline.checkpoints = store

    # Transcription + analysis must NOT run again — prove it by making
    # them explode if called.
    def _boom(*a, **k):
        raise AssertionError("checkpointed step must not re-run")

    monkeypatch.setattr(pipeline.ai, "transcribe_clip", _boom)
    monkeypatch.setattr(pipeline.ai, "analyze_story_and_generate_metadata", _boom)
    edit_calls = []
    monkeypatch.setattr(
        pipeline.editor, "edit_clip",
        lambda **kw: edit_calls.append(kw) or str(tmp_path / "edited.mp4"))

    result = pipeline.resume_clip(clip_id)
    assert result["status"] == "ready_to_upload"
    assert result["category"] == "Big Wins"
    assert result["ai_score"] == 92
    # The stored words reached the editor (subtitle + silence input).
    assert edit_calls and len(edit_calls[0]["words"]) == 1
    assert edit_calls[0]["words"][0]["word"] == "hello"
    conn = database.get_connection()
    row = conn.execute("SELECT status FROM clips WHERE id = ?",
                       (clip_id,)).fetchone()
    conn.close()
    assert row["status"] == "ready_to_upload"


# ---------------------------------------------------------------------------
# Real ffmpeg render with BGM (Tier 5 end-to-end)
# ---------------------------------------------------------------------------

@needs_ffmpeg
def test_real_render_with_bgm(tmp_path, monkeypatch):
    import clip_editor

    monkeypatch.setattr(clip_editor, "EDITED_DIR", tmp_path)
    bgm_dir = tmp_path / "assets" / "bgm"
    (bgm_dir / "chill").mkdir(parents=True)
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=6:size=1280x720:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src)],
        check=True, timeout=120)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=6",
         "-c:a", "libmp3lame", "-q:a", "4",
         str(bgm_dir / "chill" / "tone.mp3")],
        check=True, timeout=120)
    monkeypatch.setenv("BGM_DIR", str(bgm_dir))

    editor = clip_editor.ClipEditor()
    out = tmp_path / "out_bgm.mp4"
    result = editor.edit_clip(
        str(src), str(out), streamer="xqc", layout="blurred_backdrop",
        hook_banner="TEST BANNER", bgm=True, category="Big Wins",
        generate_thumbnail=False)
    assert result == str(out)
    assert out.exists() and out.stat().st_size > 0

    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-v", "error",
         "-show_entries", "stream=codec_type,codec_name,width,height",
         "-show_entries", "format=duration", "-of", "json", str(out)],
        capture_output=True, text=True, check=True, timeout=60)
    info = json.loads(probe.stdout)
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(video) == 1
    assert video[0]["width"] == 1080 and video[0]["height"] == 1920
    assert len(audio) == 1  # ducked speech + music mix
    assert abs(float(info["format"]["duration"]) - 6.0) < 0.5
