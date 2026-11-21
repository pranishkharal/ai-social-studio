"""Tests for the full-body tracking feature (FEATURE 3): body_tracker.py,
the TRACKING_TARGET config, the pipeline passthrough, and the dashboard
tracking-target option.

cv2 / mediapipe are NOT installed under system python3, so this file
reuses the fake cv2 from tests/test_face_tracker_mp.py (importing that
module installs it — one fake cv2 for the whole session) and injects a
fake mediapipe Tasks API via sys.modules monkeypatching — exactly the
optional-import path the tracker uses in production.

The fake mediapipe mirrors the REAL mediapipe 1.2.3 surface the tracker
uses: mp.tasks IS mediapipe.tasks.python (mp.tasks.BaseOptions,
mp.tasks.vision.PoseLandmarker / PoseLandmarkerOptions / RunningMode),
plus mp.Image / mp.ImageFormat. There is no mp.solutions in 1.2.3.
"""
import asyncio
import io
import sys
import types
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the shared fake cv2 (installs it into sys.modules on import).
from test_face_tracker_mp import FAKE_VIDEO  # noqa: E402,F401

from body_tracker import BodyTracker  # noqa: E402
import pipeline as pipeline_module  # noqa: E402
from pipeline import ClipPipeline, _resolve_tracking_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Fake mediapipe Tasks API
# --------------------------------------------------------------------------

POSE_PLAN = None  # list of poses, or callable(frame_idx) -> list of poses
# Each pose is a list of (x, y) or (x, y, visibility) tuples in
# normalized coords. The frame index comes from the fake video's
# marker byte (frame[0, 0, 0]), like the face-tracker fakes.


def _lm(point):
    x, y = point[0], point[1]
    visibility = point[2] if len(point) > 2 else 1.0
    return SimpleNamespace(x=x, y=y, visibility=visibility)


class FakeImage:
    def __init__(self, image_format=None, data=None):
        self.image_format = image_format
        self.data = data


class FakeBaseOptions:
    def __init__(self, model_asset_path=None):
        self.model_asset_path = model_asset_path


class FakePoseLandmarkerOptions:
    def __init__(self, base_options=None, running_mode=None, num_poses=1,
                 min_pose_detection_confidence=0.5, **kwargs):
        self.base_options = base_options
        self.running_mode = running_mode
        self.num_poses = num_poses
        self.min_pose_detection_confidence = min_pose_detection_confidence


class FakePoseLandmarker:
    def __init__(self, options=None):
        self.options = options

    @classmethod
    def create_from_options(cls, options):
        return cls(options)

    def detect(self, image):
        idx = int(image.data[0, 0, 0])
        plan = POSE_PLAN
        poses = plan(idx) if callable(plan) else (plan or [])
        return SimpleNamespace(
            pose_landmarks=[[_lm(pt) for pt in pose] for pose in poses])


def make_fake_mediapipe():
    mp = types.ModuleType("mediapipe")
    mp.Image = FakeImage
    mp.ImageFormat = SimpleNamespace(SRGB="SRGB")
    # NOTE: mp.tasks is mediapipe.tasks.python — NOT mp.tasks.python.
    mp.tasks = SimpleNamespace(
        BaseOptions=FakeBaseOptions,
        vision=SimpleNamespace(
            PoseLandmarker=FakePoseLandmarker,
            PoseLandmarkerOptions=FakePoseLandmarkerOptions,
            RunningMode=SimpleNamespace(IMAGE="IMAGE")))
    return mp


def _pose(x0, y0, x1, y1, visibility=1.0):
    """A rectangular 5-landmark pose (corners + centre)."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [(x0, y0, visibility), (x1, y0, visibility),
            (x0, y1, visibility), (x1, y1, visibility),
            (cx, cy, visibility)]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    global POSE_PLAN
    POSE_PLAN = None
    FAKE_VIDEO.update({"width": 1920, "height": 1080, "fps": 30.0,
                       "n_frames": 60})
    monkeypatch.delitem(sys.modules, "mediapipe", raising=False)


@pytest.fixture
def models_dir(tmp_path):
    """Models dir with a (fake, non-empty) pose model already present,
    so the tracker never tries to download anything."""
    d = tmp_path / "models"
    d.mkdir()
    (d / "pose_landmarker_lite.task").write_bytes(b"fake-pose-model")
    return d


@pytest.fixture
def empty_models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


def _tracker(models_dir, **kwargs):
    return BodyTracker(str(models_dir), **kwargs)


def _analyze(tracker):
    smoothed, rw, rh, cw = tracker.analyze_video_trajectory("fake.mp4")
    assert (rw, rh, cw) == (1920, 1080, 607)
    assert len(smoothed) == 60
    return smoothed


# --------------------------------------------------------------------------
# Body box computation (pure function, no video needed)
# --------------------------------------------------------------------------

def test_landmarks_to_box_exact_without_margin():
    # margin_ratio=0 still leaves the 10 px minimum margin floor.
    box = BodyTracker.landmarks_to_box(
        [_lm(p) for p in _pose(0.25, 0.25, 0.75, 0.75)],
        1920, 1080, margin_ratio=0.0)
    assert box == (470, 260, 980, 560)


def test_landmarks_to_box_applies_margin():
    # 960x540 box, 15% margin -> 144 px x, 81 px y.
    box = BodyTracker.landmarks_to_box(
        [_lm(p) for p in _pose(0.25, 0.25, 0.75, 0.75)], 1920, 1080)
    assert box == (336, 189, 1248, 702)


def test_landmarks_to_box_clamps_to_frame():
    box = BodyTracker.landmarks_to_box(
        [_lm(p) for p in _pose(0.0, 0.0, 0.1, 0.1)], 1920, 1080)
    assert box[0] == 0 and box[1] == 0  # margin would go negative
    box = BodyTracker.landmarks_to_box(
        [_lm(p) for p in _pose(0.9, 0.9, 1.0, 1.0)], 1920, 1080)
    assert box[0] + box[2] == 1920 and box[1] + box[3] == 1080


def test_landmarks_to_box_empty_is_none():
    assert BodyTracker.landmarks_to_box([], 1920, 1080) is None
    assert BodyTracker.landmarks_to_box(None, 1920, 1080) is None


def test_landmarks_to_box_ignores_low_visibility_landmarks():
    # Four visible landmarks span x 0.4-0.6; one low-visibility landmark
    # sits far away at (0.95, 0.95) and must NOT stretch the box.
    pose = _pose(0.4, 0.2, 0.6, 0.8) + [(0.95, 0.95, 0.1)]
    box = BodyTracker.landmarks_to_box([_lm(p) for p in pose], 1920, 1080)
    # Visible span 768-1152 px, 15% margin (57.6 px) -> x 710, w 499.
    assert box[0] == 710 and box[2] == 499


def test_landmarks_to_box_uses_all_when_too_few_visible():
    # Only 2 of 5 landmarks are visible (< 4), so all 5 are used and
    # the box spans the full 0.1-0.9 range instead of giving up.
    pose = [(0.4, 0.4, 0.9), (0.6, 0.6, 0.9),
            (0.1, 0.1, 0.1), (0.9, 0.9, 0.1), (0.5, 0.5, 0.1)]
    box = BodyTracker.landmarks_to_box([_lm(p) for p in pose], 1920, 1080)
    assert box[0] == 0 and box[2] == 1920  # clamped to the full width


# --------------------------------------------------------------------------
# Backend selection / fallback signals
# --------------------------------------------------------------------------

def test_no_mediapipe_falls_back_cleanly(models_dir):
    tracker = _tracker(models_dir)
    assert tracker.backend == "unavailable"
    smoothed = _analyze(tracker)  # must not raise
    assert smoothed == [656] * 60  # centred default
    assert tracker.body_found is False
    assert tracker.detection_rate == 0.0


def test_close_is_safe_without_backend(models_dir, monkeypatch):
    tracker = _tracker(models_dir)
    tracker.close()  # never initialized — must not raise
    tracker.close()  # twice — must not raise
    assert tracker.backend == "unavailable"


def test_close_releases_landmarker(models_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"
    assert tracker._landmarker is not None
    tracker.close()
    assert tracker._landmarker is None


def test_old_mediapipe_without_tasks_api_falls_back(models_dir, monkeypatch):
    mp = types.ModuleType("mediapipe")  # no .tasks at all
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    assert _tracker(models_dir).backend == "unavailable"


def test_mediapipe_init_failure_falls_back(models_dir, monkeypatch):
    mp = make_fake_mediapipe()

    class BrokenLandmarker(FakePoseLandmarker):
        @classmethod
        def create_from_options(cls, options):
            raise RuntimeError("no model")

    mp.tasks.vision.PoseLandmarker = BrokenLandmarker
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    assert _tracker(models_dir).backend == "unavailable"


def test_existing_model_is_not_redownloaded(models_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())

    def _boom(*a, **k):
        raise AssertionError("download must not run when the model exists")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert _tracker(models_dir).backend == "mediapipe"


def test_model_download_failure_falls_back_cleanly(empty_models_dir,
                                                   monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())

    def _fail(*a, **k):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr("urllib.request.urlopen", _fail)
    tracker = _tracker(empty_models_dir)
    assert tracker.backend == "unavailable"
    smoothed = _analyze(tracker)
    assert smoothed == [656] * 60
    assert tracker.body_found is False
    # No partial file left behind.
    assert not (empty_models_dir / "pose_landmarker_lite.task").exists()


def test_model_download_success_writes_file(empty_models_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: io.BytesIO(b"downloaded-model"))
    tracker = _tracker(empty_models_dir)
    assert tracker.backend == "mediapipe"
    assert (empty_models_dir / "pose_landmarker_lite.task").read_bytes() == \
        b"downloaded-model"


def test_pose_runtime_failure_falls_back_mid_video(models_dir, monkeypatch):
    mp = make_fake_mediapipe()

    class ExplodingLandmarker(FakePoseLandmarker):
        def detect(self, image):
            raise RuntimeError("boom")

    mp.tasks.vision.PoseLandmarker = ExplodingLandmarker
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"
    smoothed = _analyze(tracker)  # must not raise
    assert smoothed == [656] * 60
    assert tracker.body_found is False
    assert tracker.backend == "unavailable"


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------

def test_body_tracks_centres_crop_on_body(monkeypatch, models_dir):
    global POSE_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    # Landmarks x 0.5-0.6 (960-1152 px), 15% margin -> box x=931, w=249,
    # centre 1055 -> crop_x = 1055 - 303 = 752.
    POSE_PLAN = [_pose(0.5, 0.2, 0.6, 0.8)]
    tracker = _tracker(models_dir)
    smoothed = _analyze(tracker)
    assert smoothed == [752] * 60
    assert tracker.body_found is True
    assert tracker.detection_rate == 1.0
    assert tracker.last_body_box == (931, 118, 249, 843)


def test_no_bodies_gives_fallback_signal(monkeypatch, models_dir):
    global POSE_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    POSE_PLAN = []  # Pose runs fine, finds nobody
    tracker = _tracker(models_dir)
    smoothed = _analyze(tracker)
    assert smoothed == [656] * 60
    assert tracker.body_found is False
    assert tracker.detection_rate == 0.0
    assert tracker.last_body_box is None


def test_largest_body_wins_with_multiple_people(monkeypatch, models_dir):
    global POSE_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    small_left = _pose(0.05, 0.2, 0.10, 0.8)   # narrow box, left
    large_right = _pose(0.6, 0.1, 0.8, 0.9)    # wide box, right
    POSE_PLAN = [small_left, large_right]
    tracker = _tracker(models_dir)
    smoothed = _analyze(tracker)
    # Large box: x 1152-1536 px, 15% margin -> x=1094, w=499,
    # centre 1343 -> crop_x = 1343 - 303 = 1040. (The small box would
    # give crop_x 0, so this proves the largest box won.)
    assert smoothed == [1040] * 60
    assert tracker.body_found is True


def test_body_holds_last_target_when_body_temporarily_lost(monkeypatch,
                                                           models_dir):
    global POSE_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())

    def plan(frame_idx):
        # Body present for the first half of the video, then gone.
        return [_pose(0.5, 0.2, 0.6, 0.8)] if frame_idx < 30 else []

    POSE_PLAN = plan
    tracker = _tracker(models_dir)
    smoothed = _analyze(tracker)
    assert tracker.body_found is True  # found at least once
    assert 0.4 < tracker.detection_rate < 0.6
    # Camera stays on the last known body position, never snaps centre.
    assert smoothed == [752] * 60


def test_deadzone_holds_camera_still_on_small_body_jitter(monkeypatch,
                                                          models_dir):
    global POSE_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())

    def plan(frame_idx):
        # Body centre jitters between 940 and 1000 px every sample
        # (targets 636 / 696) — inside the 15%-of-crop deadzone (~91 px).
        sample = frame_idx // 5
        centre = 940 if sample % 2 == 0 else 1000
        return [_pose((centre - 96) / 1920, 0.2, (centre + 96) / 1920, 0.8)]

    POSE_PLAN = plan
    smoothed = _analyze(_tracker(models_dir))
    assert len(set(smoothed)) == 1  # no jitter at all


# --------------------------------------------------------------------------
# get_ffmpeg_crop_filter (same behaviour as SpeakerTracker)
# --------------------------------------------------------------------------

def test_crop_filter_static_camera(models_dir):
    tracker = _tracker(models_dir)
    assert tracker.get_ffmpeg_crop_filter([656] * 90, 30.0, 607, 1080) == \
        "crop=607:1080:656:0,scale=1080:1920"


def test_crop_filter_moving_camera_uses_piecewise_expression(models_dir):
    tracker = _tracker(models_dir)
    filt = tracker.get_ffmpeg_crop_filter(list(range(0, 900, 10)), 30.0, 607, 1080)
    assert filt.startswith("crop=607:1080:'if(lt(t,")
    assert filt.endswith(":0,scale=1080:1920")


def test_crop_filter_empty_list_centres(models_dir):
    tracker = _tracker(models_dir)
    assert tracker.get_ffmpeg_crop_filter([], 30.0, 607, 1080) == \
        "crop=607:1080:(iw-607)/2:0"


# --------------------------------------------------------------------------
# Config + pipeline passthrough (mocked editor)
# --------------------------------------------------------------------------

def test_config_tracking_target_defaults_to_face():
    import config
    assert config.TRACKING_TARGET == "face"


def test_resolve_tracking_target():
    assert _resolve_tracking_target("body") == "body"
    assert _resolve_tracking_target("BODY") == "body"
    assert _resolve_tracking_target("face") == "face"
    assert _resolve_tracking_target("bogus") == "face"
    assert _resolve_tracking_target("") == "face"


def test_resolve_tracking_target_none_uses_config(monkeypatch):
    monkeypatch.setattr(pipeline_module, "TRACKING_TARGET", "body")
    assert _resolve_tracking_target(None) == "body"
    monkeypatch.setattr(pipeline_module, "TRACKING_TARGET", "face")
    assert _resolve_tracking_target(None) == "face"


@pytest.fixture
def pipeline_setup(tmp_path, monkeypatch, fresh_db):
    """ClipPipeline with a mocked editor + AI, like
    tests/test_tiktok_pipeline.py::pipeline_setup."""
    monkeypatch.setattr(pipeline_module, "SUBTITLES_DIR", tmp_path / "subtitles")
    monkeypatch.setattr(pipeline_module, "EDITED_DIR", tmp_path / "edited")
    monkeypatch.setattr(pipeline_module, "CLIPS_DIR", tmp_path / "clips")
    monkeypatch.setattr(pipeline_module, "SUBTITLES_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "SUBTITLE_STYLE", "classic")
    monkeypatch.setattr(pipeline_module, "BGM_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "BROLL_ENABLED", False)
    monkeypatch.setattr(pipeline_module, "TRACKING_TARGET", "face")
    (tmp_path / "subtitles").mkdir()
    (tmp_path / "edited").mkdir()
    (tmp_path / "clips").mkdir()

    pipe = ClipPipeline()
    monkeypatch.setattr(pipe, "_mark", lambda *a, **k: None)
    monkeypatch.setattr(
        pipe.detector, "evaluate_clip",
        lambda path, **kwargs: {"hype_score": 80.0, "trigger_reason": "test"})
    monkeypatch.setattr(
        pipe.ai, "transcribe_clip",
        lambda path, **kwargs: {"full_text": "hello world", "words": []})
    monkeypatch.setattr(
        pipe.ai, "analyze_story_and_generate_metadata",
        lambda **kw: {"category": "Funny Moments", "ai_score": 85,
                      "selection_reason": "test", "story_summary": "s",
                      "viral_title": "T", "hook_banner": "H",
                      "tiktok_caption": "c", "youtube_title": "Y",
                      "youtube_description": "D", "instagram_caption": "I",
                      "blue_comment": "B", "hashtags": "#x"})

    captured = {}

    def fake_edit_clip(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "edited" / "out.mp4")

    monkeypatch.setattr(pipe.editor, "edit_clip", fake_edit_clip)

    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"fake-video")
    return pipe, raw, tmp_path, captured


def test_process_moment_passes_tracking_target_body(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    result = pipe.process_moment("adinross", duration=10,
                                 raw_clip_path=str(raw),
                                 tracking_target="body")
    assert result["status"] == "ready_to_upload"
    assert captured["tracking_target"] == "body"


def test_process_moment_tracking_target_defaults_to_face(pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw))
    assert captured["tracking_target"] == "face"


def test_process_moment_tracking_target_invalid_falls_back_to_face(
        pipeline_setup):
    pipe, raw, tmp_path, captured = pipeline_setup
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw),
                        tracking_target="bogus")
    assert captured["tracking_target"] == "face"


def test_process_moment_tracking_target_uses_config_when_none(
        pipeline_setup, monkeypatch):
    pipe, raw, tmp_path, captured = pipeline_setup
    monkeypatch.setattr(pipeline_module, "TRACKING_TARGET", "body")
    pipe.process_moment("adinross", duration=10, raw_clip_path=str(raw))
    assert captured["tracking_target"] == "body"


def test_resume_clip_passes_tracking_target(pipeline_setup, monkeypatch):
    from database import add_clip, update_clip_status
    from retry_utils import CheckpointStore

    pipe, raw, tmp_path, captured = pipeline_setup
    clip_id = add_clip({
        "streamer": "xqc", "title": "Moment from xqc",
        "story_summary": "Analyzing...", "hook_banner": "WAIT FOR IT...",
        "caption": "", "tags": "", "hype_score": 80.0,
        "trigger_reason": "test", "duration": 25.0,
        "raw_path": str(raw), "edited_path": None, "status": "failed",
    })
    update_clip_status(clip_id, "failed")
    store = CheckpointStore(tmp_path / "checkpoints")
    store.mark_step_done(clip_id, "transcribed",
                         {"transcript": "hello world", "words": []})
    store.mark_step_done(clip_id, "analyzed", {
        "category": "Big Wins", "ai_score": 92, "selection_reason": "test",
        "story_summary": "s", "viral_title": "T", "hook_banner": "H",
        "tiktok_caption": "c", "youtube_title": "Y",
        "youtube_description": "D", "instagram_caption": "I",
        "blue_comment": "B", "hashtags": "#x"})
    pipe.checkpoints = store

    result = pipe.resume_clip(clip_id, tracking_target="body")
    assert result["status"] == "ready_to_upload"
    assert captured["tracking_target"] == "body"

    captured.clear()
    result = pipe.resume_clip(clip_id)
    assert result["status"] == "ready_to_upload"
    assert captured["tracking_target"] == "face"


def test_rerender_clip_passes_tracking_target(pipeline_setup):
    from database import add_clip

    pipe, raw, tmp_path, captured = pipeline_setup
    clip_id = add_clip({
        "streamer": "xqc", "title": "Moment from xqc",
        "story_summary": "s", "hook_banner": "H", "caption": "",
        "tags": "", "hype_score": 80.0, "trigger_reason": "test",
        "duration": 25.0, "raw_path": str(raw),
        "edited_path": None, "status": "ready_to_upload",
    })
    result = pipe.rerender_clip(clip_id, tracking_target="body")
    assert result["status"] == "success"
    assert captured["tracking_target"] == "body"

    captured.clear()
    result = pipe.rerender_clip(clip_id)
    assert result["status"] == "success"
    assert captured["tracking_target"] == "face"


# --------------------------------------------------------------------------
# Dashboard request models + endpoints
# --------------------------------------------------------------------------

def _import_web_dashboard():
    """Import web_dashboard, upgrading the conftest fastapi stubs first:
    the stub StaticFiles takes no kwargs and the stub FastAPI lacks
    delete/put/patch/api_route, which the dashboard uses at import time.
    Additive only — no other test module is affected."""
    fastapi_stub = sys.modules["fastapi"]
    staticfiles = types.ModuleType("fastapi.staticfiles")

    class StaticFiles:
        def __init__(self, directory=None, **kwargs):
            self.directory = directory

    staticfiles.StaticFiles = StaticFiles
    sys.modules["fastapi.staticfiles"] = staticfiles
    for method in ("delete", "put", "patch"):
        if not hasattr(fastapi_stub.FastAPI, method):
            setattr(fastapi_stub.FastAPI, method, fastapi_stub.FastAPI.get)
    if not hasattr(fastapi_stub.FastAPI, "api_route"):
        def _api_route(self, path, methods=None, **kwargs):
            return self._decorator(methods[0] if methods else "GET", path)

        fastapi_stub.FastAPI.api_route = _api_route
    import web_dashboard
    return web_dashboard


class _SyncBackgroundTasks:
    """BackgroundTasks stand-in that runs the task immediately."""

    def add_task(self, func, *args, **kwargs):
        func(*args, **kwargs)


def test_clip_trigger_req_tracking_target_defaults_to_face():
    wd = _import_web_dashboard()
    assert wd.ClipTriggerReq(streamer="xqc").tracking_target == "face"
    assert wd.ClipTriggerReq(streamer="xqc",
                             tracking_target="body").tracking_target == "body"


def test_rerender_req_tracking_target_defaults_to_face():
    wd = _import_web_dashboard()
    assert wd.RerenderReq().tracking_target == "face"
    assert wd.RerenderReq(tracking_target="body").tracking_target == "body"


def test_trigger_endpoint_passes_tracking_target_to_pipeline(monkeypatch):
    wd = _import_web_dashboard()
    calls = {}
    monkeypatch.setattr(wd.pipeline, "process_moment",
                        lambda **kw: calls.update(kw))
    req = wd.ClipTriggerReq(streamer="xqc", duration=10,
                            tracking_target="body")
    asyncio.run(wd.trigger_clip(req, _SyncBackgroundTasks()))
    assert calls["tracking_target"] == "body"
    assert calls["streamer"] == "xqc"


def test_rerender_endpoint_passes_tracking_target_to_pipeline(monkeypatch):
    wd = _import_web_dashboard()
    calls = {}
    monkeypatch.setattr(wd.pipeline, "rerender_clip",
                        lambda **kw: calls.update(kw))
    req = wd.RerenderReq(tracking_target="body")
    asyncio.run(wd.rerender_clip_endpoint(1, req, _SyncBackgroundTasks()))
    assert calls["tracking_target"] == "body"
    assert calls["clip_id"] == 1


# --------------------------------------------------------------------------
# Dashboard template wiring
# --------------------------------------------------------------------------

def test_template_has_tracking_target_select():
    html = (REPO_ROOT / "templates" / "index.html").read_text()
    assert 'id="tracking-target-select"' in html
    assert '<option value="face" selected>' in html
    assert '<option value="body">' in html
    # The trigger + re-render JS both send it to the API.
    assert html.count("tracking_target: trackingTarget") == 2
