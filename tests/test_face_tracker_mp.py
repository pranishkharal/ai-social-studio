"""Tests for the upgraded speaker_tracker.py.

cv2 / mediapipe / ultralytics are NOT installed in this VM, so cv2 is stubbed
with a fake VideoCapture + fake cascades, and mediapipe / ultralytics are
injected as fake modules via sys.modules monkeypatching — exactly the
optional-import paths the tracker uses in production.

The fake mediapipe mirrors the real Tasks API (mediapipe >= 1.0):
``mp.tasks.vision.FaceDetector`` built from ``FaceDetectorOptions`` with a
``BaseOptions(model_asset_path=...)``, ``mp.Image`` wrapping the frame, and
pixel ``bounding_box`` results. A legacy ``mp.solutions`` fake is kept for
the old-mediapipe fallback path.
"""
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --------------------------------------------------------------------------
# Fake cv2 (installed before importing speaker_tracker)
# --------------------------------------------------------------------------

FAKE_VIDEO = {"width": 1920, "height": 1080, "fps": 30.0, "n_frames": 60}
HAAR_PLAN = None  # None -> no faces; list -> static boxes; callable(call#) -> boxes


class FakeVideoCapture:
    def __init__(self, path):
        self._idx = 0

    def isOpened(self):
        return True

    def read(self):
        if self._idx >= FAKE_VIDEO["n_frames"]:
            return False, None
        frame = np.zeros((FAKE_VIDEO["height"], FAKE_VIDEO["width"], 3), dtype=np.uint8)
        frame[0, 0, 0] = self._idx % 256  # marker so fakes know the frame index
        self._idx += 1
        return True, frame

    def get(self, prop):
        return {
            FakeCV2.CAP_PROP_FRAME_WIDTH: FAKE_VIDEO["width"],
            FakeCV2.CAP_PROP_FRAME_HEIGHT: FAKE_VIDEO["height"],
            FakeCV2.CAP_PROP_FRAME_COUNT: FAKE_VIDEO["n_frames"],
            FakeCV2.CAP_PROP_FPS: FAKE_VIDEO["fps"],
        }[prop]

    def release(self):
        pass


class FakeCascadeClassifier:
    def __init__(self, path):
        self._calls = 0

    def detectMultiScale(self, *args, **kwargs):
        plan = HAAR_PLAN
        boxes = plan(self._calls) if callable(plan) else (plan or [])
        self._calls += 1
        return np.array(boxes, dtype=np.float64).reshape(-1, 4)


class FakeCV2(types.ModuleType):
    # Enough of the real cv2 surface that other test modules importing
    # cv2 (e.g. the repo's test_face_tracker.py smoke test) do not crash
    # on attribute access. haarcascades points nowhere real on purpose:
    # there are no real OpenCV cascades in this VM.
    __version__ = "0.0.0-fake"
    data = SimpleNamespace(haarcascades="/nonexistent/haarcascades/")
    COLOR_BGR2GRAY = 6
    COLOR_BGR2RGB = 4
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FPS = 5
    VideoCapture = FakeVideoCapture
    CascadeClassifier = FakeCascadeClassifier

    @staticmethod
    def cvtColor(frame, code):
        if code == FakeCV2.COLOR_BGR2GRAY:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        return frame

    @staticmethod
    def resize(img, dsize, fx=1.0, fy=1.0):
        h, w = img.shape[:2]
        return np.zeros((int(h * fy), int(w * fx)), dtype=np.uint8)


sys.modules.setdefault("cv2", FakeCV2("cv2"))

import speaker_tracker  # noqa: E402
from speaker_tracker import SpeakerTracker  # noqa: E402

# The real model-download function, saved before the autouse fixture
# replaces it with a no-network stub (the download tests restore it).
_REAL_ENSURE_BLAZEFACE = speaker_tracker.ensure_blazeface_model

# --------------------------------------------------------------------------
# Fake mediapipe / ultralytics
# --------------------------------------------------------------------------

MP_PLAN = None    # list of pixel (x, y, w, h) boxes, or callable(frame_idx)
LEGACY_MP_PLAN = None  # list of relative (xmin, ymin, w, h) boxes, or callable(frame_idx)
YOLO_PLAN = None  # (xyxy list, conf list), or callable(frame_idx)


# --- Fake mediapipe Tasks API (mediapipe >= 1.0) ---------------------------

class FakeBoundingBox:
    """Stand-in for the Tasks API BoundingBox (pixel origin + size)."""

    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height


class FakeTasksDetection:
    def __init__(self, box):
        self.bounding_box = FakeBoundingBox(*box)


class FakeImage:
    """Stand-in for mp.Image: keeps the numpy frame reachable as .data."""

    def __init__(self, image_format=None, data=None):
        self.image_format = image_format
        self.data = data


class FakeBaseOptions:
    def __init__(self, model_asset_path=None, model_asset_buffer=None,
                 delegate=None):
        self.model_asset_path = model_asset_path


class FakeFaceDetectorOptions:
    def __init__(self, base_options=None, model_asset_path=None,
                 min_detection_confidence=0.5, running_mode=None, **kwargs):
        self.base_options = base_options
        self.model_asset_path = model_asset_path
        self.min_detection_confidence = min_detection_confidence
        self.running_mode = running_mode


class FakeFaceDetector:
    """Stand-in for mp.tasks.vision.FaceDetector (IMAGE mode)."""

    created_options = None  # last options object, for assertions
    closed = 0              # how many detectors were closed

    def __init__(self, options=None):
        self.options = options

    @classmethod
    def create_from_options(cls, options):
        cls.created_options = options
        return cls(options)

    def detect(self, image):
        arr = getattr(image, "data", image)
        idx = int(arr[0, 0, 0])
        plan = MP_PLAN
        boxes = plan(idx) if callable(plan) else (plan or [])
        return SimpleNamespace(detections=[FakeTasksDetection(b) for b in boxes])

    def close(self):
        type(self).closed += 1


def make_fake_mediapipe():
    mp = types.ModuleType("mediapipe")
    mp.Image = FakeImage
    mp.ImageFormat = SimpleNamespace(SRGB="SRGB")
    mp.tasks = SimpleNamespace(
        BaseOptions=FakeBaseOptions,
        vision=SimpleNamespace(
            FaceDetector=FakeFaceDetector,
            FaceDetectorOptions=FakeFaceDetectorOptions,
            RunningMode=SimpleNamespace(IMAGE="IMAGE"),
        ),
    )
    return mp


# --- Fake mediapipe legacy API (mp.solutions, mediapipe < 1.0) --------------

def _mp_detection(xmin, ymin, w, h):
    return SimpleNamespace(
        location_data=SimpleNamespace(
            relative_bounding_box=SimpleNamespace(
                xmin=xmin, ymin=ymin, width=w, height=h)))


class FakeLegacyFaceDetection:
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        pass

    def process(self, image):
        idx = int(image[0, 0, 0])
        plan = LEGACY_MP_PLAN
        boxes = plan(idx) if callable(plan) else (plan or [])
        return SimpleNamespace(detections=[_mp_detection(*b) for b in boxes])


def make_fake_mediapipe_legacy():
    mp = types.ModuleType("mediapipe")
    mp.solutions = SimpleNamespace(
        face_detection=SimpleNamespace(FaceDetection=FakeLegacyFaceDetection))
    return mp


class FakeBoxes:
    def __init__(self, xyxy, conf):
        self._xyxy = xyxy
        self._conf = conf

    def __len__(self):
        return len(self._xyxy)

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def conf(self):
        return self._conf


class FakeYOLO:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, frame, verbose=False):
        idx = int(frame[0, 0, 0])
        plan = YOLO_PLAN
        boxes, confs = plan(idx) if callable(plan) else (plan or ([], []))
        return [SimpleNamespace(boxes=FakeBoxes(boxes, confs))]


def make_fake_ultralytics():
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    return mod


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    global HAAR_PLAN, MP_PLAN, LEGACY_MP_PLAN, YOLO_PLAN
    HAAR_PLAN = None
    MP_PLAN = None
    LEGACY_MP_PLAN = None
    YOLO_PLAN = None
    FakeFaceDetector.created_options = None
    FakeFaceDetector.closed = 0
    FAKE_VIDEO.update({"width": 1920, "height": 1080, "fps": 30.0, "n_frames": 60})
    monkeypatch.delenv("FACE_TRACKER_BACKEND", raising=False)
    monkeypatch.delenv("YOLO_FACE_MODEL", raising=False)
    # Block the REAL optional imports (None in sys.modules makes the import
    # raise ImportError) so these tests behave the same whether or not
    # mediapipe / ultralytics happen to be installed in the test env.
    for name in ("mediapipe", "ultralytics"):
        monkeypatch.setitem(sys.modules, name, None)
    # Never touch the network in tests: pretend the BlazeFace model is
    # already in models/. The download tests restore the real function.
    monkeypatch.setattr(
        speaker_tracker, "ensure_blazeface_model",
        lambda d: Path(d) / "blaze_face_short_range.tflite")


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "haarcascade_frontalface_default.xml").write_text("<cascade/>")
    (d / "haarcascade_profileface.xml").write_text("<cascade/>")
    return d


def _tracker(models_dir, **kwargs):
    return SpeakerTracker(str(models_dir), **kwargs)


def _analyze(tracker):
    smoothed, rw, rh, cw = tracker.analyze_video_trajectory("fake.mp4")
    assert (rw, rh, cw) == (1920, 1080, 607)
    assert len(smoothed) == 60
    return smoothed


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------

def test_no_optional_deps_falls_back_to_haar(models_dir):
    tracker = _tracker(models_dir)
    assert tracker.backend == "haar"


def test_mediapipe_preferred_when_available(monkeypatch, models_dir):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"


def test_env_haar_forces_haar_even_with_mediapipe(monkeypatch, models_dir):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    monkeypatch.setenv("FACE_TRACKER_BACKEND", "haar")
    assert _tracker(models_dir).backend == "haar"


def test_env_yolo_uses_yolo_when_available(monkeypatch, models_dir):
    monkeypatch.setitem(sys.modules, "ultralytics", make_fake_ultralytics())
    monkeypatch.setenv("FACE_TRACKER_BACKEND", "yolo")
    assert _tracker(models_dir).backend == "yolo"


def test_env_yolo_falls_back_to_haar_when_missing(monkeypatch, models_dir):
    monkeypatch.setenv("FACE_TRACKER_BACKEND", "yolo")
    assert _tracker(models_dir).backend == "haar"


def test_unknown_backend_env_falls_back_to_haar(monkeypatch, models_dir):
    monkeypatch.setenv("FACE_TRACKER_BACKEND", "bogus")
    assert _tracker(models_dir).backend == "haar"


def test_mediapipe_init_failure_falls_back_to_haar(monkeypatch, models_dir):
    mp = make_fake_mediapipe()

    class BrokenFaceDetector:
        @classmethod
        def create_from_options(cls, options):
            raise RuntimeError("no model")

    mp.tasks.vision.FaceDetector = BrokenFaceDetector
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    assert _tracker(models_dir).backend == "haar"


def test_mediapipe_model_unavailable_falls_back_to_haar(monkeypatch, models_dir):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    monkeypatch.setattr(speaker_tracker, "ensure_blazeface_model",
                        lambda d: None)
    assert _tracker(models_dir).backend == "haar"


# --------------------------------------------------------------------------
# Haar path (must behave exactly as before the upgrade)
# --------------------------------------------------------------------------

def test_haar_no_faces_stays_centred(models_dir):
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed == [656] * 60  # (1920 - 607) // 2


def test_haar_single_face_centres_crop_on_it(models_dir):
    global HAAR_PLAN
    # Small-coord box (400,100,100,100) -> full-res (800,200,200,200),
    # centre x=900 -> crop_x = 900 - 607//2 = 597.
    HAAR_PLAN = [(400, 100, 100, 100)]
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed == [597] * 60


def test_haar_ema_glides_instead_of_snapping(models_dir):
    global HAAR_PLAN

    def plan(call):
        # No face for the first 6 samples (frames 0-25), then a face.
        return [] if call < 6 else [(400, 100, 100, 100)]

    HAAR_PLAN = plan
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed[0] == 656
    assert smoothed[29] > 640  # EMA has barely moved yet (lag is the old behaviour)
    assert 597 <= smoothed[-1] <= 615  # gliding towards 597
    assert smoothed[30] > smoothed[-1]  # monotonic approach


# --------------------------------------------------------------------------
# MediaPipe path
# --------------------------------------------------------------------------

def test_mediapipe_tracks_face(monkeypatch, models_dir):
    global MP_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    # Pixel box x=1152, w=192 -> centre 1248 -> crop_x = 1248 - 303 = 945.
    MP_PLAN = [(1152, 216, 192, 162)]
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed == [945] * 60


def test_mediapipe_detector_uses_model_and_image_mode(monkeypatch, models_dir):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"
    options = FakeFaceDetector.created_options
    assert options is not None
    assert options.base_options.model_asset_path == str(
        models_dir / "blaze_face_short_range.tflite")
    assert options.min_detection_confidence == 0.5
    assert options.running_mode == "IMAGE"


def test_mediapipe_close_releases_detector(monkeypatch, models_dir):
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"
    tracker.close()
    assert FakeFaceDetector.closed == 1
    assert tracker._mp_detector is None
    tracker.close()  # second close is a no-op, not an error
    assert FakeFaceDetector.closed == 1


def test_close_is_noop_without_mediapipe(models_dir):
    tracker = _tracker(models_dir)
    assert tracker.backend == "haar"
    tracker.close()  # must not raise


def test_mediapipe_deadzone_holds_camera_still(monkeypatch, models_dir):
    global MP_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())

    def plan(frame_idx):
        # Face centre jitters between 940 and 1000 px every sample
        # (targets 637 / 697) — inside the 15%-of-crop deadzone (~91 px).
        sample = frame_idx // 5
        centre = 940 if sample % 2 == 0 else 1000
        return [(centre - 96, 216, 192, 162)]

    MP_PLAN = plan
    smoothed = _analyze(_tracker(models_dir))
    assert len(set(smoothed)) == 1  # no jitter at all


def test_mediapipe_snaps_on_large_jump(monkeypatch, models_dir):
    global MP_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())

    def plan(frame_idx):
        centre = 500 if frame_idx < 30 else 1500
        return [(centre - 96, 216, 192, 162)]

    MP_PLAN = plan
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed[0] == 197    # 500 - 303
    assert smoothed[29] == 997   # snapping along the ramp, not lagging
    assert smoothed[30] == 1197  # 1500 - 303
    assert smoothed[-1] == 1197


def test_mediapipe_runtime_failure_falls_back_to_haar_mid_video(monkeypatch, models_dir):
    mp = make_fake_mediapipe()

    class ExplodingFaceDetector(FakeFaceDetector):
        def detect(self, image):
            raise RuntimeError("boom")

    mp.tasks.vision.FaceDetector = ExplodingFaceDetector
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"
    smoothed = _analyze(tracker)  # must not raise
    assert smoothed == [656] * 60  # Haar found no faces -> centred
    assert tracker.backend == "haar"


def test_legacy_solutions_mediapipe_still_works(monkeypatch, models_dir):
    global LEGACY_MP_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe_legacy())
    # Relative box xmin=0.6, w=0.1 -> x=1152, w=192, centre 1248
    # -> crop_x = 1248 - 303 = 945.
    LEGACY_MP_PLAN = [(0.6, 0.2, 0.1, 0.15)]
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"
    smoothed = _analyze(tracker)
    assert smoothed == [945] * 60


# --------------------------------------------------------------------------
# BlazeFace model auto-download (real ensure function, fake _download_file)
# --------------------------------------------------------------------------

@pytest.fixture
def real_ensure(monkeypatch):
    """Restore the real ensure_blazeface_model (the autouse fixture stubs
    it) and cut retries to 1 so failure tests do not sleep."""
    monkeypatch.setattr(speaker_tracker, "ensure_blazeface_model",
                        _REAL_ENSURE_BLAZEFACE)
    monkeypatch.setattr(speaker_tracker, "MEDIAPIPE_MODEL_DOWNLOAD_RETRIES", 1)
    return speaker_tracker.ensure_blazeface_model


def test_blazeface_model_download_success(monkeypatch, models_dir, real_ensure):
    def fake_download(url, dest, timeout=30):
        assert url == speaker_tracker.MEDIAPIPE_MODEL_URL
        Path(dest).write_bytes(b"x" * 200_000)

    monkeypatch.setattr(speaker_tracker, "_download_file", fake_download)
    path = real_ensure(models_dir)
    assert path == models_dir / "blaze_face_short_range.tflite"
    assert path.stat().st_size == 200_000
    # No temp file left behind.
    assert not (models_dir / "blaze_face_short_range.tflite.download").exists()


def test_blazeface_model_existing_file_reused(monkeypatch, models_dir, real_ensure):
    model = models_dir / "blaze_face_short_range.tflite"
    model.write_bytes(b"x" * 200_000)
    calls = []
    monkeypatch.setattr(speaker_tracker, "_download_file",
                        lambda *a, **k: calls.append(1))
    assert real_ensure(models_dir) == model
    assert calls == []  # no download attempted


def test_blazeface_model_download_failure_returns_none(monkeypatch, models_dir, real_ensure):
    def failing_download(url, dest, timeout=30):
        raise ConnectionError("no network")

    monkeypatch.setattr(speaker_tracker, "_download_file", failing_download)
    assert real_ensure(models_dir) is None
    assert not (models_dir / "blaze_face_short_range.tflite").exists()
    assert not (models_dir / "blaze_face_short_range.tflite.download").exists()


def test_blazeface_model_bad_download_rejected(monkeypatch, models_dir, real_ensure):
    def tiny_download(url, dest, timeout=30):
        Path(dest).write_bytes(b"tiny")  # far below the 50 KB minimum

    monkeypatch.setattr(speaker_tracker, "_download_file", tiny_download)
    assert real_ensure(models_dir) is None
    assert not (models_dir / "blaze_face_short_range.tflite").exists()


def test_blazeface_model_truncated_existing_file_redownloaded(monkeypatch, models_dir, real_ensure):
    # A truncated file from a previous failed run must not be trusted.
    (models_dir / "blaze_face_short_range.tflite").write_bytes(b"x" * 100)

    def good_download(url, dest, timeout=30):
        Path(dest).write_bytes(b"x" * 200_000)

    monkeypatch.setattr(speaker_tracker, "_download_file", good_download)
    path = real_ensure(models_dir)
    assert path.stat().st_size == 200_000


# --------------------------------------------------------------------------
# YOLO path
# --------------------------------------------------------------------------

def test_yolo_tracks_face(monkeypatch, models_dir):
    global YOLO_PLAN
    monkeypatch.setitem(sys.modules, "ultralytics", make_fake_ultralytics())
    monkeypatch.setenv("FACE_TRACKER_BACKEND", "yolo")
    # Box (1000,200,1200,400) -> centre 1100 -> crop_x = 1100 - 303 = 797.
    YOLO_PLAN = ([[1000, 200, 1200, 400]], [0.9])
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed == [797] * 60


def test_yolo_low_confidence_boxes_ignored(monkeypatch, models_dir):
    global YOLO_PLAN
    monkeypatch.setitem(sys.modules, "ultralytics", make_fake_ultralytics())
    monkeypatch.setenv("FACE_TRACKER_BACKEND", "yolo")
    YOLO_PLAN = ([[1000, 200, 1200, 400]], [0.3])  # below 0.5 threshold
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed == [656] * 60


# --------------------------------------------------------------------------
# get_ffmpeg_crop_filter (unchanged API)
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
# Adaptive backend fallback (low detection rate -> Haar re-run)
# --------------------------------------------------------------------------

def test_low_detection_rate_falls_back_to_haar(monkeypatch, models_dir):
    global MP_PLAN, HAAR_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    # MediaPipe sees a face on the first sample only (1/12 = 8% < 20%).
    # Pixel box x=1152, w=192 -> centre 1248 -> crop_x 945.
    MP_PLAN = lambda frame_idx: [(1152, 216, 192, 162)] if frame_idx < 5 else []
    # Haar sees a face on every sample: centre x=900 -> crop_x 597.
    HAAR_PLAN = [(400, 100, 100, 100)]
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"
    smoothed = _analyze(tracker)
    assert tracker.backend == "haar"
    assert smoothed == [597] * 60


def test_fallback_disabled_keeps_mediapipe(monkeypatch, models_dir):
    global MP_PLAN, HAAR_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    MP_PLAN = lambda frame_idx: [(1152, 216, 192, 162)] if frame_idx < 5 else []
    HAAR_PLAN = [(400, 100, 100, 100)]
    tracker = _tracker(models_dir)
    smoothed, rw, rh, cw = tracker.analyze_video_trajectory(
        "fake.mp4", fallback_to_haar=False)
    assert (rw, rh, cw) == (1920, 1080, 607)
    assert tracker.backend == "mediapipe"
    # MediaPipe target 945 (centre 1248 - 303), held for all samples.
    assert smoothed == [945] * 60


def test_high_detection_rate_keeps_mediapipe(monkeypatch, models_dir):
    global MP_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())
    MP_PLAN = [(1152, 216, 192, 162)]  # every sample
    tracker = _tracker(models_dir)
    smoothed = _analyze(tracker)
    assert tracker.backend == "mediapipe"
    assert smoothed == [945] * 60
