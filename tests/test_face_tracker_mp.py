"""Tests for the upgraded speaker_tracker.py.

cv2 / mediapipe / ultralytics are NOT installed in this VM, so cv2 is stubbed
with a fake VideoCapture + fake cascades, and mediapipe / ultralytics are
injected as fake modules via sys.modules monkeypatching — exactly the
optional-import paths the tracker uses in production.
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

from speaker_tracker import SpeakerTracker  # noqa: E402

# --------------------------------------------------------------------------
# Fake mediapipe / ultralytics
# --------------------------------------------------------------------------

MP_PLAN = None    # list of relative (xmin, ymin, w, h) boxes, or callable(frame_idx)
YOLO_PLAN = None  # (xyxy list, conf list), or callable(frame_idx)


def _mp_detection(xmin, ymin, w, h):
    return SimpleNamespace(
        location_data=SimpleNamespace(
            relative_bounding_box=SimpleNamespace(
                xmin=xmin, ymin=ymin, width=w, height=h)))


class FakeFaceDetection:
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        pass

    def process(self, image):
        idx = int(image[0, 0, 0])
        plan = MP_PLAN
        boxes = plan(idx) if callable(plan) else (plan or [])
        return SimpleNamespace(detections=[_mp_detection(*b) for b in boxes])


def make_fake_mediapipe():
    mp = types.ModuleType("mediapipe")
    mp.solutions = SimpleNamespace(
        face_detection=SimpleNamespace(FaceDetection=FakeFaceDetection))
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
    global HAAR_PLAN, MP_PLAN, YOLO_PLAN
    HAAR_PLAN = None
    MP_PLAN = None
    YOLO_PLAN = None
    FAKE_VIDEO.update({"width": 1920, "height": 1080, "fps": 30.0, "n_frames": 60})
    monkeypatch.delenv("FACE_TRACKER_BACKEND", raising=False)
    monkeypatch.delenv("YOLO_FACE_MODEL", raising=False)
    for name in ("mediapipe", "ultralytics"):
        monkeypatch.delitem(sys.modules, name, raising=False)


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

    class BrokenFaceDetection:
        def __init__(self, *a, **k):
            raise RuntimeError("no model")

    mp.solutions.face_detection.FaceDetection = BrokenFaceDetection
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
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
    # Relative box xmin=0.6, w=0.1 -> x=1152, w=192, centre 1248
    # -> crop_x = 1248 - 303 = 945.
    MP_PLAN = [(0.6, 0.2, 0.1, 0.15)]
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed == [945] * 60


def test_mediapipe_deadzone_holds_camera_still(monkeypatch, models_dir):
    global MP_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())

    def plan(frame_idx):
        # Face centre jitters between 940 and 1000 px every sample
        # (targets 637 / 697) — inside the 15%-of-crop deadzone (~91 px).
        sample = frame_idx // 5
        centre = 940 if sample % 2 == 0 else 1000
        return [((centre - 96) / 1920, 0.2, 0.1, 0.15)]

    MP_PLAN = plan
    smoothed = _analyze(_tracker(models_dir))
    assert len(set(smoothed)) == 1  # no jitter at all


def test_mediapipe_snaps_on_large_jump(monkeypatch, models_dir):
    global MP_PLAN
    monkeypatch.setitem(sys.modules, "mediapipe", make_fake_mediapipe())

    def plan(frame_idx):
        centre = 500 if frame_idx < 30 else 1500
        return [((centre - 96) / 1920, 0.2, 0.1, 0.15)]

    MP_PLAN = plan
    smoothed = _analyze(_tracker(models_dir))
    assert smoothed[0] == 197    # 500 - 303
    assert smoothed[29] == 997   # snapping along the ramp, not lagging
    assert smoothed[30] == 1197  # 1500 - 303
    assert smoothed[-1] == 1197


def test_mediapipe_runtime_failure_falls_back_to_haar_mid_video(monkeypatch, models_dir):
    mp = make_fake_mediapipe()

    class ExplodingFaceDetection(FakeFaceDetection):
        def process(self, image):
            raise RuntimeError("boom")

    mp.solutions.face_detection.FaceDetection = ExplodingFaceDetection
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    tracker = _tracker(models_dir)
    assert tracker.backend == "mediapipe"
    smoothed = _analyze(tracker)  # must not raise
    assert smoothed == [656] * 60  # Haar found no faces -> centred
    assert tracker.backend == "haar"


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
