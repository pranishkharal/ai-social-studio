"""
BodyTracker — full-body-tracked auto-framing for 9:16 vertical clips.

FEATURE 3: an alternative tracking target for the speaker_tracking layout.
Instead of following the streamer's face, the crop follows their whole
body (MediaPipe Pose landmarks). Best for IRL / standing / dancing
streams where the streamer moves around the frame; for a seated facecam
stream the face tracker is usually the better choice, and the editor
falls back to it whenever no body is found (see body_found below).

Backend: MediaPipe Tasks API PoseLandmarker (IMAGE running mode,
pose_landmarker_lite.task). mediapipe 1.2.3 has NO legacy solutions API
(mp.solutions does not exist) — the Tasks API (mp.tasks.vision) is the
only way in. The model is auto-downloaded into models/ on first use;
if mediapipe is missing, the download fails, or no body is detected in
the whole video, the tracker reports body_found=False and returns a
centred trajectory so the caller can fall back to face tracking. It
never raises for a detection/backend problem (an unreadable video
still raises RuntimeError, exactly like SpeakerTracker).

Multi-person rule: when several poses are detected in one frame, the
LARGEST body box (by area) wins — the streamer is normally the biggest
person on screen, and game characters are usually smaller. This is a
deliberate choice, documented here.

YOLO person detection was deliberately NOT implemented: ultralytics is
not installed in this VM and is too heavy to install here, so a YOLO
body path could not be verified. MediaPipe Pose is the body backend;
face tracking is the fallback.

Public API mirrors SpeakerTracker: analyze_video_trajectory() and
get_ffmpeg_crop_filter() keep the same signatures and return shapes.
Camera smoothing reuses SpeakerTracker._smooth_trajectory_opensource
(deadzone + exponential follow + snap, adapted from
opensource-clipping) so both trackers move the camera identically.
"""

import os
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from speaker_tracker import SpeakerTracker

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_FILENAME = "pose_landmarker_lite.task"
POSE_MIN_CONFIDENCE = 0.5
POSE_DOWNLOAD_TIMEOUT_SECONDS = 60
DEFAULT_NUM_POSES = 2  # streamer + one more person on screen
BODY_BOX_MARGIN_RATIO = 0.15  # pad the landmark box by 15% on each side
BODY_BOX_MIN_MARGIN_PX = 10
# Landmarks below this visibility are ignored when building the box
# (a partial body — e.g. a facecam showing only the upper body — has
# low-visibility leg landmarks with unreliable positions). If too few
# landmarks pass, all of them are used instead of giving up.
POSE_MIN_LANDMARK_VISIBILITY = 0.5
POSE_MIN_VISIBLE_LANDMARKS = 4


class BodyTracker:
    def __init__(self, models_dir: str = "models",
                 min_pose_detection_confidence: float = POSE_MIN_CONFIDENCE,
                 num_poses: int = DEFAULT_NUM_POSES,
                 model_path: str = None):
        self.models_dir = Path(models_dir)
        self.model_path = (Path(model_path) if model_path
                           else self.models_dir / POSE_MODEL_FILENAME)
        self.min_pose_detection_confidence = min_pose_detection_confidence
        self.num_poses = num_poses

        # Lazy backend state (mirrors SpeakerTracker): None = not tried
        # yet, True/False = MediaPipe Pose ready or unavailable.
        self._backend_ready = None
        self._landmarker = None
        self._mp = None

        # Results of the last analyze_video_trajectory() run. body_found
        # is the fallback signal the editor checks: False means "no body
        # in this video (or no Pose backend) — use face tracking".
        self.body_found = False
        self.sampled_frames = 0
        self.detected_frames = 0
        self.last_body_box = None  # (x, y, w, h) of the latest detection

    # ------------------------------------------------------------------
    # Backend selection (lazy, never crashes: ends on "unavailable")
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        """The backend in use: 'mediapipe' or 'unavailable'."""
        if self._backend_ready is None:
            self._backend_ready = self._init_mediapipe()
        return "mediapipe" if self._backend_ready else "unavailable"

    @property
    def detection_rate(self) -> float:
        """Fraction of sampled frames with at least one body (0.0-1.0)."""
        if not self.sampled_frames:
            return 0.0
        return self.detected_frames / self.sampled_frames

    def close(self):
        """Release the native landmarker (callers that create a tracker
        per render should call this in a finally block — each landmarker
        holds a LiteRT environment, and an unclosed one also logs a
        dispatcher error when the process exits). Safe to call twice
        or when the backend never started."""
        landmarker, self._landmarker = self._landmarker, None
        self._backend_ready = False
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass

    def _init_mediapipe(self) -> bool:
        """Optional import of mediapipe + PoseLandmarker setup.

        Returns True only when a landmarker is actually ready. Every
        failure mode (no mediapipe, old mediapipe without the Tasks
        API, no model file, broken model) returns False instead of
        raising, so the caller can fall back to face tracking.
        """
        try:
            import mediapipe as mp
        except ImportError:
            return False
        try:
            # NOTE: in mediapipe 1.2.3, mp.tasks IS mediapipe.tasks.python
            # (the tasks package re-exports it) — mp.tasks.python.vision
            # does NOT exist. There is also no mp.solutions at all.
            vision = mp.tasks.vision
            PoseLandmarker = vision.PoseLandmarker
            PoseLandmarkerOptions = vision.PoseLandmarkerOptions
            BaseOptions = mp.tasks.BaseOptions
            RunningMode = vision.RunningMode
        except AttributeError:
            return False
        if not self._ensure_model():
            return False
        try:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(self.model_path)),
                running_mode=RunningMode.IMAGE,
                num_poses=self.num_poses,
                min_pose_detection_confidence=self.min_pose_detection_confidence,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[BodyTracker] MediaPipe Pose init failed ({e}); "
                  "body tracking unavailable.")
            self._landmarker = None
            return False
        self._mp = mp
        return True

    # ------------------------------------------------------------------
    # Model auto-download (fail soft: retry once, then give up cleanly)
    # ------------------------------------------------------------------
    def _ensure_model(self) -> bool:
        """Make sure the pose model file exists and is non-empty.

        Downloads it into models/ on first use (same URL the model is
        published at), retrying once. A failed or empty download leaves
        no partial file behind and returns False.
        """
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            return True
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for attempt in (1, 2):
            try:
                print(f"[BodyTracker] Downloading pose model "
                      f"(attempt {attempt}/2)...")
                req = urllib.request.Request(
                    POSE_MODEL_URL,
                    headers={"User-Agent": "ai-social-studio/1.0"})
                with urllib.request.urlopen(
                        req, timeout=POSE_DOWNLOAD_TIMEOUT_SECONDS) as resp, \
                        open(self.model_path, "wb") as f:
                    shutil.copyfileobj(resp, f)
                if (self.model_path.exists()
                        and self.model_path.stat().st_size > 0):
                    return True
                print("[BodyTracker] Pose model download was empty.")
            except Exception as e:
                print(f"[BodyTracker] Pose model download failed "
                      f"(attempt {attempt}/2): {e}")
            # Remove a partial file so the next run retries cleanly.
            try:
                if self.model_path.exists():
                    self.model_path.unlink()
            except OSError:
                pass
        print("[BodyTracker] Could not download the pose model; "
              "body tracking unavailable.")
        return False

    # ------------------------------------------------------------------
    # Pose detection — returns [(x, y, w, h)] in full-res px
    # ------------------------------------------------------------------
    @staticmethod
    def landmarks_to_box(landmarks, frame_w: int, frame_h: int,
                         margin_ratio: float = BODY_BOX_MARGIN_RATIO):
        """Body bounding box (x, y, w, h) in pixels from pose landmarks.

        landmarks: objects with .x/.y in normalized [0, 1] coords
        (MediaPipe NormalizedLandmark). The box is the min/max of the
        landmarks, padded by margin_ratio on each side and clamped to
        the frame. Landmarks with visibility < 0.5 are ignored when
        enough visible ones remain (see POSE_MIN_LANDMARK_VISIBILITY).
        Returns None when there are no usable landmarks.
        """
        points = list(landmarks or [])
        if not points:
            return None
        visible = [lm for lm in points
                   if getattr(lm, "visibility", 1.0) >= POSE_MIN_LANDMARK_VISIBILITY]
        use = visible if len(visible) >= POSE_MIN_VISIBLE_LANDMARKS else points
        xs = [min(1.0, max(0.0, float(lm.x))) for lm in use]
        ys = [min(1.0, max(0.0, float(lm.y))) for lm in use]
        if not xs:
            return None
        x1 = min(xs) * frame_w
        x2 = max(xs) * frame_w
        y1 = min(ys) * frame_h
        y2 = max(ys) * frame_h
        mx = max((x2 - x1) * margin_ratio, BODY_BOX_MIN_MARGIN_PX)
        my = max((y2 - y1) * margin_ratio, BODY_BOX_MIN_MARGIN_PX)
        x = int(max(0, x1 - mx))
        y = int(max(0, y1 - my))
        x_end = int(min(frame_w, x2 + mx))
        y_end = int(min(frame_h, y2 + my))
        if x_end <= x or y_end <= y:
            return None
        return (x, y, x_end - x, y_end - y)

    def _detect_bodies(self, frame):
        """One (x, y, w, h) box per detected pose, in full-res px."""
        if self.backend != "mediapipe":
            return []
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb)
            results = self._landmarker.detect(image)
        except Exception as e:
            print(f"[BodyTracker] Pose detection failed ({e}); "
                  "body tracking unavailable for this video.")
            self._backend_ready = False
            self._landmarker = None
            return []
        boxes = []
        for pose in (getattr(results, "pose_landmarks", None) or []):
            box = self.landmarks_to_box(pose, frame_w, frame_h)
            if box is not None:
                boxes.append(box)
        return boxes

    # ------------------------------------------------------------------
    # Trajectory (same shape as SpeakerTracker.analyze_video_trajectory)
    # ------------------------------------------------------------------
    def analyze_video_trajectory(self, video_path: str, target_aspect: float = 9/16, sample_fps: int = 6) -> list:
        """
        Analyzes video frames to track the largest detected body.

        Returns (smoothed_x, raw_w, raw_h, crop_w) — smoothed_x has one
        crop_x per frame of the source video, exactly like
        SpeakerTracker. After the call, self.body_found says whether
        any body was detected at all: when it is False the caller
        should fall back to face tracking (the returned trajectory is
        just a centred default in that case).

        Note: unlike SpeakerTracker, the empty-video path returns the
        same 4-tuple instead of a bare list.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # In vertical 9:16 crop from 16:9 (e.g. 1920x1080)
        crop_w = int(raw_h * target_aspect)
        max_x = max(0, raw_w - crop_w)
        center_default_x = max_x // 2

        # Fresh per-run stats (body_found is the fallback signal).
        self.body_found = False
        self.sampled_frames = 0
        self.detected_frames = 0
        self.last_body_box = None

        frame_interval = max(1, int(orig_fps / sample_fps))
        sample_x_targets = []
        sample_frame_indices = []

        last_known_target = center_default_x
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                self.sampled_frames += 1
                detected_boxes = self._detect_bodies(frame)

                if detected_boxes:
                    self.detected_frames += 1
                    self.body_found = True
                    # Multi-person rule (documented in the module
                    # docstring): the LARGEST body box wins.
                    bx, by, bw, bh = max(detected_boxes,
                                         key=lambda b: b[2] * b[3])
                    self.last_body_box = (bx, by, bw, bh)
                    person_center_x = bx + (bw // 2)
                    target_x = person_center_x - (crop_w // 2)
                    last_known_target = max(0, min(max_x, target_x))
                # No body in this frame: hold the last known target
                # (same as the face tracker holding its last face).

                sample_x_targets.append(last_known_target)
                sample_frame_indices.append(frame_idx)

            frame_idx += 1

        cap.release()

        if not sample_x_targets:
            return [center_default_x] * total_frames, raw_w, raw_h, crop_w

        # Interpolate trajectory across all frames, then smooth with
        # the SAME deadzone + exponential follow + snap camera the
        # MediaPipe face path uses (shared helper, not a reimplementation).
        raw_interpolated = np.interp(range(total_frames), sample_frame_indices, sample_x_targets)
        smoothed_x = SpeakerTracker._smooth_trajectory_opensource(
            raw_interpolated, crop_w, raw_w, max_x)
        return smoothed_x, raw_w, raw_h, crop_w

    def get_ffmpeg_crop_filter(self, smoothed_x: list, orig_fps: float = 30.0, crop_w: int = 608, raw_h: int = 1080) -> str:
        """
        Creates an ultra-efficient FFmpeg crop filter expression.
        If movement is subtle, uses static optimal crop.
        If movement shifts between speakers, uses smooth pan expression.

        Duplicated from speaker_tracker.SpeakerTracker.get_ffmpeg_crop_filter
        (same logic, kept in sync deliberately) so BodyTracker does not
        need a SpeakerTracker instance to build its filter.
        """
        if not smoothed_x:
            return f"crop={crop_w}:{raw_h}:(iw-{crop_w})/2:0"

        avg_x = int(np.mean(smoothed_x))
        std_x = np.std(smoothed_x)

        # If camera is mostly stationary, stick to optimal centered speaker crop
        if std_x < 25:
            return f"crop={crop_w}:{raw_h}:{avg_x}:0,scale=1080:1920"

        # If camera tracks between speakers, sample keypoints every 1.5 seconds
        step = int(orig_fps * 1.5)
        keyframes = []
        for i in range(0, len(smoothed_x), step):
            t_sec = round(i / orig_fps, 2)
            keyframes.append((t_sec, smoothed_x[i]))

        # Build piecewise linear interpolation expression for FFmpeg
        # x(t) transitions smoothly between keyframe timestamps
        expr = f"{keyframes[-1][1]}"
        for k in range(len(keyframes) - 2, -1, -1):
            t_start, x_start = keyframes[k]
            t_end, x_end = keyframes[k + 1]
            slope = (x_end - x_start) / max(0.01, (t_end - t_start))
            current_seg = f"({x_start}+{slope}*(t-{t_start}))"
            expr = f"if(lt(t,{t_end}),{current_seg},{expr})"

        return f"crop={crop_w}:{raw_h}:'{expr}':0,scale=1080:1920"


if __name__ == "__main__":
    import sys
    tracker = BodyTracker("models")
    print(f"Body tracker backend: {tracker.backend}")
    if len(sys.argv) > 1:
        test_clip = sys.argv[1]
    else:
        test_clip = str(Path(__file__).resolve().parent / "clips"
                        / "adinross_20261116_063325_raw.mp4")
    if os.path.exists(test_clip):
        smooth_x, rw, rh, cw = tracker.analyze_video_trajectory(test_clip)
        filt = tracker.get_ffmpeg_crop_filter(smooth_x, 30.0, cw, rh)
        print("Body tracking successful!")
        print(f"Frames: {len(smooth_x)}, Raw: {rw}x{rh}, CropW: {cw}")
        print(f"Body found: {tracker.body_found} "
              f"({tracker.detected_frames}/{tracker.sampled_frames} sampled frames, "
              f"rate {tracker.detection_rate:.0%})")
        print(f"Last body box: {tracker.last_body_box}")
        print(f"Generated FFmpeg Filter: {filt[:120]}...")
        tracker.close()
    else:
        print(f"Test clip not found at {test_clip}")
