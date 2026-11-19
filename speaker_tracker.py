"""
SpeakerTracker — face-tracked auto-framing for 9:16 vertical clips.

Backends, selected by the FACE_TRACKER_BACKEND env var
(auto / mediapipe / haar / yolo, default auto):

  1. MediaPipe BlazeFace (``mp.solutions.face_detection``) — the most robust
     detector; an *optional* dependency. If mediapipe is not installed the
     tracker falls back silently instead of crashing.
  2. YOLO face detection (``ultralytics``) — optional; only used when the
     package is installed. Model file from the YOLO_FACE_MODEL env var
     (default ``yolov8n-face.pt``).
  3. OpenCV Haar cascades — always available, the original behaviour.

The MediaPipe/YOLO camera behaviour (deadzone, exponential follow, snap) is
adapted from opensource-clipping (https://github.com/NaufalRizqullah/opensource-clipping,
MIT licence), clipping/studio/face_detection.py and
clipping/studio/render_camera_switch.py.

Public API is unchanged: analyze_video_trajectory() and
get_ffmpeg_crop_filter() keep their signatures and return shapes, and the
Haar path produces identical results to before this upgrade (same detection
code, same EMA smoothing).
"""

import os
import cv2
import numpy as np
from pathlib import Path

# --- Camera tunables adapted from opensource-clipping's renderers ---
TRACK_DEADZONE_RATIO = 0.15    # deadzone as a fraction of crop width
TRACK_SMOOTH_FACTOR = 0.30     # exponential follow factor per sample
TRACK_SNAP_RATIO = 0.08        # hard snap when the target jumps this far
MEDIAPIPE_MIN_CONFIDENCE = 0.5
YOLO_MIN_CONFIDENCE = 0.5
DEFAULT_YOLO_FACE_MODEL = "yolov8n-face.pt"


class SpeakerTracker:
    def __init__(self, models_dir: str = "models", backend: str = None):
        self.models_dir = Path(models_dir)
        self.frontal_cascade_path = str(self.models_dir / "haarcascade_frontalface_default.xml")
        self.profile_cascade_path = str(self.models_dir / "haarcascade_profileface.xml")

        if not os.path.exists(self.frontal_cascade_path):
            raise FileNotFoundError(f"Cascade not found at {self.frontal_cascade_path}")

        self.frontal_face = cv2.CascadeClassifier(self.frontal_cascade_path)
        self.profile_face = cv2.CascadeClassifier(self.profile_cascade_path) if os.path.exists(self.profile_cascade_path) else None

        # Backend preference: explicit arg wins, else FACE_TRACKER_BACKEND env.
        # Valid values: auto / mediapipe / haar / yolo. Default: auto.
        if backend is None:
            backend = os.getenv("FACE_TRACKER_BACKEND", "auto")
        self.backend_preference = str(backend).strip().lower()
        self._active_backend = None  # resolved lazily: "mediapipe" | "yolo" | "haar"
        self._mp_detector = None
        self._yolo_model = None

    # ------------------------------------------------------------------
    # Backend selection (lazy, never crashes: always ends on Haar)
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        """The backend actually in use: 'mediapipe', 'yolo' or 'haar'."""
        if self._active_backend is None:
            self._resolve_backend()
        return self._active_backend

    def _resolve_backend(self):
        pref = self.backend_preference
        tried = []
        if pref in ("auto", "mediapipe"):
            if self._init_mediapipe():
                self._active_backend = "mediapipe"
                return
            tried.append("mediapipe")
        if pref in ("auto", "yolo"):
            if self._init_yolo():
                self._active_backend = "yolo"
                return
            tried.append("yolo")
        if pref not in ("auto", "mediapipe", "yolo", "haar"):
            print(f"[SpeakerTracker] Unknown FACE_TRACKER_BACKEND={pref!r}; falling back to Haar.")
        elif tried:
            print(f"[SpeakerTracker] {', '.join(tried)} unavailable; falling back to Haar cascades.")
        self._active_backend = "haar"

    def _init_mediapipe(self) -> bool:
        """Optional import of mediapipe. Returns True if a detector is ready."""
        try:
            import mediapipe as mp
        except ImportError:
            return False
        try:
            face_detection = mp.solutions.face_detection
        except AttributeError:
            return False
        try:
            self._mp_detector = face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=MEDIAPIPE_MIN_CONFIDENCE,
            )
        except Exception as e:
            print(f"[SpeakerTracker] MediaPipe init failed ({e}); falling back.")
            self._mp_detector = None
            return False
        return True

    def _init_yolo(self) -> bool:
        """Optional import of ultralytics. Returns True if a model is ready."""
        try:
            from ultralytics import YOLO
        except ImportError:
            return False
        model_name = os.getenv("YOLO_FACE_MODEL", DEFAULT_YOLO_FACE_MODEL)
        try:
            import logging
            logging.getLogger("ultralytics").setLevel(logging.ERROR)
            self._yolo_model = YOLO(model_name)
        except Exception as e:
            print(f"[SpeakerTracker] YOLO init failed ({e}); falling back.")
            self._yolo_model = None
            return False
        return True

    # ------------------------------------------------------------------
    # Face detection — every backend returns [(x, y, w, h)] in full-res px
    # ------------------------------------------------------------------
    def _detect_faces_haar(self, frame):
        """Original Haar cascade detection (behaviour unchanged)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downsample for ultra-fast face detection
        scale_factor = 0.5
        small = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)

        # Detect frontal faces
        faces = self.frontal_face.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
        detected_boxes = []
        for (x, y, w, h) in faces:
            detected_boxes.append((int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)))

        # If no frontal face, try profile faces
        if len(detected_boxes) == 0 and self.profile_face:
            profiles = self.profile_face.detectMultiScale(small, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30))
            for (x, y, w, h) in profiles:
                detected_boxes.append((int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)))

        return detected_boxes

    def _detect_faces_mediapipe(self, frame):
        """MediaPipe BlazeFace (short-range) via mp.solutions.face_detection."""
        frame_h, frame_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_detector.process(rgb)
        detected_boxes = []
        for det in (getattr(results, "detections", None) or []):
            bb = det.location_data.relative_bounding_box
            x = int(max(0.0, bb.xmin) * frame_w)
            y = int(max(0.0, bb.ymin) * frame_h)
            w = int(min(1.0, bb.width) * frame_w)
            h = int(min(1.0, bb.height) * frame_h)
            if w > 0 and h > 0:
                detected_boxes.append((x, y, w, h))
        return detected_boxes

    def _detect_faces_yolo(self, frame):
        """YOLO face detection via ultralytics (yolov8n-face.pt by default)."""
        results = self._yolo_model(frame, verbose=False)
        detected_boxes = []
        if not results:
            return detected_boxes
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detected_boxes
        xyxy = boxes.xyxy
        xyxy = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
        confs = getattr(boxes, "conf", None)
        if confs is not None:
            confs = confs.tolist() if hasattr(confs, "tolist") else list(confs)
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            if confs is not None and i < len(confs) and confs[i] < YOLO_MIN_CONFIDENCE:
                continue
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            if w > 0 and h > 0:
                detected_boxes.append((x, y, w, h))
        return detected_boxes

    def _detect_faces(self, frame):
        """Dispatch to the active backend; any runtime failure drops to Haar."""
        backend = self.backend
        try:
            if backend == "mediapipe":
                return self._detect_faces_mediapipe(frame)
            if backend == "yolo":
                return self._detect_faces_yolo(frame)
        except Exception as e:
            print(f"[SpeakerTracker] {backend} detection failed ({e}); falling back to Haar for this video.")
            self._active_backend = "haar"
        return self._detect_faces_haar(frame)

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------
    def analyze_video_trajectory(self, video_path: str, target_aspect: float = 9/16, sample_fps: int = 6) -> list:
        """
        Analyzes video frames to track the active talking person(s).
        Returns a smoothed list of crop_x positions for the video duration.

        Signature and return shape are unchanged:
        (smoothed_x, raw_w, raw_h, crop_w) — smoothed_x has one crop_x per
        frame of the source video. Pass sample_fps=4 for the 0.25 s sampling
        cadence opensource-clipping uses with MediaPipe.
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
                detected_boxes = self._detect_faces(frame)

                if len(detected_boxes) == 1:
                    # Single person in frame - center crop on this person
                    fx, fy, fw, fh = detected_boxes[0]
                    person_center_x = fx + (fw // 2)
                    target_x = person_center_x - (crop_w // 2)
                    last_known_target = max(0, min(max_x, target_x))
                elif len(detected_boxes) > 1:
                    # Multiple people in frame (e.g. Amanda & Zavala)
                    # Calculate bounding box of the group
                    min_x = min(b[0] for b in detected_boxes)
                    max_person_x = max(b[0] + b[2] for b in detected_boxes)
                    group_span = max_person_x - min_x

                    if group_span <= crop_w:
                        # Both fit in vertical frame! Center on both
                        mid_x = (min_x + max_person_x) // 2
                        target_x = mid_x - (crop_w // 2)
                    else:
                        # Too far apart: prioritize largest / most central face
                        sorted_faces = sorted(detected_boxes, key=lambda b: b[2] * b[3], reverse=True)
                        fx, fy, fw, fh = sorted_faces[0]
                        target_x = (fx + fw // 2) - (crop_w // 2)

                    last_known_target = max(0, min(max_x, target_x))

                sample_x_targets.append(last_known_target)
                sample_frame_indices.append(frame_idx)

            frame_idx += 1

        cap.release()

        if not sample_x_targets:
            return [center_default_x] * total_frames

        # Interpolate trajectory across all frames
        raw_interpolated = np.interp(range(total_frames), sample_frame_indices, sample_x_targets)

        if self._active_backend == "haar":
            # Original EMA glide — kept identical for the Haar path.
            # Use Exponential Moving Average (EMA) to ensure smooth camera dolly glide without jitter
            smoothed_x = []
            cur_x = raw_interpolated[0]
            alpha = 0.05  # Smooth cinematic camera gliding factor

            for val in raw_interpolated:
                cur_x = cur_x * (1 - alpha) + val * alpha
                smoothed_x.append(int(max(0, min(max_x, round(cur_x)))))

            return smoothed_x, raw_w, raw_h, crop_w

        # MediaPipe / YOLO: opensource-clipping style deadzone + exponential
        # follow + snap, so the crop never jitters on small face movements
        # but still jumps promptly when the speaker changes.
        smoothed_x = self._smooth_trajectory_opensource(
            raw_interpolated, crop_w, raw_w, max_x
        )
        return smoothed_x, raw_w, raw_h, crop_w

    @staticmethod
    def _smooth_trajectory_opensource(raw_interpolated, crop_w: int, raw_w: int, max_x: int) -> list:
        """
        Deadzone + exponential follow + snap camera smoothing.

        Adapted from opensource-clipping's render_camera_switch.py
        (_smooth_positions_cs): the camera seeds from the median of the first
        samples, ignores movement inside a deadzone (15% of crop width),
        follows exponentially (factor 0.30) outside it, and snaps instantly
        when the target jumps more than 8% of the frame width.
        """
        values = [float(v) for v in raw_interpolated]
        if not values:
            return []
        cam_x = float(np.median(values[:5]))
        deadzone_px = crop_w * TRACK_DEADZONE_RATIO
        snap_px = raw_w * TRACK_SNAP_RATIO
        smoothed = []
        for target in values:
            if abs(target - cam_x) > snap_px:
                cam_x = target
            elif target > cam_x + deadzone_px:
                cam_x += (target - (cam_x + deadzone_px)) * TRACK_SMOOTH_FACTOR
            elif target < cam_x - deadzone_px:
                cam_x += (target - (cam_x - deadzone_px)) * TRACK_SMOOTH_FACTOR
            # else: inside the deadzone — hold the camera still.
            smoothed.append(int(max(0, min(max_x, round(cam_x)))))
        return smoothed

    def get_ffmpeg_crop_filter(self, smoothed_x: list, orig_fps: float = 30.0, crop_w: int = 608, raw_h: int = 1080) -> str:
        """
        Creates an ultra-efficient FFmpeg crop filter expression.
        If movement is subtle, uses static optimal crop.
        If movement shifts between speakers, uses smooth pan expression.
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
    tracker = SpeakerTracker("models")
    print(f"Face tracker backend: {tracker.backend}")
    # Test clip: first CLI argument, else the example clip in this repo's
    # clips/ folder (no machine-specific path).
    if len(sys.argv) > 1:
        test_clip = sys.argv[1]
    else:
        test_clip = str(Path(__file__).resolve().parent / "clips"
                        / "adinross_20261116_063325_raw.mp4")
    if os.path.exists(test_clip):
        smooth_x, rw, rh, cw = tracker.analyze_video_trajectory(test_clip)
        filt = tracker.get_ffmpeg_crop_filter(smooth_x, 30.0, cw, rh)
        print("Speaker tracking successful!")
        print(f"Frames: {len(smooth_x)}, Raw: {rw}x{rh}, CropW: {cw}")
        print(f"Generated FFmpeg Filter: {filt[:120]}...")
    else:
        print(f"Test clip not found at {test_clip}")
