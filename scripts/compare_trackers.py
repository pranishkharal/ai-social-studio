"""Dev tool: compare the MediaPipe and Haar face-tracking backends on a real video.

Runs SpeakerTracker with backend=mediapipe and backend=haar over the same
clip and reports honest per-backend numbers:

  - backend actually selected (proves auto-selection / fallback)
  - detection rate: % of sampled frames with >=1 face
  - raw-target jitter: mean |delta| between consecutive per-sample crop
    targets, in px (before any smoothing — the detector's own stability)
  - smoothed-trajectory std (px): how much the final camera actually moves
  - wall-clock time for the whole analysis

Usage (needs the real cv2 + mediapipe, i.e. the venv python):

    ~/workspace/venv-whisper/bin/python scripts/compare_trackers.py [video] [seconds]

Defaults to the adinross 75 s capture in ~/workspace/demo/capture/ if it
exists, else the first .mp4 in clips/. Pass [seconds] to analyse only the
first N seconds (a temp subclip is cut with ffmpeg, stream copy).
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from speaker_tracker import SpeakerTracker  # noqa: E402

DEFAULT_VIDEO = Path.home() / "workspace" / "demo" / "capture" / "adinross_75s.mp4"
SAMPLE_FPS = 6


def _pick_video(arg):
    if arg:
        return Path(arg)
    if DEFAULT_VIDEO.exists():
        return DEFAULT_VIDEO
    clips = sorted((REPO_ROOT / "clips").glob("*.mp4"))
    if clips:
        return clips[0]
    raise SystemExit("No video given and no default clip found.")


def _subclip(video, seconds):
    """Cut the first *seconds* of *video* (stream copy) into a temp file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-t", str(seconds), "-c", "copy", tmp.name],
        check=True)
    return Path(tmp.name)


def _raw_targets(calls, crop_w, max_x):
    """Rebuild the per-sample crop targets from recorded detections.

    Mirrors the target logic in SpeakerTracker.analyze_video_trajectory:
    single face -> centre on it; group -> centre on the group if it fits,
    else the largest face; no face -> hold the last target.
    """
    centre = max_x // 2
    last = centre
    targets = []
    for boxes in calls:
        if len(boxes) == 1:
            fx, _fy, fw, _fh = boxes[0]
            last = max(0, min(max_x, (fx + fw // 2) - (crop_w // 2)))
        elif len(boxes) > 1:
            min_x = min(b[0] for b in boxes)
            max_person_x = max(b[0] + b[2] for b in boxes)
            if max_person_x - min_x <= crop_w:
                mid_x = (min_x + max_person_x) // 2
                last = max(0, min(max_x, mid_x - (crop_w // 2)))
            else:
                fx, _fy, fw, _fh = max(boxes, key=lambda b: b[2] * b[3])
                last = max(0, min(max_x, (fx + fw // 2) - (crop_w // 2)))
        targets.append(last)
    return targets


def run_backend(video, backend):
    """Analyse *video* with one backend; return a dict of metrics."""
    tracker = SpeakerTracker(models_dir=str(REPO_ROOT / "models"),
                             backend=backend)
    selected = tracker.backend  # forces lazy backend resolution
    calls = []
    orig_detect = tracker._detect_faces

    def recording(frame, _orig=orig_detect, _calls=calls):
        boxes = _orig(frame)
        _calls.append(list(boxes))
        return boxes

    tracker._detect_faces = recording
    start = time.monotonic()
    # fallback_to_haar=False: measure the RAW backend. With the default
    # adaptive fallback, a MediaPipe run that barely detects anything
    # would silently re-run with Haar and the numbers would be Haar's.
    smoothed, raw_w, raw_h, crop_w = tracker.analyze_video_trajectory(
        str(video), sample_fps=SAMPLE_FPS, fallback_to_haar=False)
    elapsed = time.monotonic() - start
    tracker.close()

    max_x = max(0, raw_w - crop_w)
    targets = _raw_targets(calls, crop_w, max_x)
    n = len(calls)
    with_face = sum(1 for b in calls if b)
    diffs = np.abs(np.diff(np.array(targets, dtype=float))) if n > 1 else np.array([0.0])
    return {
        "requested": backend,
        "selected": selected,
        "frames": f"{raw_w}x{raw_h}",
        "samples": n,
        "detection_rate_pct": 100.0 * with_face / n if n else 0.0,
        "faces_per_sample": sum(len(b) for b in calls) / n if n else 0.0,
        "raw_jitter_px": float(np.mean(diffs)),
        "raw_max_jump_px": float(np.max(diffs)) if len(diffs) else 0.0,
        "smoothed_std_px": float(np.std(smoothed)),
        "smoothed_range_px": int(max(smoothed) - min(smoothed)) if smoothed else 0,
        "wall_time_s": elapsed,
    }


def main():
    video = _pick_video(sys.argv[1] if len(sys.argv) > 1 else None)
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else None
    tmp = None
    if seconds:
        tmp = _subclip(video, seconds)
        video = tmp
        print(f"Analysing first {seconds:g} s of the clip "
              f"(temp subclip {video.name}).")
    print(f"Video: {video}")
    print(f"Sample rate: {SAMPLE_FPS} fps\n")

    # Auto-selection check: no backend requested, no env override.
    os.environ.pop("FACE_TRACKER_BACKEND", None)
    auto = SpeakerTracker(models_dir=str(REPO_ROOT / "models"))
    print(f"Auto-selection (FACE_TRACKER_BACKEND unset): backend = "
          f"{auto.backend!r}")
    auto.close()
    print()

    results = {}
    for backend in ("mediapipe", "haar"):
        print(f"Running backend={backend} ...", flush=True)
        results[backend] = run_backend(video, backend)
        print(f"  done in {results[backend]['wall_time_s']:.1f} s "
              f"(selected backend: {results[backend]['selected']!r})")

    print()
    print(f"{'metric':<28} {'mediapipe':>12} {'haar':>12}")
    print("-" * 54)
    rows = [
        ("backend selected", "selected", "{}"),
        ("video", "frames", "{}"),
        ("samples", "samples", "{:d}"),
        ("detection rate (%)", "detection_rate_pct", "{:.1f}"),
        ("faces / sample", "faces_per_sample", "{:.2f}"),
        ("raw jitter mean |d| (px)", "raw_jitter_px", "{:.1f}"),
        ("raw max jump (px)", "raw_max_jump_px", "{:.0f}"),
        ("smoothed traj std (px)", "smoothed_std_px", "{:.1f}"),
        ("smoothed traj range (px)", "smoothed_range_px", "{:d}"),
        ("wall time (s)", "wall_time_s", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:<28} "
              f"{fmt.format(results['mediapipe'][key]):>12} "
              f"{fmt.format(results['haar'][key]):>12}")

    if tmp is not None:
        os.unlink(tmp)


if __name__ == "__main__":
    main()
