"""
speaker_framing.py — Active-speaker framing: follow whoever is talking.

For shots with people sitting apart (podcasts, couch streams), this module
works out *who is speaking when* and produces the speaker data the
camera-switch / split-screen layouts need:

    {
        "speakers": ["speaker_1", "speaker_2"],
        "speaker_segments": [{"speaker", "start", "end"}, ...],
        "trajectories": {"speaker_1": {"x", "crop_w", "raw_h"}, ...},
        "split_speakers": [{"speaker", "x", "crop_w", "raw_h"}, ...],
        "split_spans": [(start, end, ("speaker_1", "speaker_2")), ...],
        "method": "landmarker" | "fallback",
    }

``speaker_segments`` + ``trajectories`` plug straight into
``render_modes.build_camera_switch_filter`` (and
``build_speaker_layout_filter`` for the split spans), and
``split_speakers`` into ``render_modes.build_split_screen_filter``. The
pipeline's camera-switch / split-screen layouts used to fall back to
plain face tracking because no speaker data was produced ("no
diarization") — this module is what produces it.

How it works:
  1. Faces are sampled with ``SpeakerTracker.analyze_face_boxes`` (the
     same backends + adaptive Haar fallback the face tracker uses) and
     clustered by x-centre into people (a couch hardly moves).
  2. Lip motion is measured per person, 10 times a second:
       * FaceLandmarker path (preferred): MediaPipe's FaceLandmarker
         reads the ``jawOpen`` blendshape on a crop of each face; the
         score is the smoothed change in jaw openness. The model file
         (``face_landmarker.task``, 3.6 MB) is *not* in the repo — it is
         downloaded at runtime on first use from MediaPipe's official
         hosting into ``models/`` (see ``ensure_landmarker_model``), or
         pointed at with the ``SPEAKER_FRAMING_MODEL_PATH`` env var.
       * Fallback path (no model): mouth-region frame difference per
         face (the lower third of the face box, where the mouth is)
         stands in for lip motion.
     In both paths the score is multiplied by a speech-loudness gate
     (the clip's own audio, per-sample energy above its quietest third),
     so a listener's grin cannot steal the crop.
  3. The speaker is labelled per sample (switch only to a clearly more
     active mouth: 1.3x the runner-up and above ``SPEECH_MOTION``), the
     label is held for at least ``MIN_HOLD`` (1.5 s) so the camera never
     flickers, and spans where two people trade lines fast become
     split-screen spans.
  4. When unsure — fewer than 2 people, or lips barely moving anywhere —
     ``analyze_speakers`` returns None and the caller keeps its
     fallback (blurred fill / plain face tracking).

Technique adapted from gta6-clipper (https://gitlab.com/nocodehack.io/
gta6-clipper, MIT licence), reframe.py ``speakers()`` — reimplemented
here on top of this repo's SpeakerTracker and render_modes contract.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# FaceLandmarker model (Google's public hosting; the file is NOT in the
# repo — binaries cannot be pushed. Downloaded on first use, like the
# BlazeFace model in speaker_tracker.py).
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
LANDMARKER_FILENAME = "face_landmarker.task"
LANDMARKER_MIN_BYTES = 3_000_000   # real file is 3,758,596 bytes
LANDMARKER_MAX_BYTES = 5_000_000
LANDMARKER_DOWNLOAD_RETRIES = 2
ENV_MODEL_PATH = "SPEAKER_FRAMING_MODEL_PATH"
ENV_LANDMARKER = "SPEAKER_FRAMING_LANDMARKER"  # "0" forces the fallback

SPEECH_MOTION = 0.01   # mean jawOpen change per sample that counts as talking
MIN_HOLD = 1.5         # seconds a speaker stays on screen before a cut
SAMPLE_HZ = 10.0       # lip-motion samples per second (every 3rd frame at 30 fps)
SWITCH_MARGIN = 1.3    # new speaker must beat the current one by this factor
MIN_SPEECH_COVERAGE = 0.3  # fraction of samples with real lip motion, else unsure
MAX_PEOPLE = 5


def _creationflags():
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ---------------------------------------------------------------------------
# model download (runtime, never committed)
# ---------------------------------------------------------------------------


def _valid_model_file(path):
    try:
        size = Path(path).stat().st_size
    except OSError:
        return False
    return LANDMARKER_MIN_BYTES <= size <= LANDMARKER_MAX_BYTES


def _download_file(url, dest_path, timeout=60):
    """Download *url* to *dest_path*. Raises on any failure. Kept as its
    own function so tests can monkeypatch it without touching the network."""
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "ai-social-studio/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, \
            open(dest_path, "wb") as fh:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            fh.write(chunk)


def ensure_landmarker_model(models_dir="models"):
    """
    Return the FaceLandmarker model path, downloading it on first use.

    ``SPEAKER_FRAMING_MODEL_PATH`` points at an existing .task file and
    wins over everything (useful when the model is already on disk).
    Returns None when the model cannot be obtained — callers fall back
    to mouth-motion scoring. Downloads go to a ``.download`` temp file
    that is only renamed into place after validation, so a failed
    download never leaves a bad model behind.
    """
    override = os.environ.get(ENV_MODEL_PATH, "").strip()
    if override:
        path = Path(override)
        return path if _valid_model_file(path) else None
    models_dir = Path(models_dir)
    model_path = models_dir / LANDMARKER_FILENAME
    if _valid_model_file(model_path):
        return model_path
    tmp_path = model_path.with_suffix(".task.download")
    for attempt in range(1, LANDMARKER_DOWNLOAD_RETRIES + 1):
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            _download_file(LANDMARKER_URL, tmp_path)
        except Exception as e:
            print(f"[SpeakerFraming] FaceLandmarker download attempt "
                  f"{attempt}/{LANDMARKER_DOWNLOAD_RETRIES} failed ({e}).")
            try:
                tmp_path.unlink()
            except OSError:
                pass
            if attempt < LANDMARKER_DOWNLOAD_RETRIES:
                time.sleep(1)
            continue
        if _valid_model_file(tmp_path):
            os.replace(tmp_path, model_path)
            return model_path
        print("[SpeakerFraming] Downloaded FaceLandmarker model failed "
              "validation; discarding it.")
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return None


def landmarker_enabled():
    """False when SPEAKER_FRAMING_LANDMARKER=0 (force the fallback)."""
    return os.environ.get(ENV_LANDMARKER, "1").strip().lower() not in (
        "0", "false", "no", "off")


# ---------------------------------------------------------------------------
# pure helpers (no cv2 / ffmpeg — fully testable offline)
# ---------------------------------------------------------------------------


def cluster_people(face_samples, min_presence=0.5, max_people=MAX_PEOPLE):
    """
    Cluster sampled face boxes into people by x-centre.

    Args:
        face_samples: [(frame_idx, [(x, y, w, h), ...]), ...] as returned
            by ``SpeakerTracker.analyze_face_boxes``.
        min_presence: a cluster must appear in at least this fraction of
            sampled frames to count as a person (drops stray detections).

    Returns:
        List of median (x, y, w, h) boxes, left to right, at most
        ``max_people``. A box joins the cluster whose median x-centre is
        within 0.6 face-widths of it.
    """
    clusters = []
    n_frames = max(len(face_samples), 1)
    for _, boxes in face_samples:
        for box in boxes:
            cx = box[0] + box[2] / 2
            match = None
            for cluster in clusters:
                median_cx = float(np.median([b[0] + b[2] / 2 for b in cluster]))
                if abs(median_cx - cx) < 0.6 * box[2]:
                    match = cluster
                    break
            if match is not None:
                match.append(box)
            else:
                clusters.append([box])
    people = []
    for cluster in clusters:
        if len(cluster) >= min_presence * n_frames:
            people.append(tuple(float(v) for v in np.median(cluster, axis=0)))
    people.sort(key=lambda b: b[0])
    return people[:max_people]


def smooth_scores(scores, window_seconds, hz=SAMPLE_HZ):
    """Moving-average smoothing of per-person lip-motion scores.

    ``scores`` is (n_people, n_samples); the window is at least 3 samples
    and never longer than the clip.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    n = scores.shape[1]
    win = min(max(3, int(hz * window_seconds)), n)
    if win % 2 == 0:
        win = max(3, win - 1)
    kernel = np.ones(win) / win
    return np.array([np.convolve(row, kernel, mode="same") for row in scores])


def runs(labels):
    """Consecutive equal labels -> [[label, start, end), ...]."""
    out = []
    for i, value in enumerate(labels):
        if out and out[-1][0] == value:
            out[-1][2] = i + 1
        else:
            out.append([value, i, i + 1])
    return out


def hold(labels, n):
    """Give runs shorter than ``n`` samples to their longer neighbour,
    shortest first, so nobody flashes by."""
    labels = list(labels)
    runs_list = runs(labels)
    while len(runs_list) > 1:
        i = min(range(len(runs_list)),
                key=lambda j: runs_list[j][2] - runs_list[j][1])
        if runs_list[i][2] - runs_list[i][1] >= n:
            break
        neighbours = [j for j in (i - 1, i + 1)
                      if 0 <= j < len(runs_list)]
        nb = max(neighbours, key=lambda j: runs_list[j][2] - runs_list[j][1])
        labels[runs_list[i][1]:runs_list[i][2]] = (
            [runs_list[nb][0]] * (runs_list[i][2] - runs_list[i][1]))
        runs_list = runs(labels)
    return labels


def trades(labels, n):
    """Spans where two people swap lines fast: >= 3 runs in a row, each
    shorter than ``n`` samples, only two people -> [(start, end, (a, b))]."""
    runs_list, out, i = runs(labels), [], 0
    while i < len(runs_list):
        j = i
        while j < len(runs_list) and runs_list[j][2] - runs_list[j][1] < n:
            j += 1
        who = sorted({r[0] for r in runs_list[i:j]})
        if j - i >= 3 and len(who) == 2:
            out.append((runs_list[i][1], runs_list[j - 1][2], tuple(who)))
        i = max(j, i + 1)
    return out


def label_speakers(scores, hz=SAMPLE_HZ, min_hold=MIN_HOLD):
    """
    Turn per-person lip-motion scores into per-sample speaker labels.

    The speaker switches only to a clearly more active mouth (1.3x the
    runner-up and above SPEECH_MOTION); through silence and ties the
    current speaker is kept. Short runs are merged so every speaker
    holds the screen for at least ``min_hold`` seconds.

    Returns (labels, split_spans): labels is a list of person indices
    (or (a, b) tuples for fast-trade split spans), split_spans is
    [(start_sample, end_sample, (a, b)), ...].
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or scores.shape[1] == 0:
        return [], []
    n_samples = scores.shape[1]
    labels = [int(scores[:, 0].argmax())]
    for t in range(1, n_samples):
        ranked = np.sort(scores[:, t])
        top, second = ranked[-1], ranked[-2] if len(ranked) > 1 else 0.0
        if top > SWITCH_MARGIN * second and top > SPEECH_MOTION:
            labels.append(int(scores[:, t].argmax()))
        else:
            labels.append(labels[-1])
    fast = hold(labels, int(0.5 * hz))
    slow = hold(fast, int(min_hold * hz))
    split_spans = trades(fast, int(2.5 * hz))
    for start, end, pair in split_spans:
        slow[start:end] = [pair] * (end - start)
    return slow, split_spans


def segments_from_labels(labels, hz=SAMPLE_HZ, names=None):
    """
    Per-sample labels -> [{"speaker", "start", "end"}] turns in seconds.

    A label that is a (a, b) tuple becomes a segment with
    ``"speaker": "<a>+<b>"`` and a ``"split": [name_a, name_b]`` key, so
    ``render_modes.build_speaker_layout_filter`` can render it as a
    split screen.
    """
    names = names or []
    segments = []
    for label, start, end in runs(labels):
        seg = {
            "start": round(start / hz, 2),
            "end": round(end / hz, 2),
        }
        if isinstance(label, tuple):
            a, b = label
            name_a = names[a] if a < len(names) else f"speaker_{a + 1}"
            name_b = names[b] if b < len(names) else f"speaker_{b + 1}"
            seg["speaker"] = f"{name_a}+{name_b}"
            seg["split"] = [name_a, name_b]
        else:
            seg["speaker"] = (names[label] if label < len(names)
                              else f"speaker_{label + 1}")
        segments.append(seg)
    return segments


def trajectory_for_box(box, raw_w, raw_h, target_aspect=9 / 16):
    """
    render_modes trajectory for one person: a static 9:16 crop centred
    on the face (a couch hardly moves, so a fixed crop per person is
    right — the camera *cuts* between people, it does not pan).
    """
    x, _y, w, _h = box
    crop_w = int(raw_h * target_aspect)
    max_x = max(0, raw_w - crop_w)
    crop_x = int(min(max(x + w / 2 - crop_w / 2, 0), max_x))
    return {"x": crop_x, "crop_w": crop_w, "raw_h": int(raw_h)}


# ---------------------------------------------------------------------------
# audio gate + lip-motion measurement (need ffmpeg / cv2 — lazy imports)
# ---------------------------------------------------------------------------


def speech_loudness_mask(video_path, n_samples, hz=SAMPLE_HZ):
    """
    Boolean per-sample mask: True where the clip is louder than its
    quietest third — somebody is talking. A listener's grin must not
    steal the crop. Returns all-True when the audio cannot be read
    (the lip-motion scores still work, just ungated).
    """
    try:
        res = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video_path),
             "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
            capture_output=True, timeout=120,
            creationflags=_creationflags())
        if res.returncode != 0 or not res.stdout:
            raise RuntimeError("ffmpeg audio decode failed")
        samples = np.frombuffer(res.stdout, np.int16).astype(np.float32) ** 2
        step = 8000 / hz
        needed = int(step * n_samples)
        if len(samples) < needed:
            samples = np.pad(samples, (0, needed - len(samples)))
        energy = 10 * np.log10(
            np.add.reduceat(samples, (np.arange(n_samples) * step).astype(int))
            / step + 1)
        return energy > np.percentile(energy, 33)
    except Exception:
        return np.ones(n_samples, dtype=bool)


def _jawopen_scores(video_path, boxes, fps, n_samples, model_path):
    """
    FaceLandmarker path: per-person jawOpen blendshape sampled at
    SAMPLE_HZ, scored as smoothed |change|. Returns (n_people,
    n_samples) or None when the landmarker cannot run.
    """
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError:
        return None
    try:
        landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(model_path)),
                output_face_blendshapes=True,
                num_faces=1))
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    jaw = np.zeros((len(boxes), n_samples))
    step = max(1, int(round(fps / SAMPLE_HZ)))
    try:
        frame_idx = 0
        sample_idx = 0
        while sample_idx < n_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                for person, (x, y, w, h) in enumerate(boxes):
                    # Landmarks on a crop of full-size pixels: a couch
                    # face is small in the whole frame.
                    x0 = int(max(x - w / 3, 0))
                    y0 = int(max(y - h / 3, 0))
                    x1 = int(min(x + 4 * w / 3, frame_w))
                    y1 = int(min(y + 4 * h / 3, frame_h))
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    crop = cv2.cvtColor(frame[y0:y1, x0:x1],
                                        cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=crop))
                    if result.face_blendshapes:
                        jaw[person, sample_idx] = next(
                            (s.score for s in result.face_blendshapes[0]
                             if s.category_name == "jawOpen"), 0.0)
                sample_idx += 1
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()
    scores = np.array([
        np.abs(np.diff(row, prepend=row[0])) for row in jaw])
    return smooth_scores(scores, 0.8)


def _mouth_motion_scores(video_path, boxes, fps, n_samples):
    """
    Fallback path (no FaceLandmarker model): per-person mouth-region
    frame difference. The mouth lives in the lower third of the face
    box; a talking mouth moves there, a listening face does not.
    Returns (n_people, n_samples) or None when OpenCV cannot run.
    """
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    motion = np.zeros((len(boxes), n_samples))
    step = max(1, int(round(fps / SAMPLE_HZ)))
    prev_gray = None
    try:
        frame_idx = 0
        sample_idx = 0
        while sample_idx < n_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
                    for person, (x, y, w, h) in enumerate(boxes):
                        # Mouth region: lower third of the face box.
                        x0 = int(max(x, 0))
                        y0 = int(max(y + h * 0.6, 0))
                        x1 = int(min(x + w, frame_w))
                        y1 = int(min(y + h, frame_h))
                        if x1 - x0 < 4 or y1 - y0 < 4:
                            continue
                        motion[person, sample_idx] = float(
                            diff[y0:y1, x0:x1].mean()) / 255.0
                prev_gray = gray
                sample_idx += 1
            frame_idx += 1
    finally:
        cap.release()
    return smooth_scores(motion, 0.8)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def analyze_speakers(video_path, models_dir="models", sample_fps=6):
    """
    Analyse a clip and produce speaker data for the camera-switch /
    split-screen layouts.

    Returns a dict (see the module docstring) or None when unsure
    (fewer than 2 people, lips barely moving, or the video cannot be
    read) — the caller keeps its fallback in that case. Never raises
    for analysis failures; it logs and returns None instead.
    """
    from database import log_event

    def _fail(reason):
        log_event("INFO", "SpeakerFraming",
                  f"No speaker data for {video_path}: {reason}")
        return None

    if not os.path.exists(video_path):
        return _fail("file not found")
    try:
        from speaker_tracker import SpeakerTracker
        tracker = SpeakerTracker()
        try:
            face_samples = tracker.analyze_face_boxes(
                str(video_path), sample_fps=sample_fps)
        finally:
            tracker.close()
    except Exception as e:
        return _fail(f"face sampling failed ({e})")

    people = cluster_people(face_samples)
    if len(people) < 2:
        return _fail(f"found {len(people)} person(s), need at least 2")
    names = [f"speaker_{i + 1}" for i in range(len(people))]

    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    except Exception as e:
        return _fail(f"cannot read video properties ({e})")
    if raw_w <= 0 or raw_h <= 0 or total_frames <= 0:
        return _fail("video has no frames")

    n_samples = max(1, int(total_frames / fps * SAMPLE_HZ))
    method = "fallback"
    scores = None
    if landmarker_enabled():
        model_path = ensure_landmarker_model(models_dir)
        if model_path is not None:
            scores = _jawopen_scores(video_path, people, fps, n_samples,
                                     model_path)
            if scores is not None:
                method = "landmarker"
    if scores is None:
        scores = _mouth_motion_scores(video_path, people, fps, n_samples)
    if scores is None:
        return _fail("no lip-motion measurement available (need OpenCV)")

    # Gate by speech loudness: a listener's grin must not steal the crop.
    gate = speech_loudness_mask(video_path, n_samples)
    scores = scores * gate[np.newaxis, :]

    active = (scores.max(axis=0) > SPEECH_MOTION).mean()
    if active < MIN_SPEECH_COVERAGE:
        return _fail(f"lips barely move (active {active:.0%} of samples)")

    labels, split_spans = label_speakers(scores)
    segments = segments_from_labels(labels, names=names)
    trajectories = {name: trajectory_for_box(box, raw_w, raw_h)
                    for name, box in zip(names, people)}
    split_speakers = [{"speaker": name, **trajectories[name]}
                      for name in names[:2]]
    log_event("INFO", "SpeakerFraming",
              f"{len(people)} people, {len(segments)} turns, "
              f"{len(split_spans)} split spans ({method})")
    return {
        "speakers": names,
        "speaker_segments": segments,
        "trajectories": trajectories,
        "split_speakers": split_speakers,
        "split_spans": [(round(s / SAMPLE_HZ, 2), round(e / SAMPLE_HZ, 2),
                         tuple(names[i] for i in pair))
                        for s, e, pair in split_spans],
        "method": method,
    }
