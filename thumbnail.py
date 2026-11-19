"""
Auto-thumbnails for ai-social-studio clips (feature 10).

Picks the best frame from a rendered clip, overlays the hook text in bold
high-contrast type, and saves a JPG next to the clip. Used by
ClipEditor.edit_clip (and useful for YouTube Shorts + the dashboard clip
list, which currently have no thumbnails at all).

The frame is grabbed from the pre-overlay source clip when the caller
provides one (see generate_thumbnail's frame_source_path): the rendered
clip already has the white hook banner burned in, so the thumbnail's
own text box must never be drawn on top of it — one frame, one text box.
That source frame is 16:9 while the clip is 9:16, so the thumbnail is
cropped to 9:16 first, centred on the streamer's face when one is
detected (else centred) — the thumbnail stays portrait like the clip.

Frame selection, in priority order:
    1. ``peak_time`` argument, when the caller already knows it.
    2. Peak-audio moment: ffmpeg ``astats`` RMS levels are measured over
       1-second windows and the loudest window wins (screams, laughter
       and big reactions are usually the most clickable frame).
    3. Largest detected face: frames are sampled and scored with the
       OpenCV Haar cascades in ``models/`` (same cascades speaker_tracker
       uses). Skipped when OpenCV is not installed.
    4. Word-density midpoint when Whisper words are available, else the
       middle frame of the clip.

Only Pillow + ffmpeg are required. OpenCV is optional.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ASTATS_WINDOW_SECONDS = 1.0
FACE_SAMPLE_COUNT = 5


def _creationflags():
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _run(cmd, timeout=120):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        creationflags=_creationflags(),
    )


def probe_duration(video_path):
    """Clip duration in seconds via ffprobe, or None if unavailable."""
    try:
        res = _run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video_path),
        ], timeout=30)
        if res.returncode == 0:
            return float(res.stdout.strip())
    except Exception:
        pass
    return None


def parse_astats_rms_by_window(stderr_text, window_seconds=ASTATS_WINDOW_SECONDS):
    """
    Parse ffmpeg astats output into {window_start: max RMS level dB}.

    Expected input is the stderr of:
        ffmpeg -i <video> -vn -af
          "astats=metadata=1:reset=1,ametadata=mode=print:
           key=lavfi.astats.Overall.RMS_level" -f null -

    ametadata prints one block per audio frame:
        frame:12 pts:12345 pts_time:0.278
        lavfi.astats.Overall.RMS_level=-23.456789
    RMS level dB is negative (0 dB = full scale); "-inf" means silence.
    Returns an empty dict when nothing parses.
    """
    rms_by_window = {}
    current_time = None
    pts_re = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
    rms_re = re.compile(r"lavfi\.astats\.Overall\.RMS_level=(-inf|[0-9\-\.]+)")

    for line in (stderr_text or "").splitlines():
        pts_match = pts_re.search(line)
        if pts_match:
            try:
                current_time = float(pts_match.group(1))
            except ValueError:
                current_time = None
            continue
        rms_match = rms_re.search(line)
        if rms_match and current_time is not None:
            raw = rms_match.group(1)
            if raw == "-inf":
                continue
            try:
                rms = float(raw)
            except ValueError:
                continue
            window = int(current_time // window_seconds) * window_seconds
            rms_by_window[window] = max(rms, rms_by_window.get(window, float("-inf")))
    return rms_by_window


def estimate_peak_time(video_path, window_seconds=ASTATS_WINDOW_SECONDS):
    """
    Estimate the loudest moment of a clip (seconds) via ffmpeg astats.

    Returns the midpoint of the loudest 1-second window, or None when the
    clip has no audio track, ffmpeg fails, or no RMS data parses.
    """
    if not os.path.exists(video_path):
        return None
    try:
        res = _run([
            "ffmpeg", "-i", str(video_path), "-vn",
            "-af",
            "astats=metadata=1:reset=1,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
            "-f", "null", "-",
        ])
    except Exception:
        return None
    if res.returncode != 0:
        return None
    rms_by_window = parse_astats_rms_by_window(res.stderr, window_seconds)
    if not rms_by_window:
        return None
    loudest_window = max(rms_by_window, key=lambda w: rms_by_window[w])
    return round(loudest_window + window_seconds / 2.0, 2)


def _grab_frame(video_path, timestamp, output_path):
    """Extract one frame with ffmpeg. Raises RuntimeError on failure."""
    res = _run([
        "ffmpeg", "-y", "-ss", str(max(0.0, timestamp)),
        "-i", str(video_path), "-frames:v", "1", "-q:v", "2",
        str(output_path),
    ])
    if res.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg frame grab failed: {res.stderr[-300:]}")
    return output_path


def _largest_face_time(video_path, duration):
    """
    Timestamp of the sampled frame with the largest detected face.

    Samples FACE_SAMPLE_COUNT evenly spaced frames and scores them with
    the Haar frontal-face cascade from models/. Returns None when OpenCV
    is not installed, the cascade is missing, no face is found, or the
    duration is unknown.
    """
    if not duration or duration <= 0:
        return None
    try:
        import cv2
    except ImportError:
        return None

    cascade_path = Path(__file__).resolve().parent / "models" / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        cascade_path = Path("models") / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None

    # A broken or incomplete OpenCV (missing attributes, unreadable video)
    # is the same as no OpenCV for this best-effort helper: skip gracefully
    # instead of failing thumbnail generation.
    try:
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if face_cascade.empty():
            return None

        best_time = None
        best_area = 0
        with tempfile.TemporaryDirectory(prefix="thumb_faces_") as tmp_dir:
            for i in range(FACE_SAMPLE_COUNT):
                t = duration * (i + 0.5) / FACE_SAMPLE_COUNT
                frame_path = os.path.join(tmp_dir, f"frame_{i}.jpg")
                try:
                    _grab_frame(video_path, t, frame_path)
                except Exception:
                    continue
                img = cv2.imread(frame_path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30)
                )
                for (x, y, w, h) in faces:
                    if w * h > best_area:
                        best_area = w * h
                        best_time = t
        return best_time
    except Exception:
        return None


def _densest_speech_time(words):
    """
    Midpoint of the 1-second window containing the most Whisper words.
    Returns None when there are no usable word timestamps.
    """
    counts = {}
    for w in words or []:
        try:
            start = float(w["start"])
        except (KeyError, TypeError, ValueError):
            continue
        window = int(start // ASTATS_WINDOW_SECONDS)
        counts[window] = counts.get(window, 0) + 1
    if not counts:
        return None
    best_window = max(counts, key=lambda k: counts[k])
    return round(best_window * ASTATS_WINDOW_SECONDS + ASTATS_WINDOW_SECONDS / 2.0, 2)


def pick_thumbnail_time(video_path, words=None, peak_time=None):
    """
    Choose the thumbnail timestamp (seconds) using the documented
    priority order: explicit peak_time > peak audio > largest face >
    densest speech > middle frame. Falls back to 1.0s when nothing else
    is measurable.
    """
    if peak_time is not None:
        return float(peak_time)

    peak = estimate_peak_time(video_path)
    if peak is not None:
        return peak

    duration = probe_duration(video_path)
    face_time = _largest_face_time(video_path, duration)
    if face_time is not None:
        return face_time

    speech_time = _densest_speech_time(words)
    if speech_time is not None:
        return speech_time

    if duration:
        return round(duration / 2.0, 2)
    return 1.0


def _largest_face_box(frame_path):
    """Bounding box (x, y, w, h) of the largest face in a frame file.

    Best-effort: None when OpenCV or the Haar cascade is missing, or
    no face is found. Used to centre the 9:16 thumbnail crop on the
    streamer (x) and to keep the hook-text band off their face (y).
    """
    try:
        import cv2
    except ImportError:
        return None

    cascade_path = Path(__file__).resolve().parent / "models" / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        cascade_path = Path("models") / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None

    try:
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if face_cascade.empty():
            return None
        img = cv2.imread(str(frame_path))
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (int(x), int(y), int(w), int(h))
    except Exception:
        return None


def _crop_to_vertical(image, center_x=None):
    """Crop a landscape frame to 9:16; portrait frames pass through.

    The crop window is centred on ``center_x`` (the streamer's face)
    when given, else on the frame centre, clamped to the frame edges
    so the window never runs off the sides.
    """
    width, height = image.size
    target_w = int(round(height * 9 / 16))
    if width <= target_w:
        return image
    cx = width // 2 if center_x is None else center_x
    left = max(0, min(width - target_w, cx - target_w // 2))
    return image.crop((left, 0, left + target_w, height))


def _load_bold_font(size):
    """
    Cross-platform bold font lookup (Windows, macOS, Linux — see
    platform_utils.load_font). Never crashes: falls back to Pillow's
    built-in default when no bold font is installed.
    """
    from platform_utils import load_font
    return load_font(size, bold=True)


def _overlay_hook_text(image, hook_text, min_y=None):
    """Draw bold, high-contrast wrapped hook text over the frame.

    ``min_y`` (pixels) pushes the text band down: when a face was
    detected at the top of the frame (the streamer's facecam), the
    band starts below the face instead of covering it. Ignored when
    the band would not fit below ``min_y`` — then it stays at the top.
    """
    from PIL import ImageDraw

    if not hook_text or not hook_text.strip():
        return image

    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Scale the font to the frame: ~6% of frame width, clamped.
    font_size = max(28, min(110, int(width * 0.062)))
    font = _load_bold_font(font_size)

    # Word-wrap to ~88% of the frame width.
    max_text_width = int(width * 0.88)
    words = hook_text.strip().split()
    lines = []
    cur = []
    for word in words:
        test = " ".join(cur + [word])
        if draw.textlength(test, font=font) <= max_text_width:
            cur.append(word)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    lines = lines[:4]  # keep the thumbnail readable

    line_height = int(font_size * 1.25)
    pad_x, pad_y = 28, 20
    block_h = len(lines) * line_height + pad_y * 2
    # Top placement: clear of TikTok/Shorts bottom UI, visible in feeds.
    box_y = int(height * 0.06)
    if min_y is not None and min_y + block_h <= int(height * 0.85):
        # Keep the band off the detected face when there is room.
        box_y = max(box_y, int(min_y))
    box_x = int(width * 0.06)
    box_w = width - 2 * box_x

    # Semi-transparent black band behind the text for contrast.
    overlay = image.convert("RGBA")
    band = ImageDraw.Draw(overlay, "RGBA")
    band.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + block_h],
        radius=18, fill=(0, 0, 0, 170),
    )
    image = overlay.convert("RGB")
    draw = ImageDraw.Draw(image)

    y = box_y + pad_y
    for line in lines:
        line_w = draw.textlength(line, font=font)
        x = box_x + (box_w - line_w) // 2
        # White text with a thick black stroke: readable on any frame.
        draw.text((x, y), line, font=font, fill=(255, 255, 255),
                  stroke_width=max(2, font_size // 18), stroke_fill=(0, 0, 0))
        y += line_height
    return image


def generate_thumbnail(video_path, hook_text, output_path, words=None, peak_time=None,
                       frame_source_path=None):
    """
    Generate a JPG thumbnail for a clip.

    Args:
        video_path: rendered clip to grab a frame from.
        hook_text: text overlaid on the frame (may be "" for a clean frame).
        output_path: where the JPG is saved.
        words: optional Whisper word dicts, used as a fallback frame signal.
        peak_time: optional known peak moment in seconds.
        frame_source_path: optional video to grab the frame from instead
            of ``video_path``. ClipEditor passes the pre-overlay source
            clip here: the rendered clip already has the white hook
            banner burned in, so drawing the thumbnail's own hook-text
            box on top of it puts two text boxes on one frame. Grabbing
            the frame from the clean source means the thumbnail's text
            box is the only text on the thumbnail. Falls back to
            ``video_path`` when the path is missing.

    Returns:
        output_path.

    Raises:
        FileNotFoundError: video does not exist.
        RuntimeError: frame extraction failed.
    """
    from PIL import Image

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # The frame comes from the clean pre-overlay source when one is
    # given (same timeline as the render, so peak/face/speech timing
    # still applies); otherwise from the rendered clip itself.
    frame_source = frame_source_path
    if not frame_source or not os.path.exists(frame_source):
        frame_source = video_path

    timestamp = pick_thumbnail_time(frame_source, words=words, peak_time=peak_time)

    with tempfile.TemporaryDirectory(prefix="thumb_") as tmp_dir:
        frame_path = os.path.join(tmp_dir, "frame.jpg")
        _grab_frame(frame_source, timestamp, frame_path)
        face_box = _largest_face_box(frame_path)
        image = Image.open(frame_path).convert("RGB")

    # The clean source frame is 16:9 but the clip (and its thumbnail)
    # is 9:16: crop to portrait, centred on the streamer's face when
    # one was found. A frame grabbed from the rendered clip is already
    # 9:16, so the crop is a no-op there.
    face_cx = None
    text_min_y = None
    if face_box is not None:
        face_cx = face_box[0] + face_box[2] // 2
        # Keep the hook-text band below the facecam: the crop is
        # full-height, so the face's bottom edge is unchanged by it.
        text_min_y = face_box[1] + face_box[3] + 24
    image = _crop_to_vertical(image, face_cx)
    image = _overlay_hook_text(image, hook_text, min_y=text_min_y)

    parent = Path(output_path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, "JPEG", quality=90)
    return output_path
