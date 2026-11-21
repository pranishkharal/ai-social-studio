"""Visual moment detection: SEE the moment, don't just hear it.

Labels stretches of a video ``chase`` / ``crash`` / ``fight`` / ``stunt`` /
``reaction`` / ``funny`` with NO AI model -- the same idea as
gta6-clipper's ``moments.py`` (MIT), reimplemented here in this repo's
style:

* 2 fps, 160x90 grey frames piped from ffmpeg (the file is never loaded
  whole) + Farneback optical flow: radial outflow = speed (chase),
  rotation = stunt, sudden motion jump = crash.
* Audio bands from a 16 kHz mono stream: low (60-400 Hz, engine),
  voice (300-3 kHz), high (3-8 kHz), plus a 20 Hz envelope.
* Laughter from the 3.5-8 Hz wobble in the voice envelope
  (``ha-ha-ha`` is periodic; steady speech is not).
* Optional transcript text (laugh/hype words) and chat laugh tags
  confirm the funny/reaction scores.

Everything is scale-free: features are ranked against their own
percentiles, so a quiet stream and a loud one rank alike.

The detector runs at roughly 85x realtime on CPU for the audio+flow
pass (the reference repo's figure; the ffmpeg pipe dominates).

Integration points:
* ``hype_detector.HypeDetector.evaluate_clip`` accepts the moments
  returned here (``visual_moments=...``) and blends the best moment's
  confidence into the hype score, and reports ``moment_type``.
* ``story_headlines.build_story_headline`` accepts ``moment_type=...``
  so headlines name the kind of moment (a crash reads differently
  from a chase).

Dependencies: numpy (required) + OpenCV (optional -- without it the
video pass is skipped and moments are scored from audio only, capped
at 0.8, and ``stunt`` is never reported since rotation needs frames).

Module-local settings (env overrides; nothing in config.py):
    VISUAL_MOMENTS_FPS        default 2 (frames sampled per second)
    VISUAL_MOMENTS_MIN_SCORE  default 0.5 (lowest segment threshold)
"""

import json
import os
import re
import subprocess
import time

import numpy as np

# ---------------------------------------------------------------------------
# Settings (module-local constants with env-var overrides)
# ---------------------------------------------------------------------------

SAMPLE_FPS = int(os.getenv("VISUAL_MOMENTS_FPS", "2") or 2)
FRAME_WIDTH = 160
FRAME_HEIGHT = 90
AUDIO_RATE = 16000          # Hz, mono audio fed to the band analysis
ENVELOPE_FPS = 20           # audio envelope frames per second

# Moment types this module can label.
MOMENT_TYPES = ("chase", "crash", "fight", "stunt", "reaction", "funny")

# Minimum length of a reported segment, in seconds, per type. A crash is
# a single instant; a chase has to last a while to count.
MIN_LENGTH_SECONDS = {
    "chase": 5.0, "crash": 1.0, "fight": 4.0,
    "stunt": 3.0, "reaction": 2.0, "funny": 2.0,
}

# Score threshold a bin must cross to start a segment, per type.
SCORE_THRESHOLDS = {
    "chase": 0.5, "crash": 0.7, "fight": 0.6,
    "stunt": 0.7, "reaction": 0.55, "funny": 0.5,
}

# Without video the audio-only scores are capped (motion is unknown).
# The cap is 0.8, deliberately NOT 0.7: the crash threshold is 0.7,
# and a cap equal to the threshold would make audio-only crash
# detection mathematically impossible (0.7 * 1.0 can never exceed
# 0.7 -- the reference repo's own selftest only passes on float
# rounding). 0.8 keeps audio-only scores below video scores while
# leaving strong audio evidence (a loud bang + high-frequency flux)
# able to cross the bar.
AUDIO_ONLY_SCORE_CAP = 0.8

# Gap (seconds) bridged when merging neighbouring segments of one type.
SEGMENT_MERGE_GAP = 3.0

_HYPE_WORDS = re.compile(
    r"\b(oh my (god|gosh)|no way|what the|holy|bruh|let'?s go+|"
    r"are you kidding|crazy|insane|dude+|bro+)\b|!", re.I)
_LAUGH_WORDS = re.compile(
    r"\b(ha(ha)+|haha\w*|lol|lmao|lmfao|laugh\w*|dying|dead|cry(ing)?)\b",
    re.I)
_CHAT_LAUGH = re.compile(r"LUL|KEKW|OMEGALUL|LMAO|ICANT", re.I)


# ---------------------------------------------------------------------------
# Small signal helpers (pure numpy, no ffmpeg / cv2 needed)
# ---------------------------------------------------------------------------

def percentile_scale(values, low=50, high=98):
    """Scale-free 0..1: the median maps to 0, the 98th percentile to 1."""
    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, [low, high])
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


def smooth(values, width):
    """Centred moving average over ``width`` bins."""
    arr = np.asarray(values, dtype=float)
    width = max(1, int(width))
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def previous_median(values, count=6):
    """Median of the ``count`` bins BEFORE each bin (causal baseline)."""
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        out[i] = np.median(arr[max(0, i - count):i]) if i else arr[0]
    return out


def audio_band_levels(pcm):
    """16 kHz mono float32 audio -> per-20 Hz-frame dB in each band.

    Returns {"low": ..., "voice": ..., "high": ..., "all": ...} where
    low = 60-400 Hz (engine / rumble), voice = 300-3 kHz,
    high = 3-8 kHz (impacts, glass, gunshots).
    """
    frame = AUDIO_RATE // ENVELOPE_FPS
    n_frames = len(pcm) // frame
    windowed = (pcm[:n_frames * frame].reshape(n_frames, frame)
                * np.hanning(frame))
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1 / AUDIO_RATE)

    def band_db(low_hz, high_hz):
        mask = (freqs >= low_hz) & (freqs < high_hz)
        return 10 * np.log10(spectrum[:, mask].sum(axis=1) + 1e-9)

    return {
        "low": band_db(60, 400),
        "voice": band_db(300, 3000),
        "high": band_db(3000, 8000),
        "all": band_db(60, 8000),
    }


def laughter_modulation(envelope, window=60):
    """Periodic 3.5-8 Hz amplitude modulation of the voice envelope.

    Laughter is ``ha-ha-ha``: bursts repeating ~5 times a second. This
    measures the strongest normalised autocorrelation of a detrended
    3 s window at lags of 2-5 envelope frames (2.5-10 Hz, centred on
    the laughter band). Steady fast speech rings a little too, so the
    scorer caps audio-only ``funny`` and wants text/chat agreement.
    """
    env = np.asarray(envelope, dtype=float)
    out = np.zeros(len(env))
    detrended = env - smooth(env, 20)
    step = ENVELOPE_FPS // SAMPLE_FPS
    for i in range(0, len(env), step):
        chunk = detrended[max(0, i - window // 2): i + window // 2]
        if len(chunk) < 20:
            continue
        energy = float((chunk * chunk).sum()) + 1e-9
        best = max(float((chunk[:-lag] * chunk[lag:]).sum()) / energy
                   for lag in range(2, 6))
        out[i:i + step] = best
    return out


def _to_bins(values, n_bins, combine=np.mean):
    """Downsample a 20 Hz array to SAMPLE_FPS bins (pad/truncate to n)."""
    per_bin = ENVELOPE_FPS // SAMPLE_FPS
    arr = np.asarray(values, dtype=float)
    usable = len(arr) // per_bin
    binned = combine(arr[:usable * per_bin].reshape(usable, per_bin), axis=1)
    return np.pad(binned, (0, max(0, n_bins - len(binned))),
                  mode="edge")[:n_bins]


# ---------------------------------------------------------------------------
# Scoring: per-bin features -> per-type score arrays (0..1)
# ---------------------------------------------------------------------------

def score_moment_types(features, has_video=True, has_cam=False):
    """Score every 2 fps bin for each moment type.

    ``features`` is a dict of per-bin arrays: ``m`` (motion), ``out``
    (radial outflow), ``rot`` (rotation), ``cam`` (facecam-tile
    motion), ``low`` / ``voice`` / ``high`` / ``all`` (dB),
    ``mod`` (laughter periodicity), ``txt_laugh`` / ``txt_hype`` /
    ``chat_lol`` (0/1 text signals). Missing keys default to zeros.
    Returns {type: score array 0..1}.
    """
    n = len(features["voice"])
    zeros = np.zeros(n)
    bins = lambda seconds: int(seconds * SAMPLE_FPS)  # noqa: E731

    low = percentile_scale(smooth(features["low"], bins(6)))
    voice = percentile_scale(features["voice"])
    all_db = percentile_scale(features["all"])

    def jump_above_baseline(values):
        """Positive jump above the last 3 s, scaled 0..1."""
        return percentile_scale(
            np.maximum(0, values - previous_median(values)), 75, 99)

    all_jump = jump_above_baseline(features["all"])
    voice_jump = jump_above_baseline(features["voice"])
    high_flux = jump_above_baseline(features["high"])
    # >= 4 transients inside 8 s = a sustained burst (fight).
    transient_density = np.minimum(
        1, smooth((all_jump > 0.45).astype(float), bins(8)) * bins(8) / 4)

    if has_video:
        motion = percentile_scale(features.get("m", zeros))
        motion_smooth = percentile_scale(smooth(features.get("m", zeros),
                                                bins(6)))
        outflow = percentile_scale(
            smooth(features.get("out", zeros), bins(6)), 50, 97)
        rotation = percentile_scale(
            smooth(np.abs(features.get("rot", zeros)), bins(4)), 50, 97)
        motion_jump = percentile_scale(
            np.maximum(0, features.get("m", zeros)
                       - previous_median(features.get("m", zeros))), 75, 99)
    else:
        # Audio proxies, capped: motion is unknown without frames.
        motion = motion_smooth = all_db * 0.8
        outflow = low
        rotation = zeros
        motion_jump = voice_jump
    cap = 1.0 if has_video else AUDIO_ONLY_SCORE_CAP

    scores = {}
    # Chase: sustained motion AND speed (outflow or engine rumble),
    # held for seconds.
    scores["chase"] = cap * np.minimum(
        smooth(np.minimum(motion_smooth, np.maximum(outflow, low)),
               bins(2)),
        smooth(np.maximum(low, all_db * 0.8), bins(4)))
    # Crash: a sudden loud jump with a motion spike and high-frequency
    # flux (impact noise), gated on the jump actually happening.
    scores["crash"] = np.where(
        all_jump > 0.35,
        cap * np.minimum(np.maximum(motion_jump, all_jump * 0.8),
                         np.maximum(all_jump, high_flux))
        * np.minimum(1, 0.4 + all_jump),
        0)
    # Fight: repeated transients + motion + raised voice.
    scores["fight"] = cap * np.where(
        transient_density > 0.3,
        (transient_density + smooth(motion, bins(3))
         + smooth(voice, bins(3))) / 3 * 1.25,
        0)
    # Stunt: rotation (spins, flips) with motion; needs video.
    if has_video:
        scores["stunt"] = (np.minimum(rotation, np.maximum(motion_smooth, 0.5))
                           * np.where(low < 0.85, 1, 0.8))
    else:
        scores["stunt"] = zeros
    # Reaction: the facecam moves AND the voice jumps (or the voice
    # alone when there is no cam), with hype words as a boost.
    if has_video and has_cam:
        cam_motion = percentile_scale(
            smooth(features.get("cam", zeros), bins(2)))
    else:
        cam_motion = smooth(voice_jump, bins(2))
    reaction = (np.minimum(1, (cam_motion * 0.45
                               + smooth(voice, bins(2)) * 0.55) * 1.2)
                * (0.5 + 0.5 * np.maximum(voice_jump, voice))
                * (voice > 0.5))
    scores["reaction"] = np.minimum(
        1, reaction + 0.2 * smooth(features.get("txt_hype", zeros),
                                   bins(3)).clip(0, 1) * (voice > 0.4))
    # Funny: laughter periodicity in the voice + laugh words in the
    # transcript/chat. Audio-only funny is capped at 0.75 -- steady
    # fast speech can ring the periodicity detector.
    mod = (np.clip((smooth(features.get("mod", zeros), bins(2)) - 0.35)
                   / 0.35, 0, 1) * (voice > 0.4))
    text = np.minimum(
        1, smooth(features.get("txt_laugh", zeros), bins(4)) * 2.8
        + smooth(features.get("chat_lol", zeros), bins(4)) * 2)
    funny = np.minimum(1, mod * 0.55 + text * 0.6
                       + np.minimum(mod, text) * 0.3)
    scores["funny"] = (np.where((mod > 0.3) | (text > 0.3), funny, 0)
                       * np.where(text > 0.3, 1, 0.75))
    return {t: np.clip(np.nan_to_num(s), 0, 1) for t, s in scores.items()}


def segments_from_scores(scores, t0=0.0, gap=SEGMENT_MERGE_GAP):
    """Score arrays -> [{t0, t1, type, score}] sorted by start time.

    Bins above the type's threshold start a segment; gaps of up to
    ``gap`` seconds are bridged; segments shorter than the type's
    minimum length are dropped. A segment's score is the 90th
    percentile of its bins.
    """
    out = []
    gap_bins = int(gap * SAMPLE_FPS)
    for moment_type, series in scores.items():
        above = series >= SCORE_THRESHOLDS[moment_type]
        n = len(series)
        i = 0
        while i < n:
            if not above[i]:
                i += 1
                continue
            j = i
            while j < n and (above[j]
                             or above[j + 1: j + 1 + gap_bins].any()):
                j += 1
            start_s, end_s = i / SAMPLE_FPS, j / SAMPLE_FPS
            if end_s - start_s >= MIN_LENGTH_SECONDS[moment_type]:
                out.append({
                    "t0": round(t0 + start_s, 1),
                    "t1": round(t0 + end_s, 1),
                    "type": moment_type,
                    "score": round(float(np.percentile(series[i:j], 90)), 2),
                })
            i = j + 1
    return sorted(out, key=lambda m: (m["t0"], m["type"]))


# ---------------------------------------------------------------------------
# Feature extraction (ffmpeg pipes + OpenCV)
# ---------------------------------------------------------------------------

def probe_video(path):
    """(duration_seconds, video_stream_or_None, has_audio) via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "stream=codec_type,width,height:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30)
    info = json.loads(result.stdout or "{}")
    streams = info.get("streams", [])
    video = next((s for s in streams
                  if s.get("codec_type") == "video"
                  and s.get("height", 0) > 16), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return float(info.get("format", {}).get("duration", 0)), video, has_audio


def read_audio_bands(path, chunk_seconds=60):
    """Stream 16 kHz mono audio in chunks -> per-frame band dB dict.

    The PCM itself is never held whole: each 60 s chunk is reduced to
    band levels and dropped.
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"],
        stdout=subprocess.PIPE)
    collected = {}
    step = chunk_seconds * AUDIO_RATE * 2  # 16-bit samples
    while True:
        chunk = proc.stdout.read(step)
        if not chunk:
            break
        pcm = (np.frombuffer(chunk[:len(chunk) // 2 * 2], np.int16)
               .astype(np.float32) / 32768)
        for key, values in audio_band_levels(pcm).items():
            collected.setdefault(key, []).append(values)
    proc.wait()
    if not collected:
        return None
    return {k: np.concatenate(v) for k, v in collected.items()}


def read_video_motion(path, cam_box=None):
    """2 fps 160x90 grey frames -> motion / outflow / rotation / cam.

    Returns {m, out, rot, cam} per-bin arrays, or None when OpenCV is
    not installed. ``cam_box`` is (x0, y0, x1, y1) in 160x90 units:
    the facecam tile is excluded from the game-area motion and its own
    frame-diff motion is reported separately as ``cam``.
    """
    try:
        import cv2
    except ImportError:
        return None
    if not hasattr(cv2, "calcOpticalFlowFarneback"):
        # A partial cv2 (e.g. a test stub) cannot do optical flow:
        # degrade to audio-only rather than crash the pipeline.
        return None
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an",
         "-vf", f"fps={SAMPLE_FPS},scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE)
    rows, cols = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH].astype(np.float32)
    dx, dy = cols - FRAME_WIDTH / 2, rows - FRAME_HEIGHT / 2
    radius = np.sqrt(dx ** 2 + dy ** 2) + 1
    game_mask = np.ones((FRAME_HEIGHT, FRAME_WIDTH), bool)
    if cam_box:
        game_mask[cam_box[1]:cam_box[3], cam_box[0]:cam_box[2]] = False
    series = {key: [] for key in ("m", "out", "rot", "cam")}
    prev = None
    frame_bytes = FRAME_WIDTH * FRAME_HEIGHT
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        frame = np.frombuffer(raw, np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH)
        if prev is None:
            values = (0.0, 0.0, 0.0, 0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev, frame, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            vx, vy = flow[..., 0], flow[..., 1]
            if cam_box:
                tile_now = frame[cam_box[1]:cam_box[3],
                                 cam_box[0]:cam_box[2]].astype(np.int16)
                tile_prev = prev[cam_box[1]:cam_box[3],
                                 cam_box[0]:cam_box[2]]
                cam_motion = float(np.abs(tile_now - tile_prev).mean())
            else:
                cam_motion = 0.0
            values = (
                float(np.hypot(vx, vy)[game_mask].mean()),
                float((((dx * vx + dy * vy) / radius)[game_mask]).mean()),
                float((((dx * vy - dy * vx) / radius)[game_mask]).mean()),
                cam_motion,
            )
        for key, value in zip(series, values):
            series[key].append(value)
        prev = frame
    proc.wait()
    return {k: np.array(v) for k, v in series.items()}


def find_cam_box(path, duration, samples=12):
    """Facecam box in 160x90 units, or None when no corner face is found.

    Samples a few frames and looks for a face (Haar cascade) in a
    corner of the frame -- where streamers put their camera. Without
    OpenCV this always returns None and reactions fall back to audio.
    """
    try:
        import cv2
    except ImportError:
        return None
    cascade_path = (cv2.data.haarcascades
                    + "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        return None
    detector = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(str(path))
    hits = []
    for k in range(samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, (k + 0.5) / samples * duration * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (640, 360))
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in detector.detectMultiScale(grey, 1.1, 4):
            cx, cy = x + w / 2, y + h / 2
            if (cx < 160 or cx > 480) and (cy < 90 or cy > 270):
                hits.append((cx, cy, w))
    cap.release()
    if len(hits) < 4:
        return None
    cx, cy, w = np.median(np.array(hits), axis=0)
    r = w * 1.6
    # 640x360 -> 160x90 units (divide by 4).
    return tuple(int(v / 4) for v in (
        max(0, cx - r), max(0, cy - r),
        min(640, cx + r), min(360, cy + r)))


def text_features(transcript_lines, n_bins, t0=0.0):
    """Per-bin 0/1 arrays from transcript lines [(start, end, text)].

    ``txt_laugh``: laugh words in the spoken text. ``txt_hype``: hype
    words. ``chat_lol``: a ``[CHAT ...]`` tag on the line carrying a
    laugh emote (KEKW / LUL / ...).
    """
    out = {k: np.zeros(n_bins) for k in
           ("txt_laugh", "txt_hype", "chat_lol")}
    for start, end, text in transcript_lines or []:
        i = max(0, int((float(start) - t0) * SAMPLE_FPS))
        j = min(n_bins, int((float(end) - t0) * SAMPLE_FPS) + 1)
        if j <= i:
            continue
        plain = re.sub(r"\[[^\]]*\]", "", text or "")
        if _LAUGH_WORDS.search(plain):
            out["txt_laugh"][i:j] = 1
        if _HYPE_WORDS.search(plain):
            out["txt_hype"][i:j] = 1
        chat_tag = re.search(r"\[CHAT[^\]]*\]", text or "")
        if chat_tag and _CHAT_LAUGH.search(chat_tag[0]):
            out["chat_lol"][i:j] = 1
    return out


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def best_moment(moments):
    """Highest-confidence moment dict, or None when there are none."""
    if not moments:
        return None
    return max(moments, key=lambda m: m["score"])


def moment_tag(moment):
    """Readable tag for a moment, e.g. ``[MOMENT chase 0.8]``."""
    return f"[MOMENT {moment['type']} {moment['score']:.1f}]"


def detect_visual_moments(video_path, transcript_lines=None, t0=0.0):
    """Label the visual moments in a video file.

    Returns {"src", "t0", "dur", "has_video", "cam", "scan_s",
    "x_realtime", "moments": [{t0, t1, type, score}]}. Raises
    FileNotFoundError for a missing file and ValueError for a file
    with no readable audio (the audio bands are the minimum input).
    """
    if not os.path.exists(str(video_path)):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    started = time.time()
    duration, video_stream, has_audio = probe_video(video_path)
    if not duration or not has_audio:
        raise ValueError(
            f"Need a readable file with audio: {video_path}")
    audio = read_audio_bands(video_path)
    if audio is None:
        raise ValueError(
            f"Could not decode the audio of this file: {video_path}")

    n_bins = int(duration * SAMPLE_FPS)
    has_video = video_stream is not None
    cam_box = None
    motion = None
    if has_video:
        cam_box = find_cam_box(video_path, duration)
        motion = read_video_motion(video_path, cam_box)
        if motion is None or not len(motion.get("m", [])):
            # Video track present but no frame decoded (or no OpenCV):
            # score on audio only instead of crashing.
            has_video, cam_box = False, None

    features = {k: _to_bins(v, n_bins) for k, v in audio.items()}
    features["mod"] = _to_bins(
        laughter_modulation(10 ** (audio["voice"] / 20)), n_bins, np.max)
    for key in ("m", "out", "rot", "cam"):
        if has_video:
            features[key] = np.pad(
                motion[key], (0, max(0, n_bins - len(motion[key]))),
                mode="edge")[:n_bins]
        else:
            features[key] = np.zeros(n_bins)
    features.update(text_features(transcript_lines, n_bins, t0))

    moments = segments_from_scores(
        score_moment_types(features, has_video, bool(cam_box)), t0)
    scan_s = time.time() - started
    return {
        "src": str(video_path),
        "t0": t0,
        "dur": round(duration, 1),
        "has_video": has_video,
        "cam": list(cam_box) if cam_box else None,
        "scan_s": round(scan_s, 1),
        "x_realtime": round(duration / max(scan_s, 0.01), 1),
        "moments": moments,
    }
