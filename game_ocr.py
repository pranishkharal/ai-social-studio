"""Game-event detection from the screen: read the big banner text.

Reads big centred on-screen text -- WASTED, BUSTED, MISSION PASSED /
FAILED, VICTORY, GAME OVER and friends -- the same idea as
gta6-clipper's ``vision.py`` (MIT), reimplemented here in this repo's
style. A death, a mission pass or a win is exactly the moment a clip
wants, and it is visible even when the streamer goes quiet.

How it works:
* Frames are sampled at ~1 fps (360p-equivalent, 640 px wide) and
  OCR'd with RapidOCR (``rapidocr-onnxruntime``, OPTIONAL -- lazily
  imported; when it is missing the scan reports ``ocr_available:
  False`` and returns no events instead of crashing).
* OCR text is fuzzy-matched against the known banner phrases
  (difflib ratio >= 0.8), so an OCR slip like "BUSTEO" still counts.
* Colour cues on the centre band back the OCR up: a desaturated
  (grey) frame = death screen, gold = mission passed, red = WASTED
  text. Frames are only OCR'd when a cue fires or every 4th sampled
  frame -- OCR is slow (~1 s/frame), so this keeps a scan affordable.
* Unknown big centred ALL-CAPS text is reported as a ``text`` event
  too (other games, GTA 6 banners we have not listed yet).

Integration: ``hype_detector.HypeDetector.evaluate_clip`` accepts the
events returned here (``ocr_events=...``) and adds a bonus to the
hype score; ``ocr_score_bonus`` is the shared bonus formula.

Module-local settings (env overrides; nothing in config.py):
    GAME_OCR_FPS          default 1.0 (frames sampled per second)
    GAME_OCR_MIN_CONF     default 0.6 (min OCR confidence to match)
    GAME_OCR_MERGE_GAP_S  default 15 (same label within this = 1 event)
"""

import difflib
import os
import re
import time

import numpy as np

# label -> (phrases that count, heat weight 0-1)
BANNER_WORDS = {
    "WASTED": (["WASTED"], 0.8),
    "BUSTED": (["BUSTED"], 0.7),
    "MISSION PASSED": (["MISSION PASSED", "MISSION COMPLETE"], 0.7),
    "MISSION FAILED": (["MISSION FAILED"], 0.7),
    "HEIST PASSED": (["HEIST PASSED", "HEIST COMPLETE"], 0.8),
    "HEIST FAILED": (["HEIST FAILED"], 0.7),
    "VICTORY ROYALE": (["VICTORY ROYALE"], 0.8),
    "ELIMINATED": (["ELIMINATED", "YOU ELIMINATED"], 0.5),
    "GAME OVER": (["GAME OVER"], 0.6),
    "YOU DIED": (["YOU DIED"], 0.7),
    "VICTORY": (["VICTORY"], 0.6),
    "DEFEAT": (["DEFEAT"], 0.5),
    "CHAMPION": (["CHAMPION", "YOU ARE THE CHAMPION"], 0.6),
}

# Weights for the non-banner event kinds.
WEIGHT_UNKNOWN_TEXT = 0.35
WEIGHT_HARD_CUT = 0.3

# Fuzzy-match bar for OCR text (difflib ratio). 0.8 lets "BUSTEO"
# count as BUSTED while keeping unrelated words out.
FUZZY_RATIO = 0.8

# Max hype-score points an OCR event can add (see ocr_score_bonus).
MAX_OCR_BONUS = 15.0

OCR_FPS = float(os.getenv("GAME_OCR_FPS", "1.0") or 1.0)
OCR_MIN_CONF = float(os.getenv("GAME_OCR_MIN_CONF", "0.6") or 0.6)
MERGE_GAP_SECONDS = float(os.getenv("GAME_OCR_MERGE_GAP_S", "15") or 15)

# OCR runs on cue frames + one sweep frame every N sampled frames.
OCR_SWEEP_EVERY = 4

_ocr_engine = None
_ocr_unavailable = False


# ---------------------------------------------------------------------------
# RapidOCR (optional, lazy)
# ---------------------------------------------------------------------------

def ocr_available():
    """True when rapidocr-onnxruntime imports (checked once, cached)."""
    global _ocr_unavailable
    if _ocr_engine is not None:
        return True
    if _ocr_unavailable:
        return False
    try:
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        _ocr_unavailable = True
        return False
    return True


def _get_ocr_engine():
    """Singleton RapidOCR engine, or None when the package is missing."""
    global _ocr_engine, _ocr_unavailable
    if _ocr_engine is not None:
        return _ocr_engine
    if _ocr_unavailable:
        return None
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        _ocr_unavailable = True
        return None
    _ocr_engine = RapidOCR()
    return _ocr_engine


def run_ocr(image_bgr):
    """OCR one BGR image -> [(box, text, confidence)] (empty if no OCR)."""
    engine = _get_ocr_engine()
    if engine is None:
        return []
    result, _ = engine(image_bgr, use_cls=False)
    return result or []


# ---------------------------------------------------------------------------
# Text matching + colour cues (no OCR engine needed)
# ---------------------------------------------------------------------------

def normalize_banner_text(text):
    """Uppercase letters/spaces only: "Mission Passed!" -> "MISSION PASSED"."""
    return re.sub(r"[^A-Z ]", "", (text or "").upper()).strip()


def match_banner(text):
    """Fuzzy-match OCR text against the known banner phrases.

    Matches the whole string or any same-length token window, so a
    banner with extra words around it still counts. Returns the label
    (e.g. "BUSTED") or None.
    """
    cleaned = normalize_banner_text(text)
    if len(cleaned) < 4:
        return None
    for label, (phrases, _weight) in BANNER_WORDS.items():
        for phrase in phrases:
            n_words = len(phrase.split())
            tokens = cleaned.split()
            candidates = [cleaned] + [
                " ".join(tokens[i:i + n_words])
                for i in range(max(1, len(tokens) - n_words + 1))]
            for candidate in candidates:
                if difflib.SequenceMatcher(None, candidate, phrase).ratio() >= FUZZY_RATIO:
                    return label
    return None


def colour_cues(frame_bgr):
    """Cheap colour cues on the centre band of a BGR frame.

    Returns {"sat": mean saturation 0-1, "gold": share of gold pixels,
    "red": share of red pixels}. Grey (low saturation) = death screen,
    gold = MISSION PASSED text, red = WASTED text. Needs OpenCV.
    """
    import cv2
    height, width = frame_bgr.shape[:2]
    band = frame_bgr[int(height * 0.25):int(height * 0.75)]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    full_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return {
        "sat": float(full_hsv[..., 1].mean()) / 255,
        "gold": float((((hue > 15) & (hue < 35) & (sat > 110)
                        & (val > 170))).mean()),
        "red": float((((hue < 8) | (hue > 172)) & (sat > 140)
                      & (val > 120)).mean()),
    }


def _banner_cue(cues, recent_saturation):
    """True when the colour cues look like a banner is on screen."""
    if cues["gold"] > 0.006 or cues["red"] > 0.004:
        return True
    if recent_saturation:
        median_sat = float(np.median(recent_saturation))
        if median_sat > 0 and cues["sat"] < 0.6 * median_sat:
            return True  # frame suddenly greyed out = death screen
    return False


# ---------------------------------------------------------------------------
# Per-frame detection
# ---------------------------------------------------------------------------

def read_banners(frame_bgr, ocr_fn=None):
    """OCR the centre band -> [(label_or_None, text, conf, rel_height)].

    ``ocr_fn`` is an injectable OCR callable (image -> [(box, text,
    conf)]) so tests can run without the RapidOCR package; it defaults
    to the real engine. Only the centre band (20-80% height, 8-92%
    width) is read -- banners are big and centred.
    """
    import cv2  # noqa: F401  (kept for parity: frames are cv2 images)
    height, width = frame_bgr.shape[:2]
    y0, x0 = int(height * 0.2), int(width * 0.08)
    crop = frame_bgr[y0:int(height * 0.8), x0:int(width * 0.92)]
    ocr = ocr_fn or run_ocr
    out = []
    for box, text, conf in ocr(crop):
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        rel_height = (max(ys) - min(ys)) / height
        centre_x = (float(np.mean(xs)) + x0) / width
        label = match_banner(text) if float(conf) > OCR_MIN_CONF else None
        out.append((label, text, float(conf), rel_height, centre_x))
    return out


def detect_frame(frame_bgr, ocr_fn=None):
    """Events on one frame -> [(kind, label, extra)].

    kind "word": a known banner phrase. kind "text": unknown big
    centred ALL-CAPS text (another game's banner). A frame with
    neither returns [].
    """
    events = []
    cues = colour_cues(frame_bgr)
    for label, text, conf, rel_height, centre_x in read_banners(
            frame_bgr, ocr_fn=ocr_fn):
        if label:
            events.append(("word", label, {
                "text": text, "conf": round(conf, 2), "cues": cues}))
        elif (rel_height >= 0.06 and conf > 0.85 and 0.3 < centre_x < 0.7
              and len(normalize_banner_text(text).replace(" ", "")) >= 4
              and sum(c.isupper() for c in text)
              >= 0.8 * sum(c.isalpha() for c in text)):
            events.append(("text", "TEXT " + normalize_banner_text(text), {
                "text": text, "conf": round(conf, 2), "cues": cues}))
    return events


# ---------------------------------------------------------------------------
# Whole-video scan
# ---------------------------------------------------------------------------

def scan_video(path, t0=0.0, fps=None, ocr_fn=None):
    """Scan a video for banner events -> raw [(t, kind, label, extra)].

    Samples ~``fps`` frames per second. OCR only runs on frames where
    a colour cue fires (or every 4th sampled frame as a sweep), and
    only when the centre band actually changed since the last OCR'd
    frame -- OCR is ~1 s/frame, so this is what keeps a scan
    affordable. Hard scene cuts are reported as ("cut", "CUT") events.
    Needs OpenCV; raises ImportError when it is missing.
    """
    import cv2
    fps = OCR_FPS if fps is None else float(fps)
    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(video_fps / fps))
    raw = []
    prev_hist = None
    prev_band = None
    saturations = []
    sampled = 0
    frame_index = 0
    while True:
        if not cap.grab():
            break
        if frame_index % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if frame.shape[1] > 640:
                frame = cv2.resize(
                    frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            t = t0 + frame_index / video_fps
            hist = cv2.calcHist(
                [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None,
                [16, 8], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if (prev_hist is not None
                    and cv2.compareHist(prev_hist, hist,
                                        cv2.HISTCMP_BHATTACHARYYA) > 0.55):
                raw.append((t, "cut", "CUT", {}))
            prev_hist = hist
            band = cv2.resize(
                frame[int(frame.shape[0] * 0.2):int(frame.shape[0] * 0.8)],
                (80, 40)).astype(np.int16)
            cues = colour_cues(frame)
            saturations.append(cues["sat"])
            saturations[:] = saturations[-40:]
            sampled += 1
            cue = _banner_cue(cues, saturations)
            band_changed = (prev_band is None
                            or np.abs(band - prev_band).mean() > 6)
            if (cue or sampled % OCR_SWEEP_EVERY == 0) and band_changed:
                for kind, label, extra in detect_frame(frame, ocr_fn=ocr_fn):
                    raw.append((t, kind, label, extra))
                prev_band = band
        frame_index += 1
    cap.release()
    return raw


def merge_events(raw, gap=MERGE_GAP_SECONDS):
    """Merge raw hits into events: same label within ``gap`` s = 1 event.

    A banner word seen once at low confidence is dropped unless a
    colour cue agrees with it (grey frame for a death banner, gold
    for a pass banner). Each event gets {"t", "t_end", "kind",
    "label", "hits", "weight"}.
    """
    events = []
    last_by_label = {}
    for t, kind, label, extra in sorted(raw, key=lambda r: r[0]):
        if label in last_by_label and t - last_by_label[label]["t_end"] <= gap:
            last_by_label[label]["t_end"] = round(t, 1)
            last_by_label[label]["hits"] += 1
            continue
        event = {"t": round(t, 1), "t_end": round(t, 1), "kind": kind,
                 "label": label, "hits": 1,
                 **{k: v for k, v in extra.items()}}
        if kind == "word":
            event["weight"] = BANNER_WORDS[label][1]
        else:
            event["weight"] = (WEIGHT_UNKNOWN_TEXT if kind == "text"
                               else WEIGHT_HARD_CUT)
        events.append(event)
        last_by_label[label] = event
    kept = []
    for event in events:
        if (event["kind"] == "word" and event["hits"] == 1
                and event.get("conf", 1.0) < 0.8):
            cues = event.get("cues") or {}
            agrees = (
                (cues.get("sat", 1.0) < 0.3 and event["label"] in
                 ("WASTED", "BUSTED", "YOU DIED", "GAME OVER", "DEFEAT"))
                or (cues.get("gold", 0.0) > 0.006 and "PASSED" in event["label"])
                or (cues.get("red", 0.0) > 0.004 and event["label"] == "WASTED"))
            if not agrees:
                continue  # single low-confidence hit, no colour backup
        kept.append(event)
    return kept


def ocr_score_bonus(events):
    """Hype-score bonus (0-15) for a list of merged OCR events.

    The strongest event sets the bonus (weight 0.8 -> 12 points);
    extra distinct banners add a little more, capped at 15.
    """
    if not events:
        return 0.0
    weights = sorted((float(e.get("weight", 0.0)) for e in events),
                     reverse=True)
    bonus = weights[0] * MAX_OCR_BONUS
    if len(weights) > 1:
        bonus += min(3.0, sum(weights[1:]) * 2.0)
    return round(min(MAX_OCR_BONUS, bonus), 1)


def scan_game_events(video_path, t0=0.0, fps=None, ocr_fn=None):
    """Scan a video for game-banner events.

    Returns {"src", "t0", "events": [...], "scan_s",
    "ocr_available": bool}. When RapidOCR is not installed and no
    ``ocr_fn`` was injected, the scan still runs (colour cues + cuts)
    but no banner text can be read, so ``events`` holds only cuts and
    ``ocr_available`` is False. Raises FileNotFoundError for a missing
    file and ImportError when OpenCV is missing.
    """
    if not os.path.exists(str(video_path)):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    started = time.time()
    available = ocr_fn is not None or ocr_available()
    raw = scan_video(video_path, t0=t0, fps=fps, ocr_fn=ocr_fn)
    events = merge_events(raw)
    if not available:
        # Without OCR the only trustworthy events are hard cuts.
        events = [e for e in events if e["kind"] == "cut"]
    return {
        "src": str(video_path),
        "t0": t0,
        "events": events,
        "scan_s": round(time.time() - started, 1),
        "ocr_available": available,
    }


# ---------------------------------------------------------------------------
# On-screen dialogue / objective text (for the caption generator)
# ---------------------------------------------------------------------------

# How the caption generator reads the video the way Claude does: the
# streamer may barely speak, but the game talks — GTA dialogue and
# objective text ("Go to the crash site") sit in the bottom band of
# the frame, and that is where "$500 STRAWBERRY farmer" came from.
# Sampled frames are OCR'd with the same RapidOCR engine as the
# banner scan; lines that repeat across frames are deduped.
SCREEN_TEXT_INTERVAL_S = float(
    os.getenv("SCREEN_TEXT_INTERVAL_S", "2.0") or 2.0)
SCREEN_TEXT_MAX_FRAMES = int(
    os.getenv("SCREEN_TEXT_MAX_FRAMES", "15") or 15)
SCREEN_TEXT_MIN_CONF = float(
    os.getenv("SCREEN_TEXT_MIN_CONF", "0.8") or 0.8)
SCREEN_TEXT_BAND = 0.25  # bottom 25% of the frame (subtitle band)
SCREEN_TEXT_CACHE_SIZE = 8

# Phrases RapidOCR glues together at 360p ("Gotothecrashsite"):
# repaired so the caption entity extractor can see the words.
_SCREEN_TEXT_GLUED = {
    "crashsite": "crash site",
    "vicecity": "vice city",
}

_screen_text_cache = {}
_screen_text_cache_order = []


def clear_screen_text_cache():
    """Empty the on-screen-text cache (tests)."""
    _screen_text_cache.clear()
    _screen_text_cache_order.clear()


def _sample_times(t0, t1, interval, max_frames):
    """Sample times in [t0, t1]: every ``interval`` s, capped at
    ``max_frames`` evenly spaced (a long window gets spread samples
    instead of a cut-off scan)."""
    if t1 <= t0:
        return [round(t0, 2)]
    count = int((t1 - t0) / interval)
    if count <= max_frames:
        return [round(t0 + i * interval, 2) for i in range(max(1, count))]
    step = (t1 - t0) / max_frames
    return [round(t0 + step * (i + 0.5), 2) for i in range(max_frames)]


def _unglue_screen_text(text):
    """Repair phrases RapidOCR glues at low resolution."""
    out = text or ""
    for glued, fixed in _SCREEN_TEXT_GLUED.items():
        out = re.sub(glued, fixed, out, flags=re.IGNORECASE)
    return out


def _clean_screen_lines(raw_lines, min_conf=SCREEN_TEXT_MIN_CONF):
    """Filter + dedupe OCR lines -> [line].

    Drops low-confidence reads (garbled fragments like "Gofothe-Gsl"
    score ~0.75 while real subtitle text scores 0.9+) and lines
    without letters, repairs glued phrases, and dedupes by
    normalized text — the same subtitle repeats across frames, and
    when two reads of one line differ the one with more spaces wins
    ("Go to the crash site" over "Gotothe crash site").
    """
    best = {}
    for text, conf in raw_lines:
        if float(conf) < min_conf:
            continue
        line = _unglue_screen_text(
            re.sub(r"\s+", " ", (text or "")).strip())
        if len(line) < 3:
            continue
        if not any(ch.isalpha() for ch in line):
            continue
        norm = re.sub(r"[^a-z0-9]+", "", line.lower())
        if not norm:
            continue
        if norm not in best or line.count(" ") > best[norm].count(" "):
            best[norm] = line
    return list(best.values())


def _screen_text_cache_key(path, t0, t1):
    try:
        st = os.stat(str(path))
        return (os.path.abspath(str(path)), st.st_mtime_ns, st.st_size,
                round(float(t0), 1), round(float(t1), 1))
    except (OSError, ValueError):
        return None


def read_screen_text(video_path, t0=0.0, t1=None, interval=None,
                     ocr_fn=None, use_cache=True):
    """Read the on-screen dialogue + objective text in a clip window.

    Samples frames every ~2 s (at most SCREEN_TEXT_MAX_FRAMES),
    OCRs the bottom 25% of each frame — the subtitle band where GTA
    dialogue and objective text appear — and returns
    {"text", "lines", "frames_ocr", "sampled", "scan_s"}.

    Best effort: a missing file, missing OpenCV or a missing OCR
    engine returns empty text instead of raising, so the caption
    generator falls back to transcript-only captions. Results are
    cached per (file, window), so a clip is OCR'd once, not on every
    read. ``ocr_fn`` is an injectable OCR callable (image ->
    [(box, text, conf)]) so tests can run without the RapidOCR
    package.
    """
    started = time.time()
    empty = {"text": "", "lines": [], "frames_ocr": 0, "sampled": 0,
             "scan_s": 0.0}
    if not os.path.exists(str(video_path)):
        return dict(empty)
    try:
        import cv2
    except ImportError:
        return dict(empty)
    try:
        cap = cv2.VideoCapture(str(video_path))
    except Exception:
        return dict(empty)
    if not cap.isOpened():
        return dict(empty)
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / video_fps if frame_count else 0.0
        start = max(0.0, float(t0))
        end = float(t1) if t1 is not None else duration
        if end <= 0:
            end = duration
        if end <= start:
            return dict(empty)
        key = _screen_text_cache_key(video_path, start, end)
        if use_cache and key is not None and key in _screen_text_cache:
            return dict(_screen_text_cache[key])
        times = _sample_times(start, end,
                              interval or SCREEN_TEXT_INTERVAL_S,
                              SCREEN_TEXT_MAX_FRAMES)
        ocr = ocr_fn or run_ocr
        raw_lines = []
        frames_ocr = 0
        prev_band = None
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            height = frame.shape[0]
            band = frame[int(height * (1 - SCREEN_TEXT_BAND)):, :]
            small = cv2.resize(band, (80, 24)).astype(np.int16)
            if (prev_band is not None
                    and np.abs(small - prev_band).mean() <= 6):
                continue  # subtitle band unchanged: same lines already read
            for _box, text, conf in ocr(band):
                raw_lines.append((text, conf))
            frames_ocr += 1
            prev_band = small
        lines = _clean_screen_lines(raw_lines)
        result = {"text": " ".join(lines), "lines": lines,
                  "frames_ocr": frames_ocr, "sampled": len(times),
                  "scan_s": round(time.time() - started, 1)}
        if use_cache and key is not None:
            _screen_text_cache[key] = result
            _screen_text_cache_order.append(key)
            while len(_screen_text_cache_order) > SCREEN_TEXT_CACHE_SIZE:
                old = _screen_text_cache_order.pop(0)
                _screen_text_cache.pop(old, None)
        return dict(result)
    except Exception:
        # Best effort: any read/OCR failure means transcript-only
        # captions, never a failed clip.
        return dict(empty)
    finally:
        cap.release()
