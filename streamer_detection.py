"""
Multi-streamer detection for ai-social-studio clips (feature 13).

Answers "who is in this clip?" so the Kick clipping-program POV rule can
be applied automatically: when several streamers appear in one clip, it
must be submitted under the lowest-CPM streamer shown
(see kick_program.resolve_multi_streamer_pov).

Three signals, combined:
    (a) Transcript name mentions (PRIMARY). The Whisper transcript is
        searched for known streamer usernames and aliases (case-insensitive
        word match) against the monitored list plus every CREATOR_RATES
        key. This is the only signal that names a streamer without extra
        setup, and it is the primary signal in practice.
    (b) Face count / positions (SUPPORTING). Frames are sampled and faces
        counted with the Haar cascades from models/ (the same cascades
        SpeakerTracker uses), plus SpeakerTracker's camera trajectory: a
        camera that pans between speakers suggests more than one person.
        This signal can only say HOW MANY people are on screen, never WHO
        they are. It is reported as evidence, and it confirms a
        multi-person clip when the transcript names 2+ streamers. It is
        skipped gracefully when OpenCV is not installed.
    (c) Face recognition (OPTIONAL). When the ``face_recognition`` library
        is installed AND reference images exist (one image per streamer,
        named <username>.jpg/.png, in FACE_REFERENCE_DIR or
        <repo>/face_references/), faces in sampled frames are encoded and
        matched against the references. This is the only signal that can
        identify a streamer who is never named in the transcript.

HONEST LIMITATION: without (c) installed and supplied with reference
images, detection relies on the transcript. A streamer who appears on
camera but is never named out loud will NOT be detected -- (b) can only
flag "there is another person here". The overlays/ watermark images are
not usable as face references (they are text bars, not faces). Treat the
returned list as "streamers we have evidence for", and surface the
evidence dict (see detect_streamers_with_evidence) in the dashboard when
a clip needs a human to confirm the POV streamer.
"""

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Fallback known-streamer list used only when config/kick_program cannot
# be imported (e.g. in a minimal test environment). In production the
# monitored list + CREATOR_RATES keys are used instead.
_FALLBACK_KNOWN_STREAMERS = [
    "xqc", "adinross", "westcol", "tarik", "n3on", "trainwreckstv",
    "mizkif", "asmongold", "sneako", "jidion", "lospollostv",
]

# Aliases for transcript matching. Kept deliberately conservative: an
# alias must be unambiguous, because a false match submits a clip under
# the wrong streamer. Common words (e.g. "neon" for n3on, "louie" for
# lospollostv) are NOT aliases.
ALIASES = {
    "xqc": ["xqc", "xqcow", "xqc ow", "felix lengyel"],
    "adinross": ["adinross", "adin ross", "adin"],
    "westcol": ["westcol"],
    "tarik": ["tarik", "tarik celik"],
    "n3on": ["n3on"],
    "trainwreckstv": ["trainwreckstv", "trainwreck"],
    "mizkif": ["mizkif", "miz"],
    "asmongold": ["asmongold", "asmon", "zackrawrr"],
    "sneako": ["sneako"],
    "jidion": ["jidion", "jideon"],
    "lospollostv": ["lospollostv", "los pollos tv", "los pollos"],
    "iceposeidon": ["iceposeidon", "ice poseidon", "paul denino"],
    "jackdoherty": ["jackdoherty", "jack doherty"],
    "cheesur": ["cheesur"],
    "cuffem": ["cuffem"],
    "konvy": ["konvy"],
    "samfrank": ["samfrank", "sam frank"],
    "ryangarcia": ["ryangarcia", "ryan garcia"],
    "akademiks": ["akademiks", "dj akademiks"],
    "hstikkytokky": ["hstikkytokky", "hs tikkytokky"],
    "edmatthews": ["edmatthews", "ed matthews"],
    "clavicular": ["clavicular"],
    "larrywheels": ["larrywheels", "larry wheels"],
    "blueface": ["blueface"],
    "6ix9ine": ["6ix9ine", "tekashi", "tekashi 6ix9ine"],
    "kodakblack": ["kodakblack", "kodak black"],
    "nelkboys": ["nelkboys", "nelk boys", "nelk"],
    "davido": ["davido"],
    "moon": ["moon"],
    "stake": ["stake"],
}

FACE_SAMPLE_FRAMES = 8
FACE_CLUSTER_TOLERANCE_PX = 120  # x-centres closer than this = same person
TRAJECTORY_PAN_THRESHOLD_PX = 100  # std of crop-x above this = camera pans


def _default_known_streamers():
    """Monitored streamers + every CREATOR_RATES key (lowercased)."""
    known = set(_FALLBACK_KNOWN_STREAMERS)
    try:
        from config import STREAMERS_TO_MONITOR
        known.update(s.lower() for s in STREAMERS_TO_MONITOR)
    except Exception:
        pass
    try:
        from kick_program import CREATOR_RATES
        known.update(k.lower() for k in CREATOR_RATES)
    except Exception:
        pass
    return sorted(known)


def _name_map(known_streamers):
    """
    Map every matchable name/alias (lowercased) to its username.
    The username itself is always matchable; ALIASES add extra spellings.
    """
    mapping = {}
    for username in known_streamers or []:
        clean = str(username).lower().strip().lstrip("@")
        if not clean:
            continue
        mapping[clean] = clean
        for alias in ALIASES.get(clean, []):
            mapping[alias.lower()] = clean
    return mapping


def find_mentioned_streamers(transcript_text, known_streamers=None):
    """
    Signal (a): streamers named in the transcript.

    Case-insensitive whole-word/phrase match, so "xqc" does not match
    inside "xqcow" (that is its own alias) and "adin" does not match
    inside "trading". "@username" mentions match too.
    Returns usernames in order of first mention.
    """
    if not transcript_text:
        return []
    if known_streamers is None:
        known_streamers = _default_known_streamers()
    mapping = _name_map(known_streamers)
    text = transcript_text.lower()

    first_seen = {}
    for name, username in mapping.items():
        # Word boundaries on both ends; multi-word aliases match as a phrase.
        pattern = r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])"
        match = re.search(pattern, text)
        if match:
            pos = match.start()
            if username not in first_seen or pos < first_seen[username]:
                first_seen[username] = pos
    return sorted(first_seen, key=lambda u: first_seen[u])


def _face_evidence(video_path):
    """
    Signal (b): how many distinct people appear on screen.

    Samples frames, detects faces with the Haar cascades from models/,
    clusters face x-centres into distinct positions, and checks whether
    SpeakerTracker's camera trajectory pans between speakers. Returns a
    dict, or None when OpenCV/cascades/video are unavailable (graceful
    skip -- this signal is supporting evidence only).
    """
    if not video_path or not os.path.exists(video_path):
        return None
    try:
        import cv2
    except ImportError:
        return None

    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    if not (models_dir / "haarcascade_frontalface_default.xml").exists():
        models_dir = Path("models")
    cascade_path = models_dir / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None

    # A broken or incomplete OpenCV (missing attributes, unreadable video)
    # is the same as no OpenCV for this best-effort signal: skip gracefully
    # instead of failing the whole detection.
    try:
        face_cascade = cv2.CascadeClassifier(str(cascade_path))
        if face_cascade.empty():
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total_frames <= 0:
                return None
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0

            max_faces = 0
            x_centres = []
            sample_indices = [
                int(total_frames * (i + 0.5) / FACE_SAMPLE_FRAMES)
                for i in range(FACE_SAMPLE_FRAMES)
            ]
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
                faces = face_cascade.detectMultiScale(
                    small, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30)
                )
                max_faces = max(max_faces, len(faces))
                for (x, y, w, h) in faces:
                    x_centres.append(int((x + w / 2) / 0.5))
        finally:
            cap.release()
    except Exception:
        return None

    # Cluster x-centres: centres within tolerance of each other are the
    # same person seen in different frames.
    distinct_positions = 0
    for centre in sorted(x_centres):
        if distinct_positions == 0:
            distinct_positions = 1
            last_cluster = centre
        elif abs(centre - last_cluster) > FACE_CLUSTER_TOLERANCE_PX:
            distinct_positions += 1
            last_cluster = centre

    # SpeakerTracker trajectory: a camera that pans a lot is following
    # more than one speaker. Best-effort -- never fails detection.
    trajectory_pans = False
    try:
        from speaker_tracker import SpeakerTracker
        tracker = SpeakerTracker(str(models_dir))
        smoothed_x, _, _, _ = tracker.analyze_video_trajectory(str(video_path))
        if smoothed_x:
            mean_x = sum(smoothed_x) / len(smoothed_x)
            variance = sum((x - mean_x) ** 2 for x in smoothed_x) / len(smoothed_x)
            trajectory_pans = variance ** 0.5 > TRAJECTORY_PAN_THRESHOLD_PX
    except Exception as e:
        logger.debug("SpeakerTracker trajectory unavailable: %s", e)

    return {
        "max_faces_in_frame": max_faces,
        "distinct_face_positions": distinct_positions,
        "trajectory_pans": trajectory_pans,
        "frame_width": frame_w,
    }


def _face_recognition_available():
    try:
        import face_recognition  # noqa: F401
        return True
    except ImportError:
        return False


def _reference_dir():
    env_dir = os.environ.get("FACE_REFERENCE_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parent / "face_references"


def _load_reference_encodings(reference_dir=None):
    """
    Signal (c) setup: {username: face encoding} from reference images.

    One image per streamer, named <username>.jpg/.jpeg/.png. Returns an
    empty dict when the library is absent, the directory is missing, or
    no image yields exactly one face encoding.
    """
    if not _face_recognition_available():
        return {}
    import face_recognition

    ref_dir = Path(reference_dir) if reference_dir else _reference_dir()
    encodings = {}
    if not ref_dir.exists():
        return encodings
    for image_path in sorted(ref_dir.iterdir()):
        if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        username = image_path.stem.lower()
        try:
            image = face_recognition.load_image_file(str(image_path))
            found = face_recognition.face_encodings(image)
        except Exception as e:
            logger.warning("Face reference %s unreadable: %s", image_path, e)
            continue
        if len(found) == 1:
            encodings[username] = found[0]
        elif len(found) > 1:
            logger.warning(
                "Face reference %s has %d faces; skipping (use one face per image)",
                image_path, len(found),
            )
    return encodings


def _recognize_faces_in_clip(video_path, reference_encodings, tolerance=0.6):
    """
    Signal (c): match sampled frames against reference face encodings.
    Returns usernames whose face was matched in at least one frame.
    """
    if not reference_encodings or not video_path or not os.path.exists(video_path):
        return []
    import face_recognition

    cap = None
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            return []
        sample_indices = [
            int(total_frames * (i + 0.5) / FACE_SAMPLE_FRAMES)
            for i in range(FACE_SAMPLE_FRAMES)
        ]
        known_names = list(reference_encodings.keys())
        known_encodings = [reference_encodings[name] for name in known_names]
        matched = set()
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            locations = face_recognition.face_locations(small)
            if not locations:
                continue
            encodings = face_recognition.face_encodings(small, locations)
            for encoding in encodings:
                matches = face_recognition.compare_faces(
                    known_encodings, encoding, tolerance=tolerance
                )
                distances = face_recognition.face_distance(known_encodings, encoding)
                if True in matches:
                    best = int(min(range(len(distances)), key=lambda i: distances[i]))
                    if matches[best]:
                        matched.add(known_names[best])
        return sorted(matched)
    finally:
        cap.release()


def detect_streamers_with_evidence(video_path, transcript_text, known_streamers=None,
                                   reference_dir=None):
    """
    Detect streamers in a clip, with the evidence behind the decision.

    Returns (usernames, evidence) where evidence is a dict:
        {
          "mentioned": [...],        # signal (a) transcript matches
          "recognized": [...],       # signal (c) face-recognition matches
          "face_evidence": {...}|None,  # signal (b) counts/trajectory
          "face_recognition_available": bool,
          "reference_streamers": [...], # streamers with reference images
          "multi_person_suspected": bool,
          "note": str,               # honest caveat when identification is weak
        }

    ``multi_person_suspected`` is True when the face evidence shows 2+
    distinct people (or a panning camera + 2+ faces in a frame) -- even
    when only one streamer could be named. That is the case a human
    should double-check before submitting.
    """
    mentioned = find_mentioned_streamers(transcript_text, known_streamers)
    face_evidence = _face_evidence(video_path)

    recognized = []
    reference_encodings = _load_reference_encodings(reference_dir)
    if reference_encodings:
        recognized = _recognize_faces_in_clip(video_path, reference_encodings)

    detected = []
    for username in mentioned + recognized:
        if username not in detected:
            detected.append(username)

    multi_person = False
    if face_evidence:
        multi_person = (
            face_evidence["distinct_face_positions"] >= 2
            or face_evidence["max_faces_in_frame"] >= 2
        )

    note = ""
    if multi_person and len(detected) <= 1:
        note = (
            "Face evidence suggests %d people on screen but only %d streamer(s) "
            "could be identified (%s). A streamer who is never named in the "
            "transcript cannot be identified without the face_recognition "
            "backend + reference images -- confirm the POV streamer manually "
            "before submitting."
            % (
                max(face_evidence["distinct_face_positions"],
                    face_evidence["max_faces_in_frame"]),
                len(detected),
                ", ".join(detected) if detected else "none",
            )
        )
    elif not detected:
        note = (
            "No known streamer was named in the transcript and no face "
            "recognition match was made. Detection relies on transcript "
            "mentions unless the face_recognition backend is set up."
        )

    evidence = {
        "mentioned": mentioned,
        "recognized": recognized,
        "face_evidence": face_evidence,
        "face_recognition_available": _face_recognition_available(),
        "reference_streamers": sorted(reference_encodings.keys()),
        "multi_person_suspected": multi_person,
        "note": note,
    }
    return detected, evidence


def detect_streamers_in_clip(video_path, transcript_text, known_streamers=None):
    """
    Detect which known streamers appear in a clip.

    Combines transcript name mentions (primary), face-count evidence
    (supporting) and optional face recognition (see module docstring for
    the honest limitation: without the face_recognition backend, a
    streamer who is never named in the transcript will not be detected).

    Args:
        video_path: clip file (may be None to use the transcript only).
        transcript_text: Whisper transcript of the clip.
        known_streamers: usernames to look for; defaults to the monitored
            list + CREATOR_RATES keys.

    Returns:
        List of usernames, transcript-mention order first, then any
        face-recognition matches.
    """
    detected, _ = detect_streamers_with_evidence(
        video_path, transcript_text, known_streamers
    )
    return detected


def pick_submission_streamer(detected):
    """
    Pick the POV streamer to submit a clip under (lowest CPM wins).

    Calls kick_program.resolve_multi_streamer_pov -- the official
    multi-streamer POV rule -- and returns (streamer, discord_command).
    With no detected streamers this falls back to ("other", "/other").

    Returns:
        (streamer_username, discord_command_prefix) e.g. ("xqc", "/xqc").
        Pair it with kick_program.format_bot_submission_command to build
        the full Discord command with video links.
    """
    detected = [str(s).lower().strip().lstrip("@") for s in (detected or []) if s]
    try:
        from kick_program import CREATOR_RATES, get_creator_info, resolve_multi_streamer_pov
        resolved = resolve_multi_streamer_pov(detected)
        # resolve_multi_streamer_pov returns the rate dict, not the name,
        # so re-derive which streamer it picked (lowest CPM).
        best_streamer = "other"
        best_cpm = None
        for username in detected:
            cpm = get_creator_info(username)["cpm"]
            if best_cpm is None or cpm < best_cpm:
                best_cpm = cpm
                best_streamer = username
        # A detected name outside the program submits under /other.
        if best_streamer not in CREATOR_RATES:
            best_streamer = "other"
        return best_streamer, resolved["cmd"]
    except Exception as e:
        logger.warning("kick_program unavailable (%s); using first detected streamer", e)
        if detected:
            return detected[0], f"/{detected[0]}"
        return "other", "/other"
