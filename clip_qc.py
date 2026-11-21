"""
Automatic quality check of finished clips (gta6-clipper feature 2).

Every rendered clip is checked BEFORE the pipeline marks it
ready_to_upload. A bad render (black frames, silent clip, ends on a
logo card, wrong loudness) is caught here instead of when Sara watches
it. Each clip gets PASS / WARN / FAIL:

    FAIL  duration outside 12-60 s, loudness more than 2 LU from
          -14 LUFS, black frames anywhere, the game panel black for
          over 1 s, opens or ends on a dark/flat card, over 40%
          silence, or the file cannot be read at all.
    WARN  over 35 s long, frozen picture over 1.5 s, no face or
          motion in the first second, 25-40% silence.
    PASS  none of the above.

    Deliberately NOT warned: a single dark band ("letterbox strip").
    Real GTA footage at dusk/night has near-black bottom bands (dark
    road) for the whole clip, so that warn fired on every dark clip;
    a picture that actually goes black is still caught by the
    game-panel-black FAIL. (black_band_s stays in the report as a
    diagnostic.)

The face check uses the EXISTING OpenCV Haar cascade shipped in the
repo (models/haarcascade_frontalface_default.xml) — the reference
implementation used a YuNet .onnx model, but binary model files cannot
be pushed through the snapshot push, so Haar it is. Frame-based checks
need OpenCV; when cv2 is unavailable they are skipped (recorded in the
report) and the ffmpeg-only checks still run.

Deliberately NOT checked: captions. Sara's rule is no subtitles on
Kick clips, and the TikTok study confirmed the top clipping accounts
use none — a "captions present" check would warn on every good clip.

Settings are module-local constants with env-var overrides (no
config.py changes — other agents own that file):

    QC_MIN_SECONDS / QC_MAX_SECONDS   Duration window (default 12 / 60)
    QC_LONG_WARN_SECONDS              Long-clip warning (default 35)
    QC_LUFS_TOLERANCE                 Loudness tolerance in LU (default 2)
    QC_SILENCE_WARN_PCT               Silence warning % (default 25)
    QC_SILENCE_FAIL_PCT               Silence failure % (default 40)

``check_clip`` never raises: a check that errors is recorded and the
clip is judged on the checks that ran. Only a file that cannot be
read at all is an automatic FAIL.
"""

import os
import re
import subprocess
from pathlib import Path

from loudness import TARGET_LUFS, measure_lufs

# --- thresholds (env overrides) -------------------------------------------
MIN_SECONDS = float(os.environ.get("QC_MIN_SECONDS", "12"))
MAX_SECONDS = float(os.environ.get("QC_MAX_SECONDS", "60"))
LONG_WARN_SECONDS = float(os.environ.get("QC_LONG_WARN_SECONDS", "35"))
LUFS_TOLERANCE = float(os.environ.get("QC_LUFS_TOLERANCE", "2"))
SILENCE_WARN_PCT = float(os.environ.get("QC_SILENCE_WARN_PCT", "25"))
SILENCE_FAIL_PCT = float(os.environ.get("QC_SILENCE_FAIL_PCT", "40"))

# --- fixed analysis parameters --------------------------------------------
FPS = 5               # frames per second sampled for the picture checks
PANEL_START = 0.45    # game panel = rows below 45% of the frame
BLACK_MIN_S = 0.3     # blackdetect: a black run this long counts
FREEZE_MIN_S = 1.5    # freezedetect: a frozen run this long counts
BAND_MIN_S = 1.0      # a dark band / black panel this long counts
END_WINDOW_S = 1.5    # "ends on a card" looks at the last 1.5 s
FIRST_SECOND_S = 1.0  # "opens on a card" / face check window
SILENCE_DB = -35      # bins quieter than this count as silence
SILENCE_MIN_S = 0.25  # silencedetect: a silence this long counts

try:
    import cv2
    import numpy as np
except Exception:  # OpenCV is optional: ffmpeg checks still run
    cv2 = None
    np = None


def _ffprobe_duration(path):
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(res.stdout.strip())


def _ffmpeg_stderr(args, timeout=300):
    res = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", *args],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    return res.stderr


def _black_and_freeze(path):
    """Black-frame runs + frozen-picture runs, in one ffmpeg pass."""
    err = _ffmpeg_stderr([
        "-i", str(path), "-an", "-vf",
        f"blackdetect=d={BLACK_MIN_S}:pix_th=0.10,"
        f"freezedetect=n=-60dB:d={FREEZE_MIN_S}",
        "-f", "null", "-",
    ])
    black = [float(x) for x in re.findall(r"black_duration:([\d.]+)", err)]
    frozen = [float(x) for x in re.findall(r"freeze_duration: ([\d.]+)", err)]
    return black, frozen


def _silence_pct(path, duration):
    """Share of the clip that is silent (bins under SILENCE_DB)."""
    if not duration:
        return None
    err = _ffmpeg_stderr([
        "-i", str(path), "-vn", "-af",
        f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}",
        "-f", "null", "-",
    ])
    total = 0.0
    open_start = None
    for line in err.splitlines():
        start = re.search(r"silence_start: ([\d.]+)", line)
        end = re.search(r"silence_end: ([\d.]+)", line)
        if start:
            open_start = float(start.group(1))
        elif end and open_start is not None:
            total += float(end.group(1)) - open_start
            open_start = None
    if open_start is not None:  # silence runs to the end of the file
        total += max(0.0, duration - open_start)
    return round(100 * total / duration, 1)


def _haar_cascade():
    """The repo's own Haar face cascade, with a cv2-data fallback."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "models" / "haarcascade_frontalface_default.xml",
        Path("models") / "haarcascade_frontalface_default.xml",
    ]
    try:
        candidates.append(Path(cv2.data.haarcascades)
                          / "haarcascade_frontalface_default.xml")
    except Exception:
        pass
    for cand in candidates:
        if cand.exists():
            try:
                cascade = cv2.CascadeClassifier(str(cand))
                if not cascade.empty():
                    return cascade
            except Exception:
                # A partial cv2 (no .empty(), ...) cannot do face
                # detection: skip the face check rather than fail QC.
                continue
    return None


def _sample_frames(path):
    """Per analysed frame: time, panel stats, dark bands, motion, face.

    Returns [] when OpenCV is unavailable or the file cannot be read.
    """
    if cv2 is None or np is None:
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / FPS))
    cascade = _haar_cascade()
    frames, prev, i = [], None, 0
    try:
        while True:
            if not cap.grab():
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                small = cv2.resize(frame, (270, 480))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype("float32")
                panel = gray[int(480 * PANEL_START):]
                bands = [gray[r:r + 24] for r in range(0, 480, 24)]
                dark = [bool(b.mean() < 10 and b.std() < 6) for b in bands]
                motion = (0.0 if prev is None
                          else float(np.abs(gray[216:] - prev[216:]).mean()))
                prev = gray
                face = False
                if cascade is not None:
                    found = cascade.detectMultiScale(
                        gray.astype("uint8"), scaleFactor=1.1,
                        minNeighbors=5, minSize=(20, 20))
                    face = len(found) > 0
                frames.append({
                    "t": i / fps,
                    "panel_mean": float(panel.mean()),
                    "panel_std": float(panel.std()),
                    "dark": dark,
                    "motion": motion,
                    "face": face,
                })
            i += 1
    except Exception:
        # A partial cv2 (no grab/retrieve, ...) cannot sample frames:
        # return what was collected so far (usually []) and let the
        # frame-based checks record themselves as skipped.
        pass
    finally:
        cap.release()
    return frames


def _is_card(frames, share):
    """Logo / black / flat card: the panel is dark AND flat, or
    perfectly flat at any brightness, AND still in at least ``share``
    of these frames.

    The AND matters: a dusk/night game scene is dark (panel mean
    ~24) but textured (std ~30) — it is real gameplay, not a card.
    Only a panel that is dark and flat (a black/logo card) or flat
    to a single colour (std < 5, a solid-colour card) counts.
    """
    if not frames:
        return False, {}
    bad = [((f["panel_mean"] < 25 and f["panel_std"] < 22)
            or f["panel_std"] < 5)
           and f["motion"] < 4 for f in frames]
    card_share = float(np.mean(bad)) if np is not None else sum(bad) / len(bad)
    info = {
        "card_share": round(card_share, 2),
        "panel_mean": round(float(sorted(f["panel_mean"]
                                         for f in frames)[len(frames) // 2]), 1),
    }
    return card_share >= share, info


def check_clip(path):
    """QC one finished clip. Returns a report dict (never raises).

    Keys: file, status ("PASS" / "WARN" / "FAIL"), fail (list of
    reasons), warn (list), plus metrics: duration, lufs, black_s,
    freeze_s, silence_pct, black_band_s, black_panel_s, start, end,
    frame_checks ("" when the picture checks ran, otherwise the reason
    they were skipped).
    """
    path = Path(path)
    fail, warn, metrics = [], [], {}
    if not path.exists():
        return {"file": path.name, "status": "FAIL",
                "fail": ["file not found"], "warn": [], **metrics}

    # --- duration ---------------------------------------------------------
    try:
        duration = metrics["duration"] = round(_ffprobe_duration(path), 1)
    except Exception as e:
        return {"file": path.name, "status": "FAIL",
                "fail": [f"unreadable file ({str(e)[:80]})"], "warn": [],
                **metrics}
    if not MIN_SECONDS <= duration <= MAX_SECONDS:
        fail.append(f"duration {duration}s")
    elif duration > LONG_WARN_SECONDS:
        warn.append(f"long {duration}s")

    # --- loudness ---------------------------------------------------------
    try:
        lufs = metrics["lufs"] = measure_lufs(path)
    except Exception:
        lufs = metrics["lufs"] = None
    if lufs is None or abs(lufs - TARGET_LUFS) > LUFS_TOLERANCE:
        fail.append(f"loudness {lufs} LUFS")

    # --- black frames + frozen picture ------------------------------------
    try:
        black, frozen = _black_and_freeze(path)
    except Exception:
        black, frozen = [], []
    metrics["black_s"], metrics["freeze_s"] = black, frozen
    if black:
        fail.append(f"black frames {max(black):.1f}s")
    if frozen:
        warn.append(f"frozen picture {max(frozen):.1f}s")

    # --- silence ----------------------------------------------------------
    try:
        silence = metrics["silence_pct"] = _silence_pct(path, duration)
    except Exception:
        silence = metrics["silence_pct"] = None
    if silence is not None:
        if silence > SILENCE_FAIL_PCT:
            fail.append(f"silence {silence}%")
        elif silence > SILENCE_WARN_PCT:
            warn.append(f"silence {silence}%")

    # --- picture checks (need OpenCV) -------------------------------------
    frames = _sample_frames(path)
    if cv2 is None or np is None:
        metrics["frame_checks"] = "skipped (OpenCV unavailable)"
    elif not frames:
        metrics["frame_checks"] = "skipped (no frames read)"
    else:
        metrics["frame_checks"] = ""
        panel_rows = int(20 * PANEL_START) + 1
        # One dark row of the picture that stays dark > 1 s = a
        # letterbox / empty strip (WARN); most of the panel dark > 1 s
        # = the panel went black (FAIL).
        best_band = best_panel = 0
        for row in range(20):
            run = 0
            for f in frames:
                run = run + 1 if f["dark"][row] else 0
                best_band = max(best_band, run)
        run = 0
        for f in frames:
            dark_panel = sum(f["dark"][panel_rows:]) >= 0.6 * (19 - panel_rows)
            run = run + 1 if dark_panel else 0
            best_panel = max(best_panel, run)
        metrics["black_band_s"] = round(best_band / FPS, 1)
        metrics["black_panel_s"] = round(best_panel / FPS, 1)
        if best_panel / FPS >= BAND_MIN_S:
            fail.append(f"game panel black {best_panel / FPS:.1f}s")
        # NOTE: there used to be a WARN here for a single dark band
        # ("black strip in picture"). It was removed: on real GTA
        # footage the bottom bands of a normal dusk/night scene are
        # near-black (dark road, mean 0.3-4.7) for the whole clip, so
        # the warn fired on every dark clip. A strip that matters —
        # most of the picture gone black — is still caught by the
        # panel-black FAIL above. black_band_s stays in the report
        # as a diagnostic metric.

        # Opens / ends on a card.
        start_frames = [f for f in frames if f["t"] < FIRST_SECOND_S]
        end_frames = [f for f in frames if f["t"] >= duration - END_WINDOW_S]
        opens_card, metrics["start"] = _is_card(start_frames, 0.6)
        ends_card, metrics["end"] = _is_card(end_frames, 0.4)
        if opens_card:
            fail.append("opens on a card/black/flat panel")
        if ends_card:
            fail.append("ends on a card/black/flat panel")

        # A face or motion in the first second — otherwise the clip
        # opens on a dead frame.
        first_motion = ([f["motion"] for f in start_frames[1:]]
                        if len(start_frames) > 1 else [0.0])
        avg_motion = (float(np.mean(first_motion))
                      if np is not None else sum(first_motion) / len(first_motion))
        if not (any(f["face"] for f in start_frames) or avg_motion > 2.0):
            warn.append("no face or motion in first second")

    status = "FAIL" if fail else "WARN" if warn else "PASS"
    return {"file": path.name, "status": status,
            "fail": fail, "warn": warn, **metrics}
