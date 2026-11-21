"""
bgm.py — Local background-music pool + sidechain ducking filter.

Adapted from opensource-clipping (https://github.com/NaufalRizqullah/opensource-clipping,
MIT licence), clipping/studio/audio_bgm.py: pick a random MP3 from
``assets/bgm/<mood>/`` and mix it under the speech with ffmpeg's
``sidechaincompress`` so the music dips whenever someone talks
(threshold 0.08, ratio 5.0, attack 100 ms, release 1000 ms, base volume 0.12).

One built-in track is generated automatically on first use when the file
is missing: ``assets/bgm/chill/soft_pad.mp3``, a gentle ambient pad
synthesised with ffmpeg (no downloaded audio, so it is royalty-free by
construction). It is the fallback for every mood, so BGM works out of
the box with no setup. Sara can add her own royalty-free MP3s to
``assets/bgm/<mood>/`` (see assets/bgm/README.md) — any track in a mood
folder joins that mood's pool.
"""

import os
import random
import shutil
import subprocess
from pathlib import Path

MOODS = ("chill", "epic", "upbeat", "sad", "suspense")
DEFAULT_MOOD = "chill"

# Built-in track: a gentle ambient pad (a warm A-major chord: 220 /
# 277.18 / 329.63 Hz sine tones, slow tremolo, fade in/out), 60 s,
# stereo 44.1 kHz MP3 at 128 kbps. The MP3 is NOT stored in the repo
# (binary files cannot be pushed with the snapshot push); it is
# synthesised with ffmpeg on first use when the file is missing
# (generate_builtin_track, ~1 s, measured mean -24.4 dB / max -15.0 dB).
BUILTIN_TRACK_NAME = "soft_pad.mp3"
BUILTIN_TRACK_SECONDS = 60
BUILTIN_TRACK_FREQUENCIES = (220.0, 277.18, 329.63)
BUILTIN_TRACK_VOLUME = 0.5


def _env_volume(name, default):
    """Read a linear volume (0-1) from the environment, or ``default``
    when unset, unparsable or non-positive."""
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Base BGM level when nobody is talking: 0.12 is soft, roughly -18 dB
# under the speech (which is boosted to 1.2), and the sidechain ducking
# dips the music further while someone talks. Override with the
# BGM_BASE_VOLUME env var.
BGM_BASE_VOLUME = _env_volume("BGM_BASE_VOLUME", 0.12)

# The 8 highlight categories from ai_story_engine.PRD_CATEGORIES.
CATEGORY_MOODS = {
    "Funny Moments": "upbeat",
    "Exciting Gameplay": "epic",
    "Streamer Reactions": "upbeat",
    "Unexpected Moments": "suspense",
    "Big Wins": "epic",
    "Angry or Emotional Reactions": "sad",
    "Viral Potential": "upbeat",
    "Interesting Conversations": "chill",
}


def _bgm_base_dir(bgm_dir=None) -> Path:
    """Resolve the BGM pool root: explicit arg > BGM_DIR env > assets/bgm/."""
    if bgm_dir:
        return Path(bgm_dir)
    env_dir = os.getenv("BGM_DIR")
    if env_dir:
        return Path(env_dir)
    return Path(__file__).resolve().parent / "assets" / "bgm"


def builtin_track_path(bgm_dir=None) -> Path:
    """Path of the built-in track (the file may not exist yet)."""
    return _bgm_base_dir(bgm_dir) / DEFAULT_MOOD / BUILTIN_TRACK_NAME


def generate_builtin_track(bgm_dir=None):
    """Synthesise the built-in ambient pad with ffmpeg.

    Returns the track path as a string on success, or None when ffmpeg
    is missing or the synthesis fails. Safe to call when the file
    already exists (it is left alone).
    """
    target = builtin_track_path(bgm_dir)
    if target.is_file():
        return str(target)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    inputs = []
    for freq in BUILTIN_TRACK_FREQUENCIES:
        inputs += ["-f", "lavfi", "-i",
                   f"sine=frequency={freq}:duration={BUILTIN_TRACK_SECONDS}"]
    filt = (
        "[0:a][1:a][2:a]amix=inputs=3:normalize=0,"
        f"volume={BUILTIN_TRACK_VOLUME},"
        "pan=stereo|c0=c0|c1=c0,tremolo=f=0.2:d=0.3,"
        "afade=t=in:st=0:d=3,"
        f"afade=t=out:st={BUILTIN_TRACK_SECONDS - 3}:d=3"
    )
    try:
        res = subprocess.run(
            [ffmpeg, "-y", "-v", "error", *inputs,
             "-filter_complex", filt, "-t", str(BUILTIN_TRACK_SECONDS),
             "-c:a", "libmp3lame", "-b:a", "128k", str(target)],
            capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0 or not target.is_file():
        return None
    return str(target)


def ensure_builtin_track(bgm_dir=None):
    """Return the built-in track path, generating it first if missing.

    Returns None when the track cannot be produced (no ffmpeg, or the
    synthesis failed) — callers treat that as "no BGM available".
    """
    target = builtin_track_path(bgm_dir)
    if target.is_file():
        return str(target)
    return generate_builtin_track(bgm_dir)


def list_bgm(mood, bgm_dir=None):
    """Sorted list of MP3 paths available for ``mood`` (empty list if none)."""
    mood_dir = _bgm_base_dir(bgm_dir) / mood
    if not mood_dir.is_dir():
        return []
    return sorted(str(p) for p in mood_dir.iterdir()
                  if p.is_file() and p.suffix.lower() == ".mp3")


def pick_bgm(mood, bgm_dir=None):
    """
    Pick a random MP3 for ``mood`` from assets/bgm/<mood>/.

    Falls back to the ``chill`` pool when the requested mood folder is
    missing or empty (as in opensource-clipping). When no MP3 is found
    anywhere and the default pool location is in use (no explicit
    ``bgm_dir`` and no ``BGM_DIR`` env var), the built-in track
    ``assets/bgm/chill/soft_pad.mp3`` is synthesised on the spot with
    ffmpeg, so the fallback always finds at least one track. A custom
    pool location is never written to. Returns the absolute path as a
    string, or None when no MP3 is available at all.
    """
    for candidate in (mood, DEFAULT_MOOD):
        files = list_bgm(candidate, bgm_dir)
        if files:
            return random.choice(files)
    if bgm_dir is None and not os.getenv("BGM_DIR"):
        generated = ensure_builtin_track()
        if generated:
            return generated
    return None


def mood_for_category(category):
    """
    Map one of the 8 PRD highlight categories to a BGM mood.
    Unknown categories map to "chill".
    """
    return CATEGORY_MOODS.get(category, DEFAULT_MOOD)


def bgm_ffmpeg_input_args(bgm_path):
    """
    ffmpeg input args for a BGM file: loop it forever so short tracks cover
    the whole clip (opensource-clipping uses the same -stream_loop -1).
    Insert BEFORE the -i of the main video's other inputs as its own input,
    e.g. cmd = ["ffmpeg", "-y", "-i", clip, *bgm_ffmpeg_input_args(bgm), ...]
    making the music stream [1:a] by default.
    """
    return ["-stream_loop", "-1", "-i", str(bgm_path)]


def build_ducking_filter(voc_label="[0:a]", bgm_label="[1:a]",
                         base_volume=BGM_BASE_VOLUME, threshold=0.08,
                         ratio=5.0, attack=100, release=1000,
                         output_label="[a_out]"):
    """
    Build a sidechain-ducking filter: the music (bgm_label) is compressed
    whenever speech (voc_label) is present, so dialogue stays intelligible.

    Args:
        voc_label: ffmpeg label of the speech audio stream.
        bgm_label: ffmpeg label of the BGM audio stream (the looped MP3).
        base_volume: BGM level when nobody is talking (default 0.12 —
            BGM_BASE_VOLUME, soft under the speech).
        threshold / ratio / attack / release: sidechaincompress parameters
            (defaults match opensource-clipping).
        output_label: label the mixed audio is written to.

    Returns:
        (filter_complex_string, output_label). Wire the BGM file in with
        bgm_ffmpeg_input_args() and ``-map "[a_out]"`` for audio.
    """
    voc = (f"{voc_label}aformat=sample_fmts=fltp:sample_rates=48000"
           f":channel_layouts=stereo,volume=1.2[voc]")
    bgm = (f"{bgm_label}aformat=sample_fmts=fltp:sample_rates=48000"
           f":channel_layouts=stereo,volume={base_volume}[bgm]")
    # normalize=0: amix divides by the input count by default, which
    # would quiet the ORIGINAL audio ~6 dB whenever BGM is on. With
    # normalize=0 the speech keeps its level and the music adds
    # underneath it (the voice is already at 1.2, and the music is
    # ~18 dB down, so clipping is not a practical risk).
    filt = (
        f"{voc};"
        f"{bgm};"
        f"[voc]asplit=2[voc_sc][voc_mix];"
        f"[bgm][voc_sc]sidechaincompress=threshold={threshold}:ratio={ratio}"
        f":attack={attack}:release={release}[bgm_ducked];"
        f"[voc_mix][bgm_ducked]amix=inputs=2:duration=first:normalize=0{output_label}"
    )
    return filt, output_label


def build_background_filter(voc_label="[0:a]", bgm_label="[1:a]",
                            base_volume=BGM_BASE_VOLUME,
                            output_label="[a_out]"):
    """
    Simpler constant-volume mix (no ducking): BGM stays at ``base_volume``
    throughout. Same wiring as build_ducking_filter().
    """
    voc = (f"{voc_label}aformat=sample_fmts=fltp:sample_rates=48000"
           f":channel_layouts=stereo,volume=1.2[voc]")
    bgm = (f"{bgm_label}aformat=sample_fmts=fltp:sample_rates=48000"
           f":channel_layouts=stereo,volume={base_volume}[bgm]")
    # normalize=0: see build_ducking_filter — without it amix would
    # quiet the original audio ~6 dB whenever BGM is on.
    filt = f"{voc};{bgm};[voc][bgm]amix=inputs=2:duration=first:normalize=0{output_label}"
    return filt, output_label
