"""
bgm.py — Local background-music pool + sidechain ducking filter.

Adapted from opensource-clipping (https://github.com/NaufalRizqullah/opensource-clipping,
MIT licence), clipping/studio/audio_bgm.py: pick a random MP3 from
``assets/bgm/<mood>/`` and mix it under the speech with ffmpeg's
``sidechaincompress`` so the music dips whenever someone talks
(threshold 0.08, ratio 5.0, attack 100 ms, release 1000 ms, base volume 0.25).

No music is bundled: the user adds their own royalty-free MP3s to
``assets/bgm/<mood>/`` (see assets/bgm/README.md). With no MP3s present,
pick_bgm() returns None and BGM is simply skipped.
"""

import os
import random
from pathlib import Path

MOODS = ("chill", "epic", "upbeat", "sad", "suspense")
DEFAULT_MOOD = "chill"
BGM_BASE_VOLUME = 0.25

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
    missing or empty (as in opensource-clipping). Returns the absolute path
    as a string, or None when no MP3 is available at all.
    """
    for candidate in (mood, DEFAULT_MOOD):
        files = list_bgm(candidate, bgm_dir)
        if files:
            return random.choice(files)
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
        base_volume: BGM level when nobody is talking (default 0.25).
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
    filt = (
        f"{voc};"
        f"{bgm};"
        f"[voc]asplit=2[voc_sc][voc_mix];"
        f"[bgm][voc_sc]sidechaincompress=threshold={threshold}:ratio={ratio}"
        f":attack={attack}:release={release}[bgm_ducked];"
        f"[voc_mix][bgm_ducked]amix=inputs=2:duration=first{output_label}"
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
    filt = f"{voc};{bgm};[voc][bgm]amix=inputs=2:duration=first{output_label}"
    return filt, output_label
