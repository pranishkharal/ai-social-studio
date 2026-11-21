"""Voice-over: Sara's own line on a clip (gta6-clipper feature 7).

Sara records one short line about the moment (phone or mic), and this
module places it at any second of a finished clip — usually right
after the payoff. While she talks, the game/streamer audio ducks
about 15 dB under her voice; the video is copied untouched, so the
pass is fast and never re-encodes the picture. A line longer than the
rest of the clip is trimmed at the end, not dropped.

Used two ways:
- At render time: ``ClipEditor.edit_clip(voiceover_path=..., voiceover_at=...)``
  runs this as a post-pass on the rendered clip.
- After the fact: the dashboard upload endpoint (gta6_routers.py)
  applies a voice-over to an already-rendered clip.

No keys or logins needed; only ffmpeg/ffprobe.
"""

import os
import subprocess
import sys

# How far the game/streamer audio drops while the voice-over plays.
# sidechaincompress with ratio 20 at threshold 0.015 gives roughly
# -15 dB of ducking; the 400 ms release lets the game audio come back
# smoothly instead of snapping.
DUCK_THRESHOLD = 0.015
DUCK_RATIO = 20
DUCK_ATTACK_MS = 15
DUCK_RELEASE_MS = 400

# The voice-over is normalised to broadcast-ish loudness so a quiet
# phone recording still sits clearly above the game audio.
VOICE_LOUDNORM_I = -16
VOICE_LOUDNORM_TP = -1.5


def _run(cmd, timeout=60):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


def probe_duration(path):
    """Duration of a media file in seconds, or 0.0 when unknown."""
    try:
        res = _run(["ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of", "csv=p=0", str(path)])
        return float(res.stdout.strip() or 0)
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0.0


def has_audio(path):
    """True when the file has at least one audio stream."""
    try:
        res = _run(["ffprobe", "-v", "error", "-select_streams", "a",
                    "-show_entries", "stream=index", "-of", "csv=p=0",
                    str(path)])
        return bool(res.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def build_voiceover_filter(clip_duration, at_seconds, clip_has_audio):
    """The ffmpeg filter graph for the voice-over mix.

    ``[1:a]`` is the recorded line: normalised, trimmed to the time
    left in the clip (a too-long line is cut at the end, not dropped),
    delayed to ``at_seconds``, then split into the voice itself and a
    sidechain signal. The clip audio (or silence when the clip has no
    audio stream) is ducked by that sidechain while the line plays,
    then mixed with the voice. Output label is ``[a]``.
    """
    at = max(0.0, float(at_seconds))
    ms = int(at * 1000)
    # Trim the line to the time left in the clip (when the duration is
    # known); without this a long line would stretch the clip.
    trim = f",atrim=end={clip_duration - at:.2f}" if clip_duration else ""
    voice = (
        f"[1:a]loudnorm=I={VOICE_LOUDNORM_I}:TP={VOICE_LOUDNORM_TP},"
        f"aresample=48000,aformat=channel_layouts=stereo{trim},"
        f"adelay={ms}|{ms},apad,asplit=2[v][sc]"
    )
    game = (
        "[0:a]" if clip_has_audio
        else "anullsrc=r=48000:cl=stereo,"
    ) + "aresample=48000,aformat=channel_layouts=stereo[g]"
    duck = (
        f"[g][sc]sidechaincompress=threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}"
        f":attack={DUCK_ATTACK_MS}:release={DUCK_RELEASE_MS}:makeup=1[duck]"
    )
    mix = "[duck][v]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]"
    return ";".join([voice, game, duck, mix])


def add_voiceover(clip_path, voice_path, at_seconds, output_path):
    """Mix a recorded line into a clip at ``at_seconds``.

    The video stream is copied untouched; only the audio is rebuilt.
    Returns the output path. Raises ValueError for a silent recording
    or a start time past the end of the clip, and RuntimeError when
    ffmpeg fails.
    """
    clip_path, voice_path, output_path = str(clip_path), str(voice_path), str(output_path)
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Voice-over recording not found: {voice_path}")
    if not has_audio(voice_path):
        raise ValueError("Your recording has no sound in it: record again or pick another file")
    at = max(0.0, float(at_seconds))
    duration = probe_duration(clip_path)
    if duration and at > duration - 1:
        raise ValueError(
            f"The voice starts at {at:.1f} s but the clip is only {duration:.1f} s long: "
            "pick an earlier moment")
    graph = build_voiceover_filter(duration, at, has_audio(clip_path))
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", clip_path, "-i", voice_path,
           "-filter_complex", graph,
           "-map", "0:v", "-map", "[a]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-t", f"{duration:.3f}" if duration else "86400",
           output_path]
    res = _run(cmd, timeout=max(120, int(duration * 4) if duration else 120))
    if res.returncode != 0:
        last_line = (res.stderr.strip().splitlines() or ["ffmpeg failed"])[-1][:200]
        raise RuntimeError(f"Could not add the voice-over: {last_line}")
    return output_path
