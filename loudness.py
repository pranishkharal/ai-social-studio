"""
Loudness normalisation to -14 LUFS (gta6-clipper feature 10).

Every rendered clip should sound the same volume on a phone, whatever
stream it came from. -14 LUFS is the level YouTube Shorts / TikTok
normalise to, so renders target it directly:

1.  **Two-pass loudnorm in the render** — the source audio is measured
    first (``measure_loudnorm_stats``), then the render applies
    ``loudnorm`` with those measured values and ``linear=true``. The
    linear (two-pass) mode avoids the pumping a single-pass dynamic
    loudnorm causes on speech.
2.  **Safety pass** (``fix_loudness``) — after the render, the finished
    file is re-measured; if it lands more than 1.5 LU from -14 LUFS
    (BGM mixing and ducking shift the level a little), only the audio
    is re-normalised and the video is stream-copied untouched.

``measure_lufs`` is also what the QC module (clip_qc.py) checks against,
so the render target and the QC threshold can never drift apart.

Settings are module-local constants with env-var overrides (no config.py
changes — other agents own that file):

    LOUDNESS_TARGET_LUFS   Target integrated loudness (default -14)
    LOUDNESS_TRUE_PEAK     True-peak ceiling in dBTP (default -1.5)
    LOUDNESS_LRA           Loudness range target (default 11)
    LOUDNESS_SAFETY_LU     Safety-pass tolerance in LU (default 1.5)

All helpers shell out to ffmpeg and never raise on a measurement
failure — they return None and the caller renders / proceeds without
normalisation rather than failing a clip over a loudness probe.
"""

import json
import os
import re
import subprocess
from pathlib import Path

TARGET_LUFS = float(os.environ.get("LOUDNESS_TARGET_LUFS", "-14"))
TRUE_PEAK_DBTP = float(os.environ.get("LOUDNESS_TRUE_PEAK", "-1.5"))
LOUDNESS_RANGE = float(os.environ.get("LOUDNESS_LRA", "11"))
SAFETY_TOLERANCE_LU = float(os.environ.get("LOUDNESS_SAFETY_LU", "1.5"))

# Every render's audio leaves at 48 kHz: linear loudnorm upsamples to
# 192 kHz internally and the AAC encoder caps at 96 kHz, so without an
# explicit resample a render comes out at 96 kHz — and the format
# joiner (concat demuxer) fails hard on mixed sample rates.
OUTPUT_SAMPLE_RATE = 48000

# A loudnorm stats JSON block must have all of these as plain numbers.
# Silent audio reports "-inf" for input_i, which is not a number and
# would make ffmpeg fail with "Result too large" if passed back in.
_NUMERIC = re.compile(r"-?\d+(\.\d+)?")
_STATS_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh",
               "target_offset")


def _fmt(value):
    """Format a loudness number for a filter string (no trailing .0)."""
    return f"{value:g}"


def _run_ffmpeg(args, timeout=300):
    return subprocess.run(
        ["ffmpeg", "-hide_banner", *args],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )


def measure_loudnorm_stats(path):
    """First loudnorm pass on a file's audio: measure only, no output.

    Returns the parsed loudnorm JSON (input_i, input_tp, input_lra,
    input_thresh, target_offset, ...) or None when the file has no
    audio, the audio is silent (-inf loudness), or ffmpeg fails.
    """
    try:
        res = _run_ffmpeg([
            "-v", "info", "-i", str(path),
            "-vn", "-af",
            f"loudnorm=I={_fmt(TARGET_LUFS)}:TP={_fmt(TRUE_PEAK_DBTP)}"
            f":LRA={_fmt(LOUDNESS_RANGE)}:print_format=json",
            "-f", "null", "-",
        ])
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        stats = json.loads(re.findall(r"\{[^{}]*\}", res.stderr, re.S)[-1])
    except (IndexError, ValueError):
        return None
    if not all(_NUMERIC.fullmatch(str(stats.get(k, "")))
               for k in _STATS_KEYS):
        return None  # silent audio ("-inf") or a partial parse
    return stats


def loudnorm_filter(stats=None):
    """The loudnorm filter for the render's audio chain.

    With measured stats this is the second pass of two-pass loudnorm
    (``linear=true``); without stats it falls back to a single-pass
    loudnorm, which still normalises but may pump on speech.

    The chain always ends with ``aresample=48000``: linear loudnorm
    upsamples to 192 kHz internally and the AAC encoder caps at
    96 kHz, so without the resample a render comes out at 96 kHz —
    and the format joiner (concat demuxer) fails hard when inputs
    have mixed sample rates. Resampling here keeps every render's
    audio at one uniform rate.
    """
    base = (f"loudnorm=I={_fmt(TARGET_LUFS)}:TP={_fmt(TRUE_PEAK_DBTP)}"
            f":LRA={_fmt(LOUDNESS_RANGE)}")
    if not stats:
        filt = base
    else:
        try:
            filt = (f"{base}:measured_I={stats['input_i']}"
                    f":measured_TP={stats['input_tp']}"
                    f":measured_LRA={stats['input_lra']}"
                    f":measured_thresh={stats['input_thresh']}"
                    f":offset={stats['target_offset']}:linear=true")
        except KeyError:
            filt = base
    return f"{filt},aresample={OUTPUT_SAMPLE_RATE}"


def measure_lufs(path):
    """Integrated loudness of a file in LUFS, or None if unmeasurable.

    Reads the ``input_i`` field of a loudnorm measure pass (the same
    number the two-pass render targets); falls back to an ebur128
    summary read when the loudnorm JSON cannot be parsed.
    """
    stats = measure_loudnorm_stats(path)
    if stats is not None:
        try:
            return float(stats["input_i"])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        res = _run_ffmpeg(["-nostats", "-i", str(path),
                           "-vn", "-af", "ebur128", "-f", "null", "-"])
        matches = re.findall(r"I:\s+(-?[\d.]+) LUFS", res.stderr)
        return float(matches[-1]) if matches else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def fix_loudness(path):
    """Safety pass: re-normalise a finished clip's audio if it is off.

    Measures the finished file; if it is within SAFETY_TOLERANCE_LU of
    the target it is left untouched. Otherwise the audio is
    re-normalised with two-pass linear loudnorm while the video is
    stream-copied (no re-encode, no quality loss). The fix happens in
    a temp file that replaces the original only on success.

    Returns the measured LUFS after the fix (or the original reading
    when no fix was needed), or None when the file could not be
    measured. Never raises — a loudness fix must never fail a render.
    """
    try:
        path = Path(path)
        before = measure_lufs(path)
        if before is None or abs(before - TARGET_LUFS) <= SAFETY_TOLERANCE_LU:
            return before
        stats = measure_loudnorm_stats(path)
        if stats is None:
            return before
        tmp = path.with_name(path.stem + ".loudfix" + path.suffix)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                 "-c:v", "copy", "-af", loudnorm_filter(stats),
                 "-c:a", "aac", "-b:a", "192k", str(tmp)],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=600, check=True,
            )
            after = measure_lufs(tmp)
            os.replace(tmp, path)
            return after
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        return None
