"""
Dead-space removal for ai-social-studio clips (feature 9).

Uses Whisper word-level timestamps to drop long silent gaps between words
before rendering, so clips stay fast-paced (a proven retention trick for
shorts). The pipeline docstring promises "dead space elimination" and
``clip_intelligence_prxno.py`` already measures silence seconds -- this
module is the part that actually cuts it out.

Environment variables (read at call time, so tests can monkeypatch them):
    SILENCE_MAX_GAP_SECONDS  Silent gap longer than this is cut.
                             Default: 1.5
    SILENCE_PAD_SECONDS      Speech kept around each word, so cuts never
                             clip the start/end of a word. Default: 0.15

How cutting works:
    ``compute_keep_segments`` turns word timings into a list of
    (start, end) keep-segments. ``cut_silence`` extracts each segment with
    ffmpeg stream copy (``-ss``/``-to`` + ``-c copy``), then joins them with
    the concat demuxer. Stream copy is fast and lossless, but cuts land on
    the nearest keyframe, so boundaries can shift by a few frames -- the
    pad exists partly to absorb that. If stream copy fails (some
    containers/codecs refuse it), the concat falls back to a re-encode
    (libx264 + AAC).

Caveat for the pipeline: cutting removes time, so word timestamps from the
original clip no longer line up with the trimmed clip. Use
``remap_words`` to shift word timings into trimmed-clip time before
generating ASS subtitles from them.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MAX_GAP_SECONDS = 1.5
DEFAULT_PAD_SECONDS = 0.15


def _env_float(name, default):
    """Read a float env var, falling back to ``default`` on bad values."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def max_gap_seconds():
    """Silent gap threshold in seconds (env SILENCE_MAX_GAP_SECONDS)."""
    return _env_float("SILENCE_MAX_GAP_SECONDS", DEFAULT_MAX_GAP_SECONDS)


def pad_seconds():
    """Padding kept around speech in seconds (env SILENCE_PAD_SECONDS)."""
    return _env_float("SILENCE_PAD_SECONDS", DEFAULT_PAD_SECONDS)


def _creationflags():
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def compute_keep_segments(words, max_gap=None, pad=None):
    """
    Compute keep-segments from Whisper word timestamps.

    Args:
        words: list of {"word": str, "start": float, "end": float} dicts,
            as returned by AIStoryEngine.transcribe_clip().
        max_gap: silent gap longer than this (seconds) is dropped.
            Defaults to env SILENCE_MAX_GAP_SECONDS (1.5).
        pad: seconds of audio kept around each speech run so words are
            never clipped. Defaults to env SILENCE_PAD_SECONDS (0.15).

    Returns:
        List of (start, end) tuples in seconds, sorted and non-overlapping.
        Empty list when there are no words.
    """
    if max_gap is None:
        max_gap = max_gap_seconds()
    if pad is None:
        pad = pad_seconds()

    timed = []
    for w in words or []:
        try:
            start = float(w["start"])
            end = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        timed.append((start, end))
    if not timed:
        return []

    timed.sort()
    segments = []
    seg_start = max(0.0, timed[0][0] - pad)
    seg_end = timed[0][1] + pad

    for start, end in timed[1:]:
        gap = start - (seg_end - pad)  # gap between this word and previous word's end
        if gap > max_gap:
            segments.append((round(seg_start, 3), round(seg_end, 3)))
            seg_start = max(0.0, start - pad)
            seg_end = end + pad
        else:
            seg_end = max(seg_end, end + pad)

    segments.append((round(seg_start, 3), round(seg_end, 3)))

    # Merge any segments that overlap (can happen when pad bridges a gap
    # that was only just under the threshold).
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def remap_words(words, segments):
    """
    Shift word timestamps from original-clip time into trimmed-clip time.

    Words falling inside a dropped gap are dropped (they should not exist
    -- segments are built from the words -- but be safe). Use this before
    generating ASS subtitles for a silence-trimmed clip.

    Args:
        words: original word dicts with "start"/"end".
        segments: keep-segments from compute_keep_segments().

    Returns:
        New list of word dicts with remapped "start"/"end" (other keys kept).
    """
    if not segments:
        return []
    remapped = []
    offset = 0.0
    prev_end = 0.0
    for seg_start, seg_end in segments:
        # Time removed before this segment starts.
        offset += seg_start - prev_end
        for w in words or []:
            try:
                start = float(w["start"])
                end = float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if seg_start <= start < seg_end:
                new_word = dict(w)
                new_word["start"] = round(max(0.0, start - offset), 2)
                new_word["end"] = round(max(new_word["start"], end - offset), 2)
                remapped.append(new_word)
        prev_end = seg_end
    remapped.sort(key=lambda w: w["start"])
    return remapped


def _probe_duration(path):
    """Total duration in seconds via ffprobe, or None if unavailable."""
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
            creationflags=_creationflags(),
        )
        if res.returncode == 0:
            return float(res.stdout.strip())
    except Exception:
        pass
    return None


def _run_ffmpeg(cmd, timeout=300):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        creationflags=_creationflags(),
    )


def cut_silence(input_path, words, output_path, max_gap=None, pad=None):
    """
    Cut silent gaps out of a clip using Whisper word timestamps.

    Args:
        input_path: source video file.
        words: word dicts with "start"/"end" (see compute_keep_segments).
        output_path: where the trimmed clip is written.
        max_gap / pad: overrides for the env-driven defaults.

    Returns:
        (output_path, seconds_removed). When there is nothing to cut
        (no words, or no gap over the threshold) the input is copied
        unchanged and seconds_removed is 0.0.

    Raises:
        FileNotFoundError: input does not exist.
        RuntimeError: ffmpeg failed even with the re-encode fallback.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source clip not found: {input_path}")

    segments = compute_keep_segments(words, max_gap=max_gap, pad=pad)

    if not segments:
        # No word timings to cut by: leave the clip untouched.
        shutil.copyfile(input_path, output_path)
        return output_path, 0.0

    total_duration = _probe_duration(input_path)
    kept_duration = sum(end - start for start, end in segments)

    if len(segments) == 1:
        # Single segment: only worth cutting if it trims leading/trailing
        # silence. Leading silence before the first word is part of the
        # gap logic (segment starts at first word - pad), trailing silence
        # after the last word is only cut when we know the total duration.
        seg_start, seg_end = segments[0]
        trims_head = seg_start > 0.05
        trims_tail = (
            total_duration is not None and total_duration - seg_end > 0.05
        )
        if not trims_head and not trims_tail:
            shutil.copyfile(input_path, output_path)
            return output_path, 0.0

    if total_duration is not None:
        seconds_removed = max(0.0, round(total_duration - kept_duration, 2))
    else:
        # No ffprobe: estimate from the dropped gaps (gap minus the pads
        # kept on both sides). Under-reports leading/trailing trims.
        seconds_removed = 0.0
        for (_, prev_end), (next_start, _) in zip(segments, segments[1:]):
            seconds_removed += max(0.0, next_start - prev_end)
        seconds_removed = round(seconds_removed, 2)

    tmp_dir = Path(tempfile.mkdtemp(prefix="silence_cut_"))
    try:
        ext = Path(input_path).suffix or ".mp4"
        segment_files = []
        for i, (start, end) in enumerate(segments):
            seg_file = tmp_dir / f"seg_{i:03d}{ext}"
            res = _run_ffmpeg([
                "ffmpeg", "-y",
                "-ss", str(start), "-to", str(end),
                "-i", str(input_path),
                "-c", "copy",
                str(seg_file),
            ])
            if res.returncode != 0 or not seg_file.exists():
                raise RuntimeError(
                    f"ffmpeg segment cut failed ({start}-{end}s): "
                    f"{res.stderr[-300:]}"
                )
            segment_files.append(seg_file)

        concat_list = tmp_dir / "concat.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg_file in segment_files:
                # Concat demuxer: single-quoted paths, escape inner quotes.
                safe = str(seg_file).replace("'", "'\\''")
                f.write(f"file '{safe}'\n")

        # Fast path: stream copy concat (lossless, near-instant).
        res = _run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(output_path),
        ])
        if res.returncode != 0:
            # Fallback: some containers/codecs refuse stream-copy concat.
            res = _run_ffmpeg([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ])
        if res.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError(f"ffmpeg concat failed: {res.stderr[-300:]}")

        return output_path, seconds_removed
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
