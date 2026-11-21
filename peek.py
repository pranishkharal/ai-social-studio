"""Peek before rendering (gta6-clipper feature 8).

For each picked moment, save a 4-frame strip — start, one third, two
thirds, end — as one JPG. A quick look at the strip tells Sara (or the
pipeline log) whether the pick is the right game, whether the hook
matches the picture, and whether a face is visible, so a bad pick
(wrong game, menu screen, logo card) can be dropped before spending
minutes rendering it.

- Local files: frames are extracted directly with ffmpeg.
- Remote moments (a VOD URL): only that moment is downloaded, at
  240p, via yt-dlp ``--download-sections`` — a few seconds of work
  per pick instead of a whole video.

A failed peek never blocks a render: callers treat it as best-effort
and log the failure.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Frames are scaled to this height so the strip stays small.
FRAME_HEIGHT = 240
# How long a remote 240p section download may take before one retry.
DOWNLOAD_TIMEOUT_SECONDS = 90
DOWNLOAD_RETRIES = 2


def peek_times(start, end):
    """The 4 sample times inside [start, end): start+1, a third, two
    thirds, end-1. One second in from each edge so the frames show the
    moment itself, not the cut point."""
    start, end = float(start), float(end)
    return [start + 1, start + (end - start) / 3,
            start + 2 * (end - start) / 3, end - 1]


def _run(cmd, timeout=60):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


def _download_moment_section(url, start, end, dest, timeout=DOWNLOAD_TIMEOUT_SECONDS):
    """Download just [start, end) of a remote video at 240p with yt-dlp.

    Retries once on a stall (YouTube sometimes throttles a section to
    a crawl); raises RuntimeError when both attempts fail.
    """
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "-S", "res:240",
           "--download-sections", f"*{start:.2f}-{end:.2f}",
           "--merge-output-format", "mp4", "-o", str(dest), url]
    last_error = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            res = _run(cmd, timeout=timeout)
            if res.returncode == 0 and Path(dest).exists():
                return str(dest)
            last_error = RuntimeError("yt-dlp failed: " + (res.stderr or "").strip()[-120:])
        except subprocess.TimeoutExpired:
            last_error = RuntimeError(
                f"download stalled ({timeout} s) for {start:.0f}-{end:.0f}s")
    raise last_error


def _grab_frame(source, at, dest):
    """One frame from ``source`` at ``at`` seconds, scaled to 240p."""
    res = _run(["ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{max(at, 0):.2f}", "-i", str(source),
                "-frames:v", "1", "-vf", f"scale=-2:{FRAME_HEIGHT}",
                str(dest)], timeout=60)
    return res.returncode == 0 and Path(dest).exists()


def make_peek_strip(source, start, end, output_path, download_fn=None):
    """Save a 4-frame strip for the moment [start, end) to ``output_path``.

    ``source`` is a local video path or a remote URL. Returns the
    output path as a string. ``download_fn`` (tests only) replaces the
    yt-dlp download with a fake ``(url, start, end, dest) -> path``.
    """
    start, end = float(start), float(end)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if Path(source).is_file():
            video, offset = str(source), 0.0
        else:
            video = str(tmp / "moment.mp4")
            download = download_fn or _download_moment_section
            download(str(source), start, end, video)
            offset = start  # the download starts at the moment, so frame times are relative
        frame_inputs = []
        for i, t in enumerate(peek_times(start, end)):
            frame = tmp / f"frame{i}.jpg"
            if _grab_frame(video, t - offset, frame):
                frame_inputs += ["-i", str(frame)]
        if not frame_inputs:
            raise RuntimeError(f"Peek failed: no frames extracted from {source}")
        n = len(frame_inputs) // 2
        hstack = f"hstack=inputs={n}" if n > 1 else "null"
        res = _run(["ffmpeg", "-y", "-loglevel", "error", *frame_inputs,
                    "-filter_complex", hstack, str(output_path)], timeout=60)
        if res.returncode != 0 or not output_path.exists():
            raise RuntimeError(f"Peek failed: could not assemble the strip for {source}")
    return str(output_path)


def peek_moment(name, source, start, end, out_dir, download_fn=None):
    """Peek one moment and save ``<out_dir>/<name>_peek.jpg``.

    Returns the strip path, or None when the peek failed (the caller
    logs the reason and carries on — a failed peek never blocks a
    render).
    """
    try:
        return make_peek_strip(source, start, end,
                               Path(out_dir) / f"{name}_peek.jpg",
                               download_fn=download_fn)
    except Exception as e:
        print(f"peek failed for {name}: {e}", flush=True)
        return None


def peek_picks(picks_file, out_dir="clips/peek", download_fn=None):
    """Peek every pick in a picks.json file (``[{"name", "url",
    "start", "end", ...}]``). Returns {name: strip_path or None}."""
    picks = json.loads(Path(picks_file).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for pick in picks:
        results[pick["name"]] = peek_moment(
            pick["name"], pick["url"], pick["start"], pick["end"],
            out_dir, download_fn=download_fn)
    return results
