"""VOD / multi-source clipping (Feature 6 of the TikTok clipping upgrades).

Clip past videos, not just live Kick streams: paste a Kick VOD, Twitch
VOD, YouTube video or podcast link (or point at a local video file) and
the engine finds the loudest moment in it and runs it through the normal
clip pipeline (hype check, Whisper transcription, metadata, 9:16 render).

This closes the gap with @kicksclipping, the biggest of the three TikTok
clipping accounts studied: it clips Twitch and podcasts too, while this
engine was Kick-live-only.

Flow:
    source URL or file
      -> resolve_vod_source()   (download with yt-dlp, or use the file)
      -> find_best_moment()     (scan windows with HypeDetector)
      -> extract the best window (ffmpeg stream copy, fast)
      -> ClipPipeline.process_moment() (the existing pipeline, unchanged)

Only the standard library plus repo modules are imported here. yt-dlp is
imported lazily inside download_vod(), so local-file clipping works
without it installed; downloading a URL without yt-dlp raises a
plain-English VodError with the install command.

TLS is verified normally (yt-dlp's default). No certificate checks are
disabled and no custom CA handling lives here.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from config import VOD_DOWNLOAD_DIR
from hype_detector import HypeDetector
from pipeline import ClipPipeline

# yt-dlp format: best video at 720p or below plus best audio, falling back
# to a single combined file at 720p or below. 720p keeps downloads light
# (the clip is rendered at 1080x1920 from it, which is fine for TikTok).
YT_DLP_FORMAT = "bv*[height<=720]+ba/b[height<=720]"

# URL path segments that are NOT streamer names (Kick/Twitch routes).
_NON_STREAMER_SEGMENTS = {
    "video", "videos", "clip", "clips", "directory", "embed",
    "search", "downloads", "login", "signup", "settings",
}


class VodError(Exception):
    """Any failure in the VOD path, with a plain-English message."""


def detect_source_kind(url_or_path) -> str:
    """Classify a VOD source.

    Returns "local_file" if the path exists on disk, "kick" /
    "twitch" / "youtube" for those sites' URLs, and "unknown_url"
    for anything else (a missing path or an unrecognised URL).
    """
    source = str(url_or_path or "").strip()
    if source and os.path.exists(source):
        return "local_file"
    lowered = source.lower()
    if "kick.com" in lowered:
        return "kick"
    if "twitch.tv" in lowered:
        return "twitch"
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    return "unknown_url"


def download_vod(url, out_dir=None) -> str:
    """Download a video URL with yt-dlp and return the saved file path.

    yt-dlp is imported lazily so this module works without it; if it is
    missing, a VodError explains how to install it. Downloads are capped
    at 720p to keep them light. The default download folder is
    config.VOD_DOWNLOAD_DIR (``<repo>/vod``).
    """
    try:
        import yt_dlp
    except ImportError:
        raise VodError(
            "yt-dlp is not installed — run: pip install yt-dlp"
        ) from None

    target_dir = Path(out_dir) if out_dir else Path(VOD_DOWNLOAD_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": YT_DLP_FORMAT,
        # restrictfilenames sanitizes the title into a safe ASCII
        # filename (no spaces, quotes or characters that break shells).
        "restrictfilenames": True,
        "outtmpl": str(target_dir / "%(title)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # After a download, requested_downloads holds the final
            # file path (after any video+audio merge).
            for entry in info.get("requested_downloads") or []:
                filepath = entry.get("filepath")
                if filepath and os.path.exists(filepath):
                    return str(filepath)
            # Fallback: the filename yt-dlp would use for this video.
            candidate = ydl.prepare_filename(info)
            if candidate and os.path.exists(candidate):
                return str(candidate)
    except VodError:
        raise
    except Exception as e:
        raise VodError(f"Could not download {url}: {e}") from None
    raise VodError(
        f"Download finished but the video file could not be found in {target_dir}"
    )


def resolve_vod_source(url_or_path, out_dir=None) -> str:
    """Turn a source (URL or local path) into a local video file path.

    Local file -> returned as-is (after verifying it exists).
    Kick / Twitch / YouTube URL -> downloaded with yt-dlp.
    Anything else -> VodError.
    """
    kind = detect_source_kind(url_or_path)
    if kind == "local_file":
        return str(url_or_path)
    if kind in ("kick", "twitch", "youtube"):
        return download_vod(url_or_path, out_dir=out_dir)
    if kind == "unknown_url" and str(url_or_path or "").strip():
        raise VodError(
            f"Not a file on this computer and not a Kick, Twitch or "
            f"YouTube link: {url_or_path}"
        )
    raise VodError("No video source given — pass a video file or a video link.")


def _get_duration(video_path: str):
    """Video duration in seconds (float), or None if it cannot be read."""
    # ffprobe is the precise route...
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=30)
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    # ...fall back to parsing ffmpeg's "Duration: HH:MM:SS.xx" line.
    try:
        res = subprocess.run(
            ["ffmpeg", "-i", str(video_path)],
            capture_output=True, text=True, timeout=30)
        match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", res.stderr)
        if match:
            return (int(match.group(1)) * 3600
                    + int(match.group(2)) * 60
                    + float(match.group(3)))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _extract_window(video_path: str, start: float, length: float,
                    output_path: str) -> str:
    """Cut one window out of a video with ffmpeg stream copy (fast).

    ``-ss``/``-t`` are output options (after ``-i``) so the cut is
    accurate: input-side ``-ss`` with stream copy can only start on a
    keyframe, which would shift the window.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-ss", str(start), "-t", str(length),
        "-c", "copy", str(output_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise VodError(f"Could not cut the video at {start}s: {e}") from None
    if res.returncode != 0 or not os.path.exists(output_path):
        detail = (res.stderr or "").strip()
        if len(detail) > 200:
            detail = detail[-200:]
        raise VodError(
            f"Could not cut the video at {start}s"
            + (f": {detail}" if detail else "")
        )
    return str(output_path)


def find_best_moment(video_path, window_seconds=30, max_windows=40,
                     signals=None) -> dict:
    """Find the highest-hype window in a video.

    Scans the video in ``window_seconds`` windows, scoring each one with
    the existing HypeDetector (audio energy via ffmpeg volumedetect) so
    the scoring logic is shared with live clipping, never duplicated.
    For long videos the windows are strided so at most ``max_windows``
    are scanned across the whole duration.

    ``signals`` is the viewer-signal dict from
    ``vod_signals.fetch_signals_for_source`` ({"clips": [...],
    "heatmap": [...]}): a window viewers already clipped or replayed
    earns a bonus of up to 35 points on top of its audio score, so a
    moment the audience voted for beats an equally loud one nobody
    cared about. Without signals the scoring is pure audio, as before.

    Returns {"start": float, "end": float, "score": float} for the best
    window (plus "signal_bonus" and "signal_tags" when signals were
    given). Raises VodError if the file is missing or cannot be scanned.
    """
    if not os.path.exists(str(video_path)):
        raise VodError(f"Video file not found: {video_path}")
    if window_seconds <= 0:
        raise VodError("window_seconds must be more than 0.")
    if max_windows < 1:
        raise VodError("max_windows must be at least 1.")

    duration = _get_duration(str(video_path))
    if duration is None or duration <= window_seconds:
        starts = [0.0]
    else:
        # Non-overlapping windows when they fit, otherwise stride so the
        # scan covers the whole video in at most max_windows windows.
        stride = max(float(window_seconds), duration / float(max_windows))
        starts = []
        start = 0.0
        while start < duration and len(starts) < max_windows:
            starts.append(start)
            start += stride

    detector = HypeDetector()
    signal_clips = (signals or {}).get("clips") or []
    signal_heatmap = (signals or {}).get("heatmap") or []
    have_signals = bool(signal_clips or signal_heatmap)
    if have_signals:
        # Lazy import: vod_signals is only needed when signals exist.
        from vod_signals import window_signal_bonus
    best = None
    with tempfile.TemporaryDirectory(prefix="vod_windows_") as tmpdir:
        for index, start in enumerate(starts):
            window_path = os.path.join(tmpdir, f"window_{index}.mp4")
            try:
                _extract_window(str(video_path), start, window_seconds,
                                window_path)
            except VodError:
                continue
            result = detector.analyze_audio_energy(window_path)
            score = float(result.get("score", 0.0))
            end = start + window_seconds
            if duration is not None:
                end = min(end, duration)
            candidate = {"start": round(float(start), 2),
                         "end": round(float(end), 2),
                         "score": score}
            if have_signals:
                sig = window_signal_bonus(signal_clips, signal_heatmap,
                                          start, end)
                candidate["score"] = round(
                    min(100.0, score + sig["bonus"]), 1)
                candidate["signal_bonus"] = sig["bonus"]
                tags = (sig["clipped"]["tag"] + sig["replayed"]["tag"])
                candidate["signal_tags"] = tags.strip()
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        raise VodError(
            f"Could not scan the video for hype moments: {video_path}"
        )
    return best


def _streamer_from_url(url: str):
    """Best-effort streamer/channel name from a Kick or Twitch URL.

    kick.com/<slug> or twitch.tv/<slug> -> <slug>. YouTube watch links
    carry no channel in the URL, so they return None (caller falls back
    to "vod"). Route words like /videos/ or /clip/ are not streamers.
    """
    match = re.search(r"(?:kick\.com|twitch\.tv)/([A-Za-z0-9_]+)",
                      str(url or ""))
    if match and match.group(1).lower() not in _NON_STREAMER_SEGMENTS:
        return match.group(1)
    return None


def clip_vod(source, streamer=None, duration=25, layout=None,
             out_dir=None, window_seconds=None, use_vod_signals=True) -> dict:
    """Clip the best moment from a past video (VOD) end to end.

    1. Resolves the source (downloads the URL, or uses the local file).
    2. Finds the highest-hype window (scan window defaults to the clip
       duration). For Twitch/YouTube URLs, viewer signals are fetched
       first (Twitch clips viewers already made, YouTube most-replayed
       peaks) and folded into the window scoring, unless
       ``use_vod_signals`` is False. A failed fetch just means
       audio-only scoring -- it never fails the clip.
    3. Extracts that window and runs it through the normal pipeline
       (hype check, Whisper transcription, metadata, 9:16 render).

    Returns the pipeline result dict plus:
        "vod_source": the source URL or file that was clipped
        "vod_window": {"start", "end", "score"} of the chosen moment

    The streamer name defaults to the slug in a Kick/Twitch URL, else
    "vod". The extracted window is KEPT in vod/windows/ (named after
    the source and its start/end times), so the pipeline's resume and
    re-render paths keep working — they need the raw clip file on disk.
    """
    video_path = resolve_vod_source(source, out_dir=out_dir)
    scan_window = window_seconds if window_seconds is not None else duration
    signals = None
    if use_vod_signals and detect_source_kind(source) in ("twitch", "youtube"):
        try:
            from vod_signals import fetch_signals_for_source
            signals = fetch_signals_for_source(source)
        except Exception:
            signals = None  # viewer signals are a bonus, never a blocker
    if signals:
        moment = find_best_moment(video_path, window_seconds=scan_window,
                                  signals=signals)
    else:
        moment = find_best_moment(video_path, window_seconds=scan_window)
    name = streamer or _streamer_from_url(str(source)) or "vod"

    windows_dir = Path(out_dir or VOD_DOWNLOAD_DIR) / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(video_path).stem)[:40]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_path = str(windows_dir /
                      f"{stem}_{int(moment['start'])}-{int(moment['end'])}_{stamp}.mp4")
    _extract_window(video_path, moment["start"],
                    moment["end"] - moment["start"], window_path)
    result = ClipPipeline().process_moment(
        streamer=name, duration=duration,
        raw_clip_path=window_path, layout=layout)

    result = dict(result)
    result["vod_source"] = source
    result["vod_window"] = moment
    return result


def main(argv=None) -> int:
    """CLI: python3 vod_clipper.py <url-or-file> [options]."""
    parser = argparse.ArgumentParser(
        description="Clip the best moment from a past video: a Kick VOD, "
                    "Twitch VOD, YouTube video or local file.")
    parser.add_argument("source",
                        help="Video URL (kick.com, twitch.tv, youtube.com) "
                             "or a local video file.")
    parser.add_argument("--streamer", default=None,
                        help="Streamer name for the clip (default: taken "
                             "from a Kick/Twitch URL, else 'vod').")
    parser.add_argument("--duration", type=int, default=25,
                        help="Clip length in seconds (default: 25).")
    parser.add_argument("--layout", default=None,
                        help="Render layout, e.g. blurred_backdrop "
                             "(default: the streamer's profile layout).")
    parser.add_argument("--out-dir", default=None,
                        help="Where to save downloaded VODs "
                             "(default: the vod/ folder).")
    parser.add_argument("--window-seconds", type=int, default=None,
                        help="Seconds scanned per window when hunting for "
                             "the best moment (default: same as --duration).")
    parser.add_argument("--no-vod-signals", action="store_true",
                        help="Skip the Twitch-clips / YouTube most-replayed "
                             "viewer signals and score windows on audio only.")
    args = parser.parse_args(argv)
    try:
        result = clip_vod(
            args.source, streamer=args.streamer, duration=args.duration,
            layout=args.layout, out_dir=args.out_dir,
            window_seconds=args.window_seconds,
            use_vod_signals=not args.no_vod_signals)
    except VodError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
