"""
broll.py — Contextual stock B-roll from Pexels + Ken Burns / crossfade helpers.

Pexels search/download flow adapted from opensource-clipping
(https://github.com/NaufalRizqullah/opensource-clipping, MIT licence),
clipping/studio/broll.py (download_pexels_broll): same API endpoint and
params, HD-MP4 preference, and a run-wide de-duplication set so the same
stock clip is never reused twice in one run.

Needs a Pexels API key in the PEXELS_API_KEY env var (free at
https://www.pexels.com/api/). With no key set, search_broll() is a graceful
no-op returning [] — B-roll is simply skipped, never an error.

Only the standard library is used (urllib) — no new dependency.
"""

import json
import os
import shutil
import urllib.parse
import urllib.request

PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
USER_AGENT = "ai-social-studio/1.0 (B-roll fetcher)"
DEFAULT_PER_PAGE = 10
DEFAULT_SEARCH_TIMEOUT = 15
DEFAULT_DOWNLOAD_TIMEOUT = 60

# Run-wide de-duplication: Pexels video ids already handed out, so the same
# stock clip is not reused across B-roll slots in one process.
USED_PEXELS_IDS = set()


def _best_mp4_file(video):
    """Pick the best downloadable file for a Pexels video: HD MP4 first, then largest."""
    files = [vf for vf in video.get("video_files", []) if vf.get("file_type") == "video/mp4"]
    if not files:
        return None
    files.sort(key=lambda vf: (
        vf.get("quality") != "hd",
        -(vf.get("width") or 0),
        -(vf.get("height") or 0),
    ))
    return files[0]


def search_broll(query, per_page=DEFAULT_PER_PAGE, api_key=None, orientation="portrait"):
    """
    Search Pexels for stock video matching ``query``.

    Args:
        query: search terms, e.g. "crowd cheering" or "city night".
        per_page: max results to request from Pexels (default 10).
        api_key: Pexels API key; defaults to the PEXELS_API_KEY env var.
            With no key, returns [] (graceful no-op).
        orientation: "portrait" for 9:16 clips (default) or "landscape".

    Returns:
        A list of {"id", "url", "width", "height", "duration"} dicts
        (duration in seconds). Videos already returned earlier in this run
        are excluded when fresh ones are available; if the whole pool is
        exhausted the pool resets, as in opensource-clipping.
        Returns [] on any API/network error.
    """
    key = api_key or os.getenv("PEXELS_API_KEY")
    if not key:
        return []
    params = urllib.parse.urlencode({
        "query": query,
        "orientation": orientation,
        "per_page": per_page,
        "size": "large",
    })
    req = urllib.request.Request(
        f"{PEXELS_SEARCH_URL}?{params}",
        headers={"Authorization": key, "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_SEARCH_TIMEOUT) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"[broll] Pexels search failed for {query!r}: {e}")
        return []

    videos = data.get("videos") or []
    if not videos:
        return []
    fresh = [v for v in videos if v.get("id") not in USED_PEXELS_IDS]
    if not fresh:
        # Pool exhausted for this query — reset and reuse.
        fresh = videos
    results = []
    for video in fresh:
        best = _best_mp4_file(video)
        if best is None or not best.get("link"):
            continue
        USED_PEXELS_IDS.add(video.get("id"))
        results.append({
            "id": video.get("id"),
            "url": best["link"],
            "width": best.get("width"),
            "height": best.get("height"),
            "duration": video.get("duration"),
        })
    return results


def download_broll(url, dest, timeout=DEFAULT_DOWNLOAD_TIMEOUT):
    """
    Download a B-roll file (typically a Pexels CDN link) to ``dest``.

    Downloads to ``dest + ".part"`` and renames atomically on success so a
    failed download never leaves a truncated file at ``dest``. Parent
    directories are created. Returns the destination path as a string, or
    None on failure. No API key is needed (CDN links are public).
    """
    dest = str(dest)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temp_path = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(temp_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(temp_path, dest)
        return dest
    except Exception as e:
        print(f"[broll] Download failed for {url!r}: {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        return None


def build_kenburns_filter(duration, fps=30, width=1080, height=1920,
                          zoom_start=1.0, zoom_end=1.10,
                          input_label="[broll]", output_label="[broll_kb]"):
    """
    Build a Ken Burns (slow zoom-in) filter snippet for a B-roll clip.

    The clip is upscaled 2x for zoom headroom, then zoompan zooms smoothly
    from ``zoom_start`` to ``zoom_end`` (default 1.0 -> 1.10, as in
    opensource-clipping) across the clip's ``duration`` seconds.

    Args:
        duration: B-roll segment length in seconds.
        fps: frame rate of the B-roll stream.
        width / height: output canvas size (default 1080x1920).
        input_label: ffmpeg label of the B-roll video stream.
        output_label: label the zoomed stream is written to.

    Returns:
        A filter string (one chain, no trailing semicolon) to splice into a
        filter_complex, e.g. ``[1:v]scale=...,zoompan=...[broll_kb]``.
    """
    frames = max(1, int(round(float(duration) * fps)))
    increment = (zoom_end - zoom_start) / frames
    return (
        f"{input_label}scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
        f"zoompan=z='min({zoom_start:.4f}+{increment:.6f}*on,{zoom_end:.4f})':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps}"
        f"{output_label}"
    )


def build_broll_xfade_filter(base_label="[base]", broll_label="[broll_kb]",
                             start=0.0, end=5.0, duration=30.0, fade=0.3,
                             output_label="[broll_out]"):
    """
    Build an xfade crossfade chain that cuts a full-screen B-roll segment
    into the base clip between ``start`` and ``end`` seconds, fading in and
    out over ``fade`` seconds (default 0.3 s, as in opensource-clipping).

    Args:
        base_label: ffmpeg label of the base video stream (full clip).
        broll_label: ffmpeg label of the B-roll video stream (already
            Ken-Burnsed / scaled to the canvas; at least end-start long).
        start / end: B-roll window in seconds on the clip timeline.
        duration: total clip duration in seconds.
        fade: crossfade duration in seconds; must be shorter than the
            shortest segment it touches.
        output_label: label the finished stream is written to.

    Returns:
        (filter_complex_string, output_label). VIDEO ONLY — audio is
        untouched, so the caller still maps it separately (``-map 0:a``).
    """
    start = float(start)
    end = float(end)
    duration = float(duration)
    fade = float(fade)
    if not (0 <= start < end <= duration):
        raise ValueError(f"Invalid B-roll window: 0 <= {start} < {end} <= {duration} required")
    if fade <= 0 or fade >= (end - start):
        raise ValueError(f"fade ({fade}) must be > 0 and shorter than the B-roll window ({end - start})")

    parts = []
    segments = []  # (label, length_seconds)
    if start > 0:
        parts.append(f"{base_label}trim=0:{start:.2f},setpts=PTS-STARTPTS[br_pre]")
        segments.append(("[br_pre]", start))
    parts.append(f"{broll_label}trim=0:{end - start:.2f},setpts=PTS-STARTPTS[br_mid]")
    segments.append(("[br_mid]", end - start))
    if end < duration:
        parts.append(f"{base_label}trim={end:.2f}:{duration:.2f},setpts=PTS-STARTPTS[br_post]")
        segments.append(("[br_post]", duration - end))

    if len(segments) == 1:
        # B-roll covers the whole clip: no crossfade needed.
        parts[-1] = parts[-1].replace("[br_mid]", output_label)
        return ";".join(parts), output_label

    cur_label, cur_len = segments[0]
    for i, (next_label, next_len) in enumerate(segments[1:]):
        is_last = i == len(segments) - 2
        out = output_label if is_last else f"[br_x{i}]"
        # The transition starts `fade` before the end of the accumulated
        # stream so it overlaps the cut point.
        offset = max(0.0, cur_len - fade)
        parts.append(
            f"{cur_label}{next_label}xfade=transition=fade:duration={fade:.2f}"
            f":offset={offset:.2f}{out}"
        )
        cur_len = cur_len + next_len - fade
        cur_label = out
    return ";".join(parts), output_label


def build_broll_overlay_filter(base_label="[base]", broll_label="[broll_kb]",
                               start=0.0, end=5.0, fade=0.3,
                               output_label="[broll_out]"):
    """
    Alternative to xfade: overlay the B-roll full-screen over the base clip
    between ``start`` and ``end`` seconds, fading in/out over ``fade``
    seconds. The B-roll stream must run on the same timeline as the base
    (e.g. trimmed to the full clip duration with setpts=PTS-STARTPTS).

    Returns (filter_complex_string, output_label). VIDEO ONLY.
    """
    start = float(start)
    end = float(end)
    fade = float(fade)
    if not (0 <= start < end):
        raise ValueError(f"Invalid B-roll window: 0 <= {start} < {end} required")
    if fade <= 0 or fade >= (end - start):
        raise ValueError(f"fade ({fade}) must be > 0 and shorter than the B-roll window ({end - start})")
    filt = (
        f"{broll_label}fade=t=in:st={start:.2f}:d={fade:.2f},"
        f"fade=t=out:st={end - fade:.2f}:d={fade:.2f}[br_faded];"
        f"{base_label}[br_faded]overlay=0:0:enable='between(t,{start:.2f},{end:.2f})'{output_label}"
    )
    return filt, output_label
