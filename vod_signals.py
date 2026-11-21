"""Viewer-made clips + most-replayed as VOD signals.

Other viewers already voted with their clips and replays -- two free
signals that cost one API-free fetch each, the same idea as
gta6-clipper's ``chat.py`` ``fetch_clips`` / ``clip_tag`` /
``replay_tag`` (MIT), reimplemented here in this repo's style:

* ``[CLIPPED]`` -- a channel's top-100 Twitch clips per period
  (public Twitch GQL, no login) filtered to this VOD; a transcript
  line / window counts when it lies inside a viewer clip or up to 5 s
  before it starts (clips are made right AFTER the moment).
* ``[REPLAYED]`` -- YouTube's most-replayed heatmap from yt-dlp
  metadata; a window counts when the heatmap peaks inside it at >= 3x
  the median (the first 3% of a video is ignored -- intros are always
  replayed).

Integration: ``vod_clipper.find_best_moment`` accepts the signals
returned here (``signals=...``) and adds
``window_signal_bonus`` to each window's audio score, so a window
viewers already clipped or replayed beats an equally loud one that
nobody cared about.

All network access goes through injectable callables (``opener`` /
``metadata_fn``) so tests never touch the network. Every fetch fails
soft: a dead endpoint or a missing yt-dlp returns [] and the caller
just scores on audio alone.

Module-local settings (env overrides; nothing in config.py):
    VOD_SIGNALS_TIMEOUT   default 30 (seconds per fetch)
    VOD_SIGNALS_MAX_BONUS default 35 (cap on the combined bonus)
"""

import json
import math
import os
import re
import urllib.error
import urllib.request

# Twitch's public web client id (used by twitch.tv itself; no login).
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
TWITCH_GQL_URL = "https://gql.twitch.tv/gql"

# Clip periods queried, most recent first. A period can answer "server
# error"; the others still work.
CLIP_PERIODS = ("LAST_DAY", "LAST_WEEK", "LAST_MONTH", "ALL_TIME")

# Clips are made right AFTER the moment, so a window counts when it
# lies inside the clip or up to this many seconds before it starts.
CLIP_LEAD_SECONDS = 5.0

# A heatmap peak counts at >= 3x the median value (and >= 0.5 absolute).
REPLAY_MIN_RATIO = 3.0
REPLAY_MIN_VALUE = 0.5
# Ignore heatmap peaks in the first 3% of the video (intro replays).
REPLAY_IGNORE_START_FRACTION = 0.03

# Bonus budget: clipped moments up to 20, replayed peaks up to 15.
CLIPPED_MAX_BONUS = 20.0
REPLAYED_MAX_BONUS = 15.0

FETCH_TIMEOUT = float(os.getenv("VOD_SIGNALS_TIMEOUT", "30") or 30)
MAX_SIGNAL_BONUS = float(os.getenv("VOD_SIGNALS_MAX_BONUS", "35") or 35)


# ---------------------------------------------------------------------------
# Twitch clips ([CLIPPED])
# ---------------------------------------------------------------------------

def _twitch_gql(query, opener=None, timeout=FETCH_TIMEOUT):
    """One public Twitch GQL query -> parsed JSON ``data`` dict."""
    if opener is not None:
        return opener(query)
    request = urllib.request.Request(
        TWITCH_GQL_URL,
        json.dumps({"query": query}).encode(),
        {"Client-ID": TWITCH_CLIENT_ID, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["data"]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return {}


def fetch_twitch_clips(vod_id, opener=None, timeout=FETCH_TIMEOUT):
    """Viewer-made Twitch clips of VOD ``vod_id``.

    GQL cannot list clips by VOD, so this reads the channel's top-100
    clips per period (day / week / month / all time) and keeps the
    ones whose video is this VOD. Top 100 only: a huge channel loses
    the small clips. Returns [{offset, duration, views, title, slug}]
    sorted by offset; [] on any failure.
    """
    try:
        owner = _twitch_gql(
            '{video(id:"%s"){owner{login}}}' % vod_id,
            opener=opener, timeout=timeout)
        login = (owner.get("video") or {}).get("owner", {}).get("login")
        if not login:
            return []
    except Exception:
        return []
    found = {}
    for period in CLIP_PERIODS:
        try:
            data = _twitch_gql(
                '{user(login:"%s"){clips(first:100,criteria:{period:%s,'
                'sort:VIEWS_DESC}){edges{node{slug title viewCount '
                'videoOffsetSeconds durationSeconds video{id}}}}}}'
                % (login, period),
                opener=opener, timeout=timeout)
            edges = ((data.get("user") or {}).get("clips") or {}).get(
                "edges") or []
        except Exception:
            continue  # one period failing must not lose the others
        for edge in edges:
            node = edge.get("node") or {}
            video = node.get("video") or {}
            offset = node.get("videoOffsetSeconds")
            if video.get("id") == str(vod_id) and offset is not None:
                found[node.get("slug")] = {
                    "offset": float(offset),
                    "duration": float(node.get("durationSeconds") or 0),
                    "views": int(node.get("viewCount") or 0),
                    "title": node.get("title") or "",
                    "slug": node.get("slug") or "",
                }
    return sorted(found.values(), key=lambda c: c["offset"])


def clipped_signal(clips, start, end):
    """[CLIPPED] signal for window [start, end].

    Returns {"count", "views", "tag"} -- e.g. tag
    ``[CLIPPED x2 5.6k views]`` -- or {"count": 0, "views": 0,
    "tag": ""} when no viewer clip covers the window.
    """
    hits = [c for c in clips or []
            if c["offset"] - CLIP_LEAD_SECONDS < end
            and c["offset"] + c["duration"] > start]
    if not hits:
        return {"count": 0, "views": 0, "tag": ""}
    views = sum(c["views"] for c in hits)
    if views >= 1000:
        tag = f" [CLIPPED x{len(hits)} {views / 1000:.1f}k views]"
    else:
        tag = f" [CLIPPED x{len(hits)} {views} views]"
    return {"count": len(hits), "views": views, "tag": tag}


# ---------------------------------------------------------------------------
# YouTube most-replayed ([REPLAYED])
# ---------------------------------------------------------------------------

def fetch_youtube_heatmap(url, metadata_fn=None):
    """YouTube most-replayed heatmap for a video URL.

    Reads yt-dlp metadata (no download). ``metadata_fn`` is an
    injectable ``url -> info dict`` for tests. Returns the heatmap
    list [{start_time, end_time, value}] or [] when yt-dlp is
    missing, the fetch fails, or the video has no heatmap.
    """
    try:
        if metadata_fn is not None:
            info = metadata_fn(url)
        else:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                   "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False)
    except Exception:
        return []
    heatmap = (info or {}).get("heatmap") or []
    return [h for h in heatmap
            if isinstance(h, dict) and "value" in h
            and "start_time" in h and "end_time" in h]


def replayed_signal(heatmap, start, end):
    """[REPLAYED] signal for window [start, end].

    Returns {"ratio", "tag"} -- e.g. tag ``[REPLAYED x8]`` where 8 is
    the peak heat inside the window divided by the video's median
    heat -- or {"ratio": 0.0, "tag": ""} when nothing peaks here.
    """
    if not heatmap:
        return {"ratio": 0.0, "tag": ""}
    values = sorted(float(h["value"]) for h in heatmap)
    median = max(values[len(values) // 2], 0.05)
    video_end = float(heatmap[-1]["end_time"])
    ignore_before = REPLAY_IGNORE_START_FRACTION * video_end
    hits = [float(h["value"]) for h in heatmap
            if h["start_time"] < end and h["end_time"] > start
            and h["value"] >= max(REPLAY_MIN_VALUE,
                                  REPLAY_MIN_RATIO * median)
            and h["start_time"] > ignore_before]
    if not hits:
        return {"ratio": 0.0, "tag": ""}
    ratio = max(hits) / median
    return {"ratio": round(ratio, 1), "tag": f" [REPLAYED x{ratio:.0f}]"}


# ---------------------------------------------------------------------------
# Combined scoring (used by vod_clipper.find_best_moment)
# ---------------------------------------------------------------------------

def window_signal_bonus(clips, heatmap, start, end):
    """Bonus points (0-35) for one VOD window from viewer signals.

    Clipped: log-scaled on views (100 views ~0.35, 5k views ~0.75 of
    the 20-point budget) plus a little per extra clip. Replayed:
    the peak ratio scaled to the 15-point budget (8x = full).
    Returns {"bonus", "clipped", "replayed"} with the two signals'
    tag dicts attached for logging.
    """
    clipped = clipped_signal(clips, start, end)
    replayed = replayed_signal(heatmap, start, end)
    bonus = 0.0
    if clipped["count"]:
        views = max(clipped["views"], 1)
        strength = min(1.0, 0.35 + 0.15 * math.log10(views / 100)
                       + 0.05 * (clipped["count"] - 1))
        bonus += max(0.0, strength) * CLIPPED_MAX_BONUS
    if replayed["ratio"]:
        bonus += min(1.0, replayed["ratio"] / 8.0) * REPLAYED_MAX_BONUS
    return {
        "bonus": round(min(MAX_SIGNAL_BONUS, bonus), 1),
        "clipped": clipped,
        "replayed": replayed,
    }


def fetch_signals_for_source(source, opener=None, metadata_fn=None):
    """Fetch the viewer signals for a VOD source URL.

    Twitch video URLs -> Twitch clips; YouTube URLs -> most-replayed
    heatmap. Anything else (Kick, local files) -> empty signals: Kick
    has no public clips feed like this. Never raises; a failed fetch
    just means empty signals and audio-only scoring.
    Returns {"clips": [...], "heatmap": [...]}.
    """
    source = str(source or "")
    signals = {"clips": [], "heatmap": []}
    try:
        twitch_match = re.search(r"twitch\.tv/videos/(\d+)", source)
        if twitch_match:
            signals["clips"] = fetch_twitch_clips(
                twitch_match.group(1), opener=opener)
            return signals
        if "youtube.com" in source.lower() or "youtu.be" in source.lower():
            signals["heatmap"] = fetch_youtube_heatmap(
                source, metadata_fn=metadata_fn)
    except Exception:
        pass
    return signals
