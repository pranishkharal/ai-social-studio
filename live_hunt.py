"""Multi-platform live hunt (gta6-clipper feature 14).

One panel for Kick, Twitch and YouTube: who of Sara's streamers is
live right now, with viewers, title and a start/stop button per
streamer, plus each streamer's last 5 past streams to clip from.
Public endpoints only, no keys, read-only.

Honest limit: Kick's past-videos lookup is sometimes
Cloudflare-blocked — when it fails, the caller gets a row with an
"error" field instead of an exception, so the panel still renders.

Live checks are cached for 60 s and past-stream lookups for 10 min,
so refreshing the panel does not hammer the platforms.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

USER_AGENT = {"User-Agent": "Mozilla/5.0 (ai-social-studio live hunt)"}

LIVE_CACHE_SECONDS = 60
PAST_CACHE_SECONDS = 600
PAST_STREAMS_COUNT = 5

_CACHE = {}   # (platform, handle) -> (time, info)
_PCACHE = {}  # (platform, handle) -> (time, rows)
_PROCS = {}   # "platform:handle" -> Popen (streamlink recorder)
_LOCK = threading.Lock()
_WORKSPACE = None  # last workspace seen, so pid files survive a restart


def _get(url, timeout=10):
    req = urllib.request.Request(url, headers=USER_AGENT)
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def _twitch_gql(query):
    """Twitch's public GraphQL endpoint (no login). The client id is
    the public web client id, the same one the Twitch website uses."""
    req = urllib.request.Request(
        "https://gql.twitch.tv/gql",
        json.dumps({"query": query}).encode(),
        {"Client-ID": "kimne78kx3ncx6brgo4mv6wki5h1ko",
         "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))["data"]


def look_twitch(handle):
    data = _twitch_gql(
        '{user(login:%s){stream{title viewersCount createdAt}}}' % json.dumps(handle))
    user = data.get("user") or {}
    stream = user.get("stream")
    return {
        "live": bool(stream),
        "title": (stream or {}).get("title", "") or "",
        "viewers": (stream or {}).get("viewersCount", 0) or 0,
        "started": (stream or {}).get("createdAt", "") or "",
    }


def look_kick(handle):
    data = json.loads(_get("https://kick.com/api/v2/channels/" + handle))
    live = data.get("livestream")
    return {
        "live": bool(live),
        "title": (live or {}).get("session_title", "") or "",
        "viewers": (live or {}).get("viewer_count", 0) or 0,
        "started": (live or {}).get("created_at", "") or "",
    }


def look_youtube(handle):
    handle = handle if handle.startswith("@") else "@" + handle
    page = _get(f"https://www.youtube.com/{handle}/live")
    is_live = '"isLive":true' in page
    match = re.search(r'<meta property="og:title" content="([^"]*)"', page)
    return {
        "live": is_live,
        "title": match.group(1) if match and is_live else "",
        "viewers": 0,  # the channel page does not give a viewer count
        "started": "",
    }


LOOK = {"twitch": look_twitch, "kick": look_kick, "youtube": look_youtube}


def lookup(platform, handle, now=None):
    """Live status for one streamer (cached 60 s). Failures return a
    not-live row with an "error" field instead of raising."""
    now = now or time.time()
    cached = _CACHE.get((platform, handle))
    if cached and now - cached[0] < LIVE_CACHE_SECONDS:
        return cached[1]
    try:
        result = LOOK[platform](handle)
    except Exception as e:
        result = {"live": False, "title": "", "viewers": 0, "started": "",
                  "error": str(e)[:80]}
    _CACHE[(platform, handle)] = (now, result)
    return result


def _key(platform, handle):
    return f"{platform}:{handle.lower()}"


def outdir(workspace, platform, handle):
    return Path(workspace) / "hunt" / f"{platform}_{handle.lstrip('@').lower()}"


def _alive(pid):
    if os.name == "nt":
        res = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True)
        return "python" in res.stdout.lower()
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_file(workspace, platform, handle):
    return outdir(workspace, platform, handle) / "hunt.pid"


def _read_pid(workspace, platform, handle):
    try:
        return int(_pid_file(workspace, platform, handle).read_text().strip())
    except (OSError, ValueError):
        return None


def watching(platform, handle, workspace=None):
    """True when a hunt recorder is running for this streamer — either
    started by this process or by an earlier run (pid file)."""
    proc = _PROCS.get(_key(platform, handle))
    if proc:
        return proc.poll() is None
    workspace = workspace or _WORKSPACE
    if not workspace:
        return False
    pid = _read_pid(workspace, platform, handle)
    return bool(pid and _alive(pid))


def hunt(workspace, streamers, lookup_fn=None, limit=25):
    """Live status for a list of {"platform", "handle", "name"?}
    streamers, live first, most viewers first. Each row also says
    whether a hunt recorder is running for that streamer.
    ``lookup_fn`` defaults to lookup() (resolved at call time, so
    tests can replace the module-level lookup)."""
    lookup_fn = lookup_fn or lookup
    global _WORKSPACE
    _WORKSPACE = workspace
    streamers = streamers[:limit]
    with ThreadPoolExecutor(8) as pool:
        infos = list(pool.map(
            lambda s: lookup_fn(s["platform"], s["handle"]), streamers))
    out = []
    for streamer, info in zip(streamers, infos):
        out.append({
            "handle": streamer["handle"],
            "name": streamer.get("name", streamer["handle"]),
            "platform": streamer["platform"],
            "live": bool(info.get("live")),
            "title": info.get("title", ""),
            "viewers": info.get("viewers", 0),
            "started": info.get("started", ""),
            "error": info.get("error", ""),
            "watching": watching(streamer["platform"], streamer["handle"],
                                 workspace),
        })
    return sorted(out, key=lambda r: (not r["live"], -r["viewers"]))


def stream_url(platform, handle):
    """The URL streamlink records for a streamer on each platform."""
    if platform == "twitch":
        return f"https://www.twitch.tv/{handle}"
    if platform == "kick":
        return f"https://kick.com/{handle}"
    if platform == "youtube":
        handle = handle if handle.startswith("@") else "@" + handle
        return f"https://www.youtube.com/{handle}/live"
    raise ValueError(f"Unknown platform: {platform}")


def start(workspace, platform, handle, python=sys.executable):
    """Start recording a streamer with the streamlink recorder
    (streamlink_recorder.py) in the background. Returns
    {"ok", "already", "out", "log"}. Raises ValueError on a bad
    platform or handle."""
    if platform not in LOOK or not re.fullmatch(r"@?[\w.\-]{1,60}", handle or ""):
        raise ValueError("Bad platform or handle")
    global _WORKSPACE
    _WORKSPACE = workspace
    if watching(platform, handle, workspace):
        return {"ok": True, "already": True}
    dest = outdir(workspace, platform, handle)
    dest.mkdir(parents=True, exist_ok=True)
    log = open(dest / "live.log", "ab")
    with _LOCK:
        _PROCS[_key(platform, handle)] = subprocess.Popen(
            [python, str(HERE / "streamlink_recorder.py"),
             stream_url(platform, handle), "--out", str(dest)],
            cwd=str(dest), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)
        _pid_file(workspace, platform, handle).write_text(
            str(_PROCS[_key(platform, handle)].pid))
    return {"ok": True, "out": str(dest), "log": str(dest / "live.log")}


def stop(platform, handle, workspace=None):
    """Stop a hunt recorder started by start() (or by an earlier run,
    via its pid file). Returns {"ok", "was_running"}."""
    proc = _PROCS.pop(_key(platform, handle), None)
    workspace = workspace or _WORKSPACE
    pid = None if proc else (_read_pid(workspace, platform, handle) if workspace else None)
    was_running = False
    if proc and proc.poll() is None:
        was_running = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True)
        else:
            proc.terminate()
    elif pid and _alive(pid):
        was_running = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
        else:
            os.kill(pid, 15)
    if workspace:
        try:
            _pid_file(workspace, platform, handle).unlink(missing_ok=True)
        except OSError:
            pass
    return {"ok": True, "was_running": was_running}


# ---------------------------------------------------------------------------
# Past streams (last 5 per streamer, for VOD clipping)
# ---------------------------------------------------------------------------

def past_twitch(handle):
    data = _twitch_gql(
        '{user(login:%s){videos(type:ARCHIVE,first:5){edges{node{id title createdAt lengthSeconds viewCount}}}}}' % json.dumps(handle))
    user = data.get("user") or {}
    edges = ((user.get("videos") or {}).get("edges") or [])
    return [{
        "title": node.get("title") or "",
        "url": "https://www.twitch.tv/videos/" + node.get("id", ""),
        "date": (node.get("createdAt") or "")[:10],
        "duration_s": node.get("lengthSeconds") or 0,
        "views": node.get("viewCount") or 0,
    } for node in (edge.get("node") or {} for edge in edges)]


def past_kick(handle):
    # Often Cloudflare-blocked: the caller turns that into an error row.
    data = json.loads(_get(f"https://kick.com/api/v2/channels/{handle}/videos"))
    videos = data if isinstance(data, list) else data.get("data", [])
    rows = []
    for video in videos[:PAST_STREAMS_COUNT]:
        uuid = (video.get("video") or {}).get("uuid") or video.get("uuid", "")
        rows.append({
            "title": video.get("session_title") or "",
            "url": f"https://kick.com/{handle}/videos/{uuid}",
            "date": (video.get("created_at") or video.get("start_time") or "")[:10],
            "duration_s": int((video.get("duration") or 0) / 1000),
            "views": video.get("views") or 0,
        })
    return rows


def past_youtube(handle):
    handle = handle if handle.startswith("@") else "@" + handle
    res = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--flat-playlist", "-j",
         "--playlist-end", str(PAST_STREAMS_COUNT), "--no-warnings",
         f"https://www.youtube.com/{handle}/streams"],
        capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    rows = [json.loads(line) for line in res.stdout.splitlines()
            if line.startswith("{")]
    if not rows:
        raise RuntimeError((res.stderr.strip().splitlines() or ["no streams found"])[-1][:100])
    out = []
    for row in rows[:PAST_STREAMS_COUNT]:
        uploaded = row.get("upload_date") or ""
        date = f"{uploaded[:4]}-{uploaded[4:6]}-{uploaded[6:8]}" if len(uploaded) == 8 else ""
        out.append({
            "title": row.get("title") or "",
            "url": row.get("webpage_url") or row.get("url") or "",
            "date": date,
            "duration_s": int(row.get("duration") or 0),
            "views": row.get("view_count") or 0,
        })
    return out


PAST = {"twitch": past_twitch, "kick": past_kick, "youtube": past_youtube}


def past(platform, handle, now=None, fns=None):
    """Last 5 past streams for a streamer (cached 10 min). On failure
    returns one row with an "error" field — Kick's lookup is often
    Cloudflare-blocked, and that must not break the panel."""
    if platform not in PAST or not re.fullmatch(r"@?[\w.\-]{1,60}", handle or ""):
        raise ValueError("Bad platform or handle")
    now = now or time.time()
    cached = _PCACHE.get((platform, handle))
    if cached and now - cached[0] < PAST_CACHE_SECONDS:
        return cached[1]
    try:
        rows = (fns or PAST)[platform](handle)
    except Exception as e:
        message = str(e)[:100]
        prefix = "Kick blocks this lookup: " if platform == "kick" else "Could not load: "
        rows = [{"title": "", "url": "", "date": "", "duration_s": 0,
                 "views": 0, "error": prefix + message}]
    _PCACHE[(platform, handle)] = (now, rows)
    return rows
