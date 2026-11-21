"""YouTube performance analytics (gta6-clipper feature 20).

How is a Short doing on YouTube, and why not better — deeper than the
view counts in analytics.py, because retention shows *where* viewers
leave, which is the real lesson for the next clip.

Level 1: a YouTube Data API key gives views/likes/comments per Short.
Level 2 (optional): the user's own OAuth client JSON gives Analytics
data — retention curve, average view percentage — via a loopback
sign-in flow. The OAuth client JSON is checked before use: its auth
and token URLs must point at Google, so a crafted client file cannot
send the code or secret elsewhere.

Needs Sara's own YouTube API key (and her own OAuth client JSON for
retention). Nothing here runs live without them. Read-only calls,
stdlib only; ``_get`` is a test hook that replaces the HTTP GET.
"""

import json
import re
import secrets
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DATA_API = "https://www.googleapis.com/youtube/v3"
ANALYTICS_API = "https://youtubeanalytics.googleapis.com/v2/reports"
SCOPES = ("https://www.googleapis.com/auth/yt-analytics.readonly "
          "https://www.googleapis.com/auth/youtube.readonly")

# Verbs a title should start with (the diagnosis flags titles that do not).
VERBS = set("watch see look stop wait why how make build get find learn try "
            "use spot hear meet beat break catch clip drop expose reveal show "
            "survive steal crash escape ruin save take unlock leak hit run "
            "shoot play ask say tell think guess".split())

_get = None  # test hook: replaces the HTTP GET (url, headers) -> dict


def http(url, headers=None, data=None):
    """GET (or POST with ``data``) returning parsed JSON. Raises
    RuntimeError with YouTube's own message on an HTTP error."""
    if _get is not None and data is None:
        return _get(url, headers or {})
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        message = e.read().decode("utf-8", "ignore")[:300]
        try:
            message = json.loads(message)["error"]["message"]
        except Exception:
            pass
        raise RuntimeError(f"YouTube said {e.code}: {message}")


def vid_id(text):
    """The 11-character video id from a YouTube URL or a bare id."""
    match = (re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([\w-]{11})", text or "")
             or re.fullmatch(r"([\w-]{11})", (text or "").strip()))
    return match.group(1) if match else None


def parse_duration(iso):
    """An ISO 8601 duration (PT1M5S) -> seconds."""
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    hours, minutes, seconds = (int(x or 0) for x in match.groups()) if match else (0, 0, 0)
    return hours * 3600 + minutes * 60 + seconds


def data_api(path, key, **params):
    return http(f"{DATA_API}/{path}?" + urllib.parse.urlencode(dict(params, key=key)))


def _normalise(item):
    snippet, stats = item["snippet"], item.get("statistics", {})
    return {
        "id": item["id"],
        "title": snippet["title"],
        "published": snippet["publishedAt"],
        "seconds": parse_duration(item.get("contentDetails", {}).get("duration")),
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
    }


def videos(key, ids):
    """Stats for up to any number of video ids (batched 50 at a time)."""
    out = []
    for i in range(0, len(ids), 50):
        batch = data_api("videos", key, part="snippet,statistics,contentDetails",
                         id=",".join(ids[i:i + 50])).get("items", [])
        out += [_normalise(item) for item in batch]
    return out


def channel_id(key, channel):
    """A UC... channel id from an id or an @handle."""
    if re.fullmatch(r"UC[\w-]{22}", channel or ""):
        return channel
    items = data_api("channels", key, part="id",
                     forHandle=(channel or "").lstrip("@")).get("items")
    if not items:
        raise RuntimeError("Channel not found: use the UC... id or the @handle")
    return items[0]["id"]


def latest_shorts(key, channel, count=20):
    """The channel's latest uploads that are Shorts (<= 3 minutes)."""
    uploads_playlist = "UU" + channel_id(key, channel)[2:]
    items = data_api("playlistItems", key, part="contentDetails",
                     playlistId=uploads_playlist, maxResults=50).get("items", [])
    ids = [item["contentDetails"]["videoId"] for item in items]
    return [v for v in videos(key, ids) if 0 < v["seconds"] <= 180][:count]


# ---------------------------------------------------------------------------
# OAuth (the user's own client, loopback flow)
# ---------------------------------------------------------------------------

def token_path(workspace):
    return Path(workspace) / "yt" / "token.json"


def client_path(workspace):
    return Path(workspace) / "yt" / "client.json"


def check_client(client_json):
    """The uploaded client JSON must point at Google only: the code
    and client_secret are posted to token_uri, so a crafted file
    pointing elsewhere would leak them. Returns the client block."""
    client = client_json.get("installed") or client_json.get("web") or client_json
    for key, host in (("auth_uri", "accounts.google.com"),
                      ("token_uri", "oauth2.googleapis.com")):
        url = urllib.parse.urlparse(str(client.get(key) or ""))
        if (url.scheme != "https" or url.hostname != host
                or url.port not in (None, 443) or url.username):
            raise ValueError(f"OAuth client JSON rejected: {key} must be https://{host}/...")
    return client


def oauth_start(workspace, client_json):
    """Start the loopback sign-in: returns the Google auth URL to open.
    The code comes back to a one-shot local server, is exchanged for a
    token, and the token is saved under <workspace>/yt/token.json.
    The server shuts itself down after 5 minutes."""
    client = check_client(client_json)
    box = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if query.get("state", [""])[0] == box["state"] and query.get("code"):
                try:
                    token = http(client["token_uri"], data={
                        "code": query["code"][0],
                        "client_id": client["client_id"],
                        "client_secret": client["client_secret"],
                        "redirect_uri": box["redirect_uri"],
                        "grant_type": "authorization_code"})
                    token["at"] = time.time()
                    token_path(workspace).parent.mkdir(parents=True, exist_ok=True)
                    token_path(workspace).write_text(json.dumps(token))
                    message = "Signed in. You can close this tab."
                except Exception as e:
                    message = "Sign-in failed: " + str(e)
            else:
                message = "Wrong or missing code."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(message.encode())
            threading.Thread(target=server.shutdown, daemon=True).start()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    box["state"] = secrets.token_urlsafe(16)
    box["redirect_uri"] = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    timer = threading.Timer(300, server.shutdown)
    timer.daemon = True
    timer.start()
    return client["auth_uri"] + "?" + urllib.parse.urlencode({
        "client_id": client["client_id"],
        "redirect_uri": box["redirect_uri"],
        "response_type": "code",
        "scope": SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "state": box["state"],
    })


def access_token(workspace):
    """A valid Analytics access token (refreshing when expired), or
    None when not signed in."""
    token_file, client_file = token_path(workspace), client_path(workspace)
    if not token_file.exists() or not client_file.exists():
        return None
    token = json.loads(token_file.read_text())
    client = check_client(json.loads(client_file.read_text()))
    if time.time() - token.get("at", 0) > token.get("expires_in", 0) - 60:
        if not token.get("refresh_token"):
            return None
        token.update(http(client["token_uri"], data={
            "client_id": client["client_id"],
            "client_secret": client["client_secret"],
            "refresh_token": token["refresh_token"],
            "grant_type": "refresh_token"}), at=time.time())
        token_file.write_text(json.dumps(token))
    return token["access_token"]


def analytics(token, start, end, metrics, dimensions=None, filters=None,
              sort=None, max_results=None):
    params = {"ids": "channel==MINE", "startDate": start, "endDate": end,
              "metrics": metrics}
    for key, value in (("dimensions", dimensions), ("filters", filters),
                       ("sort", sort), ("maxResults", max_results)):
        if value:
            params[key] = value
    result = http(ANALYTICS_API + "?" + urllib.parse.urlencode(params),
                  {"Authorization": "Bearer " + token})
    columns = [h["name"] for h in result.get("columnHeaders", [])]
    return [dict(zip(columns, row)) for row in result.get("rows", [])]


def deep(workspace, video):
    """Analytics for one video + the channel median of the top 20
    videos (90 days): {} when not signed in or no data yet."""
    try:
        token = access_token(workspace)
    except Exception as e:
        return {"error": "YouTube sign-in expired: sign in again "
                         "(Settings > YouTube) (" + str(e)[:120] + ")"}
    if not token:
        return {}
    start = video["published"][:10]
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = {}
    metrics = "views,averageViewDuration,averageViewPercentage,subscribersGained"
    try:
        rows = analytics(token, start, end, metrics, filters="video==" + video["id"])
        if rows:
            out.update(rows[0])
        retention = analytics(token, start, end, "audienceWatchRatio",
                              dimensions="elapsedVideoTimeRatio",
                              filters="video==" + video["id"])
        if retention and video["seconds"]:
            target = 3 / video["seconds"]
            out["kept_at_3s"] = min(
                retention,
                key=lambda x: abs(x["elapsedVideoTimeRatio"] - target))["audienceWatchRatio"]
        ninety_days_ago = datetime.fromordinal(
            datetime.now(timezone.utc).date().toordinal() - 90).strftime("%Y-%m-%d")
        top = analytics(token, ninety_days_ago, end, metrics,
                        dimensions="video", sort="-views", max_results=20)
        if top:
            out["median"] = {k: statistics.median(x[k] for x in top)
                             for k in ("averageViewDuration",
                                       "averageViewPercentage",
                                       "subscribersGained")}
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


# ---------------------------------------------------------------------------
# Diagnosis (rules, no AI)
# ---------------------------------------------------------------------------

def age_hours(video):
    published = datetime.fromisoformat(video["published"].replace("Z", "+00:00"))
    return max(1, (datetime.now(timezone.utc) - published).total_seconds() / 3600)


def diagnose(video, others=(), deep_stats=None, hook=""):
    """Why a Short underperformed: a list of {key, level, text} reasons.
    ``others`` are peer Shorts for the median comparison; ``deep_stats``
    is the Level 2 analytics dict from deep()."""
    deep_stats = deep_stats or {}
    reasons = []
    peers = [o for o in others if o["id"] != video["id"]]
    median_views = statistics.median(o["views"] for o in peers) if peers else None
    age = age_hours(video)
    if median_views is not None and age >= 24 and video["views"] < 0.5 * median_views:
        reasons.append({"key": "views", "level": "bad",
                        "text": f"{video['views']} views after {age:.0f} h; your last "
                                f"{len(peers)} Shorts have a median of {median_views:.0f}. "
                                "Low views this early usually means the first second "
                                "did not earn the swipe-stay."})
    elif age < 24:
        reasons.append({"key": "views", "level": "info",
                        "text": f"Only {age:.0f} h old ({video['views']} views): "
                                "judge it at 24 h and 48 h."})
    if video["views"] >= 50 and video["likes"] / video["views"] < 0.02:
        reasons.append({"key": "likes", "level": "warn",
                        "text": f"Like rate {100 * video['likes'] / video['views']:.1f}% "
                                "(typical Shorts 3-6%): viewers do not feel enough to "
                                "react; end on the strongest beat."})
    avg_pct = deep_stats.get("averageViewPercentage")
    median_pct = (deep_stats.get("median") or {}).get("averageViewPercentage")
    if avg_pct is not None and (avg_pct < 60 or (median_pct and avg_pct < 0.8 * median_pct)):
        reasons.append({"key": "avp", "level": "bad",
                        "text": f"Average viewed {avg_pct:.0f}%"
                                + (f" vs your median {median_pct:.0f}%" if median_pct else "")
                                + ": people leave before the end. Cut the slow middle "
                                  "or shorten the clip."})
    kept_3s = deep_stats.get("kept_at_3s")
    if kept_3s is not None and kept_3s < 0.7:
        reasons.append({"key": "hook3", "level": "bad",
                        "text": f"Only {100 * kept_3s:.0f}% are still watching at "
                                "second 3: the hook is not working. Start on the peak "
                                "line or action."})
    title = video["title"]
    words = re.findall(r"[\w']+", title)
    if len(title) > 60:
        reasons.append({"key": "title_len", "level": "warn",
                        "text": f"Title is {len(title)} characters (over about 60): "
                                "the end is cut in the feed."})
    if words and words[0].lower() not in VERBS:
        reasons.append({"key": "verb", "level": "warn",
                        "text": "Title does not start with a verb: try an action "
                                "opener (Watch, Spot, Stop...)."})
    if not any(w[0].isupper() or w.isdigit() for w in words[1:]):
        reasons.append({"key": "name", "level": "warn",
                        "text": "No name or number in the title (GTA 6, Rockstar, the "
                                "streamer): searchers have nothing to match."})
    hook_words = {w.lower() for w in re.findall(r"\w{4,}", hook)}
    title_words = {w.lower() for w in re.findall(r"\w{4,}", title)}
    if hook_words and title_words and len(hook_words & title_words) / len(hook_words | title_words) < 0.1:
        reasons.append({"key": "mismatch", "level": "warn",
                        "text": "Title and on-screen hook say different things: pick "
                                "one promise and use it in both."})
    if len(peers) >= 6:
        hour = lambda o: datetime.fromisoformat(o["published"].replace("Z", "+00:00")).hour
        best_hours = {hour(o) for o in sorted(peers, key=lambda o: -o["views"])[:len(peers) // 3]}
        if hour(video) not in best_hours:
            reasons.append({"key": "hour", "level": "info",
                            "text": f"Posted at {hour(video):02d}:00 UTC; your best Shorts "
                                    f"went out around {', '.join(f'{h:02d}:00' for h in sorted(best_hours))} UTC."})
    return {
        "reasons": reasons,
        "median_views": median_views,
        "age_hours": round(age, 1),
        "level": "analytics" if deep_stats.get("averageViewPercentage") is not None else "key",
    }


def ai_prompt(video, reasons, transcript="", hook=""):
    """Prompt for any AI provider; the answer is JSON
    {titles[5], hooks[3], recut} — fresh packaging for an
    underperforming Short."""
    problems = "\n".join("- " + r["text"] for r in reasons)
    return ("A YouTube Short about GTA 6 is underperforming. Give fresh packaging.\n"
            f"Current title: {video['title']}\nOn-screen hook: {hook or '(none)'}\n"
            f"Stats: {video['views']} views, {video['likes']} likes, "
            f"{video['comments']} comments, {video['seconds']} s, posted {video['published']}\n"
            "Problems found:\n" + problems + f"\nTranscript of the Short:\n{transcript[:3000]}\n\n"
            'Return ONLY JSON: {"titles":[5 titles, each starts with a verb, under 60 chars, '
            'contains a name or number],"hooks":[3 on-screen hooks, max 7 words, same promise '
            'the clip delivers],"recut":"one idea to re-cut the clip (where to start, what to remove)"}')
