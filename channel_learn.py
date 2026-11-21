"""
Learn from your own channel (gta6-clipper feature 3).

Reads the channel's own YouTube Shorts and their public view counts
with yt-dlp (one flat-playlist request, cached a day, NO API key),
matches them to clips this app made, and writes rules about what
performs: which streamer, length and post time get well above (or
below) the median views, and whether name-first hooks, numbers or
questions help. Needs 8+ Shorts before it says anything — fewer than
that is noise, not a pattern.

HONEST LIMIT — YouTube only. TikTok has no public view-count feed
like this, so TikTok performance stays manual (typed in on the
dashboard) until a TikTok API connection exists. Nothing in this
module pretends to know TikTok numbers.

Integration points:
    - analytics.record_performance: sync_channel_performance()
      auto-fills YouTube view counts for matched clips, so the
      analytics summary stops depending on hand-typed numbers.
    - analytics.analytics_summary(): exposes the learned rules.
    - pipeline: score_tiebreak() nudges the AI score by at most
      +/-3 when moments tie — a tie-break, not a new score.

Settings are module-local constants with env-var overrides (no
config.py changes — other agents own that file):

    YOUTUBE_CHANNEL_URL   Channel to learn from, e.g.
                          https://www.youtube.com/@handle (empty = not
                          configured; sync returns "not_configured")
    LEARN_MIN_SHORTS      Shorts needed before rules are written (8)
    LEARN_CACHE_TTL       Cache lifetime in seconds (86400 = 1 day)
    LEARN_LIMIT           Max Shorts fetched per sync (60)
"""

import difflib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import CLIPS_DIR

CHANNEL_URL = os.environ.get("YOUTUBE_CHANNEL_URL", "").strip()
MIN_SHORTS = int(os.environ.get("LEARN_MIN_SHORTS", "8"))
CACHE_TTL_SECONDS = int(os.environ.get("LEARN_CACHE_TTL", "86400"))
FETCH_LIMIT = int(os.environ.get("LEARN_LIMIT", "60"))

# A Short only counts as "one of ours" when its title matches a clip
# we made this closely (difflib ratio).
MATCH_THRESHOLD = 0.75
# A category only becomes a rule when its median is this far from the
# channel median (1.5x up or 0.67x down) with at least 3 samples.
RULE_RATIO_HIGH = 1.5
RULE_RATIO_LOW = 0.67
RULE_MIN_SAMPLES = 3
# Hook features (name-first / number / question) need a smaller gap
# because they split the whole set in two rather than slicing a group.
HOOK_RATIO_HIGH = 1.4
HOOK_RATIO_LOW = 0.7

STOP_WORDS = set(
    "the a an this that when how why what who i he she they we you it "
    "my his her their top best new first last just so and but if watch "
    "can did does is are was".split()
)

RULES_FILENAME = "learned_rules.json"


def _median(values):
    return statistics.median(values) if values else 0


def _length_bucket(seconds):
    if not seconds:
        return None
    if seconds < 20:
        return "under 20 s"
    if seconds <= 30:
        return "20-30 s"
    return "over 30 s"


def _slug(text):
    return re.sub(r"\W", "_", text)


def cache_path(channel_url):
    """Where one channel's fetched Shorts list is cached (24 h)."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    return CLIPS_DIR / f"learn_cache_{_slug(channel_url)}.json"


def rules_path():
    """Where the learned rules JSON lives."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    return CLIPS_DIR / RULES_FILENAME


def fetch_shorts(channel_url, limit=FETCH_LIMIT, refresh=False,
                 runner=None):
    """Fetch a channel's Shorts + public view counts via yt-dlp.

    One flat-playlist request, cached for CACHE_TTL_SECONDS so a
    re-run costs nothing. ``runner`` is the subprocess function to
    call (injected by tests — no network in tests). Returns the
    parsed yt-dlp JSON (``{"entries": [...], "channel": ...}``).
    Raises RuntimeError when yt-dlp fails or times out.
    """
    run = runner or subprocess.run
    cache = cache_path(channel_url)
    if (cache.exists() and not refresh
            and time.time() - cache.stat().st_mtime < CACHE_TTL_SECONDS):
        return json.loads(cache.read_text(encoding="utf-8"))

    url = channel_url if channel_url.startswith("http") else \
        "https://www.youtube.com/" + channel_url.lstrip("/")
    if not url.rstrip("/").endswith("/shorts"):
        url = url.rstrip("/") + "/shorts"
    try:
        res = run([sys.executable, "-m", "yt_dlp", "-J",
                   "--flat-playlist", "--playlist-end", str(limit),
                   "--skip-download", url],
                  capture_output=True, text=True,
                  encoding="utf-8", timeout=300)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "YouTube did not answer in 5 minutes: check the internet "
            "and try again")
    if res.returncode or not res.stdout:
        raise RuntimeError("yt-dlp failed: " + (res.stderr or "")[-300:])
    try:
        cache.write_text(res.stdout, encoding="utf-8")
    except OSError:
        pass  # a failed cache write must not fail the sync
    return json.loads(res.stdout)


def local_clip_index():
    """Clips this app made, for matching against the channel's Shorts.

    Each entry carries both titles the clip is known by — the YouTube
    title is what actually appears on the Short, the clip title is
    the fallback — plus the streamer and the clip length. An empty
    list when the clips table cannot be read.
    """
    try:
        from database import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, title, youtube_title, streamer, duration "
            "FROM clips").fetchall()
        conn.close()
    except Exception:
        return []
    index = []
    for row in rows:
        row = dict(row)
        titles = [t for t in (row.get("youtube_title"), row.get("title"))
                  if t]
        if not titles:
            continue
        index.append({
            "clip_id": row["id"],
            "titles": titles,
            "source": row.get("streamer"),
            "length": row.get("duration"),
        })
    return index


def _best_match(title, index):
    """The local clip whose title best matches a Short's title, or None."""
    best, best_ratio = None, 0.0
    lowered = title.lower()
    for entry in index:
        for candidate in entry["titles"]:
            ratio = difflib.SequenceMatcher(
                None, candidate.lower(), lowered).ratio()
            if ratio > best_ratio:
                best, best_ratio = entry, ratio
    return best if best_ratio >= MATCH_THRESHOLD else None


def build_rows(entries, index):
    """Turn raw yt-dlp entries into comparable per-Short rows.

    Entries without a view count are skipped (private / removed
    Shorts). Each row records the views, the matched local clip (if
    any), the streamer, the length bucket, the UTC 6-hour post-time
    block, and the hook features (name first, number, question).
    """
    rows = []
    for entry in entries:
        if entry.get("view_count") is None:
            continue
        title = entry.get("title") or ""
        match = _best_match(title, index) or {}
        words = re.findall(r"[A-Za-z0-9']+", title)
        first = words[0] if words else ""
        timestamp = entry.get("timestamp")
        duration = match.get("length") or entry.get("duration")
        rows.append({
            "title": title,
            "views": entry["view_count"],
            "clip_id": match.get("clip_id"),
            "matched": bool(match),
            "source": match.get("source") or (
                first if first.lower() not in STOP_WORDS else None),
            "length": _length_bucket(duration),
            "hour": (datetime.fromtimestamp(timestamp, timezone.utc).hour
                     // 6 * 6) if timestamp else None,
            "name_first": bool(first[:1].isupper()
                               and first.lower() not in STOP_WORDS),
            "number": bool(re.search(r"\d", title)),
            "question": "?" in title,
        })
    return rows


def learn_rules(rows):
    """Rules for the next picks, from per-Short rows.

    A rule is a dict: kind (source/length/hour/name_first/number/
    question), value, ratio vs the median (or vs the other half for
    hook features), samples, and a human-readable sentence. Returns
    [] when there are fewer than MIN_SHORTS rows — too few to learn
    anything. Sorted strongest effect first.
    """
    if len(rows) < MIN_SHORTS:
        return []
    base = _median([r["views"] for r in rows])
    if not base:
        return []
    found = []
    categories = [
        ("source", "Sources"), ("length", "Length"),
        ("hour", "Post time (UTC 6 h block)"),
    ]
    for key, label in categories:
        for value in {r[key] for r in rows if r[key] is not None}:
            group = [r["views"] for r in rows if r[key] == value]
            if len(group) < RULE_MIN_SAMPLES:
                continue
            ratio = _median(group) / base
            if ratio >= RULE_RATIO_HIGH or ratio <= RULE_RATIO_LOW:
                shown = (f"{value:02d}-{value + 6:02d}h" if key == "hour"
                         else str(value))
                found.append((abs(ratio - 1), {
                    "kind": key, "value": value, "ratio": round(ratio, 2),
                    "samples": len(group),
                    "text": (f"{label}: {shown} gets {ratio:.1f}x the "
                             f"median views ({len(group)} Shorts)."),
                }))
    for key, label in [("name_first", "hooks with a name first"),
                       ("number", "titles with a number"),
                       ("question", "titles that ask a question")]:
        with_it = [r["views"] for r in rows if r[key]]
        without = [r["views"] for r in rows if not r[key]]
        if len(with_it) < RULE_MIN_SAMPLES or len(without) < RULE_MIN_SAMPLES:
            continue
        ratio = _median(with_it) / max(_median(without), 1)
        if ratio >= HOOK_RATIO_HIGH or ratio <= HOOK_RATIO_LOW:
            found.append((abs(ratio - 1), {
                "kind": key, "value": True, "ratio": round(ratio, 2),
                "samples": len(with_it),
                "text": (f"{label[0].upper() + label[1:]}: {ratio:.1f}x "
                         f"the views of those without "
                         f"({len(with_it)} vs {len(without)})."),
            }))
    return [rule for _, rule in sorted(found, reverse=True)]


def summarize_learning(rows, channel_name=""):
    """The learning summary: counts, median, rules, too-few flag."""
    rules = learn_rules(rows)
    return {
        "channel": channel_name,
        "shorts": len(rows),
        "matched": sum(1 for r in rows if r["matched"]),
        "median_views": _median([r["views"] for r in rows]),
        "too_few": len(rows) < MIN_SHORTS,
        "min_shorts": MIN_SHORTS,
        "rules": rules,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def save_learned_rules(summary, path=None):
    target = Path(path) if path else rules_path()
    target.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                      encoding="utf-8")
    return target


def load_learned_rules(path=None):
    """The last saved learning summary, or None when there is none
    (never synced, or the file is broken)."""
    target = Path(path) if path else rules_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _latest_youtube_views(clip_id):
    """Latest recorded YouTube views for a clip, or None."""
    from database import get_connection
    conn = get_connection()
    row = conn.execute(
        "SELECT views FROM clip_performance "
        "WHERE clip_id = ? AND platform = 'youtube' "
        "ORDER BY id DESC LIMIT 1", (clip_id,)).fetchone()
    conn.close()
    return row["views"] if row else None


def sync_channel_performance(channel_url=None, limit=FETCH_LIMIT,
                             refresh=False, runner=None):
    """Fetch the channel's Shorts, learn rules, auto-fill YouTube views.

    Matches each Short to a clip this app made and records the public
    view count via analytics.record_performance — but only when the
    count changed since the last snapshot, so a daily sync does not
    pile up identical rows. The learned rules are saved for
    analytics_summary() and score_tiebreak().

    Returns a status dict; "not_configured" when no channel URL is
    set, "error" when the fetch fails. Never raises.
    """
    url = (channel_url or CHANNEL_URL or "").strip()
    if not url:
        return {"status": "not_configured",
                "reason": "Set YOUTUBE_CHANNEL_URL to the channel to "
                          "learn from (e.g. https://www.youtube.com/@handle)."}
    try:
        from analytics import record_performance
        data = fetch_shorts(url, limit=limit, refresh=refresh,
                            runner=runner)
    except Exception as e:
        return {"status": "error", "reason": str(e)}
    entries = data.get("entries") or []
    dated = any(e.get("timestamp") for e in entries)
    if not dated:
        # No upload dates: views are raw, and the 3 newest Shorts are
        # still gaining — skip them so they do not skew the median.
        entries = entries[3:]
    rows = build_rows(entries, local_clip_index())
    summary = summarize_learning(rows, data.get("channel") or url)
    try:
        save_learned_rules(summary)
    except OSError:
        pass
    recorded = 0
    for row in rows:
        if not row["matched"] or row["clip_id"] is None:
            continue
        try:
            if _latest_youtube_views(row["clip_id"]) == row["views"]:
                continue
            record_performance(row["clip_id"], "youtube", row["views"])
            recorded += 1
        except Exception:
            continue  # one bad row must not stop the sync
    summary.update({"status": "ok", "recorded": recorded})
    return summary


def score_tiebreak(streamer, duration_seconds, title, rules=None):
    """A small score nudge (+/-3 max) from the learned rules.

    Used when the pipeline scores moments: when two moments are
    otherwise tied, the one the channel's own history favours wins.
    Returns 0 when there are no rules or nothing applies — so clips
    are never punished for a channel with no learning yet.
    """
    if rules is None:
        saved = load_learned_rules()
        rules = saved.get("rules", []) if saved else []
    if not rules:
        return 0
    delta = 0
    bucket = _length_bucket(duration_seconds)
    words = re.findall(r"[A-Za-z0-9']+", title or "")
    first = words[0] if words else ""
    traits = {
        "name_first": bool(first[:1].isupper()
                           and first.lower() not in STOP_WORDS),
        "number": bool(re.search(r"\d", title or "")),
        "question": "?" in (title or ""),
    }
    for rule in rules:
        kind, ratio = rule.get("kind"), rule.get("ratio") or 1
        if ratio == 1:
            continue
        favoured = ratio > 1  # this trait beats the median
        if kind == "source":
            if (streamer or "").lower() != str(rule.get("value", "")).lower():
                continue
            delta += 2 if favoured else -2
        elif kind == "length":
            if bucket != rule.get("value"):
                continue
            delta += 1 if favoured else -1
        elif kind in traits:
            # The trait helps: having it is +1, missing it is -1 (and
            # vice versa when the trait hurts).
            has_trait = traits[kind]
            if has_trait == favoured:
                delta += 1
            else:
                delta -= 1
        # "hour" rules cannot apply here: post time is not known when
        # a moment is scored, only when it is posted.
    return max(-3, min(3, delta))
