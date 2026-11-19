"""
Earnings tracker for the Kick clipping program (Feature 7).

The Kick program pays per 100K views at a per-creator CPM rate
(see kick_program.CREATOR_RATES). This module records every submitted video
link, tracks its views, and turns views into estimated earnings:

    estimated_earnings = views / 100_000 * cpm

View counts start as manual entry (paste the number from TikTok / YouTube /
Instagram) and can grow to API pulls: for YouTube links the tracker tries a
best-effort view-count pull through the YouTube Data API when the client is
configured — it never fails the call if that pull is unavailable.

No new env vars: CPM rates come from kick_program.py, YouTube pulls reuse
the YOUTUBE_* settings from youtube_uploader.py.
"""

from datetime import datetime, timezone

from database import get_connection, log_event
from kick_program import get_creator_info, resolve_multi_streamer_pov

VIEWS_PER_CPM_UNIT = 100_000


def init_submissions_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id INTEGER,
            streamer TEXT NOT NULL,
            platform TEXT NOT NULL,
            video_url TEXT NOT NULL,
            cpm REAL NOT NULL,
            views INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0,
            estimated_earnings REAL DEFAULT 0.0,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def _ensure_tables():
    init_submissions_table()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _earnings_for(views: int, cpm: float) -> float:
    return round((views or 0) / VIEWS_PER_CPM_UNIT * (cpm or 0), 2)


def _get_clip_row(clip_id):
    if clip_id is None:
        return None
    conn = get_connection()
    row = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def record_submission(clip_id: int, platform: str, video_url: str,
                      streamer: str = None, streamers: list = None) -> dict:
    """Records a submitted video link and returns the submission row.

    The CPM is looked up from kick_program: if ``streamers`` (a list of every
    streamer shown in the clip) is given, the multi-streamer POV rule applies
    and the LOWEST CPM wins; otherwise the single ``streamer`` (or the clip's
    own streamer) is used. Unknown creators default to the $40 whitelist rate.
    """
    _ensure_tables()
    platform = (platform or "").strip().lower()
    if platform not in ("tiktok", "youtube", "instagram"):
        raise ValueError(f"Unknown platform '{platform}': use tiktok, youtube or instagram")
    if not video_url or not str(video_url).strip():
        raise ValueError("video_url is required to record a submission")

    clip = _get_clip_row(clip_id)
    if streamers:
        info = resolve_multi_streamer_pov(streamers)
        # The POV rule submits under the lowest-CPM streamer shown.
        streamer = min(streamers, key=lambda s: get_creator_info(s)["cpm"]).lower()
    else:
        streamer = (streamer or (clip or {}).get("streamer") or "other").lower()
        info = get_creator_info(streamer)
    cpm = info["cpm"]

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO submissions (clip_id, streamer, platform, video_url, cpm,
                                 views, likes, estimated_earnings, submitted_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 0, 0, 0.0, ?, ?)
    """, (clip_id, streamer, platform, str(video_url).strip(), cpm, _now_iso(), _now_iso()))
    submission_id = cursor.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
    conn.close()
    log_event("INFO", "Earnings",
              f"Recorded {platform} submission #{submission_id} for {streamer} "
              f"(${cpm}/100K): {video_url}")
    return dict(row)


def _try_youtube_pull(video_url: str):
    """Best-effort (views, likes) pull for a YouTube URL, or None.

    Never raises: returns None when the URL is not YouTube, the client is not
    configured, or the API call fails.
    """
    try:
        from youtube_uploader import fetch_video_statistics
        stats = fetch_video_statistics(video_url)
        if stats:
            return stats["views"], stats["likes"]
    except Exception as e:
        log_event("WARNING", "Earnings", f"YouTube view pull failed: {e}")
    return None


def update_views(submission_id: int, views: int = None, likes: int = None) -> dict:
    """Updates the view/like counts of a submission and recomputes earnings.

    Pass ``views`` explicitly for manual entry. If ``views`` is None and the
    submission URL is a YouTube link, a best-effort YouTube Data API pull is
    attempted; if that pull fails the stored counts are left untouched and the
    call still succeeds (``views_pulled`` in the result says what happened).
    """
    _ensure_tables()
    conn = get_connection()
    row = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Submission #{submission_id} not found")
    sub = dict(row)
    conn.close()

    pulled = False
    if views is None:
        result = _try_youtube_pull(sub["video_url"])
        if result is not None:
            views, pulled_likes = result
            if likes is None:
                likes = pulled_likes
            pulled = True
    if views is None:
        views = sub["views"]
    if likes is None:
        likes = sub["likes"]

    views = max(0, int(views))
    likes = max(0, int(likes))
    earnings = _earnings_for(views, sub["cpm"])

    conn = get_connection()
    conn.execute("""
        UPDATE submissions
        SET views = ?, likes = ?, estimated_earnings = ?, updated_at = ?
        WHERE id = ?
    """, (views, likes, earnings, _now_iso(), submission_id))
    conn.commit()
    row = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
    conn.close()
    out = dict(row)
    out["views_pulled"] = pulled
    log_event("INFO", "Earnings",
              f"Submission #{submission_id} updated: {views} views, "
              f"${earnings:.2f} estimated")
    return out


def get_submission(submission_id: int):
    _ensure_tables()
    conn = get_connection()
    row = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_submissions(streamer: str = None, platform: str = None, limit: int = 50) -> list:
    _ensure_tables()
    query = "SELECT * FROM submissions"
    clauses, params = [], []
    if streamer:
        clauses.append("streamer = ?")
        params.append(streamer.lower())
    if platform:
        clauses.append("platform = ?")
        params.append(platform.lower())
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def earnings_summary(streamer: str = None, month: str = None) -> dict:
    """Totals + breakdowns of estimated earnings.

    Args:
        streamer: Optional Kick username to filter by.
        month: Optional "YYYY-MM" to filter by submission month.

    Returns:
        {total_earnings, total_views, total_likes, submission_count,
         by_streamer, by_platform, by_month}
        where each breakdown maps a key to {earnings, views, submissions}.
    """
    _ensure_tables()
    clauses, params = [], []
    if streamer:
        clauses.append("streamer = ?")
        params.append(streamer.lower())
    if month:
        clauses.append("strftime('%Y-%m', submitted_at) = ?")
        params.append(month)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    conn = get_connection()
    totals = conn.execute(
        f"SELECT COUNT(*) AS n, COALESCE(SUM(views), 0) AS views, "
        f"COALESCE(SUM(likes), 0) AS likes, "
        f"COALESCE(SUM(estimated_earnings), 0.0) AS earnings FROM submissions{where}",
        params,
    ).fetchone()

    def _breakdown(column):
        rows = conn.execute(
            f"SELECT {column} AS k, COUNT(*) AS n, COALESCE(SUM(views), 0) AS views, "
            f"COALESCE(SUM(estimated_earnings), 0.0) AS earnings "
            f"FROM submissions{where} GROUP BY {column} ORDER BY earnings DESC",
            params,
        ).fetchall()
        return {
            r["k"]: {"earnings": round(r["earnings"], 2), "views": r["views"],
                    "submissions": r["n"]}
            for r in rows
        }

    by_streamer = _breakdown("streamer")
    by_platform = _breakdown("platform")
    by_month = _breakdown("strftime('%Y-%m', submitted_at)")
    conn.close()

    return {
        "total_earnings": round(totals["earnings"], 2),
        "total_views": totals["views"],
        "total_likes": totals["likes"],
        "submission_count": totals["n"],
        "by_streamer": by_streamer,
        "by_platform": by_platform,
        "by_month": by_month,
    }
