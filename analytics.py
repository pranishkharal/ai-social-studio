"""
Analytics feedback loop (Feature 8).

The AI score is a guess that is never checked against reality — until now.
This module records each posted clip's real performance (views / likes /
comments) and compares it against the clip's AI score, category and streamer,
so the auto-clip threshold and the Gemini prompt can be tuned with evidence
instead of vibes.

Data sources:
    - clip_performance rows written by :func:`record_performance`
      (manual entry from the dashboard, or fed from the earnings tracker).
    - The clips table for ai_score / category / streamer.

Outputs:
    - analytics_summary(): average views by AI-score bucket (60-69, 70-79,
      80-89, 90+), by category and by streamer, plus a threshold
      recommendation (which score bucket actually averages the most views,
      and a suggested AUTO_CLIP_MIN_SCORE).
    - monthly_report(year, month): totals + top clips for a calendar month.

No new env vars.
"""

from database import get_connection, log_event

# AI-score buckets: (label, low, high) — high is exclusive except for 90+.
SCORE_BUCKETS = [("60-69", 60, 70), ("70-79", 70, 80),
                 ("80-89", 80, 90), ("90+", 90, 101)]

# A bucket needs at least this many clips before its average is trusted
# for the threshold recommendation.
MIN_SAMPLES_FOR_RECOMMENDATION = 3


def init_performance_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clip_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clip_id INTEGER NOT NULL,
            platform TEXT NOT NULL,
            views INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def _ensure_tables():
    init_performance_table()


def _bucket_for(score: int) -> str:
    for label, low, high in SCORE_BUCKETS:
        if low <= score < high:
            return label
    return "below-60" if score < 60 else "90+"


def record_performance(clip_id: int, platform: str, views: int,
                       likes: int = 0, comments: int = 0) -> dict:
    """Records one performance snapshot for a posted clip.

    Call this whenever fresh view counts come in (manual entry, or the
    earnings tracker's update_views). Snapshots accumulate, so the summary
    always uses the LATEST snapshot per (clip, platform) pair.
    """
    _ensure_tables()
    platform = (platform or "").strip().lower()
    if platform not in ("tiktok", "youtube", "instagram"):
        raise ValueError(f"Unknown platform '{platform}': use tiktok, youtube or instagram")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO clip_performance (clip_id, platform, views, likes, comments)
        VALUES (?, ?, ?, ?, ?)
    """, (clip_id, platform, max(0, int(views)), max(0, int(likes)), max(0, int(comments))))
    perf_id = cursor.lastrowid
    conn.commit()
    row = conn.execute("SELECT * FROM clip_performance WHERE id = ?", (perf_id,)).fetchone()
    conn.close()
    log_event("INFO", "Analytics",
              f"Recorded performance for clip #{clip_id} ({platform}): {views} views")
    return dict(row)


def _latest_performance_rows():
    """Latest snapshot per (clip_id, platform), joined with clip metadata."""
    _ensure_tables()
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.clip_id, p.platform, p.views, p.likes, p.comments, p.recorded_at,
               c.ai_score, c.category, c.streamer, c.title
        FROM clip_performance p
        JOIN clips c ON c.id = p.clip_id
        WHERE p.id IN (
            SELECT MAX(id) FROM clip_performance GROUP BY clip_id, platform
        )
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _avg_views(rows, key_fn):
    """Groups rows by key_fn -> {key: {clips, avg_views, total_views}}."""
    groups = {}
    for r in rows:
        key = key_fn(r)
        g = groups.setdefault(key, {"clips": 0, "total_views": 0})
        g["clips"] += 1
        g["total_views"] += r["views"] or 0
    return {
        k: {"clips": v["clips"], "total_views": v["total_views"],
            "avg_views": round(v["total_views"] / v["clips"], 1)}
        for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["total_views"])
    }


def _recommend_threshold(by_bucket: dict) -> dict:
    """Picks the score bucket with the best average views and suggests a
    minimum auto-clip score from it.

    Buckets with fewer than MIN_SAMPLES_FOR_RECOMMENDATION clips are ignored
    unless no bucket qualifies, in which case the best available bucket is
    used and flagged low-confidence.
    """
    if not by_bucket:
        return {
            "suggested_auto_clip_min_score": 80,
            "best_bucket": None,
            "confidence": "none",
            "reason": "No performance data yet — keeping the default threshold of 80.",
        }

    trusted = {k: v for k, v in by_bucket.items()
               if v["clips"] >= MIN_SAMPLES_FOR_RECOMMENDATION}
    pool = trusted if trusted else by_bucket
    best_label = max(pool, key=lambda k: pool[k]["avg_views"])
    best = pool[best_label]

    low = dict((label, lo) for label, lo, _hi in SCORE_BUCKETS).get(best_label, 80)
    suggested = max(60, min(90, low))
    return {
        "suggested_auto_clip_min_score": suggested,
        "best_bucket": best_label,
        "best_bucket_avg_views": best["avg_views"],
        "best_bucket_clips": best["clips"],
        "confidence": "high" if trusted else "low",
        "reason": (
            f"Clips scored {best_label} average {best['avg_views']:,.0f} views "
            f"({best['clips']} clips) — the best of any score bucket. "
            f"{'Enough samples to trust this.' if trusted else 'Small sample — treat as a hint, not proof.'} "
            f"Suggested AUTO_CLIP_MIN_SCORE: {suggested}."
        ),
    }


def analytics_summary() -> dict:
    """Average views by AI-score bucket, category and streamer, plus a
    threshold recommendation. Returns empty breakdowns when no data exists."""
    rows = _latest_performance_rows()
    by_bucket = _avg_views(rows, lambda r: _bucket_for(r.get("ai_score") or 0))
    by_category = _avg_views(rows, lambda r: r.get("category") or "Unknown")
    by_streamer = _avg_views(rows, lambda r: r.get("streamer") or "Unknown")
    total_views = sum(r["views"] or 0 for r in rows)
    return {
        "tracked_clips": len({r["clip_id"] for r in rows}),
        "tracked_posts": len(rows),
        "total_views": total_views,
        "avg_views_per_post": round(total_views / len(rows), 1) if rows else 0,
        "by_score_bucket": by_bucket,
        "by_category": by_category,
        "by_streamer": by_streamer,
        "threshold_recommendation": _recommend_threshold(by_bucket),
    }


def monthly_report(year: int, month: int) -> dict:
    """Totals + top clips for one calendar month (by recorded_at)."""
    _ensure_tables()
    month_key = f"{int(year):04d}-{int(month):02d}"
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.clip_id, p.platform, p.views, p.likes, p.comments, p.recorded_at,
               c.ai_score, c.category, c.streamer, c.title
        FROM clip_performance p
        JOIN clips c ON c.id = p.clip_id
        WHERE strftime('%Y-%m', p.recorded_at) = ?
          AND p.id IN (SELECT MAX(id) FROM clip_performance GROUP BY clip_id, platform)
        ORDER BY p.views DESC
    """, (month_key,)).fetchall()
    conn.close()
    rows = [dict(r) for r in rows]
    return {
        "year": int(year),
        "month": int(month),
        "month_key": month_key,
        "tracked_clips": len({r["clip_id"] for r in rows}),
        "tracked_posts": len(rows),
        "total_views": sum(r["views"] or 0 for r in rows),
        "total_likes": sum(r["likes"] or 0 for r in rows),
        "total_comments": sum(r["comments"] or 0 for r in rows),
        "by_platform": _avg_views(rows, lambda r: r["platform"]),
        "top_clips": [
            {
                "clip_id": r["clip_id"],
                "title": r["title"],
                "streamer": r["streamer"],
                "category": r["category"],
                "ai_score": r["ai_score"],
                "platform": r["platform"],
                "views": r["views"],
                "likes": r["likes"],
            }
            for r in rows[:10]
        ],
    }


def list_performance(limit: int = 50) -> list:
    _ensure_tables()
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.*, c.title, c.streamer, c.ai_score, c.category
        FROM clip_performance p
        LEFT JOIN clips c ON c.id = p.clip_id
        ORDER BY p.id DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
