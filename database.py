import sqlite3
import json
from datetime import datetime
from config import DATABASE_FILE

def get_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Streamers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS streamers (
            username TEXT PRIMARY KEY,
            is_live INTEGER DEFAULT 0,
            channel_id INTEGER,
            chatroom_id INTEGER,
            title TEXT,
            category TEXT,
            viewer_count INTEGER DEFAULT 0,
            playback_url TEXT,
            last_checked TIMESTAMP,
            monitoring_enabled INTEGER DEFAULT 1,
            monitoring_mode TEXT DEFAULT 'all',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Clips table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            streamer TEXT NOT NULL,
            title TEXT,
            story_summary TEXT,
            hook_banner TEXT,
            caption TEXT,
            tags TEXT,
            hype_score REAL DEFAULT 0.0,
            ai_score INTEGER DEFAULT 85,
            selection_reason TEXT,
            category TEXT DEFAULT 'Viral Potential',
            approval_status TEXT DEFAULT 'pending', -- pending, approved, rejected, published
            trigger_reason TEXT,
            duration REAL,
            raw_path TEXT,
            edited_path TEXT,
            status TEXT DEFAULT 'detected',
            blue_comment TEXT,
            youtube_title TEXT,
            youtube_description TEXT,
            instagram_caption TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # In-App Notifications table (PRD Section 13)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            type TEXT DEFAULT 'highlight_ready',
            streamer TEXT,
            clip_id INTEGER,
            ai_score REAL,
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # System logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT,
            source TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Safe migrations for existing SQLite database
    def ensure_column(table, col_name, col_def):
        existing_cols = [c[1] for c in cursor.execute(f"PRAGMA table_info({table})").fetchall()]
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")

    ensure_column("clips", "category", "TEXT DEFAULT 'Viral Potential'")
    ensure_column("clips", "ai_score", "INTEGER DEFAULT 85")
    ensure_column("clips", "selection_reason", "TEXT")
    ensure_column("clips", "approval_status", "TEXT DEFAULT 'pending'")
    ensure_column("clips", "youtube_title", "TEXT")
    ensure_column("clips", "youtube_description", "TEXT")
    ensure_column("clips", "instagram_caption", "TEXT")
    ensure_column("clips", "detected_streamers", "TEXT")
    ensure_column("clips", "quote_line", "TEXT")
    ensure_column("clips", "qc_report", "TEXT")
    ensure_column("streamers", "monitoring_enabled", "INTEGER DEFAULT 1")
    ensure_column("streamers", "monitoring_mode", "TEXT DEFAULT 'all'")
    ensure_column("streamers", "platform", "TEXT DEFAULT 'kick'")

    conn.commit()
    conn.close()

def log_event(level: str, source: str, message: str):
    try:
        conn = get_connection()
        conn.execute("INSERT INTO system_logs (level, source, message) VALUES (?, ?, ?)", (level, source, message))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Log Error: {e}")

def update_streamer_status(data: dict):
    conn = get_connection()
    data.setdefault("platform", "kick")
    conn.execute('''
        INSERT INTO streamers (username, platform, is_live, channel_id, chatroom_id, title, category, viewer_count, playback_url, last_checked)
        VALUES (:username, :platform, :is_live, :channel_id, :chatroom_id, :title, :category, :viewer_count, :playback_url, :last_checked)
        ON CONFLICT(username) DO UPDATE SET
            platform = COALESCE(excluded.platform, streamers.platform, 'kick'),
            is_live = excluded.is_live,
            channel_id = excluded.channel_id,
            chatroom_id = excluded.chatroom_id,
            title = excluded.title,
            category = excluded.category,
            viewer_count = excluded.viewer_count,
            playback_url = excluded.playback_url,
            last_checked = excluded.last_checked
    ''', data)
    conn.commit()
    conn.close()

def add_clip(clip_data: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO clips (streamer, title, story_summary, hook_banner, caption, tags, hype_score, trigger_reason, duration, raw_path, edited_path, status)
        VALUES (:streamer, :title, :story_summary, :hook_banner, :caption, :tags, :hype_score, :trigger_reason, :duration, :raw_path, :edited_path, :status)
    ''', clip_data)
    clip_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return clip_id

def update_clip_status(clip_id: int, status: str, edited_path: str = None, title: str = None,
                       caption: str = None, tags: str = None, hook_banner: str = None,
                       quote_line: str = None,
                       story_summary: str = None, blue_comment: str = None, category: str = None,
                       ai_score: int = None, selection_reason: str = None,
                       youtube_title: str = None, youtube_description: str = None,
                       instagram_caption: str = None, approval_status: str = None,
                       detected_streamers: str = None, qc_report: str = None):
    conn = get_connection()
    updates = ["status = ?"]
    params = [status]
    
    if edited_path:
        updates.append("edited_path = ?")
        params.append(edited_path)
    if title:
        updates.append("title = ?")
        params.append(title)
    if caption:
        updates.append("caption = ?")
        params.append(caption)
    if tags:
        updates.append("tags = ?")
        params.append(tags)
    if hook_banner:
        updates.append("hook_banner = ?")
        params.append(hook_banner)
    if quote_line:
        updates.append("quote_line = ?")
        params.append(quote_line)
    if story_summary:
        updates.append("story_summary = ?")
        params.append(story_summary)
    if blue_comment:
        updates.append("blue_comment = ?")
        params.append(blue_comment)
    if category:
        updates.append("category = ?")
        params.append(category)
    if ai_score is not None:
        updates.append("ai_score = ?")
        params.append(ai_score)
    if selection_reason:
        updates.append("selection_reason = ?")
        params.append(selection_reason)
    if youtube_title:
        updates.append("youtube_title = ?")
        params.append(youtube_title)
    if youtube_description:
        updates.append("youtube_description = ?")
        params.append(youtube_description)
    if instagram_caption:
        updates.append("instagram_caption = ?")
        params.append(instagram_caption)
    if approval_status:
        updates.append("approval_status = ?")
        params.append(approval_status)
    if detected_streamers:
        updates.append("detected_streamers = ?")
        params.append(detected_streamers)
    if qc_report:
        updates.append("qc_report = ?")
        params.append(qc_report)

    params.append(clip_id)
    query = f"UPDATE clips SET {', '.join(updates)} WHERE id = ?"
    conn.execute(query, params)
    conn.commit()
    conn.close()

def update_clip_approval(clip_id: int, approval_status: str):
    """Updates human approval status: pending, approved, rejected, published."""
    conn = get_connection()
    conn.execute("UPDATE clips SET approval_status = ? WHERE id = ?", (approval_status, clip_id))
    conn.commit()
    conn.close()

def add_notification(title: str, message: str, notif_type: str = "highlight_ready", 
                     streamer: str = None, clip_id: int = None, ai_score: float = None) -> int:
    """Creates an in-app notification for the customer (PRD Section 13)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO notifications (title, message, type, streamer, clip_id, ai_score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (title, message, notif_type, streamer, clip_id, ai_score))
    notif_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return notif_id

def get_notifications(unread_only: bool = False, limit: int = 30):
    conn = get_connection()
    query = "SELECT * FROM notifications"
    if unread_only:
        query += " WHERE is_read = 0"
    query += " ORDER BY id DESC LIMIT ?"
    rows = conn.execute(query, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def mark_notification_read(notif_id: int):
    conn = get_connection()
    conn.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notif_id,))
    conn.commit()
    conn.close()

def delete_clip(clip_id: int):
    """Permanently deletes clip from DB and removes both raw and edited MP4 files from disk."""
    conn = get_connection()
    row = conn.execute("SELECT raw_path, edited_path FROM clips WHERE id = ?", (clip_id,)).fetchone()
    if row:
        import os
        for p in [row["raw_path"], row["edited_path"]]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    conn.execute("DELETE FROM clips WHERE id = ?", (clip_id,))
    conn.commit()
    conn.close()

def mark_all_notifications_read():
    conn = get_connection()
    conn.execute("UPDATE notifications SET is_read = 1")
    conn.commit()
    conn.close()

def toggle_streamer_monitoring(username: str, enabled: bool):
    conn = get_connection()
    conn.execute("UPDATE streamers SET monitoring_enabled = ? WHERE username = ?", (1 if enabled else 0, username.lower()))
    conn.commit()
    conn.close()

def get_all_streamers():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM streamers ORDER BY is_live DESC, username ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_clips():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM clips ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_recent_logs(limit: int = 50):
    conn = get_connection()
    rows = conn.execute("SELECT * FROM system_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

init_db()
