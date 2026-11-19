import json
import os
import sys
from pathlib import Path

# One-off analysis script: parse a saved task log for @prestigeclipn video
# metadata. The log path comes from the first CLI argument or the
# PRESTIGE_LOG_PATH env var — there is no machine-specific default.
log_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PRESTIGE_LOG_PATH", "")
if not log_path:
    print("Usage: python parse_prestige_videos.py <path-to-task-log>")
    print("   or set PRESTIGE_LOG_PATH to the log file path.")
    sys.exit(1)

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

videos = []
for line in lines:
    line = line.strip()
    if line.startswith('{') and '"id":' in line:
        try:
            d = json.loads(line)
            videos.append({
                "id": d.get("id"),
                "title": d.get("title"),
                "description": d.get("description"),
                "duration": d.get("duration"),
                "views": d.get("view_count"),
                "likes": d.get("like_count"),
                "comments": d.get("comment_count"),
                "shares": d.get("repost_count"),
                "saves": d.get("save_count"),
                "track": d.get("track"),
                "artist": d.get("artist"),
                "upload_date": d.get("upload_date")
            })
        except Exception as e:
            pass

print(f"Parsed {len(videos)} videos from @prestigeclipn:")
for idx, v in enumerate(videos, 1):
    print(f"\n--- VIDEO #{idx} ---")
    print(f"Title / Caption: {v['description']}")
    print(f"Duration: {v['duration']}s | Views: {v['views']:,} | Likes: {v['likes']:,} | Comments: {v['comments']}")
    print(f"Audio Track: {v['track']} by {v['artist']}")
    print(f"Link: https://www.tiktok.com/@prestigeclipn/video/{v['id']}")

out_path = Path(__file__).resolve().parent / "prestige_analysis.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(videos, f, indent=2)
