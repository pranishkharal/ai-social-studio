import json

log_path = r'C:\Users\MSI\.gemini\antigravity\brain\69f96515-da9d-4411-be11-cb7576b6448a\.system_generated\tasks\task-345.log'

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

with open(r'D:\kick-ai-clip-engine\prestige_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(videos, f, indent=2)
