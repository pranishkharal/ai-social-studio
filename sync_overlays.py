import urllib.request
import json
import os
from pathlib import Path

overlay_dir = Path(r"D:\kick-ai-clip-engine\overlays")
overlay_dir.mkdir(parents=True, exist_ok=True)

try:
    req = urllib.request.Request("https://api.github.com/repos/pranishkharal/prxno/contents/overlays", headers={"User-Agent": "Mozilla/5.0"})
    res = urllib.request.urlopen(req)
    files = json.loads(res.read().decode())
    print(f"Found {len(files)} items in prxno overlays.")
    for item in files:
        name = item.get("name", "")
        dl_url = item.get("download_url")
        if dl_url and (name.endswith(".webp") or name.endswith(".png")):
            dest = overlay_dir / name
            if not dest.exists():
                urllib.request.urlretrieve(dl_url, dest)
                print("Downloaded:", name)
            else:
                print("Cached:", name)
    print("Overlay sync complete!")
except Exception as e:
    print("Download error:", e)
