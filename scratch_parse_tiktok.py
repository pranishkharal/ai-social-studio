import re
import json
from pathlib import Path
from curl_cffi import requests

html_path = Path(__file__).resolve().parent / "static" / "tiktok_page.html"
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

m = re.search(r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>', html, re.DOTALL)
if m:
    data = json.loads(m.group(1))
    scope = data.get('__DEFAULT_SCOPE__', {})
    user = scope.get('webapp.user-detail', {}).get('userInfo', {}).get('user', {})
    secUid = user.get('secUid')
    userId = user.get('id')
    print('secUid:', secUid)
    print('userId:', userId)

    # Let's test fetching videos via yt-dlp using tiktokuser:secUid or direct API
    if secUid:
        cmd = f'tiktokuser:{secUid}'
        print('Target:', cmd)
