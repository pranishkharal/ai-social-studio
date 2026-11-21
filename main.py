"""Cross-platform launcher for the AI Social Studio dashboard.

Works the same on Windows, macOS and Linux:

    python main.py                     # localhost:8500 + open browser
    python main.py --host 0.0.0.0      # reachable on the local network
                                       # (e.g. from an Android phone on
                                       # the same Wi-Fi — set a
                                       # DASHBOARD_PASSWORD first!)
    python main.py --port 9000 --no-browser   # server mode, no browser

Host/port can also come from the environment (DASHBOARD_HOST /
DASHBOARD_PORT in .env); CLI flags win over env vars.
"""

import argparse
import threading
import time
import webbrowser

import uvicorn

from config import DASHBOARD_HOST, DASHBOARD_PORT
from platform_utils import get_lan_ip


def open_browser(url):
    time.sleep(1.5)
    print(f"\n🌐 Opening Kick AI Clip Engine Dashboard: {url}\n")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(
        description="AI Social Studio — Kick live clipping dashboard")
    parser.add_argument("--host", default=DASHBOARD_HOST,
                        help="Bind address (default: DASHBOARD_HOST or "
                             "127.0.0.1; use 0.0.0.0 for LAN/phone access)")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT,
                        help="Port (default: DASHBOARD_PORT or 8500)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the dashboard in a browser "
                             "(for servers / VPS)")
    args = parser.parse_args()

    print("=" * 65)
    print("  🚀 KICK AI CLIP ENGINE & AUTONOMOUS SOCIAL PUBLISHER")
    print("  🟢 Lightweight Ingestion: Direct AWS IVS Stream-Copy (0% Screen Strain)")
    print("  🧠 AI Narrative Engine: Gemini 2.5 Flash + Local Whisper")
    print("  📱 TikTok Studio: Auto 9:16 Vertical Crop, Subtitles & Hooks")
    print("=" * 65)

    # Lite mode (gta6-clipper feature 17): auto-detects a weak PC
    # (under 12 GB RAM) and adapts the engine; the disk guard stops
    # the run early with a clear message instead of failing later.
    try:
        import lite_mode
        lite_note = lite_mode.note()
        if lite_note:
            print(f"  💡 {lite_note}")
        lite_mode.check_disk()
    except Exception as e:
        print(f"  ⚠️  {e}")

    local_url = f"http://localhost:{args.port}"
    print(f"\n  Dashboard: {local_url}")
    if args.host == "0.0.0.0":
        # The address to type into a phone on the same Wi-Fi.
        print(f"  On your phone (same Wi-Fi): http://{get_lan_ip()}:{args.port}")
        print("  ⚠️  The dashboard is visible to your whole local network —")
        print("     set DASHBOARD_PASSWORD in .env if you have not already.")

    # Open dashboard in default browser (unless --no-browser)
    if not args.no_browser:
        threading.Thread(target=open_browser, args=(local_url,),
                         daemon=True).start()

    # Run FastAPI server
    uvicorn.run("web_dashboard:app", host=args.host, port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
