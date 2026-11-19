"""Native desktop app launcher (Windows, macOS, Linux).

Starts the dashboard backend, then opens it in a standalone window:
pywebview first (native window on all three platforms), falling back to
the system browser in app mode (Edge/Chrome on Windows, Chrome on macOS,
Chrome/Chromium on Linux), and finally to the default browser.
"""

import os
import shutil
import subprocess
import sys
import threading
import time

import uvicorn

from config import DASHBOARD_PORT
from platform_utils import get_os, is_macos, is_windows

PORT = DASHBOARD_PORT
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"


def is_server_running(host=HOST, port=PORT):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_backend():
    print("🚀 Starting AI Social Studio Backend Engine on port 8500...")
    uvicorn.run("web_dashboard:app", host=HOST, port=PORT, log_level="warning")


def launch_pywebview():
    try:
        import webview
        print("🖥️ Launching native desktop window (pywebview)...")
        window = webview.create_window(
            title="AI Social Studio v1.0 - Kick Live Stream Clipping & Publishing",
            url=URL,
            width=1650,
            height=980,
            min_size=(1200, 750),
            confirm_close=False,
            text_select=True
        )
        webview.start()
        return True
    except Exception as e:
        print(f"⚠️ pywebview failed: {e}. Falling back to browser app mode...")
        return False


def _windows_browser_exe():
    """Edge or Chrome on Windows, or None."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def launch_browser_app_mode():
    """Open the dashboard in a standalone browser app window.

    Windows: Edge/Chrome --app. macOS: `open -a "Google Chrome" --args
    --app=...`. Linux: google-chrome / chromium --app. Falls back to the
    default browser when no app-mode browser is available.
    """
    os_name = get_os()

    if is_windows():
        browser_exe = _windows_browser_exe()
        if browser_exe:
            print(f"🖥️ Launching standalone window via {os.path.basename(browser_exe)}...")
            cmd = [browser_exe, f"--app={URL}",
                   "--window-size=1650,980", "--window-position=50,50"]
            proc = subprocess.Popen(cmd)
            proc.wait()
            return

    elif is_macos():
        chrome_app = "/Applications/Google Chrome.app"
        if os.path.exists(chrome_app):
            print("🖥️ Launching standalone window via Google Chrome...")
            cmd = ["open", "-a", "Google Chrome", "--args",
                   f"--app={URL}", "--window-size=1650,980"]
            proc = subprocess.Popen(cmd)
            proc.wait()
            return

    else:  # Linux and anything else with a Chromium-based browser
        for binary in ("google-chrome", "google-chrome-stable",
                       "chromium", "chromium-browser"):
            exe = shutil.which(binary)
            if exe:
                print(f"🖥️ Launching standalone window via {binary}...")
                cmd = [exe, f"--app={URL}", "--window-size=1650,980"]
                proc = subprocess.Popen(cmd)
                proc.wait()
                return

    import webbrowser
    print("🌐 Opening in default web browser...")
    webbrowser.open(URL)


# Backwards-compatible alias (the fallback used to be Windows-only).
launch_edge_app_mode = launch_browser_app_mode


if __name__ == "__main__":
    # Run from the repo directory, wherever it lives on this machine.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Start backend server if not already alive
    if not is_server_running():
        t = threading.Thread(target=start_backend, daemon=True)
        t.start()
        # Wait until server is listening
        for _ in range(30):
            if is_server_running():
                break
            time.sleep(0.2)

    # Launch native desktop app
    success = launch_pywebview()
    if not success:
        launch_browser_app_mode()
