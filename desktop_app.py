import os
import sys
import time
import socket
import threading
import subprocess
import uvicorn

PORT = 8500
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"

def is_server_running(host=HOST, port=PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def start_backend():
    print("🚀 Starting AI Social Studio Backend Engine on port 8500...")
    uvicorn.run("web_dashboard:app", host=HOST, port=PORT, log_level="warning")

def launch_pywebview():
    try:
        import webview
        print("🖥️ Launching Native Windows Desktop Software Window (pywebview)...")
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
        print(f"⚠️ pywebview failed: {e}. Falling back to Windows Native Edge App Mode...")
        return False

def launch_edge_app_mode():
    edge_paths = [
        "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    ]
    browser_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            browser_exe = p
            break
            
    if browser_exe:
        print(f"🖥️ Launching Native Windows Standalone Window via {os.path.basename(browser_exe)}...")
        cmd = [
            browser_exe,
            f"--app={URL}",
            "--window-size=1650,980",
            "--window-position=50,50"
        ]
        proc = subprocess.Popen(cmd)
        proc.wait()
    else:
        import webbrowser
        print("🌐 Opening in default web browser...")
        webbrowser.open(URL)

if __name__ == "__main__":
    # Ensure working directory is D:\kick-ai-clip-engine
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
        launch_edge_app_mode()
