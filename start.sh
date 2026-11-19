#!/bin/bash
# AI Social Studio — macOS / Linux launcher (mirrors start_engine.bat).
# Usage:
#   ./start.sh                  # localhost dashboard + open browser
#   ./start.sh --host 0.0.0.0   # reachable from your phone on the same Wi-Fi
#   ./start.sh --no-browser     # server mode (VPS), no browser
set -e
cd "$(dirname "$0")"

# --- Pick a Python -----------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[ERROR] Python 3.11+ is required but no python3 was found."
    echo "  macOS:   brew install python@3.11 ffmpeg"
    echo "  Ubuntu:  sudo apt install python3 python3-pip ffmpeg"
    exit 1
fi

# --- ffmpeg is required for clipping ----------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[ERROR] ffmpeg is required but was not found in PATH."
    echo "  macOS:   brew install ffmpeg"
    echo "  Ubuntu:  sudo apt install ffmpeg"
    exit 1
fi

# --- Use the project venv when present, else the system Python --------------
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
fi

# --- Install dependencies on first run ---------------------------------------
if ! "$PY" -c "import fastapi, uvicorn, PIL, psutil" >/dev/null 2>&1; then
    echo "[SETUP] Installing required dependencies (first run only)..."
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
fi

# --- Go -----------------------------------------------------------------------
echo "[START] Launching AI Social Studio on http://localhost:8500 ..."
exec "$PY" main.py "$@"
