@echo off
title Kick AI Clip Engine - Control Center
cd /d "%~dp0"

echo ================================================================
echo           KICK AI CLIP ENGINE - OFFICIAL PAYOUT EDITION
echo ================================================================
echo  * Working Directory: %~dp0
echo  * 100% Portable on D: Drive (Ready for Laptop or Any PC)
echo  * Live Stream Watch & Clip Studio
echo  * Full-Width Kick Banner Overlay & Clean On-Screen Text
echo ================================================================
echo.

python -c "import fastapi, uvicorn, PIL, psutil" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Installing required dependencies...
    pip install fastapi uvicorn pillow psutil requests faster-whisper yt-dlp
)

echo [START] Launching Autonomous Server on http://localhost:8500 ...
start "" "http://localhost:8500"
python main.py

pause
