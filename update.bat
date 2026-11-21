@echo off
title AI Social Studio - GitHub Auto Updater
cd /d "%~dp0"

echo ================================================================
echo           AI SOCIAL STUDIO - GITHUB UPDATER
echo ================================================================
echo  * Checking for new updates from GitHub...
echo.

git fetch origin main >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to connect to GitHub. Please check your internet connection.
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('git rev-parse --short HEAD') do set "LOCAL_HASH=%%a"
for /f "tokens=*" %%b in ('git rev-parse --short origin/main') do set "REMOTE_HASH=%%b"

if "%LOCAL_HASH%"=="%REMOTE_HASH%" goto UP_TO_DATE

echo [UPDATE FOUND] New updates detected on GitHub!
echo Current local commit: %LOCAL_HASH%
echo Latest GitHub commit: %REMOTE_HASH%
echo.
echo Pulling updates...
git pull origin main
echo.
echo [SETUP] Updating dependencies...
pip install -r requirements.txt --quiet
echo.
echo ================================================================
echo [SUCCESS] Your computer is now up-to-date with GitHub!
echo ================================================================
echo.
pause
exit /b 0

:UP_TO_DATE
echo [OK] Your files are already up-to-date with GitHub [%LOCAL_HASH%].
echo.
pause
exit /b 0
