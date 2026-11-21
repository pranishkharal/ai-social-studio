@echo off
title AI Social Studio v1.0 - Control Center
cd /d "%~dp0"

:MENU
cls
echo ================================================================
echo           AI SOCIAL STUDIO v1.0 - KICK CLIP ENGINE
echo ================================================================
echo  * Working Directory: %~dp0
echo  * Database & Storage: 100%% Portable on D: Drive
echo  * Live Stream Watcher, AI Scoring & Prestige Safe Zone Overlays
echo ================================================================
echo.
echo  CHOOSE LAUNCH MODE:
echo  ----------------------------------------------------------------
echo   [1] Native Windows Desktop App (Standalone Window Mode)
echo   [2] Web App / Website (Opens in Browser: http://localhost:8500)
echo   [3] Create Desktop Shortcut (Add 1-Click Icon to Windows Desktop)
echo   [4] Update from GitHub (Pull latest changes & dependencies)
echo   [5] Exit
echo  ----------------------------------------------------------------
echo.

set /p CHOICE="Enter choice [1, 2, 3, 4, or 5] (Default is 1): "
if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" goto NATIVE_APP
if "%CHOICE%"=="2" goto WEB_APP
if "%CHOICE%"=="3" goto SHORTCUT
if "%CHOICE%"=="4" goto UPDATE_GIT
if "%CHOICE%"=="5" exit /b
goto MENU

:NATIVE_APP
echo.
echo [START] Launching Native Windows Desktop Software Window...
python desktop_app.py
pause
goto MENU

:WEB_APP
echo.
echo [START] Launching Web App in Default Browser...
start "" "http://localhost:8500"
python main.py
pause
goto MENU

:SHORTCUT
echo.
echo [SETUP] Creating 1-Click Desktop Shortcut on Windows Desktop...
python create_desktop_shortcut.py
echo.
pause
goto MENU

:UPDATE_GIT
echo.
call update.bat
goto MENU
