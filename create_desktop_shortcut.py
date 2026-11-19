"""Create a 1-click desktop launcher for AI Social Studio.

Windows: a .lnk shortcut to start.bat (via VBScript, as before — but with
paths derived from this repo instead of a hardcoded D:\\ drive).
macOS: a .command script on the Desktop that runs start.sh.
Linux: a .desktop launcher on the Desktop (and in
~/.local/share/applications/ so it appears in the app menu).
"""

import os
import stat
import sys
from pathlib import Path

from platform_utils import desktop_dir, get_os

REPO_DIR = Path(__file__).resolve().parent
DESCRIPTION = "AI Social Studio v1.0 - Commercial Kick Live Stream Clipping"


def create_windows_shortcut():
    desktop = desktop_dir()
    shortcut_path = desktop / "AI Social Studio.lnk"
    target_path = REPO_DIR / "start.bat"
    icon_path = REPO_DIR / "static" / "app_icon.ico"

    vbs_script = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sLinkFile = "{shortcut_path}"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{target_path}"
oLink.WorkingDirectory = "{REPO_DIR}"
oLink.Description = "{DESCRIPTION}"
oLink.IconLocation = "{icon_path},0"
oLink.Save
"""
    vbs_file = REPO_DIR / "temp_shortcut.vbs"
    vbs_file.write_text(vbs_script, encoding="utf-8")
    try:
        os.system(f'cscript //nologo "{vbs_file}"')
    finally:
        if vbs_file.exists():
            vbs_file.unlink()

    if shortcut_path.exists():
        print(f"SUCCESS: Desktop shortcut created at: {shortcut_path}")
        return True
    print("FAILED to create shortcut.")
    return False


def create_macos_launcher():
    """A .command file: double-clicking it opens Terminal and runs start.sh."""
    launcher = desktop_dir() / "AI Social Studio.command"
    launcher.write_text(
        "#!/bin/bash\n"
        f'cd "{REPO_DIR}"\n'
        "./start.sh\n",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"SUCCESS: Desktop launcher created at: {launcher}")
    print("NOTE: the first double-click may ask you to allow it in "
          "System Settings > Privacy & Security (downloaded-script Gatekeeper check).")
    return True


def create_linux_launcher():
    """A .desktop file on the Desktop + in the app menu."""
    entry = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=AI Social Studio\n"
        f"Comment={DESCRIPTION}\n"
        f"Exec={REPO_DIR / 'start.sh'}\n"
        f"Path={REPO_DIR}\n"
        f"Icon={REPO_DIR / 'static' / 'icon-512.png'}\n"
        "Terminal=true\n"
        "Categories=Utility;\n"
    )
    desktop_file = desktop_dir() / "AI Social Studio.desktop"
    desktop_file.write_text(entry, encoding="utf-8")
    desktop_file.chmod(desktop_file.stat().st_mode | stat.S_IXUSR)
    apps_dir = Path.home() / ".local" / "share" / "applications"
    try:
        apps_dir.mkdir(parents=True, exist_ok=True)
        (apps_dir / "ai-social-studio.desktop").write_text(entry, encoding="utf-8")
    except Exception as e:
        print(f"NOTE: could not add to the app menu: {e}")
    print(f"SUCCESS: Desktop launcher created at: {desktop_file}")
    print("NOTE: on some desktops, right-click the icon and choose "
          "\"Allow Launching\" the first time.")
    return True


def create_shortcut():
    os_name = get_os()
    if os_name == "windows":
        return create_windows_shortcut()
    if os_name == "macos":
        return create_macos_launcher()
    if os_name == "linux":
        return create_linux_launcher()
    print(f"Unsupported platform for desktop shortcuts: {sys.platform}")
    return False


if __name__ == "__main__":
    create_shortcut()
