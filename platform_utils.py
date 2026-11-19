"""
platform_utils.py — OS detection, font discovery, and small cross-platform
helpers shared by every module that touches fonts, paths, or subprocesses.

Supported platforms: Windows, macOS, Linux. Stdlib only. Nothing here
raises for a missing font, directory, or tool — callers get None or a
documented fallback instead, so the engine runs anywhere.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# OS detection (the `sys_platform` override exists so tests can pretend to
# be another OS without actually running on it)
# ---------------------------------------------------------------------------

def get_os(sys_platform=None):
    """Return "windows", "macos", "linux", or "other"."""
    p = sys_platform if sys_platform is not None else sys.platform
    if p.startswith("win"):
        return "windows"
    if p == "darwin":
        return "macos"
    if p.startswith("linux"):
        return "linux"
    return "other"


def is_windows(sys_platform=None):
    return get_os(sys_platform) == "windows"


def is_macos(sys_platform=None):
    return get_os(sys_platform) == "macos"


def is_linux(sys_platform=None):
    return get_os(sys_platform) == "linux"


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

def windows_fonts_dir():
    """The Windows Fonts directory (from WINDIR/SystemRoot, with the
    classic C:\\Windows fallback). Returned on every platform so tests
    can inspect it; it simply does not exist off Windows."""
    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows"
    return Path(windir) / "Fonts"


def font_directories(os_name=None):
    """Font directories for the platform, in preference order.

    - Windows: %WINDIR%\\Fonts.
    - macOS: ~/Library/Fonts, /Library/Fonts, /System/Library/Fonts (+
      Supplemental, where Impact and Arial Bold live).
    - Linux: the truetype package dirs, /usr/share/fonts,
      /usr/local/share/fonts, then the user's ~/.fonts and
      ~/.local/share/fonts.
    """
    os_name = os_name or get_os()
    home = Path.home()
    if os_name == "windows":
        return [windows_fonts_dir()]
    if os_name == "macos":
        return [
            home / "Library" / "Fonts",
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
        ]
    # Linux and anything else: system dirs first, then user dirs.
    return [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype/noto"),
        Path("/usr/share/fonts/truetype/freefont"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        home / ".fonts",
        home / ".local" / "share" / "fonts",
    ]


# Bold sans fonts that can stand in for Arial Bold / Impact when the
# requested Windows names are not installed (Linux/macOS).
FALLBACK_BOLD_FONT_NAMES = [
    "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "NotoSans-Bold.ttf",
    "FreeSansBold.ttf", "Arial Bold.ttf", "Helvetica-Bold.ttf",
]
FALLBACK_FONT_NAMES = FALLBACK_BOLD_FONT_NAMES + [
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf",
    "FreeSans.ttf", "Arial.ttf", "Helvetica.ttf",
]

EMOJI_FONT_NAMES = [
    "seguiemj.ttf",           # Windows
    "Apple Color Emoji.ttc",  # macOS
    "NotoColorEmoji.ttf",     # Linux
]


def find_font(preferred_names, os_name=None, fallback_names=None):
    """Return the first existing font file for `preferred_names`, or None.

    Every preferred name is tried in every font directory first; when none
    of them exists, platform equivalents are tried (DejaVu / Liberation /
    Noto on Linux, Arial/Helvetica on macOS), so a request for
    "arialbd.ttf" still finds a real bold font on a Mac or Linux box.
    Pass fallback_names=[] to disable the equivalents.
    """
    names = list(preferred_names or [])
    dirs = font_directories(os_name)
    for directory in dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    if fallback_names is None:
        fallback_names = FALLBACK_FONT_NAMES
    for directory in dirs:
        for name in fallback_names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


def find_emoji_font(os_name=None):
    """Return the first colour-emoji font file found, or None."""
    for directory in font_directories(os_name):
        for name in EMOJI_FONT_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


def load_font(size, bold=True, emoji=False):
    """Load a Pillow font cross-platform; never raises.

    Emoji requests try the colour-emoji font first (Segoe UI Emoji on
    Windows, Apple Color Emoji on macOS, Noto Color Emoji on Linux) and
    fall through to the regular bold font when none is installed. With no
    font file available at all, Pillow's built-in default font is used.
    """
    from PIL import ImageFont

    if emoji:
        emoji_path = find_emoji_font()
        if emoji_path:
            try:
                return ImageFont.truetype(emoji_path, size)
            except Exception:
                pass
        # No colour-emoji font: fall through to the regular bold font.
    preferred = (["arialbd.ttf", "arial.ttf", "impact.ttf"] if bold
                 else ["arial.ttf", "arialbd.ttf"])
    font_path = find_font(preferred)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: load_default() takes no size
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Subprocess / network / paths
# ---------------------------------------------------------------------------

def ffmpeg_creationflags():
    """subprocess creationflags for ffmpeg/ffprobe: CREATE_NO_WINDOW on
    Windows (no console popup per render), 0 everywhere else."""
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def get_lan_ip():
    """Best-effort LAN IPv4 address of this machine, for phone access.

    Uses a UDP connect (which sends no packets) to find the address the
    machine would use to reach the internet. Falls back to the hostname
    lookup, then 127.0.0.1. Never raises.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"


def desktop_dir():
    """The user's Desktop folder (~/Desktop on all three platforms)."""
    return Path.home() / "Desktop"
