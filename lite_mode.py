"""Lite mode for weak PCs (gta6-clipper feature 17).

Auto-detects a machine with under 12 GB of RAM and switches the
engine to gentler settings: 1 render at a time, 30 fps renders, only
the first hour of very long streams, and audio downloads deleted
once transcribed. Plus a disk-space guard that stops a run early
with a clear message instead of failing after an hour.

Control: LITE_MODE=1 forces it on, LITE_MODE=0 forces it off; unset
means auto-detect by RAM. main.py prints the note at startup and the
pipeline checks the disk guard before a run starts.
"""

import ctypes
import os
import shutil
import sys

# Under this much RAM, lite mode switches itself on.
RAM_THRESHOLD_GB = 12.0
# A run needs roughly this much free disk for downloads and renders.
DISK_NEED_GB = 3.0

ENV_VAR = "LITE_MODE"


class DiskFullError(Exception):
    """Raised by the disk guard when a run would not have room."""


def ram_gb():
    """Total physical RAM in GB (16.0 when it cannot be measured —
    unknown means assume enough, so lite mode stays off)."""
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong),
                            ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong)] + [
                                (f"x{i}", ctypes.c_ulonglong) for i in range(5)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 2 ** 30
        if sys.platform == "darwin":
            # macOS has no SC_PHYS_PAGES; sysctl hw.memsize is the way.
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True, timeout=10)
            return int(out.stdout.strip()) / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return 16.0


def lite_enabled():
    """True when lite mode is on: LITE_MODE=1 forces on, LITE_MODE=0
    forces off, otherwise on when RAM is under 12 GB."""
    value = os.environ.get(ENV_VAR, "")
    if value in ("0", "1"):
        return value == "1"
    return ram_gb() < RAM_THRESHOLD_GB


def settings():
    """The engine settings lite mode changes. Full mode returns the
    normal values, so callers can use this unconditionally."""
    if lite_enabled():
        return {
            "lite": True,
            "max_concurrent_renders": 1,
            "render_fps": 30,
            "max_stream_hours": 1,       # first hour of long streams only
            "delete_audio_after_transcribe": True,
        }
    return {
        "lite": False,
        "max_concurrent_renders": 3,
        "render_fps": 60,
        "max_stream_hours": None,        # no cap
        "delete_audio_after_transcribe": False,
    }


def disk_free_gb(path="."):
    """Free disk space in GB at ``path``."""
    return shutil.disk_usage(os.path.abspath(path)).free / 2 ** 30


def check_disk(path=".", need_gb=DISK_NEED_GB):
    """Disk guard: returns the free GB, or raises DiskFullError with a
    clear message when there is not enough room for a run. Call this
    before a run starts — failing early beats failing after an hour."""
    free = disk_free_gb(path)
    if free < need_gb:
        raise DiskFullError(
            f"Only {free:.1f} GB free on this disk; a run needs about "
            f"{need_gb:.0f} GB. Free some space (delete old downloads and "
            "renders), then run again.")
    return free


def note():
    """One-line status for the startup banner ("" when lite is off)."""
    if lite_enabled():
        return (f"Lite mode on ({ram_gb():.0f} GB RAM): 1 render at a time, "
                "30 fps, first hour of long streams.")
    return ""
