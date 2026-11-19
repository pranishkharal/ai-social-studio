"""Tests for the cross-platform helpers (platform_utils) and their consumers.

The OS is mocked via platform_utils' `sys_platform` / `os_name` parameters
and via monkeypatched Path.home() / env vars — nothing here depends on
which machine the suite runs on.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import platform_utils
from platform_utils import (
    EMOJI_FONT_NAMES,
    FALLBACK_BOLD_FONT_NAMES,
    desktop_dir,
    ffmpeg_creationflags,
    find_emoji_font,
    find_font,
    font_directories,
    get_lan_ip,
    get_os,
    load_font,
)


# --- OS detection ---------------------------------------------------------

@pytest.mark.parametrize("platform,expected", [
    ("win32", "windows"),
    ("darwin", "macos"),
    ("linux", "linux"),
    ("freebsd", "other"),
])
def test_get_os(platform, expected):
    assert get_os(sys_platform=platform) == expected


def test_get_os_defaults_to_current_platform():
    assert get_os() in ("windows", "macos", "linux", "other")


# --- Font directories per platform ----------------------------------------

def test_windows_font_dirs_use_windir(monkeypatch):
    monkeypatch.setenv("WINDIR", r"D:\Windows")
    dirs = font_directories(os_name="windows")
    assert any(str(d).endswith(os.path.join("Windows", "Fonts")) for d in dirs)


def test_macos_font_dirs_include_supplemental():
    dirs = font_directories(os_name="macos")
    assert any("Supplemental" in str(d) for d in dirs)
    assert any(str(d) == "/Library/Fonts" for d in dirs)


def test_linux_font_dirs_include_truetype():
    dirs = font_directories(os_name="linux")
    assert any("dejavu" in str(d) for d in dirs)
    assert any("liberation" in str(d) for d in dirs)


def test_font_directories_are_paths():
    for os_name in ("windows", "macos", "linux", "other"):
        assert all(isinstance(d, Path) for d in font_directories(os_name=os_name))


# --- find_font ------------------------------------------------------------

def test_find_font_finds_real_font_in_fake_dir(tmp_path, monkeypatch):
    fake_fonts = tmp_path / "Fonts"
    fake_fonts.mkdir()
    (fake_fonts / "arialbd.ttf").write_bytes(b"not a real font")
    monkeypatch.setattr(
        platform_utils, "font_directories",
        lambda os_name=None: [fake_fonts])
    found = find_font(["arialbd.ttf"], os_name="windows")
    assert found == str(fake_fonts / "arialbd.ttf")


def test_find_font_returns_none_when_nothing_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [tmp_path])
    assert find_font(["arialbd.ttf"], os_name="linux", fallback_names=[]) is None


def test_find_font_prefers_preferred_name_over_equivalent(tmp_path, monkeypatch):
    d = tmp_path / "fonts"
    d.mkdir()
    (d / "arialbd.ttf").write_bytes(b"x")
    (d / "DejaVuSans-Bold.ttf").write_bytes(b"x")
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [d])
    found = find_font(["arialbd.ttf"], os_name="linux")
    assert found == str(d / "arialbd.ttf")


def test_find_font_maps_windows_name_to_linux_equivalent(tmp_path, monkeypatch):
    """arialbd.ttf must resolve to DejaVuSans-Bold on a Linux box that only
    has the DejaVu fonts (this is what the thumbnail/editor tests rely on)."""
    d = tmp_path / "fonts"
    d.mkdir()
    (d / "DejaVuSans-Bold.ttf").write_bytes(b"x")
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [d])
    assert find_font(["arialbd.ttf"], os_name="linux") == str(d / "DejaVuSans-Bold.ttf")


def test_find_font_macos_equivalent(tmp_path, monkeypatch):
    d = tmp_path / "Fonts"
    d.mkdir()
    (d / "Arial Bold.ttf").write_bytes(b"x")
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [d])
    assert find_font(["arialbd.ttf"], os_name="macos") == str(d / "Arial Bold.ttf")


# --- Emoji font -----------------------------------------------------------

def test_find_emoji_font_finds_noto(tmp_path, monkeypatch):
    d = tmp_path / "fonts"
    d.mkdir()
    (d / "NotoColorEmoji.ttf").write_bytes(b"x")
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [d])
    assert find_emoji_font(os_name="linux") == str(d / "NotoColorEmoji.ttf")


def test_find_emoji_font_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [tmp_path])
    assert find_emoji_font(os_name="windows") is None


def test_emoji_font_names_cover_all_platforms():
    assert "seguiemj.ttf" in EMOJI_FONT_NAMES          # Windows
    assert "Apple Color Emoji.ttc" in EMOJI_FONT_NAMES  # macOS
    assert "NotoColorEmoji.ttf" in EMOJI_FONT_NAMES     # Linux


# --- load_font never crashes ----------------------------------------------

def test_load_font_never_crashes_without_fonts(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [tmp_path])
    assert load_font(38, bold=True) is not None
    assert load_font(36, emoji=True) is not None


def test_load_font_returns_pillow_font(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [tmp_path])
    font = load_font(24)
    assert font is not None
    # Pillow default font still exposes getbbox / getlength.
    assert font.getlength("hello") >= 0


# --- Consumers delegate to platform_utils ---------------------------------

def test_clip_editor_resolve_font_path_maps_to_linux_equivalent(tmp_path, monkeypatch):
    import clip_editor
    d = tmp_path / "fonts"
    d.mkdir()
    (d / "DejaVuSans-Bold.ttf").write_bytes(b"x")
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [d])
    assert clip_editor.resolve_font_path(["arialbd.ttf"]) == str(d / "DejaVuSans-Bold.ttf")


def test_clip_editor_load_font_uses_module_resolve_font_path(monkeypatch):
    """The monkeypatch seam test_thumbnail relies on: load_font must call
    the module-level resolve_font_path, so stubbing it to None still
    yields a font (Pillow default)."""
    import clip_editor
    monkeypatch.setattr(clip_editor, "resolve_font_path", lambda names: None)
    assert clip_editor.load_font(38, bold=True) is not None


def test_thumbnail_bold_font_never_crashes(tmp_path, monkeypatch):
    from thumbnail import _load_bold_font
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [tmp_path])
    assert _load_bold_font(36) is not None


def test_kick_program_candidate_fonts_include_impact_and_linux(tmp_path, monkeypatch):
    from kick_program import _candidate_display_fonts
    monkeypatch.setattr(platform_utils, "font_directories",
                        lambda os_name=None: [Path("/usr/share/fonts/truetype/dejavu")])
    candidates = _candidate_display_fonts()
    assert any("Impact" in c or "impact" in c for c in candidates)
    assert any("DejaVuSans-Bold.ttf" in c for c in candidates)
    assert all(isinstance(c, str) for c in candidates)


# --- Misc helpers ---------------------------------------------------------

def test_ffmpeg_creationflags_is_int():
    assert isinstance(ffmpeg_creationflags(), int)


def test_desktop_dir_is_under_home():
    assert str(desktop_dir()).startswith(str(Path.home()))
    assert desktop_dir().name == "Desktop"


def test_get_lan_ip_returns_string():
    ip = get_lan_ip()
    assert isinstance(ip, str) and ip
    assert ip.count(".") == 3  # dotted-quad IPv4


def test_escape_path_posix_unchanged():
    from clip_editor import ClipEditor
    editor = ClipEditor.__new__(ClipEditor)
    assert editor.escape_path("/home/user/clips/Clip 01.mp4") == "/home/user/clips/Clip 01.mp4"


def test_fallback_bold_names_include_dejavu_and_liberation():
    assert "DejaVuSans-Bold.ttf" in FALLBACK_BOLD_FONT_NAMES
    assert "LiberationSans-Bold.ttf" in FALLBACK_BOLD_FONT_NAMES


# --- Launchers & dashboard template ---------------------------------------

def test_start_sh_exists_and_is_executable():
    repo = Path(__file__).resolve().parent.parent
    start_sh = repo / "start.sh"
    assert start_sh.exists()
    assert start_sh.read_text().startswith("#!/bin/bash")
    assert os.access(start_sh, os.X_OK)
    assert "main.py" in start_sh.read_text()


def test_main_py_accepts_host_flag():
    """main.py must parse --host/--port/--no-browser (phone access)."""
    repo = Path(__file__).resolve().parent.parent
    src = (repo / "main.py").read_text()
    assert "--host" in src and "--port" in src and "--no-browser" in src
    assert "get_lan_ip" in src


def test_config_has_dashboard_host_port():
    import config
    assert hasattr(config, "DASHBOARD_HOST")
    assert hasattr(config, "DASHBOARD_PORT")
    assert config.DASHBOARD_PORT == 8500


def test_template_is_mobile_friendly():
    repo = Path(__file__).resolve().parent.parent
    html = (repo / "templates" / "index.html").read_text()
    assert 'name="viewport"' in html and "width=device-width" in html
    assert "mobile-web-app-capable" in html
    assert "@media (max-width: 640px)" in html
    assert "min-height: 44px" in html  # roomy tap targets
    assert "w-[calc(100vw-2rem)]" in html  # notification panel fits a phone


def test_no_hardcoded_windows_font_paths_in_core_modules():
    """The main modules must not hardcode C:/Windows/Fonts any more."""
    repo = Path(__file__).resolve().parent.parent
    for name in ("clip_editor.py", "thumbnail.py", "kick_program.py",
                 "generate_app_icons.py", "test_card.py", "test_emoji.py",
                 "test_safe_zone.py"):
        src = (repo / name).read_text()
        assert "C:/Windows/Fonts" not in src, f"{name} still has a hardcoded font path"
        assert "C:\\\\Windows\\Fonts" not in src, f"{name} still has a hardcoded font path"


def test_no_hardcoded_repo_drive_paths():
    """No module may hardcode the old D:\\kick-ai-clip-engine location."""
    repo = Path(__file__).resolve().parent.parent
    for name in ("scratch_parse_tiktok.py", "sync_overlays.py",
                 "speaker_tracker.py", "parse_prestige_videos.py",
                 "create_desktop_shortcut.py", "desktop_app.py",
                 "templates/index.html"):
        src = (repo / name).read_text()
        assert "kick-ai-clip-engine" not in src, f"{name} still has a hardcoded repo path"
