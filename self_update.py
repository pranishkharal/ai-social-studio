"""Safe self-update with backup + rollback (gta6-clipper feature 21).

Checks a manifest, verifies the sha256 of the download, backs up the
current install, replaces ONLY shipped files, and rolls back on any
failure. The user's own files are never touched: clips, renders,
buffer, database, .env, streamer profiles, OAuth tokens, logs and
the git directory are all protected.

Manifest: {"version": "1.4.0", "date": "...", "url": "<zip url or
local path>", "sha256": "<hex of the zip>"}. The zip holds the app
files (an optional single top folder is stripped). The manifest URL
defaults to the UPDATE_MANIFEST_URL env var; with none set, check()
reports "no_info" instead of guessing a URL — Sara updates via git
today, so this is convenience, not a gap.

Version: read from version.json in the app root when present
("0.0.0" otherwise — this repo does not ship one yet).
"""

import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

MANIFEST_ENV_VAR = "UPDATE_MANIFEST_URL"

# Directories never replaced and never backed up (big, or yours).
KEEP_DIRS = {"models", "clips", "edited", "buffer", "vod", "subtitles",
             "checkpoints", "backups", "test", "tests", "__pycache__",
             ".git", ".venv", "venv", "chrome_user_data", "assets",
             "static", "templates"}
# Files never replaced (your config, data and secrets).
KEEP_FILES = {".env", ".env.example", "kick_engine.db",
              "streamer_profiles.json", "client_secrets.json", "token.json",
              "clip_rules.json", "caption_guide.json",
              "prestige_analysis.json"}


def protected(relative_path):
    """True when a path inside the app root must never be replaced."""
    path = Path(relative_path)
    return (path.parts[0] in KEEP_DIRS or path.name in KEEP_FILES
            or ".bak" in path.name or path.name.endswith(".pyc"))


def current_version(root=HERE):
    """The installed version from version.json, or "0.0.0"."""
    try:
        return json.loads((Path(root) / "version.json").read_text(encoding="utf-8"))["version"]
    except Exception:
        return "0.0.0"


def _version_tuple(version):
    return tuple(int(part) for part in str(version).split(".") if part.isdigit())


def _fetch(url, timeout=20):
    if "://" not in url:
        return Path(url).read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "ai-social-studio-update"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def check(url=None, root=HERE):
    """-> {"current", "latest", "update", "status", "manifest"?}.

    status is "ok", "no_info" (no manifest URL configured, or a 404)
    or "error". Never raises for a bad manifest or a network failure.
    """
    current = current_version(root)
    url = url or os.environ.get(MANIFEST_ENV_VAR)
    if not url:
        return {"current": current, "update": False, "status": "no_info",
                "detail": "No update manifest URL configured "
                          f"(set {MANIFEST_ENV_VAR})."}
    try:
        manifest = json.loads(_fetch(url))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"current": current, "update": False, "status": "no_info",
                    "detail": "No update info yet"}
        return {"current": current, "update": False, "status": "error",
                "detail": f"HTTP {e.code}"}
    except Exception as e:
        return {"current": current, "update": False, "status": "error",
                "detail": str(e)[:200]}
    if not isinstance(manifest, dict) or "version" not in manifest:
        return {"current": current, "update": False, "status": "error",
                "detail": "Bad manifest (no version)"}
    return {"current": current, "latest": manifest["version"],
            "date": manifest.get("date", ""),
            "update": _version_tuple(manifest["version"]) > _version_tuple(current),
            "status": "ok", "manifest": manifest}


def apply(workspace, url=None, root=HERE, dry=False):
    """Download, verify sha256, back up to
    <workspace>/backups/<current-version>/, replace shipped files;
    roll back on any failure. ``dry=True`` only lists what would be
    replaced. Returns a result dict; nothing is half-applied."""
    root = Path(root)
    checked = check(url, root)
    if checked["status"] != "ok":
        return {"ok": False, **{k: checked[k] for k in ("status", "detail") if k in checked}}
    if not checked["update"]:
        return {"ok": True, "changed": False, "current": checked["current"],
                "detail": "Already up to date"}
    manifest = checked["manifest"]
    try:
        data = _fetch(manifest["url"], timeout=120)
    except Exception as e:
        return {"ok": False, "detail": "Download failed, nothing changed: " + str(e)[:200]}
    if hashlib.sha256(data).hexdigest().lower() != str(manifest.get("sha256", "")).lower():
        return {"ok": False, "detail": "sha256 mismatch: nothing changed"}
    tmp = Path(tempfile.mkdtemp(prefix="ai-social-studio-upd-"))
    zip_path = tmp / "update.zip"
    zip_path.write_bytes(data)
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        shutil.rmtree(tmp, True)
        return {"ok": False, "detail": "Damaged download, nothing changed"}
    with archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        for name in names:
            if (name.startswith(("/", "\\")) or ".." in Path(name).parts
                    or ":" in name):
                shutil.rmtree(tmp, True)
                return {"ok": False, "detail": "Unsafe path in zip: " + name}
        tops = {Path(n).parts[0] for n in names}
        strip_top = len(tops) == 1 and all(len(Path(n).parts) > 1 for n in names)
        new = {(str(Path(*Path(n).parts[1:])) if strip_top else n): n for n in names}
        # A kept file is still installed when it is missing.
        new = {rel: name for rel, name in new.items()
               if not protected(rel) or not (root / rel).exists()}
        # version.json last: a crash never shows the new number on old files.
        new = dict(sorted(new.items(), key=lambda kv: Path(kv[0]).name == "version.json"))
        if dry:
            shutil.rmtree(tmp, True)
            return {"ok": True, "dry": True, "current": checked["current"],
                    "latest": checked["latest"], "would_replace": sorted(new)}
        backup = Path(workspace) / "backups" / checked["current"]
        replaced = []
        try:
            if backup.exists():
                shutil.rmtree(backup)
            for path in root.rglob("*"):
                relative = path.relative_to(root)
                if path.is_file() and not protected(relative):
                    (backup / relative).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup / relative)
            for relative, name in new.items():
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                staged = dest.with_name(dest.name + ".new")
                staged.write_bytes(archive.read(name))
                replaced.append((dest, dest.exists()))
                os.replace(staged, dest)
        except Exception as e:
            for dest, existed in replaced:
                saved = backup / dest.relative_to(root)
                try:
                    if existed and saved.exists():
                        shutil.copy2(saved, dest)
                    elif not existed and dest.exists():
                        dest.unlink()
                except OSError:
                    pass  # keep rolling back the rest
            for dest, _ in replaced:
                dest.with_name(dest.name + ".new").unlink(missing_ok=True)
            shutil.rmtree(tmp, True)
            return {"ok": False, "detail": "Failed, rolled back: " + str(e)[:200]}
    shutil.rmtree(tmp, True)
    return {"ok": True, "changed": True, "current": checked["current"],
            "latest": checked["latest"], "backup": str(backup),
            "replaced": len(new), "restart_needed": True}
