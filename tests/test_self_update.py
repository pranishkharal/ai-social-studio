"""Tests for self_update.py (gta6-clipper feature 21).

The manifest and the update zip are local fakes — no network.
"""

import hashlib
import json
import zipfile
from pathlib import Path

import self_update


def _make_zip(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _make_manifest(path, zip_path, version="1.1.0"):
    data = zip_path.read_bytes()
    manifest = {
        "version": version,
        "date": "2026-11-21",
        "url": str(zip_path),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    path.write_text(json.dumps(manifest))
    return manifest


def _make_root(tmp_path, version="1.0.0"):
    root = tmp_path / "app"
    root.mkdir()
    (root / "version.json").write_text(json.dumps({"version": version}))
    (root / "app.py").write_text("old app\n")
    (root / ".env").write_text("SECRET=keep-me\n")
    (root / "clips").mkdir()
    (root / "clips" / "clip1.mp4").write_bytes(b"clip")
    return root


def test_check_no_manifest_url(tmp_path, monkeypatch):
    monkeypatch.delenv("UPDATE_MANIFEST_URL", raising=False)
    root = _make_root(tmp_path)
    result = self_update.check(root=root)
    assert result["status"] == "no_info"
    assert result["update"] is False


def test_check_update_available(tmp_path):
    root = _make_root(tmp_path)
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path, {"app.py": "new app\n"})
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, zip_path, version="1.1.0")
    result = self_update.check(url=str(manifest), root=root)
    assert result["status"] == "ok"
    assert result["current"] == "1.0.0"
    assert result["latest"] == "1.1.0"
    assert result["update"] is True


def test_check_already_up_to_date(tmp_path):
    root = _make_root(tmp_path, version="1.1.0")
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path, {"app.py": "new app\n"})
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, zip_path, version="1.1.0")
    result = self_update.check(url=str(manifest), root=root)
    assert result["update"] is False


def test_check_bad_manifest(tmp_path):
    root = _make_root(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"no_version": True}))
    result = self_update.check(url=str(manifest), root=root)
    assert result["status"] == "error"


def test_apply_replaces_and_backs_up(tmp_path):
    root = _make_root(tmp_path)
    workspace = tmp_path / "workspace"
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path, {
        "app.py": "new app\n",
        "new_module.py": "brand new\n",
        "version.json": json.dumps({"version": "1.1.0"}),
        ".env": "SECRET=evil-overwrite\n",       # protected: must NOT replace
        "clips/clip1.mp4": b"evil-overwrite",    # protected dir: must NOT replace
    })
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, zip_path, version="1.1.0")
    result = self_update.apply(workspace, url=str(manifest), root=root)
    assert result["ok"] is True
    assert result["changed"] is True
    assert result["latest"] == "1.1.0"
    assert (root / "app.py").read_text() == "new app\n"
    assert (root / "new_module.py").read_text() == "brand new\n"
    assert self_update.current_version(root) == "1.1.0"
    # Protected files untouched.
    assert (root / ".env").read_text() == "SECRET=keep-me\n"
    assert (root / "clips" / "clip1.mp4").read_bytes() == b"clip"
    # Backup holds the old version.
    backup = Path(result["backup"])
    assert (backup / "app.py").read_text() == "old app\n"
    assert (backup / "version.json").read_text() == json.dumps({"version": "1.0.0"})


def test_apply_dry_run_changes_nothing(tmp_path):
    root = _make_root(tmp_path)
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path, {"app.py": "new app\n"})
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, zip_path, version="1.1.0")
    result = self_update.apply(tmp_path / "workspace", url=str(manifest),
                               root=root, dry=True)
    assert result["ok"] is True and result["dry"] is True
    assert result["would_replace"] == ["app.py"]
    assert (root / "app.py").read_text() == "old app\n"


def test_apply_rejects_sha_mismatch(tmp_path):
    root = _make_root(tmp_path)
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path, {"app.py": "new app\n"})
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, zip_path, version="1.1.0")
    data = json.loads(manifest.read_text())
    data["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    result = self_update.apply(tmp_path / "workspace", url=str(manifest), root=root)
    assert result["ok"] is False
    assert "sha256 mismatch" in result["detail"]
    assert (root / "app.py").read_text() == "old app\n"


def test_apply_rejects_unsafe_zip_paths(tmp_path):
    root = _make_root(tmp_path)
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path, {"../escape.py": "evil\n"})
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, zip_path, version="1.1.0")
    result = self_update.apply(tmp_path / "workspace", url=str(manifest), root=root)
    assert result["ok"] is False
    assert "Unsafe path" in result["detail"]
    assert not (tmp_path / "escape.py").exists()


def test_apply_strips_single_top_folder(tmp_path):
    root = _make_root(tmp_path)
    zip_path = tmp_path / "update.zip"
    _make_zip(zip_path, {"ai-social-studio-1.1.0/app.py": "new app\n"})
    manifest = tmp_path / "manifest.json"
    _make_manifest(manifest, zip_path, version="1.1.0")
    result = self_update.apply(tmp_path / "workspace", url=str(manifest), root=root)
    assert result["ok"] is True
    assert (root / "app.py").read_text() == "new app\n"


def test_protected():
    assert self_update.protected("clips/x.mp4")
    assert self_update.protected(".env")
    assert self_update.protected("kick_engine.db")
    assert self_update.protected("streamer_profiles.json")
    assert self_update.protected("something.bak")
    assert self_update.protected("module.pyc")
    assert not self_update.protected("app.py")
    assert not self_update.protected("version.json")
