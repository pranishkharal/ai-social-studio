"""Tests for auth.py (feature 15) — password verify + HMAC session tokens.

Pure-function tests; no TestClient needed. FastAPI is stubbed in
sys.modules when it is not installed, so these tests run in a bare
environment (only the real FastAPI is used when available).
"""

import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- FastAPI stub (only what auth.py imports) -------------------------------
# FastAPI is not installed in this VM. Another worker's tests/conftest.py
# may already have stubbed `fastapi` (with only APIRouter/HTTPException),
# so instead of replacing the module we patch in whatever is missing:
# Request, fastapi.responses, and an HTTPException that accepts `headers`
# (needed for the /login redirect). When the real FastAPI is installed
# (e.g. CI) everything already exists and nothing is patched.
import inspect  # noqa: E402


class _HTTPException(Exception):
    def __init__(self, status_code, detail=None, headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}


class _Request:
    pass


class _APIRouter:
    def __init__(self):
        self.routes = []

    def get(self, path, **kwargs):
        def deco(fn):
            self.routes.append(("GET", path, fn))
            return fn
        return deco

    def post(self, path, **kwargs):
        def deco(fn):
            self.routes.append(("POST", path, fn))
            return fn
        return deco


class _HTMLResponse:
    def __init__(self, content="", status_code=200, **kwargs):
        self.body = content.encode() if isinstance(content, str) else content
        self.status_code = status_code
        self.headers = {}
        self.cookies = {}

    def set_cookie(self, key, value, **kwargs):
        self.cookies[key] = value

    def delete_cookie(self, key, **kwargs):
        self.cookies.pop(key, None)


class _RedirectResponse(_HTMLResponse):
    def __init__(self, url, status_code=307, **kwargs):
        super().__init__(content="", status_code=status_code)
        self.headers["location"] = url


def _ensure_fastapi_stub():
    fastapi_mod = sys.modules.get("fastapi")
    if fastapi_mod is None:
        fastapi_mod = types.ModuleType("fastapi")
        sys.modules["fastapi"] = fastapi_mod
    if not hasattr(fastapi_mod, "APIRouter"):
        fastapi_mod.APIRouter = _APIRouter
    if not hasattr(fastapi_mod, "Request"):
        fastapi_mod.Request = _Request
    existing_exc = getattr(fastapi_mod, "HTTPException", None)
    needs_exc = existing_exc is None
    if not needs_exc:
        try:
            params = inspect.signature(existing_exc.__init__).parameters
            needs_exc = "headers" not in params
        except (TypeError, ValueError):
            needs_exc = True
    if needs_exc:
        fastapi_mod.HTTPException = _HTTPException
    responses_mod = sys.modules.get("fastapi.responses")
    if responses_mod is None:
        responses_mod = types.ModuleType("fastapi.responses")
        sys.modules["fastapi.responses"] = responses_mod
        fastapi_mod.responses = responses_mod
    if not hasattr(responses_mod, "HTMLResponse"):
        responses_mod.HTMLResponse = _HTMLResponse
    if not hasattr(responses_mod, "RedirectResponse"):
        responses_mod.RedirectResponse = _RedirectResponse


_ensure_fastapi_stub()

import auth  # noqa: E402
from auth import (  # noqa: E402
    SESSION_COOKIE_NAME,
    create_session_token,
    hash_password,
    is_auth_enabled,
    require_login,
    verify_password,
    verify_session_token,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with no auth env vars set."""
    for var in ("DASHBOARD_PASSWORD", "DASHBOARD_PASSWORD_HASH", "DASHBOARD_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


def _request(path="/", cookies=None):
    return SimpleNamespace(cookies=cookies or {}, url=SimpleNamespace(path=path))


# --- hash_password / verify_password ---------------------------------------

def test_hash_password_matches_sha256():
    assert hash_password("s3cret") == hashlib.sha256(b"s3cret").hexdigest()


def test_verify_password_with_plain_env(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    assert is_auth_enabled() is True
    assert verify_password("hunter2") is True
    assert verify_password("wrong") is False


def test_verify_password_with_hash_env(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", hash_password("topsecret"))
    assert is_auth_enabled() is True
    assert verify_password("topsecret") is True
    assert verify_password("TopSecret") is False


def test_hash_env_is_preferred_over_plain(monkeypatch):
    # Both set: the hash wins — the plain password must NOT authenticate.
    monkeypatch.setenv("DASHBOARD_PASSWORD", "plainpw")
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", hash_password("hashpw"))
    assert verify_password("hashpw") is True
    assert verify_password("plainpw") is False


def test_verify_password_unconfigured_returns_false():
    assert is_auth_enabled() is False
    assert verify_password("anything") is False


# --- session tokens ---------------------------------------------------------

def test_token_roundtrip(monkeypatch):
    monkeypatch.setenv("DASHBOARD_SECRET_KEY", "test-secret-key")
    token = create_session_token("admin")
    assert verify_session_token(token) == "admin"


def test_token_tampered_rejected(monkeypatch):
    monkeypatch.setenv("DASHBOARD_SECRET_KEY", "test-secret-key")
    token = create_session_token("admin")
    tampered = ("A" if token[0] != "A" else "B") + token[1:]
    assert verify_session_token(tampered) is None


def test_token_expired_rejected(monkeypatch):
    monkeypatch.setenv("DASHBOARD_SECRET_KEY", "test-secret-key")
    token = create_session_token("admin", max_age_seconds=-10)
    assert verify_session_token(token) is None


def test_token_wrong_secret_rejected(monkeypatch):
    monkeypatch.setenv("DASHBOARD_SECRET_KEY", "secret-one")
    token = create_session_token("admin")
    monkeypatch.setenv("DASHBOARD_SECRET_KEY", "secret-two")
    assert verify_session_token(token) is None


def test_token_malformed_rejected():
    assert verify_session_token("not-a-token!!!") is None
    assert verify_session_token("") is None
    assert verify_session_token(None) is None


# --- require_login dependency ----------------------------------------------

def test_require_login_disabled_allows_access():
    assert require_login(_request("/")) == "local"
    assert require_login(_request("/api/clips")) == "local"


def test_require_login_valid_cookie(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    monkeypatch.setenv("DASHBOARD_SECRET_KEY", "test-secret-key")
    token = create_session_token("admin")
    req = _request("/api/clips", cookies={SESSION_COOKIE_NAME: token})
    assert require_login(req) == "admin"


def test_require_login_missing_cookie_api_returns_401(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    with pytest.raises(Exception) as excinfo:
        require_login(_request("/api/clips"))
    assert excinfo.value.status_code == 401


def test_require_login_missing_cookie_page_redirects(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    with pytest.raises(Exception) as excinfo:
        require_login(_request("/"))
    assert excinfo.value.status_code == 303
    assert excinfo.value.headers["Location"] == "/login"


def test_require_login_bad_cookie_rejected(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "hunter2")
    req = _request("/api/clips", cookies={SESSION_COOKIE_NAME: "bogus"})
    with pytest.raises(Exception) as excinfo:
        require_login(req)
    assert excinfo.value.status_code == 401
