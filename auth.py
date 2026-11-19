"""
Dashboard login for the FastAPI web dashboard (feature 15).

Single-password login that protects the dashboard before it ever leaves
localhost (e.g. on a VPS deployment — see docker-compose.yml).

Configuration (environment variables, no secrets in code):
    DASHBOARD_PASSWORD       Plain-text dashboard password. Compared in
                             constant time via hmac.compare_digest.
    DASHBOARD_PASSWORD_HASH  SHA-256 hex digest of the password (see
                             hash_password()). Preferred over the plain
                             password when both are set.
    DASHBOARD_SECRET_KEY     Key used to sign session cookies. If unset, a
                             random key is generated per process start (a
                             warning is logged) — sessions then do not
                             survive restarts, which is acceptable.

Behaviour:
    * If neither password variable is set, auth is disabled: a warning is
      logged once and require_login() allows access, preserving the current
      localhost behaviour.
    * Sessions are HMAC-SHA256 signed cookies (username + expiry), valid
      for 7 days (SESSION_MAX_AGE_SECONDS).

Wiring (done by the coordinator in web_dashboard.py — NOT here):
    from auth import router as auth_router, require_login
    app.include_router(auth_router)
    # then add `user: str = Depends(require_login)` to protected routes,
    # or: app = FastAPI(dependencies=[Depends(require_login)]) to protect
    # everything except the /login + /logout routes on this router.
"""

import base64
import binascii
import hashlib
import hmac
import logging
import os
import secrets
import time
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

logger = logging.getLogger(__name__)

SESSION_COOKIE_NAME = "ai_social_studio_session"
SESSION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60  # 7 days
DEFAULT_USERNAME = "admin"  # single-user dashboard; the token carries this name

_generated_secret_key = None
_warned_auth_disabled = False


# ---------------------------------------------------------------------------
# Password handling
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """SHA-256 hex digest of a password.

    Use this to generate a value for DASHBOARD_PASSWORD_HASH, e.g.:
        python -c "from auth import hash_password; print(hash_password('s3cret'))"
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _configured_password_hash():
    """The expected password hash, or None when no password is configured.

    DASHBOARD_PASSWORD_HASH wins when both variables are set.
    """
    pw_hash = os.environ.get("DASHBOARD_PASSWORD_HASH", "").strip()
    if pw_hash:
        return pw_hash.lower()
    password = os.environ.get("DASHBOARD_PASSWORD", "")
    if password:
        return hash_password(password)
    return None


def is_auth_enabled() -> bool:
    """True when a dashboard password (plain or hashed) is configured."""
    return _configured_password_hash() is not None


def verify_password(password: str) -> bool:
    """Constant-time password check. Returns False when auth is disabled."""
    expected = _configured_password_hash()
    if expected is None:
        return False
    return hmac.compare_digest(hash_password(password), expected)


# ---------------------------------------------------------------------------
# Session tokens (HMAC-SHA256 signed, base64url-encoded)
# ---------------------------------------------------------------------------

def get_secret_key() -> str:
    """Signing key for session cookies.

    Uses DASHBOARD_SECRET_KEY when set; otherwise generates a random key
    once per process start and logs a warning (sessions then do not survive
    restarts).
    """
    global _generated_secret_key
    key = os.environ.get("DASHBOARD_SECRET_KEY", "").strip()
    if key:
        return key
    if _generated_secret_key is None:
        _generated_secret_key = secrets.token_hex(32)
        logger.warning(
            "DASHBOARD_SECRET_KEY is not set — generated a random signing "
            "key for this process only; dashboard sessions will not survive "
            "a restart. Set DASHBOARD_SECRET_KEY in .env to fix this."
        )
    return _generated_secret_key


def create_session_token(username: str = DEFAULT_USERNAME,
                         max_age_seconds: int = SESSION_MAX_AGE_SECONDS) -> str:
    """Create a signed session token for `username`, valid for `max_age_seconds`."""
    expiry = int(time.time()) + int(max_age_seconds)
    payload = f"{username}:{expiry}"
    signature = hmac.new(
        get_secret_key().encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    raw = f"{payload}:{signature}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def verify_session_token(token):
    """Return the username if `token` is valid and unexpired, else None."""
    if not token:
        return None
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        username, expiry_str, signature = raw.rsplit(":", 2)
        expiry = int(expiry_str)
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return None
    if time.time() >= expiry:
        return None
    expected = hmac.new(
        get_secret_key().encode("utf-8"),
        f"{username}:{expiry}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    return username


# ---------------------------------------------------------------------------
# FastAPI dependency + login/logout routes
# ---------------------------------------------------------------------------

def require_login(request: Request) -> str:
    """FastAPI dependency: returns the logged-in username, or blocks access.

    * Auth disabled (no password configured): logs a warning once and
      allows access, keeping the current localhost behaviour.
    * Valid session cookie: returns the username from the token.
    * Otherwise: raises 401 for /api/* requests (so API clients get a clean
      error) and a 303 redirect to /login for browser page requests.
    """
    global _warned_auth_disabled
    if not is_auth_enabled():
        if not _warned_auth_disabled:
            _warned_auth_disabled = True
            logger.warning(
                "No DASHBOARD_PASSWORD / DASHBOARD_PASSWORD_HASH configured — "
                "the dashboard is accessible without login. Set one before "
                "exposing the dashboard beyond localhost."
            )
        return "local"
    user = verify_session_token(request.cookies.get(SESSION_COOKIE_NAME))
    if user:
        return user
    if request.url.path.startswith("/api"):
        raise HTTPException(status_code=401, detail="Login required")
    raise HTTPException(status_code=303, headers={"Location": "/login"})


LOGIN_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Login — AI Social Studio</title>
<style>
  body {{ margin: 0; min-height: 100vh; display: flex; align-items: center;
         justify-content: center; background: #0f0f0f; color: #f0f0f0;
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  .card {{ background: #1c1c1c; border: 1px solid #333; border-radius: 12px;
           padding: 32px; width: 320px; box-shadow: 0 8px 32px rgba(0,0,0,.5); }}
  h1 {{ margin: 0 0 4px; font-size: 20px; }}
  h1 span {{ color: #53fc18; }}
  p {{ margin: 0 0 20px; color: #999; font-size: 14px; }}
  label {{ display: block; font-size: 13px; color: #bbb; margin-bottom: 6px; }}
  input {{ width: 100%; box-sizing: border-box; padding: 10px 12px; font-size: 15px;
           border: 1px solid #444; border-radius: 8px; background: #111; color: #fff; }}
  input:focus {{ outline: none; border-color: #53fc18; }}
  button {{ width: 100%; margin-top: 16px; padding: 11px; font-size: 15px; font-weight: 600;
            border: none; border-radius: 8px; background: #53fc18; color: #000; cursor: pointer; }}
  button:hover {{ background: #6bff35; }}
  .error {{ background: #3a1414; border: 1px solid #7a2222; color: #ff9d9d;
            border-radius: 8px; padding: 10px 12px; font-size: 14px; margin-bottom: 16px; }}
</style>
</head>
<body>
  <form class="card" method="post" action="/login">
    <h1><span>KICK</span> AI Clip Engine</h1>
    <p>Sign in to the dashboard.</p>
    {error_html}
    <label for="password">Password</label>
    <input id="password" name="password" type="password" autocomplete="current-password"
           autofocus required>
    <button type="submit">Log in</button>
  </form>
</body>
</html>
"""


def _render_login_form(error=None) -> str:
    error_html = f'<div class="error">{error}</div>' if error else ""
    return LOGIN_PAGE_TEMPLATE.format(error_html=error_html)


router = APIRouter()


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Show the login form (or bounce to the dashboard when auth is off)."""
    if not is_auth_enabled():
        return RedirectResponse(url="/", status_code=303)
    return HTMLResponse(_render_login_form())


@router.post("/login")
async def login_submit(request: Request):
    """Check the password; on success set the session cookie and go home.

    The form body is parsed with stdlib urllib.parse so no extra
    dependency (python-multipart) is needed.
    """
    body = await request.body()
    form = parse_qs(body.decode("utf-8", "ignore"))
    password = (form.get("password") or [""])[0]
    if verify_password(password):
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            create_session_token(),
            max_age=SESSION_MAX_AGE_SECONDS,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response
    return HTMLResponse(_render_login_form("Incorrect password."), status_code=401)


@router.post("/logout")
async def logout():
    """Clear the session cookie and return to the login page."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return response
