"""Shared test setup for the Tier 2 (reach & money) feature tests.

- Puts the repo root on sys.path.
- Stubs ``dotenv`` (config.py imports it; it is not installed in this VM).
- Stubs ``fastapi`` / ``pydantic`` (dashboard_routers.py imports them; they
  are not installed in this VM). The stub APIRouter records routes and
  returns the handler unchanged so tests can call endpoints directly.
- Points the database at a throwaway sqlite file BEFORE importing database,
  so database.init_db() (which runs at import time) never touches the
  repo's real kick_engine.db.
- Provides a ``fresh_db`` fixture: a brand-new sqlite file per test, with
  all tables created, plus a ``seed_clip`` helper.
"""

import os
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- Stub dotenv (config.py does `from dotenv import load_dotenv`) --------
dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda *a, **k: None
sys.modules.setdefault("dotenv", dotenv_stub)

# --- Point config at a throwaway DB *before* importing database ------------
import config  # noqa: E402

_SESSION_TMP = Path(tempfile.mkdtemp(prefix="ai-social-studio-test-"))
config.DATABASE_FILE = _SESSION_TMP / "session.db"

# --- Stub fastapi + pydantic (not installed in this VM) --------------------
# Coordinator note: this stub must cover everything the tested modules
# import from fastapi — APIRouter, HTTPException (with headers), Request,
# the fastapi.responses classes, plus staticfiles/templating for the
# dashboard app itself. Keep it in sync with the real imports.
fastapi_stub = types.ModuleType("fastapi")


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str = "", headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


class Request:
    """Minimal stand-in for starlette Request (cookies/headers/path)."""

    def __init__(self, headers=None, cookies=None, url_path="/"):
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.query_params = {}
        self.url = types.SimpleNamespace(path=url_path)
        self.client = None


class APIRouter:
    """Minimal stand-in: route decorators register and return the handler."""

    def __init__(self):
        self.routes = []

    def _decorator(self, method, path):
        def deco(func):
            self.routes.append((method, path, func))
            return func
        return deco

    def get(self, path, **kwargs):
        return self._decorator("GET", path)

    def post(self, path, **kwargs):
        return self._decorator("POST", path)

    def patch(self, path, **kwargs):
        return self._decorator("PATCH", path)

    def delete(self, path, **kwargs):
        return self._decorator("DELETE", path)

    def put(self, path, **kwargs):
        return self._decorator("PUT", path)


class FastAPI:
    """Minimal stand-in so app modules can be imported in tests."""

    def __init__(self, *args, **kwargs):
        self.routes = []
        self.title = kwargs.get("title", "")

    def include_router(self, router, **kwargs):
        self.routes.extend(getattr(router, "routes", []))

    def mount(self, path, app, name=None):
        return None

    def middleware(self, middleware_type):
        def deco(func):
            return func
        return deco

    def _decorator(self, method, path):
        def deco(func):
            self.routes.append((method, path, func))
            return func
        return deco

    def get(self, path, **kwargs):
        return self._decorator("GET", path)

    def post(self, path, **kwargs):
        return self._decorator("POST", path)


def _make_response_class(name):
    class _Response:
        def __init__(self, content=None, status_code=200, headers=None, **kwargs):
            self.body = content
            self.content = content
            self.status_code = status_code
            self.headers = headers or {}
            for key, value in kwargs.items():
                setattr(self, key, value)

    _Response.__name__ = name
    return _Response


responses_stub = types.ModuleType("fastapi.responses")
responses_stub.Response = _make_response_class("Response")
responses_stub.HTMLResponse = _make_response_class("HTMLResponse")
responses_stub.JSONResponse = _make_response_class("JSONResponse")
responses_stub.RedirectResponse = _make_response_class("RedirectResponse")
responses_stub.FileResponse = _make_response_class("FileResponse")

staticfiles_stub = types.ModuleType("fastapi.staticfiles")
staticfiles_stub.StaticFiles = object

templating_stub = types.ModuleType("fastapi.templating")


class Jinja2Templates:
    def __init__(self, directory=None):
        self.directory = directory

    def TemplateResponse(self, request_or_name, name_or_context=None, context=None):
        # New Starlette signature: TemplateResponse(request, name, context).
        # Also tolerate the legacy TemplateResponse(name, context) form.
        if isinstance(request_or_name, str):
            return {"template": request_or_name,
                    "context": name_or_context or {}}
        ctx = dict(context or {})
        ctx.setdefault("request", request_or_name)
        return {"template": name_or_context, "context": ctx}


templating_stub.Jinja2Templates = Jinja2Templates

fastapi_stub.APIRouter = APIRouter
fastapi_stub.HTTPException = HTTPException
fastapi_stub.Request = Request
fastapi_stub.FastAPI = FastAPI
fastapi_stub.BackgroundTasks = object
sys.modules.setdefault("fastapi", fastapi_stub)
sys.modules.setdefault("fastapi.responses", responses_stub)
sys.modules.setdefault("fastapi.staticfiles", staticfiles_stub)
sys.modules.setdefault("fastapi.templating", templating_stub)

pydantic_stub = types.ModuleType("pydantic")


class BaseModel:
    """Minimal stand-in: class-level defaults + kwargs become attributes."""

    def __init__(self, **kwargs):
        for name, value in vars(type(self)).items():
            if not name.startswith("_") and not callable(value):
                setattr(self, name, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


pydantic_stub.BaseModel = BaseModel
sys.modules.setdefault("pydantic", pydantic_stub)

# --- Now it is safe to import the database (init_db runs on the tmp file) --
import database  # noqa: E402
import analytics  # noqa: E402
import earnings  # noqa: E402
import upload_scheduler  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    """Fresh throwaway sqlite DB with every table created.

    Returns a dict with a ``seed_clip`` helper for inserting clip rows.
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DATABASE_FILE", db_path)
    database.init_db()
    earnings.init_submissions_table()
    upload_scheduler.init_upload_queue_table()
    analytics.init_performance_table()

    def seed_clip(streamer="xqc", title="Test clip", ai_score=85,
                  category="Funny Moments", youtube_title=None,
                  youtube_description=None, instagram_caption=None,
                  tags="#kick #clip", caption="test caption",
                  edited_path=None, hype_score=70.0):
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO clips (streamer, title, caption, tags, hype_score,
                               ai_score, category, edited_path, status,
                               approval_status, youtube_title,
                               youtube_description, instagram_caption)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready_to_upload', 'approved',
                    ?, ?, ?)
        """, (streamer, title, caption, tags, hype_score, ai_score, category,
              edited_path, youtube_title, youtube_description,
              instagram_caption))
        clip_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return clip_id

    return {"seed_clip": seed_clip, "db_path": db_path}
