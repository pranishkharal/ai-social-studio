"""Tests for the bulk Gemini API key manager (gemini_keys.py), the
dashboard routes in web_dashboard.py, and the ai_story_engine failover.

All keys here are FAKE ("AIza" + repeated letters). No real key is
ever used. Several tests assert the full fake key NEVER appears in
status output, API responses, or logs -- that is the point of masking.
"""

import asyncio
import json
import sys
import time
import types

import pytest

import gemini_keys
from gemini_keys import (
    GeminiKeyManager,
    GeminiUnavailableError,
    NoHealthyKeyError,
    RotatingGeminiClient,
    classify_error,
    mask_key,
    parse_keys,
)

# Fake keys only -- 39 chars, valid format ("AIza" + 35 chars).
KEY1 = "AIza" + "a" * 35
KEY2 = "AIza" + "b" * 35
KEY3 = "AIza" + "c" * 35
KEY4 = "AIza" + "d" * 35


def make_manager(tmp_path, env_keys="", factory=None, **kwargs):
    """A manager with a throwaway store file (never the real repo one)."""
    if factory is not None:
        kwargs.setdefault("client_factory", factory)
    return GeminiKeyManager(
        store_path=str(tmp_path / "gemini_keys.json"),
        env_keys=env_keys,
        **kwargs)


# ---------------------------------------------------------------------------
# Fake genai clients (the real SDK is not installed in this VM)
# ---------------------------------------------------------------------------

class FakeApiError(Exception):
    """An API failure carrying a numeric .code, like the SDK's errors."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FakeResponse:
    def __init__(self, text):
        self.text = text


VALID_METADATA_JSON = json.dumps({
    "category": "Funny Moments",
    "ai_score": 90,
    "selection_reason": "Fake test moment.",
    "story_summary": "Test summary.",
    "viral_title": "XQC DID SOMETHING FUNNY",
    "hook_banner": "Xqc was SHOCKED",
    "screen_caption": "Xqc was SHOCKED",
    "quote_line": "what a win",
    "tiktok_caption": "funny caption",
    "youtube_title": "Funny #Shorts",
    "youtube_description": "desc",
    "instagram_caption": "insta",
    "blue_comment": "comment",
    "hashtags": "#xqc",
})


class FakeModels:
    """Behaviour is driven by a shared state dict: outcome per key is
    either "ok", a response string, or an Exception to raise."""

    def __init__(self, state, key):
        self._state = state
        self._key = key

    def _outcome(self, kind):
        return self._state[kind].get(self._key, "ok")

    def generate_content(self, model, contents, **kwargs):
        outcome = self._outcome("generate")
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "ok":
            return FakeResponse(VALID_METADATA_JSON)
        return FakeResponse(outcome)  # custom response string

    def list(self, **kwargs):
        outcome = self._outcome("list")
        if isinstance(outcome, Exception):
            raise outcome
        return iter([{"name": "models/gemini-2.5-flash"}])


class FakeClient:
    def __init__(self, state, key):
        self.models = FakeModels(state, key)


def make_factory(state):
    def factory(key, timeout=None):
        return FakeClient(state, key)
    return factory


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_newlines():
    parsed = parse_keys(f"{KEY1}\n{KEY2}\n{KEY3}")
    assert parsed["valid"] == [KEY1, KEY2, KEY3]
    assert parsed["invalid"] == 0


def test_parse_commas_and_semicolons():
    parsed = parse_keys(f"{KEY1},{KEY2};{KEY3}")
    assert parsed["valid"] == [KEY1, KEY2, KEY3]
    assert parsed["invalid"] == 0


def test_parse_mixed_separators_whitespace_and_empties():
    text = f"  {KEY1} ,\n\n  {KEY2} ;\r\n{KEY3}  \n"
    parsed = parse_keys(text)
    assert parsed["valid"] == [KEY1, KEY2, KEY3]
    assert parsed["invalid"] == 0


def test_parse_dedupes_exact_matches():
    parsed = parse_keys(f"{KEY1}\n{KEY1},{KEY2}\n{KEY2}")
    assert parsed["valid"] == [KEY1, KEY2]


def test_parse_rejects_wrong_prefix_and_too_short():
    parsed = parse_keys("sk-1234567890abcdefghij\nAIzaShort\n" + KEY1)
    assert parsed["valid"] == [KEY1]
    assert parsed["invalid"] == 2
    assert parsed["invalid_reasons"] == {"wrong_prefix": 1, "too_short": 1}
    # Rejected values are never echoed back.
    assert "sk-1234567890abcdefghij" not in json.dumps(parsed)
    assert "AIzaShort" not in json.dumps(parsed)


def test_parse_strips_surrounding_quotes():
    parsed = parse_keys(f'"{KEY1}"\n\'{KEY2}\'')
    assert parsed["valid"] == [KEY1, KEY2]


def test_parse_empty_and_none():
    assert parse_keys("") == {"valid": [], "invalid": 0,
                              "invalid_reasons": {"wrong_prefix": 0,
                                                  "too_short": 0}}
    assert parse_keys(None)["valid"] == []


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def test_mask_key_shows_first_and_last_4_only():
    assert mask_key(KEY1) == "AIza..." + "a" * 4
    assert KEY1 not in mask_key(KEY1)
    assert len(mask_key(KEY1)) == 11


def test_mask_key_short_or_empty():
    assert mask_key("") == "****"
    assert mask_key("AIza") == "****"
    assert mask_key(None) == "****"


# ---------------------------------------------------------------------------
# Sources: empty, env fallback, store wins
# ---------------------------------------------------------------------------

def test_no_keys_anywhere(tmp_path):
    mgr = make_manager(tmp_path, env_keys="")
    assert not mgr.has_keys()
    assert mgr.source == "none"
    assert mgr.key_count() == 0
    status = mgr.get_status()
    assert status["counts"] == {"total": 0, "healthy": 0,
                                "cooling_down": 0, "invalid": 0}
    assert status["keys"] == []
    with pytest.raises(NoHealthyKeyError):
        mgr.get_client()


def test_env_fallback_keys(tmp_path):
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}")
    assert mgr.source == "env"
    assert mgr.key_count() == 2
    status = mgr.get_status()
    assert [k["id"] for k in status["keys"]] == ["env-1", "env-2"]
    assert all(k["source"] == "env" for k in status["keys"])
    assert all(k["deletable"] is False for k in status["keys"])
    assert mgr.delete_key("env-1") is False  # env keys live in .env


def test_env_fallback_splits_on_all_separators(tmp_path):
    mgr = make_manager(tmp_path, env_keys=f"{KEY1};{KEY2}\n{KEY3}")
    assert mgr.key_count() == 3


def test_store_wins_over_env_and_falls_back_when_emptied(tmp_path):
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}")
    assert mgr.source == "env"
    mgr.add_keys(KEY3)
    assert mgr.source == "store"
    assert mgr.key_count() == 1
    assert mgr.get_status()["env_keys_ignored"] == 2
    key_id = mgr.get_status()["keys"][0]["id"]
    assert mgr.delete_key(key_id) is True
    assert mgr.source == "env"
    assert mgr.key_count() == 2


# ---------------------------------------------------------------------------
# Adding keys + store round-trip
# ---------------------------------------------------------------------------

def test_add_keys_counts(tmp_path):
    mgr = make_manager(tmp_path)
    result = mgr.add_keys(f"{KEY1}\n{KEY2}\nbad-key\n{KEY1}")
    assert result == {"added": 2, "duplicates": 0, "invalid": 1,
                      "total": 2}
    # Adding the same keys again: all duplicates, nothing new.
    result = mgr.add_keys(f"{KEY1},{KEY2},{KEY3}")
    assert result == {"added": 1, "duplicates": 2, "invalid": 0,
                      "total": 3}


def test_store_round_trip_and_health_persists(tmp_path):
    mgr = make_manager(tmp_path)
    mgr.add_keys(f"{KEY1}\n{KEY2}")
    key_id = mgr.get_status()["keys"][0]["id"]
    mgr.report_failure(key_id, FakeApiError(401, "API key not valid"))
    # A fresh manager on the same file sees the same keys + status.
    mgr2 = make_manager(tmp_path)
    assert mgr2.key_count() == 2
    status = {k["id"]: k for k in mgr2.get_status()["keys"]}
    assert status[key_id]["status"] == "invalid"
    # The file itself is valid JSON with the expected shape.
    data = json.loads((tmp_path / "gemini_keys.json").read_text())
    assert data["version"] == 1
    assert len(data["keys"]) == 2


def test_corrupt_store_file_starts_empty(tmp_path):
    store = tmp_path / "gemini_keys.json"
    store.write_text("{not json", encoding="utf-8")
    mgr = make_manager(tmp_path)
    assert mgr.key_count() == 0
    assert mgr.source == "none"


def test_delete_key(tmp_path):
    mgr = make_manager(tmp_path)
    mgr.add_keys(f"{KEY1}\n{KEY2}")
    key_id = mgr.get_status()["keys"][0]["id"]
    assert mgr.delete_key(key_id) is True
    assert mgr.key_count() == 1
    assert mgr.delete_key(key_id) is False  # already gone
    assert mgr.delete_key("no-such-id") is False


def test_reenable_invalid_key(tmp_path):
    mgr = make_manager(tmp_path)
    mgr.add_keys(KEY1)
    key_id = mgr.get_status()["keys"][0]["id"]
    mgr.report_failure(key_id, FakeApiError(403, "Permission denied"))
    assert mgr.get_status()["keys"][0]["status"] == "invalid"
    assert mgr.reenable_key(key_id) is True
    assert mgr.get_status()["keys"][0]["status"] == "healthy"


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def test_round_robin_cycles_across_keys(tmp_path):
    state = {"generate": {}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2},{KEY3}",
                       factory=make_factory(state))
    ids = [mgr.get_client()[1] for _ in range(6)]
    assert ids == ["env-1", "env-2", "env-3", "env-1", "env-2", "env-3"]


def test_clients_are_cached_per_key(tmp_path):
    state = {"generate": {}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=KEY1, factory=make_factory(state))
    client1, _ = mgr.get_client()
    client2, _ = mgr.get_client()
    assert client1 is client2


def test_cooling_down_key_is_skipped_then_returns(tmp_path):
    state = {"generate": {}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}",
                       factory=make_factory(state))
    mgr.report_failure("env-1", FakeApiError(429, "Quota exceeded"))
    ids = [mgr.get_client()[1] for _ in range(3)]
    assert ids == ["env-2", "env-2", "env-2"]
    # Expire the cooldown: the key is healthy again (manipulating the
    # stored timestamp directly -- deterministic, no sleeps).
    mgr._entries[0]["cooldown_until"] = time.time() - 1
    ids = [mgr.get_client()[1] for _ in range(2)]
    assert "env-1" in ids


def test_invalid_key_is_skipped(tmp_path):
    state = {"generate": {}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}",
                       factory=make_factory(state))
    mgr.report_failure("env-1", FakeApiError(401, "API key not valid"))
    ids = [mgr.get_client()[1] for _ in range(3)]
    assert ids == ["env-2", "env-2", "env-2"]


def test_all_keys_down_raises(tmp_path):
    state = {"generate": {}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}",
                       factory=make_factory(state))
    mgr.report_failure("env-1", FakeApiError(401, "bad key"))
    mgr.report_failure("env-2", FakeApiError(429, "quota"))
    with pytest.raises(NoHealthyKeyError):
        mgr.get_client()


def test_missing_sdk_raises_unavailable(tmp_path):
    def boom_factory(key, timeout=None):
        raise ImportError("No module named 'google'")

    mgr = make_manager(tmp_path, env_keys=KEY1, client_factory=boom_factory)
    with pytest.raises(GeminiUnavailableError):
        mgr.get_client()


# ---------------------------------------------------------------------------
# Failure classification + health reporting
# ---------------------------------------------------------------------------

def test_classify_error_by_code():
    assert classify_error(FakeApiError(429, "x")) == "rate_limited"
    assert classify_error(FakeApiError(401, "x")) == "invalid"
    assert classify_error(FakeApiError(403, "x")) == "invalid"
    assert classify_error(FakeApiError(500, "x")) == "other"


def test_classify_error_by_message():
    assert classify_error(Exception("quota exceeded")) == "rate_limited"
    assert classify_error(Exception("Rate limit hit")) == "rate_limited"
    assert classify_error(Exception("RESOURCE_EXHAUSTED")) == "rate_limited"
    assert classify_error(Exception("API key not valid")) == "invalid"
    assert classify_error(Exception("unauthenticated")) == "invalid"
    assert classify_error(Exception("connection reset")) == "other"


def test_report_failure_rate_limit_cools_down(tmp_path):
    mgr = make_manager(tmp_path, env_keys=KEY1)
    kind = mgr.report_failure("env-1", FakeApiError(429, "quota"))
    assert kind == "rate_limited"
    entry = mgr.get_status()["keys"][0]
    assert entry["status"] == "cooling_down"
    assert entry["failure_count"] == 1
    assert entry["last_error"] == "quota"


def test_report_failure_auth_marks_invalid(tmp_path):
    mgr = make_manager(tmp_path, env_keys=KEY1)
    kind = mgr.report_failure("env-1", FakeApiError(403, "denied"))
    assert kind == "invalid"
    assert mgr.get_status()["keys"][0]["status"] == "invalid"


def test_report_failure_other_counts_only(tmp_path):
    mgr = make_manager(tmp_path, env_keys=KEY1)
    kind = mgr.report_failure("env-1", Exception("boom"))
    assert kind == "other"
    entry = mgr.get_status()["keys"][0]
    assert entry["status"] == "healthy"
    assert entry["failure_count"] == 1
    assert entry["last_error"] == "boom"


def test_report_failure_scrubs_key_from_error(tmp_path):
    # If an SDK error ever echoed the key, the stored error must not
    # carry it -- last_error is shown in the dashboard.
    mgr = make_manager(tmp_path, env_keys=KEY1)
    mgr.report_failure("env-1", Exception(f"rejected key {KEY1}"))
    entry = mgr.get_status()["keys"][0]
    assert KEY1 not in entry["last_error"]
    assert "[key]" in entry["last_error"]


def test_report_success_clears_streak(tmp_path):
    mgr = make_manager(tmp_path, env_keys=KEY1)
    mgr.report_failure("env-1", Exception("boom"))
    mgr.report_success("env-1")
    entry = mgr.get_status()["keys"][0]
    assert entry["failure_count"] == 0
    assert entry["last_error"] is None
    assert entry["last_used_at"] is not None


def test_get_status_never_contains_full_keys(tmp_path):
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}")
    mgr.report_failure("env-1", Exception("boom"))
    blob = json.dumps(mgr.get_status())
    assert KEY1 not in blob
    assert KEY2 not in blob
    assert mask_key(KEY1) in blob
    assert mask_key(KEY2) in blob


def test_rotation_is_thread_safe(tmp_path):
    import threading

    state = {"generate": {}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2},{KEY3}",
                       factory=make_factory(state))
    seen = []

    def worker():
        for _ in range(20):
            _, key_id = mgr.get_client()
            seen.append(key_id)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 100
    # Round-robin across 3 keys over 100 picks: each key used 33-34 times.
    assert all(30 <= seen.count(f"env-{i}") <= 36 for i in (1, 2, 3))


# ---------------------------------------------------------------------------
# Rotating client (captions.py compatibility)
# ---------------------------------------------------------------------------

def test_rotating_client_rotates_and_reports(tmp_path):
    state = {"generate": {KEY1: FakeApiError(429, "quota")}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}",
                       factory=make_factory(state))
    client = RotatingGeminiClient(mgr)
    with pytest.raises(FakeApiError):
        client.models.generate_content(model="m", contents="hi")
    # Key 1 is now cooling down; the next call uses key 2 and works.
    resp = client.models.generate_content(model="m", contents="hi")
    assert json.loads(resp.text)["category"] == "Funny Moments"
    status = {k["id"]: k for k in mgr.get_status()["keys"]}
    assert status["env-1"]["status"] == "cooling_down"
    assert status["env-2"]["last_used_at"] is not None


def test_rotating_client_no_keys_raises(tmp_path):
    mgr = make_manager(tmp_path, env_keys="")
    client = RotatingGeminiClient(mgr)
    with pytest.raises(NoHealthyKeyError):
        client.models.generate_content(model="m", contents="hi")


# ---------------------------------------------------------------------------
# Key testing
# ---------------------------------------------------------------------------

def test_test_all_keys_format_only_without_sdk(tmp_path):
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}")
    result = mgr.test_all_keys()
    # google-genai is not installed in this VM.
    assert result["mode"] == "format_only"
    assert [r["result"] for r in result["results"]] == ["ok", "ok"]
    assert all(KEY1 not in json.dumps(r) and KEY2 not in json.dumps(r)
               for r in result["results"])


def test_test_all_keys_live_mode(monkeypatch, tmp_path):
    # Pretend the SDK is importable (the probe only checks that) --
    # the actual calls go through the injected fake factory.
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("google.genai")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    state = {"generate": {}, "list": {
        KEY1: FakeApiError(401, "API key not valid"),
        KEY2: FakeApiError(429, "quota exceeded"),
    }}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2},{KEY3}",
                       factory=make_factory(state))
    result = mgr.test_all_keys()
    assert result["mode"] == "live"
    by_id = {r["id"]: r for r in result["results"]}
    assert by_id["env-1"]["result"] == "invalid"
    assert by_id["env-2"]["result"] == "rate_limited"
    assert by_id["env-3"]["result"] == "ok"
    # Testing updated health: key 1 invalid, key 2 cooling down.
    status = {k["id"]: k for k in mgr.get_status()["keys"]}
    assert status["env-1"]["status"] == "invalid"
    assert status["env-2"]["status"] == "cooling_down"


# ---------------------------------------------------------------------------
# ai_story_engine failover integration
# ---------------------------------------------------------------------------

def test_engine_fails_over_to_next_key(monkeypatch, tmp_path):
    import ai_story_engine

    state = {"generate": {KEY1: FakeApiError(429, "quota")}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}",
                       factory=make_factory(state))
    monkeypatch.setattr(ai_story_engine, "gemini_keys", mgr)

    engine = ai_story_engine.AIStoryEngine()
    meta = engine.analyze_story_and_generate_metadata(
        "xqc", "what a win clutch victory", 30)
    assert meta["llm_provider"] == "gemini"
    assert meta["category"] == "Funny Moments"
    status = {k["id"]: k for k in mgr.get_status()["keys"]}
    assert status["env-1"]["status"] == "cooling_down"
    assert status["env-2"]["last_used_at"] is not None


def test_engine_falls_back_to_heuristic_when_all_keys_fail(
        monkeypatch, tmp_path):
    import ai_story_engine
    import llm_providers

    state = {"generate": {
        KEY1: FakeApiError(401, "bad key"),
        KEY2: FakeApiError(401, "bad key"),
    }, "list": {}}
    mgr = make_manager(tmp_path, env_keys=f"{KEY1},{KEY2}",
                       factory=make_factory(state))
    monkeypatch.setattr(ai_story_engine, "gemini_keys", mgr)
    monkeypatch.setattr(
        llm_providers.OpenAICompatibleClient, "generate_json",
        lambda self, prompt, **kw: (_ for _ in ()).throw(
            RuntimeError("no local server")))
    engine = ai_story_engine.AIStoryEngine()
    meta = engine.analyze_story_and_generate_metadata(
        "xqc", "what a win clutch victory", 30)
    assert meta["llm_provider"] == "heuristic"
    assert meta["category"] == "Big Wins"


def test_engine_no_keys_uses_heuristic(monkeypatch, tmp_path):
    import ai_story_engine
    import llm_providers

    mgr = make_manager(tmp_path, env_keys="")
    monkeypatch.setattr(ai_story_engine, "gemini_keys", mgr)
    monkeypatch.setattr(
        llm_providers.OpenAICompatibleClient, "generate_json",
        lambda self, prompt, **kw: (_ for _ in ()).throw(
            RuntimeError("no local server")))
    engine = ai_story_engine.AIStoryEngine()
    meta = engine.analyze_story_and_generate_metadata(
        "xqc", "what a win clutch victory", 30)
    assert meta["llm_provider"] == "heuristic"


# ---------------------------------------------------------------------------
# Dashboard routes (direct handler calls -- no TestClient in this repo)
# ---------------------------------------------------------------------------

def _load_dashboard(monkeypatch, tmp_path, factory=None, env_keys=""):
    """Import web_dashboard with its key manager pointed at a throwaway
    store. Returns (web_dashboard, manager)."""
    import web_dashboard

    state = {"generate": {}, "list": {}}
    mgr = make_manager(tmp_path, env_keys=env_keys,
                       factory=factory or make_factory(state))
    monkeypatch.setattr(web_dashboard, "get_key_manager", lambda: mgr)
    return web_dashboard, mgr


def test_route_list_empty(monkeypatch, tmp_path):
    web_dashboard, _ = _load_dashboard(monkeypatch, tmp_path)
    data = asyncio.run(web_dashboard.list_gemini_keys())
    assert data["counts"]["total"] == 0
    assert data["keys"] == []
    assert data["source"] == "none"


def test_route_add_then_list_masked(monkeypatch, tmp_path):
    web_dashboard, _ = _load_dashboard(monkeypatch, tmp_path)
    req = web_dashboard.GeminiKeysReq(
        keys_text=f"{KEY1}\n{KEY2}\nnot-a-key")
    result = asyncio.run(web_dashboard.add_gemini_keys(req))
    assert result == {"added": 2, "duplicates": 0, "invalid": 1,
                      "total": 2}
    data = asyncio.run(web_dashboard.list_gemini_keys())
    assert data["counts"]["total"] == 2
    blob = json.dumps(data)
    assert KEY1 not in blob
    assert KEY2 not in blob
    assert mask_key(KEY1) in blob
    assert data["source"] == "store"


def test_route_add_duplicates_reported(monkeypatch, tmp_path):
    web_dashboard, _ = _load_dashboard(monkeypatch, tmp_path)
    req = web_dashboard.GeminiKeysReq(keys_text=KEY1)
    asyncio.run(web_dashboard.add_gemini_keys(req))
    result = asyncio.run(web_dashboard.add_gemini_keys(req))
    assert result["added"] == 0
    assert result["duplicates"] == 1
    assert result["total"] == 1


def test_route_delete(monkeypatch, tmp_path):
    import fastapi

    web_dashboard, mgr = _load_dashboard(monkeypatch, tmp_path)
    mgr.add_keys(f"{KEY1}\n{KEY2}")
    key_id = mgr.get_status()["keys"][0]["id"]
    result = asyncio.run(web_dashboard.delete_gemini_key(key_id))
    assert result == {"status": "deleted", "key_id": key_id}
    assert mgr.key_count() == 1
    with pytest.raises(fastapi.HTTPException) as exc_info:
        asyncio.run(web_dashboard.delete_gemini_key(key_id))
    assert exc_info.value.status_code == 404


def test_route_test_keys_format_only(monkeypatch, tmp_path):
    web_dashboard, mgr = _load_dashboard(monkeypatch, tmp_path)
    mgr.add_keys(f"{KEY1}\n{KEY2}")
    result = asyncio.run(web_dashboard.test_gemini_keys())
    assert result["mode"] == "format_only"  # no genai SDK in this VM
    assert len(result["results"]) == 2
    blob = json.dumps(result)
    assert KEY1 not in blob
    assert KEY2 not in blob


# ---------------------------------------------------------------------------
# Repo wiring checks
# ---------------------------------------------------------------------------

def test_store_file_is_gitignored():
    from config import BASE_DIR

    gitignore = (BASE_DIR / ".gitignore").read_text(encoding="utf-8")
    assert "gemini_keys.json" in gitignore


def test_dashboard_has_keys_section():
    from config import BASE_DIR

    html = (BASE_DIR / "templates" / "index.html").read_text(
        encoding="utf-8")
    for marker in ("gemini-keys-textarea", "gemini-keys-list",
                   "gemini-keys-summary", "addGeminiKeys()",
                   "testGeminiKeys()", "fetchGeminiKeys()"):
        assert marker in html
    # Held/local-only dashboard features stay OUT of this port: the
    # captions toggle and the clip-permission badge belong to other,
    # unpushed work on the old local copy -- they must not leak in.
    assert "modal-caption-options-toggle" not in html
    assert "clip-rule-value" not in html
