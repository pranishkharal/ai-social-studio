"""
gemini_keys.py -- bulk Gemini API key manager with rotation.

Pranish holds MANY free-tier Gemini API keys. This module is the single
source of truth for all of them: it stores them, hands out clients in
round-robin order across healthy keys, cools keys down on rate limits,
and marks dead keys invalid -- so one exhausted key never stops a clip.

Key sources, in priority order:
  (a) gemini_keys.json in the repo root (gitignored, never committed),
      filled by pasting keys in the dashboard (Settings -> Gemini API
      Keys) or by the API. Once this file has any keys, it is the only
      source -- the .env value is ignored until the store is emptied.
  (b) Otherwise the GEMINI_API_KEY env var (from .env), split on
      comma/semicolon/newline -- so existing single-key setups keep
      working with zero changes.

Security rules (never break these):
  - A full key is NEVER logged, returned by an API, rendered in the
    dashboard, or included in an error message. Only mask_key() output
    (first 4 + last 4 chars) ever leaves this module.
  - The store file is written atomically (temp file + replace) and
    chmod 600 where the OS supports it.

Thread safety: the app is multithreaded (watcher threads, render
threads, dashboard). All rotation and status updates go through one
re-entrant lock.

The google-genai SDK is imported LAZILY, only when a client is first
created. Importing this module -- and constructing a manager -- never
needs the SDK, so the heuristic fallback keeps working on machines
without it.
"""

import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from config import BASE_DIR, GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Split bulk-pasted keys the same way main does: comma, semicolon, or
# any newline. (main: re.split(r"[,;\n\r]+", ...) in ai_story_engine.)
_KEY_SPLIT_RE = re.compile(r"[,;\n\r]+")

# Loose format check for a Gemini API key: the public prefix plus a
# minimum length. This is a typo filter, not a proof the key works --
# only a real API call proves that (see test_all_keys).
_KEY_PREFIX = "AIza"
_MIN_KEY_LENGTH = 20

DEFAULT_COOLDOWN_SECONDS = 60

# Statuses a key can be in. "healthy" keys are used for rotation;
# "cooling_down" keys are skipped until their cooldown expires (then
# they return to healthy automatically); "invalid" keys are never
# retried until re-added or manually re-enabled.
STATUS_HEALTHY = "healthy"
STATUS_COOLING_DOWN = "cooling_down"
STATUS_INVALID = "invalid"

STORE_VERSION = 1


class NoHealthyKeyError(RuntimeError):
    """Raised when no key is usable right now (none configured, all on
    cooldown, or all invalid). Callers treat this like "Gemini is not
    configured" and fall back to heuristics."""


class GeminiUnavailableError(RuntimeError):
    """Raised when the google-genai SDK is not installed, so no client
    can be created at all. Distinct from NoHealthyKeyError so callers
    can stop retrying immediately -- more attempts cannot help."""


def mask_key(key):
    """Mask a key for display/logs: first 4 + last 4 chars.

    Example: "AIzaSyD..." + 31 chars -> "AIza...Ab12". A full key must
    never appear in any masked output.
    """
    key = key or ""
    if len(key) < 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_keys(text):
    """Parse bulk-pasted keys into valid keys + rejection counts.

    Accepts one key per line, comma-separated, semicolon-separated, or
    mixed. Strips whitespace (and surrounding quotes, for keys pasted
    out of JSON/CSV), drops empties, dedupes exact matches. A key is
    valid when it starts with "AIza" and is at least 20 chars long.

    Returns {"valid": [keys...], "invalid": int,
             "invalid_reasons": {"wrong_prefix": int, "too_short": int}}.
    Rejected values are NEVER echoed back -- only counts and reasons.
    """
    valid = []
    invalid = 0
    reasons = {"wrong_prefix": 0, "too_short": 0}
    seen = set()
    for token in _KEY_SPLIT_RE.split(text or ""):
        token = token.strip().strip("\"'").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        if not token.startswith(_KEY_PREFIX):
            invalid += 1
            reasons["wrong_prefix"] += 1
            continue
        if len(token) < _MIN_KEY_LENGTH:
            invalid += 1
            reasons["too_short"] += 1
            continue
        valid.append(token)
    return {"valid": valid, "invalid": invalid, "invalid_reasons": reasons}


def split_env_keys(env_value):
    """Split a GEMINI_API_KEY env value into candidate keys (same
    regex as parse_keys, but without validation -- env keys are
    trusted as configured, and a bad one is marked invalid by its
    first real API failure)."""
    return [t.strip() for t in _KEY_SPLIT_RE.split(env_value or "")
            if t.strip()]


def classify_error(error):
    """Classify a Gemini API failure for rotation decisions.

    Returns "rate_limited" (429 / quota / resource-exhausted),
    "invalid" (401 / 403 / bad key), or "other". Checks a numeric
    .code/.status attribute first (the genai SDK's ClientError
    carries one), then falls back to message substrings.
    """
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(error, "status", None)
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):
        code = None
    if code == 429:
        return "rate_limited"
    if code in (401, 403):
        return "invalid"
    text = str(error).lower()
    if ("429" in text or "rate limit" in text or "rate_limit" in text
            or "quota" in text or "resource_exhausted" in text
            or "resource exhausted" in text):
        return "rate_limited"
    if ("401" in text or "403" in text or "unauthenticated" in text
            or "permission denied" in text or "api key not valid" in text
            or "invalid api key" in text or "api_key_invalid" in text):
        return "invalid"
    return "other"


def _default_client_factory(key, timeout=None):
    """Create a real genai.Client for one key. Imported lazily so this
    module stays importable without the SDK. ``timeout`` is in
    SECONDS; the SDK's HttpOptions.timeout is in MILLISECONDS, so
    it is converted here (passing seconds straight through made
    every call time out instantly)."""
    from google import genai
    if timeout:
        try:
            from google.genai import types
            return genai.Client(
                api_key=key,
                http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        except Exception:
            pass  # fall through to a plain client
    return genai.Client(api_key=key)


def _new_entry(key, source):
    return {
        "id": secrets.token_hex(4),
        "key": key,
        "source": source,
        "added_at": _now_iso(),
        "status": STATUS_HEALTHY,
        "last_used_at": None,
        "last_error": None,
        "failure_count": 0,
        # Epoch seconds until which the key is skipped, or None.
        "cooldown_until": None,
    }


class GeminiKeyManager:
    """Owns the Gemini key pool: storage, rotation, health tracking.

    ``store_path`` defaults to <repo root>/gemini_keys.json (override
    with the GEMINI_KEYS_FILE env var, or pass it explicitly -- tests
    use a temp file). ``env_keys`` defaults to config.GEMINI_API_KEY.
    ``client_factory`` is injectable for tests: factory(key, timeout)
    -> an object with .models.generate_content / .models.list.
    ``cooldown_seconds`` defaults to GEMINI_KEY_COOLDOWN_SECONDS or 60.
    """

    def __init__(self, store_path=None, env_keys=None, client_factory=None,
                 cooldown_seconds=None):
        if store_path is None:
            store_path = os.getenv("GEMINI_KEYS_FILE") or str(
                BASE_DIR / "gemini_keys.json")
        self.store_path = Path(store_path)
        if env_keys is None:
            env_keys = GEMINI_API_KEY
        if cooldown_seconds is None:
            try:
                cooldown_seconds = int(
                    os.getenv("GEMINI_KEY_COOLDOWN_SECONDS", "60"))
            except ValueError:
                cooldown_seconds = DEFAULT_COOLDOWN_SECONDS
        self.cooldown_seconds = max(1, cooldown_seconds)
        self._client_factory = client_factory or _default_client_factory
        self._lock = threading.RLock()
        self._clients = {}  # key_id -> genai.Client (created lazily)
        self._next_index = 0
        self._store_entries = self._load_store()
        self._env_entries = [
            dict(_new_entry(key, "env"), id=f"env-{i + 1}")
            for i, key in enumerate(split_env_keys(env_keys))
        ]
        self._entries = (self._store_entries if self._store_entries
                         else self._env_entries)

    # -- storage ------------------------------------------------------

    def _load_store(self):
        """Read gemini_keys.json. A missing file means "no stored keys"
        (env fallback applies). A corrupt file is logged (never with
        key material) and treated as empty -- a bad store file must
        never stop the app from starting."""
        if not self.store_path.exists():
            return []
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Gemini key store %s unreadable (%s); "
                           "starting with no stored keys",
                           self.store_path, e)
            return []
        entries = data.get("keys") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            return []
        cleaned = []
        for raw in entries:
            if not isinstance(raw, dict) or not raw.get("key"):
                continue
            entry = _new_entry(raw["key"], "store")
            entry.update({k: raw.get(k, entry[k]) for k in
                          ("id", "added_at", "status", "last_used_at",
                           "last_error", "failure_count",
                           "cooldown_until")})
            if entry["status"] not in (STATUS_HEALTHY, STATUS_COOLING_DOWN,
                                       STATUS_INVALID):
                entry["status"] = STATUS_HEALTHY
            cleaned.append(entry)
        return cleaned

    def _save_locked(self):
        """Persist the store atomically (temp file + replace) with
        owner-only permissions where the OS supports it."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": STORE_VERSION, "keys": self._store_entries}
        tmp_path = self.store_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2),
                            encoding="utf-8")
        try:
            os.chmod(tmp_path, 0o600)
        except (OSError, NotImplementedError):
            pass  # Windows / filesystems without POSIX modes
        os.replace(tmp_path, self.store_path)

    def _refresh_locked(self):
        """Rebind the active entries after the store changed: the store
        wins whenever it has any keys, otherwise the env keys apply."""
        self._entries = (self._store_entries if self._store_entries
                         else self._env_entries)

    # -- introspection --------------------------------------------------

    @property
    def source(self):
        """Where the active keys come from: "store", "env", or "none"."""
        if self._store_entries:
            return "store"
        if self._env_entries:
            return "env"
        return "none"

    def has_keys(self):
        with self._lock:
            return bool(self._entries)

    def key_count(self):
        with self._lock:
            return len(self._entries)

    def healthy_count(self):
        with self._lock:
            return len(self._healthy_entries_locked())

    def get_status(self):
        """Masked per-key status for the dashboard/API. Full keys NEVER
        appear here -- see the test asserting exactly that."""
        with self._lock:
            healthy = self._healthy_entries_locked()
            healthy_ids = {e["id"] for e in healthy}
            keys = [{
                "id": e["id"],
                "masked": mask_key(e["key"]),
                "status": e["status"],
                "source": e["source"],
                "added_at": e.get("added_at"),
                "last_used_at": e.get("last_used_at"),
                "last_error": e.get("last_error"),
                "failure_count": e.get("failure_count", 0),
                # Env keys live in .env, not in the store -- they can
                # only be removed by editing .env, so the dashboard
                # hides their Delete button.
                "deletable": e["source"] == "store",
            } for e in self._entries]
            counts = {
                "total": len(self._entries),
                "healthy": len(healthy_ids),
                "cooling_down": sum(
                    1 for e in self._entries
                    if e["status"] == STATUS_COOLING_DOWN),
                "invalid": sum(
                    1 for e in self._entries
                    if e["status"] == STATUS_INVALID),
            }
            return {
                "source": self.source,
                "keys": keys,
                "counts": counts,
                # Transparency: when the store is active, env keys are
                # ignored -- say so instead of hiding it.
                "env_keys_ignored": (len(self._env_entries)
                                     if self._store_entries else 0),
            }

    # -- adding / removing keys -----------------------------------------

    def add_keys(self, text):
        """Parse bulk-pasted text, add the new keys to the store, and
        persist. Returns {"added", "duplicates", "invalid", "total"}.
        Duplicates are keys already in the store (exact match)."""
        parsed = parse_keys(text)
        with self._lock:
            existing = {e["key"] for e in self._store_entries}
            added = 0
            duplicates = 0
            for key in parsed["valid"]:
                if key in existing:
                    duplicates += 1
                    continue
                existing.add(key)
                self._store_entries.append(_new_entry(key, "store"))
                added += 1
            if added:
                self._save_locked()
            self._refresh_locked()
            total = len(self._entries)
        if added:
            logger.info("Gemini keys: added %d, %d duplicates skipped, "
                        "%d invalid rejected (total %d)",
                        added, duplicates, parsed["invalid"], total)
        return {"added": added, "duplicates": duplicates,
                "invalid": parsed["invalid"], "total": total}

    def delete_key(self, key_id):
        """Remove one stored key. Returns True when removed, False when
        the id is unknown or the key is env-sourced (not deletable)."""
        with self._lock:
            for i, entry in enumerate(self._store_entries):
                if entry["id"] == key_id:
                    del self._store_entries[i]
                    self._clients.pop(key_id, None)
                    self._save_locked()
                    self._refresh_locked()
                    logger.info("Gemini key deleted: id=%s", key_id)
                    return True
        return False

    def reenable_key(self, key_id):
        """Manually re-enable an invalid key (e.g. after fixing the
        key at the provider). Returns True when re-enabled."""
        with self._lock:
            for entry in self._store_entries:
                if entry["id"] == key_id:
                    entry["status"] = STATUS_HEALTHY
                    entry["cooldown_until"] = None
                    entry["last_error"] = None
                    entry["failure_count"] = 0
                    self._save_locked()
                    return True
        return False

    # -- rotation ---------------------------------------------------------

    def _healthy_entries_locked(self):
        """Healthy entries, expiring finished cooldowns first. Caller
        must hold the lock."""
        now = time.time()
        changed = False
        for entry in self._entries:
            if (entry["status"] == STATUS_COOLING_DOWN
                    and entry.get("cooldown_until") is not None
                    and now >= entry["cooldown_until"]):
                entry["status"] = STATUS_HEALTHY
                entry["cooldown_until"] = None
                entry["last_error"] = None
                changed = True
        if changed and self._entries is self._store_entries:
            self._save_locked()
        return [e for e in self._entries
                if e["status"] == STATUS_HEALTHY]

    def _client_for_locked(self, entry):
        """Cached genai.Client for one entry, created lazily. Raises
        GeminiUnavailableError when the SDK is not installed."""
        client = self._clients.get(entry["id"])
        if client is None:
            try:
                client = self._client_factory(entry["key"])
            except ImportError as e:
                raise GeminiUnavailableError(
                    "google-genai is not installed; cannot create a "
                    "Gemini client") from e
            self._clients[entry["id"]] = client
        return client

    def get_client(self):
        """Return (client, key_id) for the next healthy key in
        round-robin order -- never always key 0. Raises
        NoHealthyKeyError when no key is usable, or
        GeminiUnavailableError when the SDK is missing. The caller
        MUST report the outcome with report_success/report_failure
        so rotation health stays accurate."""
        with self._lock:
            healthy = self._healthy_entries_locked()
            if not healthy:
                raise NoHealthyKeyError(
                    "No healthy Gemini key available "
                    f"({self.key_count()} configured, "
                    f"{self.source} source)")
            entry = healthy[self._next_index % len(healthy)]
            self._next_index += 1
            client = self._client_for_locked(entry)
            return client, entry["id"]

    def get_client_for(self, key_id):
        """Return the cached (or fresh) client for one specific key,
        regardless of health -- used by the key tester."""
        with self._lock:
            for entry in self._entries:
                if entry["id"] == key_id:
                    return self._client_for_locked(entry)
        raise KeyError(f"Unknown Gemini key id: {key_id}")

    # -- health reporting ---------------------------------------------------

    def _sanitize_error(self, entry, error):
        """Error text safe to store/display: truncated, and with the
        entry's own key value scrubbed out if the SDK ever echoed it."""
        text = str(error)[:200]
        key = entry.get("key") or ""
        if key and key in text:
            text = text.replace(key, "[key]")
        return text

    def report_success(self, key_id):
        """Record a successful API call: refreshes last_used, clears
        the failure streak and any stored error."""
        with self._lock:
            for entry in self._entries:
                if entry["id"] == key_id:
                    entry["last_used_at"] = _now_iso()
                    entry["failure_count"] = 0
                    entry["last_error"] = None
                    if self._entries is self._store_entries:
                        self._save_locked()
                    return True
        return False

    def report_failure(self, key_id, error):
        """Record a failed API call and update the key's health:
        rate limits -> 60 s cooldown (then healthy again); auth
        errors -> invalid (never retried); anything else -> failure
        count only. Returns the classification."""
        kind = classify_error(error)
        with self._lock:
            for entry in self._entries:
                if entry["id"] != key_id:
                    continue
                entry["failure_count"] = entry.get("failure_count", 0) + 1
                entry["last_error"] = self._sanitize_error(entry, error)
                if kind == "rate_limited":
                    entry["status"] = STATUS_COOLING_DOWN
                    entry["cooldown_until"] = (time.time()
                                               + self.cooldown_seconds)
                    logger.warning(
                        "Gemini key %s rate-limited; cooling down for "
                        "%ds", mask_key(entry["key"]),
                        self.cooldown_seconds)
                elif kind == "invalid":
                    entry["status"] = STATUS_INVALID
                    entry["cooldown_until"] = None
                    logger.warning("Gemini key %s marked invalid: %s",
                                   mask_key(entry["key"]),
                                   entry["last_error"])
                if self._entries is self._store_entries:
                    self._save_locked()
                return kind
        logger.warning("Gemini failure reported for unknown key id=%s",
                       key_id)
        return kind

    # -- key testing ----------------------------------------------------------

    def test_all_keys(self):
        """Validate every configured key with one tiny real API call
        each (list models -- no generation tokens spent). Returns
        {"mode": "live"|"format_only", "results": [{id, masked,
        result, detail}]}. When the SDK is unavailable, falls back to
        format validation only and says so. Testing updates key
        health: invalid keys are marked invalid, rate-limited keys go
        on cooldown."""
        try:
            from google import genai  # noqa: F401 -- availability probe
            sdk_available = True
        except ImportError:
            sdk_available = False

        with self._lock:
            entries = list(self._entries)

        if not sdk_available:
            results = [{
                "id": e["id"],
                "masked": mask_key(e["key"]),
                "result": ("ok" if e["key"].startswith(_KEY_PREFIX)
                           and len(e["key"]) >= _MIN_KEY_LENGTH
                           else "invalid"),
                "detail": "Format check only -- google-genai is not "
                          "installed, so no live call was made.",
            } for e in entries]
            return {"mode": "format_only", "results": results}

        results = []
        for entry in entries:
            try:
                # A dedicated short-timeout client per key: one slow
                # key must not stall the whole test run.
                client = self._client_factory(entry["key"], timeout=15)
                pager = client.models.list()
                next(iter(pager), None)  # one request; discard the page
                self.report_success(entry["id"])
                results.append({
                    "id": entry["id"],
                    "masked": mask_key(entry["key"]),
                    "result": "ok",
                    "detail": "Key works.",
                })
            except Exception as e:  # noqa: BLE001 -- classified below
                kind = self.report_failure(entry["id"], e)
                results.append({
                    "id": entry["id"],
                    "masked": mask_key(entry["key"]),
                    "result": ("invalid" if kind == "invalid"
                               else "rate_limited" if kind == "rate_limited"
                               else "error"),
                    "detail": self._sanitize_error(entry, e),
                })
        return {"mode": "live", "results": results}


class _RotatingModels:
    """The .models namespace of a rotating client: every call goes to
    the next healthy key, and the outcome is reported back."""

    def __init__(self, manager):
        self._manager = manager

    def generate_content(self, model, contents, **kwargs):
        client, key_id = self._manager.get_client()
        try:
            response = client.models.generate_content(
                model=model, contents=contents, **kwargs)
        except Exception as e:  # noqa: BLE001 -- reported, then raised
            self._manager.report_failure(key_id, e)
            raise
        self._manager.report_success(key_id)
        return response

    def list(self, **kwargs):
        client, key_id = self._manager.get_client()
        try:
            result = client.models.list(**kwargs)
        except Exception as e:  # noqa: BLE001 -- reported, then raised
            self._manager.report_failure(key_id, e)
            raise
        self._manager.report_success(key_id)
        return result


class RotatingGeminiClient:
    """Drop-in stand-in for genai.Client with a rotating key pool.

    Exposes the same .models.generate_content() / .models.list()
    surface the app already uses, so existing callers (captions.py's
    optional LLM path) get rotation with no code changes. Raises
    NoHealthyKeyError when no key is usable -- callers already treat
    any raise as "fall back", so behaviour is unchanged when no keys
    exist.
    """

    def __init__(self, manager):
        self._manager = manager
        self.models = _RotatingModels(manager)


# -- module-level singleton ---------------------------------------------------

_manager = None
_manager_lock = threading.Lock()


def get_key_manager(store_path=None, env_keys=None):
    """Return the process-wide key manager, creating it on first use.
    ``store_path`` / ``env_keys`` are honoured only on creation (tests
    construct GeminiKeyManager directly, or call reset_key_manager()
    first)."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = GeminiKeyManager(store_path=store_path,
                                            env_keys=env_keys)
    return _manager


def reset_key_manager():
    """Drop the cached singleton (tests only). The next
    get_key_manager() call builds a fresh manager."""
    global _manager
    with _manager_lock:
        _manager = None
