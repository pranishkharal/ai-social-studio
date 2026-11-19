"""Tests for kick_webhooks.py (Tier 1, feature 3: Kick webhooks).

Self-contained: this VM lacks python-dotenv and fastapi, and importing the
real database.py would create the repo's real kick_engine.db, so all three
are stubbed in sys.modules before importing the module under test -- but
only where tests/conftest.py has not already provided them (it stubs dotenv
and fastapi and points the real database at a throwaway file; reusing that
keeps this file a good citizen in combined pytest runs). Conftest's fastapi
stub lacks ``Request`` and the ``fastapi.responses`` submodule, so this file
adds just those two pieces instead of replacing the stub. Stubs this file
installs itself are removed again in teardown_module. No real HTTP is
performed: urllib is monkeypatched and webhook deliveries are fed to the
registered route handler directly.

RSA note: the test key pair below was generated with OpenSSL for these
tests only (it is NOT Kick's key and NOT a secret -- it exists only so the
pure-stdlib verifier can be exercised end to end). The pure-Python signer
in this file was cross-checked against OpenSSL during development:
``openssl dgst -sha256 -verify`` accepts its signatures, and the module's
verifier accepts OpenSSL-produced signatures.
"""

import asyncio
import base64
import hashlib
import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- test-only stubs (see module docstring) ---
_INSTALLED_STUBS = []


def _install_stub(name, module):
    """Install *module* under *name* only if nothing is there yet."""
    if name not in sys.modules:
        sys.modules[name] = module
        _INSTALLED_STUBS.append(name)


_dotenv_stub = types.ModuleType("dotenv")
_dotenv_stub.load_dotenv = lambda *a, **k: False
_install_stub("dotenv", _dotenv_stub)

_database_stub = types.ModuleType("database")
_database_stub.log_event = lambda *a, **k: None
_database_stub.update_streamer_status = lambda *a, **k: None
_database_stub.add_notification = lambda *a, **k: 0
_database_stub.add_clip = lambda *a, **k: 0
_database_stub.get_connection = lambda *a, **k: (_ for _ in ()).throw(
    RuntimeError("database stubbed in tests"))
_database_stub.get_all_clips = lambda: []
_install_stub("database", _database_stub)


class _FakeJSONResponse:
    def __init__(self, content=None, status_code=200, **kwargs):
        self.content = content
        self.status_code = status_code
        self.body = (json.dumps(content).encode("utf-8")
                     if content is not None else b"")


class _FakeAPIRouter:
    """Fallback when no fastapi stub exists yet (standalone runs)."""

    def __init__(self, *a, **k):
        self.routes = []  # (method, path, handler)

    def post(self, path, **kwargs):
        def deco(fn):
            self.routes.append(("POST", path, fn))
            return fn
        return deco

    def get(self, path, **kwargs):
        def deco(fn):
            self.routes.append(("GET", path, fn))
            return fn
        return deco


_fastapi_stub = sys.modules.get("fastapi")
_added_request_attr = False
if _fastapi_stub is None:
    _fastapi_stub = types.ModuleType("fastapi")
    _fastapi_stub.APIRouter = _FakeAPIRouter
    _fastapi_stub.Request = object  # only used as an annotation
    _install_stub("fastapi", _fastapi_stub)
elif not hasattr(_fastapi_stub, "Request"):
    # Reuse conftest's stub; it just lacks Request.
    _fastapi_stub.Request = object
    _added_request_attr = True

_fastapi_responses_stub = types.ModuleType("fastapi.responses")
_fastapi_responses_stub.JSONResponse = _FakeJSONResponse
_install_stub("fastapi.responses", _fastapi_responses_stub)


def teardown_module(module):
    for name in _INSTALLED_STUBS:
        sys.modules.pop(name, None)
    if _added_request_attr:
        try:
            delattr(sys.modules["fastapi"], "Request")
        except (KeyError, AttributeError):
            pass


import kick_webhooks as kw  # noqa: E402

# ----------------------------------------------------------------------
# Test-only RSA key pair (generated with OpenSSL; see module docstring).
# ----------------------------------------------------------------------
TEST_PRIVATE_KEY_PEM = """-----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEAqQtVARn8ns7Z2V4h4kqGHQkiK8GclbCsWTyUNqTSaC8zasBw
34vQNryR/7EUDgRQrN86jpdZpP3yIQtopNJh3Ut+vghaD2iPc3bqL5R9n2H8rIz6
KvCJu/2malz5Y78suzSeMur0e4YGhw7Liyfbwz8ZXnJ75yqWKeB+KDmaXBoo/e08
BnzPcU/UT0W70PAvihq3I0s+ewzmR8GVZeI7Xr0p4gTs9EylAkiQz10pRXAqbWXU
JBdWPu8rHmlQdyGHO5Me5zDplKldueE8tsJHkflDYdlqYuA4Ese2ianksDupQlgn
hJbMMUx1vQxM6wjnzekoKjtJlEeaGOsKu5xLIQIDAQABAoIBAAa5luTBCHKhavDU
XLpPHxbT3zuuJhkdkuexwCQ3NPsp0zdI8gdEqBsrWo65Yta8FMu3IudbYvi26mgR
N1Uk8k7T/cDDMzZXzAAquojmH4QaE/Z3MnhyQrUmE2skaSlO6KbHF7rnFmQ9H1tG
CzKCpWH3C57Hz2vRfwaUt0lz9t2j00acaYXlnMcVc4bIaIcmBzZBBvD+P944oLx4
NhCxVNiXTdI7jNQfUHwtSoG/dRdP4zFohPlxm+Tt7zN5wvTAs+RdPxBWnQB8vuRQ
MWO113MQLX0dUOf5BI+H068Imxx6P0qFwSq2TBKnjTNV2BNN0gf9lhQvv8YCZaQR
C38lwWkCgYEA6yXvUNUK+jBineEDJArHrsKALVUyMaTC9tDbDPU2hJEsCeaIKX+8
eCRaZGCtwoxyy8y/ABB0iG6bPY7hlyN5QPT6FAqCmtVI5HZ6Xu5kkHgTVQPcTWmS
Z15Smgez2oMOm8NhHXG25GA5eFbkN66SFq8KO46InzJSAgz87JVSw6kCgYEAuAjI
HC8N6ptdeRxp2KSZwOokkKsLxXoNeKmEp79VqvRJsESaGHuMmexh0HloXWWVh6RC
fY8lVXusOWNVEKDB1aNLn4VBKlilgjBKpUjDQpFnDI9wIsaF4+3Mfbds5V4kL2oS
G9ZFU9u7W5s/gXq+JJ7aZpXyOPcI70ukD3nVdrkCgYEAmkncuaPfvQMsYbz3K5nr
JU63fYcuFQavafTmFZ17dQYv6Bsb8ma+yfHIYZ5RxJoU5IXmOOuTGsXAFGcAvANp
luZmg+nVgjQ5zLW0Wz9+OpCe1qRe02hWyH5KIoi1SMW/OxK6UPCA+V3tt3kD60ke
z8VwKF3AZPxGdNQCzFS55mkCgYEAh+osZ+w0eezOZtCeN1qxFvGVmaZJZknSwydn
CkBbs4sBFQZFTAmK+G7EeqP/eOK6QVgFpxdd+jkBI80BmOsw3GfiROi0DOM1MOYb
auBHHNWn5nXU2koFviT0nnAmI2uYi8/pmddwkhM41tncV/LflMohAByDrKB9X8Vc
X/WZujkCgYEAmw8i8yI9KjvoAkiFYHEDLoP22N8eEGQLezHkKyLt9IbZKUZANA7m
GediL4phYCxyOdyiHrn3phvQ7qoA9OO5Gl+xMZ/Po8CHwMYOwI4LH2gUFVRZd9PO
Kli4HLc82gtLBKe5GH6b9KC9erqZTLs83FrA4GwbezG8TYBq/Iaood4=
-----END RSA PRIVATE KEY-----"""

TEST_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAqQtVARn8ns7Z2V4h4kqG
HQkiK8GclbCsWTyUNqTSaC8zasBw34vQNryR/7EUDgRQrN86jpdZpP3yIQtopNJh
3Ut+vghaD2iPc3bqL5R9n2H8rIz6KvCJu/2malz5Y78suzSeMur0e4YGhw7Liyfb
wz8ZXnJ75yqWKeB+KDmaXBoo/e08BnzPcU/UT0W70PAvihq3I0s+ewzmR8GVZeI7
Xr0p4gTs9EylAkiQz10pRXAqbWXUJBdWPu8rHmlQdyGHO5Me5zDplKldueE8tsJH
kflDYdlqYuA4Ese2ianksDupQlgnhJbMMUx1vQxM6wjnzekoKjtJlEeaGOsKu5xL
IQIDAQAB
-----END PUBLIC KEY-----"""


def _der_read_length(data, pos):
    first = data[pos]
    pos += 1
    if first < 0x80:
        return first, pos
    num_bytes = first & 0x7F
    return int.from_bytes(data[pos:pos + num_bytes], "big"), pos + num_bytes


def _der_read_tlv(data, pos):
    tag = data[pos]
    length, pos = _der_read_length(data, pos + 1)
    return tag, data[pos:pos + length], pos + length


def _parse_pkcs1_private_key(pem_text):
    """Parse a PKCS#1 RSA private key PEM -> (n, d). Test helper only."""
    lines = [ln.strip() for ln in pem_text.splitlines()
             if ln.strip() and not ln.startswith("-----")]
    der = base64.b64decode("".join(lines))
    _, seq, _ = _der_read_tlv(der, 0)
    values, pos = [], 0
    while pos < len(seq):
        _, content, pos = _der_read_tlv(seq, pos)
        values.append(int.from_bytes(content, "big"))
    return values[1], values[3]  # modulus, private exponent


_TEST_N, _TEST_D = _parse_pkcs1_private_key(TEST_PRIVATE_KEY_PEM)


def _test_sign(message: bytes) -> bytes:
    """RSA PKCS#1 v1.5 SHA-256 signature with the test key (test helper)."""
    digest = hashlib.sha256(message).digest()
    t = kw._SHA256_DIGESTINFO_PREFIX + digest
    k = (_TEST_N.bit_length() + 7) // 8
    em = b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t
    return pow(int.from_bytes(em, "big"), _TEST_D, _TEST_N).to_bytes(k, "big")


def _signed_headers(message_id, timestamp, raw_body):
    """Kick-style delivery headers for a payload, signed with the test key."""
    signed = f"{message_id}.{timestamp}.".encode("utf-8") + raw_body
    sig = base64.b64encode(_test_sign(signed)).decode("ascii")
    return {
        "kick-event-message-id": message_id,
        "kick-event-message-timestamp": timestamp,
        "kick-event-signature": sig,
        "kick-event-subscription-id": "sub-1",
        "kick-event-version": "1",
    }


# ----------------------------------------------------------------------
# RSA signature verification
# ----------------------------------------------------------------------
def test_kick_public_key_parses_as_2048_bit():
    n, e = kw._parse_rsa_public_key_pem(kw.KICK_PUBLIC_KEY_PEM)
    assert n.bit_length() == 2048
    assert e == 65537


def test_verify_valid_signature():
    body = b'{"broadcaster_user_login":"xqc","is_live":true}'
    headers = _signed_headers("msg-1", "1700000000", body)
    assert kw.verify_kick_signature(
        headers["kick-event-message-id"],
        headers["kick-event-message-timestamp"],
        body, headers["kick-event-signature"],
        public_key_pem=TEST_PUBLIC_KEY_PEM) is True


def test_verify_rejects_tampered_body():
    body = b'{"broadcaster_user_login":"xqc","is_live":true}'
    headers = _signed_headers("msg-1", "1700000000", body)
    assert kw.verify_kick_signature(
        "msg-1", "1700000000", body + b" ",
        headers["kick-event-signature"],
        public_key_pem=TEST_PUBLIC_KEY_PEM) is False


def test_verify_rejects_wrong_message_id():
    body = b'{"is_live":true}'
    headers = _signed_headers("msg-1", "1700000000", body)
    assert kw.verify_kick_signature(
        "msg-2", "1700000000", body,
        headers["kick-event-signature"],
        public_key_pem=TEST_PUBLIC_KEY_PEM) is False


def test_verify_rejects_wrong_timestamp():
    body = b'{"is_live":true}'
    headers = _signed_headers("msg-1", "1700000000", body)
    assert kw.verify_kick_signature(
        "msg-1", "1700000001", body,
        headers["kick-event-signature"],
        public_key_pem=TEST_PUBLIC_KEY_PEM) is False


def test_verify_rejects_missing_headers():
    body = b'{"is_live":true}'
    assert kw.verify_kick_signature(
        None, "1700000000", body, "AAAA",
        public_key_pem=TEST_PUBLIC_KEY_PEM) is False
    assert kw.verify_kick_signature(
        "msg-1", None, body, "AAAA",
        public_key_pem=TEST_PUBLIC_KEY_PEM) is False
    assert kw.verify_kick_signature(
        "msg-1", "1700000000", body, None,
        public_key_pem=TEST_PUBLIC_KEY_PEM) is False


def test_verify_rejects_bad_base64():
    assert kw.verify_kick_signature(
        "msg-1", "1700000000", b"{}", "!!!not-base64!!!",
        public_key_pem=TEST_PUBLIC_KEY_PEM) is False


def test_verify_rejects_wrong_length_signature():
    assert kw.verify_kick_signature(
        "msg-1", "1700000000", b"{}",
        base64.b64encode(b"too short").decode(),
        public_key_pem=TEST_PUBLIC_KEY_PEM) is False


def test_verify_rejects_bad_public_key():
    body = b'{"is_live":true}'
    headers = _signed_headers("msg-1", "1700000000", body)
    assert kw.verify_kick_signature(
        "msg-1", "1700000000", body,
        headers["kick-event-signature"],
        public_key_pem="not a pem at all") is False


def test_verify_accepts_str_body():
    body = '{"is_live":true}'
    headers = _signed_headers("msg-1", "1700000000", body.encode())
    assert kw.verify_kick_signature(
        "msg-1", "1700000000", body,
        headers["kick-event-signature"],
        public_key_pem=TEST_PUBLIC_KEY_PEM) is True


def test_webhook_secret_pem_override(monkeypatch):
    monkeypatch.setenv("KICK_WEBHOOK_SECRET", TEST_PUBLIC_KEY_PEM)
    assert kw.get_webhook_public_key_pem() == TEST_PUBLIC_KEY_PEM


def test_webhook_secret_non_pem_ignored(monkeypatch):
    monkeypatch.setenv("KICK_WEBHOOK_SECRET", "some-hmac-secret")
    assert kw.get_webhook_public_key_pem() == kw.KICK_PUBLIC_KEY_PEM


def test_webhooks_configured(monkeypatch):
    monkeypatch.delenv("KICK_CLIENT_ID", raising=False)
    monkeypatch.delenv("KICK_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("KICK_WEBHOOK_URL", raising=False)
    assert kw.webhooks_configured() is False
    monkeypatch.setenv("KICK_CLIENT_ID", "cid")
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("KICK_WEBHOOK_URL", "https://example.com/api/webhooks/kick")
    assert kw.webhooks_configured() is True


# ----------------------------------------------------------------------
# Subscription management (urllib mocked -- no real HTTP)
# ----------------------------------------------------------------------
class _FakeHTTPResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def mock_urlopen(monkeypatch):
    calls = []

    def _fake_urlopen(req, context=None, timeout=None):
        calls.append(req)
        return _FakeHTTPResponse({"data": [{"id": "sub-1", "name": "x"}]})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(kw.urllib.request, "urlopen", _fake_urlopen)
    return calls


def _request_body(req):
    return json.loads(req.data.decode("utf-8")) if req.data else None


def test_create_subscription_request_shape(mock_urlopen, monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    result = kw.create_subscription(
        events=[{"name": "livestream.status.updated", "version": 1}],
        broadcaster_user_id=12345)
    assert result["ok"] is True
    assert len(mock_urlopen) == 1
    req = mock_urlopen[0]
    assert req.full_url == "https://api.kick.com/public/v1/events/subscriptions"
    assert req.get_method() == "POST"
    assert req.headers["Authorization"] == "Bearer tok-123"
    body = _request_body(req)
    assert body["method"] == "webhook"
    assert body["broadcaster_user_id"] == 12345
    assert body["events"] == [{"name": "livestream.status.updated", "version": 1}]


def test_create_subscription_defaults_to_livestream_event(mock_urlopen, monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    kw.create_subscription()
    body = _request_body(mock_urlopen[0])
    assert body["events"] == [{"name": "livestream.status.updated", "version": 1}]
    assert "broadcaster_user_id" not in body


def test_create_subscription_plain_event_names(mock_urlopen, monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    kw.create_subscription(events=["chat.message.sent", "channel.followed"])
    body = _request_body(mock_urlopen[0])
    assert body["events"] == [{"name": "chat.message.sent", "version": 1},
                              {"name": "channel.followed", "version": 1}]


def test_subscribe_livestream_status_convenience(mock_urlopen, monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    result = kw.subscribe_livestream_status(broadcaster_user_id=777)
    assert result["ok"] is True
    body = _request_body(mock_urlopen[0])
    assert body["events"] == [{"name": "livestream.status.updated", "version": 1}]
    assert body["broadcaster_user_id"] == 777


def test_list_subscriptions_request_shape(mock_urlopen, monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    result = kw.list_subscriptions(broadcaster_user_id=999)
    assert result["ok"] is True
    req = mock_urlopen[0]
    assert req.get_method() == "GET"
    assert req.full_url == ("https://api.kick.com/public/v1/events/subscriptions"
                            "?broadcaster_user_id=999")


def test_list_subscriptions_no_filter(mock_urlopen, monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    kw.list_subscriptions()
    assert mock_urlopen[0].full_url == (
        "https://api.kick.com/public/v1/events/subscriptions")


def test_delete_subscription_request_shape(mock_urlopen, monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    result = kw.delete_subscription(subscription_id="sub-abc")
    assert result["ok"] is True
    req = mock_urlopen[0]
    assert req.get_method() == "DELETE"
    assert req.full_url == ("https://api.kick.com/public/v1/events/subscriptions"
                            "?id=sub-abc")


def test_delete_subscriptions_loops_ids(mock_urlopen, monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    results = kw.delete_subscriptions(subscription_ids=["a", "b"])
    assert len(results) == 2
    assert len(mock_urlopen) == 2


def test_subscription_calls_not_configured_without_token(monkeypatch):
    monkeypatch.delenv("KICK_ACCESS_TOKEN", raising=False)
    assert kw.create_subscription()["error"] == "not_configured"
    assert kw.list_subscriptions()["error"] == "not_configured"
    assert kw.delete_subscription(subscription_id="x")["error"] == "not_configured"


def test_delete_subscription_missing_id(monkeypatch):
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    assert kw.delete_subscription()["error"] == "missing subscription_id"


def test_api_http_error_is_reported_not_raised(monkeypatch):
    def _raising_urlopen(req, context=None, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(kw.urllib.request, "urlopen", _raising_urlopen)
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    result = kw.list_subscriptions()
    assert result["ok"] is False
    assert result["status"] == 401


def test_api_network_error_is_reported_not_raised(monkeypatch):
    def _raising_urlopen(req, context=None, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(kw.urllib.request, "urlopen", _raising_urlopen)
    monkeypatch.setenv("KICK_ACCESS_TOKEN", "tok-123")
    result = kw.list_subscriptions()
    assert result["ok"] is False
    assert "network down" in result["error"]


# ----------------------------------------------------------------------
# Event handling (no FastAPI needed)
# ----------------------------------------------------------------------
class FakeMonitor:
    def __init__(self):
        self.started = []
        self.ended = []
        self.streamer_cache = {
            "xqc": {"channel_id": 1001, "chatroom_id": 2001},
        }

    def handle_stream_started(self, username):
        self.started.append(username)
        return True

    def handle_stream_ended(self, username):
        self.ended.append(username)
        return True


def test_handle_livestream_started_by_login():
    monitor = FakeMonitor()
    result = kw.handle_livestream_status_event(
        {"broadcaster_user_login": "XQC", "is_live": True}, monitor=monitor)
    assert result == {"handled": True, "action": "stream_started",
                      "streamer": "xqc"}
    assert monitor.started == ["xqc"]


def test_handle_livestream_ended_by_login():
    monitor = FakeMonitor()
    result = kw.handle_livestream_status_event(
        {"broadcaster_user_login": "xqc", "is_live": False}, monitor=monitor)
    assert result["handled"] is True
    assert result["action"] == "stream_ended"
    assert monitor.ended == ["xqc"]


def test_handle_livestream_resolves_via_channel_id():
    monitor = FakeMonitor()
    result = kw.handle_livestream_status_event(
        {"broadcaster_user_id": 1001, "is_live": True}, monitor=monitor)
    assert result["streamer"] == "xqc"
    assert monitor.started == ["xqc"]


def test_handle_livestream_unknown_broadcaster():
    monitor = FakeMonitor()
    result = kw.handle_livestream_status_event(
        {"broadcaster_user_id": 424242, "is_live": True}, monitor=monitor)
    assert result["handled"] is False
    assert result["reason"] == "unknown_broadcaster"
    assert monitor.started == []


def test_handle_livestream_no_monitor(monkeypatch):
    kw.set_monitor(None)
    result = kw.handle_livestream_status_event(
        {"broadcaster_user_login": "xqc", "is_live": True}, monitor=None)
    assert result["handled"] is False
    assert result["reason"] == "monitor_not_configured"


def test_handle_livestream_missing_is_live():
    monitor = FakeMonitor()
    result = kw.handle_livestream_status_event(
        {"broadcaster_user_login": "xqc"}, monitor=monitor)
    assert result["handled"] is False
    assert result["reason"] == "missing is_live"


def test_set_get_monitor_roundtrip():
    monitor = FakeMonitor()
    kw.set_monitor(monitor)
    try:
        assert kw.get_monitor() is monitor
    finally:
        kw.set_monitor(None)


# ----------------------------------------------------------------------
# FastAPI route handler (called directly with a fake Request)
# ----------------------------------------------------------------------
class FakeRequest:
    def __init__(self, body: bytes, headers: dict):
        self._body = body
        self.headers = headers

    async def body(self):
        return self._body


def _route_handler():
    for method, path, handler in kw.router.routes:
        if method == "POST" and path == "/api/webhooks/kick":
            return handler
    raise AssertionError("POST /api/webhooks/kick route not registered")


@pytest.fixture(autouse=True)
def _clean_webhook_state(monkeypatch):
    # Route tests sign with the TEST key; point the module at it.
    monkeypatch.setenv("KICK_WEBHOOK_SECRET", TEST_PUBLIC_KEY_PEM)
    kw._seen_message_ids.clear()
    kw.set_monitor(None)
    yield
    kw._seen_message_ids.clear()
    kw.set_monitor(None)


def _deliver(payload_dict, event_type, message_id="msg-1"):
    body = json.dumps(payload_dict).encode("utf-8")
    headers = _signed_headers(message_id, "1700000000", body)
    headers["kick-event-type"] = event_type
    req = FakeRequest(body, headers)
    return asyncio.run(_route_handler()(req))


def test_route_registers_post_webhook_endpoint():
    assert _route_handler() is not None


def test_route_livestream_started_calls_monitor():
    monitor = FakeMonitor()
    kw.set_monitor(monitor)
    resp = _deliver({"broadcaster_user_login": "xqc", "is_live": True},
                    "livestream.status.updated")
    assert resp.status_code == 200
    assert resp.content["received"] is True
    assert resp.content["result"]["action"] == "stream_started"
    assert monitor.started == ["xqc"]


def test_route_livestream_ended_calls_monitor():
    monitor = FakeMonitor()
    kw.set_monitor(monitor)
    resp = _deliver({"broadcaster_user_login": "xqc", "is_live": False},
                    "livestream.status.updated")
    assert resp.status_code == 200
    assert monitor.ended == ["xqc"]


def test_route_invalid_signature_returns_403():
    monitor = FakeMonitor()
    kw.set_monitor(monitor)
    body = json.dumps({"broadcaster_user_login": "xqc",
                       "is_live": True}).encode("utf-8")
    headers = _signed_headers("msg-1", "1700000000", body)
    headers["kick-event-type"] = "livestream.status.updated"
    headers["kick-event-signature"] = base64.b64encode(b"0" * 256).decode()
    resp = asyncio.run(_route_handler()(FakeRequest(body, headers)))
    assert resp.status_code == 403
    assert monitor.started == []


def test_route_missing_signature_returns_403():
    body = json.dumps({"is_live": True}).encode("utf-8")
    resp = asyncio.run(_route_handler()(
        FakeRequest(body, {"kick-event-type": "livestream.status.updated"})))
    assert resp.status_code == 403


def test_route_duplicate_delivery_is_idempotent():
    monitor = FakeMonitor()
    kw.set_monitor(monitor)
    payload = {"broadcaster_user_login": "xqc", "is_live": True}
    first = _deliver(payload, "livestream.status.updated", message_id="dup-1")
    second = _deliver(payload, "livestream.status.updated", message_id="dup-1")
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.content.get("duplicate") is True
    assert monitor.started == ["xqc"]  # handled exactly once


def test_route_unknown_event_is_acknowledged():
    monitor = FakeMonitor()
    kw.set_monitor(monitor)
    resp = _deliver({"broadcaster_user_login": "xqc"},
                    "channel.followed")
    assert resp.status_code == 200
    assert resp.content["received"] is True
    assert monitor.started == [] and monitor.ended == []


def test_route_invalid_json_returns_400():
    body = b"{not json"
    headers = _signed_headers("msg-1", "1700000000", body)
    headers["kick-event-type"] = "livestream.status.updated"
    resp = asyncio.run(_route_handler()(FakeRequest(body, headers)))
    assert resp.status_code == 400


def test_route_nested_data_payload_is_tolerated():
    monitor = FakeMonitor()
    kw.set_monitor(monitor)
    resp = _deliver(
        {"data": {"broadcaster_user_login": "xqc", "is_live": True}},
        "livestream.status.updated")
    assert resp.status_code == 200
    assert monitor.started == ["xqc"]


def test_route_without_monitor_still_returns_200():
    resp = _deliver({"broadcaster_user_login": "xqc", "is_live": True},
                    "livestream.status.updated")
    assert resp.status_code == 200
    assert resp.content["result"]["handled"] is False


# ----------------------------------------------------------------------
# KickMonitor webhook hooks (kick_monitor.py is Worker A's file to edit)
# ----------------------------------------------------------------------
def test_monitor_ssl_is_verified():
    import ssl
    import kick_monitor
    monitor = kick_monitor.KickMonitor()
    assert monitor.ssl_ctx.verify_mode == ssl.CERT_REQUIRED
    assert monitor.ssl_ctx.check_hostname is True


def test_monitor_handle_stream_started(monkeypatch):
    import kick_monitor

    monitor = kick_monitor.KickMonitor()
    calls = {}

    def _fake_fetch(username):
        calls["fetch"] = username
        return {"username": username, "is_live": 1, "title": "t",
                "category": "Gaming", "viewer_count": 5,
                "playback_url": "https://example.com/stream.m3u8"}

    def _fake_start(username, playback_url):
        calls["start"] = (username, playback_url)

    monkeypatch.setattr(monitor, "fetch_channel_info", _fake_fetch)
    monkeypatch.setattr(monitor, "start_rolling_buffer", _fake_start)
    assert monitor.handle_stream_started("XQC") is True
    assert calls["fetch"] == "xqc"
    assert calls["start"] == ("xqc", "https://example.com/stream.m3u8")
    assert "xqc" in monitor.known_live


def test_monitor_handle_stream_started_api_disagrees(monkeypatch):
    import kick_monitor

    monitor = kick_monitor.KickMonitor()
    monkeypatch.setattr(
        monitor, "fetch_channel_info",
        lambda username: {"username": username, "is_live": 0,
                          "playback_url": None})
    started = []
    monkeypatch.setattr(monitor, "start_rolling_buffer",
                        lambda u, p: started.append(u))
    assert monitor.handle_stream_started("xqc") is False
    assert started == []
    assert "xqc" not in monitor.known_live


def test_monitor_handle_stream_ended(monkeypatch):
    import kick_monitor

    monitor = kick_monitor.KickMonitor()
    monitor.known_live.add("xqc")
    stopped = []
    monkeypatch.setattr(monitor, "stop_recorder",
                        lambda u: stopped.append(u))
    assert monitor.handle_stream_ended("xqc") is True
    assert "xqc" not in monitor.known_live
    assert stopped == ["xqc"]
