"""
Kick webhooks (official public API) for instant live detection.

Tier 1, feature 3 (see ai-social-studio-feature-suggestions.md). Polling
stays as the fallback; webhooks make recording start the second a stream
goes live instead of up to ~23 s later.

Verified 2026-11-19 against Kick's public API documentation (via
docs.kick.com summaries), the kick-api Rust crate, kicksdk, KickLib and a
reference Node implementation:

* Base URL: ``https://api.kick.com/public/v1``
* OAuth 2.1 + PKCE at ``https://id.kick.com/oauth/authorize`` and
  ``https://id.kick.com/oauth/token``. Event subscriptions need a **user**
  access token with the ``events:subscribe`` scope (an app access token is
  not enough -- events belong to a broadcaster). Complete the OAuth flow
  once in a browser and store the token as ``KICK_ACCESS_TOKEN``.
* Create: ``POST /events/subscriptions`` with body::

      {"broadcaster_user_id": 12345,           # optional
       "method": "webhook",
       "events": [{"name": "livestream.status.updated", "version": 1}]}

  Each event gets its own subscription id in the response.
* List: ``GET /events/subscriptions`` (optional ``?broadcaster_user_id=``).
* Delete: ``DELETE /events/subscriptions?id=<subscription_id>``
  (one request per id).
* Events relevant here: ``livestream.status.updated`` (v1, stream goes
  live/offline), ``chat.message.sent`` (v1), ``channel.followed`` (v1),
  ``channel.subscription.new`` (v1). This module acts on
  ``livestream.status.updated`` and acknowledges everything else.
* Delivery: Kick POSTs to the public HTTPS URL configured in the Kick
  developer settings (https://kick.com/settings/developer) -- the URL is
  *not* part of the subscribe call. ``KICK_WEBHOOK_URL`` is read here so
  the app can report what it expects to be configured. Kick retries failed
  deliveries 3 times and then auto-unsubscribes, so the endpoint always
  answers 200 once the signature checks out, even for unknown events.

Signature verification -- READ THIS, it is not HMAC:
Kick signs each delivery with RSA-SHA256 (PKCS#1 v1.5) over the string::

    <Kick-Event-Message-Id>.<Kick-Event-Message-Timestamp>.<raw body>

The base64 signature is in the ``Kick-Event-Signature`` header and is
verified against Kick's *published RSA public key* (below). There is no
shared HMAC secret in Kick's scheme, so ``KICK_WEBHOOK_SECRET`` is treated
as an **optional PEM override for the public key** (for key rotation), not
as an HMAC key. Verification is implemented in pure stdlib (``hashlib`` +
``pow``) so no extra crypto dependency is needed; it is cross-checked
against OpenSSL in the test suite.

Headers on every delivery: ``Kick-Event-Message-Id``,
``Kick-Event-Subscription-Id``, ``Kick-Event-Signature``,
``Kick-Event-Message-Timestamp``, ``Kick-Event-Type``, ``Kick-Event-Version``.
``Kick-Event-Message-Id`` is used as an idempotency key (Kick may deliver
the same event more than once).

Wiring (the dashboard owns the monitor singleton; this module does not
import it, to avoid import cycles)::

    from kick_webhooks import router as kick_webhook_router, set_monitor
    set_monitor(monitor)                    # the KickMonitor instance
    app.include_router(kick_webhook_router) # mounts POST /api/webhooks/kick

When nothing is configured, every function below is a clean no-op that
returns an ``{"ok": False, "error": "not_configured", ...}`` dict or logs
and returns 200 -- the app keeps working on polling alone.

Environment variables:
    KICK_CLIENT_ID       OAuth app id (from kick.com/settings/developer)
    KICK_CLIENT_SECRET   OAuth app secret (kept for the OAuth flow; the
                         subscription calls below use KICK_ACCESS_TOKEN)
    KICK_ACCESS_TOKEN    user OAuth token with events:subscribe scope
    KICK_WEBHOOK_URL     public HTTPS URL configured in Kick dev settings
    KICK_WEBHOOK_SECRET  optional PEM override for Kick's public key
                         (NOT an HMAC secret -- see above)
"""

import base64
import binascii
import hashlib
import hmac
import json
import os
import ssl
import threading
import urllib.error
import urllib.parse
import urllib.request

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from database import log_event

KICK_API_BASE = "https://api.kick.com/public/v1"
KICK_OAUTH_AUTHORIZE_URL = "https://id.kick.com/oauth/authorize"
KICK_OAUTH_TOKEN_URL = "https://id.kick.com/oauth/token"

EVENT_LIVESTREAM_STATUS = "livestream.status.updated"
EVENT_CHAT_MESSAGE = "chat.message.sent"
EVENT_CHANNEL_FOLLOWED = "channel.followed"

# Kick's published RSA public key for webhook signature verification.
# This is a PUBLIC key (published by Kick for everyone to verify with); it
# is not a secret. Override via KICK_WEBHOOK_SECRET (PEM) if Kick rotates it.
KICK_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAq/+l1WnlRrGSolDMA+A8
6rAhMbQGmQ2SapVcGM3zq8ANXjnhDWocMqfWcTd95btDydITa10kDvHzw9WQOqp2
MZI7ZyrfzJuz5nhTPCiJwTwnEtWft7nV14BYRDHvlfqPUaZ+1KR4OCaO/wWIk/rQ
L/TjY0M70gse8rlBkbo2a8rKhu69RQTRsoaf4DVhDPEeSeI5jVrRDGAMGL3cGuyY
6CLKGdjVEM78g3JfYOvDU/RvfqD7L89TZ3iN94jrmWdGz34JNlEI5hqK8dd7C5EF
BEbZ5jgB8s8ReQV8H+MkuffjdAj3ajDDX3DOJMIut1lBrUVD1AaSrGCKHooWoL2e
twIDAQAB
-----END PUBLIC KEY-----"""

# DER prefix of the SHA-256 DigestInfo used in PKCS#1 v1.5 signatures.
_SHA256_DIGESTINFO_PREFIX = bytes.fromhex(
    "3031300d060960864801650304020105000420")

# Module state (wired by the dashboard at startup).
_monitor = None
_seen_message_ids = set()
_seen_lock = threading.Lock()
_MAX_SEEN_IDS = 5000


# ----------------------------------------------------------------------
# Pure-stdlib RSA-SHA256 (PKCS#1 v1.5) verification
# ----------------------------------------------------------------------
def _der_read_length(data, pos):
    """Read a DER length field at *pos*; return (length, new_pos)."""
    first = data[pos]
    pos += 1
    if first < 0x80:
        return first, pos
    num_bytes = first & 0x7F
    if num_bytes == 0 or num_bytes > 4:
        raise ValueError("unsupported DER length encoding")
    length = int.from_bytes(data[pos:pos + num_bytes], "big")
    return length, pos + num_bytes


def _der_read_tlv(data, pos):
    """Read one DER TLV at *pos*; return (tag, content_bytes, next_pos)."""
    tag = data[pos]
    length, pos = _der_read_length(data, pos + 1)
    end = pos + length
    if end > len(data):
        raise ValueError("truncated DER data")
    return tag, data[pos:end], end


def _parse_rsa_public_key_pem(pem_text):
    """Parse a PEM SubjectPublicKeyInfo block -> (modulus, exponent) ints."""
    lines = []
    in_block = False
    for line in pem_text.strip().splitlines():
        line = line.strip()
        if line.startswith("-----BEGIN"):
            in_block = True
            continue
        if line.startswith("-----END"):
            break
        if in_block and line:
            lines.append(line)
    if not lines:
        raise ValueError("no PEM body found")
    try:
        der = base64.b64decode("".join(lines))
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"invalid PEM base64: {e}")

    # SubjectPublicKeyInfo: SEQUENCE { SEQUENCE { OID, NULL },
    #                                 BIT STRING { RSAPublicKey } }
    tag, spki, _ = _der_read_tlv(der, 0)
    if tag != 0x30:
        raise ValueError("expected SEQUENCE (SubjectPublicKeyInfo)")
    _, _, pos = _der_read_tlv(spki, 0)  # skip algorithm identifier
    tag, bitstring, _ = _der_read_tlv(spki, pos)
    if tag != 0x03 or not bitstring or bitstring[0] != 0:
        raise ValueError("expected BIT STRING with 0 unused bits")
    tag, rsapub, _ = _der_read_tlv(bitstring, 1)
    if tag != 0x30:
        raise ValueError("expected SEQUENCE (RSAPublicKey)")
    tag, modulus_bytes, pos = _der_read_tlv(rsapub, 0)
    if tag != 0x02:
        raise ValueError("expected INTEGER (modulus)")
    tag, exponent_bytes, _ = _der_read_tlv(rsapub, pos)
    if tag != 0x02:
        raise ValueError("expected INTEGER (exponent)")
    return (int.from_bytes(modulus_bytes, "big"),
            int.from_bytes(exponent_bytes, "big"))


def _rsa_pkcs1v15_verify_sha256(modulus, exponent, message, signature):
    """Verify an RSA PKCS#1 v1.5 SHA-256 signature (pure stdlib)."""
    key_bytes = (modulus.bit_length() + 7) // 8
    if len(signature) != key_bytes:
        return False
    recovered = pow(int.from_bytes(signature, "big"), exponent, modulus)
    encoded = recovered.to_bytes(key_bytes, "big")
    digest = hashlib.sha256(message).digest()
    t = _SHA256_DIGESTINFO_PREFIX + digest
    if key_bytes < len(t) + 11:
        return False
    expected = b"\x00\x01" + b"\xff" * (key_bytes - len(t) - 3) + b"\x00" + t
    return hmac.compare_digest(encoded, expected)


def get_webhook_public_key_pem():
    """Public key used to verify Kick signatures.

    ``KICK_WEBHOOK_SECRET`` may hold a replacement PEM (Kick key rotation);
    anything that is not a PEM is ignored and the built-in key is used.
    """
    override = os.getenv("KICK_WEBHOOK_SECRET", "").strip()
    if override and "BEGIN PUBLIC KEY" in override:
        return override
    return KICK_PUBLIC_KEY_PEM


def verify_kick_signature(message_id, timestamp, raw_body, signature_b64,
                          public_key_pem=None):
    """Verify a Kick webhook delivery.

    ``raw_body`` must be the exact request bytes (before any JSON parsing).
    Returns False for missing headers, bad base64, bad key material, or a
    signature that does not match -- never raises.
    """
    if not message_id or not timestamp or not signature_b64:
        return False
    if isinstance(raw_body, str):
        raw_body = raw_body.encode("utf-8")
    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except (binascii.Error, ValueError):
        return False
    pem = public_key_pem or get_webhook_public_key_pem()
    try:
        modulus, exponent = _parse_rsa_public_key_pem(pem)
    except ValueError as e:
        log_event("ERROR", "KickWebhooks",
                  f"Cannot parse Kick webhook public key: {e}")
        return False
    signed_payload = f"{message_id}.{timestamp}.".encode("utf-8") + raw_body
    return _rsa_pkcs1v15_verify_sha256(
        modulus, exponent, signed_payload, signature)


# ----------------------------------------------------------------------
# Configuration helpers
# ----------------------------------------------------------------------
def get_access_token():
    """User OAuth token for subscription management (env KICK_ACCESS_TOKEN)."""
    return os.getenv("KICK_ACCESS_TOKEN", "").strip()


def webhooks_configured():
    """True when the minimum webhook setup is present in env vars."""
    return bool(
        os.getenv("KICK_CLIENT_ID", "").strip()
        and get_access_token()
        and os.getenv("KICK_WEBHOOK_URL", "").strip()
    )


def get_setup_checklist():
    """Human-readable setup steps (used by docs / the dashboard later)."""
    return [
        "1. Create an app at kick.com/settings/developer (note CLIENT_ID and CLIENT_SECRET).",
        "2. In the app settings, enable webhooks and set the public HTTPS webhook URL "
        "to KICK_WEBHOOK_URL (must be public: ngrok or a VPS -- localhost never receives deliveries).",
        "3. Complete Kick OAuth 2.1 (PKCE) once in a browser with scope 'events:subscribe' "
        "and store the user access token as KICK_ACCESS_TOKEN.",
        "4. Call create_subscription() / subscribe_livestream_status() for each broadcaster "
        "user id to watch, or use list_subscriptions() to inspect what is active.",
        "5. Mount this module's router in the FastAPI app and call set_monitor(monitor) "
        "so livestream events start/stop recording.",
    ]


# ----------------------------------------------------------------------
# Subscription management (stdlib urllib, no requests dependency)
# ----------------------------------------------------------------------
def _api_request(method, path, access_token, body=None, timeout=15):
    """One authenticated Kick public API call. Returns a result dict and
    never raises for HTTP or network errors."""
    url = KICK_API_BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    ctx = ssl.create_default_context()  # verified TLS, unlike the old monitor
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8") or "{}"
            return {"ok": True, "status": resp.status,
                    "data": json.loads(raw)}
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode("utf-8") or "{}")
        except Exception:
            payload = {"message": str(e)}
        log_event("WARNING", "KickWebhooks",
                  f"Kick API {method} {path} failed ({e.code}): {payload}")
        return {"ok": False, "status": e.code, "error": payload}
    except Exception as e:
        log_event("WARNING", "KickWebhooks",
                  f"Kick API {method} {path} error: {e}")
        return {"ok": False, "status": None, "error": str(e)}


def create_subscription(access_token=None, events=None,
                        broadcaster_user_id=None):
    """Subscribe to Kick events via webhook.

    *events* is a list of ``{"name": ..., "version": 1}`` dicts (or plain
    names, which default to version 1). Defaults to
    ``livestream.status.updated`` v1. ``broadcaster_user_id`` scopes the
    subscription to one broadcaster's channel (the numeric user id from the
    public API / v2 channel payload). Returns the parsed API response, or
    ``{"ok": False, "error": "not_configured", ...}`` when no token exists.
    """
    token = access_token or get_access_token()
    if not token:
        return {"ok": False, "error": "not_configured",
                "detail": "Set KICK_ACCESS_TOKEN (user OAuth token with "
                          "events:subscribe scope) to manage subscriptions."}
    events = events or [{"name": EVENT_LIVESTREAM_STATUS, "version": 1}]
    normalised = [
        {"name": ev, "version": 1} if isinstance(ev, str) else
        {"name": ev["name"], "version": ev.get("version", 1)}
        for ev in events
    ]
    body = {"method": "webhook", "events": normalised}
    if broadcaster_user_id is not None:
        body["broadcaster_user_id"] = int(broadcaster_user_id)
    result = _api_request("POST", "/events/subscriptions", token, body)
    if result.get("ok"):
        log_event("INFO", "KickWebhooks",
                  f"Subscribed to {[e['name'] for e in normalised]} "
                  f"(broadcaster={broadcaster_user_id})")
    return result


def subscribe_livestream_status(access_token=None, broadcaster_user_id=None):
    """Convenience wrapper: subscribe to livestream.status.updated (v1)."""
    return create_subscription(
        access_token=access_token,
        events=[{"name": EVENT_LIVESTREAM_STATUS, "version": 1}],
        broadcaster_user_id=broadcaster_user_id,
    )


def list_subscriptions(access_token=None, broadcaster_user_id=None):
    """List active event subscriptions, optionally for one broadcaster."""
    token = access_token or get_access_token()
    if not token:
        return {"ok": False, "error": "not_configured",
                "detail": "Set KICK_ACCESS_TOKEN to list subscriptions."}
    path = "/events/subscriptions"
    if broadcaster_user_id is not None:
        path += "?" + urllib.parse.urlencode(
            {"broadcaster_user_id": int(broadcaster_user_id)})
    return _api_request("GET", path, token)


def delete_subscription(access_token=None, subscription_id=None):
    """Delete one event subscription by its id."""
    token = access_token or get_access_token()
    if not token:
        return {"ok": False, "error": "not_configured",
                "detail": "Set KICK_ACCESS_TOKEN to delete subscriptions."}
    if not subscription_id:
        return {"ok": False, "error": "missing subscription_id"}
    path = "/events/subscriptions?" + urllib.parse.urlencode(
        {"id": str(subscription_id)})
    result = _api_request("DELETE", path, token)
    if result.get("ok"):
        log_event("INFO", "KickWebhooks",
                  f"Deleted event subscription {subscription_id}")
    return result


def delete_subscriptions(access_token=None, subscription_ids=None):
    """Delete several event subscriptions (one DELETE per id)."""
    results = []
    for sub_id in (subscription_ids or []):
        results.append(delete_subscription(access_token, sub_id))
    return results


# ----------------------------------------------------------------------
# FastAPI receiver
# ----------------------------------------------------------------------
def set_monitor(monitor):
    """Wire the KickMonitor instance livestream events should drive."""
    global _monitor
    _monitor = monitor


def get_monitor():
    return _monitor


def _resolve_username(payload, monitor):
    """Best-effort streamer username for a webhook payload.

    Prefers ``broadcaster_user_login``; falls back to matching
    ``broadcaster_user_id`` against the monitor's channel cache.
    """
    login = payload.get("broadcaster_user_login")
    if login:
        return str(login).lower()
    broadcaster_id = payload.get("broadcaster_user_id")
    if broadcaster_id is not None and monitor is not None:
        cache = getattr(monitor, "streamer_cache", {}) or {}
        for username, info in cache.items():
            if isinstance(info, dict) and info.get("channel_id") == broadcaster_id:
                return str(username).lower()
    return None


def handle_livestream_status_event(payload, monitor=None):
    """Act on a livestream.status.updated payload.

    Kept separate from the HTTP handler so it is unit-testable without
    FastAPI. Returns a dict describing what happened.
    """
    mon = monitor if monitor is not None else _monitor
    username = _resolve_username(payload, mon)
    if username is None:
        log_event("WARNING", "KickWebhooks",
                  "livestream.status.updated without a resolvable broadcaster: "
                  f"{str(payload)[:200]}")
        return {"handled": False, "reason": "unknown_broadcaster"}
    if mon is None:
        log_event("WARNING", "KickWebhooks",
                  f"No monitor wired; ignoring livestream event for {username}")
        return {"handled": False, "reason": "monitor_not_configured",
                "streamer": username}
    is_live = payload.get("is_live")
    if is_live is True:
        mon.handle_stream_started(username)
        log_event("INFO", "KickWebhooks",
                  f"Webhook: {username} went LIVE, recording started")
        return {"handled": True, "action": "stream_started",
                "streamer": username}
    if is_live is False:
        mon.handle_stream_ended(username)
        log_event("INFO", "KickWebhooks",
                  f"Webhook: {username} went offline, recording stopped")
        return {"handled": True, "action": "stream_ended",
                "streamer": username}
    log_event("WARNING", "KickWebhooks",
              f"livestream.status.updated for {username} without is_live")
    return {"handled": False, "reason": "missing is_live",
            "streamer": username}


def _is_duplicate_delivery(message_id):
    """Idempotency: True if this message id was already processed."""
    if not message_id:
        return False
    with _seen_lock:
        if message_id in _seen_message_ids:
            return True
        _seen_message_ids.add(message_id)
        if len(_seen_message_ids) > _MAX_SEEN_IDS:
            _seen_message_ids.clear()
            _seen_message_ids.add(message_id)
        return False


router = APIRouter()


@router.post("/api/webhooks/kick")
async def kick_webhook(request: Request):
    """Receive Kick event deliveries.

    Verifies the RSA signature, dedupes by message id, handles
    livestream.status.updated, and acknowledges everything else with 200
    (Kick auto-unsubscribes after repeated failures, so unknown events are
    logged, not rejected).
    """
    raw_body = await request.body()
    headers = request.headers
    message_id = headers.get("kick-event-message-id")
    timestamp = headers.get("kick-event-message-timestamp")
    signature = headers.get("kick-event-signature")
    event_type = headers.get("kick-event-type")

    if not verify_kick_signature(message_id, timestamp, raw_body, signature):
        log_event("WARNING", "KickWebhooks",
                  "Rejected webhook with missing/invalid signature")
        return JSONResponse(status_code=403,
                            content={"error": "invalid signature"})

    if _is_duplicate_delivery(message_id):
        return JSONResponse({"received": True, "duplicate": True})

    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(status_code=400, content={"error": "invalid json"})
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"error": "invalid json"})

    # Kick sends the event object at the top level; tolerate a nested
    # "data" wrapper in case the shape varies by event.
    if isinstance(payload.get("data"), dict) and \
            "broadcaster_user_id" not in payload:
        payload = payload["data"]

    event_type = event_type or payload.get("event") or payload.get("type") or ""
    if event_type == EVENT_LIVESTREAM_STATUS:
        result = handle_livestream_status_event(payload)
        return JSONResponse({"received": True, "result": result})

    log_event("INFO", "KickWebhooks",
              f"Ignored event type: {event_type or '(unknown)'}")
    return JSONResponse({"received": True})
