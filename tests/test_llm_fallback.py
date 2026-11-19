"""Tests for llm_providers.py — HTTP is mocked, no server needed."""
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_providers import (  # noqa: E402
    OpenAICompatibleClient,
    generate_metadata_with_fallback,
    parse_json_response,
    strip_code_fences,
)


def _chat_response(content):
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }).encode("utf-8")


class FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def read(self, *args, **kwargs):
        return self._buf.read(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------- parsing

def test_strip_code_fences_json_block():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_parse_json_response_plain_and_fenced():
    assert parse_json_response('{"a": 1}') == {"a": 1}
    assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_response_extracts_object_from_prose():
    assert parse_json_response('Here it is: {"a": 1} hope that helps') == {"a": 1}


def test_parse_json_response_rejects_garbage():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_response("no json here at all")
    with pytest.raises(ValueError, match="not an object"):
        parse_json_response("[1, 2, 3]")


# ---------------------------------------------------------------- client

def test_client_generate_json_success(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode())
        seen["auth"] = req.get_header("Authorization")
        return FakeHTTPResponse(_chat_response('{"category": "Big Wins", "ai_score": 90}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient(base_url="http://localhost:1234/v1",
                                    api_key="lm-studio", model="local-model")
    result = client.generate_json("analyse this")
    assert result == {"category": "Big Wins", "ai_score": 90}
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    assert seen["auth"] == "Bearer lm-studio"
    assert seen["payload"]["model"] == "local-model"
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["messages"][1]["content"] == "analyse this"


def test_client_reads_env_config(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gpu-box:8001/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key")
    monkeypatch.setenv("OPENAI_MODEL", "gemma-4")
    client = OpenAICompatibleClient()
    assert client.base_url == "http://gpu-box:8001/v1"
    assert client.api_key == "secret-key"
    assert client.model == "gemma-4"


def test_client_defaults_to_lm_studio(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    client = OpenAICompatibleClient()
    assert client.base_url == "http://localhost:1234/v1"
    assert client.api_key == "lm-studio"
    assert client.model == "local-model"


def test_client_retries_without_response_format_on_400(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode()))
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {}, io.BytesIO(b""))
        return FakeHTTPResponse(_chat_response('```json\n{"ok": true}\n```'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient()
    assert client.generate_json("hi") == {"ok": True}
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_client_http_error_raises_runtime_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Server Error", {}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 500"):
        OpenAICompatibleClient().generate_json("hi")


def test_client_connection_error_raises_runtime_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="connection refused"):
        OpenAICompatibleClient().generate_json("hi")


# ---------------------------------------------------------------- fallback chain

class FakeClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.base_url = "http://fake:1234/v1"
        self.calls = 0

    def generate_json(self, prompt):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_fallback_gemini_wins():
    calls = {"openai": 0}

    def gemini_fn(prompt):
        return {"category": "Big Wins"}

    def heuristic_fn(prompt):
        raise AssertionError("heuristic must not run")

    client = FakeClient(result={"category": "Funny Moments"})
    result = generate_metadata_with_fallback(
        "prompt", gemini_fn, heuristic_fn, client=client)
    assert result == {"category": "Big Wins", "llm_provider": "gemini"}
    assert client.calls == 0


def test_fallback_gemini_failure_uses_openai_compatible():
    def gemini_fn(prompt):
        raise RuntimeError("gemini down")

    def heuristic_fn(prompt):
        raise AssertionError("heuristic must not run")

    client = FakeClient(result={"category": "Funny Moments"})
    result = generate_metadata_with_fallback(
        "prompt", gemini_fn, heuristic_fn, client=client)
    assert result == {"category": "Funny Moments", "llm_provider": "openai_compatible"}


def test_fallback_empty_gemini_result_falls_through():
    def gemini_fn(prompt):
        return {}

    client = FakeClient(result={"category": "Funny Moments"})
    result = generate_metadata_with_fallback(
        "prompt", gemini_fn, lambda p: {"category": "Viral Potential"}, client=client)
    assert result["llm_provider"] == "openai_compatible"


def test_fallback_both_llms_down_uses_heuristic():
    def gemini_fn(prompt):
        raise RuntimeError("gemini down")

    def heuristic_fn(prompt):
        return {"category": "Viral Potential", "ai_score": 70}

    client = FakeClient(error=RuntimeError("local server down"))
    result = generate_metadata_with_fallback(
        "prompt", gemini_fn, heuristic_fn, client=client)
    assert result == {"category": "Viral Potential", "ai_score": 70,
                      "llm_provider": "heuristic"}


def test_fallback_no_gemini_fn_goes_straight_to_openai():
    client = FakeClient(result={"category": "Big Wins"})
    result = generate_metadata_with_fallback(
        "prompt", None, lambda p: {"category": "Viral Potential"}, client=client)
    assert result["llm_provider"] == "openai_compatible"
