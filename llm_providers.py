"""
llm_providers.py — OpenAI-compatible LLM client (stdlib only) + provider fallback.

Fallback chain (the multi-provider idea from opensource-clipping, MIT licence,
clipping/providers/ — Gemini, NVIDIA NIM, local vLLM):

    Gemini -> OpenAI-compatible endpoint -> heuristic fallback

The OpenAI-compatible step talks to any server exposing the
``POST /chat/completions`` API: LM Studio (default, http://localhost:1234/v1),
a local vLLM server, NVIDIA NIM, or any other OpenAI-compatible endpoint.
It uses ONLY the standard library (urllib) — no new dependency.

Env vars:
    OPENAI_BASE_URL  Base URL of the OpenAI-compatible server
                     (default http://localhost:1234/v1 for LM Studio).
    OPENAI_API_KEY   API key, sent as a Bearer token
                     (default "lm-studio"; local servers usually ignore it).
    OPENAI_MODEL     Model name to request (default "local-model").

The coordinator wires generate_metadata_with_fallback() into
AIStoryEngine.analyze_story_and_generate_metadata by wrapping the existing
Gemini call as ``gemini_fn`` and the existing heuristic block as
``heuristic_fn`` — both take the prompt string and return a metadata dict.
"""

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "local-model"

SYSTEM_PROMPT = (
    "You are a professional video editor and content strategist. "
    "Return JSON only."
)


class _ResponseFormatRejected(Exception):
    """Raised when the server rejects response_format (retry without it)."""


def strip_code_fences(text: str) -> str:
    """Remove surrounding ``` or ```json fences from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop the opening fence (``` or ```json)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop the closing fence
        text = "\n".join(lines).strip()
    return text


def parse_json_response(text: str) -> dict:
    """
    Parse an LLM reply into a dict: strip code fences, json.loads, and as a
    last resort extract the substring between the first '{' and last '}'.
    Raises ValueError when no JSON object can be recovered.
    """
    cleaned = strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("LLM response contains no JSON object")
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON is not an object")
    return data


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client using only stdlib urllib."""

    def __init__(self, base_url=None, api_key=None, model=None, timeout=120):
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.timeout = timeout

    def generate_json(self, prompt, temperature=0.5, max_tokens=4096) -> dict:
        """
        Ask the model for a JSON object. Requests ``response_format``
        json_object when supported; if the server rejects it (HTTP 400),
        retries once without it and parses the reply leniently.
        Raises RuntimeError / ValueError on transport or parse failure.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            return self._post(url, payload, expect_json_format=True)
        except _ResponseFormatRejected:
            logger.info("Server rejected response_format=json_object; retrying without it")
            payload.pop("response_format", None)
            return self._post(url, payload, expect_json_format=False)

    def _post(self, url, payload, expect_json_format: bool) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 400 and expect_json_format:
                raise _ResponseFormatRejected(
                    "Server rejected response_format=json_object"
                ) from e
            raise RuntimeError(
                f"OpenAI-compatible request failed: HTTP {e.code}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"OpenAI-compatible request failed: {e.reason}"
            ) from e
        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI-compatible response shape: {exc}"
            ) from exc
        return parse_json_response(content)


def generate_metadata_with_fallback(prompt, gemini_fn, heuristic_fn, client=None) -> dict:
    """
    Generate clip metadata through the provider chain:

        1. gemini_fn(prompt)            — primary (Gemini)
        2. client.generate_json(prompt) — OpenAI-compatible fallback
                                           (LM Studio / vLLM / NIM)
        3. heuristic_fn(prompt)         — always-available local fallback

    ``gemini_fn`` and ``heuristic_fn`` are callables taking the prompt string
    and returning a metadata dict (the coordinator adapts the existing
    ai_story_engine code to this shape). ``client`` defaults to a new
    OpenAICompatibleClient built from the environment.

    Returns the winning provider's dict with an added "llm_provider" key
    ("gemini", "openai_compatible" or "heuristic"); the provider used is
    also logged. An empty/non-dict Gemini result counts as a failure and
    falls through to the next provider.
    """
    if gemini_fn is not None:
        try:
            data = gemini_fn(prompt)
            if isinstance(data, dict) and data:
                data = dict(data)
                data["llm_provider"] = "gemini"
                logger.info("LLM metadata via gemini")
                return data
            logger.warning("Gemini returned empty/non-dict metadata; trying next provider")
        except Exception as exc:
            logger.warning("Gemini metadata failed (%s); trying OpenAI-compatible provider", exc)

    try:
        client = client or OpenAICompatibleClient()
        data = client.generate_json(prompt)
        data = dict(data)
        data["llm_provider"] = "openai_compatible"
        logger.info("LLM metadata via openai_compatible (%s)", client.base_url)
        return data
    except Exception as exc:
        logger.warning("OpenAI-compatible metadata failed (%s); using heuristic fallback", exc)

    data = heuristic_fn(prompt)
    data = dict(data)
    data["llm_provider"] = "heuristic"
    logger.info("LLM metadata via heuristic fallback")
    return data
