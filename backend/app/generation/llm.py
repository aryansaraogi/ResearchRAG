"""Gemini client with client-side rate limiting and retry on 429/5xx."""

from __future__ import annotations

import itertools
import json
import threading
import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import get_settings


class LLMNotConfigured(RuntimeError):
    pass


class _RateLimiter:
    """Spaces requests so we stay under the per-minute quota of the free tier."""

    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


def _quota_violations(e: errors.APIError) -> list[dict]:
    body = e.details if isinstance(e.details, dict) else {}
    return [v for d in body.get("error", {}).get("details", []) for v in d.get("violations", [])]


def _daily_quota_exhausted(e: BaseException) -> bool:
    """Free-tier per-day quotas are small (20 requests/day for some Flash models) and retrying can't help."""
    return (isinstance(e, errors.APIError) and e.code == 429
            and any("PerDay" in v.get("quotaId", "") for v in _quota_violations(e)))


def _retryable(e: BaseException) -> bool:
    if not isinstance(e, errors.APIError) or _daily_quota_exhausted(e):
        return False
    return e.code == 429 or (e.code or 0) >= 500


_retry = retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
# Someone is watching a chat answer: give up sooner (~30 s of backoff) than batch jobs do
_stream_retry = retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(multiplier=2, min=4, max=16),
    stop=stop_after_attempt(4),
    reraise=True,
)


def describe_error(e: BaseException) -> str:
    """A message fit for the UI; Gemini's raw errors are JSON dumps."""
    if isinstance(e, errors.APIError):
        if _daily_quota_exhausted(e):
            v = _quota_violations(e)[0]
            model = v.get("quotaDimensions", {}).get("model", "this model")
            limit = f" ({v['quotaValue']} requests/day)" if v.get("quotaValue") else ""
            return (f"Gemini's free daily quota for {model} is used up{limit}. It resets at midnight Pacific time. "
                    "To keep going, set GEMINI_MODEL in backend/.env to another model, e.g. gemini-3.5-flash-lite.")
        if e.code == 429:
            return "Gemini rate limit reached. Wait a minute and try again, or lower GEMINI_RPM in backend/.env."
        if (e.code or 0) >= 500:
            return (f"Gemini is overloaded right now ({e.code}). Try again in a moment, "
                    "or set GEMINI_MODEL in backend/.env to a less busy model.")
        return f"Gemini request failed ({e.code}): {e.message}"
    return str(e)


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        if not s.gemini_api_key:
            raise LLMNotConfigured("GEMINI_API_KEY is not set in backend/.env")
        self.model = s.gemini_model
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._limiter = _RateLimiter(s.gemini_rpm)
        self._thinking_budget = s.gemini_thinking_budget

    def _config(self, system: str | None, temperature: float, **extra: Any) -> types.GenerateContentConfig:
        cfg: dict[str, Any] = {"system_instruction": system, "temperature": temperature, **extra}
        if self._thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self._thinking_budget)
        return types.GenerateContentConfig(**cfg)

    @_retry
    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        self._limiter.wait()
        resp = self._client.models.generate_content(
            model=self.model, contents=prompt, config=self._config(system, temperature)
        )
        return resp.text or ""

    @_retry
    def generate_json(self, prompt: str, system: str | None = None, schema: Any = None,
                      temperature: float = 0.0) -> Any:
        self._limiter.wait()
        extra: dict[str, Any] = {"response_mime_type": "application/json"}
        if schema is not None:
            extra["response_schema"] = schema
        resp = self._client.models.generate_content(
            model=self.model, contents=prompt, config=self._config(system, temperature, **extra)
        )
        return json.loads(resp.text or "null")

    @_stream_retry
    def _open_stream(self, prompt: str, system: str | None, temperature: float):
        # The request is sent on the first next(), so fetch the first chunk inside the retry
        self._limiter.wait()
        events = iter(self._client.models.generate_content_stream(
            model=self.model, contents=prompt, config=self._config(system, temperature)
        ))
        return next(events, None), events

    def stream(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> Iterator[str]:
        """Retries on 429/5xx only until the first chunk arrives; later errors propagate."""
        first, events = self._open_stream(prompt, system, temperature)
        for event in itertools.chain([first] if first else [], events):
            if event.text:
                yield event.text


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()
