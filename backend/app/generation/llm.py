"""Gemini client: model fallback, client-side rate limiting, and retry on 429/5xx."""

from __future__ import annotations

import itertools
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any, TypeVar

from google import genai
from google.genai import errors, types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import get_settings

log = logging.getLogger(__name__)
T = TypeVar("T")


class LLMNotConfigured(RuntimeError):
    pass


class _RateLimiter:
    """At most `rpm` requests in any rolling 60 s window, which is how Gemini counts its quota.
    A short burst (rewrite a follow-up, then answer it) goes out at once; long runs still stay under it."""

    def __init__(self, rpm: int):
        self.rpm = max(rpm, 1)
        self._lock = threading.Lock()
        self._sent: deque[float] = deque()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._sent and now - self._sent[0] >= 60:
                    self._sent.popleft()
                if len(self._sent) < self.rpm:
                    self._sent.append(now)
                    return
                delay = 60 - (now - self._sent[0])
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


def _cooldown_seconds(e: errors.APIError) -> float | None:
    """How long to skip a model after this error; None = not a model-availability problem, don't fall back."""
    if _daily_quota_exhausted(e):
        return 3600  # resets at midnight Pacific; look again hourly
    if e.code == 404:
        return 86400  # model not offered to this key
    if e.code == 429 or (e.code or 0) >= 500:
        return 60
    return None


_retry = retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
# Someone is waiting on the result (a chat answer): give up sooner (~30 s of backoff) than batch jobs do
_quick_retry = retry(
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
            return (f"Gemini's free daily quota for {model} is used up{limit}, and no fallback model answered. "
                    "It resets at midnight Pacific time. To keep going, add another model to "
                    "GEMINI_FALLBACK_MODELS in backend/.env, e.g. gemini-3.5-flash-lite.")
        if e.code == 429:
            return "Gemini rate limit reached. Wait a minute and try again, or lower GEMINI_RPM in backend/.env."
        if (e.code or 0) >= 500:
            return (f"Gemini is overloaded right now ({e.code}), including the fallback models. Try again in a "
                    "moment, or add a less busy model to GEMINI_FALLBACK_MODELS in backend/.env.")
        return f"Gemini request failed ({e.code}): {e.message}"
    return str(e)


@lru_cache
def _shared_limiter(rpm: int) -> _RateLimiter:
    return _RateLimiter(rpm)


class LLMClient:
    def __init__(self, models: list[str] | None = None) -> None:
        s = get_settings()
        if not s.gemini_api_key:
            raise LLMNotConfigured("GEMINI_API_KEY is not set in backend/.env")
        chain = models or [s.gemini_model, *s.fallback_models]
        self.models = list(dict.fromkeys(m for m in chain if m))  # de-duplicated, order kept
        self.model = self.models[0]
        self._client = genai.Client(api_key=s.gemini_api_key)
        self._limiter = _shared_limiter(s.gemini_rpm)
        self._thinking_budget = s.gemini_thinking_budget
        self._cooldown: dict[str, float] = {}

    def _config(self, system: str | None, temperature: float, **extra: Any) -> types.GenerateContentConfig:
        cfg: dict[str, Any] = {"system_instruction": system, "temperature": temperature, **extra}
        if self._thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self._thinking_budget)
        return types.GenerateContentConfig(**cfg)

    def _candidates(self) -> list[str]:
        now = time.monotonic()
        ready = [m for m in self.models if self._cooldown.get(m, 0.0) <= now]
        return ready or list(self.models)  # everything cooling down: try them all again

    def _call(self, fn: Callable[[str], T]) -> tuple[T, str]:
        """Run fn(model) on the first model that answers. Overloaded or out-of-quota models are
        skipped for a while, so later calls don't pay for the same failure again."""
        failures: list[errors.APIError] = []
        for model in self._candidates():
            self._limiter.wait()
            try:
                return fn(model), model
            except errors.APIError as e:
                pause = _cooldown_seconds(e)
                if pause is None:
                    raise
                self._cooldown[model] = time.monotonic() + pause
                log.warning("Gemini model %s unavailable (%s); trying the next model", model, e.code)
                failures.append(e)
        # Prefer an error worth retrying, so the caller's backoff gets another go at the chain
        raise next((e for e in failures if _retryable(e)), failures[-1])

    def _generate_once(self, prompt: str, system: str | None, temperature: float, info: dict | None) -> str:
        resp, model = self._call(lambda m: self._client.models.generate_content(
            model=m, contents=prompt, config=self._config(system, temperature)))
        if info is not None:
            info["model"] = model
        return resp.text or ""

    _generate = _retry(_generate_once)
    _generate_quick = _quick_retry(_generate_once)

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.2, *,
                 quick: bool = False, info: dict | None = None) -> str:
        """quick=True gives up after ~30 s of backoff instead of minutes; info["model"] reports who answered."""
        return (self._generate_quick if quick else self._generate)(prompt, system, temperature, info)

    def _generate_json_once(self, prompt: str, system: str | None, schema: Any, temperature: float,
                            info: dict | None) -> Any:
        extra: dict[str, Any] = {"response_mime_type": "application/json"}
        if schema is not None:
            extra["response_schema"] = schema
        resp, model = self._call(lambda m: self._client.models.generate_content(
            model=m, contents=prompt, config=self._config(system, temperature, **extra)))
        if info is not None:
            info["model"] = model
        return json.loads(resp.text or "null")

    _generate_json = _retry(_generate_json_once)
    _generate_json_quick = _quick_retry(_generate_json_once)

    def generate_json(self, prompt: str, system: str | None = None, schema: Any = None,
                      temperature: float = 0.0, info: dict | None = None, *, quick: bool = False) -> Any:
        return (self._generate_json_quick if quick else self._generate_json)(prompt, system, schema, temperature, info)

    @_quick_retry
    def _open_stream(self, prompt: str, system: str | None, temperature: float):
        def start(model: str):
            # The request is sent on the first next(), so fetch the first chunk inside the retry
            events = iter(self._client.models.generate_content_stream(
                model=model, contents=prompt, config=self._config(system, temperature)))
            return next(events, None), events

        (first, events), model = self._call(start)
        return first, events, model

    def stream(self, prompt: str, system: str | None = None, temperature: float = 0.2,
               info: dict | None = None) -> Iterator[str]:
        """Retries and falls back only until the first chunk arrives; later errors propagate."""
        first, events, model = self._open_stream(prompt, system, temperature)
        if info is not None:
            info["model"] = model
        for event in itertools.chain([first] if first else [], events):
            if event.text:
                yield event.text


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()


@lru_cache
def get_judge_llm() -> LLMClient:
    """GEMINI_JUDGE_MODEL first (e.g. a stronger model), then the answer models as fallbacks."""
    s = get_settings()
    if not s.gemini_judge_model:
        return get_llm()
    return LLMClient([s.gemini_judge_model, s.gemini_model, *s.fallback_models])
