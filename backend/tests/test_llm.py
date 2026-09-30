from types import SimpleNamespace

import pytest
from google.genai import errors
from tenacity import wait_none

from app.generation.llm import LLMClient, describe_error


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}})


def _daily_quota_error() -> errors.APIError:
    # Shape of the real free-tier response (trimmed)
    violation = {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                 "quotaDimensions": {"location": "global", "model": "gemini-3.5-flash"}, "quotaValue": "20"}
    return errors.APIError(429, {"error": {"code": 429, "message": "You exceeded your current quota",
                                           "status": "RESOURCE_EXHAUSTED",
                                           "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                                        "violations": [violation]}]}})


class _FlakyModels:
    """generate_content_stream fails on the first chunk `failures` times, then streams."""

    def __init__(self, failures: int, chunks: list[str], mid_stream_error: bool = False, error=None):
        self.failures, self.chunks, self.mid_stream_error, self.calls = failures, chunks, mid_stream_error, 0
        self.error = error or (lambda: _api_error(503))

    def generate_content_stream(self, **_):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error()
        for i, text in enumerate(self.chunks):
            if self.mid_stream_error and i == 1:
                raise _api_error(503)
            yield SimpleNamespace(text=text)


def _client(models: _FlakyModels) -> LLMClient:
    llm = object.__new__(LLMClient)  # skip __init__: no API key or network needed
    llm.model, llm._thinking_budget = "test-model", None
    llm._client = SimpleNamespace(models=models)
    llm._limiter = SimpleNamespace(wait=lambda: None)
    return llm


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(LLMClient._open_stream.retry, "wait", wait_none())


def test_stream_retries_until_first_chunk():
    models = _FlakyModels(failures=2, chunks=["a", "b", "c"])
    assert "".join(_client(models).stream("q")) == "abc"
    assert models.calls == 3


def test_stream_gives_up_after_retry_budget():
    models = _FlakyModels(failures=10, chunks=["a"])
    with pytest.raises(errors.APIError):
        list(_client(models).stream("q"))
    assert models.calls == 4


def test_stream_does_not_retry_mid_stream():
    # Tokens already reached the client, so a later failure must surface rather than restart the answer
    models = _FlakyModels(failures=0, chunks=["a", "b"], mid_stream_error=True)
    out = []
    with pytest.raises(errors.APIError):
        for tok in _client(models).stream("q"):
            out.append(tok)
    assert out == ["a"] and models.calls == 1


def test_daily_quota_is_not_retried():
    models = _FlakyModels(failures=10, chunks=["a"], error=_daily_quota_error)
    with pytest.raises(errors.APIError):
        list(_client(models).stream("q"))
    assert models.calls == 1


def test_describe_daily_quota_names_model_and_limit():
    msg = describe_error(_daily_quota_error())
    assert "daily quota for gemini-3.5-flash" in msg and "(20 requests/day)" in msg


def test_describe_error_is_readable():
    assert describe_error(_api_error(503)).startswith("Gemini is overloaded right now (503)")
    assert "rate limit" in describe_error(_api_error(429))
    assert describe_error(ValueError("boom")) == "boom"
