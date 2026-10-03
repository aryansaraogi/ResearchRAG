from types import SimpleNamespace

import pytest
from google.genai import errors
from tenacity import wait_none

from app.generation import llm as llm_module
from app.generation.llm import LLMClient, _RateLimiter, describe_error


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


class _PerModel:
    """generate_content fails for the models in `down` and otherwise answers with the model's name."""

    def __init__(self, down: dict):
        self.down, self.calls = down, []

    def generate_content(self, *, model, **_):
        self.calls.append(model)
        if model in self.down:
            raise self.down[model]()
        return SimpleNamespace(text=f"answer from {model}")


def _client(models, chain: list[str] | None = None) -> LLMClient:
    llm = object.__new__(LLMClient)  # skip __init__: no API key or network needed
    llm.models = chain or ["test-model"]
    llm.model, llm._thinking_budget, llm._cooldown = llm.models[0], None, {}
    llm._client = SimpleNamespace(models=models)
    llm._limiter = SimpleNamespace(wait=lambda: None)
    return llm


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    for fn in (LLMClient._open_stream, LLMClient._generate, LLMClient._generate_quick):
        monkeypatch.setattr(fn.retry, "wait", wait_none())


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


def test_falls_back_when_primary_is_overloaded():
    models = _PerModel({"a": lambda: _api_error(503)})
    llm, info = _client(models, ["a", "b"]), {}
    assert llm.generate("q", info=info) == "answer from b" and info["model"] == "b"
    # The overloaded model cools down, so the next call goes straight to the fallback
    llm.generate("q2")
    assert models.calls == ["a", "b", "b"]


def test_falls_back_when_primary_daily_quota_is_used_up():
    models = _PerModel({"a": _daily_quota_error})
    assert _client(models, ["a", "b"]).generate("q") == "answer from b"


def test_bad_request_does_not_fall_back():
    models = _PerModel({"a": lambda: _api_error(400)})
    with pytest.raises(errors.APIError):
        _client(models, ["a", "b"]).generate("q")
    assert models.calls == ["a"]


def test_stream_falls_back_before_first_chunk():
    class Models:
        calls: list[str] = []

        def generate_content_stream(self, *, model, **_):
            self.calls.append(model)
            if model == "a":
                raise _api_error(503)
            yield SimpleNamespace(text="ok")

    info = {}
    assert "".join(_client(Models(), ["a", "b"]).stream("q", info=info)) == "ok" and info["model"] == "b"


def test_rate_limiter_allows_a_burst_then_waits_for_the_window(monkeypatch):
    clock, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        clock[0] += s

    monkeypatch.setattr(llm_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm_module.time, "sleep", sleep)
    limiter = _RateLimiter(2)
    limiter.wait()
    limiter.wait()
    assert slept == []  # two requests fit in the minute
    clock[0] = 10
    limiter.wait()
    assert slept == [50]  # the third waits until the first leaves the 60 s window
