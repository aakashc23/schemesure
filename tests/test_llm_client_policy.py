"""
Tests for the LLM client's failure handling: error classification, the retry
policy, the per-request deadline, and the guarantee that logging can never cause
a duplicate call.

Each test pins down one finding from the code audit (H3, L1, L2, L3):

- H3: a daily-quota 429 used to be retried for ~260 s per call, and "500" matched
  any message containing those digits. Now errors are classified by status code
  and type, quota / too-large / auth are never retried, and a server wait hint
  over MAX_SERVER_WAIT_SECONDS (60 s) is not honoured.
- L1: raw provider text (organisation ids, usage figures) must never reach a
  response; only a kind-based public message does.
- L2: a failure while LOGGING must not be mistaken for an LLM failure.
- L3: the call log records the real number of attempts.

No network: the SDK client is replaced by a scripted fake.
"""

from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError

from app.llm_client import (
    MAX_SERVER_WAIT_SECONDS,
    CallCounter,
    LLMClient,
    LLMError,
    classify_error,
    extract_json,
)


class StatusError(Exception):
    """An SDK-style exception: carries .status_code like openai.APIStatusError."""

    def __init__(self, status_code: int, message: str = "error"):
        super().__init__(message)
        self.status_code = status_code


# ==========================================================================
# Classification
# ==========================================================================


class TestClassifyError:
    def test_per_minute_429_is_a_retryable_rate_limit(self):
        err = StatusError(429, "Rate limit reached ... (TPM) Please try again in 1.68s")
        assert classify_error(err) == ("rate_limit", True)

    @pytest.mark.parametrize("message", [
        "Rate limit reached for model on tokens per day (TPD): Limit 200000",
        "Rate limit reached on requests per day (RPD)",
    ])
    def test_daily_quota_is_never_retried(self, message):
        assert classify_error(StatusError(429, message)) == ("quota", False)

    def test_request_too_large_is_never_retried(self):
        err = StatusError(429, "Request too large for model ... (OTPM)")
        assert classify_error(err) == ("too_large", False)
        assert classify_error(StatusError(413, "payload")) == ("too_large", False)

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_errors_are_never_retried(self, status):
        assert classify_error(StatusError(status)) == ("auth", False)

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 409])
    def test_server_errors_are_retryable(self, status):
        assert classify_error(StatusError(status)) == ("upstream", True)

    def test_other_4xx_is_not_retried(self):
        assert classify_error(StatusError(400, "bad request")) == ("upstream", False)
        assert classify_error(StatusError(404, "no such model")) == ("upstream", False)

    def test_408_is_a_retryable_timeout(self):
        assert classify_error(StatusError(408)) == ("timeout", True)

    def test_the_digits_500_inside_a_message_do_not_make_it_retryable(self):
        """The old classifier matched the substring "500" anywhere ('Limit 7500')."""
        assert classify_error(StatusError(400, "Limit 7500 exceeded")) == ("upstream", False)

    def test_sdk_connection_and_timeout_types(self):
        request = httpx.Request("POST", "http://example.invalid")
        assert classify_error(APITimeoutError(request=request)) == ("timeout", True)
        assert classify_error(APIConnectionError(request=request)) == ("upstream", True)

    def test_unknown_exception_is_not_blindly_retried(self):
        assert classify_error(ValueError("boom")) == ("upstream", False)


class TestLLMError:
    def test_public_message_hides_the_raw_detail(self):
        err = LLMError("failed", kind="quota", detail="organization `org_01abc` TPD 200000")
        assert "org_01abc" not in err.public_message
        assert "org_01abc" not in str(err)
        assert err.detail

    def test_every_kind_has_a_public_message(self):
        for kind in ("rate_limit", "quota", "auth", "too_large", "timeout",
                     "bad_output", "upstream"):
            assert LLMError("x", kind=kind).public_message

    def test_unknown_kind_falls_back_to_upstream(self):
        assert LLMError("x", kind="nonsense").kind == "upstream"

    def test_unparseable_json_keeps_model_text_out_of_the_message(self):
        with pytest.raises(LLMError) as info:
            extract_json("MODEL-WORDS-HERE {broken")
        assert info.value.kind == "bad_output"
        assert "MODEL-WORDS-HERE" not in str(info.value)


# ==========================================================================
# complete(): fake SDK
# ==========================================================================


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Usage:
    prompt_tokens = 30
    completion_tokens = 12


class _Response:
    def __init__(self, content="hello"):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class FakeSDK:
    """Stands in for OpenAI(): plays back a script of exceptions / responses."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.timeouts: list[float] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        self.timeouts.append(kwargs.get("timeout"))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


@pytest.fixture()
def client_with(monkeypatch):
    """Build an LLMClient over a FakeSDK; sleeping and DB logging are recorded."""
    sleeps: list[float] = []
    monkeypatch.setattr("app.llm_client.time.sleep", lambda seconds: sleeps.append(seconds))
    logged: list[dict] = []

    def fake_log(settings, prompt_name, model, pt, ct, latency, attempts, ok, error=""):
        logged.append({"prompt": prompt_name, "pt": pt, "ct": ct,
                       "attempts": attempts, "ok": ok, "error": error})

    monkeypatch.setattr("app.llm_client._log_call", fake_log)

    def build(script):
        llm = LLMClient()           # conftest provides a dummy key; no network call
        llm._client = FakeSDK(script)
        return llm

    build.sleeps = sleeps
    build.logged = logged
    return build


class TestRetryPolicy:
    def test_daily_quota_raises_immediately_without_sleeping(self, client_with):
        llm = client_with([StatusError(429, "tokens per day (TPD) Please try again in 3m12.24s")])
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u")
        assert info.value.kind == "quota"
        assert llm._client.calls == 1
        assert client_with.sleeps == []

    def test_request_too_large_is_not_retried(self, client_with):
        llm = client_with([StatusError(429, "Request too large (OTPM)")])
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u")
        assert info.value.kind == "too_large"
        assert llm._client.calls == 1

    def test_auth_error_is_not_retried(self, client_with):
        llm = client_with([StatusError(401, "bad key")])
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u")
        assert info.value.kind == "auth"
        assert llm._client.calls == 1

    def test_long_server_wait_hint_is_not_honoured(self, client_with):
        """A 429 asking for more than a minute is a daily quota: it will not clear inside one request."""
        hint = MAX_SERVER_WAIT_SECONDS + 5
        llm = client_with([StatusError(429, f"Please try again in {hint}s")])
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u")
        assert info.value.kind == "rate_limit"
        assert llm._client.calls == 1
        assert client_with.sleeps == []

    def test_per_minute_output_token_limit_is_waited_out(self, client_with):
        """Groq's 1,000 output-tokens/minute limit asks for ~25 s: honour it."""
        llm = client_with([
            StatusError(429, "output tokens per minute (OTPM): Limit 1000. Please try again in 23.76s"),
            _Response("ok"),
        ])
        result = llm.complete("p", "s", "u")
        assert result.text == "ok"
        assert client_with.sleeps == [pytest.approx(24.26)]

    def test_short_rate_limit_is_waited_out_then_succeeds(self, client_with):
        llm = client_with([StatusError(429, "Please try again in 1.5s"), _Response("ok")])
        result = llm.complete("p", "s", "u")
        assert result.text == "ok"
        assert result.attempts == 2
        assert client_with.sleeps == [pytest.approx(2.0)]    # 1.5 s hint + 0.5 s margin
        # The log records the REAL attempt count (audit finding L3).
        assert client_with.logged[-1]["attempts"] == 2
        assert client_with.logged[-1]["ok"] is True

    def test_failure_log_records_the_real_attempt_count(self, client_with):
        llm = client_with([StatusError(401, "bad key")])
        with pytest.raises(LLMError):
            llm.complete("p", "s", "u")
        assert client_with.logged[-1]["attempts"] == 1       # not llm_max_retries
        assert client_with.logged[-1]["ok"] is False

    def test_failed_attempts_and_tokens_are_counted(self, client_with):
        llm = client_with([StatusError(503), _Response("fine")])
        counter = CallCounter()
        llm.complete("p", "s", "u", counter=counter)
        assert counter.count == 1
        assert counter.failed_attempts == 1
        assert counter.prompt_tokens == 30
        assert counter.completion_tokens == 12
        assert counter.total_tokens == 42

    def test_error_text_never_contains_provider_details(self, client_with):
        llm = client_with([StatusError(429, "organization `org_01xyz` TPD (per day) used")])
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u")
        assert "org_01xyz" not in str(info.value)
        assert "org_01xyz" not in info.value.public_message
        assert "org_01xyz" in info.value.detail               # kept for server logs only

    def test_empty_content_is_a_bad_output_error_not_retried(self, client_with):
        llm = client_with([_Response("   ")])
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u")
        assert info.value.kind == "bad_output"
        assert llm._client.calls == 1


class TestRequestDeadline:
    def test_expired_deadline_means_no_attempt(self, client_with):
        llm = client_with([_Response()])
        counter = CallCounter()
        counter.set_deadline(-1)
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u", counter=counter)
        assert info.value.kind == "timeout"
        assert llm._client.calls == 0

    def test_never_sleeps_past_the_deadline(self, client_with):
        llm = client_with([StatusError(429, "Please try again in 15s"), _Response()])
        counter = CallCounter()
        counter.set_deadline(5)          # a 15.5 s sleep would overshoot this
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u", counter=counter)
        assert info.value.kind == "timeout"
        assert client_with.sleeps == []
        assert llm._client.calls == 1

    def test_per_call_timeout_is_capped_by_the_time_left(self, client_with):
        llm = client_with([_Response()])
        counter = CallCounter()
        counter.set_deadline(10)
        llm.complete("p", "s", "u", counter=counter)
        # The per-phase httpx limits together never exceed the time that is left.
        limits = llm._client.timeouts[0]
        assert limits.connect + limits.read <= 10

    def test_no_deadline_means_no_limit(self):
        assert CallCounter().remaining() is None


class TestLoggingNeverCausesARetry:
    def test_a_logging_crash_does_not_trigger_a_second_llm_call(self, monkeypatch):
        """
        Audit finding L2: a non-sqlite error while logging used to be treated as
        an LLM failure, throwing away a good answer and asking the model again.
        """
        monkeypatch.setattr("app.llm_client.time.sleep", lambda s: None)
        llm = LLMClient()      # built first: its constructor also touches the DB

        def exploding_connect(path):
            raise OSError("disk is full")

        monkeypatch.setattr("app.llm_client._connect", exploding_connect)
        llm._client = FakeSDK([_Response("answer"), _Response("MUST NOT BE USED")])
        result = llm.complete("p", "s", "u")
        assert result.text == "answer"
        assert llm._client.calls == 1
