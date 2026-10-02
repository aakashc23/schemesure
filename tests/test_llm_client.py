"""
Tests for the LLM client's pure helpers: JSON extraction and retry timing.

Both exist because of real failures hit while building:

- `extract_json` — models wrap JSON in ``` fences or add a sentence of preamble
  even when the prompt forbids it. Failing the request over that would make the
  guardrail fragile for no good reason.

- `_retry_after_seconds` — Groq's free tier enforces a rolling 8,000
  tokens-per-minute budget and its 429 body says exactly how long to wait
  ("Please try again in 1.68s"). Fixed exponential backoff either sleeps far
  longer than needed or retries too early and burns an attempt; during the
  evaluation run that difference is the difference between finishing and not.
"""

from __future__ import annotations

import pytest

from app.llm_client import CallCounter, LLMError, LLMResult, extract_json, _retry_after_seconds


class FakeError(Exception):
    """Stands in for an SDK exception whose message carries the retry hint."""


class FakeHeaders(dict):
    def get(self, key, default=None):  # dict.get, but case-insensitive-ish
        return super().get(key, super().get(key.lower(), default))


class FakeResponse:
    def __init__(self, headers: dict):
        self.headers = FakeHeaders(headers)


# ==========================================================================
# extract_json
# ==========================================================================


class TestExtractJson:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_plain_array(self):
        assert extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_json_fence(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_bare_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_preamble_before_object(self):
        """The commonest real violation: a sentence before the JSON."""
        assert extract_json('Sure! Here is the JSON:\n{"a": 1}') == {"a": 1}

    def test_trailing_commentary(self):
        assert extract_json('{"a": 1}\nHope that helps!') == {"a": 1}

    def test_nested_object_survives_span_extraction(self):
        text = 'noise {"a": {"b": [1, 2]}, "c": "}"} more noise'
        assert extract_json(text) == {"a": {"b": [1, 2]}, "c": "}"}

    def test_unicode_is_preserved(self):
        parsed = extract_json('{"q": "पीएम किसान"}')
        assert parsed["q"] == "पीएम किसान"

    @pytest.mark.parametrize("text", ["", "   ", "not json at all", "{broken", "{'single': 'quotes'}"])
    def test_unparseable_raises_llm_error(self, text):
        with pytest.raises(LLMError):
            extract_json(text)


# ==========================================================================
# _retry_after_seconds
# ==========================================================================


class TestRetryAfter:
    def test_parses_groq_rate_limit_message(self):
        message = (
            "Error code: 429 - Rate limit reached for model `qwen/qwen3.8-27b` on "
            "input tokens per minute (ITPM): Limit 7000, Used 6208, Requested 988. "
            "Please try again in 1.68s. Need more tokens? Upgrade to Dev Tier"
        )
        assert _retry_after_seconds(FakeError(message)) == pytest.approx(1.68)

    def test_parses_minutes_and_seconds(self):
        assert _retry_after_seconds(FakeError("Please try again in 2m30.5s")) == pytest.approx(150.5)

    def test_parses_sub_second(self):
        assert _retry_after_seconds(FakeError("please try again in 0.5s")) == pytest.approx(0.5)

    def test_prefers_the_retry_after_header(self):
        exc = FakeError("Please try again in 30s")
        exc.response = FakeResponse({"retry-after": "3"})
        # The header is authoritative when present.
        assert _retry_after_seconds(exc) == pytest.approx(3.0)

    def test_no_hint_returns_none(self):
        """Caller falls back to exponential backoff in this case."""
        assert _retry_after_seconds(FakeError("Internal server error")) is None

    def test_malformed_header_falls_through_to_the_message(self):
        exc = FakeError("Please try again in 7s")
        exc.response = FakeResponse({"retry-after": "not-a-number"})
        assert _retry_after_seconds(exc) == pytest.approx(7.0)


# ==========================================================================
# CallCounter
# ==========================================================================


class TestCallCounter:
    def test_counts_and_names_calls(self):
        """`llm_calls_per_question` in the eval is only as honest as this."""
        counter = CallCounter()
        assert counter.count == 0

        for name in ("normalize_query", "generate_answer", "split_claims", "judge_claims"):
            counter.record(LLMResult(text="", prompt_name=name, model="m"))

        assert counter.count == 4
        assert counter.prompt_names == [
            "normalize_query", "generate_answer", "split_claims", "judge_claims"
        ]

    def test_total_tokens(self):
        result = LLMResult(text="", prompt_name="p", model="m",
                           prompt_tokens=100, completion_tokens=25)
        assert result.total_tokens == 125
