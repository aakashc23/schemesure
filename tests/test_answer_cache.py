"""
Tests for the /ask answer cache.

The cache exists to protect a free-tier token budget during demos, so the tests
pin down the properties that matter: a repeat is a hit, an old entry expires, the
cache stays bounded, it can be switched off, and a temporary outage is never
replayed to the next visitor. Time is injected, so nothing sleeps.
"""

from __future__ import annotations

from app.schemas import (
    AnswerStatus,
    AskResponse,
    GuardrailDecision,
    GuardrailReport,
    Language,
    RefusalReason,
)
from core.answer_cache import AnswerCache, make_key


def response(answer: str = "an answer", reason: RefusalReason | None = None) -> AskResponse:
    return AskResponse(
        question="q", normalized_query="q", language=Language.ENGLISH, answer=answer,
        status=AnswerStatus.REFUSED if reason else AnswerStatus.VERIFIED,
        guardrail=GuardrailReport(decision=GuardrailDecision.PASS),
        refusal_reason=reason,
    )


class Clock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestKey:
    def test_case_and_whitespace_are_ignored(self):
        assert make_key("How much does  PM-KISAN pay?", None, True) == make_key(
            "  how much does pm-kisan pay?  ", None, True
        )

    def test_language_override_is_part_of_the_key(self):
        assert make_key("q", "hi", True) != make_key("q", None, True)
        assert make_key("q", None, True) == make_key("q", "", True)

    def test_guardrail_flag_is_part_of_the_key(self):
        assert make_key("q", None, True) != make_key("q", None, False)


class TestHitAndMiss:
    def test_miss_then_hit(self):
        cache = AnswerCache(ttl_seconds=60, max_entries=4)
        key = make_key("q", None, True)
        assert cache.get(key) is None
        assert cache.put(key, response("first")) is True
        hit = cache.get(key)
        assert hit is not None and hit.answer == "first"

    def test_a_hit_is_a_copy_so_callers_cannot_corrupt_the_cache(self):
        cache = AnswerCache(ttl_seconds=60, max_entries=4)
        key = make_key("q", None, True)
        cache.put(key, response("first"))
        cache.get(key).answer = "tampered"
        cache.get(key).cached = True
        again = cache.get(key)
        assert again.answer == "first" and again.cached is False

    def test_stored_value_is_a_copy_of_what_was_put(self):
        cache = AnswerCache(ttl_seconds=60, max_entries=4)
        original = response("first")
        key = make_key("q", None, True)
        cache.put(key, original)
        original.answer = "changed after caching"
        assert cache.get(key).answer == "first"


class TestExpiry:
    def test_entry_expires_after_the_ttl(self):
        clock = Clock()
        cache = AnswerCache(ttl_seconds=60, max_entries=4, clock=clock)
        key = make_key("q", None, True)
        cache.put(key, response())
        clock.now += 59
        assert cache.get(key) is not None
        clock.now += 2                      # 61 s after it was stored
        assert cache.get(key) is None
        assert len(cache) == 0              # the expired entry was dropped


class TestLRU:
    def test_least_recently_used_entry_is_evicted(self):
        cache = AnswerCache(ttl_seconds=60, max_entries=2)
        a, b, c = (make_key(name, None, True) for name in "abc")
        cache.put(a, response("A"))
        cache.put(b, response("B"))
        cache.get(a)                        # touch a: b is now the oldest
        cache.put(c, response("C"))         # over capacity -> evict b
        assert cache.get(a) is not None
        assert cache.get(b) is None
        assert cache.get(c) is not None
        assert len(cache) == 2

    def test_overwriting_a_key_does_not_grow_the_cache(self):
        cache = AnswerCache(ttl_seconds=60, max_entries=2)
        key = make_key("q", None, True)
        cache.put(key, response("one"))
        cache.put(key, response("two"))
        assert len(cache) == 1 and cache.get(key).answer == "two"


class TestDisabled:
    def test_zero_entries_disables_the_cache(self):
        cache = AnswerCache(ttl_seconds=60, max_entries=0)
        key = make_key("q", None, True)
        assert cache.enabled is False
        assert cache.put(key, response()) is False
        assert cache.get(key) is None

    def test_zero_ttl_disables_the_cache(self):
        cache = AnswerCache(ttl_seconds=0, max_entries=10)
        assert cache.enabled is False
        assert cache.put(make_key("q", None, True), response()) is False


class TestNeverCacheOutages:
    def test_service_unavailable_is_not_stored(self):
        cache = AnswerCache(ttl_seconds=60, max_entries=4)
        key = make_key("q", None, True)
        assert cache.put(key, response(reason=RefusalReason.SERVICE_UNAVAILABLE)) is False
        assert cache.get(key) is None

    def test_honest_refusals_are_cached(self):
        """A low-evidence refusal is a stable, correct answer and saves tokens."""
        cache = AnswerCache(ttl_seconds=60, max_entries=4)
        key = make_key("q", None, True)
        assert cache.put(key, response(reason=RefusalReason.LOW_EVIDENCE)) is True
        assert cache.get(key).refusal_reason == RefusalReason.LOW_EVIDENCE
