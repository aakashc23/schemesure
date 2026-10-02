"""
Tests for language detection and the normalizer's failure behaviour.

The heuristic path matters more than it looks: it is what the system falls back
to when the LLM is unreachable, so it must never raise and must never silently
mislabel Devanagari as English.
"""

from __future__ import annotations

import pytest

from app.llm_client import LLMError, LLMResult
from app.schemas import Language
from core.query_normalizer import detect_language_heuristic, normalize_query


class TestLanguageHeuristic:
    @pytest.mark.parametrize("text", [
        "How much money does PM-KISAN give?",
        "What documents are required for Ujjwala?",
        "Who is eligible for Stand-Up India?",
    ])
    def test_english(self, text):
        assert detect_language_heuristic(text) == Language.ENGLISH

    @pytest.mark.parametrize("text", [
        "पीएम किसान में कितने पैसे मिलते हैं?",
        "अटल पेंशन योजना के लिए आयु सीमा क्या है?",
        "उज्ज्वला योजना",
    ])
    def test_devanagari_is_hindi(self, text):
        assert detect_language_heuristic(text) == Language.HINDI

    @pytest.mark.parametrize("text", [
        "PM Kisan ka paisa kitna milta hai?",
        "Mujhe kaun si yojana mil sakti hai?",
        "Ujjwala ke liye kya documents chahiye?",
        "Mera age 25 hai, kya main eligible hun?",
    ])
    def test_romanised_hindi_is_hinglish(self, text):
        assert detect_language_heuristic(text) == Language.HINGLISH

    def test_english_sentence_containing_yojana_stays_english(self):
        """'yojana' is a scheme-name word that appears in English sentences.
        Only Hindi *function* words should tip the balance to Hinglish."""
        assert detect_language_heuristic(
            "What are the benefits of the Ujjwala Yojana scheme?"
        ) == Language.ENGLISH

    def test_single_marker_is_not_enough(self):
        """One ambiguous token must not flip the verdict."""
        assert detect_language_heuristic("Where is the PM office") == Language.ENGLISH

    def test_mixed_script_prefers_hindi(self):
        assert detect_language_heuristic("PM-KISAN में कितना पैसा") == Language.HINDI

    @pytest.mark.parametrize("text", ["", "   ", "?", "12345"])
    def test_degenerate_input_does_not_raise(self, text):
        assert detect_language_heuristic(text) in set(Language)


# --------------------------------------------------------------------------
# normalize_query
# --------------------------------------------------------------------------


class FakeLLM:
    """Stand-in for LLMClient. Returns a scripted payload or raises."""

    def __init__(self, payload=None, error: bool = False):
        self.payload = payload
        self.error = error
        self.calls = 0

    def complete_json(self, prompt_name, system, user, max_tokens=None, counter=None):
        self.calls += 1
        if self.error:
            raise LLMError("simulated failure")
        result = LLMResult(text="{}", prompt_name=prompt_name, model="fake")
        if counter is not None:
            counter.record(result)
        return self.payload, result


class TestNormalizeQuery:
    def test_no_llm_uses_the_heuristic(self):
        result = normalize_query(None, "PM Kisan ka paisa kitna milta hai?")
        assert result.language == Language.HINGLISH
        assert result.fallback_used is True
        assert result.normalized_query == "PM Kisan ka paisa kitna milta hai?"

    def test_happy_path(self):
        llm = FakeLLM({
            "normalized_query": "How much money does PM-KISAN give per year?",
            "language": "hinglish",
        })
        result = normalize_query(llm, "PM Kisan ka paisa kitna milta hai?")
        assert result.normalized_query == "How much money does PM-KISAN give per year?"
        assert result.language == Language.HINGLISH
        assert result.fallback_used is False

    def test_llm_failure_degrades_instead_of_raising(self):
        """A normalizer outage must not turn into a 500 for the user."""
        result = normalize_query(FakeLLM(error=True), "पीएम किसान क्या है?")
        assert result.fallback_used is True
        assert result.language == Language.HINDI
        assert result.normalized_query == "पीएम किसान क्या है?"

    def test_empty_rewrite_falls_back_to_the_original(self):
        """An empty normalized query would destroy retrieval silently."""
        llm = FakeLLM({"normalized_query": "   ", "language": "en"})
        result = normalize_query(llm, "What is PM-KISAN?")
        assert result.normalized_query == "What is PM-KISAN?"

    def test_unknown_language_tag_falls_back_to_the_heuristic(self):
        llm = FakeLLM({"normalized_query": "What is PM-KISAN?", "language": "hindi"})
        result = normalize_query(llm, "पीएम किसान क्या है?")
        assert result.language == Language.HINDI

    def test_non_dict_payload_is_handled(self):
        result = normalize_query(FakeLLM(["not", "a", "dict"]), "What is PM-KISAN?")
        assert result.fallback_used is True
        assert result.language == Language.ENGLISH

    def test_counter_records_the_call(self):
        from app.llm_client import CallCounter

        counter = CallCounter()
        llm = FakeLLM({"normalized_query": "x", "language": "en"})
        normalize_query(llm, "What is PM-KISAN?", counter=counter)
        assert counter.count == 1
        assert counter.prompt_names == ["normalize_query"]
