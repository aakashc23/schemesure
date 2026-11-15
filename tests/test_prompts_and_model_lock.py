"""
Small tests for pieces that have no other natural home:

- refusal_message(): the right canned text for every refusal reason and language.
- the repair prompt: it must carry the NORMALIZED query, never raw user text
  (audit finding H1).
- get_embedding_model(): two threads asking at once must load the model ONCE
  (audit finding M1: without a lock each start-up loaded its own ~470 MB copy).
"""

from __future__ import annotations

import threading
import time

import pytest

import core.ingest as ingest
from app.prompts import (
    ANSWER_REPAIR_SYSTEM,
    ANSWER_REPAIR_USER,
    BLOCKED_MESSAGES,
    REFUSAL_MESSAGES,
    SERVICE_UNAVAILABLE_MESSAGES,
    refusal_for,
    refusal_message,
)
from app.schemas import RefusalReason


class TestRefusalMessage:
    @pytest.mark.parametrize("language", ["en", "hi", "hinglish"])
    def test_each_reason_maps_to_its_own_text(self, language):
        assert refusal_message(language, RefusalReason.LOW_EVIDENCE) == REFUSAL_MESSAGES[language]
        assert refusal_message(language, RefusalReason.NO_CLAIMS) == REFUSAL_MESSAGES[language]
        assert refusal_message(language, RefusalReason.NOT_SUPPORTED) == BLOCKED_MESSAGES[language]
        assert (
            refusal_message(language, RefusalReason.SERVICE_UNAVAILABLE)
            == SERVICE_UNAVAILABLE_MESSAGES[language]
        )

    def test_reason_can_be_given_as_a_plain_string(self):
        assert refusal_message("en", "service_unavailable") == SERVICE_UNAVAILABLE_MESSAGES["en"]

    def test_unknown_language_falls_back_to_english(self):
        assert refusal_message("fr", RefusalReason.LOW_EVIDENCE) == REFUSAL_MESSAGES["en"]

    def test_unknown_reason_falls_back_to_the_strict_text(self):
        assert refusal_message("en", "mystery") == BLOCKED_MESSAGES["en"]

    def test_service_unavailable_text_matches_the_spec_and_never_claims_missing_info(self):
        english = SERVICE_UNAVAILABLE_MESSAGES["en"]
        assert english.startswith("I could not verify an answer right now because the verification service")
        assert "try again in a minute" in english
        assert "do not have official information" not in english

    def test_all_languages_are_covered(self):
        assert set(SERVICE_UNAVAILABLE_MESSAGES) == {"en", "hi", "hinglish"}
        assert "सत्यापन" in SERVICE_UNAVAILABLE_MESSAGES["hi"]

    def test_refusal_for_still_works(self):
        assert refusal_for("en") == REFUSAL_MESSAGES["en"]
        assert refusal_for("hi", blocked=True) == BLOCKED_MESSAGES["hi"]


class TestRepairPrompt:
    def test_repair_user_prompt_has_no_raw_question_placeholder(self):
        assert "{question}" not in ANSWER_REPAIR_USER
        assert "{query}" in ANSWER_REPAIR_USER
        assert "Original question" not in ANSWER_REPAIR_USER

    def test_repair_system_prompt_explains_the_evidence_markers(self):
        assert "[n]" in ANSWER_REPAIR_SYSTEM
        assert "position" in ANSWER_REPAIR_SYSTEM.lower()


class TestEmbeddingModelIsLoadedOnce:
    def test_concurrent_callers_share_one_load(self, monkeypatch):
        constructed: list[int] = []

        class SlowFakeModel:
            def __init__(self, *args, **kwargs):
                constructed.append(1)
                time.sleep(0.2)            # wide window for a second thread to sneak in

        monkeypatch.setattr("sentence_transformers.SentenceTransformer", SlowFakeModel)
        monkeypatch.setattr(ingest, "_MODEL", None)

        results: list[object] = []

        def worker():
            results.append(ingest.get_embedding_model())

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(constructed) == 1
        assert len(results) == 6 and all(model is results[0] for model in results)
