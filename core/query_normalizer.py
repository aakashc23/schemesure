"""
Turn whatever the user typed into (a) an English search query and (b) a language tag.

WHY THIS STEP EXISTS — AND WHY IT IS NOT OPTIONAL
------------------------------------------------
Our scheme documents are in English. The embedding model is multilingual, so you
might expect a Hindi or Hinglish question to retrieve the right chunk directly.
Measured against the real index, it does not reliably:

    "How much money does PM-KISAN give per year?"   -> 0.574  correct scheme
    "पीएम किसान में कितने पैसे मिलते हैं?"              -> 0.610  WRONG scheme
    "PM Kisan ka paisa kitna milta hai?"           -> 0.436  WRONG scheme

The Hinglish score (0.436) is below our 0.45 refusal threshold, so without this
step the system would tell a Hinglish speaker it has no information — about a
scheme it has fully indexed. Romanised Hindi is the hardest case: the model was
trained on Devanagari Hindi and English, not on Hindi spelled in Latin letters.

One LLM call fixes both problems at once: it translates to English (so retrieval
works) and reports the language (so we answer back in the user's own language).

FAILURE BEHAVIOUR: if the call fails we fall back to the raw question plus a
script-based language guess, and mark `fallback_used=True`. Degraded retrieval
beats a 500 error, and the flag makes the degradation visible instead of silent.
"""

from __future__ import annotations

import re

from app.llm_client import CallCounter, LLMClient, LLMError
from app.prompts import QUERY_NORMALIZER_SYSTEM, QUERY_NORMALIZER_USER
from app.schemas import Language, NormalizedQuery

# Devanagari block. Presence of these characters is a reliable "this is Hindi".
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Common romanised Hindi function words. Deliberately function words rather than
# nouns: "yojana" appears in English sentences too ("the Ujjwala Yojana"), but
# nobody writes "kitna" or "mujhe" in an English sentence.
_HINGLISH_MARKERS = {
    "kya", "kaise", "kaisa", "kaun", "kon", "kitna", "kitne", "kitni", "kab",
    "kahan", "kahaan", "hai", "hain", "ho", "hoga", "hogi", "milta", "milti",
    "milega", "milegi", "mil", "mujhe", "mera", "meri", "mere", "aap", "aapko",
    "apne", "liye", "nahi", "nahin", "kar", "karna", "karne", "chahiye",
    "paisa", "paise", "rupaye", "sakta", "sakti", "sakte", "wala", "wali",
    "bata", "batao", "jankari", "jaankari", "labh", "patra", "patrata",
}


def detect_language_heuristic(text: str) -> Language:
    """
    Script/word-based language guess with no LLM call.

    Used as the fallback when the LLM is unavailable, and in tests where we do
    not want a network call. Cheap and surprisingly serviceable.
    """
    if _DEVANAGARI.search(text):
        return Language.HINDI

    words = set(re.findall(r"[a-z]+", text.lower()))
    # Two markers, not one: "hai" alone could be a stray token, but two romanised
    # Hindi function words in one short question is decisive.
    if len(words & _HINGLISH_MARKERS) >= 2:
        return Language.HINGLISH

    return Language.ENGLISH


def normalize_query(
    llm: LLMClient | None,
    question: str,
    counter: CallCounter | None = None,
) -> NormalizedQuery:
    """
    Normalize `question` to English and detect its language.

    Pass `llm=None` to skip the LLM entirely and use the heuristic path (handy
    for tests and for the retrieval-only calibration script).
    """
    question = (question or "").strip()

    if llm is None:
        return NormalizedQuery(
            normalized_query=question,
            language=detect_language_heuristic(question),
            fallback_used=True,
        )

    try:
        parsed, _ = llm.complete_json(
            prompt_name="normalize_query",
            system=QUERY_NORMALIZER_SYSTEM,
            user=QUERY_NORMALIZER_USER.format(question=question),
            max_tokens=256,
            counter=counter,
        )

        if not isinstance(parsed, dict):
            raise LLMError("normalizer did not return a JSON object")

        normalized = str(parsed.get("normalized_query") or "").strip()
        raw_language = str(parsed.get("language") or "").strip().lower()

        # Validate rather than trust: an empty rewrite would silently destroy
        # retrieval, so fall back to the original question instead.
        if not normalized:
            normalized = question

        try:
            language = Language(raw_language)
        except ValueError:
            # Unknown tag (e.g. "hindi", "en-IN"): decide it ourselves.
            language = detect_language_heuristic(question)

        return NormalizedQuery(
            normalized_query=normalized,
            language=language,
            fallback_used=False,
        )

    except (LLMError, KeyError, TypeError, ValueError):
        # Degrade, do not fail. Retrieval will be weaker for Hindi/Hinglish, and
        # fallback_used=True records that this happened.
        return NormalizedQuery(
            normalized_query=question,
            language=detect_language_heuristic(question),
            fallback_used=True,
        )
