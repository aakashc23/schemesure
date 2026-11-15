"""
Tests for how eligibility INPUT is cleaned before the rule engine sees it.

The rule engine itself is pure arithmetic and is tested in test_eligibility.py.
This file covers the two places where messy human / LLM input could silently
produce a wrong verdict:

- income parsing (audit finding M4): the old code joined every digit it found, so
  "2.5 lakh" became Rs 25, which passed every income cap -> a false ELIGIBLE.
- occupation vocabulary (audit finding L11): "kisan" or "vegetable seller" used
  to produce NOT_ELIGIBLE where "tell me more" was the honest answer.
"""

from __future__ import annotations

import json

import pytest

from app.llm_client import LLMError, LLMResult
from app.prompts import profile_extractor_system
from app.schemas import EligibilityStatus, Gender, UserProfile
from core.eligibility import (
    KNOWN_CATEGORIES,
    KNOWN_OCCUPATIONS,
    OCCUPATION_SYNONYMS,
    _normalize_occupation,
    check_all_schemes,
    extract_profile,
    load_rules,
    normalize_profile,
    parse_age,
    parse_income,
)

# ==========================================================================
# Income (M4)
# ==========================================================================


class TestParseIncome:
    @pytest.mark.parametrize("value,expected", [
        (200000, 200000),
        (200000.0, 200000),
        ("200000", 200000),
        ("1,50,000", 150000),
        ("1,500,000", 1500000),
        ("2.5 lakh", 250000),
        ("2 lakh", 200000),
        ("2.5 Lakhs", 250000),
        ("3 lac", 300000),
        ("1.5 crore", 15_000_000),
        (" 12,00,000 ", 1_200_000),
    ])
    def test_accepted_forms(self, value, expected):
        assert parse_income(value) == expected

    @pytest.mark.parametrize("value", [
        "1-2 lakh",                       # a range is not a number
        "1,50,000-2,00,000",              # joined digits would be 150000200000
        "12 LPA",                         # unit not on the accepted list
        "about 2 lakh",
        "Rs 2 lakh",                      # strict on purpose: only the listed forms
        "poor", "", "   ", "lakh",
        "200000.0",                       # decimal digit-strings are not plain digit groups
        None, True, False, [], {},
    ])
    def test_rejected_forms_are_none(self, value):
        assert parse_income(value) is None

    @pytest.mark.parametrize("value", [0, 1, 25, 999, 2.5, "500", "999", "0"])
    def test_below_one_thousand_is_a_probable_unit_slip(self, value):
        """A JSON float in lakh (2.5) or a stray small number must not become a
        tiny 'income' that passes every cap."""
        assert parse_income(value) is None

    def test_exactly_one_thousand_is_accepted(self):
        assert parse_income(1000) == 1000

    def test_the_audit_repro_no_longer_yields_a_false_eligible(self):
        """'2.5 lakh' used to become Rs 25 and pass the NMMSS cap of Rs 3,50,000."""
        assert parse_income("2.5 lakh") == 250000      # correct value, not 25
        assert parse_income("3.5 LPA") is None          # unknown unit -> ask, do not guess


class TestParseAge:
    @pytest.mark.parametrize("value,expected", [
        (35, 35), (35.0, 35), ("35", 35), ("35 years", 35), ("35 yrs", 35), (0, 0), (120, 120),
    ])
    def test_accepted(self, value, expected):
        assert parse_age(value) == expected

    @pytest.mark.parametrize("value", ["35-40", "thirty", "young", None, True, -1, 121, "200", []])
    def test_rejected(self, value):
        assert parse_age(value) is None


# ==========================================================================
# Occupation vocabulary (L11)
# ==========================================================================


class TestOccupationNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("farmer", "farmer"),
        ("Farmer", "farmer"),
        ("Street Vendor", "street_vendor"),
        ("street-vendor", "street_vendor"),
        ("kisan", "farmer"),
        ("Cultivator", "farmer"),
        ("agriculturist", "farmer"),
        ("hawker", "street_vendor"),
        ("rehri", "street_vendor"),
        ("thela", "street_vendor"),
        ("vendor", "street_vendor"),
        ("students", "student"),
        ("housewife", "homemaker"),
        ("unorganised worker", "unorganized_worker"),
    ])
    def test_known_words_and_synonyms(self, raw, expected):
        assert _normalize_occupation(raw) == expected

    @pytest.mark.parametrize("raw", [
        "vegetable seller", "astronaut", "agricultural labourer", "labourer",
        "farm labourer", "", "   ", None,
    ])
    def test_unknown_words_become_none(self, raw):
        assert _normalize_occupation(raw) is None

    def test_a_labourer_is_never_treated_as_a_farmer(self):
        """A labourer works on someone else's land; PM-KISAN needs land in the
        applicant's own name. Mapping labourer -> farmer would flip a verdict."""
        assert "labourer" not in OCCUPATION_SYNONYMS
        assert "agricultural_labourer" not in OCCUPATION_SYNONYMS
        assert _normalize_occupation("agricultural labourer") is None

    def test_every_synonym_target_is_in_the_vocabulary(self):
        for target in OCCUPATION_SYNONYMS.values():
            assert target in KNOWN_OCCUPATIONS

    def test_every_occupation_used_by_a_rules_file_is_in_the_vocabulary(self):
        """If a rules file used a word outside the vocabulary, no extracted
        occupation could ever match it. This keeps the data and the code in sync."""
        for rules in load_rules().values():
            for occupation in rules.allowed_occupations or []:
                assert occupation in KNOWN_OCCUPATIONS, (rules.scheme_id, occupation)

    def test_unknown_occupation_gives_need_more_info_not_not_eligible(self):
        profile = normalize_profile(UserProfile(age=35, occupation="vegetable seller"))
        assert profile.occupation is None
        results = {r.scheme_id: r for r in check_all_schemes(profile, load_rules())}
        assert results["pm_kisan"].status == EligibilityStatus.NEED_MORE_INFO
        assert "occupation" in results["pm_kisan"].missing_fields

    def test_kisan_is_now_eligible_for_pm_kisan(self):
        profile = normalize_profile(UserProfile(age=35, occupation="kisan"))
        results = {r.scheme_id: r for r in check_all_schemes(profile, load_rules())}
        assert profile.occupation == "farmer"
        assert results["pm_kisan"].status == EligibilityStatus.ELIGIBLE


class TestNormalizeProfile:
    def test_category_casing_is_canonicalised(self):
        assert normalize_profile(UserProfile(category="obc")).category == "OBC"
        assert normalize_profile(UserProfile(category="general")).category == "General"

    def test_unrecognised_category_is_kept_as_typed(self):
        assert normalize_profile(UserProfile(category="Minority")).category == "Minority"

    def test_other_fields_are_untouched(self):
        original = UserProfile(age=0, annual_income=0, state="Bihar", gender=Gender.FEMALE)
        cleaned = normalize_profile(original)
        assert (cleaned.age, cleaned.annual_income, cleaned.state, cleaned.gender) == (
            0, 0, "Bihar", Gender.FEMALE)

    def test_missing_stays_missing(self):
        cleaned = normalize_profile(UserProfile())
        assert cleaned == UserProfile()


# ==========================================================================
# The prompt is built from the same vocabulary
# ==========================================================================


class TestPromptVocabulary:
    def test_prompt_lists_every_known_occupation_and_category(self):
        prompt = profile_extractor_system(KNOWN_OCCUPATIONS, KNOWN_CATEGORIES)
        for occupation in KNOWN_OCCUPATIONS:
            assert f'"{occupation}"' in prompt
        for category in KNOWN_CATEGORIES:
            assert category in prompt
        assert "<<<" not in prompt                      # no unfilled placeholder

    def test_prompt_defines_income_as_household_income(self):
        prompt = profile_extractor_system(KNOWN_OCCUPATIONS, KNOWN_CATEGORIES)
        assert "household" in prompt.lower()
        assert "family" in prompt.lower()


# ==========================================================================
# extract_profile
# ==========================================================================


class FakeExtractor:
    def __init__(self, payload=None, error: LLMError | None = None):
        self.payload = payload
        self.error = error
        self.system_seen = ""

    def complete_json(self, prompt_name, system, user, max_tokens=None, counter=None):
        self.system_seen = system
        if self.error:
            raise self.error
        result = LLMResult(text=json.dumps(self.payload), prompt_name=prompt_name, model="fake")
        if counter is not None:
            counter.record(result)
        return self.payload, result


class TestExtractProfile:
    def test_llm_failure_is_not_swallowed(self):
        """The old behaviour (an empty profile) passed an outage off as 'you told
        us nothing'. The error must propagate so the API can answer 503."""
        extractor = FakeExtractor(error=LLMError("down", kind="quota"))
        with pytest.raises(LLMError) as info:
            extract_profile(extractor, "I am a farmer")
        assert info.value.kind == "quota"

    def test_non_object_payload_raises(self):
        with pytest.raises(LLMError):
            extract_profile(FakeExtractor(payload=["not", "a", "dict"]), "x")

    def test_fields_are_cleaned_and_normalised(self):
        extractor = FakeExtractor(payload={
            "age": "35 years", "annual_income": "2.5 lakh", "state": "Bihar",
            "occupation": "Kisan", "gender": "Male", "category": "obc",
        })
        profile = extract_profile(extractor, "x")
        assert profile.age == 35
        assert profile.annual_income == 250000
        assert profile.state == "Bihar"
        assert profile.occupation == "farmer"
        assert profile.gender == Gender.MALE
        assert profile.category == "OBC"

    def test_unusable_values_become_none_not_guesses(self):
        extractor = FakeExtractor(payload={
            "age": "young", "annual_income": "1-2 lakh", "state": "unknown",
            "occupation": "vegetable seller", "gender": "n/a", "category": "null",
        })
        assert extract_profile(extractor, "x") == UserProfile()

    def test_the_vocabulary_reaches_the_prompt(self):
        extractor = FakeExtractor(payload={})
        extract_profile(extractor, "x")
        assert '"street_vendor"' in extractor.system_seen
        assert "household" in extractor.system_seen.lower()
