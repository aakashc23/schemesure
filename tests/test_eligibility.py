"""
Tests for the pure-Python eligibility engine.

Focus: boundary values (off-by-one errors here change who gets told they qualify
for money), null rules, and missing profile fields. No LLM involved — the engine
is deliberately a pure function so it can be tested like arithmetic.
"""

from __future__ import annotations

import pytest

from app.schemas import EligibilityStatus, Gender, UserProfile
from core.eligibility import SchemeRules, check_all_schemes, check_scheme, load_rules


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def rules(**kwargs) -> SchemeRules:
    """A SchemeRules with everything null unless overridden."""
    base = dict(scheme_id="test_scheme", scheme_name="Test Scheme")
    base.update(kwargs)
    return SchemeRules(**base)


AGE_18_50 = rules(scheme_id="pmjjby", scheme_name="PMJJBY", min_age=18, max_age=50)
INCOME_350K = rules(scheme_id="nmmss", scheme_name="NMMSS", max_annual_income=350_000)
FARMERS = rules(scheme_id="pm_kisan", scheme_name="PM-KISAN", allowed_occupations=["farmer"])
WOMEN_18_PLUS = rules(scheme_id="pmuy", scheme_name="Ujjwala", min_age=18, gender="female")


# ==========================================================================
# Age boundaries
# ==========================================================================


class TestAgeBoundaries:
    """Bounds are inclusive: the sources say "between 18 to 50 years"."""

    @pytest.mark.parametrize("age,expected", [
        (17, EligibilityStatus.NOT_ELIGIBLE),   # one below
        (18, EligibilityStatus.ELIGIBLE),       # exactly the minimum
        (19, EligibilityStatus.ELIGIBLE),
        (49, EligibilityStatus.ELIGIBLE),
        (50, EligibilityStatus.ELIGIBLE),       # exactly the maximum
        (51, EligibilityStatus.NOT_ELIGIBLE),   # one above
    ])
    def test_inclusive_bounds(self, age, expected):
        result = check_scheme(UserProfile(age=age), AGE_18_50)
        assert result.status == expected

    def test_below_minimum_explains_why(self):
        result = check_scheme(UserProfile(age=16), AGE_18_50)
        assert result.status == EligibilityStatus.NOT_ELIGIBLE
        reasons = " ".join(check.reason for check in result.checks)
        assert "16" in reasons and "18" in reasons

    def test_missing_age_is_need_more_info_not_a_guess(self):
        result = check_scheme(UserProfile(), AGE_18_50)
        assert result.status == EligibilityStatus.NEED_MORE_INFO
        assert "age" in result.missing_fields

    def test_age_zero_is_a_real_value_not_missing(self):
        """0 is falsy in Python — a naive `if profile.age:` would treat a
        newborn as "age not provided" and answer NEED_MORE_INFO instead of
        NOT_ELIGIBLE. This pins the correct behaviour."""
        result = check_scheme(UserProfile(age=0), AGE_18_50)
        assert result.status == EligibilityStatus.NOT_ELIGIBLE
        assert "age" not in result.missing_fields

    def test_min_only(self):
        only_min = rules(min_age=60)
        assert check_scheme(UserProfile(age=60), only_min).status == EligibilityStatus.ELIGIBLE
        assert check_scheme(UserProfile(age=59), only_min).status == EligibilityStatus.NOT_ELIGIBLE
        assert check_scheme(UserProfile(age=120), only_min).status == EligibilityStatus.ELIGIBLE

    def test_max_only(self):
        only_max = rules(max_age=25)
        assert check_scheme(UserProfile(age=25), only_max).status == EligibilityStatus.ELIGIBLE
        assert check_scheme(UserProfile(age=26), only_max).status == EligibilityStatus.NOT_ELIGIBLE
        assert check_scheme(UserProfile(age=0), only_max).status == EligibilityStatus.ELIGIBLE


# ==========================================================================
# Income boundaries
# ==========================================================================


class TestIncomeBoundaries:
    @pytest.mark.parametrize("income,expected", [
        (0, EligibilityStatus.ELIGIBLE),
        (349_999, EligibilityStatus.ELIGIBLE),
        (350_000, EligibilityStatus.ELIGIBLE),      # "not more than" is inclusive
        (350_001, EligibilityStatus.NOT_ELIGIBLE),
        (10_000_000, EligibilityStatus.NOT_ELIGIBLE),
    ])
    def test_inclusive_cap(self, income, expected):
        result = check_scheme(UserProfile(annual_income=income), INCOME_350K)
        assert result.status == expected

    def test_zero_income_is_a_real_value(self):
        """Same falsy trap as age zero: income 0 must not read as "not provided"."""
        result = check_scheme(UserProfile(annual_income=0), INCOME_350K)
        assert result.status == EligibilityStatus.ELIGIBLE
        assert "annual_income" not in result.missing_fields

    def test_missing_income_asks(self):
        result = check_scheme(UserProfile(age=30), INCOME_350K)
        assert result.status == EligibilityStatus.NEED_MORE_INFO
        assert "annual_income" in result.missing_fields

    def test_reason_shows_both_numbers(self):
        result = check_scheme(UserProfile(annual_income=500_000), INCOME_350K)
        reasons = " ".join(check.reason for check in result.checks)
        assert "500,000" in reasons and "350,000" in reasons


# ==========================================================================
# Occupation, gender, category, state
# ==========================================================================


class TestCategoricalRules:
    def test_matching_occupation(self):
        result = check_scheme(UserProfile(occupation="farmer"), FARMERS)
        assert result.status == EligibilityStatus.ELIGIBLE

    def test_wrong_occupation(self):
        result = check_scheme(UserProfile(occupation="salaried"), FARMERS)
        assert result.status == EligibilityStatus.NOT_ELIGIBLE

    def test_occupation_matching_is_normalised(self):
        """'Street Vendor', 'street-vendor' and 'street_vendor' are the same thing."""
        vendors = rules(allowed_occupations=["street_vendor"])
        for spelling in ("street_vendor", "Street Vendor", "street-vendor", "STREET VENDOR"):
            result = check_scheme(UserProfile(occupation=spelling), vendors)
            assert result.status == EligibilityStatus.ELIGIBLE, spelling

    def test_multiple_allowed_occupations(self):
        artisans = rules(allowed_occupations=["artisan", "craftsperson"])
        assert check_scheme(UserProfile(occupation="artisan"), artisans).status == EligibilityStatus.ELIGIBLE
        assert check_scheme(UserProfile(occupation="craftsperson"), artisans).status == EligibilityStatus.ELIGIBLE
        assert check_scheme(UserProfile(occupation="farmer"), artisans).status == EligibilityStatus.NOT_ELIGIBLE

    def test_gender_match_and_mismatch(self):
        ok = check_scheme(UserProfile(age=25, gender=Gender.FEMALE), WOMEN_18_PLUS)
        assert ok.status == EligibilityStatus.ELIGIBLE
        no = check_scheme(UserProfile(age=25, gender=Gender.MALE), WOMEN_18_PLUS)
        assert no.status == EligibilityStatus.NOT_ELIGIBLE

    def test_missing_gender_asks(self):
        result = check_scheme(UserProfile(age=25), WOMEN_18_PLUS)
        assert result.status == EligibilityStatus.NEED_MORE_INFO
        assert "gender" in result.missing_fields

    def test_state_restriction(self):
        bihar_only = rules(allowed_states=["Bihar", "Jharkhand"])
        assert check_scheme(UserProfile(state="Bihar"), bihar_only).status == EligibilityStatus.ELIGIBLE
        assert check_scheme(UserProfile(state="bihar"), bihar_only).status == EligibilityStatus.ELIGIBLE
        assert check_scheme(UserProfile(state="Kerala"), bihar_only).status == EligibilityStatus.NOT_ELIGIBLE
        assert check_scheme(UserProfile(), bihar_only).status == EligibilityStatus.NEED_MORE_INFO

    def test_category_restriction(self):
        sc_only = rules(category="SC")
        assert check_scheme(UserProfile(category="SC"), sc_only).status == EligibilityStatus.ELIGIBLE
        assert check_scheme(UserProfile(category="sc"), sc_only).status == EligibilityStatus.ELIGIBLE
        assert check_scheme(UserProfile(category="General"), sc_only).status == EligibilityStatus.NOT_ELIGIBLE


# ==========================================================================
# Null rules and verdict precedence
# ==========================================================================


class TestNullRulesAndPrecedence:
    def test_all_rules_null_is_eligible_with_an_explanation(self):
        """A scheme stating no mechanically checkable restriction (PM Mudra)
        passes, but says so explicitly rather than silently."""
        result = check_scheme(UserProfile(), rules())
        assert result.status == EligibilityStatus.ELIGIBLE
        assert len(result.checks) == 1
        assert result.checks[0].rule == "no_machine_checkable_rules"

    def test_empty_profile_against_all_null_rules(self):
        result = check_scheme(UserProfile(), rules())
        assert result.status == EligibilityStatus.ELIGIBLE
        assert result.missing_fields == []

    def test_null_rules_are_not_checked_at_all(self):
        """An unrestricted dimension must produce no check line, so the reasoning
        shown to the user contains only rules that actually exist."""
        result = check_scheme(
            UserProfile(age=30, annual_income=100, state="Bihar", occupation="farmer"),
            rules(min_age=18),
        )
        assert [check.rule for check in result.checks] == ["min_age"]

    def test_definite_failure_beats_missing_information(self):
        """
        Age 15 against min_age 18 with income unknown: the age failure is
        decisive, so NOT_ELIGIBLE. Reporting NEED_MORE_INFO here would send
        someone to collect documents for a scheme they cannot have.
        """
        combined = rules(min_age=18, max_annual_income=350_000)
        result = check_scheme(UserProfile(age=15), combined)
        assert result.status == EligibilityStatus.NOT_ELIGIBLE

    def test_passing_one_rule_while_missing_another_asks(self):
        combined = rules(min_age=18, max_annual_income=350_000)
        result = check_scheme(UserProfile(age=30), combined)
        assert result.status == EligibilityStatus.NEED_MORE_INFO
        assert result.missing_fields == ["annual_income"]

    def test_all_rules_satisfied(self):
        combined = rules(
            min_age=18, max_age=60, max_annual_income=350_000,
            allowed_occupations=["farmer"], gender="female",
        )
        result = check_scheme(
            UserProfile(age=35, annual_income=200_000, occupation="farmer", gender=Gender.FEMALE),
            combined,
        )
        assert result.status == EligibilityStatus.ELIGIBLE
        assert all(check.passed is True for check in result.checks)

    def test_age_reported_once_when_both_bounds_miss_it(self):
        result = check_scheme(UserProfile(), AGE_18_50)
        assert result.missing_fields == ["age"]  # not ["age", "age"]

    def test_unmodelled_conditions_are_always_surfaced(self):
        """ELIGIBLE must never be mistaken for a guarantee."""
        with_conditions = rules(
            min_age=18,
            unmodelled_conditions=["Must hold an individual bank account."],
        )
        result = check_scheme(UserProfile(age=30), with_conditions)
        assert result.status == EligibilityStatus.ELIGIBLE
        assert result.conditions_to_verify == ["Must hold an individual bank account."]


# ==========================================================================
# check_all_schemes
# ==========================================================================


class TestCheckAllSchemes:
    def test_results_are_ordered_eligible_first(self):
        all_rules = {
            "too_old": rules(scheme_id="too_old", scheme_name="Z Too Old", max_age=10),
            "ok": rules(scheme_id="ok", scheme_name="A Fine", min_age=18),
            "unknown": rules(scheme_id="unknown", scheme_name="M Unknown", max_annual_income=100),
        }
        results = check_all_schemes(UserProfile(age=30), all_rules)
        assert [result.status for result in results] == [
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.NEED_MORE_INFO,
            EligibilityStatus.NOT_ELIGIBLE,
        ]

    def test_every_scheme_is_reported(self):
        all_rules = {f"s{i}": rules(scheme_id=f"s{i}", scheme_name=f"S{i}") for i in range(5)}
        assert len(check_all_schemes(UserProfile(), all_rules)) == 5


# ==========================================================================
# The real rules files on disk
# ==========================================================================


class TestRealRulesFiles:
    """Guards the actual data, not just the engine."""

    def test_all_fifteen_schemes_load(self):
        loaded = load_rules()
        assert len(loaded) == 15

    def test_every_rule_has_a_source_sentence(self):
        """A number with no official sentence behind it is an unverifiable claim."""
        for scheme_id, scheme_rules in load_rules().items():
            for field in ("min_age", "max_age", "max_annual_income",
                          "allowed_occupations", "gender", "category", "allowed_states"):
                if getattr(scheme_rules, field) is not None:
                    assert field in scheme_rules.rule_sources, (
                        f"{scheme_id}.{field} is set but cites no official text"
                    )

    def test_known_values_from_the_official_sources(self):
        """Spot-check a few figures that were read off the official pages."""
        loaded = load_rules()
        assert (loaded["pmjjby"].min_age, loaded["pmjjby"].max_age) == (18, 50)
        assert (loaded["pmsby"].min_age, loaded["pmsby"].max_age) == (18, 70)
        assert (loaded["atal_pension_yojana"].min_age, loaded["atal_pension_yojana"].max_age) == (18, 40)
        assert (loaded["pmkvy_stt"].min_age, loaded["pmkvy_stt"].max_age) == (15, 45)
        assert loaded["nmmss"].max_annual_income == 350_000
        assert loaded["pmay_urban"].max_annual_income == 1_800_000
        assert loaded["pm_matru_vandana"].gender == "female"
        assert loaded["pm_kisan"].allowed_occupations == ["farmer"]

    def test_a_realistic_profile_produces_a_sensible_spread(self):
        """A 35-year-old male farmer should match the farmer schemes and be
        excluded from the ones with a lower age ceiling or a different
        occupation."""
        profile = UserProfile(
            age=35, annual_income=200_000, occupation="farmer",
            state="Bihar", gender=Gender.MALE,
        )
        results = {r.scheme_id: r.status for r in check_all_schemes(profile, load_rules())}

        assert results["pm_kisan"] == EligibilityStatus.ELIGIBLE
        assert results["pmjjby"] == EligibilityStatus.ELIGIBLE        # 18-50
        assert results["atal_pension_yojana"] == EligibilityStatus.ELIGIBLE  # 18-40
        assert results["pm_svanidhi"] == EligibilityStatus.NOT_ELIGIBLE      # street vendors
        assert results["pm_matru_vandana"] == EligibilityStatus.NOT_ELIGIBLE # women only

    def test_gender_is_never_inferred_from_occupation(self):
        """
        The same farmer profile *without* a stated gender must report
        NEED_MORE_INFO for a women-only scheme, not NOT_ELIGIBLE. Guessing
        either way would be wrong: "farmer" says nothing about gender, and
        quietly assuming male would hide a scheme a woman is entitled to.
        """
        profile = UserProfile(age=35, annual_income=200_000, occupation="farmer")
        results = {r.scheme_id: r for r in check_all_schemes(profile, load_rules())}

        matru = results["pm_matru_vandana"]
        assert matru.status == EligibilityStatus.NEED_MORE_INFO
        assert "gender" in matru.missing_fields

    def test_profile_too_old_for_age_capped_schemes(self):
        profile = UserProfile(age=75, occupation="farmer")
        results = {r.scheme_id: r.status for r in check_all_schemes(profile, load_rules())}
        assert results["pmsby"] == EligibilityStatus.NOT_ELIGIBLE               # max 70
        assert results["atal_pension_yojana"] == EligibilityStatus.NOT_ELIGIBLE  # max 40
        assert results["pm_kisan"] == EligibilityStatus.ELIGIBLE                 # no age rule
