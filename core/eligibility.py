"""
Rule-based eligibility checking.

THE CENTRAL DESIGN RULE OF THIS FILE
------------------------------------
The LLM extracts, Python decides. The model's only job is turning "I'm a 35 year
old farmer from Bihar earning about 2 lakh a year" into
{age: 35, occupation: "farmer", state: "Bihar", annual_income: 200000}.
Every comparison after that is an `if` statement.

Why not let the LLM decide eligibility? Three reasons:
  1. It is arithmetic and set membership. Code is exactly right, every time;
     an LLM is approximately right, most of the time.
  2. It must be auditable. "Not eligible: your age 35 is above the scheme's
     maximum of 25" is a reason a person can check and dispute. "The model said
     no" is not.
  3. These answers affect whether someone applies for money they are entitled
     to. A silent arithmetic slip is unacceptable in a way that a slightly
     awkward sentence is not.

THE THIRD STATE
---------------
Verdicts are ELIGIBLE / NOT_ELIGIBLE / NEED_MORE_INFO. The third one is the
important one. If a scheme has an income cap and the user never mentioned their
income, the honest answer is "tell me your income", not a guess in either
direction. Missing data is a question, not a default.

WHAT A FLAT RULE SCHEMA CANNOT DO
---------------------------------
Some official conditions do not fit seven numeric fields — Stand-Up India's
"if the applicant is male, he must be SC/ST" is conditional; PM Ujjwala's
categories are alternatives, not requirements. Those are carried in
`unmodelled_conditions` and always surfaced as things a human must still check.
So ELIGIBLE here means "passes every rule we can mechanically check", never
"you will definitely receive this benefit", and the UI says so.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, get_settings
from app.llm_client import CallCounter, LLMClient, LLMError
from app.prompts import PROFILE_EXTRACTOR_SYSTEM, PROFILE_EXTRACTOR_USER
from app.schemas import (
    EligibilityStatus,
    Gender,
    RuleCheck,
    SchemeEligibility,
    UserProfile,
)
from core.ingest import parse_front_matter


# ==========================================================================
# Rules loading
# ==========================================================================


@dataclass
class SchemeRules:
    """One scheme's machine-checkable rules, plus what could not be modelled."""

    scheme_id: str
    scheme_name: str
    source_url: str = ""
    min_age: int | None = None
    max_age: int | None = None
    max_annual_income: int | None = None
    allowed_states: list[str] | None = None
    allowed_occupations: list[str] | None = None
    gender: str | None = None
    category: str | None = None
    rule_sources: dict[str, str] = field(default_factory=dict)
    unmodelled_conditions: list[str] = field(default_factory=list)


def _normalize(value: str | None) -> str:
    """Lowercase, trim, and collapse separators so 'Street Vendor' == 'street_vendor'."""
    if value is None:
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def load_rules(settings: Settings | None = None) -> dict[str, SchemeRules]:
    """
    Read every `*.rules.json` and pair it with its scheme's name and URL.

    Returns {scheme_id: SchemeRules}. A scheme without a rules file is skipped
    (it simply cannot be checked), not faked with empty rules.
    """
    settings = settings or get_settings()
    data_dir: Path = settings.data_dir
    rules: dict[str, SchemeRules] = {}

    for rules_path in sorted(data_dir.glob("*.rules.json")):
        try:
            payload = json.loads(io.open(rules_path, encoding="utf-8").read())
        except (OSError, json.JSONDecodeError):
            continue

        scheme_id = str(payload.get("scheme_id") or rules_path.stem.replace(".rules", ""))

        # Pull the display name and source URL from the matching markdown file,
        # so the rules file never has to repeat them.
        scheme_name, source_url = scheme_id, ""
        md_path = data_dir / f"{scheme_id}.md"
        if md_path.exists():
            front, _ = parse_front_matter(io.open(md_path, encoding="utf-8").read())
            scheme_name = front.get("name") or scheme_id
            source_url = front.get("source_url", "")

        def as_list(value) -> list[str] | None:
            if value is None:
                return None
            if isinstance(value, list):
                items = [str(item).strip() for item in value if str(item).strip()]
                return items or None
            text = str(value).strip()
            return [text] if text else None

        def as_int(value) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        rules[scheme_id] = SchemeRules(
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            source_url=source_url,
            min_age=as_int(payload.get("min_age")),
            max_age=as_int(payload.get("max_age")),
            max_annual_income=as_int(payload.get("max_annual_income")),
            allowed_states=as_list(payload.get("allowed_states")),
            allowed_occupations=as_list(payload.get("allowed_occupations")),
            gender=(str(payload.get("gender")).strip() if payload.get("gender") else None),
            category=(
                str(payload.get("category")).strip() if payload.get("category") else None
            ),
            rule_sources=dict(payload.get("rule_sources") or {}),
            unmodelled_conditions=list(payload.get("unmodelled_conditions") or []),
        )

    return rules


# ==========================================================================
# Profile extraction (the only LLM call in this flow)
# ==========================================================================


def extract_profile(
    llm: LLMClient,
    description: str,
    counter: CallCounter | None = None,
) -> UserProfile:
    """
    Turn free text into a UserProfile. Unstated fields stay None.

    On failure we return an empty profile rather than raising: every scheme then
    reports NEED_MORE_INFO, which is the correct, safe outcome when we know
    nothing about the user.
    """
    try:
        parsed, _ = llm.complete_json(
            prompt_name="extract_profile",
            system=PROFILE_EXTRACTOR_SYSTEM,
            user=PROFILE_EXTRACTOR_USER.format(description=description),
            max_tokens=400,
            counter=counter,
        )
    except LLMError:
        return UserProfile()

    if not isinstance(parsed, dict):
        return UserProfile()

    def clean(key: str):
        value = parsed.get(key)
        # Models sometimes write the string "null"/"none"/"unknown" instead of JSON null.
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in (
            "", "null", "none", "n/a", "na", "unknown", "not specified", "not stated",
        ):
            return None
        return value

    def as_int(value):
        if value is None:
            return None
        try:
            # Tolerate "35", "35 years", 35.0
            if isinstance(value, str):
                digits = "".join(ch for ch in value if ch.isdigit())
                return int(digits) if digits else None
            return int(value)
        except (TypeError, ValueError):
            return None

    gender = None
    raw_gender = clean("gender")
    if raw_gender is not None:
        try:
            gender = Gender(str(raw_gender).strip().lower())
        except ValueError:
            gender = None

    age = as_int(clean("age"))
    income = as_int(clean("annual_income"))

    try:
        return UserProfile(
            age=age if (age is None or 0 <= age <= 120) else None,
            annual_income=income if (income is None or income >= 0) else None,
            state=(str(clean("state")) if clean("state") is not None else None),
            occupation=(
                _normalize(str(clean("occupation")))
                if clean("occupation") is not None
                else None
            ),
            gender=gender,
            category=(str(clean("category")) if clean("category") is not None else None),
        )
    except ValueError:
        # Pydantic rejected something; an empty profile is the safe fallback.
        return UserProfile()


# ==========================================================================
# The rule engine (pure Python — no LLM, no I/O, fully unit-tested)
# ==========================================================================


def check_scheme(profile: UserProfile, rules: SchemeRules) -> SchemeEligibility:
    """
    Test `profile` against one scheme's rules.

    Pure function. Returns a per-rule trace alongside the verdict, because the
    reasoning is the product here — a bare ELIGIBLE/NOT_ELIGIBLE is not useful
    to someone deciding whether to spend a morning at a government office.
    """
    checks: list[RuleCheck] = []
    missing: list[str] = []
    failed = False

    def source_for(rule: str) -> str:
        return rules.rule_sources.get(rule, "")

    def need(rule: str, field_name: str, message: str) -> None:
        """Record "cannot tell without this field"."""
        nonlocal missing
        if field_name not in missing:
            missing.append(field_name)
        checks.append(
            RuleCheck(rule=rule, passed=None, reason=message, official_text=source_for(rule))
        )

    # ---- age -------------------------------------------------------------
    if rules.min_age is not None:
        if profile.age is None:
            need("min_age", "age", f"Minimum age is {rules.min_age}, but your age was not provided.")
        elif profile.age < rules.min_age:
            failed = True
            checks.append(RuleCheck(
                rule="min_age", passed=False,
                reason=f"Your age {profile.age} is below the minimum age of {rules.min_age}.",
                official_text=source_for("min_age"),
            ))
        else:
            checks.append(RuleCheck(
                rule="min_age", passed=True,
                reason=f"Your age {profile.age} meets the minimum age of {rules.min_age}.",
                official_text=source_for("min_age"),
            ))

    if rules.max_age is not None:
        if profile.age is None:
            need("max_age", "age", f"Maximum age is {rules.max_age}, but your age was not provided.")
        elif profile.age > rules.max_age:
            failed = True
            checks.append(RuleCheck(
                rule="max_age", passed=False,
                reason=f"Your age {profile.age} is above the maximum age of {rules.max_age}.",
                official_text=source_for("max_age"),
            ))
        else:
            checks.append(RuleCheck(
                rule="max_age", passed=True,
                reason=f"Your age {profile.age} is within the maximum age of {rules.max_age}.",
                official_text=source_for("max_age"),
            ))

    # ---- income ----------------------------------------------------------
    if rules.max_annual_income is not None:
        limit = rules.max_annual_income
        if profile.annual_income is None:
            need(
                "max_annual_income", "annual_income",
                f"Annual income must not exceed Rs {limit:,}, but your income was not provided.",
            )
        elif profile.annual_income > limit:
            failed = True
            checks.append(RuleCheck(
                rule="max_annual_income", passed=False,
                reason=f"Your annual income Rs {profile.annual_income:,} is above the limit of Rs {limit:,}.",
                official_text=source_for("max_annual_income"),
            ))
        else:
            checks.append(RuleCheck(
                rule="max_annual_income", passed=True,
                reason=f"Your annual income Rs {profile.annual_income:,} is within the limit of Rs {limit:,}.",
                official_text=source_for("max_annual_income"),
            ))

    # ---- state -----------------------------------------------------------
    if rules.allowed_states:
        allowed = {_normalize(state) for state in rules.allowed_states}
        if profile.state is None:
            need("allowed_states", "state", "This scheme is limited to certain states, but your state was not provided.")
        elif _normalize(profile.state) not in allowed:
            failed = True
            checks.append(RuleCheck(
                rule="allowed_states", passed=False,
                reason=f"This scheme is limited to {', '.join(rules.allowed_states)}; you are in {profile.state}.",
                official_text=source_for("allowed_states"),
            ))
        else:
            checks.append(RuleCheck(
                rule="allowed_states", passed=True,
                reason=f"{profile.state} is covered by this scheme.",
                official_text=source_for("allowed_states"),
            ))

    # ---- occupation ------------------------------------------------------
    if rules.allowed_occupations:
        allowed = {_normalize(occupation) for occupation in rules.allowed_occupations}
        readable = ", ".join(rules.allowed_occupations)
        if profile.occupation is None:
            need("allowed_occupations", "occupation", f"This scheme is for: {readable}. Your occupation was not provided.")
        elif _normalize(profile.occupation) not in allowed:
            failed = True
            checks.append(RuleCheck(
                rule="allowed_occupations", passed=False,
                reason=f"This scheme is for: {readable}. Your occupation is {profile.occupation}.",
                official_text=source_for("allowed_occupations"),
            ))
        else:
            checks.append(RuleCheck(
                rule="allowed_occupations", passed=True,
                reason=f"Your occupation ({profile.occupation}) is covered by this scheme.",
                official_text=source_for("allowed_occupations"),
            ))

    # ---- gender ----------------------------------------------------------
    if rules.gender:
        if profile.gender is None:
            need("gender", "gender", f"This scheme is only for applicants of gender: {rules.gender}. Your gender was not provided.")
        elif _normalize(profile.gender.value) != _normalize(rules.gender):
            failed = True
            checks.append(RuleCheck(
                rule="gender", passed=False,
                reason=f"This scheme is only for {rules.gender} applicants.",
                official_text=source_for("gender"),
            ))
        else:
            checks.append(RuleCheck(
                rule="gender", passed=True,
                reason=f"This scheme is for {rules.gender} applicants, which matches your profile.",
                official_text=source_for("gender"),
            ))

    # ---- category --------------------------------------------------------
    if rules.category:
        if profile.category is None:
            need("category", "category", f"This scheme requires category: {rules.category}. Your category was not provided.")
        elif _normalize(profile.category) != _normalize(rules.category):
            failed = True
            checks.append(RuleCheck(
                rule="category", passed=False,
                reason=f"This scheme requires category {rules.category}; your category is {profile.category}.",
                official_text=source_for("category"),
            ))
        else:
            checks.append(RuleCheck(
                rule="category", passed=True,
                reason=f"Your category ({profile.category}) matches the requirement.",
                official_text=source_for("category"),
            ))

    # ---- no modelled rules at all ----------------------------------------
    if not checks:
        checks.append(RuleCheck(
            rule="no_machine_checkable_rules", passed=True,
            reason=(
                "The official source states no age, income, state, occupation, gender "
                "or category restriction that can be checked automatically. See the "
                "conditions below, which still need to be verified by hand."
            ),
        ))

    # ---- verdict ---------------------------------------------------------
    # A definite failure outranks missing information: if you are 15 and the
    # minimum age is 18, you are not eligible regardless of unknown income.
    if failed:
        status = EligibilityStatus.NOT_ELIGIBLE
    elif missing:
        status = EligibilityStatus.NEED_MORE_INFO
    else:
        status = EligibilityStatus.ELIGIBLE

    return SchemeEligibility(
        scheme_id=rules.scheme_id,
        scheme_name=rules.scheme_name,
        status=status,
        checks=checks,
        missing_fields=missing,
        conditions_to_verify=list(rules.unmodelled_conditions),
        source_url=rules.source_url,
    )


def check_all_schemes(
    profile: UserProfile,
    all_rules: dict[str, SchemeRules],
) -> list[SchemeEligibility]:
    """
    Check every scheme and return the results most-useful-first:
    ELIGIBLE, then NEED_MORE_INFO, then NOT_ELIGIBLE; alphabetical within each.
    """
    results = [check_scheme(profile, rules) for rules in all_rules.values()]

    order = {
        EligibilityStatus.ELIGIBLE: 0,
        EligibilityStatus.NEED_MORE_INFO: 1,
        EligibilityStatus.NOT_ELIGIBLE: 2,
    }
    results.sort(key=lambda result: (order[result.status], result.scheme_name))
    return results
