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
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.config import Settings, get_settings
from app.llm_client import CallCounter, LLMClient, LLMError
from app.prompts import PROFILE_EXTRACTOR_USER, profile_extractor_system
from app.schemas import (
    EligibilityStatus,
    Gender,
    RuleCheck,
    SchemeEligibility,
    UserProfile,
)
from core.ingest import parse_front_matter


# ==========================================================================
# Vocabularies (ONE source of truth)
# ==========================================================================
# The extraction prompt, the form options (GET /eligibility/options), the synonym
# map and the *.rules.json files must all speak the same words. If they did not,
# an occupation the LLM invents ("vegetable seller") would be compared against a
# rule list and wrongly produce NOT_ELIGIBLE. So the vocabulary lives here, the
# prompt is built from it, and a test checks every rules file only uses it.

KNOWN_OCCUPATIONS: tuple[str, ...] = (
    "farmer", "student", "street_vendor", "artisan", "craftsperson",
    "unorganized_worker", "salaried", "self_employed", "homemaker",
    "unemployed", "retired",
)

# UMBRELLA occupations: they overlap the specific ones. A street vendor, an
# artisan and a farmer are all "self employed" and all "unorganized workers", so
# for these two values "not in the scheme's list" does NOT prove the person is
# outside the scheme. The engine therefore asks a follow-up question (NEED_MORE_INFO)
# instead of returning a firm NOT_ELIGIBLE. (Review finding: false NOT_ELIGIBLE.)
BROAD_OCCUPATIONS: frozenset[str] = frozenset({"self_employed", "unorganized_worker"})

# Categories the prompt asks for and the form offers. No rules file uses a
# category today; the set is what an applicant would name.
KNOWN_CATEGORIES: tuple[str, ...] = ("General", "OBC", "SC", "ST", "EWS")

# Everyday words mapped to the vocabulary above. CONSERVATIVE on purpose: a wrong
# mapping silently changes a verdict, an unmapped word merely becomes "unknown"
# and the engine asks for more information. Keys are in _normalize() form.
OCCUPATION_SYNONYMS: dict[str, str] = {
    # "kisan" is simply the Hindi word for farmer; "cultivator" and
    # "agriculturist" are the standard official English words for someone who
    # farms land. Land ownership is a separate condition that the schemes list
    # under "conditions still to check yourself", so mapping the word is safe.
    "kisan": "farmer",
    "cultivator": "farmer",
    "agriculturist": "farmer",
    # A hawker / rehri / thela is, by definition, someone who sells goods on the
    # street (the rehri and thela are the handcarts). Same trade as a vendor.
    "hawker": "street_vendor",
    "rehri": "street_vendor",
    "thela": "street_vendor",
    "vendor": "street_vendor",
    "street_hawker": "street_vendor",
    "street_seller": "street_vendor",
    # Plural of an existing value.
    "students": "student",
    "farmers": "farmer",
    "street_vendors": "street_vendor",
    # British spelling of an existing value.
    "unorganised_worker": "unorganized_worker",
    # The prompt already tells the model to use homemaker for a housewife.
    "housewife": "homemaker",
    # Deliberately NOT mapped: labourer / agricultural labourer. A labourer works
    # on someone else's land and is NOT a landholding farmer, so PM-KISAN must
    # not treat them as one. They stay unknown -> NEED_MORE_INFO.
}


def _normalize(value: str | None) -> str:
    """Lowercase, trim, and collapse separators so 'Street Vendor' == 'street_vendor'."""
    if value is None:
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_occupation(value: str | None) -> str | None:
    """
    Map a free-text occupation onto KNOWN_OCCUPATIONS, or None if unknown.

    Applied to the LLM's extraction AND to structured form input, so both paths
    behave identically. Unknown -> None -> NEED_MORE_INFO ("tell me your
    occupation"), never a false NOT_ELIGIBLE (audit finding L11).
    """
    key = _normalize(value)
    if not key:
        return None
    key = OCCUPATION_SYNONYMS.get(key, key)
    return key if key in KNOWN_OCCUPATIONS else None


def _canonical_category(value: str | None) -> str | None:
    """'obc' -> 'OBC'. Unrecognised text is kept as typed (rules compare loosely)."""
    if value is None:
        return None
    cleaned = value.strip()
    for known in KNOWN_CATEGORIES:
        if cleaned.lower() == known.lower():
            return known
    return cleaned or None


def normalize_profile(profile: UserProfile) -> UserProfile:
    """
    Apply the occupation whitelist/synonyms (and category casing) to a profile.

    Used for structured input (POST /eligibility/check), where the client sends
    a profile directly instead of free text.
    """
    return profile.model_copy(update={
        "occupation": _normalize_occupation(profile.occupation),
        "category": _canonical_category(profile.category),
    })


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

# "2.5 lakh", "1.5 crore". Decimal x is allowed; nothing else is.
_INCOME_UNIT_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)\s*$", re.IGNORECASE
)
_INCOME_DIGITS_RE = re.compile(r"^\s*\d[\d,]*\s*$")
_AGE_RE = re.compile(r"^\s*(\d{1,3})\s*(?:years?|yrs?|yo)?\s*$", re.IGNORECASE)

# Below this a "yearly income" is almost certainly a unit slip (a value in lakh,
# or a monthly figure), and guessing would produce a false ELIGIBLE.
MIN_PLAUSIBLE_ANNUAL_INCOME = 1000


def parse_income(value) -> int | None:
    """
    Turn whatever the extractor returned into rupees per year, or None.

    ACCEPTED (anything else -> None, which the engine treats as "not provided"):
      - an int or float JSON number, taken as rupees;
      - a digit string with optional separators: "150000", "1,50,000";
      - an explicit "<x> lakh" / "<x> crore" with a decimal x: "2.5 lakh".
    A result under Rs 1,000 is rejected as a probable unit mistake.

    WHY SO STRICT: the old code joined every digit it found, so "2.5 lakh"
    became 25 and "1-2 lakh" became 12; Rs 25 then passed every income cap,
    giving a false ELIGIBLE (audit finding M4).
    """
    if value is None or isinstance(value, bool):
        return None

    amount: Decimal | None = None
    if isinstance(value, (int, float)):
        try:
            amount = Decimal(str(value))
        except InvalidOperation:
            return None
    elif isinstance(value, str):
        text = value.strip()
        if _INCOME_DIGITS_RE.match(text):
            amount = Decimal(text.replace(",", ""))
        else:
            match = _INCOME_UNIT_RE.match(text)
            if match:
                unit = match.group(2).lower()
                multiplier = 10_000_000 if unit.startswith("crore") else 100_000
                amount = Decimal(match.group(1)) * multiplier
    if amount is None:
        return None

    rupees = int(amount.to_integral_value())
    return rupees if rupees >= MIN_PLAUSIBLE_ANNUAL_INCOME else None


def parse_age(value) -> int | None:
    """An int, a whole float, or "35" / "35 years". Anything else -> None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        age = int(value)
    elif isinstance(value, str):
        match = _AGE_RE.match(value)
        if not match:
            return None
        age = int(match.group(1))
    else:
        return None
    return age if 0 <= age <= 120 else None


def extract_profile(
    llm: LLMClient,
    description: str,
    counter: CallCounter | None = None,
) -> UserProfile:
    """
    Turn free text into a UserProfile. Unstated fields stay None.

    An LLM failure is NOT swallowed: it raises LLMError. The old behaviour (return
    an empty profile) made an outage look like "the person told us nothing", so
    the UI claimed fields were deliberately left blank. The API turns the error
    into a clear 503 and points to the form-based check, which needs no LLM.
    """
    parsed, _ = llm.complete_json(
        prompt_name="extract_profile",
        system=profile_extractor_system(KNOWN_OCCUPATIONS, KNOWN_CATEGORIES),
        user=PROFILE_EXTRACTOR_USER.format(description=description),
        max_tokens=400,
        counter=counter,
    )

    if not isinstance(parsed, dict):
        raise LLMError("profile extractor did not return a JSON object", kind="bad_output")

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

    gender = None
    raw_gender = clean("gender")
    if raw_gender is not None:
        try:
            gender = Gender(str(raw_gender).strip().lower())
        except ValueError:
            gender = None

    raw_state = clean("state")
    raw_category = clean("category")

    try:
        return UserProfile(
            age=parse_age(clean("age")),
            annual_income=parse_income(clean("annual_income")),
            state=str(raw_state) if raw_state is not None else None,
            # Whitelist + synonyms: unknown occupations become None.
            occupation=_normalize_occupation(
                str(clean("occupation")) if clean("occupation") is not None else None
            ),
            gender=gender,
            category=_canonical_category(
                str(raw_category) if raw_category is not None else None
            ),
        )
    except ValueError:
        # Pydantic rejected something; an empty profile is the safe fallback for
        # *malformed values* (not for a failed call, which raised above).
        return UserProfile()


# ==========================================================================
# The rule engine (pure Python — no LLM, no I/O, fully unit-tested)
# ==========================================================================


def format_inr(amount: int) -> str:
    """
    Group digits the Indian way: 1800000 -> "18,00,000" (not "1,800,000").

    The official scheme pages write amounts like this, so the reasons we show
    should match what the user will read on the government site.
    """
    sign = "-" if amount < 0 else ""
    digits = str(abs(int(amount)))
    if len(digits) <= 3:
        return sign + digits
    head, tail = digits[:-3], digits[-3:]
    # Everything before the last three digits is grouped in pairs.
    pairs = []
    while len(head) > 2:
        pairs.insert(0, head[-2:])
        head = head[:-2]
    if head:
        pairs.insert(0, head)
    return sign + ",".join(pairs) + "," + tail


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
                f"Annual income must not exceed Rs {format_inr(limit)}, but your income was not provided.",
            )
        elif profile.annual_income > limit:
            failed = True
            checks.append(RuleCheck(
                rule="max_annual_income", passed=False,
                reason=f"Your annual income Rs {format_inr(profile.annual_income)} is above the limit of Rs {format_inr(limit)}.",
                official_text=source_for("max_annual_income"),
            ))
        else:
            checks.append(RuleCheck(
                rule="max_annual_income", passed=True,
                reason=f"Your annual income Rs {format_inr(profile.annual_income)} is within the limit of Rs {format_inr(limit)}.",
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
        elif (_normalize(profile.occupation) not in allowed
              and _normalize(profile.occupation) in BROAD_OCCUPATIONS):
            # Umbrella value: could still be a vendor / farmer / artisan. Ask, don't reject.
            need("allowed_occupations", "occupation",
                 f"This scheme is for: {readable}. '{profile.occupation}' is a broad term, so please "
                 "tell us your exact work (for example street vendor, farmer or artisan).")
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
