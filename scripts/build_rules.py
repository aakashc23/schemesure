"""
Build one `<scheme_id>.rules.json` per scheme in data/schemes/.

WHY A SCRIPT INSTEAD OF HAND-WRITTEN JSON
-----------------------------------------
The machine-readable rules are a *translation* of prose eligibility text into
numbers. Keeping the translation table in one reviewable file means:
  - every rule sits next to the exact official sentence it came from
    (`rule_sources`), so the whole set can be spot-checked in one place;
  - re-running after a data refresh is deterministic.

RULES OF TRANSLATION (deliberately conservative)
------------------------------------------------
1. `null` means "no restriction stated in the official source". It never means
   "probably no restriction". If the source is silent, the value is null and the
   eligibility engine simply does not test that dimension.
2. A number is only written when the official text states that number.
3. Conditions a flat schema cannot express (conditional rules, "any one of these
   categories", documentary requirements) are NOT forced into the numeric fields.
   They go into `unmodelled_conditions` verbatim, and the engine reports them as
   things a human must still check. This is the honest version: pretending a
   conditional rule is a hard filter would produce confidently wrong answers.

All 15 schemes are central-sector, so `allowed_states` is null throughout.

Run:  python scripts/build_rules.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMES_DIR = ROOT / "data" / "schemes"

# The 7 fields the project contract requires, in a fixed order.
RULE_FIELDS = (
    "min_age",
    "max_age",
    "max_annual_income",
    "allowed_states",
    "allowed_occupations",
    "gender",
    "category",
)

# --------------------------------------------------------------------------
# The translation table.
#   rules               -> the machine-checkable filters
#   rule_sources        -> official sentence behind each non-null rule
#   unmodelled_conditions -> official conditions the flat schema cannot encode
# --------------------------------------------------------------------------

RULES: dict[str, dict] = {
    # ---------------------------------------------------------------- farmers
    "pm_kisan": {
        "rules": {
            "min_age": None,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": ["farmer"],
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "allowed_occupations": (
                "All landholding farmers' families, which have cultivable land "
                "holding in their names are eligible to get benefit under the scheme."
            )
        },
        "unmodelled_conditions": [
            "Must have cultivable land holding in their own name.",
            "Excluded: institutional land holders; income tax payers in the last "
            "assessment year; serving/retired government employees (except Group D / "
            "Class IV / Multi Tasking Staff); retired pensioners with monthly pension "
            "of Rs.10,000 or more; present and former holders of constitutional posts; "
            "registered practising professionals (doctors, engineers, lawyers, "
            "chartered accountants, architects).",
        ],
    },
    "pm_fasal_bima": {
        "rules": {
            "min_age": None,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": ["farmer"],
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "allowed_occupations": (
                "All farmers, including tenant farmers and sharecroppers growing "
                "notified crops in notified areas."
            )
        },
        "unmodelled_conditions": [
            "Must grow notified crops in notified areas.",
            "Must have an insurable interest in the insured crops.",
            "Must possess a valid and authenticated land ownership certificate or a "
            "valid land tenure agreement.",
            "Must apply within the prescribed time frame, usually within 2 weeks of "
            "the start of the sowing season.",
            "Must not have received compensation for the same crop loss from any "
            "other medium or source.",
        ],
    },
    "kisan_credit_card": {
        "rules": {
            "min_age": None,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": ["farmer"],
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "allowed_occupations": (
                "Farmers - individual/joint borrowers who are owner cultivators; "
                "tenant farmers, oral lessees & share croppers; Self Help Groups "
                "(SHGs) or Joint Liability Groups (JLGs) of farmers."
            )
        },
        "unmodelled_conditions": [
            "Eligible as an individual/joint borrower who is an owner cultivator, a "
            "tenant farmer, oral lessee or share cropper, or as a member of an SHG "
            "or Joint Liability Group of farmers."
        ],
    },
    # -------------------------------------------------------------- insurance
    "pmjjby": {
        "rules": {
            "min_age": 18,
            "max_age": 50,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "min_age": "The age of the applicant must be between 18 to 50 Years.",
            "max_age": "The age of the applicant must be between 18 to 50 Years.",
        },
        "unmodelled_conditions": [
            "Must hold an individual bank / post office account."
        ],
    },
    "pmsby": {
        "rules": {
            "min_age": 18,
            "max_age": 70,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "min_age": (
                "Individual bank account holders of participating banks aged between "
                "18 years (completed) and 70 years (age nearer birthday)."
            ),
            "max_age": (
                "Individual bank account holders of participating banks aged between "
                "18 years (completed) and 70 years (age nearer birthday)."
            ),
        },
        "unmodelled_conditions": [
            "Must be an individual bank account holder of a participating bank and "
            "give consent to join / enable auto-debit."
        ],
    },
    # ---------------------------------------------------------------- pension
    "atal_pension_yojana": {
        "rules": {
            "min_age": 18,
            "max_age": 40,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "min_age": "The minimum age of joining APY is 18 years and maximum is 40 years.",
            "max_age": "The minimum age of joining APY is 18 years and maximum is 40 years.",
        },
        "unmodelled_conditions": [
            "Must be a savings bank account holder.",
            "The official overview describes APY as being for a savings account holder "
            "'who is not an income tax-payee'. This is a tax-status condition, not a "
            "stated income ceiling, so no numeric income limit is encoded.",
            "Contributions must be made by auto-debit from the savings bank account "
            "until the age of 60.",
        ],
    },
    # ----------------------------------------------------------------- credit
    "pm_mudra": {
        "rules": {
            "min_age": None,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": None,
            "category": None,
        },
        "rule_sources": {},
        "unmodelled_conditions": [
            "Eligible borrowers: individuals, proprietary concerns, partnership firms, "
            "private limited companies, public companies and any other legal forms.",
            "The applicant should not be a defaulter to any bank or financial "
            "institution and should have a satisfactory credit track record.",
            "Individual borrowers may be required to possess the necessary skills, "
            "experience or knowledge to undertake the proposed activity.",
        ],
    },
    "stand_up_india": {
        "rules": {
            "min_age": 18,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "min_age": "The age of the applicant must be at least 18 years."
        },
        "unmodelled_conditions": [
            "CONDITIONAL RULE: 'If the applicant is a male, he must be from SC / ST "
            "category.' Women applicants are eligible regardless of category. A flat "
            "gender/category filter cannot express this, so neither field is set and "
            "the engine reports this condition for human checking.",
            "Finance is provided for Greenfield Enterprises (first-time venture).",
            "The applicant must not be in default to any bank / financial institution.",
        ],
    },
    "pm_svanidhi": {
        "rules": {
            "min_age": None,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": ["street_vendor"],
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "allowed_occupations": (
                "Street vendors in possession of Certificate of Vending / Identity "
                "Card issued by Urban Local Bodies (ULBs)."
            )
        },
        "unmodelled_conditions": [
            "Must hold a Certificate of Vending / Identity Card from the Urban Local "
            "Body, or have been identified in the ULB survey, or hold a Letter of "
            "Recommendation from the ULB / Town Vending Committee."
        ],
    },
    # ---------------------------------------------------------------- housing
    "pmay_urban": {
        "rules": {
            "min_age": None,
            "max_age": None,
            "max_annual_income": 1800000,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "max_annual_income": (
                "Middle Income Group-2 (MIG-2): households with annual income between "
                "Rs 12,00,001 and Rs 18,00,000. (Rs 18,00,000 is the highest household "
                "income covered by any of the scheme's four income categories: EWS up "
                "to Rs 3,00,000; LIG Rs 3,00,001-6,00,000; MIG-1 Rs 6,00,001-12,00,000; "
                "MIG-2 Rs 12,00,001-18,00,000.)"
            )
        },
        "unmodelled_conditions": [
            "The applicant or their family members must not own a pucca house "
            "anywhere in the country.",
            "The family must comprise husband/wife and unmarried children.",
            "The town/city where the family resides must be covered under the scheme.",
            "The family must not have previously availed benefits of any "
            "housing-related scheme of the Government of India.",
            "Income band determines which vertical of the scheme applies (EWS, LIG, "
            "MIG-1, MIG-2); this engine only checks the overall Rs 18,00,000 ceiling.",
        ],
    },
    # ------------------------------------------------------------- energy/LPG
    "pm_ujjwala": {
        "rules": {
            "min_age": 18,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": "female",
            "category": None,
        },
        "rule_sources": {
            "gender": (
                "An adult woman belonging to a poor household and not having an LPG "
                "connection in her household will be eligible under UJJWALA 2.0."
            ),
            "min_age": (
                "Derived from 'An adult woman...' in the official eligibility text. "
                "The source says 'adult' rather than a number; 18 is used as the "
                "statutory age of adulthood in India."
            ),
        },
        "unmodelled_conditions": [
            "Must belong to a poor household and must not already have an LPG "
            "connection in the household.",
            "Must qualify through ONE of: the SECC 2011 list; SC/ST household, PMAY / "
            "Antyodaya Anna Yojana beneficiary, forest dweller, Most Backward Class, "
            "Tea or Ex-Tea Garden Tribe, or resident of a river island; or a 14-point "
            "declaration as a poor household. Because these are alternatives, no "
            "single category is encoded as a hard filter.",
        ],
    },
    # -------------------------------------------------------------- maternity
    "pm_matru_vandana": {
        "rules": {
            "min_age": 19,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": "female",
            "category": None,
        },
        "rule_sources": {
            "min_age": "The applicant should be of at least 19 years old and a pregnant women.",
            "gender": "The applicant should be of at least 19 years old and a pregnant women.",
        },
        "unmodelled_conditions": [
            "Must be pregnant, employed, and experiencing wage-loss due to the pregnancy.",
            "Applicable only for the first live birth (with a separate provision for a "
            "second girl child in the case of twins/triplets/quadruplets).",
            "Must apply within 270 days from the child's birth.",
            "Must fall in one of the listed socially/economically disadvantaged "
            "categories. 'Net family income less than Rs 8 Lakh per annum' is only ONE "
            "of those alternative routes, so it is NOT encoded as a universal income "
            "ceiling.",
        ],
    },
    # ------------------------------------------------------------- skilling
    "pmkvy_stt": {
        "rules": {
            "min_age": 15,
            "max_age": 45,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": None,
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "min_age": "Is aged between 15-45 years",
            "max_age": "Is aged between 15-45 years",
        },
        "unmodelled_conditions": [
            "Must be of Indian nationality.",
            "Must possess an Aadhaar card and an Aadhaar-linked bank account.",
            "Must fulfil other criteria for the respective job role as defined by the "
            "awarding body.",
        ],
    },
    "pm_vishwakarma": {
        "rules": {
            "min_age": 18,
            "max_age": None,
            "max_annual_income": None,
            "allowed_states": None,
            "allowed_occupations": ["artisan", "craftsperson"],
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "min_age": (
                "On the date of registration for the scheme, the minimum age of the "
                "applicant should be 18 years."
            ),
            "allowed_occupations": (
                "The applicant should be an artisan or craftsperson working with hands "
                "and tools."
            ),
        },
        "unmodelled_conditions": [
            "Must be engaged in the unorganized sector on a self-employment basis.",
            "Must be engaged in one of the 18 family-based traditional trades listed "
            "in the scheme.",
            "Must not have availed loans under similar central or state credit-based "
            "self-employment schemes (e.g. PMEGP, PM SVANidhi, Mudra) in the past 5 years.",
            "Benefits are restricted to one member of the family.",
        ],
    },
    # ------------------------------------------------------------ scholarship
    "nmmss": {
        "rules": {
            "min_age": None,
            "max_age": None,
            "max_annual_income": 350000,
            "allowed_states": None,
            "allowed_occupations": ["student"],
            "gender": None,
            "category": None,
        },
        "rule_sources": {
            "max_annual_income": (
                "Students whose parental income from all sources is not more than "
                "Rs 3,50,000/- per annum are eligible to avail the scholarship."
            ),
            "allowed_occupations": "The applicant must be a student.",
        },
        "unmodelled_conditions": [
            "Must have a minimum of 55% marks or equivalent grade in the Class VII "
            "examination to appear in the selection test (relaxable by 5% for SC/ST "
            "students).",
            "Must be studying as a regular student in a Government, Government-aided "
            "or local body school.",
            "Must pass both the Mental Ability Test and the Scholastic Aptitude Test "
            "with at least 40% marks in aggregate (32% for SC/ST students).",
            "Reservation applies as per State/UT Government norms.",
        ],
    },
}


def main() -> int:
    if not SCHEMES_DIR.exists():
        print(f"error: {SCHEMES_DIR} does not exist. Run fetch_schemes.py first.")
        return 1

    written = 0
    problems: list[str] = []

    for scheme_id, spec in RULES.items():
        md_path = SCHEMES_DIR / f"{scheme_id}.md"
        if not md_path.exists():
            problems.append(f"{scheme_id}: no matching .md file")
            continue

        rules = spec["rules"]

        # Guard against typos in the table: exactly the 7 contract fields.
        missing = set(RULE_FIELDS) - set(rules)
        extra = set(rules) - set(RULE_FIELDS)
        if missing or extra:
            problems.append(f"{scheme_id}: bad rule fields (missing={missing} extra={extra})")
            continue

        # Every non-null rule must cite the official sentence it came from.
        for field, value in rules.items():
            if value is not None and field not in spec["rule_sources"]:
                problems.append(f"{scheme_id}: rule '{field}' has no rule_sources entry")

        document = {
            "scheme_id": scheme_id,
            # Ordered exactly as the project contract specifies.
            **{field: rules[field] for field in RULE_FIELDS},
            "rule_sources": spec["rule_sources"],
            "unmodelled_conditions": spec["unmodelled_conditions"],
        }

        out_path = SCHEMES_DIR / f"{scheme_id}.rules.json"
        with io.open(out_path, "w", encoding="utf-8") as fh:
            json.dump(document, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        written += 1

    print(f"Wrote {written} rules files into {SCHEMES_DIR}")

    # Any .md without rules is a gap worth shouting about.
    for md in sorted(SCHEMES_DIR.glob("*.md")):
        if not (SCHEMES_DIR / f"{md.stem}.rules.json").exists():
            problems.append(f"{md.stem}: .md exists but no rules were defined")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
