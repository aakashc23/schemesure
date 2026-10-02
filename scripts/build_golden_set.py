"""
Build eval/golden_set.jsonl — and verify every expected fact against the data.

WHY THE VERIFICATION STEP MATTERS
A golden set written from memory is worse than no golden set: it measures the
author's recollection, not the system. So for every answerable question this
script asserts that each `key_fact` string actually appears in the expected
scheme's markdown file. If a fact is not there, the build fails and the question
has to be fixed. That makes "the facts match the data files" a checked property
rather than a promise.

THE 15 UNANSWERABLE QUESTIONS ARE NOT FILLER
They are the three distinct ways this system can fail, and they are separated on
purpose because the defence against each one is different:
  - off_topic    : nothing to do with schemes. Caught by the retrieval threshold.
  - fake_scheme  : a scheme that does not exist. Must be caught by the guardrail
                   — a fake "Free Laptop Yojana" retrieves real scheme chunks at
                   0.61 similarity, well above the threshold.
  - not_indexed  : a REAL scheme we do not cover (Sukanya Samriddhi, PM-JAY).
                   The hardest case, and the one where a naive RAG system
                   confidently answers from the wrong scheme's text.

Run:  python scripts/build_golden_set.py
"""

from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMES_DIR = ROOT / "data" / "schemes"
OUT_PATH = ROOT / "eval" / "golden_set.jsonl"


def normalize(text: str) -> str:
    """
    Normalise for fact matching: lowercase, NFKC, and collapse whitespace.

    Indian digit grouping is inconsistent across the source pages ("Rs 6000",
    "Rs. 6,000", "₹6000"), so a second comma-stripped form is compared too.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text)


def fact_present(fact: str, document: str) -> bool:
    """True if `fact` appears in `document`, ignoring case, spacing and commas."""
    normalized_fact = normalize(fact)
    if normalized_fact in document:
        return True
    # Try again with all commas removed from both sides.
    return normalized_fact.replace(",", "") in document.replace(",", "")


# ==========================================================================
# ANSWERABLE QUESTIONS (35)
# (question, language, expected_scheme, [key_facts], category)
# ==========================================================================

ANSWERABLE: list[tuple[str, str, str, list[str], str]] = [
    # ---- PM-KISAN ----
    ("How much money does PM-KISAN give per year?", "en", "pm_kisan",
     ["6000"], "benefit_amount"),
    ("In how many instalments is the PM-KISAN amount paid?", "en", "pm_kisan",
     ["three equal installments", "2000"], "benefit_structure"),
    ("पीएम किसान योजना में किसानों को साल में कितने रुपये मिलते हैं?", "hi", "pm_kisan",
     ["6000"], "benefit_amount"),
    ("PM Kisan ka paisa kitne installment mein milta hai?", "hinglish", "pm_kisan",
     ["three equal installments"], "benefit_structure"),
    ("Are income tax payers eligible for PM-KISAN?", "en", "pm_kisan",
     ["paid Income Tax in last assessment year"], "exclusion"),

    # ---- PMJJBY ----
    ("What is the life cover amount under Pradhan Mantri Jeevan Jyoti Bima Yojana?",
     "en", "pmjjby", ["2.00 Lakh"], "benefit_amount"),
    ("What is the annual premium for PMJJBY?", "en", "pmjjby",
     ["436"], "cost"),
    ("पीएम जीवन ज्योति बीमा योजना के लिए आयु सीमा क्या है?", "hi", "pmjjby",
     ["18 to 50 Years"], "eligibility_age"),

    # ---- PMSBY ----
    ("How much does the nominee get on death under PM Suraksha Bima Yojana?",
     "en", "pmsby", ["2 Lakhs"], "benefit_amount"),
    ("What is the maximum age to join Pradhan Mantri Suraksha Bima Yojana?",
     "en", "pmsby", ["70 years"], "eligibility_age"),
    ("PMSBY mein ek aankh ki roshni jaane par kitna milta hai?", "hinglish", "pmsby",
     ["1 Lakh"], "benefit_amount"),

    # ---- Atal Pension Yojana ----
    ("What is the minimum and maximum age to join Atal Pension Yojana?",
     "en", "atal_pension_yojana", ["18 years and maximum is 40 years"], "eligibility_age"),
    ("अटल पेंशन योजना में पेंशन कब से शुरू होती है?", "hi", "atal_pension_yojana",
     ["60 years"], "benefit_timing"),

    # ---- PM Ujjwala ----
    ("How much deposit support is given for a 14.2 kg cylinder under PM Ujjwala?",
     "en", "pm_ujjwala", ["1600"], "benefit_amount"),
    ("What documents are required for PM Ujjwala?", "en", "pm_ujjwala",
     ["Aadhaar"], "documents"),
    ("Ujjwala yojana ke liye kaun eligible hai?", "hinglish", "pm_ujjwala",
     ["adult woman"], "eligibility"),

    # ---- NMMSS ----
    ("What is the parental income limit for the National Means-cum-Merit Scholarship?",
     "en", "nmmss", ["3,50,000"], "eligibility_income"),
    ("How much scholarship is given per year under NMMSS?", "en", "nmmss",
     ["12,000"], "benefit_amount"),
    ("What minimum marks are needed in Class VII to apply for NMMSS?", "en", "nmmss",
     ["55 %"], "eligibility_marks"),

    # ---- PM SVANidhi ----
    ("How much loan can a street vendor get under PM SVANidhi?", "en", "pm_svanidhi",
     ["10,000"], "benefit_amount"),
    ("What is the interest rate on the PM SVANidhi loan?", "en", "pm_svanidhi",
     ["7%"], "cost"),
    ("PM SVANidhi ke liye kaun apply kar sakta hai?", "hinglish", "pm_svanidhi",
     ["Street vendors"], "eligibility"),

    # ---- PM Mudra ----
    ("What are the loan categories under Pradhan Mantri Mudra Yojana?",
     "en", "pm_mudra", ["Shishu", "Kishore", "Tarun"], "benefit_structure"),
    ("What is the maximum loan under the Tarun category of Mudra Yojana?",
     "en", "pm_mudra", ["10 lakhs"], "benefit_amount"),
    ("मुद्रा योजना की शिशु श्रेणी में कितना ऋण मिलता है?", "hi", "pm_mudra",
     ["50,000"], "benefit_amount"),

    # ---- Stand-Up India ----
    ("What is the loan range under Stand-Up India?", "en", "stand_up_india",
     ["10 Lakhs", "100 Lakhs"], "benefit_amount"),
    ("Who is eligible for a Stand-Up India loan if the applicant is male?",
     "en", "stand_up_india", ["SC / ST category"], "eligibility"),

    # ---- PM Vishwakarma ----
    ("What is the toolkit incentive under PM Vishwakarma?", "en", "pm_vishwakarma",
     ["15,000"], "benefit_amount"),
    ("What training stipend is given under PM Vishwakarma?", "en", "pm_vishwakarma",
     ["500 per day"], "benefit_amount"),

    # ---- PMAY Urban ----
    ("What is the EWS annual household income limit under PMAY Urban?",
     "en", "pmay_urban", ["3,00,000"], "eligibility_income"),
    ("PMAY Urban mein MIG 2 ki income limit kya hai?", "hinglish", "pmay_urban",
     ["18,00,000"], "eligibility_income"),

    # ---- PMMVY ----
    ("How much is the first instalment under Pradhan Mantri Matru Vandana Yojana?",
     "en", "pm_matru_vandana", ["3,000"], "benefit_amount"),

    # ---- PMKVY / PMFBY ----
    ("What is the age range for Pradhan Mantri Kaushal Vikas Yojana short term training?",
     "en", "pmkvy_stt", ["15-45 years"], "eligibility_age"),
    ("What premium does a farmer pay for Kharif crops under PM Fasal Bima Yojana?",
     "en", "pm_fasal_bima", ["2%"], "cost"),
    ("Who is eligible for a Kisan Credit Card?", "en", "kisan_credit_card",
     ["owner cultivators"], "eligibility"),
]

# ==========================================================================
# UNANSWERABLE QUESTIONS (15)
# (question, language, category)
# ==========================================================================

UNANSWERABLE: list[tuple[str, str, str]] = [
    # --- off_topic: should be stopped by the retrieval threshold ---
    ("What is the capital of France?", "en", "off_topic"),
    ("Give me a good recipe for pizza dough.", "en", "off_topic"),
    ("How do I reset my Gmail password?", "en", "off_topic"),
    ("Who won the ICC Cricket World Cup in 2011?", "en", "off_topic"),
    ("भारत का राष्ट्रीय खेल कौन सा है?", "hi", "off_topic"),

    # --- fake_scheme: invented names that sound plausible ---
    ("Tell me about the Pradhan Mantri Free Laptop Yojana 2026.", "en", "fake_scheme"),
    ("How much money does the PM Berojgari Bhatta Yojana give every month?",
     "en", "fake_scheme"),
    ("प्रधानमंत्री मुफ्त स्कूटी योजना के लिए आवेदन कैसे करें?", "hi", "fake_scheme"),
    ("PM Shadi Shagun Yojana mein kitna paisa milta hai?", "hinglish", "fake_scheme"),
    ("What is the eligibility for the National Digital Pension Scheme 2027?",
     "en", "fake_scheme"),

    # --- not_indexed: REAL schemes outside our 15. The hardest case. ---
    ("What is the current interest rate on Sukanya Samriddhi Yojana?",
     "en", "not_indexed"),
    ("How do I get an Ayushman Bharat PM-JAY health card?", "en", "not_indexed"),
    ("आयुष्मान भारत में कितने लाख तक का इलाज मुफ्त है?", "hi", "not_indexed"),
    ("Atal Innovation Mission ke under startup ko kitna funding milta hai?",
     "hinglish", "not_indexed"),

    # --- trick: a real scheme, but asking for a fact the source does not state ---
    ("What is the exact bank account number I should use to receive PM-KISAN money?",
     "en", "trick_unstated_fact"),
]


def main() -> int:
    # Load every scheme document once, normalised for matching.
    documents: dict[str, str] = {}
    for path in sorted(SCHEMES_DIR.glob("*.md")):
        documents[path.stem] = normalize(io.open(path, encoding="utf-8").read())

    if not documents:
        print(f"ERROR: no scheme files in {SCHEMES_DIR}")
        return 1

    rows: list[dict] = []
    problems: list[str] = []

    # ---- answerable ----
    for index, (question, language, scheme, facts, category) in enumerate(ANSWERABLE, start=1):
        if scheme not in documents:
            problems.append(f"a{index:02d}: unknown scheme '{scheme}'")
            continue

        # The core check: every expected fact must really be in the data.
        for fact in facts:
            if not fact_present(fact, documents[scheme]):
                problems.append(
                    f"a{index:02d}: key_fact {fact!r} is NOT in {scheme}.md "
                    f"(question: {question[:50]}...)"
                )

        rows.append({
            "id": f"a{index:02d}",
            "question": question,
            "language": language,
            "answerable": True,
            "expected_scheme": scheme,
            "key_facts": facts,
            "category": category,
        })

    # ---- unanswerable ----
    for index, (question, language, category) in enumerate(UNANSWERABLE, start=1):
        rows.append({
            "id": f"u{index:02d}",
            "question": question,
            "language": language,
            "answerable": False,
            "expected_scheme": None,
            "key_facts": [],
            "category": category,
        })

    if problems:
        print(f"FAILED: {len(problems)} problem(s) — the golden set was NOT written.\n")
        for problem in problems:
            print(f"  ! {problem}")
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_PATH, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    answerable = sum(1 for row in rows if row["answerable"])
    print(f"Wrote {OUT_PATH} — {len(rows)} questions "
          f"({answerable} answerable, {len(rows) - answerable} unanswerable)")

    by_language: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for row in rows:
        by_language[row["language"]] = by_language.get(row["language"], 0) + 1
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1

    print(f"  by language: {by_language}")
    print(f"  schemes covered: "
          f"{len({row['expected_scheme'] for row in rows if row['expected_scheme']})}/15")
    print("  by category:")
    for category, count in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"    {count:3d}  {category}")
    print("\n  Every key_fact above was verified to appear in its scheme's .md file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
