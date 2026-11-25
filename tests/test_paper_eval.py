"""
Tests for the paper experiments: the translation-paired question set, the paired
language statistics, and the number-perturbation helpers. No model, no LLM.
"""

import random
from decimal import Decimal

from eval.paper_analysis import (
    build_report,
    mcnemar_exact,
    operating_point,
    pair_id_of,
    refusal,
)
from eval.run_numeric_perturbation import (
    format_variants,
    indian_grouping,
    render,
    wrong_value,
)
from scripts.build_paper_set import LANGUAGES, build, scheme_body


# ---- the question set ------------------------------------------------------


def test_paper_set_builds_without_problems():
    rows, problems = build()
    assert problems == []
    assert len(rows) % 3 == 0


def test_every_pair_has_exactly_one_question_per_language():
    rows, _ = build()
    by_pair: dict[str, list[str]] = {}
    for row in rows:
        by_pair.setdefault(row["pair_id"], []).append(row["language"])
    assert all(sorted(langs) == sorted(LANGUAGES) for langs in by_pair.values())


def test_pairs_share_labels_across_languages():
    """A translation must not change the answer key."""
    rows, _ = build()
    by_pair: dict[str, set] = {}
    for row in rows:
        key = (row["answerable"], row["category"], row["expected_scheme"],
               tuple(row["key_facts"]))
        by_pair.setdefault(row["pair_id"], set()).add(key)
    assert all(len(keys) == 1 for keys in by_pair.values())


def test_question_ids_are_unique():
    rows, _ = build()
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_scheme_body_drops_front_matter(tmp_path):
    path = tmp_path / "x.md"
    path.write_text('---\nname: "Claim Form"\n---\n# Body\ntext\n', encoding="utf-8")
    body = scheme_body(path)
    assert "Claim Form" not in body
    assert "# Body" in body


# ---- paired statistics -------------------------------------------------------


def test_pair_id_of():
    assert pair_id_of("pa01-hi") == "pa01"
    assert pair_id_of("pt12-hinglish") == "pt12"


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    # 8 discordant pairs all one way: 2 * 0.5**8
    assert abs(mcnemar_exact(8, 0) - 2 * 0.5 ** 8) < 1e-12
    assert mcnemar_exact(3, 3) == 1.0  # capped at 1
    assert mcnemar_exact(5, 2) == mcnemar_exact(2, 5)  # symmetric


def test_operating_point_respects_budget():
    answerable = [0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.86, 0.87, 0.88, 0.89]
    t = operating_point(answerable, 0.10, grid=[0.2, 0.4, 0.55, 0.65])
    # 0.55 refuses 0.30 and 0.50 (20%) -> over budget; 0.4 refuses one (10%).
    assert t == 0.4
    assert refusal(answerable, t) == (1, 10)


def test_operating_point_none_when_nothing_fits():
    assert operating_point([0.1, 0.1], 0.0, grid=[0.5]) is None


def test_build_report_runs_on_tiny_records():
    def rec(id_, answerable, category, lang, scheme, best, expected=None):
        return {"id": id_, "answerable": answerable, "category": category,
                "language": lang, "expected_scheme": expected, "best_score": best,
                "ranked": [{"chunk_id": f"{scheme}#a#0", "scheme_id": scheme,
                            "section": "a", "score": best}]}
    records = []
    for lang, hit in (("en", True), ("hi", False), ("hinglish", True)):
        records.append(rec(f"pa01-{lang}", True, "benefit", lang,
                           "pm_kisan" if hit else "pmsby", 0.7, "pm_kisan"))
        records.append(rec(f"pn01-{lang}", False, "not_indexed", lang, "pm_mudra", 0.6))
    questions = {"pn01-en": {"real_scheme": "Sukanya Samriddhi Yojana"}}
    report = build_report(records, questions, 0.45)
    assert "Sukanya Samriddhi Yojana" in report
    assert "en vs hi | 0 | 1 | 0 | 0" in report


# ---- number perturbation helpers ---------------------------------------------


def test_indian_grouping():
    assert indian_grouping(999) == "999"
    assert indian_grouping(6000) == "6,000"
    assert indian_grouping(200000) == "2,00,000"
    assert indian_grouping(12345678) == "1,23,45,678"


def test_format_variants_keep_the_value():
    variants = format_variants(Decimal(200000), is_percent=False)
    assert "2,00,000" in variants
    assert "2 lakh" in variants
    assert any(ch in "०१२३४५६७८९" for ch in "".join(variants))


def test_format_variants_no_lakh_for_percent():
    assert all("lakh" not in v for v in format_variants(Decimal(200000), is_percent=True))


def test_wrong_value_avoids_evidence_numbers():
    rng = random.Random(0)
    known = {Decimal(12000), Decimal(6000), Decimal(9000), Decimal(60000)}
    for _ in range(20):
        wrong = wrong_value(Decimal(6000), known, rng)
        assert wrong is not None
        assert wrong != Decimal(6000)
        assert wrong not in known


def test_render_has_no_exponent():
    assert render(Decimal("2E+5")) == "200000"
    assert render(Decimal("1.50")) == "1.5"
