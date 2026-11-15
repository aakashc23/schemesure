"""Tests for eval/retrieval_metrics.py using hand-made retrieval records (no model)."""

import math

from eval.retrieval_metrics import (
    recall_at_k,
    reciprocal_rank,
    scheme_reciprocal_rank,
    score_distributions,
    summarize_retrieval,
    sweep_threshold,
    threshold_aurocs,
)


def rec(id_, schemes, expected, answerable=True, language="en", category="x",
        best=None, detected=None):
    ranked = [
        {"chunk_id": f"{s}#{i}", "scheme_id": s, "section": "S", "score": 0.9 - 0.05 * i}
        for i, s in enumerate(schemes)
    ]
    if best is None:
        best = ranked[0]["score"] if ranked else 0.0
    return {
        "id": id_, "language": language, "answerable": answerable, "category": category,
        "expected_scheme": expected, "ranked": ranked, "best_score": best,
        "detected_language": detected or language,
    }


def test_recall_at_k():
    r = rec("a", ["kcc", "kcc", "pm_kisan", "x"], "pm_kisan")
    assert recall_at_k(r, 1) is False
    assert recall_at_k(r, 2) is False
    assert recall_at_k(r, 3) is True
    assert recall_at_k(rec("u", ["a"], None, answerable=False), 3) is None


def test_reciprocal_rank_chunk_vs_distinct_scheme():
    r = rec("a", ["kcc", "kcc", "kcc", "pm_kisan"], "pm_kisan")
    assert reciprocal_rank(r) == 0.25            # 4th chunk
    assert scheme_reciprocal_rank(r) == 0.5      # 2nd distinct scheme
    assert reciprocal_rank(rec("a", ["pm_kisan"], "pm_kisan")) == 1.0
    assert reciprocal_rank(rec("a", ["kcc"], "pm_kisan")) == 0.0
    # beyond the cut-off counts as a miss
    deep = rec("a", ["k"] * 10 + ["pm_kisan"], "pm_kisan")
    assert reciprocal_rank(deep, k=10) == 0.0


def test_summarize_retrieval_per_language_and_language_id():
    records = [
        rec("a1", ["p", "q"], "p", language="en"),
        rec("a2", ["q", "p"], "p", language="en"),
        rec("a3", ["q", "q"], "p", language="hi", detected="en"),
        rec("u1", ["q"], None, answerable=False, language="en", category="off_topic"),
    ]
    summary = summarize_retrieval(records)
    assert summary["all"]["n"] == 3
    assert summary["all"]["recall"][1] == (1, 3)
    assert summary["all"]["recall"][3] == (2, 3)
    assert math.isclose(summary["all"]["mrr"], (1 + 0.5 + 0) / 3, abs_tol=1e-4)
    assert summary["en"]["n"] == 2 and summary["hi"]["n"] == 1
    assert summary["all"]["lang_id"] == (3, 4)   # a3 was mis-detected
    assert summary["hi"]["lang_id"] == (0, 1)


def test_summarize_retrieval_empty_group_has_no_rates():
    summary = summarize_retrieval([rec("u1", ["q"], None, answerable=False, category="off_topic")])
    assert summary["all"]["n"] == 0
    assert summary["all"]["mrr"] is None


def _score_records():
    return [
        rec("a1", ["p"], "p", best=0.80),
        rec("a2", ["p"], "p", best=0.50, language="hi"),
        rec("a3", ["p"], "p", best=0.40, language="hinglish"),
        rec("u1", ["q"], None, answerable=False, category="off_topic", best=0.20),
        rec("u2", ["q"], None, answerable=False, category="off_topic", best=0.30),
        rec("u3", ["q"], None, answerable=False, category="fake_scheme", best=0.60),
    ]


def test_sweep_threshold_counts():
    swept = sweep_threshold(_score_records(), [0.35, 0.45, 0.55])
    rows = {row["threshold"]: row for row in swept}
    assert rows[0.35]["over_block"] == (0, 3)
    assert rows[0.35]["abstain_all"] == (2, 3)
    assert rows[0.35]["abstain_by_category"]["off_topic"] == (2, 2)
    assert rows[0.35]["abstain_by_category"]["fake_scheme"] == (0, 1)
    assert rows[0.45]["over_block"] == (1, 3)     # 0.40 < 0.45
    assert rows[0.55]["over_block"] == (2, 3)
    assert rows[0.55]["abstain_by_category"]["fake_scheme"] == (0, 1)  # 0.60 is not < 0.55


def test_sweep_is_monotone_and_default_grid_matches_plan():
    rows = sweep_threshold(_score_records())
    blocked = [row["over_block"][0] for row in rows]
    abstained = [row["abstain_all"][0] for row in rows]
    assert blocked == sorted(blocked) and abstained == sorted(abstained)
    assert rows[0]["threshold"] == 0.30 and rows[-1]["threshold"] == 0.70 and len(rows) == 41


def test_threshold_aurocs_and_distributions():
    records = _score_records()
    aurocs = threshold_aurocs(records)
    assert aurocs["answerable_vs_off_topic"] == 1.0
    # answerable {0.8, 0.5, 0.4} vs unanswerable {0.2, 0.3, 0.6}:
    # 0.8 beats 3, 0.5 beats 2, 0.4 beats 2 => 7 of 9 pairs
    assert math.isclose(aurocs["answerable_vs_all_unanswerable"], 7 / 9, abs_tol=1e-3)
    dist = score_distributions(records)
    assert dist["answerable_all"]["min"] == 0.4 and dist["answerable_all"]["max"] == 0.8
    assert dist["unanswerable_by_category"]["off_topic"]["median"] == 0.25
    assert dist["answerable_by_language"]["hi"]["n"] == 1
