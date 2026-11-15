"""Tests for eval/common.py: row selection, fact matching and small statistics."""

import math

from eval.common import (
    auroc,
    fact_present,
    fmt_pct,
    load_rows,
    median,
    meta_compatible,
    pct,
    percentile_nearest_rank,
    select_rows,
    wilson_interval,
)

UNANSWERABLE_CATEGORIES = {"off_topic", "fake_scheme", "not_indexed", "trick_unstated_fact"}


def test_stratified_covers_every_language_and_unanswerable_category():
    rows = load_rows()
    for n in (12, 20, 30):
        subset = select_rows(rows, stratified=n, seed=0)
        assert len(subset) == n
        assert {r["language"] for r in subset} >= {"en", "hi", "hinglish"}
        assert {r["category"] for r in subset if not r["answerable"]} == UNANSWERABLE_CATEGORIES
        assert any(r["answerable"] for r in subset)


def test_stratified_is_deterministic_and_keeps_file_order():
    rows = load_rows()
    first = select_rows(rows, stratified=20, seed=3)
    again = select_rows(rows, stratified=20, seed=3)
    assert [r["id"] for r in first] == [r["id"] for r in again]
    positions = [rows.index(r) for r in first]
    assert positions == sorted(positions)
    other = select_rows(rows, stratified=20, seed=4)
    assert [r["id"] for r in other] != [r["id"] for r in first]


def test_filters_and_oversized_stratified():
    rows = load_rows()
    assert {r["id"] for r in select_rows(rows, ids=["a01", "u05"])} == {"a01", "u05"}
    hi = select_rows(rows, languages=["hi"])
    assert hi and all(r["language"] == "hi" for r in hi)
    assert len(select_rows(rows, stratified=999)) == len(rows)
    fakes = select_rows(rows, categories=["fake_scheme"])
    assert all(r["category"] == "fake_scheme" for r in fakes)


def test_wilson_interval_known_values_and_edges():
    assert wilson_interval(0, 0) is None
    low, high = wilson_interval(5, 10)
    assert math.isclose(low, 0.2366, abs_tol=1e-3)
    assert math.isclose(high, 0.7634, abs_tol=1e-3)
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and 0.6 < low < 0.8
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0.2 < high < 0.4


def test_auroc():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([], [0.5]) is None
    assert auroc([0.5], []) is None
    # 2 positives x 2 negatives: wins 3 of 4 pairs
    assert auroc([0.9, 0.3], [0.5, 0.1]) == 0.75


def test_pct_and_fmt_pct_never_show_empty_as_zero():
    assert pct(1, 0) is None
    assert pct(1, 4) == 25.0
    assert fmt_pct(None, 0) == "n/a (n=0)"
    assert fmt_pct(12.5) == "12.5%"


def test_percentile_and_median():
    assert percentile_nearest_rank([], 95) is None
    assert percentile_nearest_rank(list(range(1, 101)), 95) == 95
    assert percentile_nearest_rank([5], 95) == 5
    assert median([1, 3, 2]) == 2
    assert median([1, 2, 3, 4]) == 2.5
    assert median([]) is None


def test_fact_present_boundaries_and_normalisation():
    assert fact_present("6000", "Rs. 6,000 per year")
    assert fact_present("6,000", "Rs 6000 per year")
    assert not fact_present("6000", "a loan of Rs 60000")
    assert not fact_present("2000", "Rs 20000")
    assert fact_present("2000", "teen kishton mein Rs 2000 milte hain")
    assert fact_present("Three Equal Installments", "paid in three  equal installments")
    assert fact_present("6000", "साल में ₹६००० मिलते हैं")  # Devanagari digits
    assert not fact_present("", "anything")


def test_meta_compatible():
    a = {"question_ids": ["a01"], "model": "m", "top_k": 5, "min_score": 0.45,
         "block_threshold": 0.5}
    assert meta_compatible(a, dict(a)) == []
    assert meta_compatible(a, {**a, "question_ids": ["a01", "a02"]}) == ["question_ids"]
    assert set(meta_compatible(a, {**a, "model": "x", "top_k": 3})) == {"model", "top_k"}
