"""
Pure functions that score retrieval and analyse the refusal threshold.

Everything here works on plain dicts ("retrieval records", one per question) so
it can be unit-tested with hand-made data and re-run offline from
eval/retrieval_raw.json -- zero LLM calls, zero embedding calls.

A retrieval record looks like:
    {"id": "a01", "language": "en", "answerable": True, "category": "benefit_amount",
     "expected_scheme": "pm_kisan", "detected_language": "en", "best_score": 0.61,
     "ranked": [{"chunk_id": "...", "scheme_id": "pm_kisan", "section": "...",
                 "score": 0.61}, ...]}          # ranked best-first, top 10

WHAT THE METRICS MEAN
  Recall@k : was the expected scheme among the schemes of the top-k chunks?
             (scheme-level relevance only -- the golden set has no gold chunks.)
  MRR@10   : 1 / rank of the first chunk from the expected scheme (0 if absent
             from the top 10), averaged. Rewards putting the right scheme first.
  MRR (distinct schemes): the same, but rank counts distinct schemes, so a scheme
             that fills the top three slots is not penalised for crowding.
"""

from __future__ import annotations

from eval.common import auroc, median

DEFAULT_KS = (1, 3, 5)
# Threshold sweep range requested by the plan: 0.30 .. 0.70 in steps of 0.01.
SWEEP_GRID = [round(0.30 + 0.01 * i, 2) for i in range(41)]


# ==========================================================================
# Retrieval scoring
# ==========================================================================


def recall_at_k(record: dict, k: int) -> bool | None:
    """True/False for answerable rows with an expected scheme, else None."""
    expected = record.get("expected_scheme")
    if not expected:
        return None
    return any(item["scheme_id"] == expected for item in record.get("ranked", [])[:k])


def reciprocal_rank(record: dict, k: int = 10) -> float | None:
    """1/rank of the first chunk of the expected scheme within top-k (0 if none)."""
    expected = record.get("expected_scheme")
    if not expected:
        return None
    for rank, item in enumerate(record.get("ranked", [])[:k], start=1):
        if item["scheme_id"] == expected:
            return 1.0 / rank
    return 0.0


def scheme_reciprocal_rank(record: dict, k: int = 10) -> float | None:
    """Like reciprocal_rank but ranks DISTINCT schemes (first appearance order)."""
    expected = record.get("expected_scheme")
    if not expected:
        return None
    seen: list[str] = []
    for item in record.get("ranked", [])[:k]:
        if item["scheme_id"] not in seen:
            seen.append(item["scheme_id"])
    if expected in seen:
        return 1.0 / (seen.index(expected) + 1)
    return 0.0


def _score_group(records: list[dict], ks: tuple[int, ...]) -> dict:
    """Metrics for one group of retrieval records (answerable ones are scored)."""
    scored = [r for r in records if r.get("answerable") and r.get("expected_scheme")]
    result: dict = {
        "n": len(scored),
        "recall": {k: (sum(1 for r in scored if recall_at_k(r, k)), len(scored)) for k in ks},
        "mrr": None,
        "mrr_scheme": None,
    }
    if scored:
        result["mrr"] = round(sum(reciprocal_rank(r) for r in scored) / len(scored), 4)
        result["mrr_scheme"] = round(
            sum(scheme_reciprocal_rank(r) for r in scored) / len(scored), 4
        )
    # Language identification is scored on ALL questions (answerable or not).
    labelled = [r for r in records if r.get("language")]
    correct = sum(1 for r in labelled if r.get("detected_language") == r["language"])
    result["lang_id"] = (correct, len(labelled))
    return result


def summarize_retrieval(records: list[dict], ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    """Metrics per language plus an 'all' row. Keys: 'all', 'en', 'hi', ..."""
    summary = {"all": _score_group(records, ks)}
    for language in sorted({r["language"] for r in records}):
        summary[language] = _score_group([r for r in records if r["language"] == language], ks)
    return summary


# ==========================================================================
# Threshold analysis
# ==========================================================================


def distribution(values: list[float]) -> dict:
    """min / median / max of a list of scores (None entries when empty)."""
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "median": round(median(values), 4),
        "max": round(max(values), 4),
    }


def score_distributions(records: list[dict]) -> dict:
    """best_score distributions: answerable by language, unanswerable by category."""
    answerable: dict[str, list[float]] = {}
    unanswerable: dict[str, list[float]] = {}
    for r in records:
        if r["answerable"]:
            answerable.setdefault(r["language"], []).append(r["best_score"])
        else:
            unanswerable.setdefault(r["category"], []).append(r["best_score"])
    return {
        "answerable_by_language": {k: distribution(v) for k, v in sorted(answerable.items())},
        "answerable_all": distribution([s for v in answerable.values() for s in v]),
        "unanswerable_by_category": {k: distribution(v) for k, v in sorted(unanswerable.items())},
    }


def sweep_threshold(records: list[dict], grid: list[float] | None = None) -> list[dict]:
    """
    For each threshold t, "abstain" means best_score < t.

      over_block      : answerable questions that would be refused  (the cost)
      abstain recall  : unanswerable questions that would be refused (the benefit),
                        overall and per category.
    """
    grid = grid or SWEEP_GRID
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    categories = sorted({r["category"] for r in unanswerable})

    rows = []
    for t in grid:
        blocked = sum(1 for r in answerable if r["best_score"] < t)
        abstained = sum(1 for r in unanswerable if r["best_score"] < t)
        by_category = {}
        for category in categories:
            members = [r for r in unanswerable if r["category"] == category]
            by_category[category] = (
                sum(1 for r in members if r["best_score"] < t),
                len(members),
            )
        rows.append({
            "threshold": t,
            "over_block": (blocked, len(answerable)),
            "abstain_all": (abstained, len(unanswerable)),
            "abstain_by_category": by_category,
        })
    return rows


def threshold_aurocs(records: list[dict]) -> dict:
    """
    AUROC of best_score as a detector of "this question is answerable".

    Two views: answerable vs ALL unanswerable, and answerable vs off_topic only
    (the threshold is only ever expected to catch off_topic; fake and
    not-indexed questions look like real scheme questions and score high).
    """
    pos = [r["best_score"] for r in records if r["answerable"]]
    neg_all = [r["best_score"] for r in records if not r["answerable"]]
    neg_off = [r["best_score"] for r in records
               if not r["answerable"] and r["category"] == "off_topic"]
    return {
        "answerable_vs_all_unanswerable": auroc(pos, neg_all),
        "answerable_vs_off_topic": auroc(pos, neg_off),
    }
