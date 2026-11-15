"""
Shared helpers for the evaluation scripts: loading and selecting golden-set rows,
fact matching, and the small statistics (Wilson interval, AUROC) the reports use.

WHY ONE SHARED FILE
run_eval.py (end-to-end) and run_retrieval.py (retrieval only) must select
questions, match facts and compute intervals in exactly the same way, or their
numbers could not be compared. Everything here is plain Python with no LLM and no
network, so it is all unit-tested (tests/test_eval_common.py).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "eval" / "golden_set.jsonl"


# ==========================================================================
# Golden set: loading and selection
# ==========================================================================


def load_rows(path: Path | None = None) -> list[dict]:
    """Read the golden set (one JSON object per line)."""
    path = path or GOLDEN_PATH
    with io.open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh.read().splitlines() if line.strip()]


def golden_sha1(rows: list[dict]) -> str:
    """Fingerprint of the question set, stored in run metadata."""
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def select_rows(
    rows: list[dict],
    ids: list[str] | None = None,
    languages: list[str] | None = None,
    categories: list[str] | None = None,
    stratified: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """
    Pick a subset of the golden set. Order of the original file is preserved.

    ids / languages / categories are plain filters. `stratified=N` then keeps N
    questions chosen for COVERAGE, not proportion: the subset represents every
    language and every unanswerable category, but it is NOT proportional to the
    full set (small unanswerable and non-English groups are over-represented, so
    a subset has a higher share of unanswerable questions than the whole set).
    Pooled figures from a subset (latency, calls per question, claim rates) are
    therefore preliminary and must not be read as the full benchmark:

      1. COVERAGE FIRST: every (answerable, language) pair and every unanswerable
         category present in the data gets at least one question (as long as N is
         large enough). This is what stops a "quick 20" from being all-English
         or from missing the hardest unanswerable type.
      2. THEN ROUND-ROBIN over (answerable, language, category) groups, drawing
         from each group in turn, until N questions are chosen.

    Randomness is a seeded `random.Random`, so the same seed always gives the
    same subset (a requirement for resuming and for comparing runs).
    """
    selected = list(rows)
    if ids:
        wanted = set(ids)
        selected = [r for r in selected if r["id"] in wanted]
    if languages:
        selected = [r for r in selected if r["language"] in set(languages)]
    if categories:
        selected = [r for r in selected if r["category"] in set(categories)]

    if not stratified or stratified >= len(selected):
        return selected

    rng = random.Random(seed)
    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    def take(row: dict) -> None:
        chosen.append(row)
        chosen_ids.add(row["id"])

    # ---- 1. coverage constraints ----------------------------------------
    # Each constraint is a predicate; one row can satisfy several at once.
    constraints: list = []
    pairs = sorted({(bool(r["answerable"]), r["language"]) for r in selected})
    for answerable, language in pairs:
        constraints.append(
            lambda r, a=answerable, lang=language: (
                bool(r["answerable"]) == a and r["language"] == lang
            )
        )
    for category in sorted({r["category"] for r in selected if not r["answerable"]}):
        constraints.append(
            lambda r, c=category: (not r["answerable"]) and r["category"] == c
        )

    for matches in constraints:
        if len(chosen) >= stratified:
            break
        if any(matches(r) for r in chosen):
            continue  # already covered by an earlier pick
        candidates = [r for r in selected if r["id"] not in chosen_ids and matches(r)]
        if candidates:
            take(rng.choice(candidates))

    # ---- 2. round-robin fill --------------------------------------------
    groups: dict[tuple, list[dict]] = {}
    for row in selected:
        if row["id"] in chosen_ids:
            continue
        key = (bool(row["answerable"]), row["language"], row["category"])
        groups.setdefault(key, []).append(row)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    for key in group_keys:
        rng.shuffle(groups[key])

    while len(chosen) < stratified and any(groups.values()):
        for key in group_keys:
            if len(chosen) >= stratified:
                break
            if groups[key]:
                take(groups[key].pop())

    order = {row["id"]: index for index, row in enumerate(rows)}
    return sorted(chosen, key=lambda r: order[r["id"]])


def add_selection_args(parser) -> None:
    """The row-selection flags shared by run_eval.py and run_retrieval.py."""
    parser.add_argument("--ids", default="", help="comma-separated question ids, e.g. a01,u05")
    parser.add_argument("--languages", default="", help="comma-separated: en,hi,hinglish")
    parser.add_argument("--categories", default="", help="comma-separated categories")
    parser.add_argument("--stratified", type=int, default=0,
                        help="keep N questions, spread across language/category")
    parser.add_argument("--seed", type=int, default=0, help="seed for --stratified")


def rows_from_args(args, rows: list[dict]) -> list[dict]:
    """Apply the shared selection flags to `rows`."""

    def split(text: str) -> list[str]:
        return [part.strip() for part in text.split(",") if part.strip()]

    return select_rows(
        rows,
        ids=split(args.ids),
        languages=split(args.languages),
        categories=split(args.categories),
        stratified=args.stratified or None,
        seed=args.seed,
    )


def selection_is_subset(selected: list[dict], all_rows: list[dict]) -> bool:
    """True when a run covers fewer questions than the full golden set."""
    return {r["id"] for r in selected} != {r["id"] for r in all_rows}


def subset_dir_name(selected: list[dict]) -> str:
    """Stable folder name for a subset run, derived from its question ids."""
    digest = hashlib.sha1(
        ",".join(sorted(r["id"] for r in selected)).encode("utf-8")
    ).hexdigest()[:8]
    return f"subset-{len(selected)}q-{digest}"


# ==========================================================================
# Fact matching
# ==========================================================================

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize(text: str) -> str:
    """Lowercase, NFKC, Devanagari digits -> ASCII, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", str(text)).lower().translate(_DEVANAGARI_DIGITS)
    return re.sub(r"\s+", " ", text)


def fact_present(fact: str, text: str) -> bool:
    """
    Tolerant substring match: ignores case, spacing and Indian digit grouping.

    Digit boundaries are enforced: the fact "6000" must not be found inside
    "60000" (a false positive the old plain-substring test allowed).
    """
    normalized_text = normalize(text).replace(",", "")
    normalized_fact = normalize(fact).replace(",", "")
    if not normalized_fact:
        return False
    pattern = re.escape(normalized_fact)
    if normalized_fact[0].isdigit():
        pattern = r"(?<![\d.])" + pattern
    if normalized_fact[-1].isdigit():
        pattern = pattern + r"(?!\d)"
    return re.search(pattern, normalized_text) is not None


# ==========================================================================
# Statistics
# ==========================================================================


def pct(numerator: int, denominator: int) -> float | None:
    """Percentage rounded to 1 dp, or None when there is nothing to divide by."""
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 1)


def fmt_pct(value: float | None, denominator: int | None = None) -> str:
    """Render a rate; an empty denominator is 'n/a (n=0)', never 0%."""
    if value is None:
        return f"n/a (n={denominator or 0})"
    return f"{value}%"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """
    95% Wilson score interval for a proportion, as fractions in [0, 1].

    Preferred over the textbook (normal) interval because it behaves at small n
    and near 0 or 1 -- exactly where a 35-question benchmark lives.
    """
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_interval(successes: int, n: int) -> str:
    """Percent with its Wilson interval, e.g. '62.9% [45.5-77.3]', or 'n/a (n=0)'."""
    if n <= 0:
        return "n/a (n=0)"
    low, high = wilson_interval(successes, n)
    return f"{100 * successes / n:.1f}% [{100 * low:.1f}-{100 * high:.1f}]"


def auroc(positive: list[float], negative: list[float]) -> float | None:
    """
    Probability that a random positive scores higher than a random negative
    (ties count half). 0.5 = no signal, 1.0 = perfectly separable.

    Computed directly from pairs; the sets here have tens of items, so the
    O(n*m) loop is simpler and clearer than a rank-based formula.
    """
    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for q in negative:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return round(wins / (len(positive) * len(negative)), 4)


def percentile_nearest_rank(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[rank - 1]


def median(values: list[float]) -> float | None:
    """Median; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


# ==========================================================================
# Run-metadata compatibility
# ==========================================================================

# Two condition runs may be put in one comparison table only if these agree.
# embedding_model and golden_sha1 are included so records made with another
# embedding model, or against an edited golden set that kept the same ids, are
# never merged or resumed into one table.
META_COMPARE_KEYS = (
    "question_ids", "model", "top_k", "min_score", "block_threshold",
    "embedding_model", "golden_sha1",
)


def meta_compatible(a: dict, b: dict) -> list[str]:
    """Return the list of metadata keys that differ ([] means compatible)."""
    return [key for key in META_COMPARE_KEYS if a.get(key) != b.get(key)]
