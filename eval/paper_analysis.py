"""
Paper analysis: retrieval similarity vs answerability, on the translation-paired set.

Reads the saved retrieval results (eval/runs/paper_set/retrieval_raw.json) and the
question file (eval/paper_set.jsonl). Makes 0 LLM calls and 0 embedding calls, so
it can be re-run any time.

WHAT IT REPORTS (each section answers one claim the paper makes)
  1. Separability per failure mode. AUROC of the best retrieval score, answerable
     vs each kind of unanswerable question. The claim: a similarity threshold
     separates off-topic questions well and the others barely at all.
  2. Operating points. For a target over-block rate (<= 5%, <= 10%), the highest
     threshold that meets it, and what fraction of each failure mode it refuses.
     This is the fair way to compare: "how much can the threshold catch if we only
     let it wrongly refuse 1 in 20 answerable questions?"
  3. Near misses. For real-but-not-indexed schemes, which indexed scheme gets
     retrieved instead, and how confidently. This is the dangerous case: confident
     evidence about the wrong scheme.
  4. Language effect, paired. Every question exists in en/hi/hinglish, so the
     per-language comparison holds the content fixed. Recall@1 is compared with an
     exact McNemar test on the discordant pairs; best scores with the median paired
     difference. Also: does one fixed threshold treat languages differently?

Run:  python eval/paper_analysis.py
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.common import auroc, fmt_interval, median  # noqa: E402

DEFAULT_RAW = ROOT / "eval" / "runs" / "paper_set" / "retrieval_raw.json"
DEFAULT_QUESTIONS = ROOT / "eval" / "paper_set.jsonl"
DEFAULT_OUT = ROOT / "eval" / "runs" / "paper_set" / "paper_analysis.md"

FAILURE_MODES = ("off_topic", "fake_scheme", "not_indexed", "trick_unstated_fact")
LANGUAGES = ("en", "hi", "hinglish")
GRID = [round(0.20 + 0.01 * i, 2) for i in range(61)]  # 0.20 .. 0.80


# ==========================================================================
# Small, pure, unit-tested helpers
# ==========================================================================


def pair_id_of(question_id: str) -> str:
    """'pa01-hi' -> 'pa01'."""
    return question_id.rsplit("-", 1)[0]


def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value from the discordant counts.

    b = pairs where only the first condition succeeded, c = only the second.
    Under H0 each discordant pair is a fair coin flip, so this is a two-sided
    binomial test of min(b, c) against Binomial(b + c, 0.5). Concordant pairs carry
    no information about the difference and are ignored, which is the point.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def operating_point(answerable: list[float], max_over_block: float, grid=GRID) -> float | None:
    """
    The highest threshold whose over-block rate is <= max_over_block.

    A question is refused when its best score is below the threshold, so a higher
    threshold refuses more of everything; the highest one that stays within the
    over-block budget is the most protective one we can afford.
    """
    best = None
    for t in grid:
        refused = sum(score < t for score in answerable)
        if answerable and refused / len(answerable) <= max_over_block:
            best = t
    return best


def refusal(scores: list[float], threshold: float) -> tuple[int, int]:
    """(refused, n) at a threshold."""
    return sum(score < threshold for score in scores), len(scores)


# ==========================================================================
# Report
# ==========================================================================


def _ratio(pair: tuple[int, int]) -> str:
    hits, n = pair
    return "n/a" if not n else f"{hits}/{n} ({100 * hits / n:.0f}%)"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(records: list[dict], questions: dict[str, dict], configured: float) -> str:
    by_id = {r["id"]: r for r in records}
    answerable = [r for r in records if r["answerable"]]
    ans_scores = [r["best_score"] for r in answerable]
    mode_scores = {m: [r["best_score"] for r in records if r["category"] == m]
                   for m in FAILURE_MODES}
    all_unans = [s for scores in mode_scores.values() for s in scores]

    out: list[str] = []
    out.append("# Paper analysis: retrieval similarity is not answerability\n")
    out.append(f"Question file: `paper_set.jsonl`, n={len(records)} "
               f"({len(answerable)} answerable, {len(all_unans)} unanswerable), every question "
               "in English, Hindi and Hinglish. Retrieval variant: **raw** (the question as "
               "typed, 0 LLM calls). Production refuses on the LLM-normalised query, which has "
               "not been run on this set yet.\n")

    # ---- 1. separability -------------------------------------------------
    out.append("## 1. Separability per failure mode\n")
    out.append("AUROC of the best retrieval score: answerable (positive) vs each kind of "
               "unanswerable question. 1.0 = a perfect threshold exists; 0.5 = no signal.\n")
    out.append("| Unanswerable type | n | median best score | AUROC vs answerable |")
    out.append("|---|---|---|---|")
    out.append(f"| *(answerable)* | {len(ans_scores)} | {_fmt(median(ans_scores))} | — |")
    for mode in FAILURE_MODES:
        scores = mode_scores[mode]
        out.append(f"| `{mode}` | {len(scores)} | {_fmt(median(scores))} | "
                   f"{_fmt(auroc(ans_scores, scores))} |")
    out.append(f"| **all unanswerable** | {len(all_unans)} | {_fmt(median(all_unans))} | "
               f"**{_fmt(auroc(ans_scores, all_unans))}** |\n")

    # ---- 2. operating points --------------------------------------------
    out.append("## 2. What a threshold can catch at a fixed over-block budget\n")
    out.append("For each budget, the highest threshold that wrongly refuses at most that "
               "share of answerable questions, and the share of each failure mode it "
               f"refuses. The configured production threshold ({configured:.2f}) is shown "
               "for reference.\n")
    header = ["Threshold", "Over-block (answerable)"] + [f"`{m}` refused" for m in FAILURE_MODES]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    rows = []
    for budget in (0.05, 0.10):
        t = operating_point(ans_scores, budget)
        if t is not None:
            rows.append((f"{t:.2f} (budget ≤{int(budget * 100)}%)", t))
    rows.append((f"{configured:.2f} (configured)", configured))
    for label, t in rows:
        cells = [label, _ratio(refusal(ans_scores, t))]
        cells += [_ratio(refusal(mode_scores[m], t)) for m in FAILURE_MODES]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # ---- 3. near misses ---------------------------------------------------
    out.append("## 3. Near misses: real schemes we do not index\n")
    out.append("Which indexed scheme the retriever returns first for a real scheme that is "
               "not in the corpus. A score above the threshold means retrieval alone would "
               "pass this question on to the generator with the wrong scheme's evidence.\n")
    out.append("| Real scheme asked about | top-1 indexed scheme (en / hi / hinglish) | "
               "best score (en / hi / hinglish) |")
    out.append("|---|---|---|")
    passed = 0
    pairs = sorted({pair_id_of(r["id"]) for r in records if r["category"] == "not_indexed"})
    for pair in pairs:
        recs = [by_id.get(f"{pair}-{lang}") for lang in LANGUAGES]
        real = questions.get(f"{pair}-en", {}).get("real_scheme", pair)
        tops = [(r["ranked"][0]["scheme_id"] if r and r["ranked"] else "—") for r in recs]
        scores = [(r["best_score"] if r else None) for r in recs]
        passed += sum(s is not None and s >= configured for s in scores)
        out.append(f"| {real} | {' / '.join(f'`{t}`' for t in tops)} | "
                   f"{' / '.join(_fmt(s, 2) for s in scores)} |")
    n_ni = len(mode_scores["not_indexed"])
    out.append(f"\n**{passed}/{n_ni}** not-indexed questions score at or above the configured "
               f"threshold ({configured:.2f}), i.e. retrieval alone does not stop them.\n")

    fake_tops = Counter(r["ranked"][0]["scheme_id"] for r in records
                        if r["category"] == "fake_scheme" and r["ranked"])
    out.append("Fake-scheme questions, most frequent top-1 scheme: " + ", ".join(
        f"`{s}` ×{n}" for s, n in fake_tops.most_common(5)) + ".\n")

    # ---- 4. language effect, paired ---------------------------------------
    out.append("## 4. Language effect (paired by translation)\n")
    out.append("Recall@1 = the expected scheme owns the top chunk. Each answerable concept "
               "is compared with itself across languages; McNemar uses only the discordant "
               "pairs.\n")
    concept_ids = sorted({pair_id_of(r["id"]) for r in answerable})

    def r1(rec: dict | None) -> bool | None:
        if not rec or not rec["ranked"]:
            return None
        return rec["ranked"][0]["scheme_id"] == rec["expected_scheme"]

    out.append("| Comparison | both hit | only en | only other | neither | "
               "exact McNemar p | median best-score drop (en − other) |")
    out.append("|---|---|---|---|---|---|---|")
    for other in ("hi", "hinglish"):
        both = only_en = only_other = neither = 0
        drops = []
        for concept in concept_ids:
            en_rec, ot_rec = by_id.get(f"{concept}-en"), by_id.get(f"{concept}-{other}")
            a, b = r1(en_rec), r1(ot_rec)
            if a is None or b is None:
                continue
            both += a and b
            only_en += a and not b
            only_other += b and not a
            neither += (not a) and (not b)
            drops.append(en_rec["best_score"] - ot_rec["best_score"])
        out.append(f"| en vs {other} | {both} | {only_en} | {only_other} | {neither} | "
                   f"{mcnemar_exact(only_en, only_other):.3f} | {_fmt(median(drops))} |")
    out.append("")

    out.append(f"**One threshold, three languages.** Refusals at the configured threshold "
               f"({configured:.2f}), by language:\n")
    out.append("| Language | Over-block (answerable) | Unanswerable refused |")
    out.append("|---|---|---|")
    for lang in LANGUAGES:
        a = [r["best_score"] for r in answerable if r["language"] == lang]
        u = [r["best_score"] for r in records if not r["answerable"] and r["language"] == lang]
        refused_a = refusal(a, configured)
        out.append(f"| {lang} | {refused_a[0]}/{refused_a[1]}, {fmt_interval(*refused_a)} | "
                   f"{_ratio(refusal(u, configured))} |")
    out.append("")

    out.append("## Caveats\n")
    out.append("- Raw queries only. The LLM normaliser translates Hindi/Hinglish to English "
               "before retrieval in production; its effect on this set is not yet measured.\n"
               "- Author-written questions and translations; not yet reviewed by a second "
               "native speaker.\n"
               "- Relevance is scheme-level (no gold passage ids).\n"
               "- This analysis is about the retrieval gate only. Whether claim-level "
               "verification catches what the threshold misses is measured end-to-end by "
               "`eval/run_eval.py`, which needs the LLM.\n")
    out.append("Regenerate: `python eval/paper_analysis.py`.\n")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper analysis over saved retrieval results.")
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--variant", default="raw")
    args = parser.parse_args()

    with io.open(args.raw, encoding="utf-8") as fh:
        data = json.load(fh)
    if args.variant not in data["variants"]:
        print(f"Variant {args.variant!r} not in {args.raw}; have {list(data['variants'])}")
        return 1
    records = data["variants"][args.variant]["records"]
    with io.open(args.questions, encoding="utf-8") as fh:
        questions = {row["id"]: row for row in map(json.loads, filter(str.strip, fh))}
    configured = data["meta"]["configured_min_score"]

    report = build_report(records, questions, configured)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
