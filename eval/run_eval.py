"""
Run the golden set with the guardrail OFF and ON, and compare.

THE QUESTION THIS ANSWERS
"Does the verification layer actually buy anything, and what does it cost?"
Without a measurement, a hallucination guardrail is just a claim. So every
question is run twice — once with verification disabled (plain RAG, the
baseline) and once with it enabled — and the two runs are compared on the same
inputs.

THE HEADLINE METRIC: UNSUPPORTED CLAIM RATE IN DELIVERED ANSWERS
Measuring the *draft* would flatter the guardrail. What matters is what reaches
the user. So after each run, the answer the user would actually see is audited
claim-by-claim against the same retrieved evidence. For the OFF run that audit
is pure measurement (nothing is blocked). For the ON run it also catches a real
risk: the repair step writing a new unsupported claim while fixing an old one.

Audit calls are counted separately from production calls — `llm_calls_per_question`
reports what serving one real user costs, not what measuring it costs.

WHAT "CORRECT REFUSAL" AND "OVER-BLOCKING" MEAN
  - correct refusal : an unanswerable question was refused. Higher is better.
  - over-blocking   : an answerable question was refused. Lower is better.
These trade against each other, which is the whole point of reporting both.

Run:
    python eval/run_eval.py                  # full run, both conditions
    python eval/run_eval.py --limit 6        # quick smoke test
    python eval/run_eval.py --condition on   # one condition only
    python eval/run_eval.py --no-audit       # skip the claim-level audit
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.llm_client import CallCounter, LLMClient, LLMError  # noqa: E402
from app.schemas import AnswerStatus, Verdict  # noqa: E402
from core.generator import AnswerPipeline  # noqa: E402
from core.guardrail import Guardrail  # noqa: E402
from core.retriever import Retriever  # noqa: E402

GOLDEN_PATH = ROOT / "eval" / "golden_set.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.md"
RAW_PATH = ROOT / "eval" / "raw_results.json"


# ==========================================================================
# Fact matching
# ==========================================================================


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text)


def fact_present(fact: str, text: str) -> bool:
    """Tolerant match: ignores case, spacing and Indian digit grouping."""
    normalized_text = normalize(text)
    normalized_fact = normalize(fact)
    if normalized_fact in normalized_text:
        return True
    return normalized_fact.replace(",", "") in normalized_text.replace(",", "")


# ==========================================================================
# One question, one condition
# ==========================================================================


def run_one(
    row: dict,
    pipeline: AnswerPipeline,
    retriever: Retriever,
    auditor: Guardrail | None,
) -> dict:
    """Run a single question and collect everything measurable about it."""
    question = row["question"]
    started = time.perf_counter()

    try:
        response = pipeline.ask(question)
        error = ""
    except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
        return {
            "id": row["id"],
            "error": f"{type(exc).__name__}: {exc}",
            "answerable": row["answerable"],
            "category": row["category"],
            "refused": None,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "llm_calls": 0,
            "audit_calls": 0,
        }

    refused = response.status == AnswerStatus.REFUSED

    # ---- retrieval hit rate, measured independently of the refusal -------
    # Re-running the search with the same normalized query is deterministic and
    # costs no LLM call, so retrieval quality can be judged even when the
    # pipeline refused before generating.
    retrieval = retriever.search(response.normalized_query, min_score=0.0)
    retrieved_schemes = [chunk.metadata.scheme_id for chunk in retrieval.chunks]
    expected = row.get("expected_scheme")

    record: dict = {
        "id": row["id"],
        "question": question,
        "language": row["language"],
        "answerable": row["answerable"],
        "category": row["category"],
        "expected_scheme": expected,
        "status": response.status.value,
        "refused": refused,
        "guardrail_decision": response.guardrail.decision.value,
        "answer": response.answer,
        "normalized_query": response.normalized_query,
        "detected_language": response.language.value,
        "best_retrieval_score": retrieval.best_score,
        "retrieved_schemes": retrieved_schemes[:3],
        "retrieval_hit": (expected in retrieved_schemes) if expected else None,
        "retrieval_hit_top1": (
            (retrieved_schemes[0] == expected) if (expected and retrieved_schemes) else None
        ),
        "citations": len(response.citations),
        "latency_ms": response.latency_ms,
        "llm_calls": response.llm_calls,
        "audit_calls": 0,
        "error": "",
        # Draft-level verdicts, when the guardrail ran in production.
        "draft_claims": len(response.guardrail.claims),
        "draft_unsupported": response.guardrail.unsupported_count,
        "fabricated_citations": sum(
            1 for claim in response.guardrail.claims if not claim.citation_valid
        ),
    }

    # ---- key-fact coverage ----------------------------------------------
    key_facts = row.get("key_facts") or []
    if key_facts and not refused:
        found = [fact for fact in key_facts if fact_present(fact, response.answer)]
        record["key_facts_total"] = len(key_facts)
        record["key_facts_found"] = len(found)
        record["key_facts_missing"] = [f for f in key_facts if f not in found]
    else:
        record["key_facts_total"] = len(key_facts)
        record["key_facts_found"] = 0
        record["key_facts_missing"] = list(key_facts)

    # ---- audit the DELIVERED answer -------------------------------------
    # This is the number that matters: unsupported claims the user actually saw.
    record["delivered_claims"] = None
    record["delivered_unsupported"] = None
    if auditor is not None and not refused and retrieval.chunks:
        audit_counter = CallCounter()
        try:
            audit = auditor.verify(response.answer, retrieval.chunks, counter=audit_counter)
            record["delivered_claims"] = len(audit.claims)
            record["delivered_unsupported"] = sum(
                1 for claim in audit.claims if claim.verdict != Verdict.SUPPORTED
            )
            record["audit_calls"] = audit_counter.count
        except LLMError as exc:
            record["audit_error"] = str(exc)

    return record


# ==========================================================================
# Aggregation
# ==========================================================================


def summarise(records: list[dict]) -> dict:
    """Turn per-question records into the headline numbers."""
    ok = [r for r in records if not r.get("error")]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]

    def rate(numerator: int, denominator: int) -> float:
        return round(100.0 * numerator / denominator, 1) if denominator else 0.0

    # Retrieval
    with_expected = [r for r in answerable if r["retrieval_hit"] is not None]
    retrieval_hits = sum(1 for r in with_expected if r["retrieval_hit"])
    top1_hits = sum(1 for r in with_expected if r["retrieval_hit_top1"])

    # Refusals
    answerable_refused = sum(1 for r in answerable if r["refused"])
    unanswerable_refused = sum(1 for r in unanswerable if r["refused"])

    # Key facts (over answered questions only — a refusal has no facts to find)
    answered = [r for r in answerable if not r["refused"]]
    facts_total = sum(r["key_facts_total"] for r in answered)
    facts_found = sum(r["key_facts_found"] for r in answered)

    # Unsupported claims in DELIVERED answers
    audited = [r for r in ok if r.get("delivered_claims")]
    delivered_claims = sum(r["delivered_claims"] for r in audited)
    delivered_unsupported = sum(r["delivered_unsupported"] for r in audited)
    answers_with_unsupported = sum(1 for r in audited if r["delivered_unsupported"] > 0)

    latencies = [r["latency_ms"] for r in ok]
    production_calls = [r["llm_calls"] for r in ok]

    decisions: dict[str, int] = {}
    for record in ok:
        decision = record["guardrail_decision"]
        decisions[decision] = decisions.get(decision, 0) + 1

    # Refusal rate split by *why* the question is unanswerable — the three cases
    # need different defences, so a single average would hide the interesting part.
    by_category: dict[str, dict] = {}
    for record in unanswerable:
        bucket = by_category.setdefault(record["category"], {"n": 0, "refused": 0})
        bucket["n"] += 1
        bucket["refused"] += 1 if record["refused"] else 0
    for bucket in by_category.values():
        bucket["rate"] = rate(bucket["refused"], bucket["n"])

    return {
        "questions": len(records),
        "errors": len(records) - len(ok),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),

        "retrieval_hit_rate": rate(retrieval_hits, len(with_expected)),
        "retrieval_hit_rate_top1": rate(top1_hits, len(with_expected)),

        "correct_refusal_rate": rate(unanswerable_refused, len(unanswerable)),
        "over_blocking_rate": rate(answerable_refused, len(answerable)),
        "answered_rate": rate(len(answered), len(answerable)),

        "key_fact_coverage": rate(facts_found, facts_total),
        "key_facts_found": facts_found,
        "key_facts_total": facts_total,

        "unsupported_claim_rate": rate(delivered_unsupported, delivered_claims),
        "delivered_unsupported": delivered_unsupported,
        "delivered_claims": delivered_claims,
        "answers_with_unsupported_claim_rate": rate(answers_with_unsupported, len(audited)),
        "answers_with_unsupported_claim": answers_with_unsupported,
        "answers_audited": len(audited),

        "fabricated_citations": sum(r.get("fabricated_citations", 0) for r in ok),

        "avg_latency_ms": round(statistics.mean(latencies)) if latencies else 0,
        "median_latency_ms": round(statistics.median(latencies)) if latencies else 0,
        "p95_latency_ms": (
            round(sorted(latencies)[min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))])
            if latencies else 0
        ),
        "avg_llm_calls": round(statistics.mean(production_calls), 2) if production_calls else 0,
        "total_llm_calls": sum(production_calls),
        "total_audit_calls": sum(r.get("audit_calls", 0) for r in ok),

        "guardrail_decisions": decisions,
        "refusal_by_category": by_category,
    }


# ==========================================================================
# Report
# ==========================================================================


def build_report(summaries: dict[str, dict], meta: dict) -> str:
    """Write results.md: the comparison table plus how to read it."""
    off = summaries.get("off")
    on = summaries.get("on")

    lines: list[str] = []
    lines.append("# Evaluation results\n")
    lines.append(
        "The same 50 golden-set questions run twice: once with the verification "
        "layer **off** (plain RAG — the baseline) and once with it **on**. Same "
        "retrieval, same model, same prompts; the only difference is whether "
        "answers are checked before being returned.\n"
    )
    lines.append("## Setup\n")
    lines.append(f"- **Date:** {meta['date']}")
    lines.append(f"- **LLM:** `{meta['model']}` via {meta['base_url']}")
    lines.append(f"- **Embeddings:** `{meta['embedding_model']}` (384-dim, cosine)")
    lines.append(f"- **Retrieval:** top-{meta['top_k']}, refusal threshold "
                 f"{meta['min_score']}")
    lines.append(f"- **Block threshold:** {meta['block_threshold']} "
                 "(unsupported-claim fraction above which an answer is refused)")
    lines.append(f"- **Corpus:** {meta['chunks']} chunks from 15 central schemes")
    lines.append(f"- **Golden set:** {meta['answerable']} answerable "
                 f"(English/Hindi/Hinglish) + {meta['unanswerable']} unanswerable\n")

    if off and on:
        def delta(key: str, higher_is_better: bool, unit: str = "%") -> str:
            before, after = off[key], on[key]
            difference = round(after - before, 1)
            if abs(difference) < 0.05:
                arrow = "no change"
            else:
                improved = (difference > 0) == higher_is_better
                arrow = f"{'better' if improved else 'worse'} ({difference:+}{unit})"
            return f"| {_LABELS[key]} | {before}{unit} | {after}{unit} | {arrow} |"

        lines.append("## Headline comparison\n")
        lines.append("| Metric | Guardrail OFF | Guardrail ON | Change |")
        lines.append("|---|---|---|---|")
        lines.append(delta("unsupported_claim_rate", higher_is_better=False))
        lines.append(delta("answers_with_unsupported_claim_rate", higher_is_better=False))
        lines.append(delta("correct_refusal_rate", higher_is_better=True))
        lines.append(delta("over_blocking_rate", higher_is_better=False))
        lines.append(delta("answered_rate", higher_is_better=True))
        lines.append(delta("key_fact_coverage", higher_is_better=True))
        lines.append(delta("retrieval_hit_rate", higher_is_better=True))
        lines.append(
            f"| Avg LLM calls / question | {off['avg_llm_calls']} | "
            f"{on['avg_llm_calls']} | +{round(on['avg_llm_calls'] - off['avg_llm_calls'], 2)} |"
        )
        lines.append(
            f"| Avg latency | {off['avg_latency_ms'] / 1000:.1f}s | "
            f"{on['avg_latency_ms'] / 1000:.1f}s | "
            f"{(on['avg_latency_ms'] - off['avg_latency_ms']) / 1000:+.1f}s |"
        )
        lines.append(
            f"| p95 latency | {off['p95_latency_ms'] / 1000:.1f}s | "
            f"{on['p95_latency_ms'] / 1000:.1f}s | "
            f"{(on['p95_latency_ms'] - off['p95_latency_ms']) / 1000:+.1f}s |"
        )
        lines.append("")

        lines.append("### How to read this\n")
        lines.append(
            f"- **Unsupported claim rate** is the headline number: of all factual "
            f"claims in answers the user actually saw, how many were not backed by "
            f"the retrieved official text. It went from **{off['unsupported_claim_rate']}%** "
            f"to **{on['unsupported_claim_rate']}%**.\n"
        )
        lines.append(
            f"- **Correct refusal rate** is how often an unanswerable question was "
            f"refused: **{off['correct_refusal_rate']}%** → "
            f"**{on['correct_refusal_rate']}%**. The baseline already refuses "
            "off-topic questions, because the retrieval threshold catches those "
            "before any generation happens. The guardrail's contribution is on the "
            "harder cases — see the breakdown below.\n"
        )
        lines.append(
            f"- **Over-blocking rate** is the cost: answerable questions that were "
            f"refused anyway, **{off['over_blocking_rate']}%** → "
            f"**{on['over_blocking_rate']}%**. This is the price of strictness, and "
            "it is the number to watch if the guardrail is tuned harder.\n"
        )
        lines.append(
            f"- **Cost**: verification adds "
            f"{round(on['avg_llm_calls'] - off['avg_llm_calls'], 2)} LLM calls and "
            f"{(on['avg_latency_ms'] - off['avg_latency_ms']) / 1000:.1f}s per "
            "question on average.\n"
        )

    # Per-condition detail
    for key, label in (("off", "Guardrail OFF (baseline)"), ("on", "Guardrail ON")):
        summary = summaries.get(key)
        if not summary:
            continue
        lines.append(f"## {label}\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Questions run | {summary['questions']} "
                     f"({summary['answerable']} answerable, {summary['unanswerable']} not) |")
        lines.append(f"| Errors | {summary['errors']} |")
        lines.append(f"| Retrieval hit rate (expected scheme in top-k) | "
                     f"{summary['retrieval_hit_rate']}% |")
        lines.append(f"| Retrieval hit rate (top-1) | {summary['retrieval_hit_rate_top1']}% |")
        lines.append(f"| Key fact coverage | {summary['key_fact_coverage']}% "
                     f"({summary['key_facts_found']}/{summary['key_facts_total']} facts) |")
        lines.append(f"| Answered (of answerable) | {summary['answered_rate']}% |")
        lines.append(f"| Over-blocking rate | {summary['over_blocking_rate']}% |")
        lines.append(f"| Correct refusal rate | {summary['correct_refusal_rate']}% |")
        lines.append(f"| Unsupported claim rate (delivered) | "
                     f"{summary['unsupported_claim_rate']}% "
                     f"({summary['delivered_unsupported']}/{summary['delivered_claims']} claims) |")
        lines.append(f"| Answers containing any unsupported claim | "
                     f"{summary['answers_with_unsupported_claim_rate']}% "
                     f"({summary['answers_with_unsupported_claim']}/{summary['answers_audited']}) |")
        lines.append(f"| Fabricated citations caught in code | "
                     f"{summary['fabricated_citations']} |")
        lines.append(f"| Avg / median / p95 latency | "
                     f"{summary['avg_latency_ms'] / 1000:.1f}s / "
                     f"{summary['median_latency_ms'] / 1000:.1f}s / "
                     f"{summary['p95_latency_ms'] / 1000:.1f}s |")
        lines.append(f"| Avg LLM calls per question | {summary['avg_llm_calls']} |")
        lines.append(f"| Total production LLM calls | {summary['total_llm_calls']} |")
        lines.append(f"| Extra calls used for measurement only | "
                     f"{summary['total_audit_calls']} |")
        lines.append("")

        lines.append(f"**Guardrail decisions:** "
                     f"{', '.join(f'{k}={v}' for k, v in sorted(summary['guardrail_decisions'].items()))}\n")

        lines.append("**Refusal rate by type of unanswerable question:**\n")
        lines.append("| Type | Questions | Refused | Rate |")
        lines.append("|---|---|---|---|")
        for category, bucket in sorted(summary["refusal_by_category"].items()):
            lines.append(f"| `{category}` | {bucket['n']} | {bucket['refused']} | {bucket['rate']}% |")
        lines.append("")

    lines.append("## What the categories mean\n")
    lines.append(
        "- **`off_topic`** — not about schemes at all (\"capital of France\"). "
        "Caught by the retrieval threshold before any LLM call, so these are "
        "refused in both conditions and cost nothing.\n"
        "- **`fake_scheme`** — an invented but plausible scheme name "
        "(\"PM Free Laptop Yojana 2026\"). These score *above* the retrieval "
        "threshold because they look exactly like scheme questions, so the "
        "threshold cannot help. Only claim-level verification can.\n"
        "- **`not_indexed`** — a **real** scheme we do not cover (Sukanya "
        "Samriddhi, Ayushman Bharat PM-JAY). The hardest case and the most "
        "dangerous: retrieval returns confident, real, *wrong-scheme* evidence. "
        "Measured at 0.71 similarity for the Sukanya question — far above the "
        "0.45 threshold.\n"
        "- **`trick_unstated_fact`** — a real scheme, but asking for something the "
        "official source never states.\n"
    )

    lines.append("## Honest limitations of this evaluation\n")
    lines.append(
        "- **50 questions is small.** Differences of a few percent are noise. The "
        "numbers show direction and rough magnitude, not precision.\n"
        "- **The judge is the same model family as the generator.** A shared blind "
        "spot would be invisible to this evaluation. A stronger setup would use a "
        "different model as judge, and a human-labelled subset to validate it.\n"
        "- **The audit uses an LLM too**, so the unsupported-claim rate is itself "
        "an estimate, biased toward strictness by the judge prompt.\n"
        "- **Key fact coverage uses substring matching.** A correct answer that "
        "phrases a number differently can be scored as a miss.\n"
        "- **Written by the same person who wrote the system**, so the golden set "
        "may under-represent failure modes I did not think of. Every expected fact "
        "is at least verified to exist in the source data "
        "(`scripts/build_golden_set.py` fails the build otherwise).\n"
    )

    lines.append("---\n")
    lines.append(
        "Regenerate with `python eval/run_eval.py`. Per-question records, "
        "including every answer and verdict, are written to "
        "`eval/raw_results.json`.\n"
    )
    return "\n".join(lines)


_LABELS = {
    "unsupported_claim_rate": "**Unsupported claim rate** (delivered answers)",
    "answers_with_unsupported_claim_rate": "Answers containing any unsupported claim",
    "correct_refusal_rate": "Correct refusal rate (unanswerable)",
    "over_blocking_rate": "Over-blocking rate (answerable refused)",
    "answered_rate": "Answered rate (of answerable)",
    "key_fact_coverage": "Key fact coverage",
    "retrieval_hit_rate": "Retrieval hit rate (top-5)",
}


# ==========================================================================
# Main
# ==========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SchemeSure.")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N questions")
    parser.add_argument("--condition", choices=["off", "on", "both"], default="both")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="seconds between questions (Groq rate limits)")
    parser.add_argument("--no-audit", action="store_true",
                        help="skip the claim-level audit of delivered answers")
    args = parser.parse_args()

    if not GOLDEN_PATH.exists():
        print(f"ERROR: {GOLDEN_PATH} not found. Run scripts/build_golden_set.py first.")
        return 1

    rows = [
        json.loads(line)
        for line in io.open(GOLDEN_PATH, encoding="utf-8").read().splitlines()
        if line.strip()
    ]
    if args.limit:
        # Keep both kinds of question in a smoke run, or the refusal metrics are
        # meaningless.
        answerable = [r for r in rows if r["answerable"]][: max(1, args.limit // 2)]
        unanswerable = [r for r in rows if not r["answerable"]][: max(1, args.limit // 2)]
        rows = answerable + unanswerable

    base = Settings()
    base.require_llm_key()

    print(f"Golden set    : {len(rows)} questions")
    print(f"Model         : {base.llm_model}")
    print(f"Threshold     : {base.retrieval_min_score}")
    print(f"Conditions    : {args.condition}")
    print(f"Claim audit   : {'off' if args.no_audit else 'on'}")

    # Shared: the embedding model is cached at module level, so this loads once.
    retriever = Retriever(base)
    print(f"Indexed chunks: {retriever.count()}\n")

    conditions = ["off", "on"] if args.condition == "both" else [args.condition]
    summaries: dict[str, dict] = {}
    all_records: dict[str, list[dict]] = {}

    for condition in conditions:
        settings = Settings()
        settings.guardrail_enabled = condition == "on"

        llm = LLMClient(settings)
        pipeline = AnswerPipeline(retriever, llm, settings)

        # The auditor always verifies, regardless of the condition under test.
        auditor = None
        if not args.no_audit:
            audit_settings = Settings()
            audit_settings.guardrail_enabled = True
            auditor = Guardrail(llm, audit_settings)

        print("=" * 72)
        print(f"CONDITION: guardrail {condition.upper()}")
        print("=" * 72)

        records: list[dict] = []
        for index, row in enumerate(rows, start=1):
            record = run_one(row, pipeline, retriever, auditor)
            records.append(record)

            status = record.get("status", "ERROR")
            marker = {
                "verified": "OK ", "partially_verified": "REP",
                "refused": "REF", "unverified": "RAW",
            }.get(status, "ERR")
            expected_marker = "" if row["answerable"] else " (should refuse)"
            print(
                f"  [{index:2d}/{len(rows)}] {marker} {row['id']:4s} "
                f"{record.get('latency_ms', 0) / 1000:5.1f}s "
                f"{record.get('llm_calls', 0)}c "
                f"score={record.get('best_retrieval_score', 0):.2f} "
                f"{row['question'][:40]}{expected_marker}"
            )
            if record.get("error"):
                print(f"           error: {record['error'][:100]}")

            # Groq's free tier is per-minute; a short pause keeps the run inside it.
            if index < len(rows):
                time.sleep(args.delay)

        summaries[condition] = summarise(records)
        all_records[condition] = records

        summary = summaries[condition]
        print(f"\n  -- guardrail {condition.upper()} summary --")
        print(f"     retrieval hit rate       : {summary['retrieval_hit_rate']}%")
        print(f"     key fact coverage        : {summary['key_fact_coverage']}%")
        print(f"     correct refusal rate     : {summary['correct_refusal_rate']}%")
        print(f"     over-blocking rate       : {summary['over_blocking_rate']}%")
        print(f"     unsupported claim rate   : {summary['unsupported_claim_rate']}%")
        print(f"     avg latency              : {summary['avg_latency_ms'] / 1000:.1f}s")
        print(f"     avg LLM calls / question : {summary['avg_llm_calls']}\n")

    meta = {
        "date": date.today().isoformat(),
        "model": base.llm_model,
        "base_url": base.llm_base_url,
        "embedding_model": base.embedding_model,
        "top_k": base.retrieval_top_k,
        "min_score": base.retrieval_min_score,
        "block_threshold": base.guardrail_block_threshold,
        "chunks": retriever.count(),
        "answerable": sum(1 for r in rows if r["answerable"]),
        "unanswerable": sum(1 for r in rows if not r["answerable"]),
    }

    with io.open(RAW_PATH, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summaries": summaries, "records": all_records},
                  fh, ensure_ascii=False, indent=2)

    report = build_report(summaries, meta)
    with io.open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {RAW_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
