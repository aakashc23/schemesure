"""
Experiment E9b: does the LLM claim judge catch wrong numbers?

E9 (eval/run_numeric_perturbation.py, 0 LLM calls) showed the Python number
check flags every wrong number that is ABSENT from the evidence, and by
construction none that appear ELSEWHERE in it ("perturbed_in"). In production
that blind spot is left to the LLM judge. This script measures whether the judge
actually covers it, on the same saved items (eval/runs/numeric_perturbation/
items.jsonl).

DESIGN
  * One claim per judge call, with that item's own evidence passages. Never
    batch a correct sentence with its perturbed twin: the judge could then spot
    the difference between the two claims instead of checking the evidence.
  * The exact production judge: core.guardrail.judge_claims, including the
    Python check that a cited passage really was retrieved. A claim counts as
    flagged when its final verdict is NOT_SUPPORTED.
  * Sentences are taken in a seeded random order, and for each sentence the
    kinds are judged in a fixed order (perturbed_in, faithful_original,
    perturbed_out, then faithful_format). So a run cut short by quota still
    gives a balanced, random sample of the most informative items first.
  * Checkpointed after every call (judge_results.jsonl); re-running continues.
    --max-tokens stops the run before it eats a whole day's quota.

Run:  python eval/run_numeric_judge.py --max-tokens 90000 --delay 3
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.common import fmt_interval  # noqa: E402

RUN_DIR = ROOT / "eval" / "runs" / "numeric_perturbation"
KIND_ORDER = ("perturbed_in", "faithful_original", "perturbed_out", "faithful_format")


def ordered_items(items: list[dict], seed: int) -> list[dict]:
    """Seeded random order of source sentences; fixed kind order within each."""
    by_sentence: dict[tuple, list[dict]] = {}
    for item in items:
        by_sentence.setdefault((item["source_chunk_id"], item["original"]), []).append(item)
    keys = sorted(by_sentence)
    random.Random(seed).shuffle(keys)
    out: list[dict] = []
    for key in keys:
        group = by_sentence[key]
        for kind in KIND_ORDER:
            out.extend(i for i in group if i["kind"] == kind)
    return out


def item_key(item: dict) -> str:
    """Stable identity of an item across runs."""
    return f"{item['source_chunk_id']}|{item['kind']}|{item['claim']}"


def build_report(results: list[dict], meta: dict) -> str:
    kinds = ("faithful_original", "faithful_format", "perturbed_out", "perturbed_in")
    out = ["# Number-perturbation test, LLM judge (E9b)\n",
           f"- **Date:** {meta['date']}  ·  **model:** `{meta['model']}`  ·  **seed:** {meta['seed']}",
           f"- **Judged:** {len(results)} items from {len({r['original'] for r in results})} "
           f"sentences, one claim per call, each with its own top-{meta['top_k']} evidence",
           "- A claim is **flagged** when the production judge (`core.guardrail.judge_claims`, "
           "including the Python check that the cited passage was retrieved) returns NOT_SUPPORTED.",
           "- Items are a seeded random sample of sentences; the quota decides how many.\n",
           "## Flag rates: judge vs the Python number check, on the same items\n",
           "For correct items a flag is a **false alarm**; for wrong-number items it is a "
           "**detection**.\n",
           "| Item type | n | LLM judge flags | Python number check flags | either flags |",
           "|---|---|---|---|---|"]
    for kind in kinds:
        rows = [r for r in results if r["kind"] == kind]
        judge = sum(r["judge_flag"] for r in rows)
        rule = sum(bool(r["rule_flags"]) for r in rows)
        either = sum(r["judge_flag"] or bool(r["rule_flags"]) for r in rows)
        out.append(f"| `{kind}` | {len(rows)} | {fmt_interval(judge, len(rows))} | "
                   f"{fmt_interval(rule, len(rows))} | {fmt_interval(either, len(rows))} |")
    out.append("")

    missed = [r for r in results if r["kind"] == "perturbed_in" and not r["judge_flag"]]
    out.append(f"## Wrong numbers the judge passed ({len(missed)})\n")
    out.append("These reach the user in production: the number exists elsewhere in the "
               "evidence, so the Python check cannot see them either.\n")
    for r in missed[:12]:
        out.append(f"- {r['value']} → {r['wrong_value']}: *{r['claim'][:180]}*")
    out.append("")
    alarms = [r for r in results if r["kind"].startswith("faithful") and r["judge_flag"]]
    out.append(f"## Correct sentences the judge rejected ({len(alarms)})\n")
    for r in alarms[:12]:
        out.append(f"- `{r['kind']}`: *{r['claim'][:180]}*")
    out.append("")
    out.append("Regenerate: `python eval/run_numeric_judge.py --report-only`.\n")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="E9b: LLM judge on number-perturbation items.")
    parser.add_argument("--max-tokens", type=int, default=90000,
                        help="stop after this many tokens in THIS run (quota guard)")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between calls")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    items_path = RUN_DIR / "items.jsonl"
    results_path = RUN_DIR / "judge_results.jsonl"
    items = [json.loads(line) for line in io.open(items_path, encoding="utf-8") if line.strip()]
    done: dict[str, dict] = {}
    if results_path.exists():
        for line in io.open(results_path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                done[item_key(row)] = row

    from app.config import Settings

    settings = Settings()
    meta = {"date": date.today().isoformat(), "model": settings.llm_model, "seed": args.seed,
            "top_k": len(items[0]["evidence_ids"]) if items else 0}

    if not args.report_only:
        from app.llm_client import CallCounter, LLMClient, LLMError
        from app.schemas import ChunkMetadata, RetrievedChunk, Verdict
        from core.guardrail import judge_claims
        from core.ingest import get_collection

        stored = get_collection(settings).get(include=["documents", "metadatas"])
        by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))

        def chunk(cid: str) -> RetrievedChunk:
            text, m = by_id[cid]
            return RetrievedChunk(chunk_id=cid, text=text, score=1.0, metadata=ChunkMetadata(
                **{k: m.get(k, "") for k in ChunkMetadata.model_fields}))

        llm = LLMClient(settings)
        used = 0
        todo = [i for i in ordered_items(items, args.seed) if item_key(i) not in done]
        print(f"{len(done)} already judged, {len(todo)} to go, token budget {args.max_tokens}")
        with io.open(results_path, "a", encoding="utf-8") as fh:
            for n, item in enumerate(todo, start=1):
                if used >= args.max_tokens:
                    print(f"Stopping: token budget reached ({used}).")
                    break
                counter = CallCounter()
                try:
                    checks = judge_claims(llm, [item["claim"]], [chunk(c) for c in item["evidence_ids"]],
                                          counter=counter)
                except LLMError as exc:
                    print(f"  LLM error ({exc.kind}); stopping. Re-run to continue.")
                    if exc.kind in ("quota", "auth"):
                        break
                    continue
                used += counter.prompt_tokens + counter.completion_tokens
                check = checks[0]
                row = {**item, "judge_verdict": check.verdict.value,
                       "judge_flag": check.verdict == Verdict.NOT_SUPPORTED,
                       "judge_citation_valid": check.citation_valid,
                       "judge_tokens": counter.prompt_tokens + counter.completion_tokens}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                done[item_key(row)] = row
                if n % 20 == 0:
                    print(f"  {len(done)} judged, {used} tokens this run")
                time.sleep(args.delay)

    results = list(done.values())
    (RUN_DIR / "judge_results.md").write_text(build_report(results, meta), encoding="utf-8")
    print(f"Wrote {RUN_DIR / 'judge_results.md'} ({len(results)} items)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
