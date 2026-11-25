"""
Retrieval-only benchmark and refusal-threshold analysis.

THE QUESTIONS THIS ANSWERS
  1. Does retrieval find the right scheme?  (Recall@1/3/5, MRR@10, per language)
  2. How much does the LLM query-normalizer help?  (variant "raw" vs "normalized")
  3. Is the 0.45 refusal threshold sensible?  (score distributions, a sweep over
     thresholds, AUROC)

Two VARIANTS of the query that gets embedded:
  raw         the user's question exactly as typed. Costs 0 LLM calls.
  normalized  the English rewrite from the LLM normalizer (what production uses).
              Costs 1 LLM call per question the first time; results are cached in
              eval/cache/normalized.jsonl and shared with run_eval.py.

Nothing here generates or verifies an answer, so it is cheap enough to re-run
whenever the index, the embedding model or the threshold changes.

HONESTY NOTE ABOUT THE THRESHOLD
The 0.45 threshold in config.py was chosen (docs/DECISIONS.md) by looking at
about 22 queries that overlap this golden set (a01, "capital of France", the
pizza question, the Sukanya question ...). So the analysis below is a
CONSISTENCY CHECK that the chosen value behaves sensibly on this set. It is not
a held-out validation, and it must not be quoted as one.

Run:
    python eval/run_retrieval.py --variants raw            # 0 LLM calls
    python eval/run_retrieval.py --variants raw,normalized # needs LLM_API_KEY
    python eval/run_retrieval.py --analyze-only            # re-analyse saved json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.common import (  # noqa: E402
    add_selection_args,
    fmt_interval,
    golden_sha1,
    load_rows,
    rows_from_args,
    selection_is_subset,
    subset_dir_name,
)
from eval.retrieval_metrics import (  # noqa: E402
    DEFAULT_KS,
    SWEEP_GRID,
    score_distributions,
    summarize_retrieval,
    sweep_threshold,
    threshold_aurocs,
)

DEFAULT_OUT_DIR = ROOT / "eval"
TOP_N = 10  # depth stored per question; Recall@k and MRR@10 are computed from it


# ==========================================================================
# Running the retrieval
# ==========================================================================


def retrieve_variant(rows, variant, retriever, normalizer_cache, top_n=TOP_N):
    """Run every question through one variant and keep the full ranked list."""
    from core.query_normalizer import normalize_query

    records = []
    for index, row in enumerate(rows, start=1):
        if variant == "raw":
            # llm=None is the zero-cost path: raw question + script-based language guess.
            normalized = normalize_query(None, row["question"])
        else:
            normalized = normalizer_cache.get(row["question"]).normalized

        # min_score=0.0 so the search never reports "low evidence" here: we want
        # the raw ranking and let the analysis decide where to cut.
        result = retriever.search(normalized.normalized_query, top_k=top_n, min_score=0.0)
        records.append({
            "id": row["id"],
            "question": row["question"],
            "language": row["language"],
            "answerable": row["answerable"],
            "category": row["category"],
            "expected_scheme": row.get("expected_scheme"),
            "query": normalized.normalized_query,
            "detected_language": normalized.language.value,
            # For "raw" this is always True (no LLM was asked), so it is only
            # informative for the normalized variant.
            "fallback_used": normalized.fallback_used,
            "best_score": result.best_score,
            "ranked": [
                {
                    "chunk_id": chunk.chunk_id,
                    "scheme_id": chunk.metadata.scheme_id,
                    "section": chunk.metadata.section,
                    "score": chunk.score,
                }
                for chunk in result.chunks
            ],
        })
        if index % 10 == 0 or index == len(rows):
            print(f"  [{variant}] {index}/{len(rows)} questions")
    return records


# ==========================================================================
# Markdown / CSV writers
# ==========================================================================


def _preliminary_banner(meta: dict) -> str:
    if meta.get("subset"):
        return (f"> **PRELIMINARY: n={meta['n_questions']} of {meta['n_golden']} golden-set "
                "questions (a subset chosen for language/category coverage, not in proportion "
                "to the full set). Do not quote these figures as the benchmark.**\n")
    return ""


def build_retrieval_report(data: dict, ks=DEFAULT_KS) -> str:
    meta = data["meta"]
    lines = ["# Retrieval benchmark\n"]
    banner = _preliminary_banner(meta)
    if banner:
        lines.append(banner)
    lines.append(
        "Does the retriever put the right scheme in the top results? Measured with "
        "no answer generation and no verification, so only the embedding model and "
        "the index are being tested.\n"
    )
    lines.append("## Setup\n")
    lines.append(f"- **Date:** {meta['date']}")
    source = meta.get("questions_file", "golden_set.jsonl")
    lines.append(f"- **Questions:** n={meta['n_questions']} (`{source}` has {meta['n_golden']})")
    lines.append(f"- **Embeddings:** `{meta['embedding_model']}`, cosine similarity")
    lines.append(f"- **Depth stored:** top {meta['top_n']} chunks per question")
    lines.append("- **Relevance:** scheme-level. A hit means the expected scheme owns at "
                 "least one of the top-k chunks (the golden set has no gold chunk ids).")
    lines.append("- **Intervals:** 95% Wilson score intervals in brackets.\n")

    for variant, content in data["variants"].items():
        summary = summarize_retrieval(content["records"], ks)
        title = {
            "raw": "Variant `raw` (question as typed, 0 LLM calls)",
            "normalized": "Variant `normalized` (English rewrite from the LLM, as in production)",
        }.get(variant, variant)
        lines.append(f"## {title}\n")
        header = ["Language", "n answerable"] + [f"Recall@{k}" for k in ks]
        header += ["MRR@10", "MRR@10 (distinct schemes)", "Language-ID accuracy"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for language, group in summary.items():
            label = "**all**" if language == "all" else language
            cells = [label, str(group["n"])]
            for k in ks:
                hits, n = group["recall"][k]
                cells.append(fmt_interval(hits, n))
            cells.append("n/a" if group["mrr"] is None else f"{group['mrr']:.3f}")
            cells.append("n/a" if group["mrr_scheme"] is None else f"{group['mrr_scheme']:.3f}")
            correct, total = group["lang_id"]
            cells.append(fmt_interval(correct, total))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if variant == "raw":
            lines.append(
                "*Language-ID for `raw` is the offline script/word heuristic "
                "(`detect_language_heuristic`), not an LLM.*\n"
            )
        fallbacks = sum(1 for r in content["records"] if r.get("fallback_used"))
        if variant == "normalized":
            lines.append(f"*Normalizer fallbacks (LLM failed, raw question used): {fallbacks}.*\n")

    if "normalized" not in data["variants"]:
        lines.append(
            "> The `normalized` variant has **not been run** (it needs the LLM, one "
            "call per question, cached afterwards). Until it runs, the effect of "
            "the normalizer is unmeasured and no comparison is claimed.\n"
        )

    lines.append("## How to read this\n")
    if meta.get("translation_paired"):
        language_note = (
            "- Every question is asked in English, Hindi and Hinglish (same `pair_id`), "
            "so per-language differences hold the content fixed. The translations are "
            "author-written and not yet reviewed by a second native speaker.\n")
    else:
        language_note = (
            "- Hindi and Hinglish questions in the golden set are **not translations** of "
            "the English ones, so a per-language difference mixes language with question "
            "difficulty. Treat it as a hint, not a controlled comparison.\n")
    lines.append(
        language_note +
        "- Per-language n is small (see the n column); the Wilson intervals are wide "
        "on purpose. A single question moves a small-group percentage a lot.\n"
        "- Recall@k only says the right *scheme* was retrieved, not that the "
        "retrieved chunk contains the answer. End-to-end accuracy is measured by "
        "`eval/run_eval.py`.\n"
    )
    lines.append("---\n")
    lines.append("Regenerate with `python eval/run_retrieval.py`. Full ranked lists: "
                 "`retrieval_raw.json`.\n")
    return "\n".join(lines)


def _fmt_dist(dist: dict) -> str:
    if not dist["n"]:
        return "n=0 | - | - | -"
    return f"{dist['n']} | {dist['min']:.3f} | {dist['median']:.3f} | {dist['max']:.3f}"


def build_threshold_report(data: dict, configured: float) -> str:
    meta = data["meta"]
    lines = ["# Refusal-threshold analysis\n"]
    banner = _preliminary_banner(meta)
    if banner:
        lines.append(banner)
    lines.append(
        "The retriever refuses a question outright when the **best chunk similarity "
        f"is below a threshold** (currently **{configured}**). This page looks at the "
        "best-score distributions and sweeps the threshold. It uses saved retrieval "
        "results only: 0 LLM calls.\n"
    )
    lines.append(
        "> **Read this as a consistency check, not a validation.** The value 0.45 was "
        "originally chosen (docs/DECISIONS.md) from about 22 queries that overlap this "
        "golden set, so it was not tuned on held-out data. Showing that it behaves "
        "sensibly here does not show it will generalise.\n"
    )

    for variant, content in data["variants"].items():
        records = content["records"]
        lines.append(f"## Variant `{variant}`\n")
        if variant == "raw":
            lines.append(
                "*Production refuses on the **normalized** query's score. The raw "
                "variant shows what would happen without the LLM normalizer; "
                "Hindi/Hinglish scores are expected to be lower here.*\n"
            )
        dist = score_distributions(records)

        lines.append("### Best-score distributions\n")
        lines.append("| Group | n | min | median | max |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| answerable (all) | {_fmt_dist(dist['answerable_all'])} |")
        for language, d in dist["answerable_by_language"].items():
            lines.append(f"| answerable, {language} | {_fmt_dist(d)} |")
        for category, d in dist["unanswerable_by_category"].items():
            lines.append(f"| unanswerable, `{category}` | {_fmt_dist(d)} |")
        lines.append("")

        aurocs = threshold_aurocs(records)
        def fmt_auc(value):
            return "n/a" if value is None else f"{value:.3f}"
        lines.append("### AUROC of best score (1.0 = perfectly separable, 0.5 = no signal)\n")
        lines.append(f"- answerable vs **all** unanswerable: "
                     f"**{fmt_auc(aurocs['answerable_vs_all_unanswerable'])}**")
        lines.append(f"- answerable vs **off_topic** only: "
                     f"**{fmt_auc(aurocs['answerable_vs_off_topic'])}**\n")
        lines.append(
            "The threshold is only expected to catch `off_topic`. `fake_scheme` and "
            "`not_indexed` questions look like real scheme questions and score high, "
            "so they are left to the claim-level guardrail.\n"
        )

        rows = sweep_threshold(records, SWEEP_GRID)
        categories = sorted(rows[0]["abstain_by_category"]) if rows else []
        lines.append("### Threshold sweep\n")
        lines.append("`abstain` = best score < t. Over-block = answerable questions "
                     "refused; the other columns = unanswerable questions refused "
                     "(higher is better).\n")
        header = ["t", "Over-block (answerable)", "Abstain, all unanswerable"]
        header += [f"`{c}`" for c in categories]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in rows:
            mark = " **<- configured**" if abs(row["threshold"] - configured) < 1e-9 else ""
            cells = [f"{row['threshold']:.2f}{mark}",
                     _ratio(row["over_block"]), _ratio(row["abstain_all"])]
            cells += [_ratio(row["abstain_by_category"][c]) for c in categories]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("---\n")
    lines.append("Regenerate with `python eval/run_retrieval.py --analyze-only`. "
                 "Raw numbers: `threshold_sweep.csv`.\n")
    return "\n".join(lines)


def _ratio(pair) -> str:
    hits, n = pair
    if not n:
        return "n/a (n=0)"
    return f"{hits}/{n} ({100 * hits / n:.0f}%)"


def write_threshold_csv(data: dict, path: Path) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        writer = None
        for variant, content in data["variants"].items():
            for row in sweep_threshold(content["records"], SWEEP_GRID):
                flat = {
                    "variant": variant,
                    "threshold": row["threshold"],
                    "over_block_count": row["over_block"][0],
                    "answerable_n": row["over_block"][1],
                    "abstain_all_count": row["abstain_all"][0],
                    "unanswerable_n": row["abstain_all"][1],
                }
                for category, (hits, n) in row["abstain_by_category"].items():
                    flat[f"abstain_{category}_count"] = hits
                    flat[f"{category}_n"] = n
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(flat))
                    writer.writeheader()
                writer.writerow(flat)


# ==========================================================================
# Main
# ==========================================================================


def write_outputs(data: dict, out_dir: Path, configured: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with io.open(out_dir / "retrieval_raw.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    with io.open(out_dir / "retrieval_results.md", "w", encoding="utf-8") as fh:
        fh.write(build_retrieval_report(data))
    with io.open(out_dir / "threshold_results.md", "w", encoding="utf-8") as fh:
        fh.write(build_threshold_report(data, configured))
    write_threshold_csv(data, out_dir / "threshold_sweep.csv")
    for name in ("retrieval_raw.json", "retrieval_results.md",
                 "threshold_results.md", "threshold_sweep.csv"):
        print(f"Wrote {out_dir / name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="SchemeSure retrieval benchmark.")
    parser.add_argument("--variants", default="raw",
                        help="comma-separated: raw, normalized (default raw = 0 LLM calls)")
    parser.add_argument("--out-dir", default="",
                        help="output folder (default eval/, or eval/runs/<subset> for subsets)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="skip retrieval; re-analyse the saved retrieval_raw.json")
    parser.add_argument("--questions", default="",
                        help="question file (default eval/golden_set.jsonl); "
                             "e.g. eval/paper_set.jsonl")
    add_selection_args(parser)
    args = parser.parse_args()

    from app.config import Settings

    settings = Settings()
    questions_path = Path(args.questions) if args.questions else None
    all_rows = load_rows(questions_path)
    rows = rows_from_args(args, all_rows)
    subset = selection_is_subset(rows, all_rows)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif questions_path is not None:
        # A different question file never overwrites the golden-set results.
        out_dir = ROOT / "eval" / "runs" / questions_path.stem
        if subset:
            out_dir = out_dir / subset_dir_name(rows)
    elif subset:
        out_dir = ROOT / "eval" / "runs" / subset_dir_name(rows)
    else:
        out_dir = DEFAULT_OUT_DIR
    raw_path = out_dir / "retrieval_raw.json"

    if args.analyze_only:
        if not raw_path.exists():
            print(f"ERROR: {raw_path} not found; run without --analyze-only first.")
            return 1
        data = json.loads(io.open(raw_path, encoding="utf-8").read())
        write_outputs(data, out_dir, settings.retrieval_min_score)
        return 0

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in ("raw", "normalized")]
    if unknown or not variants:
        print(f"ERROR: unknown variant(s) {unknown}; choose from raw, normalized.")
        return 1

    from core.retriever import Retriever

    normalizer_cache = None
    if "normalized" in variants:
        from app.llm_client import LLMClient
        from eval.norm_cache import NormalizationCache

        settings.require_llm_key()
        normalizer_cache = NormalizationCache(LLMClient(settings), settings.llm_model)

    retriever = Retriever(settings)
    print(f"Questions: {len(rows)}   indexed chunks: {retriever.count()}   out: {out_dir}")

    meta = {
        "date": date.today().isoformat(),
        "embedding_model": settings.embedding_model,
        "top_n": TOP_N,
        "top_k": settings.retrieval_top_k,
        "configured_min_score": settings.retrieval_min_score,
        "question_ids": [r["id"] for r in rows],
        "n_questions": len(rows),
        "n_golden": len(all_rows),
        "subset": subset,
        "golden_sha1": golden_sha1(all_rows),
        "questions_file": (questions_path or Path("eval/golden_set.jsonl")).name,
        # True when every question comes in en/hi/hinglish versions sharing a pair_id.
        "translation_paired": all("pair_id" in r for r in all_rows),
    }

    # Keep variants from an earlier run only if they are truly comparable: same
    # questions, same golden-set content AND the same embedding model. (Matching
    # question ids alone would let a model-A "normalized" sit beside a model-B "raw".)
    variants_data: dict = {}
    if raw_path.exists():
        try:
            previous = json.loads(io.open(raw_path, encoding="utf-8").read())
            earlier = previous.get("meta", {})
            if all(earlier.get(key) == meta[key]
                   for key in ("question_ids", "embedding_model", "golden_sha1")):
                variants_data = previous.get("variants", {})
        except (OSError, json.JSONDecodeError):
            pass

    for variant in variants:
        print(f"Variant: {variant}")
        # Each variant keeps its OWN date / embedding model, so a reused variant
        # is never relabelled with the newer run's metadata.
        variants_data[variant] = {
            "meta": {"date": meta["date"], "embedding_model": meta["embedding_model"]},
            "records": retrieve_variant(rows, variant, retriever, normalizer_cache),
        }

    ordered = {k: variants_data[k] for k in ("raw", "normalized") if k in variants_data}
    write_outputs({"meta": meta, "variants": ordered}, out_dir, settings.retrieval_min_score)
    return 0


if __name__ == "__main__":
    sys.exit(main())
