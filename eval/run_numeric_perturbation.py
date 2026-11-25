"""
Experiment E9: number-perturbation test for the claim verifiers.

THE QUESTION
A wrong number ("Rs 8,000" instead of "Rs 6,000") is the most consequential error
a welfare-scheme assistant can make, and the easiest one for a similarity-based
check to miss: swapping one number barely moves a sentence embedding. How well do
the verification signals catch it?

HOW THE TEST SET IS MADE (deterministic, seeded, 0 LLM calls)
Every sentence in the indexed passages that contains a checkable number (a rupee,
lakh/crore or percentage amount, or any number >= 10) becomes one test item.
Its evidence is what production would see: the top-k passages retrieved for that
sentence, with the source passage always included. From each item we make:

  faithful_format  the SAME value written differently: "6000" -> "6,000",
                   "Rs 6,000", "₹6000", Devanagari digits, and "2,00,000" <-> "2
                   lakh". A verifier must NOT flag these (false alarms).
  perturbed_out    the value replaced by a plausible wrong value that appears
                   nowhere in the evidence. A verifier SHOULD flag these.
  perturbed_in     the value replaced by ANOTHER number that does appear in the
                   evidence (e.g. the age limit swapped for the premium). This is
                   the known blind spot of an exact-match number check; measuring
                   it is the point, not hiding it.

SIGNALS MEASURED
  numeric rule  core.guardrail.numeric_and_link_grounding (flags = non-empty list)
  similarity    cosine(claim, source passage) with the production embedding model;
                reported as AUROC (faithful vs perturbed) and as the detection rate
                at the threshold that keeps false alarms on faithful items <= 5%.
  LLM judge     NOT run here (needs the API); the same items are written to
                items.jsonl so the judge and an NLI model can be scored on them later.

Run:  python eval/run_numeric_perturbation.py [--top-k 5] [--seed 0]
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.common import auroc, fmt_interval  # noqa: E402

OUT_DIR = ROOT / "eval" / "runs" / "numeric_perturbation"
_DEVANAGARI = str.maketrans("0123456789", "०१२३४५६७८९")
# Split after . ! ? ; — but not after "Rs." / "No." / "i.e." / "viz.", which are
# abbreviations inside an amount or a list, not sentence ends.
_SENTENCE_SPLIT = re.compile(
    r"(?<!\bRs\.)(?<!\brs\.)(?<!\bNo\.)(?<!\bno\.)(?<!i\.e\.)(?<!viz\.)"
    r"(?<=[.!?;])\s+|\n+")
_MARKDOWN = re.compile(r"[*_#>`|]+")


# ==========================================================================
# Pure helpers (unit-tested)
# ==========================================================================


def indian_grouping(value: int) -> str:
    """12345678 -> '1,23,45,678' (Indian digit grouping)."""
    digits = str(value)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return ",".join(groups + [tail])


def format_variants(value: Decimal, is_percent: bool) -> list[str]:
    """
    Other ways of writing exactly the same value. Each is a string that replaces
    the number token (its currency prefix / unit / % suffix handled by the caller).
    """
    variants: list[str] = []
    if value != value.to_integral_value():
        return [str(value).translate(_DEVANAGARI)]  # decimals: script change only
    whole = int(value)
    variants.append(indian_grouping(whole))
    variants.append(str(whole).translate(_DEVANAGARI))
    if not is_percent and whole >= 100000 and whole % 100000 == 0:
        variants.append(f"{whole // 100000} lakh")
    # de-duplicate, keep order
    return list(dict.fromkeys(variants))


def wrong_value(value: Decimal, known: set[Decimal], rng: random.Random) -> Decimal | None:
    """A plausible wrong value that is NOT among the evidence numbers."""
    candidates = [value * 2, value + value / 2, value * 10, value - value / 4,
                  value + 1000, value + 7]
    rng.shuffle(candidates)
    for candidate in candidates:
        candidate = candidate.quantize(Decimal(1)) if value == value.to_integral_value() \
            else candidate.quantize(Decimal("0.1"))
        if candidate > 0 and candidate != value and candidate.normalize() not in known:
            return candidate.normalize()
    return None


def render(value: Decimal) -> str:
    """Plain rendering of a Decimal without exponent notation."""
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def clean_sentence(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKDOWN.sub(" ", text)).strip()


# ==========================================================================
# Building the items
# ==========================================================================


def checkable_numbers(sentence: str):
    """(match, value) for every number the production rule would check."""
    from core.guardrail import _AMOUNT_RE, _amount_value

    for match in _AMOUNT_RE.finditer(sentence):
        value = _amount_value(match)
        if value is None:
            continue
        has_context = bool(match.group("cur") or match.group("unit") or match.group("pct"))
        if value < 10 and value == value.to_integral_value() and not has_context:
            continue
        # Years and URL-ish numbers are not facts worth perturbing here.
        if 1900 <= value <= 2100 and not has_context:
            continue
        yield match, value


def evidence_numbers(chunks) -> set[Decimal]:
    """The same set of 'known numbers' the production rule builds from passages."""
    from core.guardrail import (_AMOUNT_RE, _DEVANAGARI_DIGITS, _NUMBER_RE,
                                _amount_value, _canonical_number)

    haystack = " ".join(f"{c.text} {c.metadata.source_url}" for c in chunks)
    haystack = haystack.translate(_DEVANAGARI_DIGITS).lower()
    known = {v for t in _NUMBER_RE.findall(haystack) if (v := _canonical_number(t)) is not None}
    known |= {v for m in _AMOUNT_RE.finditer(haystack) if (v := _amount_value(m)) is not None}
    return known


def build_items(retriever, top_k: int, seed: int) -> list[dict]:
    from app.schemas import ChunkMetadata, RetrievedChunk

    rng = random.Random(seed)
    stored = retriever.collection.get(include=["documents", "metadatas"])
    items: list[dict] = []
    for chunk_id, text, meta in sorted(zip(stored["ids"], stored["documents"], stored["metadatas"])):
        source = RetrievedChunk(chunk_id=chunk_id, text=text, score=1.0,
                                metadata=ChunkMetadata(**{k: meta.get(k, "") for k in
                                                          ChunkMetadata.model_fields}))
        for raw_sentence in _SENTENCE_SPLIT.split(text):
            sentence = clean_sentence(raw_sentence)
            if not (20 <= len(sentence) <= 300) or "http" in sentence or "www." in sentence:
                continue
            numbers = list(checkable_numbers(sentence))
            if not numbers:
                continue
            match, value = rng.choice(numbers)

            # Evidence = what retrieval returns for this sentence, source included.
            result = retriever.search(sentence, top_k=top_k, min_score=0.0)
            evidence = [c for c in result.chunks if c.chunk_id != chunk_id][: top_k - 1]
            evidence = [source] + evidence
            known = evidence_numbers(evidence)

            start, end = match.span("num")
            unit = match.group("unit")
            is_pct = bool(match.group("pct"))

            def with_number(new_text: str, drop_unit: bool = False) -> str:
                if drop_unit and unit:
                    return sentence[:start] + new_text + sentence[match.end("unit"):]
                return sentence[:start] + new_text + sentence[end:]

            base = {"source_chunk_id": chunk_id, "scheme_id": meta.get("scheme_id"),
                    "original": sentence, "value": render(value),
                    "evidence_ids": [c.chunk_id for c in evidence]}

            items.append({**base, "kind": "faithful_original", "claim": sentence})
            # Format variants express the FULL value (so "2.00 Lakh" -> "2,00,000").
            for variant in format_variants(value, is_pct):
                items.append({**base, "kind": "faithful_format",
                              "claim": with_number(variant, drop_unit=True)})

            # Perturb at the level of the written token and keep its unit, so
            # "2.00 Lakh" becomes e.g. "4 Lakh"; the FULL value is what must be
            # absent from the evidence.
            from core.guardrail import _canonical_number

            token_value = _canonical_number(match.group("num"))
            multiplier = (value / token_value) if token_value else Decimal(1)
            wrong_token = wrong_value(
                token_value or value,
                {(k / multiplier).normalize() for k in known}, rng)
            if wrong_token is not None:
                items.append({**base, "kind": "perturbed_out",
                              "wrong_value": render((wrong_token * multiplier).normalize()),
                              "claim": with_number(render(wrong_token))})

            in_evidence = sorted(v for v in known if v != value and v >= 10
                                 and v == v.to_integral_value() and not (1900 <= v <= 2100))
            if in_evidence:
                swap = rng.choice(in_evidence)
                items.append({**base, "kind": "perturbed_in", "wrong_value": render(swap),
                              "claim": with_number(render(swap), drop_unit=True)})
    return items


# ==========================================================================
# Scoring
# ==========================================================================


def score_items(items: list[dict], retriever) -> None:
    """Adds `rule_flags` (list) and `similarity` (float) to every item, in place."""
    import numpy as np

    from core.guardrail import numeric_and_link_grounding

    stored = retriever.collection.get(include=["documents", "metadatas"])
    by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))

    from app.schemas import ChunkMetadata, RetrievedChunk

    def chunk(cid: str) -> RetrievedChunk:
        text, meta = by_id[cid]
        return RetrievedChunk(chunk_id=cid, text=text, score=1.0,
                              metadata=ChunkMetadata(**{k: meta.get(k, "") for k in
                                                        ChunkMetadata.model_fields}))

    for item in items:
        item["rule_flags"] = numeric_and_link_grounding(
            item["claim"], [chunk(c) for c in item["evidence_ids"]])

    claims = [item["claim"] for item in items]
    sources = [by_id[item["source_chunk_id"]][0] for item in items]
    vectors = np.asarray(retriever.embed(claims + sources), dtype=float)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    n = len(items)
    for i, item in enumerate(items):
        item["similarity"] = round(float(vectors[i] @ vectors[n + i]), 4)


def build_report(items: list[dict], meta: dict) -> str:
    kinds = ("faithful_original", "faithful_format", "perturbed_out", "perturbed_in")
    group = {k: [i for i in items if i["kind"] == k] for k in kinds}
    faithful = group["faithful_original"] + group["faithful_format"]

    # Similarity threshold: flag when cosine < t; pick the highest t that keeps
    # false alarms on faithful items <= 5%.
    sims_faithful = sorted(i["similarity"] for i in faithful)
    t = sims_faithful[max(0, int(0.05 * len(sims_faithful)) - 1)] if sims_faithful else 0.0

    out = ["# Number-perturbation test (E9)\n",
           f"- **Date:** {meta['date']}  ·  **seed:** {meta['seed']}  ·  "
           f"**evidence:** top-{meta['top_k']} retrieved passages for the claim, source "
           f"passage always included",
           f"- **Items:** {len(items)} from {meta['n_sentences']} numeric sentences in "
           f"{meta['n_chunks']} indexed passages",
           f"- **Embeddings:** `{meta['embedding_model']}`",
           "- **LLM judge:** not run (no API key in this run). The items are saved to "
           "`items.jsonl` so the judge and an NLI model can be scored on the same set.\n",
           "## Detection rates\n",
           "Flag rate per item type. For faithful items a flag is a **false alarm** "
           "(lower is better); for perturbed items it is a **detection** (higher is "
           f"better). Similarity flags when cosine(claim, source) < {t:.3f}, the "
           "threshold that keeps false alarms on faithful items at about 5%.\n",
           "| Item type | n | numeric rule flags | similarity flags |",
           "|---|---|---|---|"]
    for kind in kinds:
        rows = group[kind]
        rule = sum(bool(i["rule_flags"]) for i in rows)
        sim = sum(i["similarity"] < t for i in rows)
        out.append(f"| `{kind}` | {len(rows)} | {fmt_interval(rule, len(rows))} | "
                   f"{fmt_interval(sim, len(rows))} |")
    out.append("")

    out.append("## Can similarity tell a wrong number from a right one?\n")
    out.append("AUROC of cosine(claim, source): faithful (positive) vs perturbed. "
               "0.5 = no signal.\n")
    out.append("| Faithful vs | AUROC |")
    out.append("|---|---|")
    for kind in ("perturbed_out", "perturbed_in"):
        value = auroc([i["similarity"] for i in faithful],
                      [i["similarity"] for i in group[kind]])
        out.append(f"| `{kind}` | {value if value is not None else 'n/a'} |")
    out.append("")

    out.append("## Examples\n")
    for kind in ("faithful_format", "perturbed_out", "perturbed_in"):
        for item in group[kind][:2]:
            change = f"{item['value']} → {item.get('wrong_value', 'same value, reformatted')}"
            out.append(f"- `{kind}` ({change}) — rule flags: {item['rule_flags'] or 'none'}; "
                       f"similarity {item['similarity']:.3f}  \n  *{item['claim'][:200]}*")
    out.append("")

    misses = [i for i in faithful if i["rule_flags"]]
    out.append(f"## False alarms of the numeric rule ({len(misses)})\n")
    for item in misses[:15]:
        out.append(f"- flags {item['rule_flags']} — *{item['claim'][:200]}*")
    out.append("")
    out.append("## Reading this\n"
               "- `perturbed_in` is constructed to defeat an exact-match check, so a low "
               "rule detection rate there is expected; it measures how often the blind spot "
               "*can* occur, not how often a model *does* make that error.\n"
               "- Claims are source sentences with one number changed, not LLM-written "
               "claims; real claims are paraphrased, which matters for similarity.\n"
               "- Regenerate: `python eval/run_numeric_perturbation.py`.\n")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="E9 number-perturbation test (0 LLM calls).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    from app.config import Settings
    from core.retriever import Retriever

    settings = Settings()
    retriever = Retriever(settings)
    items = build_items(retriever, args.top_k, args.seed)
    score_items(items, retriever)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": date.today().isoformat(), "seed": args.seed, "top_k": args.top_k,
        "embedding_model": settings.embedding_model, "n_chunks": retriever.count(),
        "n_sentences": sum(i["kind"] == "faithful_original" for i in items),
    }
    with io.open(out_dir / "items.jsonl", "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out_dir / "results.md").write_text(build_report(items, meta), encoding="utf-8")
    print(f"Wrote {out_dir / 'items.jsonl'} ({len(items)} items) and {out_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
