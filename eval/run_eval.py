"""
Run the golden set with the guardrail OFF and ON, and compare.

THE QUESTION THIS ANSWERS
"Does the verification layer actually buy anything, and what does it cost?"
Without a measurement, a hallucination guardrail is just a claim. So every
question is run twice -- once with verification disabled (plain RAG, the
baseline) and once with it enabled -- on the SAME inputs.

HOW THE TWO CONDITIONS ARE KEPT COMPARABLE
  * Each question is normalized ONCE (eval/norm_cache.py) and the same
    NormalizedQuery is handed to both conditions, so they retrieve identically.
    Retrieval is therefore not a difference between OFF and ON by construction.
  * The conditions are interleaved question by question (OFF, ON, OFF, ON ...),
    so both see the same rate-limit state; running one condition entirely after
    the other would make the second one look slower.

THE HEADLINE METRIC: UNSUPPORTED CLAIMS IN DELIVERED ANSWERS
Measuring the *draft* would flatter the guardrail. What matters is what reaches
the user, so the delivered answer is audited claim by claim against the same
retrieved evidence. Verdicts the production pipeline already produced for the
delivered text are reused (a PASS draft, or the REPAIR re-check) rather than
paying twice for the same measurement; otherwise a separate audit runs.
Caveat, stated in the report: reused verdicts come from the same judge, so ON's
number is not independent evidence. `--audit-all` re-audits those answers too.

WHAT COUNTS AS A REFUSAL (one detector for both conditions)
An answer counts as refused when ANY of these hold:
  * the pipeline said status == refused, or
  * the text is one of the canned refusal / blocked / service messages, or
  * the delivered answer contains zero checkable claims (the model declined in
    its own words, e.g. "Mujhe iske baare mein jaankari nahi hai").
Using one detector stops the guardrail-ON condition from looking better only
because it re-labels the same non-answer as a refusal.

INFRASTRUCTURE FAILURES ARE NOT REFUSALS
A rate limit, quota or outage makes the backend fail closed (refuse). That is
the right behaviour for users but it is not a measurement of the guardrail, so
such questions are recorded as `infra_failure`, EXCLUDED from every rate and
reported as a count. After 3 consecutive infrastructure failures the run stops
(the budget is gone; carrying on would only produce noise) and can be continued
with --resume.

Run:
    python eval/run_eval.py                       # full run, both conditions
    python eval/run_eval.py --stratified 20       # preliminary run -> eval/runs/<name>/
    python eval/run_eval.py --ids a01,u05         # specific questions
    python eval/run_eval.py --condition on        # one condition only
    python eval/run_eval.py --resume              # continue an interrupted run
    python eval/run_eval.py --no-audit            # skip the claim-level audit
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import AnswerStatus, GuardrailDecision, Verdict  # noqa: E402
from eval.common import (  # noqa: E402
    GOLDEN_PATH,
    add_selection_args,
    auroc,
    fact_present,
    fmt_pct,
    golden_sha1,
    load_rows,
    median,
    meta_compatible,
    normalize,
    pct,
    percentile_nearest_rank,
    rows_from_args,
    selection_is_subset,
    subset_dir_name,
)

DEFAULT_OUT_DIR = ROOT / "eval"
MAX_CONSECUTIVE_INFRA_FAILURES = 3
TOP_N = 10  # depth of the retrieval list saved per question

__all__ = ["fact_present", "summarise", "build_report", "detect_refusal"]


# ==========================================================================
# Token metering (works with any backend version)
# ==========================================================================


class InfraBreaker:
    """
    Circuit breaker: abort the run after N consecutive infrastructure failures.

    One counter PER CONDITION. The questions run interleaved (OFF, ON, OFF, ON),
    so a single shared counter would be reset by every OFF success and would never
    trip when only ON is failing (ON's larger judge prompts hit the per-minute
    token limit first).
    """

    def __init__(self, conditions, limit: int = MAX_CONSECUTIVE_INFRA_FAILURES):
        self.limit = limit
        self.consecutive = {condition: 0 for condition in conditions}

    def record(self, condition: str, failed: bool) -> bool:
        """Note one result; return True when this condition has reached the limit."""
        self.consecutive[condition] = self.consecutive[condition] + 1 if failed else 0
        return self.consecutive[condition] >= self.limit


class MeteredLLM:
    """
    Thin wrapper that adds up prompt/completion tokens of every LLM call made
    through it. Everything else is forwarded to the real client untouched.

    Why a wrapper instead of reading the call log: it needs no database, and it
    still works if the backend's own per-request counter changes shape.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _add(self, result) -> None:
        self.prompt_tokens += int(getattr(result, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(result, "completion_tokens", 0) or 0)

    def complete(self, *args, **kwargs):
        result = self._inner.complete(*args, **kwargs)
        self._add(result)
        return result

    def complete_json(self, *args, **kwargs):
        # The inner complete_json calls the INNER complete, so no double count.
        parsed, result = self._inner.complete_json(*args, **kwargs)
        self._add(result)
        return parsed, result

    def snapshot(self) -> tuple[int, int]:
        return self.prompt_tokens, self.completion_tokens


# ==========================================================================
# One shared refusal detector
# ==========================================================================


def _canned_refusals() -> set[str]:
    """Every hand-written refusal sentence the backend can emit (normalized)."""
    from app import prompts

    texts: set[str] = set()
    for name in ("REFUSAL_MESSAGES", "BLOCKED_MESSAGES", "SERVICE_UNAVAILABLE_MESSAGES"):
        for message in (getattr(prompts, name, None) or {}).values():
            texts.add(normalize(message))
    return texts


def is_canned_refusal(answer: str, canned: set[str] | None = None) -> bool:
    """True if `answer` is (or contains) one of the backend's canned refusals."""
    canned = canned if canned is not None else _canned_refusals()
    text = normalize(answer or "")
    if not text:
        return True  # an empty answer is not an answer
    return any(message in text for message in canned)


def _enum_value(value) -> str | None:
    """Enum or plain string -> plain string (None stays None)."""
    if value is None:
        return None
    return str(getattr(value, "value", value))


def detect_refusal(
    status: str | None,
    answer: str,
    canned: set[str] | None = None,
    audited_claim_count: int | None = None,
) -> tuple[bool, str]:
    """
    The single refusal detector used for BOTH conditions.

    Returns (refused, source) where source is "pipeline", "canned_text",
    "no_claims" or "" (not refused).
    `audited_claim_count` is the number of claims the audit found in the delivered
    answer; None means "no audit ran", in which case that test is skipped.
    """
    if status == AnswerStatus.REFUSED.value:
        return True, "pipeline"
    if is_canned_refusal(answer, canned):
        return True, "canned_text"
    if audited_claim_count is not None and audited_claim_count == 0:
        return True, "no_claims"
    return False, ""


# ==========================================================================
# One question, one condition
# ==========================================================================


def _claim_sims(claims) -> list[list]:
    """[verdict, evidence_similarity, best_similarity] per judged claim."""
    rows = []
    for claim in claims or []:
        rows.append([
            _enum_value(claim.verdict),
            getattr(claim, "evidence_similarity", None),
            getattr(claim, "best_similarity", None),
        ])
    return rows


def base_record(row: dict, condition: str) -> dict:
    """Fields every record has, so summarise() never meets a missing key."""
    return {
        "id": row["id"],
        "condition": condition,
        "question": row["question"],
        "language": row["language"],
        "answerable": row["answerable"],
        "category": row["category"],
        "expected_scheme": row.get("expected_scheme"),
        "infra_failure": False,
        "infra_reason": "",
        "error": "",
        "status": "",
        "refusal_reason": None,
        "refused": None,
        # True when the answer was not refused by the pipeline or by canned text AND
        # no claim audit ran/succeeded, so "model declined in its own words"
        # (zero checkable claims) could not be tested. Such a record is left out of
        # the refusal rates in BOTH conditions (see pair_records).
        "refusal_undetermined": False,
        "refusal_source": "",
        "guardrail_decision": "",
        "answer": "",
        "normalized_query": "",
        "detected_language": "",
        "normalization_fallback": False,
        "normalization_from_cache": None,
        "best_retrieval_score": 0.0,
        "retrieval_ranked": [],
        "retrieval_hit": None,
        "retrieval_hit_top1": None,
        "citations": 0,
        "latency_ms": 0,
        "llm_calls": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "audit_calls": 0,
        "audit_tokens": None,
        "draft_claims": 0,
        "draft_unsupported": 0,
        "fabricated_citations": 0,
        "key_facts_total": len(row.get("key_facts") or []),
        "key_facts_found": 0,
        "key_facts_missing": list(row.get("key_facts") or []),
        "delivered_claims": None,
        "delivered_unsupported": None,
        "audit_source": None,
        "audit_error": "",
        "claim_sims": [],
    }


def run_one(
    row: dict,
    condition: str,
    pipeline,
    retriever,
    auditor,
    norm,
    meter: MeteredLLM,
    top_k: int,
    audit_all: bool = False,
    canned: set[str] | None = None,
) -> dict:
    """Run a single question under one condition and collect everything measurable."""
    record = base_record(row, condition)
    record["normalized_query"] = norm.normalized.normalized_query
    record["detected_language"] = norm.normalized.language.value
    record["normalization_from_cache"] = norm.from_cache
    record["normalization_fallback"] = bool(norm.normalized.fallback_used)

    # A normalizer fallback means the LLM was unavailable for the very first
    # step. The pipeline would run with degraded retrieval -- that is an
    # infrastructure problem, not a result, so do not spend more calls on it.
    if norm.normalized.fallback_used:
        record["infra_failure"] = True
        record["infra_reason"] = "normalizer_fallback"
        return record

    started = time.perf_counter()
    tokens_before = meter.snapshot()
    injected = True
    try:
        try:
            response = pipeline.ask(row["question"], normalized=norm.normalized)
        except TypeError as exc:
            # Backend without the `normalized=` parameter: run unpaired, and say so.
            if "normalized" not in str(exc):
                raise
            injected = False
            response = pipeline.ask(row["question"])
    except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
        record["infra_failure"] = True
        record["infra_reason"] = "exception"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return record
    tokens_after = meter.snapshot()

    status = _enum_value(response.status)
    reason = _enum_value(getattr(response, "refusal_reason", None))
    guardrail = response.guardrail
    verification_error = getattr(guardrail, "verification_error", None)

    record["status"] = status
    record["refusal_reason"] = reason
    record["guardrail_decision"] = _enum_value(guardrail.decision)
    record["answer"] = response.answer
    record["citations"] = len(response.citations)
    record["draft_claims"] = len(guardrail.claims)
    record["draft_unsupported"] = guardrail.unsupported_count
    record["fabricated_citations"] = sum(
        1 for claim in guardrail.claims if not claim.citation_valid
    )

    # Production cost = what the backend reported + the normalizer call that this
    # harness made on its behalf (so numbers match what a real request costs).
    record["llm_calls"] = response.llm_calls + (1 if injected else 0)
    record["latency_ms"] = response.latency_ms + (norm.latency_ms if injected else 0)
    record["prompt_tokens"] = (tokens_after[0] - tokens_before[0]) + (
        norm.prompt_tokens if injected else 0
    )
    record["completion_tokens"] = (tokens_after[1] - tokens_before[1]) + (
        norm.completion_tokens if injected else 0
    )

    # ---- infrastructure failure: excluded from every rate -----------------
    if reason == "service_unavailable" or verification_error:
        record["infra_failure"] = True
        record["infra_reason"] = reason or "verification_error"
        return record

    # ---- retrieval, measured independently of any refusal ----------------
    # Deterministic and free: search again with the same normalized query and
    # keep the whole top-10 so Recall@k / MRR can be recomputed later.
    retrieval = retriever.search(
        norm.normalized.normalized_query, top_k=max(TOP_N, top_k), min_score=0.0
    )
    ranked = [
        {"chunk_id": c.chunk_id, "scheme_id": c.metadata.scheme_id,
         "section": c.metadata.section, "score": c.score}
        for c in retrieval.chunks
    ]
    record["best_retrieval_score"] = retrieval.best_score
    record["retrieval_ranked"] = ranked
    expected = row.get("expected_scheme")
    top_schemes = [item["scheme_id"] for item in ranked[:top_k]]
    if expected:
        record["retrieval_hit"] = expected in top_schemes
        record["retrieval_hit_top1"] = bool(top_schemes) and top_schemes[0] == expected
    audit_chunks = retrieval.chunks[:top_k]  # the evidence production actually saw

    # ---- refusal, first pass (pipeline status or canned text) -------------
    refused, source = detect_refusal(status, response.answer, canned)
    record["refused"], record["refusal_source"] = refused, source

    # ---- audit the DELIVERED answer ----------------------------------------
    if not refused:
        delivered = None
        repair_report = getattr(response, "repair_report", None)

        if (not audit_all and guardrail.decision == GuardrailDecision.PASS
                and guardrail.claims):
            # PASS: the answer shipped unchanged, so the draft verdicts ARE the
            # verdicts on the delivered text.
            delivered, record["audit_source"] = guardrail.claims, "guardrail_pass"
        elif not audit_all and repair_report is not None and repair_report.claims:
            # REPAIR: the backend already re-checked the repaired text.
            delivered, record["audit_source"] = repair_report.claims, "repair_report"
        elif auditor is not None and audit_chunks:
            before = meter.snapshot()
            audit_counter = _new_counter()
            try:
                audit = auditor.verify(response.answer, audit_chunks, counter=audit_counter)
                failed = getattr(audit, "verification_error", None) or (
                    not audit.claims and audit.decision == GuardrailDecision.BLOCK
                )
                if failed:
                    record["audit_error"] = str(
                        getattr(audit, "verification_error", None) or "audit failed"
                    )
                else:
                    delivered, record["audit_source"] = audit.claims, "audit"
            except Exception as exc:  # noqa: BLE001 - audit trouble is data, not a crash
                record["audit_error"] = f"{type(exc).__name__}: {exc}"
            after = meter.snapshot()
            record["audit_calls"] = getattr(audit_counter, "count", 0)
            record["audit_tokens"] = (after[0] - before[0]) + (after[1] - before[1])

        if delivered is not None:
            record["delivered_claims"] = len(delivered)
            record["delivered_unsupported"] = sum(
                1 for claim in delivered if claim.verdict != Verdict.SUPPORTED
            )
            record["claim_sims"] = _claim_sims(delivered)
            # Zero checkable claims => the model declined in its own words.
            refused, source = detect_refusal(status, response.answer, canned,
                                             audited_claim_count=len(delivered))
            record["refused"], record["refusal_source"] = refused, source

    # No audit means the zero-claims test above never ran for this answer. ON's
    # pipeline applies that test itself (NO_CLAIMS), so without this flag only ON
    # would count such a decline as a refusal. Say so instead of guessing.
    if not record["refused"] and record["delivered_claims"] is None:
        record["refusal_undetermined"] = True

    # ---- key-fact coverage (only for an actual answer) ---------------------
    key_facts = row.get("key_facts") or []
    if key_facts and not record["refused"]:
        found = [fact for fact in key_facts if fact_present(fact, response.answer)]
        record["key_facts_found"] = len(found)
        record["key_facts_missing"] = [f for f in key_facts if f not in found]

    return record


def _new_counter():
    """A fresh CallCounter (imported lazily to keep module import light)."""
    from app.llm_client import CallCounter

    return CallCounter()


# ==========================================================================
# Aggregation
# ==========================================================================


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def pair_records(
    records_by_condition: dict[str, list[dict]],
) -> tuple[set[str], set[str]]:
    """
    Decide which question ids must be left out so OFF and ON are scored on the
    SAME questions.

    Returns (drop_ids, refusal_skip_ids):
      * drop_ids: ids that were an infrastructure failure / error in ANY
        condition. They are removed from EVERY condition. Otherwise an ON run
        that hits more rate limits (it makes ~2x the calls) would silently be
        scored on easier questions than OFF, and the table would compare
        different question sets.
      * refusal_skip_ids: ids whose refusal status could not be determined in
        any condition; they are left out of the refusal rates only.
    """
    drop_ids: set[str] = set()
    refusal_skip_ids: set[str] = set()
    for records in records_by_condition.values():
        for record in records:
            if record.get("infra_failure") or record.get("error"):
                drop_ids.add(record["id"])
            elif record.get("refusal_undetermined"):
                refusal_skip_ids.add(record["id"])
    return drop_ids, refusal_skip_ids - drop_ids


def summarise(
    records: list[dict],
    drop_ids: set[str] | frozenset[str] = frozenset(),
    refusal_skip_ids: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """
    Turn per-question records into the headline numbers.

    `drop_ids` / `refusal_skip_ids` come from pair_records(): questions that
    failed (or could not be classified) in the OTHER condition are excluded here
    too, so both conditions are scored on the same questions.

    Rules that keep the numbers honest:
      * infrastructure failures and errors are EXCLUDED from every rate and
        reported as a count;
      * a rate with an empty denominator is None (shown as "n/a (n=0)"), never 0;
      * accuracy is reported two ways: coverage over answered questions, and
        end-to-end over ALL answerable questions (a refusal scores zero facts).
    """
    infra = [r for r in records if r.get("infra_failure") or r.get("error")]
    ok = [r for r in records
          if not (r.get("infra_failure") or r.get("error")) and r["id"] not in drop_ids]
    paired_out = [r for r in records
                  if not (r.get("infra_failure") or r.get("error")) and r["id"] in drop_ids]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]

    # Refusal rates use only records whose refusal status is known.
    def _refusal_known(record: dict) -> bool:
        return not (record.get("refusal_undetermined") or record["id"] in refusal_skip_ids)

    ref_answerable = [r for r in answerable if _refusal_known(r)]
    ref_unanswerable = [r for r in unanswerable if _refusal_known(r)]

    # Retrieval
    with_expected = [r for r in answerable if r.get("retrieval_hit") is not None]
    retrieval_hits = sum(1 for r in with_expected if r["retrieval_hit"])
    top1_hits = sum(1 for r in with_expected if r.get("retrieval_hit_top1"))

    # Refusals
    answerable_refused = sum(1 for r in ref_answerable if r["refused"])
    unanswerable_refused = sum(1 for r in ref_unanswerable if r["refused"])
    answered = [r for r in ref_answerable if not r["refused"]]
    answered_all = [r for r in answerable if not r["refused"]]  # for key-fact coverage

    # Key facts: coverage over answered, end-to-end over all answerable.
    facts_total_answered = sum(r["key_facts_total"] for r in answered_all)
    facts_found_answered = sum(r["key_facts_found"] for r in answered_all)
    facts_total_all = sum(r["key_facts_total"] for r in answerable)
    facts_found_all = sum(r["key_facts_found"] for r in answerable)  # refusals are 0
    with_facts = [r for r in answerable if r["key_facts_total"]]
    fully_correct = sum(
        1 for r in with_facts
        if not r["refused"] and r["key_facts_found"] == r["key_facts_total"]
    )

    # Unsupported claims in DELIVERED answers (only records that were audited)
    audited = [r for r in ok if r.get("delivered_claims")]
    delivered_claims = sum(r["delivered_claims"] for r in audited)
    delivered_unsupported = sum(r["delivered_unsupported"] for r in audited)
    answers_with_unsupported = sum(1 for r in audited if r["delivered_unsupported"] > 0)
    audit_failures = sum(1 for r in ok if r.get("audit_error"))
    unaudited_answers = sum(
        1 for r in ok if not r["refused"] and r.get("delivered_claims") is None
    )

    # Cost
    latencies = [r["latency_ms"] for r in ok]
    calls = [r["llm_calls"] for r in ok]
    prompt_tokens = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None]
    completion_tokens = [
        r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None
    ]
    token_totals = [
        (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
        for r in ok if r.get("prompt_tokens") is not None
    ]

    decisions: dict[str, int] = {}
    for record in ok:
        decision = record.get("guardrail_decision") or "none"
        decisions[decision] = decisions.get(decision, 0) + 1

    # Refusals split by WHY the question is unanswerable.
    by_category: dict[str, dict] = {}
    for record in ref_unanswerable:
        bucket = by_category.setdefault(record["category"], {"n": 0, "refused": 0})
        bucket["n"] += 1
        bucket["refused"] += 1 if record["refused"] else 0
    for bucket in by_category.values():
        bucket["rate"] = pct(bucket["refused"], bucket["n"])

    by_source: dict[str, int] = {}
    for record in ok:
        if record["refused"]:
            key = record.get("refusal_source") or "unknown"
            by_source[key] = by_source.get(key, 0) + 1

    # Per-language view of answerable questions (n is small: read with care).
    by_language: dict[str, dict] = {}
    for record in answerable:
        bucket = by_language.setdefault(
            record["language"], {"n": 0, "refused": 0, "hits": 0, "hit_n": 0}
        )
        bucket["n"] += 1
        bucket["refused"] += 1 if record["refused"] else 0
        if record.get("retrieval_hit") is not None:
            bucket["hit_n"] += 1
            bucket["hits"] += 1 if record["retrieval_hit"] else 0

    # EXPLORATORY: does claim-evidence similarity separate the judge's verdicts?
    supported_sims, unsupported_sims = [], []
    for record in ok:
        for verdict, evidence_similarity, _best in record.get("claim_sims", []):
            if evidence_similarity is None:
                continue
            if verdict == Verdict.SUPPORTED.value:
                supported_sims.append(evidence_similarity)
            elif verdict == Verdict.NOT_SUPPORTED.value:
                unsupported_sims.append(evidence_similarity)
    similarity = {
        "n_supported": len(supported_sims),
        "n_not_supported": len(unsupported_sims),
        "mean_supported": _mean(supported_sims),
        "mean_not_supported": _mean(unsupported_sims),
        "auroc": auroc(supported_sims, unsupported_sims),
    }

    return {
        "questions": len(records),
        "infra_failures": len(infra),
        "infra_failure_ids": [r["id"] for r in infra],
        # Questions dropped only because the OTHER condition failed on them.
        "paired_excluded": len(paired_out),
        "paired_excluded_ids": [r["id"] for r in paired_out],
        # The exact question set these rates are computed on (compared across
        # conditions before anything is published).
        "scored_ids": sorted(r["id"] for r in ok),
        "refusal_undetermined": len(answerable) + len(unanswerable)
                                - len(ref_answerable) - len(ref_unanswerable),
        "refusal_answerable_n": len(ref_answerable),
        "refusal_unanswerable_n": len(ref_unanswerable),
        "scored": len(ok),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),

        "retrieval_hit_rate": pct(retrieval_hits, len(with_expected)),
        "retrieval_hit_rate_top1": pct(top1_hits, len(with_expected)),
        "retrieval_hits": retrieval_hits,
        "retrieval_top1_hits": top1_hits,
        "retrieval_n": len(with_expected),

        "correct_refusal_rate": pct(unanswerable_refused, len(ref_unanswerable)),
        "correct_refusals": unanswerable_refused,
        "over_blocking_rate": pct(answerable_refused, len(ref_answerable)),
        "over_blocked": answerable_refused,
        "answered_rate": pct(len(answered), len(ref_answerable)),
        "answered": len(answered),

        "key_fact_coverage": pct(facts_found_answered, facts_total_answered),
        "key_facts_found": facts_found_answered,
        "key_facts_total": facts_total_answered,
        "end_to_end_fact_accuracy": pct(facts_found_all, facts_total_all),
        "end_to_end_facts_found": facts_found_all,
        "end_to_end_facts_total": facts_total_all,
        "fully_correct_rate": pct(fully_correct, len(with_facts)),
        "fully_correct": fully_correct,
        "fully_correct_n": len(with_facts),

        "unsupported_claim_rate": pct(delivered_unsupported, delivered_claims),
        "delivered_unsupported": delivered_unsupported,
        "delivered_claims": delivered_claims,
        "answers_with_unsupported_claim_rate": pct(answers_with_unsupported, len(audited)),
        "answers_with_unsupported_claim": answers_with_unsupported,
        "answers_audited": len(audited),
        "audit_failures": audit_failures,
        "answers_not_audited": unaudited_answers,
        "audit_sources": _count_by(ok, "audit_source"),

        "fabricated_citations": sum(r.get("fabricated_citations", 0) for r in ok),

        "avg_latency_ms": round(statistics.mean(latencies)) if latencies else None,
        "median_latency_ms": round(median(latencies)) if latencies else None,
        "p95_latency_ms": (
            round(percentile_nearest_rank(latencies, 95)) if latencies else None
        ),
        "avg_llm_calls": round(statistics.mean(calls), 2) if calls else None,
        "total_llm_calls": sum(calls),
        "total_audit_calls": sum(r.get("audit_calls", 0) for r in ok),
        "avg_tokens_per_question": round(statistics.mean(token_totals)) if token_totals else None,
        "total_prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
        "total_completion_tokens": sum(completion_tokens) if completion_tokens else None,
        "total_audit_tokens": (
            sum(r["audit_tokens"] for r in ok if r.get("audit_tokens") is not None) or None
        ),

        "guardrail_decisions": decisions,
        "refusal_by_category": by_category,
        "refusal_by_source": by_source,
        "by_language": by_language,
        "similarity": similarity,
    }


def _count_by(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = record.get(key)
        if value:
            counts[value] = counts.get(value, 0) + 1
    return counts


# ==========================================================================
# Report
# ==========================================================================


def _p(summary: dict, rate_key: str, num_key: str, den_key: str) -> str:
    """'62.9% (22/35)' or 'n/a (n=0)'."""
    rate, num, den = summary[rate_key], summary[num_key], summary[den_key]
    if rate is None:
        return fmt_pct(None, den)
    return f"{rate}% ({num}/{den})"


def _num(value, unit: str = "", digits: int | None = None) -> str:
    if value is None:
        return "n/a"
    if digits is not None:
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _secs(ms) -> str:
    return "n/a" if ms is None else f"{ms / 1000:.1f}s"


def _delta_pct(off: dict, on: dict, key: str, higher_is_better: bool) -> str:
    """Difference of two percentages, in PERCENTAGE POINTS (pp), not relative %."""
    before, after = off[key], on[key]
    if before is None or after is None:
        return "n/a"
    difference = round(after - before, 1)
    if abs(difference) < 0.05:
        return "no change"
    improved = (difference > 0) == higher_is_better
    return f"{'better' if improved else 'worse'} ({difference:+} pp)"


def preliminary_label(meta: dict) -> str:
    """Empty for a full run; the loud label for any subset."""
    if meta.get("subset"):
        return (f"PRELIMINARY, n={meta['n_questions']} of {meta['n_golden']} "
                "golden-set questions (a subset chosen for language/category "
                "coverage, not proportional to the full set)")
    return ""


def build_report(
    summaries: dict[str, dict],
    meta: dict,
    condition_meta: dict | None = None,
) -> str:
    """Write results.md: the comparison table plus how to read it."""
    off = summaries.get("off")
    on = summaries.get("on")
    label = preliminary_label(meta)
    mismatch = []
    if off and on and condition_meta and "off" in condition_meta and "on" in condition_meta:
        mismatch = meta_compatible(condition_meta["off"], condition_meta["on"])
    if off and on and off.get("scored_ids") != on.get("scored_ids"):
        # Rates over different question sets are not a comparison.
        mismatch = [*mismatch, "scored question set"]
    comparable = bool(off and on and not mismatch)

    lines: list[str] = ["# Evaluation results\n"]
    if label:
        lines.append(f"> **{label}.** Do not quote these figures as the benchmark; "
                     "with this few questions a single answer moves a rate by several points.\n")
    lines.append(
        f"The same {meta['n_questions']} golden-set questions run with the verification "
        "layer **off** (plain RAG, the baseline) and **on**. Each question is "
        "normalized once and both conditions retrieve with that identical query; the "
        "only difference is whether answers are checked before being returned.\n"
    )
    if mismatch:
        lines.append(f"> **The two conditions are NOT comparable** (they differ in: "
                     f"{', '.join(mismatch)}). No comparison table is produced.\n")

    lines.append("## Setup\n")
    lines.append(f"- **Date:** {meta['date']}")
    lines.append(f"- **Questions:** n={meta['n_questions']} "
                 f"({meta['answerable']} answerable, {meta['unanswerable']} unanswerable)"
                 f" of {meta['n_golden']} in the golden set")
    lines.append(f"- **LLM:** `{meta['model']}` via {meta['base_url']}")
    lines.append(f"- **Embeddings:** `{meta['embedding_model']}` (cosine)")
    lines.append(f"- **Retrieval:** top-{meta['top_k']}, refusal threshold {meta['min_score']}")
    lines.append(f"- **Block threshold:** {meta['block_threshold']} "
                 "(unsupported-claim fraction above which an answer is refused)")
    lines.append(f"- **Corpus:** {meta['chunks']} chunks from {meta['schemes']} central schemes")
    lines.append(f"- **Claim audit:** {'on' if meta['audit'] else 'off'}"
                 f"{' (re-audits PASS/REPAIR answers too)' if meta.get('audit_all') else ''}\n")

    if comparable:
        lines.append("## Headline comparison\n")
        lines.append("| Metric | Guardrail OFF | Guardrail ON | Change |")
        lines.append("|---|---|---|---|")
        rows = [
            ("**Unsupported claim rate** (delivered answers)", "unsupported_claim_rate",
             "delivered_unsupported", "delivered_claims", False),
            ("Answers containing any unsupported claim", "answers_with_unsupported_claim_rate",
             "answers_with_unsupported_claim", "answers_audited", False),
            ("Correct refusal rate (unanswerable)", "correct_refusal_rate",
             "correct_refusals", "refusal_unanswerable_n", True),
            ("Over-blocking rate (answerable refused)", "over_blocking_rate",
             "over_blocked", "refusal_answerable_n", False),
            ("Answered rate (of answerable)", "answered_rate", "answered",
             "refusal_answerable_n", True),
            ("Key-fact coverage (answered only)", "key_fact_coverage",
             "key_facts_found", "key_facts_total", True),
            ("**End-to-end fact accuracy** (all answerable; refusal = 0)",
             "end_to_end_fact_accuracy", "end_to_end_facts_found",
             "end_to_end_facts_total", True),
            ("Questions fully correct (all key facts)", "fully_correct_rate",
             "fully_correct", "fully_correct_n", True),
        ]
        for label_text, rate_key, num_key, den_key, higher in rows:
            lines.append(
                f"| {label_text} | {_p(off, rate_key, num_key, den_key)} | "
                f"{_p(on, rate_key, num_key, den_key)} | "
                f"{_delta_pct(off, on, rate_key, higher)} |"
            )
        if off["avg_llm_calls"] is not None and on["avg_llm_calls"] is not None:
            lines.append(
                f"| Avg LLM calls / question | {off['avg_llm_calls']} | {on['avg_llm_calls']} "
                f"| {on['avg_llm_calls'] - off['avg_llm_calls']:+.2f} |"
            )
        lines.append(
            f"| Avg tokens / question | {_num(off['avg_tokens_per_question'])} | "
            f"{_num(on['avg_tokens_per_question'])} | |"
        )
        lines.append(
            f"| Median latency | {_secs(off['median_latency_ms'])} | "
            f"{_secs(on['median_latency_ms'])} | |"
        )
        lines.append(
            f"| p95 latency | {_secs(off['p95_latency_ms'])} | {_secs(on['p95_latency_ms'])} | |"
        )
        lines.append(
            f"| Infrastructure failures (excluded) | {off['infra_failures']} | "
            f"{on['infra_failures']} | |"
        )
        lines.append(
            f"| Questions scored (paired: the same ones in both) | {off['scored']} | "
            f"{on['scored']} | |"
        )
        lines.append("")
        if off["paired_excluded"] or on["paired_excluded"]:
            lines.append(
                "*Paired scoring: a question that failed for infrastructure reasons in "
                "either condition is dropped from both, so neither is scored on easier "
                f"questions (dropped only because the other condition failed: OFF "
                f"{off['paired_excluded']}, ON {on['paired_excluded']}).*\n"
            )
        if off["refusal_undetermined"] or on["refusal_undetermined"]:
            lines.append(
                "*Refusal rates leave out answers whose refusal status could not be "
                "determined (no claim audit ran for them, so a decline in the model's "
                f"own words was undetectable): OFF {off['refusal_undetermined']}, "
                f"ON {on['refusal_undetermined']}.*\n"
            )
        lines.append(
            f"*Retrieval is identical in both conditions by construction (shared cached "
            f"normalization): expected scheme in top-{meta['top_k']} for "
            f"{_p(off, 'retrieval_hit_rate', 'retrieval_hits', 'retrieval_n')} of "
            "answerable questions.*\n"
        )

        lines.append("### How to read this\n")
        lines.append(
            "- **Unsupported claim rate** is the headline: of all factual claims in "
            "answers the user actually saw, how many the judge found unsupported by the "
            "retrieved official text. For ON, a PASS answer has zero unsupported claims "
            "*by definition of PASS* and its verdicts come from the same judge, so ON's "
            "figure is not independent evidence; use `--audit-all` for an independent "
            "re-audit.\n"
        )
        lines.append(
            "- **Correct refusal vs over-blocking** trade against each other: refusing "
            "more catches more unanswerable questions and blocks more good answers. "
            "Refusal here uses one detector for both conditions (pipeline status, canned "
            "refusal text, or an answer with no checkable claims); answers the detector "
            "could not classify are left out of the refusal rates in both.\n"
        )
        lines.append(
            "- **End-to-end fact accuracy** counts a refusal as zero facts, so it does "
            "not flatter a system that simply refuses more. Key-fact coverage (answered "
            "questions only) is shown beside it for that reason.\n"
        )
        lines.append(
            "- **Latency** includes time the harness spent waiting out free-tier rate "
            "limits, so read the median, not the mean.\n"
        )
    elif off or on:
        lines.append("## Results\n")
        lines.append("Only one condition has run, so no comparison is made yet.\n")

    for key, title in (("off", "Guardrail OFF (baseline)"), ("on", "Guardrail ON")):
        summary = summaries.get(key)
        if not summary:
            continue
        lines.append(f"## {title}\n")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Questions run | {summary['questions']} |")
        lines.append(f"| Infrastructure failures (excluded from all rates) | "
                     f"{summary['infra_failures']}"
                     f"{' (' + ', '.join(summary['infra_failure_ids']) + ')' if summary['infra_failure_ids'] else ''} |")
        lines.append(f"| Scored | {summary['scored']} "
                     f"({summary['answerable']} answerable, {summary['unanswerable']} not) |")
        lines.append(f"| Retrieval: expected scheme in top-{meta['top_k']} | "
                     f"{_p(summary, 'retrieval_hit_rate', 'retrieval_hits', 'retrieval_n')} |")
        lines.append(f"| Retrieval: top-1 | "
                     f"{_p(summary, 'retrieval_hit_rate_top1', 'retrieval_top1_hits', 'retrieval_n')} |")
        lines.append(f"| Answered (of answerable) | "
                     f"{_p(summary, 'answered_rate', 'answered', 'refusal_answerable_n')} |")
        lines.append(f"| Over-blocking rate | "
                     f"{_p(summary, 'over_blocking_rate', 'over_blocked', 'refusal_answerable_n')} |")
        lines.append(f"| Correct refusal rate | "
                     f"{_p(summary, 'correct_refusal_rate', 'correct_refusals', 'refusal_unanswerable_n')} |")
        lines.append(f"| Key-fact coverage (answered only) | "
                     f"{_p(summary, 'key_fact_coverage', 'key_facts_found', 'key_facts_total')} |")
        lines.append(f"| End-to-end fact accuracy (all answerable) | "
                     f"{_p(summary, 'end_to_end_fact_accuracy', 'end_to_end_facts_found', 'end_to_end_facts_total')} |")
        lines.append(f"| Unsupported claim rate (delivered) | "
                     f"{_p(summary, 'unsupported_claim_rate', 'delivered_unsupported', 'delivered_claims')} |")
        lines.append(f"| Answers with any unsupported claim | "
                     f"{_p(summary, 'answers_with_unsupported_claim_rate', 'answers_with_unsupported_claim', 'answers_audited')} |")
        lines.append(f"| Answers that could not be audited | {summary['answers_not_audited']} "
                     f"(audit failures: {summary['audit_failures']}) |")
        lines.append(f"| Fabricated citations caught in code (draft level) | "
                     f"{summary['fabricated_citations']} |")
        lines.append(f"| Avg / median / p95 latency | {_secs(summary['avg_latency_ms'])} / "
                     f"{_secs(summary['median_latency_ms'])} / {_secs(summary['p95_latency_ms'])} |")
        lines.append(f"| Avg LLM calls per question (production) | "
                     f"{_num(summary['avg_llm_calls'])} (normalization counts as 1) |")
        lines.append(f"| Avg tokens per question (production) | "
                     f"{_num(summary['avg_tokens_per_question'])} |")
        lines.append(f"| Extra calls used for measurement only | {summary['total_audit_calls']} |")
        lines.append("")

        lines.append("**Guardrail decisions:** " + ", ".join(
            f"{k}={v}" for k, v in sorted(summary["guardrail_decisions"].items())) + "\n")
        if summary["refusal_by_source"]:
            lines.append("**Refusals by how they were detected:** " + ", ".join(
                f"{k}={v}" for k, v in sorted(summary["refusal_by_source"].items())) + "\n")
        if summary["audit_sources"]:
            lines.append("**Where delivered-answer verdicts came from:** " + ", ".join(
                f"{k}={v}" for k, v in sorted(summary["audit_sources"].items())) + "\n")

        lines.append("**Refusal rate by type of unanswerable question:**\n")
        lines.append("| Type | Questions | Refused | Rate |")
        lines.append("|---|---|---|---|")
        for category, bucket in sorted(summary["refusal_by_category"].items()):
            lines.append(f"| `{category}` | {bucket['n']} | {bucket['refused']} | "
                         f"{fmt_pct(bucket['rate'], bucket['n'])} |")
        lines.append("")

        lines.append("**Answerable questions by language** (small n: read with care):\n")
        lines.append("| Language | n | Refused | Retrieval hit |")
        lines.append("|---|---|---|---|")
        for language, bucket in sorted(summary["by_language"].items()):
            lines.append(f"| {language} | {bucket['n']} | {bucket['refused']} | "
                         f"{bucket['hits']}/{bucket['hit_n']} |")
        lines.append("")

        sim = summary["similarity"]
        lines.append("**Claim-evidence similarity (EXPLORATORY, not used in any decision):**\n")
        if sim["n_supported"] and sim["n_not_supported"]:
            lines.append(
                f"- mean cosine to cited evidence: SUPPORTED {sim['mean_supported']} "
                f"(n={sim['n_supported']}) vs NOT_SUPPORTED {sim['mean_not_supported']} "
                f"(n={sim['n_not_supported']}); AUROC {sim['auroc']}.\n"
            )
        else:
            lines.append(
                f"- not enough data: SUPPORTED n={sim['n_supported']}, NOT_SUPPORTED "
                f"n={sim['n_not_supported']} claims carry a similarity value (needs both "
                "classes; the backend must record `evidence_similarity`).\n"
            )

    lines.append("## What the categories mean\n")
    lines.append(
        "- **`off_topic`**: not about schemes at all (\"capital of France\"). The "
        "retrieval threshold refuses these before any generation, so they cost only the "
        "**1 normalization call** (normalization always runs first) and no answer or "
        "verification calls.\n"
        "- **`fake_scheme`**: an invented but plausible scheme name. These score *above* "
        "the retrieval threshold because they look like scheme questions, so only "
        "claim-level verification can catch them.\n"
        "- **`not_indexed`**: a **real** scheme we do not cover. Retrieval returns "
        "confident, real, *wrong-scheme* evidence, which makes it the hardest case.\n"
        "- **`trick_unstated_fact`**: a real scheme, but the question asks for something "
        "the official source never states.\n"
    )

    lines.append("## Honest limitations of this evaluation\n")
    lines.append(
        f"- **n={meta['n_questions']} is small.** Differences of a few points are noise; "
        "the numbers show direction and rough size, not precision.\n"
        "- **The judge is the same model family as the generator**, and the audit uses "
        "the same judge. A shared blind spot would be invisible here. Stronger: a "
        "different judge model plus a human-labelled subset.\n"
        "- **Reused verdicts are not independent.** ON's unsupported-claim rate on PASS "
        "answers is zero by construction of PASS.\n"
        "- **Key facts are matched by substring.** A correct answer that words a number "
        "differently (\"2 lakh\" vs \"2.00 Lakh\") or answers in Hindi against an "
        "English key fact is scored as a miss, which biases against Hindi/Hinglish.\n"
        "- **Some gold labels are ambiguous** (for example the SVANidhi interest rate "
        "is stated two ways in the source). Per-question misses are worth reading "
        "in `raw_results.json` before drawing conclusions.\n"
        "- **Written by the same people who built the system**, so the set may "
        "under-represent failure modes we did not think of.\n"
    )
    lines.append("---\n")
    lines.append(
        "Regenerate with `python eval/run_eval.py`. Per-question records (answers, "
        "verdicts, top-10 retrieval) are in `raw_results.json`.\n"
    )
    return "\n".join(lines)


# ==========================================================================
# Checkpoints, merging, main
# ==========================================================================


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(io.open(path, encoding="utf-8").read())
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_resume_records(out_dir: Path, condition: str, cmeta: dict, wanted_ids: set[str]) -> list[dict]:
    """
    Records already finished for this condition, if their metadata matches.

    Source: the checkpoint if present, else a finished raw_results.json. Infra
    failures are NOT kept, so a resumed run retries them.
    """
    candidates = []
    checkpoint = _read_json(out_dir / f"_checkpoint_{condition}.json")
    if checkpoint:
        candidates.append((checkpoint.get("meta", {}), checkpoint.get("records", [])))
    raw = _read_json(out_dir / "raw_results.json")
    if raw and condition in raw.get("condition_meta", {}):
        candidates.append((raw["condition_meta"][condition],
                           raw.get("records", {}).get(condition, [])))
    for meta, records in candidates:
        differing = meta_compatible(meta, cmeta)
        if differing:
            print(f"  (not resuming '{condition}' from earlier data: differs in "
                  f"{', '.join(differing)})")
            continue
        kept = [r for r in records
                if not r.get("infra_failure") and not r.get("error") and r["id"] in wanted_ids]
        return kept
    return []


def build_condition_meta(condition: str, settings, rows: list[dict], all_rows: list[dict],
                         args) -> dict:
    """Everything that must match for two condition runs to be compared."""
    return {
        "condition": condition,
        "question_ids": [r["id"] for r in rows],
        "model": settings.llm_model,
        "top_k": settings.retrieval_top_k,
        "min_score": settings.retrieval_min_score,
        "block_threshold": settings.guardrail_block_threshold,
        "guardrail_enabled": condition == "on",
        "embedding_model": settings.embedding_model,
        "audit": not args.no_audit,
        "audit_all": bool(args.audit_all),
        "golden_sha1": golden_sha1(all_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SchemeSure.")
    add_selection_args(parser)
    parser.add_argument("--limit", type=int, default=0,
                        help="DEPRECATED alias for --stratified N")
    parser.add_argument("--condition", choices=["off", "on", "both"], default="both")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="seconds between questions (Groq rate limits)")
    parser.add_argument("--no-audit", action="store_true",
                        help="skip the claim-level audit of delivered answers")
    parser.add_argument("--audit-all", action="store_true",
                        help="re-audit PASS/REPAIR answers instead of reusing their verdicts")
    parser.add_argument("--out-dir", default="",
                        help="output folder (default eval/ for a full run, "
                             "eval/runs/<name>/ for any subset)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from checkpoints / earlier results in the out dir")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing a finished condition in the out dir")
    parser.add_argument("--questions", default="",
                        help="question file (default eval/golden_set.jsonl); "
                             "e.g. eval/paper_set.jsonl")
    args = parser.parse_args()
    if args.limit and not args.stratified:
        args.stratified = args.limit

    questions_path = Path(args.questions) if args.questions else None
    if not (questions_path or GOLDEN_PATH).exists():
        print(f"ERROR: {questions_path or GOLDEN_PATH} not found. "
              "Run scripts/build_golden_set.py (or build_paper_set.py) first.")
        return 1

    # Heavy imports are deferred so importing this module (e.g. in tests) stays fast.
    from app.config import Settings
    from app.llm_client import LLMClient
    from core.generator import AnswerPipeline
    from core.guardrail import Guardrail
    from core.retriever import Retriever
    from eval.norm_cache import NormalizationCache

    all_rows = load_rows(questions_path)
    rows = rows_from_args(args, all_rows)
    if not rows:
        print("ERROR: the selection matched no questions.")
        return 1
    subset = selection_is_subset(rows, all_rows)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif questions_path is not None:
        # Another question file never overwrites the golden-set results in eval/.
        out_dir = ROOT / "eval" / "runs" / f"e2e_{questions_path.stem}"
        if subset:
            out_dir = out_dir / subset_dir_name(rows)
    elif subset:
        out_dir = ROOT / "eval" / "runs" / subset_dir_name(rows)
    else:
        out_dir = DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Settings()
    base.require_llm_key()

    conditions = ["off", "on"] if args.condition == "both" else [args.condition]
    wanted_ids = {r["id"] for r in rows}

    # Never silently overwrite a finished run.
    existing = _read_json(out_dir / "raw_results.json") or {}
    finished = [c for c in conditions if c in existing.get("condition_meta", {})]
    if finished and not args.resume and not args.overwrite:
        print(f"ERROR: {out_dir / 'raw_results.json'} already holds a finished "
              f"'{', '.join(finished)}' run.\n  Use --resume to continue/extend it, "
              "--overwrite to replace it, or --out-dir to write somewhere else.")
        return 1

    print(f"Questions     : {len(rows)} of {len(all_rows)}"
          f"{'  (PRELIMINARY subset)' if subset else ''}")
    print(f"Output folder : {out_dir}")
    print(f"Model         : {base.llm_model}")
    print(f"Threshold     : {base.retrieval_min_score}")
    print(f"Conditions    : {args.condition}   (interleaved per question)")
    print(f"Claim audit   : {'off' if args.no_audit else ('all answers' if args.audit_all else 'on')}")

    retriever = Retriever(base)
    print(f"Indexed chunks: {retriever.count()}\n")

    # One shared normalization per question (cached on disk).
    normalizer_llm = LLMClient(base)
    cache = NormalizationCache(normalizer_llm, base.llm_model)

    canned = _canned_refusals()
    runners: dict[str, dict] = {}
    for condition in conditions:
        settings = Settings()
        settings.guardrail_enabled = condition == "on"
        meter = MeteredLLM(LLMClient(settings))
        pipeline = AnswerPipeline(retriever, meter, settings)

        auditor = None
        if not args.no_audit:
            audit_settings = Settings()
            audit_settings.guardrail_enabled = True
            try:
                auditor = Guardrail(meter, audit_settings, embed_fn=retriever.embed)
            except TypeError:  # backend without the similarity hook yet
                auditor = Guardrail(meter, audit_settings)

        cmeta = build_condition_meta(condition, settings, rows, all_rows, args)
        records = load_resume_records(out_dir, condition, cmeta, wanted_ids) if args.resume else []
        if records:
            print(f"Resuming '{condition}': {len(records)} questions already done.")
        runners[condition] = {
            "pipeline": pipeline, "auditor": auditor, "meter": meter,
            "cmeta": cmeta, "records": records,
            "done": {r["id"] for r in records},
        }

    def save_checkpoint(condition: str) -> None:
        runner = runners[condition]
        try:
            _write_json(out_dir / f"_checkpoint_{condition}.json",
                        {"meta": runner["cmeta"], "records": runner["records"]})
        except OSError:
            pass  # never let checkpointing break the run

    marker_for = {"verified": "OK ", "partially_verified": "REP",
                  "refused": "REF", "unverified": "RAW"}
    breaker = InfraBreaker(conditions)
    aborted = False

    for index, row in enumerate(rows, start=1):
        todo = [c for c in conditions if row["id"] not in runners[c]["done"]]
        if not todo:
            continue
        norm = cache.get(row["question"])

        for condition in todo:
            runner = runners[condition]
            record = run_one(
                row, condition, runner["pipeline"], retriever, runner["auditor"],
                norm, runner["meter"], base.retrieval_top_k,
                audit_all=args.audit_all, canned=canned,
            )
            runner["records"].append(record)
            runner["done"].add(row["id"])
            save_checkpoint(condition)

            if record["infra_failure"]:
                tag = "INF"
            else:
                tag = marker_for.get(record["status"], "ERR")
            tripped = breaker.record(condition, bool(record["infra_failure"]))
            expected_marker = "" if row["answerable"] else " (should refuse)"
            print(f"  [{index:2d}/{len(rows)}] {condition.upper():3s} {tag} {row['id']:4s} "
                  f"{record['latency_ms'] / 1000:5.1f}s {record['llm_calls']}c "
                  f"score={record['best_retrieval_score']:.2f} "
                  f"{row['question'][:40]}{expected_marker}")
            if record["infra_failure"]:
                print(f"           infrastructure failure: {record['infra_reason']} "
                      f"{record['error'][:80]}")

            if tripped:
                aborted = True
                break
        if aborted:
            break
        if index < len(rows):
            time.sleep(args.delay)  # Groq's free tier is per-minute

    if aborted:
        print(f"\nABORTED: {MAX_CONSECUTIVE_INFRA_FAILURES} consecutive infrastructure "
              "failures (rate limit / quota / outage). Progress is saved; wait for the "
              "budget to reset and re-run the same command with --resume.")
        return 2

    # ---- summarise and write ------------------------------------------------
    order = {r["id"]: i for i, r in enumerate(rows)}
    summaries, all_records, condition_meta = {}, {}, {}
    for condition in conditions:
        records = sorted(runners[condition]["records"], key=lambda r: order[r["id"]])
        all_records[condition] = records
        condition_meta[condition] = runners[condition]["cmeta"]
        summaries[condition] = summarise(records)

    # Keep an earlier condition only if it is comparable with what just ran.
    # Anything else is moved to a backup file rather than silently dropped.
    dropped = False
    reference = condition_meta[conditions[0]]
    for key in ("off", "on"):
        if key in condition_meta or key not in existing.get("condition_meta", {}):
            continue
        differing = meta_compatible(existing["condition_meta"][key], reference)
        if differing:
            print(f"  (earlier '{key}' run differs in {', '.join(differing)}: not merged)")
            dropped = True
            continue
        condition_meta[key] = existing["condition_meta"][key]
        all_records[key] = existing["records"][key]  # summary is recomputed below
        print(f"  (merged the earlier '{key}' run from {out_dir.name}/raw_results.json)")
    if dropped:
        _write_json(out_dir / "raw_results.previous.json", existing)

    # Score OFF and ON on the SAME questions: an id that failed (or could not be
    # classified) in either condition is dropped from both. Done after merging, so
    # a merged earlier condition is paired with the one that just ran.
    drop_ids, refusal_skip_ids = pair_records(all_records)
    for condition in all_records:
        summaries[condition] = summarise(all_records[condition], drop_ids, refusal_skip_ids)
        s = summaries[condition]
        print(f"\n  -- guardrail {condition.upper()} summary "
              f"(scored {s['scored']}, infra failures {s['infra_failures']}, "
              f"dropped to pair with the other condition {s['paired_excluded']}) --")
        print(f"     retrieval hit rate     : {fmt_pct(s['retrieval_hit_rate'], s['retrieval_n'])}")
        print(f"     correct refusal rate   : {fmt_pct(s['correct_refusal_rate'], s['refusal_unanswerable_n'])}")
        print(f"     over-blocking rate     : {fmt_pct(s['over_blocking_rate'], s['refusal_answerable_n'])}")
        print(f"     unsupported claim rate : {fmt_pct(s['unsupported_claim_rate'], s['delivered_claims'])}")

    meta = {
        "date": date.today().isoformat(),
        "model": base.llm_model,
        "base_url": base.llm_base_url,
        "embedding_model": base.embedding_model,
        "top_k": base.retrieval_top_k,
        "min_score": base.retrieval_min_score,
        "block_threshold": base.guardrail_block_threshold,
        "chunks": retriever.count(),
        "schemes": len(list((ROOT / "data" / "schemes").glob("*.md"))),
        "n_questions": len(rows),
        "n_golden": len(all_rows),
        "subset": subset,
        "answerable": sum(1 for r in rows if r["answerable"]),
        "unanswerable": sum(1 for r in rows if not r["answerable"]),
        "audit": not args.no_audit,
        "audit_all": bool(args.audit_all),
    }

    _write_json(out_dir / "raw_results.json", {
        "meta": meta, "condition_meta": condition_meta,
        "summaries": summaries, "records": all_records,
    })
    with io.open(out_dir / "results.md", "w", encoding="utf-8") as fh:
        fh.write(build_report(summaries, meta, condition_meta))

    for condition in conditions:  # finished: the checkpoint is now stale
        try:
            (out_dir / f"_checkpoint_{condition}.json").unlink()
        except OSError:
            pass

    print(f"\nWrote {out_dir / 'results.md'}")
    print(f"Wrote {out_dir / 'raw_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
