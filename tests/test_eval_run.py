"""
Tests for eval/run_eval.py (summarise, refusal detection, run_one), eval/norm_cache.py
and scripts/update_readme_eval.py. Hand-made records and fakes: no model, no network.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.llm_client import LLMError
from app.schemas import (
    AnswerStatus,
    ClaimCheck,
    GuardrailDecision,
    GuardrailReport,
    Language,
    NormalizedQuery,
    RetrievalResult,
    RetrievedChunk,
    ChunkMetadata,
    Verdict,
)
from eval import run_eval
from eval.norm_cache import NormalizationCache, cache_key
from eval.run_eval import base_record, build_report, detect_refusal, run_one, summarise

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import update_readme_eval  # noqa: E402


def make_record(id_="a01", answerable=True, category="benefit_amount", **overrides):
    row = {"id": id_, "question": "q", "language": "en", "answerable": answerable,
           "category": category, "expected_scheme": "pm_kisan" if answerable else None,
           "key_facts": ["6000"] if answerable else []}
    record = base_record(row, "on")
    record.update({"refused": False, "status": "verified", "llm_calls": 4,
                   "latency_ms": 1000, "guardrail_decision": "PASS"})
    record.update(overrides)
    return record


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------


def test_infra_failures_are_excluded_and_counted_not_refusals():
    records = [
        make_record("a01", refused=False, key_facts_found=1, retrieval_hit=True,
                    retrieval_hit_top1=True, delivered_claims=2, delivered_unsupported=0),
        # An answerable question that failed because the model was unavailable:
        # must NOT count as over-blocking.
        make_record("a02", refused=True, infra_failure=True, infra_reason="service_unavailable"),
        make_record("u01", answerable=False, category="off_topic", refused=True),
    ]
    s = summarise(records)
    assert s["questions"] == 3 and s["infra_failures"] == 1 and s["scored"] == 2
    assert s["infra_failure_ids"] == ["a02"]
    assert s["answerable"] == 1
    assert s["over_blocking_rate"] == 0.0 and s["over_blocked"] == 0
    assert s["correct_refusal_rate"] == 100.0


def test_empty_denominators_are_none_not_zero():
    s = summarise([make_record("a01", infra_failure=True)])
    assert s["scored"] == 0
    for key in ("over_blocking_rate", "correct_refusal_rate", "unsupported_claim_rate",
                "key_fact_coverage", "end_to_end_fact_accuracy", "retrieval_hit_rate",
                "answers_with_unsupported_claim_rate"):
        assert s[key] is None, key
    assert s["avg_latency_ms"] is None and s["p95_latency_ms"] is None
    # no answerable questions at all
    only_unanswerable = summarise([make_record("u01", answerable=False, category="off_topic",
                                               refused=True)])
    assert only_unanswerable["over_blocking_rate"] is None
    assert only_unanswerable["correct_refusal_rate"] == 100.0


def test_end_to_end_accuracy_counts_refusals_as_zero():
    records = [
        make_record("a01", key_facts_found=1),                                   # answered, correct
        make_record("a02", refused=True, status="refused", key_facts_found=0),  # refused
    ]
    s = summarise(records)
    assert s["key_fact_coverage"] == 100.0             # answered questions only
    assert s["end_to_end_fact_accuracy"] == 50.0       # refusal = 0 facts
    assert s["answered_rate"] == 50.0 and s["over_blocking_rate"] == 50.0
    assert s["fully_correct"] == 1 and s["fully_correct_n"] == 2


def test_unsupported_claim_rate_only_over_audited_answers():
    records = [
        make_record("a01", delivered_claims=4, delivered_unsupported=1),
        make_record("a02", delivered_claims=6, delivered_unsupported=0),
        make_record("a03", delivered_claims=None),                  # never audited
        make_record("a04", delivered_claims=None, audit_error="boom"),
    ]
    s = summarise(records)
    assert s["delivered_claims"] == 10 and s["delivered_unsupported"] == 1
    assert s["unsupported_claim_rate"] == 10.0
    assert s["answers_audited"] == 2 and s["answers_with_unsupported_claim"] == 1
    assert s["answers_not_audited"] == 2 and s["audit_failures"] == 1


def test_refusal_by_category_and_p95_nearest_rank():
    records = [make_record(f"u{i}", answerable=False, category="fake_scheme",
                           refused=(i < 3), latency_ms=1000 * (i + 1)) for i in range(4)]
    s = summarise(records)
    assert s["refusal_by_category"]["fake_scheme"] == {"n": 4, "refused": 3, "rate": 75.0}
    assert s["p95_latency_ms"] == 4000          # nearest rank of 4 values
    assert s["median_latency_ms"] == 2500


def test_similarity_stats_need_both_classes():
    records = [make_record("a01", claim_sims=[
        [Verdict.SUPPORTED.value, 0.8, 0.9],
        [Verdict.SUPPORTED.value, 0.7, 0.8],
        [Verdict.NOT_SUPPORTED.value, 0.2, 0.5],
        [Verdict.NOT_SUPPORTED.value, None, 0.5],      # no similarity: skipped
    ])]
    sim = summarise(records)["similarity"]
    assert sim["n_supported"] == 2 and sim["n_not_supported"] == 1
    assert sim["mean_supported"] == 0.75 and sim["auroc"] == 1.0
    assert summarise([make_record("a01")])["similarity"]["auroc"] is None


def test_token_averages_ignore_records_without_tokens():
    records = [make_record("a01", prompt_tokens=100, completion_tokens=50),
               make_record("a02", prompt_tokens=None, completion_tokens=None)]
    s = summarise(records)
    assert s["avg_tokens_per_question"] == 150
    assert s["total_prompt_tokens"] == 100
    assert summarise([make_record("a01")])["avg_tokens_per_question"] is None


# --------------------------------------------------------------------------
# refusal detector
# --------------------------------------------------------------------------


def test_detect_refusal_sources():
    from app import prompts

    canned_text = next(iter(prompts.REFUSAL_MESSAGES.values()))
    assert detect_refusal("refused", "anything") == (True, "pipeline")
    assert detect_refusal("unverified", canned_text) == (True, "canned_text")
    assert detect_refusal("unverified", "") == (True, "canned_text")
    # model declined in its own words: only the audit can tell (zero claims)
    own_words = "Mujhe PM Kisan ke baare mein official jankari nahi hai."
    assert detect_refusal("unverified", own_words) == (False, "")
    assert detect_refusal("unverified", own_words, audited_claim_count=0) == (True, "no_claims")
    assert detect_refusal("unverified", "Rs 6000 per year [1]", audited_claim_count=2) == (False, "")


# --------------------------------------------------------------------------
# run_one with fakes
# --------------------------------------------------------------------------


def chunk(chunk_id="pm_kisan#benefits#0", scheme="pm_kisan", score=0.7):
    return RetrievedChunk(
        chunk_id=chunk_id, text="Rs 6000 per year", score=score,
        metadata=ChunkMetadata(scheme_id=scheme, scheme_name="PM-KISAN", section="Benefits",
                               source_url="https://pmkisan.gov.in", last_verified="2026-01-01"),
    )


class FakeRetriever:
    def search(self, query, top_k=None, min_score=None):
        chunks = [chunk(), chunk("pm_kisan#overview#0")]
        return RetrievalResult(query=query, chunks=chunks, best_score=0.7,
                               low_evidence=False, threshold=0.0)


class FakeMeter:
    def snapshot(self):
        return (0, 0)


def fake_norm(fallback=False):
    return SimpleNamespace(
        normalized=NormalizedQuery(normalized_query="how much", language=Language.ENGLISH,
                                   fallback_used=fallback),
        from_cache=True, latency_ms=200, prompt_tokens=30, completion_tokens=10, llm_calls=0,
    )


def claims(*verdicts):
    return [ClaimCheck(claim=f"c{i}", verdict=v, evidence_chunk_id="pm_kisan#benefits#0")
            for i, v in enumerate(verdicts)]


def fake_response(status=AnswerStatus.VERIFIED, decision=GuardrailDecision.PASS, claim_list=None,
                  answer="Rs 6000 per year [1]", reason=None, **extra):
    guardrail = GuardrailReport(decision=decision, claims=claim_list or [])
    return SimpleNamespace(
        status=status, answer=answer, guardrail=guardrail, citations=[1], latency_ms=900,
        llm_calls=3, refusal_reason=reason, normalized_query="how much", **extra,
    )


class FakePipeline:
    def __init__(self, response, accept_normalized=True):
        self.response, self.accept_normalized, self.calls = response, accept_normalized, 0

    def ask(self, question, language_override=None, normalized=None):
        if normalized is not None and not self.accept_normalized:
            raise TypeError("ask() got an unexpected keyword argument 'normalized'")
        self.calls += 1
        return self.response


class FakeAuditor:
    def __init__(self, report):
        self.report, self.calls = report, 0

    def verify(self, answer, chunks, counter=None):
        self.calls += 1
        return self.report


ROW = {"id": "a01", "question": "How much does PM-KISAN pay?", "language": "en",
       "answerable": True, "category": "benefit_amount", "expected_scheme": "pm_kisan",
       "key_facts": ["6000"]}


def go(response, auditor=None, norm=None, audit_all=False, pipeline=None):
    return run_one(ROW, "on", pipeline or FakePipeline(response), FakeRetriever(), auditor,
                   norm or fake_norm(), FakeMeter(), 5, audit_all=audit_all)


def test_run_one_pass_reuses_verdicts_and_adds_normalizer_cost():
    response = fake_response(claim_list=claims(Verdict.SUPPORTED, Verdict.SUPPORTED))
    auditor = FakeAuditor(None)
    record = go(response, auditor)
    assert auditor.calls == 0
    assert record["audit_source"] == "guardrail_pass"
    assert record["delivered_claims"] == 2 and record["delivered_unsupported"] == 0
    assert record["llm_calls"] == 4 and record["latency_ms"] == 1100   # +1 call, +200 ms
    assert record["prompt_tokens"] == 30 and record["refused"] is False
    assert record["retrieval_hit"] is True and record["retrieval_hit_top1"] is True
    assert len(record["retrieval_ranked"]) == 2 and record["key_facts_found"] == 1


def test_run_one_audit_all_forces_an_independent_audit():
    response = fake_response(claim_list=claims(Verdict.SUPPORTED))
    audit = GuardrailReport(decision=GuardrailDecision.REPAIR,
                            claims=claims(Verdict.SUPPORTED, Verdict.NOT_SUPPORTED))
    auditor = FakeAuditor(audit)
    record = go(response, auditor, audit_all=True)
    assert auditor.calls == 1 and record["audit_source"] == "audit"
    assert record["delivered_unsupported"] == 1


def test_run_one_reuses_repair_report():
    repaired = GuardrailReport(decision=GuardrailDecision.PASS, claims=claims(Verdict.SUPPORTED))
    response = fake_response(status=AnswerStatus.PARTIALLY_VERIFIED,
                             decision=GuardrailDecision.REPAIR,
                             claim_list=claims(Verdict.SUPPORTED, Verdict.NOT_SUPPORTED),
                             repair_report=repaired)
    auditor = FakeAuditor(None)
    record = go(response, auditor)
    assert auditor.calls == 0 and record["audit_source"] == "repair_report"
    assert record["delivered_claims"] == 1 and record["delivered_unsupported"] == 0


def test_run_one_service_unavailable_is_infra_failure():
    response = fake_response(status=AnswerStatus.REFUSED, decision=GuardrailDecision.BLOCK,
                             reason="service_unavailable", answer="try again")
    record = go(response)
    assert record["infra_failure"] and record["infra_reason"] == "service_unavailable"
    assert summarise([record])["infra_failures"] == 1


def test_run_one_verification_error_is_infra_failure():
    response = fake_response(status=AnswerStatus.REFUSED, decision=GuardrailDecision.BLOCK)
    response.guardrail.verification_error = "judge failed"
    assert go(response)["infra_failure"] is True


def test_run_one_off_answer_with_no_claims_counts_as_refused():
    """Guardrail OFF: the model declined in its own words -> same refusal as ON."""
    response = fake_response(status=AnswerStatus.UNVERIFIED, decision=GuardrailDecision.SKIPPED,
                             answer="Mujhe iske baare mein jaankari nahi hai")
    audit = GuardrailReport(decision=GuardrailDecision.PASS, claims=[])
    record = go(response, FakeAuditor(audit))
    assert record["refused"] is True and record["refusal_source"] == "no_claims"
    assert record["key_facts_found"] == 0


def test_run_one_audit_failure_is_recorded_not_counted_as_clean():
    response = fake_response(status=AnswerStatus.UNVERIFIED, decision=GuardrailDecision.SKIPPED)
    failed = GuardrailReport(decision=GuardrailDecision.BLOCK, claims=[])
    failed.verification_error = "rate limit"
    record = go(response, FakeAuditor(failed))
    assert record["audit_error"] and record["delivered_claims"] is None
    assert record["refused"] is False
    assert summarise([record])["answers_not_audited"] == 1


def test_run_one_normalizer_fallback_skips_the_pipeline():
    pipeline = FakePipeline(fake_response())
    record = go(None, norm=fake_norm(fallback=True), pipeline=pipeline)
    assert pipeline.calls == 0
    assert record["infra_failure"] and record["infra_reason"] == "normalizer_fallback"


def test_run_one_exception_is_infra_failure_not_a_crash():
    class Boom:
        def ask(self, *a, **k):
            raise RuntimeError("boom")

    record = go(None, pipeline=Boom())
    assert record["infra_failure"] and "RuntimeError" in record["error"]


def test_run_one_backend_without_normalized_param_still_runs_unpaired():
    response = fake_response(claim_list=claims(Verdict.SUPPORTED))
    record = go(response, pipeline=FakePipeline(response, accept_normalized=False))
    assert record["infra_failure"] is False
    assert record["llm_calls"] == 3      # normalizer call already inside response.llm_calls


# --------------------------------------------------------------------------
# normalization cache
# --------------------------------------------------------------------------


class FakeNormLLM:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def complete_json(self, prompt_name, system, user, max_tokens=None, counter=None):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        if counter is not None:
            counter.count += 1
        return {"normalized_query": "how much does pm kisan pay", "language": "hinglish"}, None


def test_normalization_cache_pays_once_and_persists(tmp_path):
    path = tmp_path / "normalized.jsonl"
    llm = FakeNormLLM()
    cache = NormalizationCache(llm, "model-x", path)
    first = cache.get("PM Kisan kitna deta hai?")
    again = cache.get("PM Kisan kitna deta hai?")
    assert llm.calls == 1 and not first.from_cache and again.from_cache
    assert again.normalized.language == Language.HINGLISH
    # a new cache object (a new run) reads the file: still no LLM call
    llm2 = FakeNormLLM()
    third = NormalizationCache(llm2, "model-x", path).get("PM Kisan kitna deta hai?")
    assert third.from_cache and llm2.calls == 0
    # a different model is a different key
    NormalizationCache(llm2, "model-y", path).get("PM Kisan kitna deta hai?")
    assert llm2.calls == 1
    assert cache_key("q", "a") != cache_key("q", "b") != cache_key("r", "b")


def test_normalization_cache_never_stores_fallbacks(tmp_path):
    path = tmp_path / "normalized.jsonl"
    llm = FakeNormLLM(fail=True)
    cache = NormalizationCache(llm, "model-x", path)
    result = cache.get("PM Kisan kitna deta hai?")
    assert result.normalized.fallback_used and not result.from_cache
    assert not path.exists() or path.read_text(encoding="utf-8").strip() == ""
    cache.get("PM Kisan kitna deta hai?")
    assert llm.calls == 2                 # tried again: the failure was not remembered


# --------------------------------------------------------------------------
# report + publishing guard
# --------------------------------------------------------------------------

META = {"date": "2026-11-15", "model": "m", "base_url": "u", "embedding_model": "e",
        "top_k": 5, "min_score": 0.45, "block_threshold": 0.5, "chunks": 10, "schemes": 15,
        "n_questions": 2, "n_golden": 50, "subset": True, "answerable": 1, "unanswerable": 1,
        "audit": True}
CMETA = {"question_ids": ["a01", "u01"], "model": "m", "top_k": 5, "min_score": 0.45,
         "block_threshold": 0.5}


def two_summaries():
    records = [make_record("a01", key_facts_found=1, delivered_claims=2, delivered_unsupported=0,
                           retrieval_hit=True, retrieval_hit_top1=True),
               make_record("u01", answerable=False, category="off_topic", refused=True)]
    return {"off": summarise(records), "on": summarise(records)}


def test_report_labels_subset_as_preliminary_and_states_n():
    report = build_report(two_summaries(), META, {"off": CMETA, "on": CMETA})
    assert "PRELIMINARY, n=2 of 50" in report
    assert "n=2" in report and "Headline comparison" in report
    assert "0 LLM calls" not in report            # the false off-topic wording is gone


def test_report_refuses_comparison_for_mismatched_conditions():
    other = {**CMETA, "question_ids": ["a01"]}
    report = build_report(two_summaries(), {**META, "subset": False},
                          {"off": CMETA, "on": other})
    assert "NOT comparable" in report and "Headline comparison" not in report
    assert "PRELIMINARY" not in report


def test_report_handles_empty_denominators():
    s = summarise([make_record("a01", infra_failure=True)])
    report = build_report({"off": s, "on": s}, META, {"off": CMETA, "on": CMETA})
    assert "n/a" in report


def test_publish_guard_requires_matching_conditions():
    s = two_summaries()
    ok, _ = update_readme_eval.check_publishable(
        {"summaries": s, "condition_meta": {"off": CMETA, "on": CMETA}})
    assert ok
    bad, reason = update_readme_eval.check_publishable(
        {"summaries": s, "condition_meta": {"off": CMETA, "on": {**CMETA, "top_k": 3}}})
    assert not bad and "top_k" in reason
    missing, reason = update_readme_eval.check_publishable(
        {"summaries": {"off": s["off"]}, "condition_meta": {"off": CMETA}})
    assert not missing
    legacy, reason = update_readme_eval.check_publishable({"summaries": s})
    assert not legacy and "metadata" in reason


def test_readme_table_states_n_and_preliminary_label():
    s = two_summaries()
    table = update_readme_eval.build_table(s["off"], s["on"], META)
    assert "PRELIMINARY: n=2 of 50" in table
    assert "n=2 questions per condition" in table
    full = update_readme_eval.build_table(s["off"], s["on"], {**META, "subset": False})
    assert "PRELIMINARY" not in full


def test_module_exports_shared_fact_matcher():
    assert run_eval.fact_present("6000", "Rs 6,000")
