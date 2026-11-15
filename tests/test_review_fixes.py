"""
Regression tests for the second-round review findings.

One group per finding, each pinning the behaviour that was wrong. No network, no
model: the LLM and the embedding model are never loaded.
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.llm_client import LLMClient, LLMError
from app.schemas import (
    AnswerStatus,
    ChunkMetadata,
    GuardrailDecision,
    GuardrailReport,
    Language,
    RefusalReason,
    RetrievalSummary,
    RetrievedChunk,
)
from core.eligibility import (
    BROAD_OCCUPATIONS,
    KNOWN_OCCUPATIONS,
    SchemeRules,
    UserProfile,
    check_scheme,
    normalize_profile,
)
from core.generator import AnswerPipeline, build_citations
from core.guardrail import numeric_and_link_grounding
from eval import run_eval
from eval.common import META_COMPARE_KEYS, meta_compatible
from eval.run_eval import InfraBreaker, base_record, build_report, pair_records, summarise

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import update_readme_eval  # noqa: E402


def _chunk(chunk_id: str, text: str = "t", url: str = "https://x.gov.in") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, text=text, score=0.9,
        metadata=ChunkMetadata(
            scheme_id=chunk_id.split("#")[0], scheme_name="S", section="Benefits",
            source_url=url, last_verified="2026-10-02",
        ),
    )


# ==========================================================================
# Broad occupations must ask, not reject
# ==========================================================================


class TestBroadOccupations:
    VENDOR_RULES = SchemeRules(scheme_id="pm_svanidhi", scheme_name="PM SVANidhi", allowed_occupations=["street_vendor"])

    @pytest.mark.parametrize("occupation", sorted(BROAD_OCCUPATIONS))
    def test_umbrella_occupation_is_need_more_info_not_not_eligible(self, occupation):
        profile = normalize_profile(UserProfile(age=30, occupation=occupation))
        result = check_scheme(profile, self.VENDOR_RULES)
        assert result.status.value == "NEED_MORE_INFO"
        assert "occupation" in result.missing_fields

    def test_umbrella_values_are_in_the_vocabulary(self):
        assert BROAD_OCCUPATIONS <= set(KNOWN_OCCUPATIONS)

    def test_a_specific_non_matching_occupation_is_still_a_firm_no(self):
        profile = normalize_profile(UserProfile(age=30, occupation="salaried"))
        assert check_scheme(profile, self.VENDOR_RULES).status.value == "NOT_ELIGIBLE"

    def test_a_matching_occupation_still_passes(self):
        profile = normalize_profile(UserProfile(age=30, occupation="street_vendor"))
        assert check_scheme(profile, self.VENDOR_RULES).status.value == "ELIGIBLE"


# ==========================================================================
# build_citations lists every verified chunk
# ==========================================================================


class TestCitationsListEveryVerifiedChunk:
    def test_a_verified_chunk_cited_under_the_wrong_marker_is_still_listed(self):
        chunks = [_chunk("a#s#0"), _chunk("b#s#0"), _chunk("c#s#0")]
        # Claim 2 is verified against chunk 2 but the model wrote [3].
        citations = build_citations(
            "Rs 6000 per year [1]. Paid in three instalments [3].",
            chunks, ["a#s#0", "b#s#0"],
        )
        assert [c.index for c in citations] == [1, 2]

    def test_never_falls_back_to_all_retrieved(self):
        chunks = [_chunk("a#s#0"), _chunk("b#s#0"), _chunk("c#s#0")]
        citations = build_citations("Fact [1] [2] [3].", chunks, ["b#s#0"])
        assert [c.chunk_id for c in citations] == ["b#s#0"]


# ==========================================================================
# A withheld answer is never counted as PASS / REPAIR in /metrics
# ==========================================================================


class TestRefusalLoggedAsBlock:
    @staticmethod
    def _refuse(monkeypatch, report_decision, reason):
        events = []
        monkeypatch.setattr(
            "core.generator.log_event",
            lambda kind, value, detail="", settings=None: events.append((kind, value)),
        )
        pipeline = object.__new__(AnswerPipeline)
        pipeline.settings = SimpleNamespace(guardrail_block_threshold=0.5)
        report = GuardrailReport(decision=report_decision, claims=[], block_threshold=0.5)
        pipeline._refusal_response(
            question="q",
            normalized=SimpleNamespace(normalized_query="q", fallback_used=False),
            language=Language.ENGLISH,
            summary=RetrievalSummary(query="q", top_k=5, threshold=0.45,
                                     best_score=0.8, low_evidence=False),
            counter=SimpleNamespace(count=0), started=0.0, trace=[],
            reason=reason, note="n", report=report,
        )
        return dict(events)

    def test_no_claims_refusal_is_not_logged_as_pass(self, monkeypatch):
        events = self._refuse(monkeypatch, GuardrailDecision.PASS, RefusalReason.NO_CLAIMS)
        assert events["guardrail_decision"] == "BLOCK"
        assert events["answer_status"] == AnswerStatus.REFUSED.value

    def test_failed_repair_is_not_logged_as_repair(self, monkeypatch):
        events = self._refuse(monkeypatch, GuardrailDecision.REPAIR, RefusalReason.NOT_SUPPORTED)
        assert events["guardrail_decision"] == "BLOCK"

    def test_outage_is_still_skipped(self, monkeypatch):
        events = self._refuse(monkeypatch, GuardrailDecision.BLOCK,
                              RefusalReason.SERVICE_UNAVAILABLE)
        assert events["guardrail_decision"] == "SKIPPED"
        assert events["answer_status"] == "service_unavailable"


# ==========================================================================
# Streamlit entry point restarts a dead API thread
# ==========================================================================


def test_dead_api_thread_is_restarted(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    started = []

    import threading

    real_thread = threading.Thread

    class FakeThread(real_thread):
        """Only the API thread is faked; AppTest's own script thread stays real."""

        def __init__(self, *args, **kwargs):
            self._fake = kwargs.get("name") == "schemesure-api"
            if not self._fake:
                # Called on the real class: threading.Timer calls Thread.__init__
                # by NAME, which is this subclass while the patch is active.
                real_thread.__init__(self, *args, **kwargs)

        def start(self):
            if self._fake:
                started.append(self)
            else:
                real_thread.start(self)

        def is_alive(self):
            return False if self._fake else real_thread.is_alive(self)

    def never_up(*args, **kwargs):
        import requests
        raise requests.exceptions.ConnectionError("down")

    st.cache_resource.clear()
    monkeypatch.setattr("threading.Thread", FakeThread)
    monkeypatch.setattr("requests.get", never_up)
    # Near-instant sleeps (not a no-op: Streamlit's test runner polls with sleep).
    import time
    real_sleep = time.sleep
    monkeypatch.setattr("time.sleep", lambda seconds: real_sleep(0.001))

    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
    app.run()
    st.cache_resource.clear()
    # First thread was dead, so a SECOND one was started instead of waiting on it.
    assert len(started) == 2


# ==========================================================================
# Numeric grounding: decimals / percentages / lakh, and "year.In"
# ==========================================================================


class TestNumericGroundingWidened:
    PM_JJBY = [_chunk("pmjjby#b#0", "Life cover of ₹ 2.00 Lakh. Premium Rs. 436 per annum. "
                                    "Interest 7.6 percent for girls aged 3 years.")]

    def test_fabricated_percentage_below_ten_is_flagged(self):
        flagged = numeric_and_link_grounding("Sukanya gives 8.2% interest [1]", self.PM_JJBY)
        assert any("8.2" in token for token in flagged)

    def test_grounded_decimal_is_not_flagged(self):
        assert numeric_and_link_grounding("The rate is 7.6 percent.", self.PM_JJBY) == []

    def test_swapped_lakh_amount_is_flagged(self):
        flagged = numeric_and_link_grounding("Cover of Rs 5 lakh [1]", self.PM_JJBY)
        assert flagged

    def test_lakh_matches_digit_groups(self):
        assert numeric_and_link_grounding("Cover of Rs 2,00,000 [1]", self.PM_JJBY) == []
        assert numeric_and_link_grounding("Cover of 2 lakh [1]", self.PM_JJBY) == []

    def test_small_whole_counts_are_still_ignored(self):
        assert numeric_and_link_grounding("Paid in 3 instalments [1]", self.PM_JJBY) == []

    def test_missing_space_after_full_stop_is_not_a_link(self):
        chunks = [_chunk("k#b#0", "Rs 6000 per year, paid in instalments.")]
        assert numeric_and_link_grounding("Rs 6000 per year.In addition to that", chunks) == []

    def test_a_made_up_real_domain_is_still_flagged(self):
        chunks = [_chunk("k#b#0", "Rs 6000 per year.")]
        assert numeric_and_link_grounding("Apply at fake-scheme.gov.in now", chunks)


# ==========================================================================
# LLM client: hard time budget and malformed responses
# ==========================================================================


class _BadSDK:
    """Fake SDK returning 200 with the given choices list."""

    def __init__(self, choices):
        self.choices = choices
        self.chat = self
        self.completions = self
        self.timeouts = []

    def create(self, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        return SimpleNamespace(choices=self.choices, usage=None)


class TestLLMClientRobustness:
    def test_empty_choices_is_a_bad_output_error_and_is_logged(self, monkeypatch):
        logged = []
        monkeypatch.setattr("app.llm_client._log_call",
                            lambda *args, **kwargs: logged.append((args, kwargs)))
        llm = LLMClient()
        llm._client = _BadSDK([])
        with pytest.raises(LLMError) as info:
            llm.complete("p", "s", "u")
        assert info.value.kind == "bad_output"
        assert logged, "the failed call must still be written to the call log"

    def test_timeout_is_a_structured_httpx_budget_that_sums_to_the_limit(self, monkeypatch):
        monkeypatch.setattr("app.llm_client._log_call", lambda *a, **k: None)
        llm = LLMClient()
        llm._client = _BadSDK([SimpleNamespace(
            message=SimpleNamespace(content="ok"), finish_reason="stop")])
        llm.complete("p", "s", "u")
        limits = llm._client.timeouts[0]
        assert isinstance(limits, httpx.Timeout)
        assert limits.connect + limits.read <= llm.settings.llm_timeout_seconds + 1e-9


# ==========================================================================
# Deploy: the Dockerfile's COPY must be uploaded
# ==========================================================================


def test_every_dockerfile_copy_source_is_uploaded_by_deploy_script():
    import deploy_hf

    dockerfile = io.open(ROOT / "Dockerfile", encoding="utf-8").read()
    sources = re.findall(r"^COPY\s+(?:--\S+\s+)*(\S+)\s+\S+\s*$", dockerfile, re.M)
    uploaded = {path.relative_to(ROOT).as_posix() for path in deploy_hf.collect_files(ROOT)}
    for source in sources:
        assert any(path == source or path.startswith(source.rstrip("/") + "/")
                   for path in uploaded), f"Dockerfile COPYs '{source}' but deploy_hf.py never uploads it"
    assert "secrets.toml" in deploy_hf.FORBIDDEN_NAMES


# ==========================================================================
# Evaluation: paired scoring, per-condition breaker, undetermined refusals
# ==========================================================================


def _row(id_, answerable=True):
    return {"id": id_, "question": "q", "language": "en", "answerable": answerable,
            "category": "benefit_amount" if answerable else "off_topic",
            "expected_scheme": "pm_kisan" if answerable else None,
            "key_facts": ["6000"] if answerable else []}


def _rec(id_, condition, answerable=True, **overrides):
    record = base_record(_row(id_, answerable), condition)
    record.update({"refused": False, "status": "verified", "llm_calls": 4,
                   "latency_ms": 1000, "delivered_claims": 1, "delivered_unsupported": 0})
    record.update(overrides)
    return record


class TestPairedScoring:
    def test_infra_failure_in_one_condition_is_dropped_from_both(self):
        # OFF answers u1 wrongly and refuses u2. ON hits an infra failure on u1.
        off = [_rec("u1", "off", False, refused=False),
               _rec("u2", "off", False, refused=True)]
        on = [_rec("u1", "on", False, infra_failure=True),
              _rec("u2", "on", False, refused=True)]
        drop, skip = pair_records({"off": off, "on": on})
        s_off, s_on = summarise(off, drop, skip), summarise(on, drop, skip)
        assert s_off["scored_ids"] == s_on["scored_ids"] == ["u2"]
        assert s_off["correct_refusal_rate"] == s_on["correct_refusal_rate"] == 100.0
        assert s_off["paired_excluded"] == 1 and s_on["infra_failures"] == 1

    def test_report_refuses_to_compare_different_scored_sets(self):
        off = [_rec("a1", "off"), _rec("a2", "off")]
        on = [_rec("a1", "on"), _rec("a2", "on", infra_failure=True)]
        meta = {"n_questions": 2, "n_golden": 2, "subset": False, "date": "d", "model": "m",
                "base_url": "b", "embedding_model": "e", "top_k": 5, "min_score": 0.45,
                "block_threshold": 0.5, "chunks": 1, "schemes": 1, "answerable": 2,
                "unanswerable": 0, "audit": True}
        text = build_report({"off": summarise(off), "on": summarise(on)}, meta)
        assert "NOT comparable" in text and "scored question set" in text

    def test_publish_is_refused_when_scored_sets_differ(self):
        off = summarise([_rec("a1", "off"), _rec("a2", "off")])
        on = summarise([_rec("a1", "on"), _rec("a2", "on", infra_failure=True)])
        cmeta = {"question_ids": ["a1", "a2"]}
        ok, reason = update_readme_eval.check_publishable(
            {"summaries": {"off": off, "on": on}, "condition_meta": {"off": cmeta, "on": cmeta}})
        assert not ok and "different question sets" in reason


class TestInfraBreaker:
    def test_on_failing_alone_trips_the_breaker_despite_off_successes(self):
        breaker = InfraBreaker(["off", "on"], limit=3)
        tripped = False
        for _ in range(3):                      # interleaved: OFF ok, ON fails
            assert breaker.record("off", False) is False
            tripped = breaker.record("on", True)
        assert tripped

    def test_a_success_resets_that_condition(self):
        breaker = InfraBreaker(["on"], limit=3)
        breaker.record("on", True)
        breaker.record("on", True)
        breaker.record("on", False)
        assert breaker.record("on", True) is False


class TestRefusalUndetermined:
    def test_unaudited_non_refusal_is_flagged_and_left_out_of_both_conditions(self):
        off = [_rec("u1", "off", False, refusal_undetermined=True, delivered_claims=None),
               _rec("u2", "off", False, refused=True)]
        on = [_rec("u1", "on", False, refused=True),     # pipeline NO_CLAIMS refusal
              _rec("u2", "on", False, refused=True)]
        drop, skip = pair_records({"off": off, "on": on})
        assert skip == {"u1"} and not drop
        s_off, s_on = summarise(off, drop, skip), summarise(on, drop, skip)
        assert s_off["refusal_unanswerable_n"] == s_on["refusal_unanswerable_n"] == 1
        assert s_off["correct_refusal_rate"] == s_on["correct_refusal_rate"] == 100.0

    def test_run_one_flags_an_answer_with_no_audit(self):
        # base record default: not flagged; flag is only set by run_one when no audit.
        assert base_record(_row("a1"), "off")["refusal_undetermined"] is False


class TestMetaKeysAndResume:
    def test_embedding_model_and_golden_set_must_match(self):
        assert "embedding_model" in META_COMPARE_KEYS and "golden_sha1" in META_COMPARE_KEYS
        a = {"embedding_model": "A", "golden_sha1": "1"}
        assert meta_compatible(a, {"embedding_model": "B", "golden_sha1": "1"}) == ["embedding_model"]
        assert meta_compatible(a, {"embedding_model": "A", "golden_sha1": "2"}) == ["golden_sha1"]


class TestPercentagePoints:
    def test_delta_is_labelled_pp_not_percent(self):
        off = {"x": 40.0}
        on = {"x": 10.0}
        assert run_eval._delta_pct(off, on, "x", higher_is_better=False) == "better (-30.0 pp)"
        assert update_readme_eval._change(off, on, "x", "lower") == "better (-30.0 pp)"


class TestResumeBullets:
    @staticmethod
    def _run(tmp_path, monkeypatch, off_rate, on_rate, off_refuse, on_refuse):
        notes = tmp_path / "notes.md"
        notes.write_text(
            "from `<OFF>%` to `<ON>%` on a 50-question benchmark.\n"
            "questions to `<ON>%` while holding over-blocking at `<X>%`, "
            "at a measured cost of 4 LLM calls per answer.\n", encoding="utf-8")
        monkeypatch.setattr(update_readme_eval, "NOTES", notes)
        off = {"unsupported_claim_rate": off_rate, "correct_refusal_rate": off_refuse,
               "over_blocking_rate": 0.0, "avg_llm_calls": 2.0}
        on = {"unsupported_claim_rate": on_rate, "correct_refusal_rate": on_refuse,
              "over_blocking_rate": 5.0, "avg_llm_calls": 4.6}
        done = update_readme_eval.fill_resume_bullets(on, off, {"n_questions": 50})
        return done, notes.read_text(encoding="utf-8")

    def test_call_cost_is_filled_from_the_run_and_both_baselines_shown(self, tmp_path, monkeypatch):
        done, text = self._run(tmp_path, monkeypatch, 20.0, 0.0, 40.0, 80.0)
        assert done
        assert "4.6 LLM calls" in text and "vs 2.0" in text and "4 LLM calls per answer" not in text
        assert "from **40.0%** to **80.0%**" in text

    def test_refuses_a_cut_that_did_not_happen(self, tmp_path, monkeypatch):
        done, text = self._run(tmp_path, monkeypatch, 10.0, 12.0, 40.0, 80.0)
        assert not done and "<OFF>" in text

    def test_refuses_a_raise_that_did_not_happen(self, tmp_path, monkeypatch):
        done, _ = self._run(tmp_path, monkeypatch, 20.0, 0.0, 40.0, 40.0)
        assert not done
