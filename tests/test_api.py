"""
Integration tests for the FastAPI layer, with the LLM mocked.

WHAT IS REAL AND WHAT IS FAKE
-----------------------------
Real: the embedding model, the Chroma index, chunking, retrieval scores, the
guardrail's decision policy, citation validation, and the eligibility rules.
Fake: only the LLM calls.

That split is deliberate. Mocking retrieval too would leave these tests
asserting that mocks return mocks. Mocking only the model keeps them fast,
free and deterministic while still exercising the real pipeline — including
whether retrieval actually finds PM-KISAN when asked about PM-KISAN.

The fake judge reads the chunk ids out of the prompt it is given, so a
"supported" verdict cites a genuinely retrieved chunk and passes the same
Python citation check the real judge faces.
"""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from app.llm_client import LLMError, LLMResult
from app.main import app
from core.generator import AnswerPipeline

# --------------------------------------------------------------------------
# The fake LLM
# --------------------------------------------------------------------------

_CHUNK_ID_RE = re.compile(r"chunk_id:\s*(\S+)")


class FakeLLM:
    """
    Scripted stand-in for LLMClient.

    `mode` controls what the judge returns, which is how each guardrail outcome
    is driven:
        "all_supported" -> PASS
        "half"          -> REPAIR
        "none"          -> BLOCK
        "fabricated"    -> SUPPORTED verdicts citing ids that do not exist
    """

    def __init__(self, mode: str = "all_supported", fail_on: str | None = None):
        self.mode = mode
        self.fail_on = fail_on
        self.calls: list[str] = []

    # -- helpers ------------------------------------------------------------

    def _record(self, prompt_name: str, counter) -> LLMResult:
        self.calls.append(prompt_name)
        if self.fail_on == prompt_name:
            raise LLMError(f"simulated failure in {prompt_name}")
        result = LLMResult(
            text="", prompt_name=prompt_name, model="fake-model",
            prompt_tokens=10, completion_tokens=5, latency_ms=1,
        )
        if counter is not None:
            counter.record(result)
        return result

    def _payload(self, prompt_name: str, user: str):
        if prompt_name == "normalize_query":
            # Pull the question back out and pretend it was already English.
            match = re.search(r'"""(.*?)"""', user, re.S)
            question = (match.group(1).strip() if match else user).strip()
            return {"normalized_query": question, "language": "en"}

        if prompt_name == "split_claims":
            return {"claims": ["PM-KISAN pays Rs 6000 per year",
                               "The amount is paid in three instalments"]}

        if prompt_name == "judge_claims":
            claims = re.findall(r"^\s*\d+\.\s+(.*)$", user, re.M)
            chunk_ids = _CHUNK_ID_RE.findall(user)
            real_id = chunk_ids[0] if chunk_ids else None

            verdicts = []
            for index, claim in enumerate(claims):
                if self.mode == "all_supported":
                    verdicts.append({"claim": claim, "verdict": "SUPPORTED",
                                     "evidence_chunk_id": real_id})
                elif self.mode == "fabricated":
                    verdicts.append({"claim": claim, "verdict": "SUPPORTED",
                                     "evidence_chunk_id": "made_up#section#99"})
                elif self.mode == "half":
                    supported = index == 0
                    verdicts.append({
                        "claim": claim,
                        "verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
                        "evidence_chunk_id": real_id if supported else None,
                    })
                else:  # "none"
                    verdicts.append({"claim": claim, "verdict": "NOT_SUPPORTED",
                                     "evidence_chunk_id": None})
            return {"verdicts": verdicts}

        if prompt_name == "extract_profile":
            return {
                "age": 35, "annual_income": 200000, "state": "Bihar",
                "occupation": "farmer", "gender": "male", "category": None,
            }

        return {}

    # -- the LLMClient surface ---------------------------------------------

    def complete(self, prompt_name, system, user, json_mode=False,
                 max_tokens=None, counter=None) -> LLMResult:
        result = self._record(prompt_name, counter)
        if prompt_name == "generate_answer":
            result.text = (
                "PM-KISAN provides Rs 6000 per year to eligible farmer families [1]. "
                "It is paid in three equal instalments of Rs 2000 each [1]."
            )
        elif prompt_name == "repair_answer":
            result.text = "PM-KISAN provides Rs 6000 per year to eligible farmer families [1]."
        else:
            result.text = json.dumps(self._payload(prompt_name, user))
        return result

    def complete_json(self, prompt_name, system, user, max_tokens=None, counter=None):
        result = self._record(prompt_name, counter)
        payload = self._payload(prompt_name, user)
        result.text = json.dumps(payload)
        return payload, result


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """
    A TestClient with the real startup sequence.

    Module-scoped because the lifespan loads the embedding model, which takes a
    few seconds — once per module rather than once per test.
    """
    with TestClient(app) as test_client:
        yield test_client


def use_fake(client: TestClient, mode: str = "all_supported", fail_on: str | None = None) -> FakeLLM:
    """Swap the app's pipeline for one backed by a FakeLLM. Returns the fake."""
    from app.config import get_settings

    fake = FakeLLM(mode=mode, fail_on=fail_on)
    settings = get_settings()
    client.app.state.llm = fake
    client.app.state.pipeline = AnswerPipeline(client.app.state.retriever, fake, settings)
    return fake


# ==========================================================================
# Read-only endpoints
# ==========================================================================


class TestHealth:
    def test_health_reports_a_populated_index(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["schemes_loaded"] == 15
        # A real number from the real index, not a mock.
        assert body["chunks_indexed"] > 200
        assert "MiniLM" in body["embedding_model"]

    def test_root_points_at_the_docs(self, client):
        assert client.get("/").json()["docs"] == "/docs"


class TestSchemes:
    def test_lists_all_fifteen(self, client):
        schemes = client.get("/schemes").json()
        assert len(schemes) == 15

    def test_every_scheme_cites_an_official_url_and_a_date(self, client):
        for scheme in client.get("/schemes").json():
            assert ".gov.in" in scheme["source_url"]
            assert scheme["last_verified"]
            assert scheme["has_rules"] is True
            assert scheme["ministry"]

    def test_sections_are_present(self, client):
        schemes = {s["scheme_id"]: s for s in client.get("/schemes").json()}
        assert "Benefits" in schemes["pm_kisan"]["sections"]
        assert "Eligibility" in schemes["pm_kisan"]["sections"]


# ==========================================================================
# POST /ask
# ==========================================================================


class TestAsk:
    def test_answerable_question_is_verified_with_citations(self, client):
        fake = use_fake(client, "all_supported")
        response = client.post("/ask", json={"question": "How much money does PM-KISAN give per year?"})
        assert response.status_code == 200
        body = response.json()

        assert body["status"] == "verified"
        assert "6000" in body["answer"]
        assert body["citations"], "a verified answer must carry citations"
        assert body["guardrail"]["decision"] == "PASS"
        assert body["guardrail"]["supported_count"] == 2
        assert body["guardrail"]["unsupported_count"] == 0

        # 4 calls: normalize, generate, split, judge.
        assert body["llm_calls"] == 4
        assert fake.calls == ["normalize_query", "generate_answer", "split_claims", "judge_claims"]

    def test_retrieval_really_finds_the_right_scheme(self, client):
        """Guards the actual index, not the mock: the citations for a PM-KISAN
        question must point at PM-KISAN."""
        use_fake(client, "all_supported")
        body = client.post("/ask", json={"question": "How much money does PM-KISAN give per year?"}).json()
        assert any(citation["scheme_id"] == "pm_kisan" for citation in body["citations"])

    def test_out_of_scope_question_is_refused_without_generating(self, client):
        """The cost argument for refusing early: 1 LLM call, not 4."""
        fake = use_fake(client, "all_supported")
        body = client.post("/ask", json={"question": "What is the best pizza recipe?"}).json()

        assert body["status"] == "refused"
        assert body["citations"] == []
        assert body["llm_calls"] == 1              # normalize only
        assert fake.calls == ["normalize_query"]   # never generated
        assert "do not have official information" in body["answer"]
        assert "below the" in " ".join(body["guardrail"]["notes"])

    def test_unsupported_claims_trigger_repair(self, client):
        fake = use_fake(client, "half")
        body = client.post("/ask", json={"question": "How much money does PM-KISAN give per year?"}).json()

        assert body["guardrail"]["decision"] == "REPAIR"
        assert body["status"] == "partially_verified"
        assert body["llm_calls"] == 5          # the extra repair call
        assert "repair_answer" in fake.calls

    def test_fully_unsupported_answer_is_blocked(self, client):
        """The hallucination case: a fluent draft with nothing backing it."""
        use_fake(client, "none")
        body = client.post("/ask", json={"question": "How much money does PM-KISAN give per year?"}).json()

        assert body["guardrail"]["decision"] == "BLOCK"
        assert body["status"] == "refused"
        assert body["citations"] == []
        assert "could not verify" in body["answer"].lower()

    def test_fabricated_citations_are_caught_in_the_api_path(self, client):
        """
        End-to-end version of the key guardrail test: the judge says everything
        is SUPPORTED but cites chunk ids that were never retrieved. Taking it at
        face value would ship the answer; the Python check blocks it.
        """
        use_fake(client, "fabricated")
        body = client.post("/ask", json={"question": "How much money does PM-KISAN give per year?"}).json()

        assert body["guardrail"]["decision"] == "BLOCK"
        assert body["status"] == "refused"
        assert all(claim["citation_valid"] is False for claim in body["guardrail"]["claims"])
        assert any("does not exist" in note for note in body["guardrail"]["notes"])

    def test_full_guardrail_report_is_returned(self, client):
        """The client must be able to show *why* an answer is trustworthy."""
        use_fake(client, "all_supported")
        report = client.post("/ask", json={"question": "What is the age limit for PMJJBY?"}).json()["guardrail"]

        assert len(report["claims"]) == 2
        for claim in report["claims"]:
            assert claim["claim"]
            assert claim["verdict"] in ("SUPPORTED", "NOT_SUPPORTED")
            assert claim["citation_valid"] is True
        assert report["notes"]

    def test_generation_failure_degrades_to_a_refusal(self, client):
        use_fake(client, "all_supported", fail_on="generate_answer")
        body = client.post("/ask", json={"question": "How much money does PM-KISAN give per year?"}).json()
        assert body["status"] == "refused"

    def test_verification_failure_fails_closed(self, client):
        """If the judge cannot run we must refuse, not ship the unverified draft."""
        use_fake(client, "all_supported", fail_on="judge_claims")
        body = client.post("/ask", json={"question": "How much money does PM-KISAN give per year?"}).json()
        assert body["guardrail"]["decision"] == "BLOCK"
        assert body["status"] == "refused"

    # -- input validation --------------------------------------------------

    def test_too_long_question_is_rejected(self, client):
        use_fake(client)
        response = client.post("/ask", json={"question": "a" * 600})
        assert response.status_code == 400
        assert "too long" in response.json()["detail"]

    @pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}])
    def test_blank_or_missing_question_is_rejected(self, client, payload):
        use_fake(client)
        assert client.post("/ask", json=payload).status_code == 422

    def test_language_override_is_respected(self, client):
        use_fake(client, "all_supported")
        body = client.post(
            "/ask",
            json={"question": "How much money does PM-KISAN give per year?", "language": "hi"},
        ).json()
        assert body["language"] == "hi"


# ==========================================================================
# POST /eligibility
# ==========================================================================


class TestEligibility:
    def test_profile_is_extracted_and_every_scheme_is_checked(self, client):
        fake = use_fake(client)
        response = client.post(
            "/eligibility",
            json={"description": "I am a 35 year old farmer from Bihar earning about 2 lakh a year"},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["profile"]["age"] == 35
        assert body["profile"]["occupation"] == "farmer"
        assert len(body["results"]) == 15
        # Exactly one LLM call: extraction. Every verdict is Python.
        assert body["llm_calls"] == 1
        assert fake.calls == ["extract_profile"]

    def test_results_are_ordered_eligible_first(self, client):
        use_fake(client)
        statuses = [r["status"] for r in client.post(
            "/eligibility", json={"description": "35 year old farmer from Bihar, income 2 lakh"}
        ).json()["results"]]
        rank = {"ELIGIBLE": 0, "NEED_MORE_INFO": 1, "NOT_ELIGIBLE": 2}
        assert statuses == sorted(statuses, key=lambda status: rank[status])

    def test_every_verdict_carries_its_reasoning(self, client):
        use_fake(client)
        for result in client.post(
            "/eligibility", json={"description": "35 year old farmer from Bihar, income 2 lakh"}
        ).json()["results"]:
            assert result["checks"], f"{result['scheme_id']} has no reasoning"
            for check in result["checks"]:
                assert check["reason"]
            assert result["source_url"]

    def test_farmer_is_eligible_for_pm_kisan_and_not_for_street_vendors(self, client):
        use_fake(client)
        results = {r["scheme_id"]: r for r in client.post(
            "/eligibility", json={"description": "35 year old farmer from Bihar, income 2 lakh"}
        ).json()["results"]}
        assert results["pm_kisan"]["status"] == "ELIGIBLE"
        assert results["pm_svanidhi"]["status"] == "NOT_ELIGIBLE"

    def test_eligible_results_still_list_conditions_to_verify(self, client):
        """ELIGIBLE must never read as a guarantee."""
        use_fake(client)
        results = {r["scheme_id"]: r for r in client.post(
            "/eligibility", json={"description": "35 year old farmer from Bihar, income 2 lakh"}
        ).json()["results"]}
        assert results["pm_kisan"]["conditions_to_verify"]

    def test_too_long_description_is_rejected(self, client):
        use_fake(client)
        assert client.post("/eligibility", json={"description": "a" * 600}).status_code == 400


# ==========================================================================
# GET /metrics
# ==========================================================================


class TestMetrics:
    def test_metrics_shape(self, client):
        body = client.get("/metrics").json()
        for key in ("llm_calls_total", "avg_latency_ms", "calls_by_prompt",
                    "guardrail_counts", "questions_answered", "questions_refused"):
            assert key in body
        for decision in ("PASS", "REPAIR", "BLOCK", "SKIPPED"):
            assert decision in body["guardrail_counts"]

    def test_guardrail_decisions_are_recorded(self, client):
        """The counters come from the SQLite log written during the tests above."""
        use_fake(client, "all_supported")
        client.post("/ask", json={"question": "How much money does PM-KISAN give per year?"})
        counts = client.get("/metrics").json()["guardrail_counts"]
        assert counts["PASS"] >= 1


# ==========================================================================
# Degraded start
# ==========================================================================


class TestWithoutAnLLM:
    def test_ask_returns_503_when_the_model_is_not_configured(self, client):
        """The container must boot and serve read-only endpoints without a key."""
        original = client.app.state.pipeline
        client.app.state.pipeline = None
        try:
            response = client.post("/ask", json={"question": "What is PM-KISAN?"})
            assert response.status_code == 503
            assert "LLM_API_KEY" in response.json()["detail"]

            # Read-only endpoints keep working.
            assert client.get("/health").status_code == 200
            assert len(client.get("/schemes").json()) == 15
        finally:
            client.app.state.pipeline = original
