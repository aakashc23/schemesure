"""
Headless render tests for the Streamlit UI.

WHY THESE EXIST: every other test covers logic the UI never touches. A broken
Streamlit call — a renamed parameter, a bad dict key, a deprecated API — would
sail through the whole suite and only appear as a stack trace on the live demo.
`st.testing.v1.AppTest` runs the real script in-process, so we can assert it
renders without a single browser or network call.

The API is replaced with a fake `requests` module, so these tests are offline,
instant and safe to run in CI.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UI_PATH = ROOT / "ui" / "streamlit_app.py"

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

# --------------------------------------------------------------------------
# Canned API responses
# --------------------------------------------------------------------------

HEALTH = {
    "status": "ok",
    "schemes_loaded": 15,
    "chunks_indexed": 312,
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "llm_model": "qwen/qwen3.8-27b",
    "llm_key_configured": True,
    "guardrail_enabled": True,
}

SCHEMES = [
    {
        "scheme_id": "pm_kisan",
        "name": "Pradhan Mantri Kisan Samman Nidhi",
        "short_name": "PM-KISAN",
        "ministry": "Ministry Of Agriculture and Farmers Welfare",
        "source_url": "https://www.myscheme.gov.in/schemes/pm-kisan",
        "last_verified": "2026-10-02",
        "sections": ["Overview", "Benefits", "Eligibility"],
        "has_rules": True,
    }
]

ASK_VERIFIED = {
    "question": "How much does PM-KISAN give?",
    "normalized_query": "How much money does PM-KISAN give per year?",
    "language": "en",
    "answer": "PM-KISAN provides Rs. 6000 per annum per family [1].",
    "status": "verified",
    "citations": [
        {
            "index": 1,
            "chunk_id": "pm_kisan#benefits#0",
            "scheme_id": "pm_kisan",
            "scheme_name": "Pradhan Mantri Kisan Samman Nidhi",
            "section": "Benefits",
            "source_url": "https://www.myscheme.gov.in/schemes/pm-kisan",
            "last_verified": "2026-10-02",
        }
    ],
    "guardrail": {
        "decision": "PASS",
        "claims": [
            {
                "claim": "PM-KISAN pays Rs 6000 per year",
                "verdict": "SUPPORTED",
                "evidence_chunk_id": "pm_kisan#benefits#0",
                "citation_valid": True,
                "note": "",
            }
        ],
        "supported_count": 1,
        "unsupported_count": 0,
        "unsupported_ratio": 0.0,
        "block_threshold": 0.5,
        "notes": ["1/1 claims supported by the evidence."],
        "llm_calls": 2,
    },
    "latency_ms": 1800,
    "llm_calls": 4,
}

ELIGIBILITY = {
    "description": "I am a 35 year old farmer from Bihar",
    "profile": {
        "age": 35, "annual_income": 200000, "state": "Bihar",
        "occupation": "farmer", "gender": None, "category": None,
    },
    "results": [
        {
            "scheme_id": "pm_kisan",
            "scheme_name": "Pradhan Mantri Kisan Samman Nidhi",
            "status": "ELIGIBLE",
            "checks": [
                {
                    "rule": "allowed_occupations",
                    "passed": True,
                    "reason": "Your occupation (farmer) is covered by this scheme.",
                    "official_text": "All landholding farmers' families...",
                }
            ],
            "missing_fields": [],
            "conditions_to_verify": ["Must have cultivable land holding."],
            "source_url": "https://www.myscheme.gov.in/schemes/pm-kisan",
        },
        {
            "scheme_id": "pm_matru_vandana",
            "scheme_name": "Pradhan Mantri Matru Vandana Yojana",
            "status": "NEED_MORE_INFO",
            "checks": [
                {
                    "rule": "gender",
                    "passed": None,
                    "reason": "This scheme is only for female applicants. Your gender was not provided.",
                    "official_text": "",
                }
            ],
            "missing_fields": ["gender"],
            "conditions_to_verify": [],
            "source_url": "https://www.myscheme.gov.in/schemes/pmmvy",
        },
    ],
    "eligible_count": 1,
    "need_more_info_count": 1,
    "latency_ms": 900,
    "llm_calls": 1,
}


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode()
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def install_fake_requests(ask_payload=None, health_payload=None, fail: bool = False):
    """
    Replace the `requests` module the UI imports.

    AppTest runs the script in-process, so swapping sys.modules["requests"]
    before the run is enough — the UI's `import requests` resolves to this.
    """
    import requests as real_requests

    module = types.ModuleType("requests")
    module.exceptions = real_requests.exceptions

    def get(url, timeout=None, **kwargs):
        if fail:
            raise real_requests.exceptions.ConnectionError("simulated outage")
        if url.endswith("/health"):
            return FakeResponse(health_payload or HEALTH)
        if url.endswith("/schemes"):
            return FakeResponse(SCHEMES)
        return FakeResponse({}, 404)

    def post(url, json=None, timeout=None, **kwargs):
        if fail:
            raise real_requests.exceptions.ConnectionError("simulated outage")
        if url.endswith("/ask"):
            return FakeResponse(ask_payload or ASK_VERIFIED)
        if url.endswith("/eligibility"):
            return FakeResponse(ELIGIBILITY)
        return FakeResponse({}, 404)

    module.get = get
    module.post = post
    sys.modules["requests"] = module
    return module


@pytest.fixture(autouse=True)
def isolate_each_run():
    """
    Restore the real `requests` module, and clear Streamlit's caches.

    The cache clearing matters: the UI wraps `/health` and `/schemes` in
    `@st.cache_data`, and AppTest runs share one process — so without this, a
    later test asserting on a *degraded* health response silently gets the
    healthy one cached by an earlier test and the assertion fails for the wrong
    reason. (Found exactly that way.)
    """
    import streamlit as st

    st.cache_data.clear()
    original = sys.modules.get("requests")
    yield
    st.cache_data.clear()
    if original is not None:
        sys.modules["requests"] = original


def run_ui(**kwargs) -> AppTest:
    install_fake_requests(**kwargs)
    app = AppTest.from_file(str(UI_PATH), default_timeout=60)
    app.run()
    return app


# ==========================================================================
# Tests
# ==========================================================================


class TestUIRenders:
    def test_the_page_renders_without_raising(self):
        """The baseline: a Streamlit API misuse would surface here, not in prod."""
        app = run_ui()
        assert not app.exception, f"UI raised: {app.exception}"

    def test_header_and_all_three_tabs_are_present(self):
        app = run_ui()
        assert any("SchemeSure" in title.value for title in app.title)
        labels = [tab.label for tab in app.tabs]
        assert len(labels) == 3
        joined = " ".join(labels)
        assert "Ask" in joined and "eligibility" in joined and "Schemes" in joined

    def test_scheme_catalogue_is_rendered(self):
        app = run_ui()
        assert app.dataframe, "the Schemes tab rendered no table"

    def test_disclaimer_is_always_shown(self):
        """A non-negotiable: the footer must state this is not an official service."""
        app = run_ui()
        captions = " ".join(c.value for c in app.caption)
        assert "not" in captions.lower() and "official" in captions.lower()
        assert "Disclaimer" in captions

    def test_backend_outage_is_reported_not_crashed(self):
        app = run_ui(fail=True)
        assert not app.exception
        errors = " ".join(e.value for e in app.error)
        assert "not reachable" in errors or "Could not reach" in errors

    def test_missing_llm_key_warns_but_still_renders(self):
        degraded = dict(HEALTH, llm_key_configured=False)
        app = run_ui(health_payload=degraded)
        assert not app.exception
        warnings = " ".join(w.value for w in app.warning)
        assert "api key" in warnings.lower() or "key" in warnings.lower()


class TestAskTab:
    def _ask(self, payload) -> AppTest:
        app = run_ui(ask_payload=payload)
        app.text_area[0].set_value("How much does PM-KISAN give?")
        app.button[0].click().run()
        return app

    def test_verified_answer_shows_badge_answer_and_citations(self):
        app = self._ask(ASK_VERIFIED)
        assert not app.exception

        successes = " ".join(s.value for s in app.success)
        assert "Verified" in successes

        body = " ".join(m.value for m in app.markdown)
        assert "6000" in body
        # The citation links back to the official source.
        assert "myscheme.gov.in" in body
        # The verification panel exists and names the claim.
        assert "PM-KISAN pays Rs 6000 per year" in body

    def test_refused_answer_shows_the_refusal_badge(self):
        refused = dict(
            ASK_VERIFIED,
            status="refused",
            answer="I do not have official information on this.",
            citations=[],
            guardrail=dict(
                ASK_VERIFIED["guardrail"],
                decision="SKIPPED", claims=[],
                notes=["Best retrieval score 0.109 is below the 0.45 threshold."],
            ),
        )
        app = self._ask(refused)
        assert not app.exception
        errors = " ".join(e.value for e in app.error)
        assert "Not enough official info" in errors

    def test_partially_verified_shows_the_warning_badge(self):
        repaired = dict(
            ASK_VERIFIED,
            status="partially_verified",
            guardrail=dict(
                ASK_VERIFIED["guardrail"],
                decision="REPAIR",
                claims=[
                    ASK_VERIFIED["guardrail"]["claims"][0],
                    {
                        "claim": "An invented extra fact",
                        "verdict": "NOT_SUPPORTED",
                        "evidence_chunk_id": None,
                        "citation_valid": True,
                        "note": "",
                    },
                ],
                supported_count=1, unsupported_count=1, unsupported_ratio=0.5,
            ),
        )
        app = self._ask(repaired)
        assert not app.exception
        warnings = " ".join(w.value for w in app.warning)
        assert "Partially verified" in warnings

    def test_fabricated_citation_is_surfaced_to_the_user(self):
        """The guardrail's most interesting catch must be visible, not just logged."""
        fabricated = dict(
            ASK_VERIFIED,
            status="refused",
            answer="I could not verify a reliable answer.",
            citations=[],
            guardrail=dict(
                ASK_VERIFIED["guardrail"],
                decision="BLOCK",
                claims=[
                    {
                        "claim": "An unsupported claim",
                        "verdict": "NOT_SUPPORTED",
                        "evidence_chunk_id": "made_up#x#9",
                        "citation_valid": False,
                        "note": "Cited chunk 'made_up#x#9' is not among the retrieved passages.",
                    }
                ],
                supported_count=0, unsupported_count=1, unsupported_ratio=1.0,
                notes=["1 claim(s) cited an evidence chunk that does not exist."],
            ),
        )
        app = self._ask(fabricated)
        assert not app.exception
        shown = " ".join(m.value for m in app.markdown) + " ".join(
            c.value for c in app.caption
        )
        assert "does not exist" in shown
