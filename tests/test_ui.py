"""
Headless render tests for the Streamlit UI.

WHY THESE EXIST: every other test covers logic the UI never touches. A broken
Streamlit call — a renamed parameter, a bad dict key, a deprecated API — would
sail through the whole suite and only appear as a stack trace on the live demo.
`st.testing.v1.AppTest` runs the real script in-process, so we can assert it
renders without a single browser or network call.

The API is replaced with a fake `requests` module, so these tests are offline,
instant and safe to run in CI. The fake records every call, which lets tests
assert on what the UI *sent* (for example that a form with an empty field posts
`null`, never a default value).

Two response shapes are tested on purpose: the full new shape, and an "old
shape" without the newer fields (evidence, trace, refusal_reason ...), because
the UI promises to keep working against an older backend.
"""

from __future__ import annotations

import copy
import json
import re
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
    "llm_ready": True,
    "guardrail_enabled": True,
    "retrieval_top_k": 5,
    "retrieval_min_score": 0.45,
    "guardrail_block_threshold": 0.5,
    "answer_cache_enabled": True,
}

# What an older server sends: none of the newer fields.
OLD_HEALTH = {
    key: HEALTH[key]
    for key in (
        "status", "schemes_loaded", "chunks_indexed", "embedding_model",
        "llm_model", "llm_key_configured", "guardrail_enabled",
    )
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

SCHEME_DETAIL = dict(
    SCHEMES[0],
    sections_detail=[
        {"title": "Benefits", "text": "Rs. 6000 per year in three instalments."},
        {"title": "Eligibility", "text": "All landholding farmer families."},
    ],
    reference_urls=[{"title": "Scheme Guidelines", "url": "https://pmkisan.gov.in/guidelines.pdf"}],
    rules={
        "min_age": None,
        "max_age": None,
        "max_annual_income": None,
        "allowed_states": None,
        "allowed_occupations": ["farmer"],
        "gender": None,
        "category": None,
        "rule_sources": {"allowed_occupations": "All landholding farmers' families are eligible."},
        "unmodelled_conditions": ["Must hold cultivable land in own name."],
    },
    chunks_indexed=21,
)

OPTIONS = {
    "occupations": ["farmer", "student", "street_vendor"],
    "genders": ["female", "male", "other"],
    "categories": ["General", "OBC", "SC", "ST", "EWS"],
}

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
                "evidence_similarity": 0.8123,
                "best_similarity": 0.8123,
                "best_similarity_chunk_id": "pm_kisan#benefits#0",
            }
        ],
        "supported_count": 1,
        "unsupported_count": 0,
        "unsupported_ratio": 0.0,
        "block_threshold": 0.5,
        "notes": ["1/1 claims supported by the evidence."],
        "llm_calls": 2,
        "ungrounded_tokens": [],
    },
    "latency_ms": 1800,
    "llm_calls": 4,
    "refusal_reason": None,
    "normalization_fallback": False,
    "cached": False,
    "retrieval": {
        "query": "How much money does PM-KISAN give per year?",
        "top_k": 5,
        "threshold": 0.45,
        "best_score": 0.712,
        "low_evidence": False,
        "passages": [
            {
                "index": 1,
                "chunk_id": "pm_kisan#benefits#0",
                "scheme_id": "pm_kisan",
                "scheme_name": "Pradhan Mantri Kisan Samman Nidhi",
                "section": "Benefits",
                "source_url": "https://www.myscheme.gov.in/schemes/pm-kisan",
                "score": 0.712,
                "above_threshold": True,
                "text": "Rs. 6000 per year in three equal instalments.",
            },
            {
                "index": 2,
                "chunk_id": "pm_kisan#overview#0",
                "scheme_id": "pm_kisan",
                "scheme_name": "Pradhan Mantri Kisan Samman Nidhi",
                "section": "Overview",
                "source_url": "https://www.myscheme.gov.in/schemes/pm-kisan",
                "score": 0.401,
                "above_threshold": False,
                "text": "An income support scheme for farmers.",
            },
        ],
    },
    "trace": [
        {"stage": "normalize", "duration_ms": 310, "llm_calls": 1, "outcome": "language=en"},
        {"stage": "retrieve", "duration_ms": 40, "llm_calls": 0, "outcome": "best 0.712 >= 0.45"},
        {"stage": "generate", "duration_ms": 700, "llm_calls": 1, "outcome": "answer drafted"},
    ],
    "repair_report": None,
}

# The same answer as an old server would send it.
ASK_OLD_SHAPE = {
    key: ASK_VERIFIED[key]
    for key in (
        "question", "normalized_query", "language", "answer", "status",
        "citations", "guardrail", "latency_ms", "llm_calls",
    )
}
ASK_OLD_SHAPE = copy.deepcopy(ASK_OLD_SHAPE)
for _claim in ASK_OLD_SHAPE["guardrail"]["claims"]:
    for _field in ("evidence_similarity", "best_similarity", "best_similarity_chunk_id"):
        _claim.pop(_field)
ASK_OLD_SHAPE["guardrail"].pop("ungrounded_tokens")

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
        {
            "scheme_id": "nmmss",
            "scheme_name": "National Means-cum-Merit Scholarship",
            "status": "NOT_ELIGIBLE",
            "checks": [
                {
                    "rule": "allowed_occupations",
                    "passed": False,
                    "reason": "This scheme is for students; your occupation is farmer.",
                    "official_text": "Students of class VIII.",
                }
            ],
            "missing_fields": [],
            "conditions_to_verify": [],
            "source_url": "https://www.myscheme.gov.in/schemes/nmmss",
        },
    ],
    "eligible_count": 1,
    "need_more_info_count": 1,
    "not_eligible_count": 1,
    "latency_ms": 900,
    "llm_calls": 1,
}

# /eligibility/check: same shape, but no description and no LLM calls.
ELIGIBILITY_CHECK = dict(ELIGIBILITY, description="", llm_calls=0)

METRICS = {
    "llm_calls_total": 42,
    "llm_calls_failed": 3,
    "avg_latency_ms": 812.4,
    "p95_latency_ms": 1500.0,
    "total_prompt_tokens": 12000,
    "total_completion_tokens": 3000,
    "calls_by_prompt": {"judge": 20, "generate": 10},
    "by_prompt": {
        "judge": {
            "calls": 20, "failed": 1, "avg_latency_ms": 700,
            "prompt_tokens": 8000, "completion_tokens": 900,
        },
        "generate": {
            "calls": 10, "failed": 0, "avg_latency_ms": 900,
            "prompt_tokens": 4000, "completion_tokens": 2100,
        },
    },
    "guardrail_counts": {"PASS": 7, "REPAIR": 2, "BLOCK": 1, "SKIPPED": 0},
    "answer_status_counts": {"verified": 7, "refused": 3},
    "cache_hits": 4,
    "log_started_at": "2026-11-15 10:00:00",
    "questions_answered": 7,
    "questions_refused": 3,
}


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode()
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeApi:
    """
    A stand-in for the `requests` module the UI imports, with a call log.

    `routes` maps (METHOD, path) to a payload, a (status, payload) pair, or an
    Exception instance to raise (to simulate a timeout or an outage).
    """

    def __init__(self, routes: dict):
        import requests as real_requests

        self.routes = routes
        self.calls: list[tuple[str, str, dict | None]] = []
        self.module = types.ModuleType("requests")
        self.module.exceptions = real_requests.exceptions
        self.module.get = lambda url, timeout=None, **kw: self._handle("GET", url, None)
        self.module.post = lambda url, json=None, timeout=None, **kw: self._handle("POST", url, json)

    def _handle(self, method: str, url: str, body):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        self.calls.append((method, path, body))
        route = self.routes.get((method, path))
        if isinstance(route, Exception):
            raise route
        if route is None:
            return FakeResponse({"detail": "not found"}, 404)
        if isinstance(route, tuple):
            status, payload = route
            return FakeResponse(payload, status)
        return FakeResponse(route)

    def posts(self, path: str) -> list[dict]:
        return [body for method, p, body in self.calls if method == "POST" and p == path]


def default_routes(overrides: dict | None = None) -> dict:
    routes = {
        ("GET", "/health"): HEALTH,
        ("GET", "/schemes"): SCHEMES,
        ("GET", "/schemes/pm_kisan"): SCHEME_DETAIL,
        ("GET", "/eligibility/options"): OPTIONS,
        ("GET", "/metrics"): METRICS,
        ("POST", "/ask"): ASK_VERIFIED,
        ("POST", "/eligibility"): ELIGIBILITY,
        ("POST", "/eligibility/check"): ELIGIBILITY_CHECK,
    }
    routes.update(overrides or {})
    return routes


@pytest.fixture(autouse=True)
def isolate_each_run():
    """
    Restore the real `requests` module, and clear Streamlit's caches.

    The cache clearing matters: the UI wraps `/health`, `/schemes` and friends in
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


def run_ui(routes: dict | None = None, overrides: dict | None = None):
    """
    Run the UI once against a fake API. Returns (AppTest, FakeApi).

    `routes` replaces the whole route table; `overrides` changes single routes
    of the default table (e.g. make /ask return a 503).
    """
    api = FakeApi(routes if routes is not None else default_routes(overrides))
    # AppTest runs the script in-process, so swapping sys.modules["requests"]
    # before the run is enough — the UI's `import requests` resolves to this.
    sys.modules["requests"] = api.module
    app = AppTest.from_file(str(UI_PATH), default_timeout=60)
    app.run()
    return app, api


def ask(app: AppTest, text: str = "How much does PM-KISAN give?") -> AppTest:
    """Type a question into the Ask form and press its submit button."""
    app.text_area[0].set_value(text)
    return app.button(key="ask_submit").click().run()


def asked(payload, health=None) -> tuple[AppTest, FakeApi]:
    """Run the UI, submit a question through the form, return the result."""
    routes = default_routes()
    routes[("POST", "/ask")] = payload
    if health is not None:
        routes[("GET", "/health")] = health
    app, api = run_ui(routes)
    ask(app)
    return app, api


def all_text(app: AppTest) -> str:
    """Every piece of rendered text, flattened (markdown, captions, alerts, metrics)."""
    parts = [m.value for m in app.markdown]
    parts += [c.value for c in app.caption]
    parts += [e.value for e in app.error] + [w.value for w in app.warning]
    parts += [i.value for i in app.info] + [s.value for s in app.success]
    parts += [str(m.value) for m in app.metric]
    return "\n".join(parts)


def frame_text(app: AppTest) -> str:
    """Every dataframe's cells, flattened into one string."""
    return "\n".join(df.value.to_string() for df in app.dataframe)


def tab_labels(app: AppTest) -> list[str]:
    return [tab.label for tab in app.tabs]


# ==========================================================================
# Page-level tests
# ==========================================================================


class TestUIRenders:
    def test_the_page_renders_without_raising(self):
        """The baseline: a Streamlit API misuse would surface here, not in prod."""
        app, _ = run_ui()
        assert not app.exception, f"UI raised: {app.exception}"

    def test_header_and_all_five_tabs_are_present(self):
        app, _ = run_ui()
        assert any("SchemeSure" in title.value for title in app.title)
        top_level = tab_labels(app)[:5]
        assert top_level == ["Ask", "Eligibility", "Schemes", "How it works", "System"]

    def test_header_explains_the_three_steps(self):
        app, _ = run_ui()
        text = all_text(app)
        assert "Retrieve official text" in text
        assert "Draft an answer" in text
        assert "Verify every claim" in text

    def test_scheme_catalogue_is_rendered(self):
        app, _ = run_ui()
        assert "PM-KISAN" in frame_text(app), "the Schemes tab rendered no catalogue"

    def test_disclaimer_is_always_shown(self):
        """A non-negotiable: the footer must state this is not an official service."""
        app, _ = run_ui()
        captions = " ".join(c.value for c in app.caption)
        assert "not" in captions.lower() and "official" in captions.lower()
        assert "Disclaimer" in captions

    def test_internal_backend_url_is_not_in_the_footer(self):
        app, _ = run_ui()
        captions = " ".join(c.value for c in app.caption)
        assert "localhost:8000" not in captions and "127.0.0.1" not in captions

    def test_sidebar_shows_system_status_and_settings(self):
        app, _ = run_ui()
        sidebar = " ".join(m.value for m in app.sidebar.markdown)
        assert "reachable" in sidebar
        assert "qwen/qwen3.8-27b" in sidebar
        assert "Answer cache" in sidebar
        assert app.sidebar.selectbox(key="answer_language").options == [
            "Auto-detect", "English", "Hindi", "Hinglish",
        ]

    def test_no_flag_emoji_anywhere_in_the_source(self):
        """Windows renders the India flag as the letters 'IN' (seen on the live app)."""
        source = UI_PATH.read_text(encoding="utf-8")
        flags = [ch for ch in source if 0x1F1E6 <= ord(ch) <= 0x1F1FF]
        assert not flags

    def test_backend_outage_is_reported_not_crashed(self):
        app, _ = run_ui(
            {("GET", "/health"): __import__("requests").exceptions.ConnectionError("simulated outage")}
        )
        assert not app.exception
        errors = " ".join(e.value for e in app.error)
        assert "not reachable" in errors

    def test_missing_llm_key_warns_disables_ask_but_keeps_the_rest(self):
        degraded = dict(HEALTH, llm_key_configured=False, llm_ready=False)
        app, _ = run_ui(overrides={("GET", "/health"): degraded})
        assert not app.exception
        warnings = " ".join(w.value for w in app.warning)
        assert "key" in warnings.lower()
        assert app.button(key="ask_submit").disabled
        assert app.button(key="example_0").disabled
        # Schemes and form-based eligibility still render.
        assert "PM-KISAN" in frame_text(app)
        # The form-based eligibility check needs no model, so it stays enabled.
        app.radio(key="elig_mode").set_value("Fill in a form").run()
        assert not app.button(key="elig_submit").disabled

    def test_llm_ready_false_disables_ask_even_if_a_key_is_present(self):
        """`llm_ready` (pipeline actually built) wins over `llm_key_configured`."""
        app, _ = run_ui(overrides={("GET", "/health"): dict(HEALTH, llm_ready=False)})
        assert app.button(key="ask_submit").disabled

    def test_old_health_without_llm_ready_falls_back_to_the_key_flag(self):
        app, _ = run_ui(overrides={("GET", "/health"): OLD_HEALTH})
        assert not app.exception
        assert not app.button(key="ask_submit").disabled


# ==========================================================================
# Ask tab
# ==========================================================================


class TestAskTab:
    def test_verified_answer_shows_pill_answer_and_sources(self):
        app, _ = asked(ASK_VERIFIED)
        assert not app.exception
        text = all_text(app)
        assert "Verified" in text and "green-badge" in text
        assert "6000" in text
        # The source links back to the official page and names the passage.
        assert "myscheme.gov.in/schemes/pm-kisan" in text
        assert "pm_kisan#benefits#0" in text

    def test_the_question_is_posted_with_auto_language_as_null(self):
        app, api = asked(ASK_VERIFIED)
        assert api.posts("/ask") == [
            {"question": "How much does PM-KISAN give?", "language": None}
        ]

    def test_answer_language_setting_is_sent(self):
        app, api = run_ui()
        app.sidebar.selectbox(key="answer_language").select("Hindi").run()
        ask(app)
        assert api.posts("/ask")[0]["language"] == "hi"

    def test_example_button_asks_immediately(self):
        app, api = run_ui()
        app.button(key="example_1").click().run()
        assert not app.exception
        posted = api.posts("/ask")
        assert len(posted) == 1 and "अटल पेंशन" in posted[0]["question"]
        assert "PM-KISAN provides Rs. 6000" in all_text(app)

    def test_refused_low_evidence_shows_red_pill_and_the_reason(self):
        refused = dict(
            ASK_VERIFIED,
            status="refused",
            refusal_reason="low_evidence",
            answer="I do not have official information on this.",
            citations=[],
            guardrail=dict(
                ASK_VERIFIED["guardrail"], decision="SKIPPED", claims=[],
                notes=["Best retrieval score 0.109 is below the 0.45 threshold."],
            ),
            retrieval=dict(
                ASK_VERIFIED["retrieval"], best_score=0.109, low_evidence=True,
                passages=[dict(ASK_VERIFIED["retrieval"]["passages"][1], score=0.109)],
            ),
        )
        app, _ = asked(refused)
        assert not app.exception
        text = all_text(app)
        assert "Not enough official info" in text and "red-badge" in text
        assert "No official passage was similar enough" in text
        # The evidence tab shows what was found and why it was not enough.
        assert "0.109" in text
        assert "nothing was drafted" in text

    def test_no_claims_refusal_is_not_labelled_pass(self):
        """Found on the live app: a refused draft with no claims read "Decision: PASS"."""
        refused = dict(
            ASK_VERIFIED,
            status="refused",
            refusal_reason="no_claims",
            answer="I do not have official information on this.",
            citations=[],
            guardrail=dict(
                ASK_VERIFIED["guardrail"], decision="PASS", claims=[], supported_count=0,
                notes=["No factual claims were extracted; nothing to verify."],
            ),
        )
        app, _ = asked(refused)
        assert not app.exception
        text = all_text(app)
        assert "Decision: PASS" not in text
        assert "nothing to verify — answer withheld" in text

    def test_old_style_refusal_without_reason_still_gets_the_red_pill(self):
        refused = dict(
            ASK_OLD_SHAPE, status="refused", answer="I do not have official information on this.",
            citations=[],
        )
        app, _ = asked(refused)
        assert not app.exception
        assert "red-badge" in all_text(app)

    def test_service_unavailable_is_grey_and_says_it_is_temporary(self):
        unavailable = dict(
            ASK_VERIFIED,
            status="refused",
            refusal_reason="service_unavailable",
            answer="I could not verify an answer right now.",
            citations=[],
            guardrail=dict(ASK_VERIFIED["guardrail"], decision="BLOCK", claims=[]),
        )
        app, _ = asked(unavailable)
        assert not app.exception
        text = all_text(app)
        assert "Verification unavailable" in text and "gray-badge" in text
        assert "red-badge" not in text
        assert "temporary" in text and "does NOT mean the information is missing" in text

    def test_partially_verified_shows_amber_pill_and_the_repair_recheck(self):
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
                        "evidence_similarity": None,
                    },
                ],
                supported_count=1, unsupported_count=1, unsupported_ratio=0.5,
            ),
            repair_report=dict(ASK_VERIFIED["guardrail"], decision="PASS"),
        )
        app, _ = asked(repaired)
        assert not app.exception
        text = all_text(app)
        assert "Partially verified" in text and "orange-badge" in text
        assert "The repaired answer was re-checked: 1/1 claims supported" in text
        assert "An invented extra fact" in frame_text(app)

    def test_unverified_eval_mode_is_grey(self):
        app, _ = asked(dict(ASK_VERIFIED, status="unverified"))
        text = all_text(app)
        assert "Unverified" in text and "gray-badge" in text

    def test_cached_and_fallback_flags_show_in_the_meta_line(self):
        app, _ = asked(dict(ASK_VERIFIED, cached=True, normalization_fallback=True))
        text = all_text(app)
        assert "served from cache" in text
        assert "language model fallback" in text

    def test_meta_line_shows_language_calls_and_latency(self):
        app, _ = asked(dict(ASK_VERIFIED, language="hinglish"))
        text = all_text(app)
        assert "Language: Hinglish" in text and "4 LLM call(s)" in text and "1.8 s" in text

    def test_claims_table_has_similarity_and_the_diagnostic_sentence(self):
        app, _ = asked(ASK_VERIFIED)
        frames = frame_text(app)
        assert "PM-KISAN pays Rs 6000 per year" in frames
        assert "Similarity to cited evidence" in frames and "0.81" in frames
        assert "not used for the decision yet" in all_text(app)

    def test_similarity_none_is_shown_as_a_dash(self):
        claim = dict(ASK_VERIFIED["guardrail"]["claims"][0], evidence_similarity=None)
        payload = dict(
            ASK_VERIFIED, guardrail=dict(ASK_VERIFIED["guardrail"], claims=[claim])
        )
        app, _ = asked(payload)
        claims_frame = next(
            df.value for df in app.dataframe if "Claim" in df.value.columns
        )
        assert claims_frame["Similarity to cited evidence"].iloc[0] == "—"

    def test_old_shape_response_still_renders_everything_it_can(self):
        app, _ = asked(ASK_OLD_SHAPE)
        assert not app.exception
        text = all_text(app)
        assert "Verified" in text and "6000" in text
        frames = frame_text(app)
        assert "PM-KISAN pays Rs 6000 per year" in frames
        assert "Similarity to cited evidence" not in frames
        # Missing evidence / trace are explained, not crashed on.
        assert "did not send retrieval details" in text
        assert "did not send a per-stage trace" in text

    def test_evidence_tab_lists_passages_with_cited_marks_and_progress_scores(self):
        app, _ = asked(ASK_VERIFIED)
        evidence = next(df.value for df in app.dataframe if "Above threshold" in df.value.columns)
        assert list(evidence["Cited"]) == ["yes", "no"]
        assert list(evidence["Above threshold"]) == ["yes", "no"]
        assert evidence["Score"].iloc[0] == pytest.approx(0.712)
        metrics = {m.label: m.value for m in app.metric}
        assert metrics["Threshold"] == "0.45" and metrics["Best score"] == "0.712"
        assert "cited in the answer" in " ".join(e.label for e in app.expander)

    def test_pipeline_tab_shows_the_stage_trace_and_the_searched_query(self):
        app, _ = asked(ASK_VERIFIED)
        frames = frame_text(app)
        assert "normalize" in frames and "best 0.712 >= 0.45" in frames
        assert any(
            "How much money does PM-KISAN give per year?" in code.value for code in app.code
        )

    def test_fabricated_citation_is_surfaced_to_the_user(self):
        """The guardrail's most interesting catch must be visible, not just logged."""
        fabricated = dict(
            ASK_VERIFIED,
            status="refused",
            refusal_reason="not_supported",
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
        app, _ = asked(fabricated)
        assert not app.exception
        shown = all_text(app)
        assert "does not exist" in shown
        assert "made_up#x#9" in frame_text(app)

    def test_ungrounded_numbers_and_links_are_listed(self):
        payload = dict(
            ASK_VERIFIED,
            guardrail=dict(ASK_VERIFIED["guardrail"], ungrounded_tokens=["9999", "example.com"]),
        )
        app, _ = asked(payload)
        text = all_text(app)
        assert "not found in the official text" in text
        assert "9999" in text and "example.com" in text

    def test_history_is_newest_first_and_can_be_cleared(self):
        app, _ = run_ui()
        ask(app, "first question")
        ask(app, "second question")
        questions = [m.value for m in app.markdown if m.value.startswith("**Question:**")]
        assert "second question" in questions[0] and "first question" in questions[1]
        app.button(key="clear_conversation").click().run()
        assert not [m for m in app.markdown if m.value.startswith("**Question:**")]
        assert "No questions yet" in all_text(app)

    def test_blank_question_warns_and_makes_no_request(self):
        app, api = run_ui()
        ask(app, "   ")
        assert "Please type a question first" in " ".join(w.value for w in app.warning)
        assert api.posts("/ask") == []

    def test_a_question_is_shown_literally_not_as_markdown(self):
        app, _ = asked(ASK_VERIFIED)
        ask(app, "what is *bold* [x]?")
        assert any("what is \\*bold\\* \\[x\\]?" in m.value for m in app.markdown)

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (400, "not accepted"),
            (502, "could not be reached"),
            (503, "not configured or not available"),
        ],
    )
    def test_http_errors_become_friendly_messages(self, status, expected):
        app, _ = run_ui(overrides={("POST", "/ask"): (status, {"detail": "boom with secrets"})})
        ask(app)
        assert not app.exception
        errors = " ".join(e.value for e in app.error)
        assert expected in errors
        if status == 502:
            assert "boom" not in errors  # provider text is never shown for a 502

    def test_timeout_is_reported_as_busy_not_crashed(self):
        import requests

        app, _ = run_ui(overrides={("POST", "/ask"): requests.exceptions.Timeout("slow")})
        ask(app)
        assert not app.exception
        assert "timed out" in " ".join(e.value for e in app.error)

    def test_connection_error_mid_session_is_friendly(self):
        import requests

        app, _ = run_ui(overrides={("POST", "/ask"): requests.exceptions.ConnectionError("down")})
        ask(app)
        assert not app.exception
        assert "Could not reach" in " ".join(e.value for e in app.error)


# ==========================================================================
# Eligibility tab
# ==========================================================================


class TestEligibilityTab:
    def test_free_text_flow_shows_counts_groups_and_rule_tables(self):
        app, api = run_ui()
        app.text_area[1].set_value("I am a 35 year old farmer from Bihar")
        app.button(key="elig_text_submit").click().run()
        assert not app.exception
        assert api.posts("/eligibility") == [{"description": "I am a 35 year old farmer from Bihar"}]

        counts = {m.label: m.value for m in app.metric}
        assert counts["Likely eligible"] == "1"
        assert counts["Need more info"] == "1"
        assert counts["Not eligible"] == "1"

        text = all_text(app)
        assert "guidance, not a decision" in text
        assert "Still to check yourself" in text and "cultivable land" in text
        frames = frame_text(app)
        assert "Occupation" in frames and "✓" in frames and "✗" in frames and "?" in frames
        labels = " ".join(e.label for e in app.expander)
        assert "Pradhan Mantri Kisan Samman Nidhi" in labels

    def test_free_text_shows_an_editable_what_we_understood_form(self):
        app, _ = run_ui()
        app.text_area[1].set_value("farmer from Bihar")
        app.button(key="elig_text_submit").click().run()
        assert app.number_input(key="understood_age_1").value == 35
        assert app.number_input(key="understood_income_1").value == 200000
        assert app.text_input(key="understood_state_1").value == "Bihar"
        assert app.selectbox(key="understood_occupation_1").value == "farmer"
        assert app.selectbox(key="understood_gender_1").value == "Not specified"

    def test_recheck_posts_the_edited_profile_to_the_check_endpoint(self):
        app, api = run_ui()
        app.text_area[1].set_value("farmer from Bihar")
        app.button(key="elig_text_submit").click().run()
        app.selectbox(key="understood_gender_1").select("female")
        app.button(key="understood_submit").click().run()
        assert not app.exception
        posted = api.posts("/eligibility/check")
        assert len(posted) == 1
        assert posted[0]["profile"]["gender"] == "female"
        assert posted[0]["profile"]["age"] == 35
        assert posted[0]["profile"]["category"] is None
        assert len(api.posts("/eligibility")) == 1  # no second LLM extraction

    def test_503_on_free_text_suggests_the_form_mode(self):
        app, _ = run_ui(overrides={("POST", "/eligibility"): (503, {"detail": "no key"})})
        app.text_area[1].set_value("farmer")
        app.button(key="elig_text_submit").click().run()
        assert not app.exception
        errors = " ".join(e.value for e in app.error)
        assert "Fill in a form" in errors

    def test_free_text_is_disabled_without_a_model_but_the_form_works(self):
        app, api = run_ui(overrides={("GET", "/health"): dict(HEALTH, llm_ready=False, llm_key_configured=False)})
        assert app.button(key="elig_text_submit").disabled
        app.radio(key="elig_mode").set_value("Fill in a form").run()
        assert not app.button(key="elig_submit").disabled

    def test_form_flow_posts_none_for_every_empty_field(self):
        app, api = run_ui()
        app.radio(key="elig_mode").set_value("Fill in a form").run()
        app.number_input(key="elig_age_0").set_value(30)
        app.button(key="elig_submit").click().run()
        assert not app.exception
        assert api.posts("/eligibility/check") == [
            {
                "profile": {
                    "age": 30, "annual_income": None, "state": None,
                    "occupation": None, "gender": None, "category": None,
                }
            }
        ]
        assert api.posts("/eligibility") == []  # the form path never uses the model
        assert {m.label: m.value for m in app.metric}["LLM calls used"] == "0"

    def test_form_flow_sends_the_chosen_values(self):
        app, api = run_ui()
        app.radio(key="elig_mode").set_value("Fill in a form").run()
        app.number_input(key="elig_age_0").set_value(22)
        app.number_input(key="elig_income_0").set_value(150000)
        app.text_input(key="elig_state_0").set_value("Bihar")
        app.selectbox(key="elig_occupation_0").select("student")
        app.selectbox(key="elig_gender_0").select("female")
        app.selectbox(key="elig_category_0").select("SC")
        app.button(key="elig_submit").click().run()
        profile = api.posts("/eligibility/check")[0]["profile"]
        assert profile == {
            "age": 22, "annual_income": 150000, "state": "Bihar",
            "occupation": "student", "gender": "female", "category": "SC",
        }

    def test_empty_form_does_not_call_the_api(self):
        app, api = run_ui()
        app.radio(key="elig_mode").set_value("Fill in a form").run()
        app.button(key="elig_submit").click().run()
        assert api.posts("/eligibility/check") == []
        assert "at least one detail" in " ".join(w.value for w in app.warning)

    def test_form_still_renders_if_the_options_endpoint_is_missing(self):
        routes = default_routes()
        del routes[("GET", "/eligibility/options")]
        app, _ = run_ui(routes)
        app.radio(key="elig_mode").set_value("Fill in a form").run()
        assert not app.exception
        # Occupation falls back to a free-text box.
        assert app.text_input(key="elig_occupation_0") is not None

    def test_income_copy_says_yearly_household_income_in_rupees(self):
        app, _ = run_ui()
        assert "total yearly household / family income in rupees" in all_text(app)


# ==========================================================================
# Schemes tab
# ==========================================================================


class TestSchemesTab:
    def test_choosing_a_scheme_shows_its_detail_view(self):
        app, _ = run_ui()
        app.selectbox(key="scheme_choice").select("pm_kisan").run()
        assert not app.exception
        text = all_text(app)
        assert "Official scheme page" in text
        assert "Scheme Guidelines" in text and "pmkisan.gov.in/guidelines.pdf" in text
        assert "21 indexed passage(s)" in text
        assert "Conditions not modelled by the rule engine" in text
        assert "Must hold cultivable land in own name." in text
        labels = [e.label for e in app.expander]
        assert "Benefits" in labels and "Eligibility" in labels
        rules = next(df.value for df in app.dataframe if "Official sentence" in df.value.columns)
        occupation = rules[rules["Rule"] == "Occupation"].iloc[0]
        assert occupation["Restriction"] == "farmer"
        assert "landholding farmers" in occupation["Official sentence"]
        assert rules[rules["Rule"] == "Gender"].iloc[0]["Restriction"] == "No restriction"

    def test_old_server_without_scheme_detail_degrades_politely(self):
        routes = default_routes()
        del routes[("GET", "/schemes/pm_kisan")]
        app, _ = run_ui(routes)
        app.selectbox(key="scheme_choice").select("pm_kisan").run()
        assert not app.exception
        assert "does not provide per-scheme details" in all_text(app)

    def test_scheme_detail_server_error_is_shown(self):
        app, _ = run_ui(overrides={("GET", "/schemes/pm_kisan"): (500, {"detail": "oops"})})
        app.selectbox(key="scheme_choice").select("pm_kisan").run()
        assert not app.exception
        assert "Something went wrong" in " ".join(e.value for e in app.error)


# ==========================================================================
# How it works and System tabs
# ==========================================================================


class TestHowItWorksTab:
    def test_two_diagrams_are_rendered_with_distinct_llm_and_python_colours(self):
        app, _ = run_ui()
        assert not app.exception
        source = UI_PATH.read_text(encoding="utf-8")
        # LLM steps (amber) and Python steps (blue) use different fill colours.
        assert "#FDEBD0" in source and "#D6EAF8" in source
        for outcome in ("PASS", "REPAIR", "BLOCK", "LOW_EVIDENCE"):
            assert outcome in source
        assert len(app.get("graphviz_chart")) == 2

    def test_stage_text_uses_live_config_values(self):
        app, _ = run_ui()
        text = all_text(app)
        assert "0.45" in text and "qwen/qwen3.8-27b" in text and "50%" in text
        for reason in ("low_evidence", "not_supported", "no_claims", "service_unavailable"):
            assert reason in text

    def test_old_server_values_are_reported_as_not_available(self):
        app, _ = run_ui(overrides={("GET", "/health"): OLD_HEALTH})
        assert not app.exception
        assert "not reported by this server" in all_text(app)


class TestSystemTab:
    def test_kpis_per_prompt_table_and_caption_render(self):
        app, _ = run_ui()
        assert not app.exception
        kpis = {m.label: m.value for m in app.metric}
        assert kpis["LLM calls"] == "42"
        assert kpis["Failed calls"] == "3"
        assert kpis["p95 latency"] == "1,500 ms"
        assert kpis["Tokens"] == "15,000"
        assert kpis["Cache hits"] == "4"
        frames = frame_text(app)
        assert "judge" in frames and "Completion tokens" in frames
        assert "since 2026-11-15 10:00:00" in all_text(app)
        assert "resets when the app is redeployed" in all_text(app)

    def test_old_metrics_shape_still_renders(self):
        old = {
            "llm_calls_total": 5, "llm_calls_failed": 0, "avg_latency_ms": 100.0,
            "p95_latency_ms": 200.0, "total_prompt_tokens": 10, "total_completion_tokens": 5,
            "calls_by_prompt": {"judge": 5},
            "guardrail_counts": {"PASS": 1, "REPAIR": 0, "BLOCK": 0, "SKIPPED": 0},
            "questions_answered": 1, "questions_refused": 0,
        }
        app, _ = run_ui(overrides={("GET", "/metrics"): old})
        assert not app.exception
        assert "judge" in frame_text(app)

    def test_empty_log_is_explained_not_charted(self):
        empty = {"guardrail_counts": {"PASS": 0, "REPAIR": 0, "BLOCK": 0, "SKIPPED": 0}}
        app, _ = run_ui(overrides={("GET", "/metrics"): empty})
        assert not app.exception
        assert "No model calls have been logged yet" in all_text(app)

    def test_metrics_failure_is_an_error_not_a_crash(self):
        app, _ = run_ui(overrides={("GET", "/metrics"): (500, {"detail": "db locked"})})
        assert not app.exception
        assert "Something went wrong" in " ".join(e.value for e in app.error)

    def test_refresh_button_reloads_without_error(self):
        app, _ = run_ui()
        app.button(key="refresh_system").click().run()
        assert not app.exception
