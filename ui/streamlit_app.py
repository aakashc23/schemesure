"""
SchemeSure frontend.

DESIGN RULE: no business logic lives here. This file renders what the API
returns and nothing more — it does not retrieve, does not call the LLM, does not
decide eligibility, and does not interpret verdicts beyond mapping them to a
badge. If you can only read one file to understand what the system *does*, read
`core/generator.py`, not this one.

Why a separate frontend at all: the same API serves the UI, the evaluation
harness and `curl`. Putting logic in the UI would mean the evaluation measured
something different from what users see.

The API base URL comes from the API_URL environment variable, so the same file
works locally (two processes) and in the container (both behind one port).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120  # a verified answer is 4-5 sequential LLM calls

st.set_page_config(
    page_title="SchemeSure",
    page_icon="🇮🇳",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==========================================================================
# API helpers
# ==========================================================================


def api_get(path: str):
    """GET from the API. Returns (data, error_message)."""
    try:
        response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return None, _detail(response)
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach the API at {API_URL} ({exc})."


def api_post(path: str, payload: dict):
    """POST to the API. Returns (data, error_message)."""
    try:
        response = requests.post(
            f"{API_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT
        )
        if response.status_code != 200:
            return None, _detail(response)
        return response.json(), None
    except requests.exceptions.Timeout:
        return None, "The request timed out. The model may be busy — please try again."
    except requests.exceptions.RequestException as exc:
        return None, f"Could not reach the API at {API_URL} ({exc})."


def _detail(response) -> str:
    """Pull FastAPI's `detail` out of an error response."""
    try:
        body = response.json()
        detail = body.get("detail", body)
        if isinstance(detail, list) and detail:
            detail = detail[0].get("msg", str(detail))
        return f"{response.status_code}: {detail}"
    except ValueError:
        return f"{response.status_code}: {response.text[:200]}"


@st.cache_data(ttl=300)
def load_schemes():
    """Cached: the scheme catalogue changes only when the data is rebuilt."""
    schemes, error = api_get("/schemes")
    return schemes or [], error


@st.cache_data(ttl=30)
def load_health():
    health, error = api_get("/health")
    return health, error


# ==========================================================================
# Rendering helpers
# ==========================================================================

# The user-facing promise for each status. The wording is deliberately plain:
# "partially verified" has to be understandable without reading any docs.
BADGES = {
    "verified": (
        "✅ Verified",
        "success",
        "Every factual claim in this answer was checked against the official "
        "documents and found to be supported.",
    ),
    "partially_verified": (
        "⚠️ Partially verified",
        "warning",
        "Some claims in the first draft were not supported by the official "
        "documents, so they were removed. What you see below is only the part "
        "that could be verified.",
    ),
    "refused": (
        "🛑 Not enough official info",
        "error",
        "SchemeSure would rather say nothing than risk telling you something "
        "wrong, so it has not answered this one.",
    ),
    "unverified": (
        "⚪ Unverified",
        "info",
        "The verification layer is switched off, so this answer has NOT been "
        "checked against the evidence. This mode exists only for evaluation.",
    ),
}


def render_badge(status: str) -> None:
    label, kind, explanation = BADGES.get(status, BADGES["unverified"])
    message = f"**{label}** — {explanation}"
    {"success": st.success, "warning": st.warning,
     "error": st.error, "info": st.info}[kind](message)


def render_citations(citations: list[dict]) -> None:
    """Sources, as clickable links back to the official pages."""
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        st.markdown(
            f"- **[{citation['index']}]** "
            f"[{citation['scheme_name']} — {citation['section']}]"
            f"({citation['source_url']})  \n"
            f"  <small>chunk <code>{citation['chunk_id']}</code> · "
            f"verified {citation['last_verified']}</small>",
            unsafe_allow_html=True,
        )


def render_verification(guardrail: dict, llm_calls: int, latency_ms: int) -> None:
    """
    The "How was this verified?" panel — the most important part of the UI.

    Showing the claim-by-claim audit trail is what makes the verification
    believable. A badge on its own is just another confident assertion.
    """
    decision = guardrail.get("decision", "?")
    claims = guardrail.get("claims") or []

    with st.expander("🔍 How was this verified?", expanded=False):
        columns = st.columns(4)
        columns[0].metric("Decision", decision)
        columns[1].metric("Claims checked", len(claims))
        columns[2].metric("LLM calls", llm_calls)
        columns[3].metric("Time", f"{latency_ms / 1000:.1f}s")

        st.markdown(
            "Every answer is split into single facts, and each fact is checked "
            "against the retrieved official text on its own. A fact that is not "
            "stated there is removed, even if it sounds right."
        )

        if claims:
            st.markdown("**Claim-by-claim result**")
            for index, claim in enumerate(claims, start=1):
                supported = claim.get("verdict") == "SUPPORTED"
                icon = "✅" if supported else "❌"
                st.markdown(f"{icon} **{index}.** {claim.get('claim', '')}")

                details = []
                if claim.get("evidence_chunk_id"):
                    details.append(f"evidence: `{claim['evidence_chunk_id']}`")
                if not claim.get("citation_valid", True):
                    # This is the fabricated-citation catch, surfaced to the user.
                    details.append("⚠️ cited a passage that does not exist")
                if claim.get("note"):
                    details.append(claim["note"])
                if details:
                    st.caption(" · ".join(details))
        else:
            st.info("No claims were checked — see the notes below.")

        if guardrail.get("notes"):
            st.markdown("**What the checker decided**")
            for note in guardrail["notes"]:
                st.caption(f"• {note}")


# ==========================================================================
# Header
# ==========================================================================

st.title("🇮🇳 SchemeSure")
st.caption(
    "Answers about Indian government schemes that are **checked against the "
    "official text before you see them**. Ask in English, Hindi or Hinglish."
)

health, health_error = load_health()
if health_error:
    st.error(
        f"The backend is not reachable at `{API_URL}`.\n\n{health_error}\n\n"
        "If you are running locally, start the API first:\n"
        "`python -m uvicorn app.main:app --port 8000`"
    )
    st.stop()

if not health.get("llm_key_configured"):
    st.warning(
        "No language-model API key is configured on the server, so asking "
        "questions is disabled. The **Schemes** tab still works."
    )

ask_tab, eligibility_tab, schemes_tab = st.tabs(
    ["💬 Ask", "✅ Check eligibility", "📚 Schemes"]
)


# ==========================================================================
# Tab 1 — Ask
# ==========================================================================

with ask_tab:
    st.markdown("#### Ask about a scheme")

    EXAMPLES = {
        "— pick an example —": "",
        "English: PM-KISAN amount": "How much money does PM-KISAN give per year?",
        "English: PMJJBY age limit": "What is the age limit for Pradhan Mantri Jeevan Jyoti Bima Yojana?",
        "Hindi: अटल पेंशन योजना": "अटल पेंशन योजना में शामिल होने की आयु सीमा क्या है?",
        "Hinglish: Ujjwala documents": "Ujjwala yojana ke liye kya documents chahiye?",
        "Trick: a scheme we do NOT cover": "What is the interest rate on Sukanya Samriddhi Yojana?",
        "Off-topic": "What is the capital of France?",
    }

    choice = st.selectbox(
        "Examples (the last two show the refusal behaviour)",
        list(EXAMPLES),
        index=0,
    )

    question = st.text_area(
        "Your question",
        value=EXAMPLES[choice],
        height=90,
        max_chars=500,
        placeholder="e.g. Who is eligible for PM SVANidhi?",
    )

    if st.button("Ask", type="primary", disabled=not health.get("llm_key_configured")):
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Retrieving official text, drafting, then verifying..."):
                data, error = api_post("/ask", {"question": question.strip()})

            if error:
                st.error(error)
            else:
                st.markdown("---")
                render_badge(data["status"])
                st.markdown(f"### {data['answer']}")

                render_citations(data.get("citations") or [])
                render_verification(
                    data.get("guardrail") or {},
                    data.get("llm_calls", 0),
                    data.get("latency_ms", 0),
                )

                with st.expander("Query details"):
                    st.write(
                        {
                            "detected language": data.get("language"),
                            "normalized query (what was searched)": data.get("normalized_query"),
                        }
                    )
                    st.caption(
                        "Your question is translated to English before searching. "
                        "The scheme documents are in English, and searching them "
                        "with romanised Hindi scores too low to find the right "
                        "passage — see docs/DECISIONS.md for the measurements."
                    )


# ==========================================================================
# Tab 2 — Eligibility
# ==========================================================================

with eligibility_tab:
    st.markdown("#### Which schemes might you qualify for?")
    st.caption(
        "Describe yourself in your own words. A language model only turns your "
        "text into fields (age, income, occupation…); **every eligibility "
        "decision below is made by plain Python code** comparing those fields "
        "against the official rules."
    )

    description = st.text_area(
        "About you",
        height=110,
        max_chars=500,
        placeholder=(
            "e.g. I am a 35 year old farmer from Bihar. My yearly income is "
            "about 2 lakh rupees."
        ),
    )

    if st.button("Check eligibility", type="primary",
                 disabled=not health.get("llm_key_configured")):
        if not description.strip():
            st.warning("Please describe yourself first.")
        else:
            with st.spinner("Reading your description, then applying the rules..."):
                data, error = api_post("/eligibility", {"description": description.strip()})

            if error:
                st.error(error)
            else:
                profile = data["profile"]
                st.markdown("**What was understood from your description**")
                st.caption(
                    "Anything shown as *not provided* was deliberately left blank "
                    "rather than guessed — a guess here would produce a confidently "
                    "wrong answer."
                )
                st.write(
                    {
                        key: (value if value is not None else "— not provided —")
                        for key, value in profile.items()
                    }
                )

                columns = st.columns(3)
                columns[0].metric("Likely eligible", data["eligible_count"])
                columns[1].metric("Need more info", data["need_more_info_count"])
                columns[2].metric("LLM calls used", data["llm_calls"])

                st.markdown("---")

                ICONS = {
                    "ELIGIBLE": "✅", "NEED_MORE_INFO": "❓", "NOT_ELIGIBLE": "❌",
                }

                # Compact overview first, then the reasoning per scheme.
                st.markdown("**Summary**")
                st.dataframe(
                    [
                        {
                            "Scheme": result["scheme_name"],
                            "Status": f"{ICONS.get(result['status'], '')} {result['status']}",
                            "Still needed": ", ".join(result["missing_fields"]) or "—",
                        }
                        for result in data["results"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("**Why — rule by rule**")
                for result in data["results"]:
                    icon = ICONS.get(result["status"], "")
                    with st.expander(f"{icon} {result['scheme_name']} — {result['status']}"):
                        for check in result["checks"]:
                            mark = {True: "✅", False: "❌", None: "❓"}[check["passed"]]
                            st.markdown(f"{mark} **{check['rule']}** — {check['reason']}")
                            if check.get("official_text"):
                                st.caption(f"Official wording: \"{check['official_text']}\"")

                        if result["conditions_to_verify"]:
                            st.markdown("**Still to check yourself**")
                            st.caption(
                                "These official conditions cannot be checked "
                                "automatically, so a green tick above is not a "
                                "guarantee."
                            )
                            for condition in result["conditions_to_verify"]:
                                st.markdown(f"- {condition}")

                        if result["source_url"]:
                            st.markdown(f"[Official page]({result['source_url']})")

                st.info(
                    "**This is guidance, not a decision.** Only the scheme's own "
                    "authority can confirm eligibility. Always check the official "
                    "page before applying."
                )


# ==========================================================================
# Tab 3 — Schemes
# ==========================================================================

with schemes_tab:
    st.markdown("#### Schemes covered")
    schemes, schemes_error = load_schemes()

    if schemes_error:
        st.error(schemes_error)
    else:
        st.caption(
            f"{len(schemes)} central government schemes, "
            f"{health.get('chunks_indexed', 0)} indexed passages. SchemeSure can "
            "only answer about these — anything else gets an honest refusal."
        )

        st.dataframe(
            [
                {
                    "Scheme": scheme["name"],
                    "Short name": scheme["short_name"] or "—",
                    "Ministry": scheme["ministry"],
                    "Last verified": scheme["last_verified"],
                    "Official source": scheme["source_url"],
                }
                for scheme in schemes
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Official source": st.column_config.LinkColumn("Official source", display_text="open")
            },
        )

        with st.expander("Where does this data come from?"):
            st.markdown(
                "Every document is built from the official **myScheme** portal "
                "(`myscheme.gov.in`), run by Digital India Corporation under the "
                "Ministry of Electronics & IT, read through the public JSON API "
                "that the portal's own pages use.\n\n"
                "Nothing is written from the model's memory. Where the official "
                "page says nothing, the document says *\"Not specified in "
                "official source\"* rather than filling the gap.\n\n"
                "`last verified` is the date the data was fetched. Schemes change, "
                "so treat anything older than a few months as needing a re-check."
            )

        with st.expander("Technical details"):
            st.write(health)


# ==========================================================================
# Footer
# ==========================================================================

st.markdown("---")
st.caption(
    "**Disclaimer** — SchemeSure is an independent educational project. It is "
    "**not** an official Government of India service and is not affiliated with "
    "any ministry. Answers are generated from official documents fetched on the "
    "date shown and may be out of date or incomplete. Always confirm on the "
    "official scheme website before applying or making any decision. Nothing "
    "here is legal or financial advice."
)
st.caption(
    f"Backend: `{API_URL}` · model `{health.get('llm_model', '?')}` · "
    f"verification {'on' if health.get('guardrail_enabled') else 'OFF'}"
)
