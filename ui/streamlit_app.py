"""
SchemeSure frontend.

DESIGN RULE: no business logic lives here. This file renders what the API
returns and nothing more — it does not retrieve, does not call the LLM, does not
decide eligibility, and does not interpret verdicts beyond mapping them to a
label. If you can only read one file to understand what the system *does*, read
`core/generator.py`, not this one.

Why a separate frontend at all: the same API serves the UI, the evaluation
harness and `curl`. Putting logic in the UI would mean the evaluation measured
something different from what users see.

BACKWARD COMPATIBILITY: the API grows new response fields over time (evidence,
stage trace, refusal reasons ...). Every optional field is read with `.get(...)`
and a default, so this UI still renders against an older backend — it simply
shows less. That also means a half-deployed release never crashes the page.

The API base URL comes from the API_URL environment variable, so the same file
works locally (two processes) and in the container (both behind one port).

Layout of this file, top to bottom:
  1. API helpers            (HTTP + friendly error messages)
  2. Rendering helpers      (status pill, sources, verification/evidence/pipeline)
  3. Page: header, sidebar, then one `with tab:` block per tab
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 120  # a verified answer is 4-5 sequential LLM calls
GITHUB_URL = "https://github.com/aakashc23/schemesure"
MAX_QUESTION_CHARS = 500
HISTORY_LIMIT = 20  # conversation cards kept per browser session

st.set_page_config(
    page_title="SchemeSure — verified answers on government schemes",
    page_icon=":material/verified_user:",  # a built-in icon: no emoji, no flag glyphs
    layout="wide",
    initial_sidebar_state="expanded",
)

# The only custom CSS: tighter page padding and a small grey style for the
# provenance line under each source. Nothing animated, no colours that carry
# meaning (status colours come from st.badge, which is theme-aware).
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; }
      .ss-small { font-size: 0.82rem; color: #5A6572; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# 1. API helpers
# ==========================================================================
# Every helper returns (data, error_message, http_status). Plain builtins only:
# the cached loaders below return these tuples, and st.cache_data pickles them.


def _detail(response) -> str:
    """Pull FastAPI's `detail` out of an error response."""
    try:
        body = response.json()
        detail = body.get("detail", body) if isinstance(body, dict) else body
        if isinstance(detail, list) and detail:
            first = detail[0]
            detail = first.get("msg", str(detail)) if isinstance(first, dict) else str(first)
        return str(detail)
    except ValueError:
        return (getattr(response, "text", "") or "")[:200]


def friendly_error(path: str, status: int, detail: str) -> str:
    """
    Turn an HTTP error into a sentence a visitor can act on.

    The rule: say what happened, say whether it is the user's fault, say what to
    do next. We never show a raw stack trace or provider error text.
    """
    is_eligibility = path.startswith("/eligibility")
    if status == 400:
        return f"Your input was not accepted: {detail}. Please shorten or rephrase it and try again."
    if status == 422:
        return "Your input was not in the expected format. Please check the fields and try again."
    if status == 502:
        # Deliberately generic: the provider's own error text is not for users.
        return (
            "The language model could not be reached right now. This is a temporary "
            "service problem, not an issue with your question — please try again in a minute."
        )
    if status == 503:
        if is_eligibility:
            return (
                "The language model is not available, so your free-text description "
                "could not be read. Use **Fill in a form** instead — it needs no "
                "language model and still checks every scheme."
            )
        return (
            "The language model is not configured or not available on this server, "
            "so questions cannot be answered right now. The Schemes tab and the "
            "form-based eligibility check still work."
        )
    if status == 404:
        return f"Not found: {detail}"
    return f"Something went wrong ({status}): {detail}"


def api_get(path: str):
    """GET from the API. Returns (data, error_message, status)."""
    try:
        response = requests.get(f"{API_URL}{path}", timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        return None, "The server took too long to respond. Please try again.", None
    except requests.exceptions.RequestException:
        return None, "Could not reach the SchemeSure API.", None
    if response.status_code != 200:
        return None, friendly_error(path, response.status_code, _detail(response)), response.status_code
    return response.json(), None, 200


def api_post(path: str, payload: dict):
    """POST to the API. Returns (data, error_message, status)."""
    try:
        response = requests.post(f"{API_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        return None, (
            f"The request timed out after {REQUEST_TIMEOUT} seconds. The model may be "
            "busy — please try again."
        ), None
    except requests.exceptions.RequestException:
        return None, "Could not reach the SchemeSure API. Please try again.", None
    if response.status_code != 200:
        return None, friendly_error(path, response.status_code, _detail(response)), response.status_code
    return response.json(), None, 200


# --- Cached read-only loaders ---------------------------------------------
# These endpoints change only when the server is redeployed, so a short cache
# keeps every Streamlit rerun (which happens on each click) from re-calling
# them. A *failure* is never kept: otherwise one blip would stick for minutes.


@st.cache_data(ttl=300)
def _load_schemes():
    return api_get("/schemes")


@st.cache_data(ttl=30)
def _load_health():
    return api_get("/health")


@st.cache_data(ttl=300)
def _load_options():
    return api_get("/eligibility/options")


@st.cache_data(ttl=300)
def _load_scheme_detail(scheme_id: str):
    return api_get(f"/schemes/{scheme_id}")


def _load_metrics():
    # Not cached: the System tab exists to show live numbers.
    return api_get("/metrics")


def _cached(loader, *args):
    """Call a cached loader; if it failed, drop the cache entry so we retry next run."""
    result = loader(*args)
    if result[1]:
        loader.clear()
    return result


# ==========================================================================
# 2. Rendering helpers
# ==========================================================================

LANGUAGE_LABELS = {"en": "English", "hi": "Hindi", "hinglish": "Hinglish"}

# The sidebar's answer-language setting -> the `language` value sent to /ask.
# None means "detect it from the question".
LANGUAGE_CHOICES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Hinglish": "hinglish",
}

# The user-facing promise for each status. Wording is deliberately plain:
# "partially verified" has to be understandable without reading any docs.
# status -> (label, st.badge colour, icon, explanation)
STATUS_VIEWS = {
    "verified": (
        "Verified", "green", ":material/verified:",
        "Every factual claim in this answer was checked against the official "
        "documents and found to be supported.",
    ),
    "partially_verified": (
        "Partially verified", "orange", ":material/warning:",
        "Some claims in the first draft were not supported by the official "
        "documents, so they were removed. What you see below is only the part "
        "that could be verified, and it was checked again after the repair.",
    ),
    "refused": (
        "Not enough official info", "red", ":material/block:",
        "SchemeSure would rather say nothing than risk telling you something "
        "wrong, so it has not answered this one.",
    ),
    "unverified": (
        "Unverified", "gray", ":material/help:",
        "The verification layer is switched off, so this answer has NOT been "
        "checked against the evidence. This mode exists only for evaluation.",
    ),
}

# refusal_reason -> plain-language explanation shown under a refusal.
REFUSAL_EXPLANATIONS = {
    "low_evidence": (
        "No official passage was similar enough to your question, so no answer "
        "was drafted. This usually means the topic is not one of the covered schemes."
    ),
    "not_supported": (
        "An answer was drafted, but too many of its claims were not backed by the "
        "official text, so it was withheld instead of risking a wrong answer."
    ),
    "no_claims": (
        "The draft did not state any checkable fact (the model declined to answer "
        "from the passages), so there was nothing to verify and nothing to show."
    ),
    "service_unavailable": (
        "A language-model call failed (rate limit, quota or outage), so the answer "
        "could not be checked. This is a temporary service problem and does NOT "
        "mean the information is missing. SchemeSure never shows an unchecked "
        "answer — please try again in a minute."
    ),
}

RULE_LABELS = {
    "min_age": "Minimum age",
    "max_age": "Maximum age",
    "max_annual_income": "Annual income limit",
    "allowed_states": "State",
    "allowed_occupations": "Occupation",
    "gender": "Gender",
    "category": "Category",
}


def status_view(data: dict):
    """Pick (label, colour, icon, explanation) for one /ask response."""
    status = data.get("status", "unverified")
    if status == "refused" and data.get("refusal_reason") == "service_unavailable":
        # Grey, not red: red says "we have no information", but here the
        # information may well exist — only the checking service failed.
        return (
            "Verification unavailable", "gray", ":material/cloud_off:",
            REFUSAL_EXPLANATIONS["service_unavailable"],
        )
    return STATUS_VIEWS.get(status, STATUS_VIEWS["unverified"])


def md_escape(text: str) -> str:
    """Escape Markdown punctuation so a user's question is shown literally."""
    for char in "\\`*_[]<>#$|~":
        text = text.replace(char, "\\" + char)
    return text


def fmt_similarity(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def render_status(data: dict) -> None:
    """The coloured status pill plus one sentence saying what it means."""
    label, colour, icon, explanation = status_view(data)
    st.badge(label, icon=icon, color=colour)
    reason = data.get("refusal_reason")
    if data.get("status") == "refused" and reason in REFUSAL_EXPLANATIONS:
        # For a refusal the specific reason is more useful than the generic line.
        st.caption(REFUSAL_EXPLANATIONS[reason])
    elif data.get("status") == "partially_verified" and not _draft_had_unsupported(data):
        # REPAIR is also chosen when every claim was supported but a number or link
        # was not found in the official text, or the draft had too many claims to
        # check. Saying "unsupported claims were removed" would contradict the
        # claims table, which shows them all supported.
        st.caption(
            "The first draft was rebuilt before it was shown, because some numbers or "
            "links in it were not found in the official text, or it contained more "
            "claims than can be checked. What you see below was checked again after "
            "the rebuild."
        )
    else:
        st.caption(explanation)


def _draft_had_unsupported(data: dict) -> bool:
    """True when the guardrail judged at least one draft claim as not supported."""
    claims = (data.get("guardrail") or {}).get("claims") or []
    return any(claim.get("verdict") != "SUPPORTED" for claim in claims)


def render_meta_line(data: dict) -> None:
    parts = [f"Language: {LANGUAGE_LABELS.get(data.get('language'), data.get('language') or '?')}"]
    parts.append(f"{data.get('llm_calls', 0)} LLM call(s)")
    parts.append(f"{data.get('latency_ms', 0) / 1000:.1f} s")
    if data.get("cached"):
        parts.append("served from cache")
    if data.get("normalization_fallback"):
        parts.append("language model fallback (question was searched as typed)")
    st.caption(" · ".join(parts))


def render_citations(citations: list[dict]) -> None:
    """Sources, as clickable links back to the official pages."""
    if not citations:
        return
    st.markdown("**Sources**")
    for citation in citations:
        st.markdown(
            f"**[{citation.get('index', '?')}]** "
            f"[{citation.get('scheme_name', '')} — {citation.get('section', '')}]"
            f"({citation.get('source_url', '')})  \n"
            f"<span class='ss-small'>passage <code>{citation.get('chunk_id', '')}</code>"
            f" · last verified {citation.get('last_verified', '?')}</span>",
            unsafe_allow_html=True,
        )


def claims_table(claims: list[dict]) -> None:
    """One row per claim: the audit trail that makes the badge believable."""
    # Older servers do not send similarity; then the column (and its caption)
    # is left out instead of showing a column of dashes.
    show_similarity = any("evidence_similarity" in claim for claim in claims)
    show_notes = any(claim.get("note") for claim in claims)
    rows = []
    for index, claim in enumerate(claims, start=1):
        row = {
            "#": index,
            "Claim": claim.get("claim", ""),
            "Verdict": "Supported" if claim.get("verdict") == "SUPPORTED" else "Not supported",
            "Evidence chunk": claim.get("evidence_chunk_id") or "—",
        }
        if show_similarity:
            row["Similarity to cited evidence"] = fmt_similarity(claim.get("evidence_similarity"))
        row["Citation valid"] = "yes" if claim.get("citation_valid", True) else "NO"
        if show_notes:
            row["Note"] = claim.get("note", "")
        rows.append(row)
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"Claim": st.column_config.TextColumn("Claim", width="large")},
    )
    if show_similarity:
        st.caption(
            "Similarity to cited evidence is an independent diagnostic (embedding "
            "cosine similarity between the claim and the passage the judge cited). "
            "It is recorded for calibration and is not used for the decision yet."
        )


def render_verification(data: dict) -> None:
    """The "Verification" tab: decision, claim table, repair re-check, warnings."""
    guardrail = data.get("guardrail") or {}
    claims = guardrail.get("claims") or []
    decision = guardrail.get("decision", "?")

    block_threshold = guardrail.get("block_threshold")
    summary = (
        f"**Decision: {decision}** — "
        f"{guardrail.get('supported_count', 0)} of {len(claims)} claim(s) supported"
    )
    if block_threshold:
        summary += f" · answer is withheld above {block_threshold:.0%} unsupported"
    st.markdown(summary)
    st.caption(
        "Every answer is split into single facts, and each fact is checked against "
        "the retrieved official text on its own. A fact that is not stated there is "
        "removed, even if it sounds right."
    )

    if claims:
        claims_table(claims)
        if any(not claim.get("citation_valid", True) for claim in claims):
            # The fabricated-citation catch, surfaced to the user.
            st.warning(
                "One or more claims cited a passage that does not exist among the "
                "retrieved passages. Such claims are treated as unsupported."
            )
    else:
        st.info("No claims were checked for this response — see the notes below.")

    ungrounded = guardrail.get("ungrounded_tokens") or []
    if ungrounded:
        st.warning(
            "Numbers or links not found in the official text: "
            + ", ".join(f"`{token}`" for token in ungrounded)
        )

    notes = guardrail.get("notes") or []
    if notes:
        st.markdown("**What the checker decided**")
        for note in notes:
            st.caption(f"• {note}")

    repair = data.get("repair_report")
    if repair:
        repair_claims = repair.get("claims") or []
        st.markdown("**Re-check of the repaired answer**")
        st.caption(
            f"The repaired answer was re-checked: {repair.get('supported_count', 0)}/"
            f"{len(repair_claims)} claims supported (decision: {repair.get('decision', '?')})."
        )
        if repair_claims:
            claims_table(repair_claims)


def render_evidence(data: dict) -> None:
    """The "Evidence" tab: what retrieval found, and which passages were cited."""
    retrieval = data.get("retrieval")
    if not retrieval:
        st.info(
            "This server did not send retrieval details. The cited passages are "
            "listed under Sources above."
        )
        return

    passages = retrieval.get("passages") or []
    threshold = retrieval.get("threshold", 0.0)
    cited_ids = {c.get("chunk_id") for c in (data.get("citations") or [])}

    columns = st.columns(3)
    columns[0].metric("Best score", f"{retrieval.get('best_score', 0.0):.3f}")
    columns[1].metric("Threshold", f"{threshold:.2f}")
    columns[2].metric("Passages retrieved", len(passages))
    if retrieval.get("low_evidence"):
        st.caption(
            "The best passage scored below the threshold, so nothing was drafted. "
            "These are the closest passages that were found."
        )
    else:
        st.caption(
            "Passages below the threshold are shown for transparency; the answer "
            "is drafted from the retrieved set."
        )

    if not passages:
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Rank": passage.get("index", position),
                    "Scheme": passage.get("scheme_name", ""),
                    "Section": passage.get("section", ""),
                    "Score": passage.get("score", 0.0),
                    "Above threshold": "yes" if passage.get("above_threshold") else "no",
                    "Cited": "yes" if passage.get("chunk_id") in cited_ids else "no",
                }
                for position, passage in enumerate(passages, start=1)
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score (cosine similarity)", min_value=0.0, max_value=1.0, format="%.3f"
            )
        },
    )

    for position, passage in enumerate(passages, start=1):
        cited = " — cited in the answer" if passage.get("chunk_id") in cited_ids else ""
        title = (
            f"[{passage.get('index', position)}] {passage.get('scheme_name', '')} — "
            f"{passage.get('section', '')} (score {passage.get('score', 0.0):.3f}){cited}"
        )
        with st.expander(title):
            st.markdown(passage.get("text", ""))
            st.caption(
                f"passage {passage.get('chunk_id', '')} · "
                f"[official page]({passage.get('source_url', '')})"
            )


def render_pipeline(data: dict) -> None:
    """The "Pipeline" tab: which stages ran, how long each took."""
    columns = st.columns(3)
    columns[0].metric("Total latency", f"{data.get('latency_ms', 0) / 1000:.1f} s")
    columns[1].metric("LLM calls", data.get("llm_calls", 0))
    columns[2].metric("Detected language", LANGUAGE_LABELS.get(data.get("language"), "?"))

    st.markdown("**Query that was searched (English)**")
    st.code(data.get("normalized_query") or "—", language=None)
    st.caption(
        "Your question is translated to English before searching. The scheme "
        "documents are in English, and searching them with romanised Hindi scores "
        "too low to find the right passage — see docs/DECISIONS.md for the measurements."
    )

    trace = data.get("trace") or []
    if not trace:
        st.info("This server did not send a per-stage trace.")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Stage": step.get("stage", ""),
                    "Duration (ms)": step.get("duration_ms", 0),
                    "LLM calls": step.get("llm_calls", 0),
                    "Outcome": step.get("outcome", ""),
                }
                for step in trace
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_answer_card(entry: dict) -> None:
    """One question + answer, in a bordered card."""
    data = entry["data"]
    with st.container(border=True):
        st.markdown(f"**Question:** {md_escape(entry['question'])}")
        render_status(data)
        render_meta_line(data)

        # Plain markdown, not a heading: a long answer in heading size is hard to read.
        st.markdown(data.get("answer", ""))
        render_citations(data.get("citations") or [])

        verification_tab, evidence_tab, pipeline_tab = st.tabs(
            ["Verification", "Evidence", "Pipeline"]
        )
        with verification_tab:
            render_verification(data)
        with evidence_tab:
            render_evidence(data)
        with pipeline_tab:
            render_pipeline(data)


# --- Eligibility rendering -------------------------------------------------

ELIGIBILITY_GROUPS = [
    ("ELIGIBLE", "Likely eligible"),
    ("NEED_MORE_INFO", "Need more information"),
    ("NOT_ELIGIBLE", "Not eligible"),
]
CHECK_MARKS = {True: "✓", False: "✗", None: "?"}


def rule_label(rule: str) -> str:
    return RULE_LABELS.get(rule, rule.replace("_", " ").capitalize())


def render_scheme_eligibility(result: dict) -> None:
    """The rule-by-rule reasoning for one scheme (shown inside an expander)."""
    checks = result.get("checks") or []
    if checks:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Rule": rule_label(check.get("rule", "")),
                        "Result": CHECK_MARKS.get(check.get("passed"), "?"),
                        "Reason": check.get("reason", ""),
                        "Official wording": check.get("official_text", "") or "—",
                    }
                    for check in checks
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={"Result": st.column_config.TextColumn("Result", width="small")},
        )
        st.caption("✓ rule met · ✗ rule not met · ? could not tell (information missing)")
    else:
        st.caption("This scheme has no machine-checkable rules.")

    if result.get("missing_fields"):
        st.caption("Still needed to decide: " + ", ".join(result["missing_fields"]))

    conditions = result.get("conditions_to_verify") or []
    if conditions:
        st.markdown("**Still to check yourself**")
        st.caption(
            "These official conditions cannot be checked automatically, so a green "
            "tick above is not a guarantee."
        )
        for condition in conditions:
            st.markdown(f"- {condition}")

    if result.get("source_url"):
        st.markdown(f"[Official page]({result['source_url']})")


def render_eligibility_results(data: dict) -> None:
    """Three counts, then the schemes grouped by outcome."""
    results = data.get("results") or []
    by_status: dict[str, list[dict]] = {}
    for result in results:
        by_status.setdefault(result.get("status", ""), []).append(result)

    columns = st.columns(4)
    columns[0].metric("Likely eligible", data.get("eligible_count", len(by_status.get("ELIGIBLE", []))))
    columns[1].metric(
        "Need more info", data.get("need_more_info_count", len(by_status.get("NEED_MORE_INFO", [])))
    )
    columns[2].metric(
        "Not eligible", data.get("not_eligible_count", len(by_status.get("NOT_ELIGIBLE", [])))
    )
    columns[3].metric("LLM calls used", data.get("llm_calls", 0))

    st.info(
        "**This is guidance, not a decision.** Only the scheme's own authority can "
        "confirm eligibility. Always check the official page before applying."
    )

    for status, heading in ELIGIBILITY_GROUPS:
        group = by_status.get(status, [])
        if not group:
            continue
        st.markdown(f"**{heading} ({len(group)})**")
        for result in group:
            with st.expander(result.get("scheme_name", result.get("scheme_id", "?"))):
                render_scheme_eligibility(result)


# --- Eligibility form ------------------------------------------------------

NOT_SPECIFIED = "Not specified"
FALLBACK_GENDERS = ["female", "male", "other"]
FALLBACK_CATEGORIES = ["General", "OBC", "SC", "ST", "EWS"]


def _match_option(options: list[str], value) -> str:
    """Find `value` in `options` ignoring case; NOT_SPECIFIED when absent."""
    if value is None:
        return NOT_SPECIFIED
    for option in options:
        if str(option).lower() == str(value).lower():
            return option
    return NOT_SPECIFIED


def profile_form(prefix: str, options: dict, defaults: dict | None, label: str, version: int = 0):
    """
    The structured profile form. Returns (submitted, profile_dict).

    A field the user leaves empty is sent as None — never a default value. The
    rule engine treats None as "unknown" and answers NEED_MORE_INFO, which is the
    honest answer; a silent default would produce a confidently wrong one.

    `version` is part of every widget key so that a fresh free-text extraction
    resets the pre-filled values instead of keeping the previous ones.
    """
    defaults = defaults or {}
    occupations = options.get("occupations") or []
    genders = options.get("genders") or FALLBACK_GENDERS
    categories = options.get("categories") or FALLBACK_CATEGORIES
    key = lambda name: f"{prefix}_{name}_{version}"  # noqa: E731

    with st.form(f"{prefix}_form"):
        first, second, third = st.columns(3)
        age = first.number_input(
            "Age (years, optional)", min_value=0, max_value=120, step=1,
            value=defaults.get("age"), key=key("age"),
        )
        income = second.number_input(
            "Yearly family income in rupees (optional)", min_value=0, step=1000,
            value=defaults.get("annual_income"), key=key("income"),
            help="Total yearly household / family income from all sources, in rupees.",
        )
        state = third.text_input(
            "State or union territory (optional)", value=defaults.get("state") or "",
            key=key("state"),
        )

        fourth, fifth, sixth = st.columns(3)
        if occupations:
            occupation_choices = [NOT_SPECIFIED, *occupations]
            current = defaults.get("occupation")
            if current and _match_option(occupations, current) == NOT_SPECIFIED:
                occupation_choices.append(current)  # keep an unusual extracted value visible
            occupation = fourth.selectbox(
                "Occupation", occupation_choices, key=key("occupation"),
                index=occupation_choices.index(_match_option(occupation_choices, current)),
                format_func=lambda value: value.replace("_", " ").capitalize()
                if value != NOT_SPECIFIED else value,
            )
        else:
            occupation = fourth.text_input(
                "Occupation (optional)", value=defaults.get("occupation") or "", key=key("occupation")
            )
        gender_choices = [NOT_SPECIFIED, *genders]
        gender = fifth.selectbox(
            "Gender", gender_choices, key=key("gender"),
            index=gender_choices.index(_match_option(gender_choices, defaults.get("gender"))),
            format_func=lambda value: value.capitalize(),
        )
        category_choices = [NOT_SPECIFIED, *categories]
        category = sixth.selectbox(
            "Category", category_choices, key=key("category"),
            index=category_choices.index(_match_option(category_choices, defaults.get("category"))),
        )
        submitted = st.form_submit_button(label, type="primary", key=f"{prefix}_submit")

    def clean(value):
        if value is None or value == NOT_SPECIFIED:
            return None
        value = str(value).strip()
        return value or None

    profile = {
        "age": int(age) if age is not None else None,
        "annual_income": int(income) if income is not None else None,
        "state": clean(state),
        "occupation": clean(occupation),
        "gender": clean(gender),
        "category": clean(category),
    }
    return submitted, profile


# --- Schemes rendering -----------------------------------------------------


def render_scheme_detail(detail: dict) -> None:
    st.markdown(f"### {detail.get('name', '')}")
    st.caption(
        f"{detail.get('ministry') or 'Ministry not stated'} · last verified "
        f"{detail.get('last_verified', '?')}"
        + (f" · {detail['chunks_indexed']} indexed passage(s)" if "chunks_indexed" in detail else "")
    )
    if detail.get("source_url"):
        st.markdown(f"[Official scheme page]({detail['source_url']})")

    references = detail.get("reference_urls") or []
    if references:
        st.markdown("**Reference documents**")
        for reference in references:
            st.markdown(f"- [{reference.get('title') or reference.get('url')}]({reference.get('url')})")

    sections = detail.get("sections_detail") or []
    if sections:
        st.markdown("**Official text, by section**")
        for section in sections:
            with st.expander(section.get("title", "Section")):
                st.markdown(section.get("text", ""))

    rules = detail.get("rules")
    if rules:
        st.markdown("**Machine-checkable rules**")
        st.caption(
            "These are the conditions the eligibility engine can test with plain "
            "Python. Each shows the official sentence it was taken from."
        )
        sources = rules.get("rule_sources") or {}
        rows = []
        for rule in RULE_LABELS:
            value = rules.get(rule)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            rows.append(
                {
                    "Rule": RULE_LABELS[rule],
                    "Restriction": "No restriction" if value in (None, "") else str(value),
                    "Official sentence": sources.get(rule, "") or "—",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        unmodelled = rules.get("unmodelled_conditions") or []
        if unmodelled:
            st.markdown("**Conditions not modelled by the rule engine**")
            st.caption("A person must still check these against the official page.")
            for condition in unmodelled:
                st.markdown(f"- {condition}")
    elif "rules" in detail:
        st.caption("This scheme has no machine-checkable rules.")


# ==========================================================================
# 3. Page
# ==========================================================================

health, health_error, _status = _cached(_load_health)
health = health or {}

# `llm_ready` means the pipeline was actually built. Older servers only report
# whether a key is configured, which is the next-best signal.
llm_ready = bool(health.get("llm_ready", health.get("llm_key_configured", False)))


def _clear_conversation() -> None:
    st.session_state["history"] = []


def _queue_question(question: str) -> None:
    """Button callback: remember the question; the page body asks it on this rerun."""
    st.session_state["pending_question"] = question


st.session_state.setdefault("history", [])

# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.markdown("### SchemeSure")
    st.markdown("**System status**")
    if health_error:
        st.markdown("- API: **unreachable**")
    else:
        st.markdown(
            "\n".join(
                [
                    "- API: **reachable**",
                    f"- Schemes: **{health.get('schemes_loaded', '?')}**",
                    f"- Indexed passages: **{health.get('chunks_indexed', '?')}**",
                    f"- LLM model: `{health.get('llm_model', '?')}`",
                    f"- Verification: **{'on' if health.get('guardrail_enabled') else 'OFF'}**",
                    f"- LLM key: **{'configured' if health.get('llm_key_configured') else 'missing'}**",
                    f"- Answer cache: **{'on' if health.get('answer_cache_enabled') else 'off'}**",
                ]
            )
        )

    st.markdown("**Settings**")
    language_choice = st.selectbox(
        "Answer language",
        list(LANGUAGE_CHOICES),
        key="answer_language",
        help="Auto-detect answers in the language you wrote the question in.",
    )
    st.button("Clear conversation", key="clear_conversation", on_click=_clear_conversation)

    st.divider()
    st.caption(
        "Independent educational project — not an official Government of India "
        "service. Answers can be out of date: confirm on the official site."
    )
    st.markdown(f"[Source code on GitHub]({GITHUB_URL})")

# --- Header ------------------------------------------------------------------
st.title("SchemeSure")
st.markdown(
    "Answers about Indian government schemes that are **checked against the "
    "official text before you see them**. Ask in English, Hindi or Hinglish."
)

steps = st.columns(3)
for column, (title, text) in zip(
    steps,
    [
        ("1. Retrieve official text",
         "Your question is matched against passages from official scheme pages."),
        ("2. Draft an answer",
         "A language model writes an answer using only those passages, with citations."),
        ("3. Verify every claim",
         "Each fact is checked against the passages. Unsupported facts are removed, "
         "or the answer is withheld."),
    ],
):
    with column.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)

if health_error:
    st.error(
        "The backend is not reachable.\n\n"
        f"{health_error}\n\n"
        "If you are running locally, start the API first:\n"
        "`python -m uvicorn app.main:app --port 8000`"
    )
    st.stop()

if not llm_ready:
    # Two different causes, two different messages: a missing key versus a key
    # that is set but a pipeline that failed to start.
    if health.get("llm_key_configured"):
        st.warning(
            "The language model is configured but is not available on this server "
            "right now, so asking questions and free-text eligibility are disabled. "
            "The **Schemes** tab, the form-based **Eligibility** check and "
            "**How it works** still work."
        )
    else:
        st.warning(
            "No language-model API key is configured on the server, so asking questions "
            "and free-text eligibility are disabled. The **Schemes** tab, the form-based "
            "**Eligibility** check and **How it works** still work."
        )

ask_tab, eligibility_tab, schemes_tab, how_tab, system_tab = st.tabs(
    ["Ask", "Eligibility", "Schemes", "How it works", "System"]
)


# ==========================================================================
# Tab 1 — Ask
# ==========================================================================

# label, question. The last two are deliberate traps that should be refused.
EXAMPLES = [
    ("PM-KISAN amount (English)", "How much money does PM-KISAN give per year?"),
    ("Atal Pension age limit (Hindi)", "अटल पेंशन योजना में शामिल होने की आयु सीमा क्या है?"),
    ("Ujjwala documents (Hinglish)", "Ujjwala yojana ke liye kya documents chahiye?"),
    ("PMJJBY age limit", "What is the age limit for Pradhan Mantri Jeevan Jyoti Bima Yojana?"),
    ("Trick: Sukanya Samriddhi interest rate", "What is the interest rate on Sukanya Samriddhi Yojana?"),
    ("Off-topic: capital of France", "What is the capital of France?"),
]

with ask_tab:
    st.markdown("#### Ask about a scheme")
    st.caption(
        "Try an example. The last two are traps — a scheme that is not indexed and "
        "an off-topic question — and should be refused."
    )

    for row_start in (0, 3):
        columns = st.columns(3)
        for column, (index, (label, example)) in zip(
            columns, list(enumerate(EXAMPLES))[row_start:row_start + 3]
        ):
            column.button(
                label, key=f"example_{index}", width="stretch", disabled=not llm_ready,
                on_click=_queue_question, args=(example,),
            )

    # A form, deliberately. A bare st.text_area only commits its value on blur or
    # Ctrl+Enter, so a visitor who types a question and clicks the button can get
    # a false "Please type a question first" on the first click. Inside a form,
    # every widget commits together when the submit button is pressed.
    with st.form("ask_form"):
        question = st.text_area(
            "Your question",
            height=90,
            max_chars=MAX_QUESTION_CHARS,
            placeholder="e.g. Who is eligible for PM SVANidhi?",
        )
        submitted = st.form_submit_button(
            "Ask", type="primary", key="ask_submit", disabled=not llm_ready
        )

    pending = st.session_state.pop("pending_question", None)
    if submitted:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            pending = question.strip()

    if pending:
        with st.spinner("Retrieving official text, drafting, then verifying every claim..."):
            data, error, _status = api_post(
                "/ask",
                {"question": pending, "language": LANGUAGE_CHOICES[language_choice]},
            )
        if error:
            st.error(error)
        else:
            st.session_state["history"].insert(0, {"question": pending, "data": data})
            del st.session_state["history"][HISTORY_LIMIT:]

    history = st.session_state["history"]
    if not history:
        st.info("No questions yet. Pick an example above or type your own.")
    for entry in history:  # newest first
        render_answer_card(entry)


# ==========================================================================
# Tab 2 — Eligibility
# ==========================================================================

with eligibility_tab:
    st.markdown("#### Which schemes might you qualify for?")
    st.caption(
        "Every eligibility decision is made by plain Python code comparing your "
        "details with the official rules — never by a language model. Income means "
        "your total yearly household / family income in rupees."
    )
    options, options_error, _status = _cached(_load_options)
    options = options or {}

    mode = st.radio(
        "How do you want to enter your details?",
        ["Describe yourself", "Fill in a form"],
        horizontal=True,
        key="elig_mode",
    )

    if mode == "Describe yourself":
        st.caption(
            "A language model only turns your text into fields (age, income, "
            "occupation ...). It uses 1 LLM call."
        )
        if not llm_ready:
            st.info("Free text needs the language model, which is not available. Use **Fill in a form**.")
        with st.form("elig_text_form"):
            description = st.text_area(
                "About you",
                height=110,
                max_chars=MAX_QUESTION_CHARS,
                placeholder=(
                    "e.g. I am a 35 year old farmer from Bihar. My yearly income is "
                    "about 2 lakh rupees."
                ),
            )
            described = st.form_submit_button(
                "Check eligibility", type="primary", key="elig_text_submit", disabled=not llm_ready
            )

        if described:
            if not description.strip():
                st.warning("Please describe yourself first.")
            else:
                with st.spinner("Reading your description, then applying the rules..."):
                    data, error, _status = api_post(
                        "/eligibility", {"description": description.strip()}
                    )
                if error:
                    st.error(error)
                else:
                    st.session_state["elig_text_result"] = data
                    # New extraction -> new widget keys -> the form below is pre-filled afresh.
                    st.session_state["elig_text_version"] = (
                        st.session_state.get("elig_text_version", 0) + 1
                    )

        text_result = st.session_state.get("elig_text_result")
        if text_result:
            st.markdown("**What we understood**")
            st.caption(
                "Anything left as *Not specified* was deliberately left blank rather "
                "than guessed. Correct anything that is wrong and re-check — this "
                "uses no language model."
            )
            rechecked, edited_profile = profile_form(
                "understood", options, text_result.get("profile"),
                "Re-check with these details",
                version=st.session_state.get("elig_text_version", 0),
            )
            if rechecked:
                with st.spinner("Applying the rules..."):
                    data, error, _status = api_post("/eligibility/check", {"profile": edited_profile})
                if error:
                    st.error(error)
                else:
                    # Keep the original description and profile shape; show the new verdicts.
                    st.session_state["elig_text_result"] = data
                    text_result = data
            st.divider()
            render_eligibility_results(text_result)

    else:
        st.caption("No language model is used in this mode — it works even when the model is down.")
        if options_error:
            st.caption("The server did not provide the list of occupations; free-text fields are shown instead.")
        checked, profile = profile_form("elig", options, None, "Check eligibility")
        if checked:
            if all(value is None for value in profile.values()):
                st.warning("Enter at least one detail first — with nothing entered every scheme is 'need more info'.")
            else:
                with st.spinner("Applying the rules..."):
                    data, error, _status = api_post("/eligibility/check", {"profile": profile})
                if error:
                    st.error(error)
                else:
                    st.session_state["elig_form_result"] = data

        form_result = st.session_state.get("elig_form_result")
        if form_result:
            st.divider()
            render_eligibility_results(form_result)


# ==========================================================================
# Tab 3 — Schemes
# ==========================================================================

with schemes_tab:
    st.markdown("#### Schemes covered")
    schemes, schemes_error, _status = _cached(_load_schemes)
    schemes = schemes or []

    if schemes_error:
        st.error(schemes_error)
    else:
        st.caption(
            f"{len(schemes)} central government schemes, "
            f"{health.get('chunks_indexed', 0)} indexed passages. SchemeSure can "
            "only answer about these — anything else gets an honest refusal."
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Scheme": scheme.get("name", ""),
                        "Short name": scheme.get("short_name") or "—",
                        "Ministry": scheme.get("ministry", ""),
                        "Last verified": scheme.get("last_verified", ""),
                        "Official source": scheme.get("source_url", ""),
                    }
                    for scheme in schemes
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Official source": st.column_config.LinkColumn("Official source", display_text="open")
            },
        )

        names = {scheme["scheme_id"]: scheme.get("name", scheme["scheme_id"]) for scheme in schemes}
        chosen = st.selectbox(
            "Open a scheme for details",
            list(names),
            index=None,
            placeholder="Choose a scheme",
            format_func=lambda scheme_id: names[scheme_id],
            key="scheme_choice",
        )
        if chosen:
            detail, detail_error, detail_status = _cached(_load_scheme_detail, chosen)
            if detail_error:
                if detail_status == 404:
                    st.info(
                        "This server does not provide per-scheme details yet. The "
                        "catalogue above is still accurate."
                    )
                else:
                    st.error(detail_error)
            else:
                render_scheme_detail(detail)

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


# ==========================================================================
# Tab 4 — How it works
# ==========================================================================

# Two fill colours separate the two kinds of step: amber = a language model does
# it, blue = plain Python code does it. Outcomes are green / red / grey.
ASK_DIAGRAM = r"""
digraph ask {
  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.35;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.14,0.07"];
  edge [fontname="Helvetica", fontsize=10, color="#5A6572"];

  q      [label="Your question\n(English, Hindi or Hinglish)", fillcolor="#FFFFFF", color="#5A6572"];
  norm   [label="1. Normalise\ntranslate to English, detect language", fillcolor="#FDEBD0", color="#B9770E"];
  ret    [label="2. Retrieve\ntop-k vector search over official text", fillcolor="#D6EAF8", color="#1F4E79"];
  gate   [label="Best score >= threshold?", shape=diamond, fillcolor="#D6EAF8", color="#1F4E79"];
  low    [label="Refuse: LOW_EVIDENCE\n(no answer is drafted)", fillcolor="#F5B7B1", color="#922B21"];
  gen    [label="3. Draft answer\nonly from the passages, with [n] citations", fillcolor="#FDEBD0", color="#B9770E"];
  split  [label="4. Split into single claims", fillcolor="#FDEBD0", color="#B9770E"];
  noclm  [label="Refuse: NO_CLAIMS\n(nothing checkable)", fillcolor="#F5B7B1", color="#922B21"];
  judge  [label="5. Judge each claim\nagainst the passages only", fillcolor="#FDEBD0", color="#B9770E"];
  check  [label="6. Python checks\ncited passage exists; numbers and links\nappear in the official text", fillcolor="#D6EAF8", color="#1F4E79"];
  sim    [label="Claim-evidence similarity\n(recorded, not used to decide)", style="rounded,filled,dashed", fillcolor="#EEF1F5", color="#5A6572"];
  policy [label="7. Decision policy (Python)", shape=diamond, fillcolor="#D6EAF8", color="#1F4E79"];
  ok     [label="PASS\nVerified", fillcolor="#C8E6C9", color="#1E6B2E"];
  repair [label="REPAIR\nrebuild from supported claims only", fillcolor="#FDEBD0", color="#B9770E"];
  recheck [label="Re-check the repaired answer\n(steps 4 to 6 again)", fillcolor="#FDEBD0", color="#B9770E"];
  part   [label="Partially verified", fillcolor="#FFE0B2", color="#B9770E"];
  block  [label="BLOCK\nRefuse: NOT_SUPPORTED", fillcolor="#F5B7B1", color="#922B21"];
  down   [label="Any model call fails\nRefuse: SERVICE_UNAVAILABLE\n(fail closed)", fillcolor="#E0E0E0", color="#5A6572"];

  q -> norm -> ret -> gate;
  gate -> low  [label="no"];
  gate -> gen  [label="yes"];
  gen -> split;
  split -> noclm [label="0 claims"];
  split -> judge;
  judge -> check;
  check -> sim [style=dashed, arrowhead=none];
  check -> policy;
  policy -> ok     [label="all supported"];
  policy -> repair [label="some unsupported"];
  policy -> block  [label="too many unsupported"];
  repair -> recheck;
  recheck -> part  [label="re-check passes"];
  recheck -> block [label="re-check fails"];
  judge -> down [style=dotted];
}
"""

ELIGIBILITY_DIAGRAM = r"""
digraph elig {
  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.4;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.14,0.07"];
  edge [fontname="Helvetica", fontsize=10, color="#5A6572"];

  text  [label="Free text\n\"35 year old farmer from Bihar\"", fillcolor="#FFFFFF", color="#5A6572"];
  form  [label="Form fields", fillcolor="#FFFFFF", color="#5A6572"];
  llm   [label="Language model\nextracts fields\n(blank if not stated)", fillcolor="#FDEBD0", color="#B9770E"];
  rules [label="Python rule engine\ncompares fields with\nthe scheme's rules", fillcolor="#D6EAF8", color="#1F4E79"];
  out   [label="ELIGIBLE\nNOT_ELIGIBLE\nNEED_MORE_INFO\n(with a reason per rule)", fillcolor="#C8E6C9", color="#1E6B2E"];

  text -> llm [label="1 LLM call"];
  llm -> rules;
  form -> rules [label="no LLM"];
  rules -> out;
}
"""


def _live(health_data: dict, key: str, template: str = "{}") -> str:
    """A live config value from /health, or a plain 'not reported' for older servers."""
    value = health_data.get(key)
    return "not reported by this server" if value is None else template.format(value)


with how_tab:
    st.markdown("#### How an answer is produced and checked")
    st.markdown(
        "**Amber** steps are done by a language model. **Blue** steps are plain "
        "Python code with fixed rules. A language model is used for language; "
        "everything that decides what is shown is code."
    )
    st.graphviz_chart(ASK_DIAGRAM, width="stretch")

    st.markdown("##### The stages in plain words")
    st.markdown(
        f"""
1. **Normalise.** The question is rewritten in English and its language is detected
   (one model call). The official documents are in English, so searching them in
   romanised Hindi would miss the right passage.
2. **Retrieve.** The question is embedded with `{_live(health, 'embedding_model')}`
   and the closest {_live(health, 'retrieval_top_k')} official passages are fetched.
   If even the best one scores below **{_live(health, 'retrieval_min_score')}**, the
   system stops here and refuses — no model is asked to improvise.
3. **Draft.** `{_live(health, 'llm_model')}` writes an answer using only the retrieved
   passages and marks each statement with a citation number.
4. **Split and judge.** The answer is split into single claims; each claim is judged
   against the passages alone as supported or not supported.
5. **Python checks.** Code confirms that every cited passage really exists, and that
   every number of 10 or more, every percentage or decimal, and every link in the
   answer appears in the official text.
6. **Decide.** All supported means the answer is shown as *Verified*. Some
   unsupported means the answer is rebuilt from the supported claims and checked
   again (*Partially verified* only if the re-check passes). If more than
   **{_live(health, 'guardrail_block_threshold', '{:.0%}')}** of the claims are unsupported the answer is
   withheld.
"""
    )

    st.markdown("##### Why an answer can be refused")
    st.markdown(
        "- **low_evidence** — no official passage matched the question well enough. "
        "Nothing was drafted.\n"
        "- **not_supported** — a draft was written, but too many claims were not backed "
        "by the official text.\n"
        "- **no_claims** — the draft stated no checkable fact, so there was nothing to verify.\n"
        "- **service_unavailable** — a language-model call failed (rate limit, quota or "
        "outage). The answer could not be checked, so it is not shown. This is a temporary "
        "service issue, not missing information."
    )

    st.markdown("##### Eligibility")
    st.graphviz_chart(ELIGIBILITY_DIAGRAM, width="stretch")
    st.markdown(
        "The language model never decides eligibility. At most it reads your free text "
        "into fields; a rule engine written in Python then compares them with each "
        "scheme's rules and gives a reason for every rule. Missing information produces "
        "*need more info*, never a guess."
    )


# ==========================================================================
# Tab 5 — System
# ==========================================================================

with system_tab:
    st.markdown("#### System")
    st.button("Refresh", key="refresh_system", on_click=st.cache_data.clear)

    metrics, metrics_error, _status = _load_metrics()
    if metrics_error:
        st.error(metrics_error)
    else:
        metrics = metrics or {}
        columns = st.columns(5)
        columns[0].metric("LLM calls", f"{metrics.get('llm_calls_total', 0):,}")
        columns[1].metric("Failed calls", f"{metrics.get('llm_calls_failed', 0):,}")
        columns[2].metric("Avg latency", f"{metrics.get('avg_latency_ms', 0):,.0f} ms")
        columns[3].metric("p95 latency", f"{metrics.get('p95_latency_ms', 0):,.0f} ms")
        tokens = metrics.get("total_prompt_tokens", 0) + metrics.get("total_completion_tokens", 0)
        columns[4].metric("Tokens", f"{tokens:,}")

        since = metrics.get("log_started_at")
        st.caption(
            "Read live from the server's SQLite call log"
            + (f" since {since}" if since else "")
            + "; resets when the app is redeployed."
        )

        by_prompt = metrics.get("by_prompt") or {}
        st.markdown("**Calls by prompt**")
        if by_prompt:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Prompt": name,
                            "Calls": stats.get("calls", 0),
                            "Failed": stats.get("failed", 0),
                            "Avg latency (ms)": stats.get("avg_latency_ms", 0),
                            "Prompt tokens": stats.get("prompt_tokens", 0),
                            "Completion tokens": stats.get("completion_tokens", 0),
                        }
                        for name, stats in by_prompt.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        elif metrics.get("calls_by_prompt"):
            st.dataframe(
                pd.DataFrame(
                    [{"Prompt": name, "Calls": count} for name, count in metrics["calls_by_prompt"].items()]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No model calls have been logged yet.")

        left, right = st.columns(2)
        with left:
            st.markdown("**Guardrail decisions**")
            counts = metrics.get("guardrail_counts") or {}
            if sum(counts.values()) > 0:
                st.bar_chart(pd.DataFrame({"answers": counts}))
            else:
                st.caption("No guardrail decisions have been recorded yet.")
        with right:
            st.markdown("**Answer outcomes**")
            status_counts = metrics.get("answer_status_counts") or {}
            if status_counts:
                st.dataframe(
                    pd.DataFrame(
                        [{"Outcome": name, "Answers": count} for name, count in status_counts.items()]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption(
                    f"{metrics.get('questions_answered', 0)} answered, "
                    f"{metrics.get('questions_refused', 0)} refused."
                )
            if "cache_hits" in metrics:
                st.metric("Cache hits", metrics["cache_hits"])

    with st.expander("Health and backend details"):
        st.markdown(f"Backend URL: `{API_URL}`")
        st.json(health)


# ==========================================================================
# Footer
# ==========================================================================

st.divider()
st.caption(
    "**Disclaimer** — SchemeSure is an independent educational project. It is "
    "**not** an official Government of India service and is not affiliated with "
    "any ministry. Answers are generated from official documents fetched on the "
    "date shown and may be out of date or incomplete. Always confirm on the "
    "official scheme website before applying or making any decision. Nothing "
    "here is legal or financial advice."
)
