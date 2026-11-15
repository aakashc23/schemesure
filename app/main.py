"""
The HTTP layer. Thin on purpose.

This file does four things: wire up startup, validate input, call into `core/`,
and translate exceptions into status codes. There is no business logic here — the
answering pipeline lives in `core/generator.py` and the rules in
`core/eligibility.py`, so they can be tested without an HTTP client and reused by
the evaluation harness.

STARTUP: the embedding model (about 470 MB of weights) and the Chroma collection
are loaded once, in the lifespan handler. Loading them per request would add
seconds of latency.

DEGRADED START: a missing `LLM_API_KEY` does not stop the server. It boots,
`/health` reports `llm_ready: false`, the read-only endpoints keep working, the
form-based eligibility check (pure Python) keeps working, and `/ask` returns a
clear 503. This matters for the container: it must come up and pass a health
check even if a secret has not been set yet.
"""

from __future__ import annotations

import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.llm_client import CallCounter, LLMClient, LLMError, init_db, log_event, read_metrics
from app.schemas import (
    AskRequest,
    AskResponse,
    EligibilityCheckRequest,
    EligibilityOptions,
    EligibilityRequest,
    EligibilityResponse,
    EligibilityStatus,
    Gender,
    HealthResponse,
    MetricsResponse,
    ReferenceLink,
    SchemeDetail,
    SchemeInfo,
    SchemeRulesView,
    SectionDetail,
    StageTrace,
    UserProfile,
)
from core.answer_cache import AnswerCache, make_key
from core.eligibility import (
    KNOWN_CATEGORIES,
    KNOWN_OCCUPATIONS,
    SchemeRules,
    check_all_schemes,
    extract_profile,
    load_rules,
    normalize_profile,
)
from core.generator import AnswerPipeline
from core.ingest import parse_front_matter, split_sections
from core.retriever import Retriever

logger = logging.getLogger("schemesure")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ==========================================================================
# Startup / shutdown
# ==========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the heavy, reusable objects once."""
    settings = get_settings()
    init_db(settings)

    logger.info("loading embedding model %s ...", settings.embedding_model)
    app.state.retriever = Retriever(settings)

    # Self-heal an empty index.
    #
    # The Docker image builds the index at build time, so this is a no-op there.
    # But hosts without a build step (Streamlit Community Cloud, a bare `git
    # clone`) start with no index at all, and an empty index means every question
    # gets refused — a failure that looks like a model problem rather than a
    # missing build step. Building it here costs ~20s once at startup.
    if app.state.retriever.count() == 0:
        logger.warning("index is empty — building it now from data/schemes ...")
        from core.ingest import build_index

        summary = build_index(settings, rebuild=False)
        logger.info("built %d chunks from %d schemes",
                    summary["chunks"], summary["schemes"])
        # Re-open so the retriever sees the newly written collection.
        app.state.retriever = Retriever(settings)

    logger.info("index ready: %d chunks", app.state.retriever.count())

    app.state.rules = load_rules(settings)
    logger.info("loaded rules for %d schemes", len(app.state.rules))

    app.state.schemes = _load_scheme_catalogue()
    # Full scheme documents (sections, rules, index size), read once like the
    # catalogue, so GET /schemes/{id} never touches the disk or Chroma per request.
    app.state.scheme_details = _load_scheme_details(
        app.state.retriever, app.state.rules
    )

    # In-memory answer cache (protects the free-tier token budget; /ask only).
    app.state.answer_cache = AnswerCache(
        ttl_seconds=settings.answer_cache_ttl_seconds,
        max_entries=settings.answer_cache_max_entries,
    )

    # The LLM client validates the key in its constructor. Keep the failure
    # local so the rest of the API still works without it.
    app.state.llm = None
    app.state.pipeline = None
    if settings.llm_api_key.strip():
        try:
            app.state.llm = LLMClient(settings)
            app.state.pipeline = AnswerPipeline(app.state.retriever, app.state.llm, settings)
            logger.info("LLM client ready (model=%s)", settings.llm_model)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM client unavailable: %s", exc)
    else:
        logger.warning("LLM_API_KEY is empty — /ask and /eligibility will return 503")

    yield

    logger.info("shutting down")


app = FastAPI(
    title="SchemeSure API",
    version="1.0.0",
    description=(
        "A hallucination-aware RAG assistant for Indian government scheme "
        "information. Every answer is verified claim-by-claim against the "
        "retrieved official evidence before it is returned, and eligibility is "
        "decided by rule-based Python rather than by the language model."
    ),
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ==========================================================================
# Helpers
# ==========================================================================


def _load_scheme_catalogue() -> list[SchemeInfo]:
    """Read the scheme documents once for GET /schemes."""
    settings = get_settings()
    catalogue: list[SchemeInfo] = []

    for path in sorted(settings.data_dir.glob("*.md")):
        try:
            front, body = parse_front_matter(io.open(path, encoding="utf-8").read())
        except OSError:
            continue
        catalogue.append(
            SchemeInfo(
                scheme_id=front.get("scheme_id", path.stem),
                name=front.get("name", path.stem),
                short_name=front.get("short_name", ""),
                ministry=front.get("ministry", ""),
                source_url=front.get("source_url", ""),
                last_verified=str(front.get("last_verified", "")),
                sections=[title for title, _ in split_sections(body)],
                has_rules=(settings.data_dir / f"{path.stem}.rules.json").exists(),
            )
        )

    catalogue.sort(key=lambda scheme: scheme.name)
    return catalogue


def _rules_view(rules: SchemeRules | None) -> SchemeRulesView | None:
    """Expose a scheme's machine-checkable rules (None if it has no rules file)."""
    if rules is None:
        return None
    return SchemeRulesView(
        min_age=rules.min_age,
        max_age=rules.max_age,
        max_annual_income=rules.max_annual_income,
        allowed_states=rules.allowed_states,
        allowed_occupations=rules.allowed_occupations,
        gender=rules.gender,
        category=rules.category,
        rule_sources=rules.rule_sources,
        unmodelled_conditions=rules.unmodelled_conditions,
    )


def _reference_links(raw) -> list[ReferenceLink]:
    """Front-matter `reference_urls` -> clean {title, url} pairs."""
    links: list[ReferenceLink] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and item.get("url"):
            links.append(ReferenceLink(title=item.get("title") or item["url"], url=item["url"]))
        elif isinstance(item, str) and item:
            links.append(ReferenceLink(title=item, url=item))
    return links


def _count_chunks(retriever: Retriever, scheme_id: str) -> int:
    """How many chunks of this scheme are in the vector index."""
    try:
        found = retriever.collection.get(where={"scheme_id": scheme_id}, include=[])
        return len(found.get("ids") or [])
    except Exception:  # noqa: BLE001 - a detail page must not fail over an index hiccup
        return 0


def _load_scheme_details(
    retriever: Retriever, rules: dict[str, SchemeRules]
) -> dict[str, SchemeDetail]:
    """Read every scheme document in full, once, for GET /schemes/{scheme_id}."""
    settings = get_settings()
    details: dict[str, SchemeDetail] = {}

    for path in sorted(settings.data_dir.glob("*.md")):
        try:
            front, body = parse_front_matter(io.open(path, encoding="utf-8").read())
        except OSError:
            continue

        scheme_id = front.get("scheme_id", path.stem)
        sections = split_sections(body)
        details[scheme_id] = SchemeDetail(
            scheme_id=scheme_id,
            name=front.get("name", path.stem),
            short_name=front.get("short_name", ""),
            ministry=front.get("ministry", ""),
            source_url=front.get("source_url", ""),
            last_verified=str(front.get("last_verified", "")),
            sections=[title for title, _ in sections],
            has_rules=(settings.data_dir / f"{path.stem}.rules.json").exists(),
            sections_detail=[SectionDetail(title=t, text=text) for t, text in sections],
            reference_urls=_reference_links(front.get("reference_urls")),
            rules=_rules_view(rules.get(scheme_id)),
            chunks_indexed=_count_chunks(retriever, scheme_id),
        )

    return details


def _require_pipeline(request: Request) -> AnswerPipeline:
    """Fetch the pipeline, or fail with a clear 503."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The language model is not configured. Set LLM_API_KEY in .env "
                "(locally) or as a Space secret (deployed), then restart."
            ),
        )
    return pipeline


def _check_length(text: str, field: str) -> str:
    """Enforce the configured input length limit."""
    limit = get_settings().max_question_length
    text = text.strip()
    if len(text) > limit:
        raise HTTPException(
            status_code=400,
            detail=f"{field} is too long ({len(text)} characters). The limit is {limit}.",
        )
    return text


# ==========================================================================
# Error handling
# ==========================================================================


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
    """
    Upstream model failure -> 502, not 500.

    The distinction matters when reading logs: 502 means "Groq failed", 500 means
    "we have a bug".
    """
    # Log the raw detail server-side; return only the kind-based public message
    # (raw provider text can carry organisation ids and usage figures).
    logger.error("LLM error on %s: %s (%s)", request.url.path, exc, exc.detail or exc.kind)
    return JSONResponse(
        status_code=502,
        content={"detail": f"The language model could not be reached: {exc.public_message}"},
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Log the detail, return a generic message (never leak internals to clients)."""
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ==========================================================================
# Endpoints
# ==========================================================================


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """
    Liveness plus the facts needed to debug a bad deployment: is the index
    populated, and is the LLM key present?
    """
    settings = get_settings()
    retriever: Retriever = request.app.state.retriever
    chunks = retriever.count()

    return HealthResponse(
        # "degraded" when the index is empty: the process is alive but cannot
        # answer anything, which a plain "ok" would hide.
        status="ok" if chunks > 0 else "degraded",
        schemes_loaded=len(request.app.state.schemes),
        chunks_indexed=chunks,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        llm_key_configured=bool(settings.llm_api_key.strip()),
        guardrail_enabled=settings.guardrail_enabled,
        # Whether the pipeline was actually built, not merely whether a key string
        # exists: a rejected or unusable key leaves the pipeline None.
        llm_ready=getattr(request.app.state, "pipeline", None) is not None,
        retrieval_top_k=settings.retrieval_top_k,
        retrieval_min_score=settings.retrieval_min_score,
        guardrail_block_threshold=settings.guardrail_block_threshold,
        # `is not None`, not truthiness: an EMPTY cache has len() == 0 and would
        # wrongly read as "no cache".
        answer_cache_enabled=(
            getattr(request.app.state, "answer_cache", None) is not None
            and request.app.state.answer_cache.enabled
        ),
    )


@app.get("/schemes", response_model=list[SchemeInfo], tags=["data"])
def list_schemes(request: Request) -> list[SchemeInfo]:
    """The schemes covered, with official source URLs and verification dates."""
    return request.app.state.schemes


@app.get("/schemes/{scheme_id}", response_model=SchemeDetail, tags=["data"])
def scheme_detail(request: Request, scheme_id: str) -> SchemeDetail:
    """One scheme in full: official text per section, rules, reference links."""
    detail = request.app.state.scheme_details.get(scheme_id)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=f"No scheme with id '{scheme_id}'. See GET /schemes for the covered ids.",
        )
    return detail


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(request: Request, payload: AskRequest) -> AskResponse:
    """
    Answer a question about a covered scheme.

    The response carries the answer, its citations, the retrieved evidence, a
    per-stage trace, and the full guardrail report (every claim and its verdict)
    so a client can show *why* an answer is trustworthy rather than just
    asserting that it is.

    Repeated identical questions are served from a small in-memory cache (zero
    LLM calls). The cache lives here, not in the pipeline, so the evaluation
    harness (which calls the pipeline directly) is never served stale results.
    """
    started = time.perf_counter()
    question = _check_length(payload.question, "question")
    pipeline = _require_pipeline(request)

    cache: AnswerCache | None = getattr(request.app.state, "answer_cache", None)
    key = make_key(
        question,
        payload.language.value if payload.language else None,
        pipeline.settings.guardrail_enabled,
    )

    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            duration = int((time.perf_counter() - started) * 1000)
            hit.cached = True
            hit.latency_ms = duration
            hit.llm_calls = 0          # this request made no LLM call
            hit.trace = [
                StageTrace(stage="cache", duration_ms=duration, outcome="served from cache"),
                *hit.trace,
            ]
            # Only the hit is logged. The guardrail decision is NOT logged again:
            # the check ran once, and counting it twice would inflate /metrics.
            log_event("cache", "hit", settings=pipeline.settings)
            return hit

    response = pipeline.ask(question, language_override=payload.language)

    # A response built on a degraded query rewrite is not worth replaying for an
    # hour; service-unavailable refusals are rejected by the cache itself.
    if cache is not None and not response.normalization_fallback:
        cache.put(key, response)
    return response


def _eligibility_response(
    description: str,
    profile: UserProfile,
    rules: dict[str, SchemeRules],
    started: float,
    llm_calls: int,
) -> EligibilityResponse:
    """Run the pure-Python rule engine and package the verdicts with their counts."""
    results = check_all_schemes(profile, rules)

    def count(status: EligibilityStatus) -> int:
        return sum(1 for result in results if result.status == status)

    return EligibilityResponse(
        description=description,
        profile=profile,
        results=results,
        eligible_count=count(EligibilityStatus.ELIGIBLE),
        need_more_info_count=count(EligibilityStatus.NEED_MORE_INFO),
        not_eligible_count=count(EligibilityStatus.NOT_ELIGIBLE),
        latency_ms=int((time.perf_counter() - started) * 1000),
        llm_calls=llm_calls,
    )


@app.post("/eligibility", response_model=EligibilityResponse, tags=["qa"])
def eligibility(request: Request, payload: EligibilityRequest) -> EligibilityResponse:
    """
    Check a free-text self-description against every scheme's rules.

    One LLM call extracts the profile; every verdict after that is plain Python.
    If that one call fails the answer is a 503, never an empty profile that would
    pass off an outage as "you told us nothing". POST /eligibility/check does the
    same check from structured fields with no LLM at all.
    """
    description = _check_length(payload.description, "description")
    pipeline = _require_pipeline(request)  # also guarantees the LLM is available

    started = time.perf_counter()
    counter = CallCounter()
    counter.set_deadline(get_settings().request_deadline_seconds)

    try:
        profile = extract_profile(pipeline.llm, description, counter=counter)
    except LLMError as exc:
        logger.error("profile extraction failed (%s): %s", exc.kind, exc.detail or exc)
        raise HTTPException(
            status_code=503,
            detail=(
                f"{exc.public_message} Your description could not be read right now. "
                "The form-based check (POST /eligibility/check) still works without "
                "the language model."
            ),
        ) from exc

    return _eligibility_response(
        description, profile, request.app.state.rules, started, counter.count
    )


@app.post("/eligibility/check", response_model=EligibilityResponse, tags=["qa"])
def eligibility_check(request: Request, payload: EligibilityCheckRequest) -> EligibilityResponse:
    """
    Check a STRUCTURED profile against every scheme's rules. No LLM involved.

    Works even when the language model is down or no key is configured, which is
    the point of keeping eligibility in plain Python. The occupation goes through
    the same whitelist and synonyms as the free-text path; an unknown occupation
    becomes "not provided" (NEED_MORE_INFO), never a false NOT_ELIGIBLE.
    """
    started = time.perf_counter()
    profile = normalize_profile(payload.profile)
    return _eligibility_response("", profile, request.app.state.rules, started, 0)


@app.get("/eligibility/options", response_model=EligibilityOptions, tags=["data"])
def eligibility_options() -> EligibilityOptions:
    """The vocabularies the eligibility form can offer (same ones the engine uses)."""
    return EligibilityOptions(
        occupations=list(KNOWN_OCCUPATIONS),
        genders=sorted(gender.value for gender in Gender),
        categories=list(KNOWN_CATEGORIES),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
def metrics() -> MetricsResponse:
    """
    Operational counters read back from the SQLite call log.

    Real measurements, not in-memory counters: they survive a restart and are
    what the evaluation numbers are derived from.
    """
    return MetricsResponse(**read_metrics())


@app.get("/", include_in_schema=False)
def root() -> dict:
    """Point a curious visitor at the docs."""
    return {
        "name": "SchemeSure API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/ask",
            "/eligibility",
            "/eligibility/check",
            "/eligibility/options",
            "/schemes",
            "/schemes/{scheme_id}",
            "/health",
            "/metrics",
        ],
    }
