"""
The HTTP layer. Thin on purpose.

This file does four things: wire up startup, validate input, call into `core/`,
and translate exceptions into status codes. There is no business logic here — the
answering pipeline lives in `core/generator.py` and the rules in
`core/eligibility.py`, so they can be tested without an HTTP client and reused by
the evaluation harness.

STARTUP: the embedding model (~120 MB) and the Chroma collection are loaded once,
in the lifespan handler. Loading them per request would add seconds of latency.

DEGRADED START: a missing `LLM_API_KEY` does not stop the server. It boots,
`/health` reports `llm_key_configured: false`, the read-only endpoints keep
working, and `/ask` returns a clear 503. This matters for the container: it must
come up and pass a health check even if a secret has not been set yet.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.llm_client import CallCounter, LLMClient, LLMError, init_db, read_metrics
from app.schemas import (
    AskRequest,
    AskResponse,
    EligibilityRequest,
    EligibilityResponse,
    EligibilityStatus,
    HealthResponse,
    MetricsResponse,
    SchemeInfo,
)
from core.eligibility import check_all_schemes, extract_profile, load_rules
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
    logger.error("LLM error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=502,
        content={"detail": f"The language model could not be reached: {exc}"},
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
    )


@app.get("/schemes", response_model=list[SchemeInfo], tags=["data"])
def list_schemes(request: Request) -> list[SchemeInfo]:
    """The schemes covered, with official source URLs and verification dates."""
    return request.app.state.schemes


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(request: Request, payload: AskRequest) -> AskResponse:
    """
    Answer a question about a covered scheme.

    The response carries the answer, its citations, and the full guardrail report
    (every claim and its verdict) so a client can show *why* an answer is
    trustworthy rather than just asserting that it is.
    """
    question = _check_length(payload.question, "question")
    pipeline = _require_pipeline(request)
    return pipeline.ask(question, language_override=payload.language)


@app.post("/eligibility", response_model=EligibilityResponse, tags=["qa"])
def eligibility(request: Request, payload: EligibilityRequest) -> EligibilityResponse:
    """
    Check a free-text self-description against every scheme's rules.

    One LLM call extracts the profile; every verdict after that is plain Python.
    """
    import time

    description = _check_length(payload.description, "description")
    pipeline = _require_pipeline(request)  # also guarantees the LLM is available

    started = time.perf_counter()
    counter = CallCounter()

    profile = extract_profile(pipeline.llm, description, counter=counter)
    results = check_all_schemes(profile, request.app.state.rules)

    return EligibilityResponse(
        description=description,
        profile=profile,
        results=results,
        eligible_count=sum(
            1 for result in results if result.status == EligibilityStatus.ELIGIBLE
        ),
        need_more_info_count=sum(
            1 for result in results if result.status == EligibilityStatus.NEED_MORE_INFO
        ),
        latency_ms=int((time.perf_counter() - started) * 1000),
        llm_calls=counter.count,
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
        "endpoints": ["/ask", "/eligibility", "/schemes", "/health", "/metrics"],
    }
