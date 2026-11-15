"""
Every data shape in the system, as Pydantic models.

WHY ONE FILE: these models are the contract between the layers (retriever ->
generator -> guardrail -> API -> UI). Keeping them together means you can read
the whole data flow in one sitting, and FastAPI gets request/response validation
and OpenAPI docs for free.

Naming convention: `*Request` / `*Response` are HTTP-facing; everything else is
internal and shared between core modules.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# ==========================================================================
# Enums — small closed vocabularies, so typos become validation errors
# ==========================================================================


class Language(str, Enum):
    """The three input languages we handle."""

    ENGLISH = "en"
    HINDI = "hi"
    # Hindi written in Latin script ("kya hai", "kisko milega"). Common in India
    # and worth detecting separately: we answer back in the same style.
    HINGLISH = "hinglish"


class Verdict(str, Enum):
    """Whether a single claim is backed by the retrieved evidence."""

    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class GuardrailDecision(str, Enum):
    """What the guardrail decided to do with a drafted answer."""

    PASS = "PASS"          # every claim supported; ship it unchanged
    REPAIR = "REPAIR"      # some claims unsupported; rebuild from supported ones
    BLOCK = "BLOCK"        # too much unsupported; refuse instead of answering
    SKIPPED = "SKIPPED"    # guardrail did not run (disabled, or refused earlier)


class AnswerStatus(str, Enum):
    """
    What the user is told about trustworthiness. This drives the UI badge.

    Deliberately separate from GuardrailDecision: the guardrail's internal
    decision is an implementation detail, this is the user-facing promise.
    """

    VERIFIED = "verified"                      # every claim checked against evidence
    PARTIALLY_VERIFIED = "partially_verified"  # unsupported claims were removed
    REFUSED = "refused"                        # not enough official information
    # Guardrail was switched off. Used only by the evaluation baseline, so that a
    # run with no verification is never labelled "verified".
    UNVERIFIED = "unverified"


class RefusalReason(str, Enum):
    """
    WHY an answer was refused. Lets the UI (and the eval) tell "we have no
    official information" apart from "the checking service was down".
    """

    LOW_EVIDENCE = "low_evidence"              # best retrieval score below threshold; no answer drafted
    NOT_SUPPORTED = "not_supported"            # guardrail BLOCK: too much of the draft unsupported
    NO_CLAIMS = "no_claims"                    # draft asserted nothing checkable (model declined)
    SERVICE_UNAVAILABLE = "service_unavailable"  # an LLM call failed (rate limit / quota / outage) -> fail closed


class EligibilityStatus(str, Enum):
    """Outcome of the rule-based eligibility check for one scheme."""

    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    # The profile is missing a field a rule needs. Honest third state: better
    # than guessing, and it tells the user exactly what to supply.
    NEED_MORE_INFO = "NEED_MORE_INFO"


class Gender(str, Enum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"


# ==========================================================================
# Retrieval
# ==========================================================================


class ChunkMetadata(BaseModel):
    """Provenance carried alongside every chunk, so citations are never invented."""

    scheme_id: str
    scheme_name: str
    section: str
    source_url: str
    last_verified: str
    ministry: str = ""


class RetrievedChunk(BaseModel):
    """One passage returned by the vector search."""

    chunk_id: str = Field(description="Deterministic ID; also the citation key.")
    text: str
    score: float = Field(description="Cosine similarity in [-1, 1]; higher is closer.")
    metadata: ChunkMetadata


class RetrievalResult(BaseModel):
    """Everything the retriever learned about one query."""

    query: str = Field(description="The (normalized) query actually embedded.")
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    best_score: float = 0.0
    # True when even the best chunk is below the configured threshold, meaning
    # we should refuse rather than let the LLM improvise.
    low_evidence: bool = True
    threshold: float = 0.0


class NormalizedQuery(BaseModel):
    """Output of the query normalizer: one LLM call, strictly shaped."""

    normalized_query: str = Field(description="The question rewritten in English.")
    language: Language = Field(description="Language the user actually wrote in.")
    # Set when the LLM call failed and we fell back to the raw query. Surfaced in
    # logs so a degraded run is visible rather than silent.
    fallback_used: bool = False


# ==========================================================================
# Citations and claims
# ==========================================================================


class Citation(BaseModel):
    """A numbered source shown under an answer. [1], [2], ... map to these."""

    index: int = Field(description="The [n] marker used in the answer text.")
    chunk_id: str
    scheme_id: str
    scheme_name: str
    section: str
    source_url: str
    last_verified: str


class ClaimCheck(BaseModel):
    """One atomic claim plus the verdict the judge returned for it."""

    claim: str
    verdict: Verdict
    evidence_chunk_id: str | None = Field(
        default=None,
        description="Chunk the judge says supports this claim.",
    )
    # Set by Python, never by the LLM: does evidence_chunk_id actually exist in
    # the retrieved set? A judge that cites a non-existent chunk is hallucinating
    # about its own evidence, so we downgrade the claim regardless of its verdict.
    citation_valid: bool = True
    note: str = ""
    # Independent second signal (semantic similarity between the claim and the
    # evidence), recorded for calibration only. Never used for the decision.
    # None = not computed (embedding unavailable, or the cited chunk is unknown).
    evidence_similarity: float | None = None
    best_similarity: float | None = None
    best_similarity_chunk_id: str | None = None


class GuardrailReport(BaseModel):
    """The full audit trail for one answer. Shown in the UI's "How was this verified?"."""

    decision: GuardrailDecision
    claims: list[ClaimCheck] = Field(default_factory=list)
    supported_count: int = 0
    unsupported_count: int = 0
    unsupported_ratio: float = 0.0
    block_threshold: float = 0.0
    # Human-readable trace of what happened and why, including failures.
    notes: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    # Set when split_claims / judge_claims raised LLMError. The decision stays
    # BLOCK (fail closed) but the pipeline reports SERVICE_UNAVAILABLE, not
    # "not supported": the answer was never judged wrong, it was never judged.
    verification_error: str | None = None
    # Wall-clock time per verification stage, in ms. Keys: split_claims,
    # judge_claims, similarity. Lets the pipeline build a stage trace without
    # the guardrail knowing about trace models.
    timings_ms: dict[str, int] = Field(default_factory=dict)
    # Numbers / links in the answer that appear nowhere in the retrieved text
    # (deterministic Python check, independent of the LLM judge).
    ungrounded_tokens: list[str] = Field(default_factory=list)


# ==========================================================================
# /ask
# ==========================================================================


class AskRequest(BaseModel):
    # No max_length here on purpose: ONE limit (settings.max_question_length)
    # is enforced in app/main.py and answered with ONE status code (400).
    question: str = Field(min_length=1)
    # Optional override; normally the language is detected.
    language: Language | None = None

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class EvidencePassage(BaseModel):
    """One retrieved passage exactly as the generator and the judge saw it."""

    index: int            # the [n] position the generator/judge saw
    chunk_id: str
    scheme_id: str
    scheme_name: str
    section: str
    source_url: str
    score: float          # cosine similarity
    above_threshold: bool
    text: str


class RetrievalSummary(BaseModel):
    """What retrieval found for a query, including why it was (not) enough."""

    query: str
    top_k: int
    threshold: float
    best_score: float
    low_evidence: bool
    passages: list[EvidencePassage] = Field(default_factory=list)


class StageTrace(BaseModel):
    """One pipeline stage that actually ran, with its real duration."""

    # normalize, retrieve, generate, split_claims, judge_claims, similarity, repair, cache
    stage: str
    duration_ms: int
    llm_calls: int = 0
    # Short human-readable result, e.g. "language=hinglish", "best 0.612 >= 0.45",
    # "4/5 supported -> REPAIR".
    outcome: str = ""


class AskResponse(BaseModel):
    question: str
    normalized_query: str
    language: Language
    answer: str
    status: AnswerStatus
    citations: list[Citation] = Field(default_factory=list)
    guardrail: GuardrailReport
    latency_ms: int = 0
    llm_calls: int = 0
    # None unless status == refused.
    refusal_reason: RefusalReason | None = None
    # True when the query normalizer fell back to the raw question (LLM failed).
    normalization_fallback: bool = False
    retrieval: RetrievalSummary | None = None
    trace: list[StageTrace] = Field(default_factory=list)
    cached: bool = False
    # The second guardrail pass over a REPAIRED answer. A repaired answer ships
    # only if this report is PASS with at least one claim.
    repair_report: GuardrailReport | None = None


# ==========================================================================
# /eligibility
# ==========================================================================


class UserProfile(BaseModel):
    """
    What we could extract from the user's free text. Every field is optional:
    a missing field must stay None so the rule engine can answer
    NEED_MORE_INFO instead of silently assuming a value.
    """

    age: int | None = Field(default=None, ge=0, le=120)
    annual_income: int | None = Field(
        default=None, ge=0,
        description="Total yearly household/family income in rupees.",
    )
    state: str | None = None
    occupation: str | None = None
    gender: Gender | None = None
    category: str | None = Field(default=None, description="e.g. SC, ST, OBC, General.")

    @field_validator("state", "occupation", "category")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        """Treat empty strings from the LLM as "not provided"."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class RuleCheck(BaseModel):
    """The result of testing one rule against the profile. One line of reasoning."""

    rule: str = Field(description="Which rule was tested, e.g. 'min_age'.")
    passed: bool | None = Field(
        default=None,
        description="True=passed, False=failed, None=could not tell (missing data).",
    )
    reason: str
    official_text: str = Field(
        default="",
        description="The official sentence this rule came from, when available.",
    )


class SchemeEligibility(BaseModel):
    """Verdict for one scheme, with the per-rule reasoning that produced it."""

    scheme_id: str
    scheme_name: str
    status: EligibilityStatus
    checks: list[RuleCheck] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    # Official conditions the flat rule schema cannot encode. Always shown, so
    # an ELIGIBLE verdict is never mistaken for a guarantee.
    conditions_to_verify: list[str] = Field(default_factory=list)
    source_url: str = ""


class EligibilityRequest(BaseModel):
    # Length limit enforced in app/main.py (same single limit as /ask).
    description: str = Field(min_length=1)

    @field_validator("description")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be blank")
        return value.strip()


class EligibilityCheckRequest(BaseModel):
    """Structured input for POST /eligibility/check (no LLM involved at all)."""

    profile: UserProfile


class EligibilityResponse(BaseModel):
    description: str
    profile: UserProfile
    results: list[SchemeEligibility] = Field(default_factory=list)
    eligible_count: int = 0
    need_more_info_count: int = 0
    not_eligible_count: int = 0
    latency_ms: int = 0
    llm_calls: int = 0


# ==========================================================================
# /schemes, /health, /metrics
# ==========================================================================


class SchemeInfo(BaseModel):
    """One row of the covered-schemes catalogue."""

    scheme_id: str
    name: str
    short_name: str = ""
    ministry: str = ""
    source_url: str
    last_verified: str
    sections: list[str] = Field(default_factory=list)
    has_rules: bool = False


class SectionDetail(BaseModel):
    """Full text of one section of a scheme's official document."""

    title: str
    text: str


class ReferenceLink(BaseModel):
    title: str
    url: str


class SchemeRulesView(BaseModel):
    """The machine-checkable rules of a scheme, as shown by GET /schemes/{id}."""

    min_age: int | None = None
    max_age: int | None = None
    max_annual_income: int | None = None
    allowed_states: list[str] | None = None
    allowed_occupations: list[str] | None = None
    gender: str | None = None
    category: str | None = None
    rule_sources: dict[str, str] = Field(default_factory=dict)
    unmodelled_conditions: list[str] = Field(default_factory=list)


class SchemeDetail(SchemeInfo):
    """A catalogue row plus the full official text, rules and index size."""

    sections_detail: list[SectionDetail] = Field(default_factory=list)
    reference_urls: list[ReferenceLink] = Field(default_factory=list)
    rules: SchemeRulesView | None = None
    chunks_indexed: int = 0


class EligibilityOptions(BaseModel):
    """The vocabularies the eligibility form may offer (GET /eligibility/options)."""

    occupations: list[str]
    genders: list[str]
    categories: list[str]


class HealthResponse(BaseModel):
    status: str
    schemes_loaded: int
    chunks_indexed: int
    embedding_model: str
    llm_model: str
    llm_key_configured: bool
    guardrail_enabled: bool
    # True only when the LLM pipeline was actually built (a key being present
    # does not guarantee that). The UI should enable Ask based on this.
    llm_ready: bool = False
    retrieval_top_k: int = 0
    retrieval_min_score: float = 0.0
    guardrail_block_threshold: float = 0.0
    answer_cache_enabled: bool = False


class GuardrailCounts(BaseModel):
    PASS: int = 0
    REPAIR: int = 0
    BLOCK: int = 0
    SKIPPED: int = 0


class PromptStats(BaseModel):
    """Per-prompt aggregate from the SQLite call log."""

    calls: int = 0
    failed: int = 0
    avg_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class MetricsResponse(BaseModel):
    """Operational counters, read back from the SQLite call log."""

    llm_calls_total: int = 0
    llm_calls_failed: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    calls_by_prompt: dict[str, int] = Field(default_factory=dict)
    guardrail_counts: GuardrailCounts = Field(default_factory=GuardrailCounts)
    questions_answered: int = 0
    questions_refused: int = 0
    by_prompt: dict[str, PromptStats] = Field(default_factory=dict)
    # Every answer_status event value (verified, refused, service_unavailable, ...).
    answer_status_counts: dict[str, int] = Field(default_factory=dict)
    cache_hits: int = 0
    # Timestamp of the first logged row, so a UI can say "since ...".
    log_started_at: str | None = None
