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


# ==========================================================================
# /ask
# ==========================================================================


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Optional override; normally the language is detected.
    language: Language | None = None

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


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
    annual_income: int | None = Field(default=None, ge=0, description="Rupees per year.")
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
    description: str = Field(min_length=1, max_length=2000)

    @field_validator("description")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be blank")
        return value.strip()


class EligibilityResponse(BaseModel):
    description: str
    profile: UserProfile
    results: list[SchemeEligibility] = Field(default_factory=list)
    eligible_count: int = 0
    need_more_info_count: int = 0
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


class HealthResponse(BaseModel):
    status: str
    schemes_loaded: int
    chunks_indexed: int
    embedding_model: str
    llm_model: str
    llm_key_configured: bool
    guardrail_enabled: bool


class GuardrailCounts(BaseModel):
    PASS: int = 0
    REPAIR: int = 0
    BLOCK: int = 0
    SKIPPED: int = 0


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
