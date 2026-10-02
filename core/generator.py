"""
Answer generation, and the pipeline that wires the whole /ask flow together.

THE PIPELINE, IN ORDER
----------------------
    question
      -> normalize   (1 LLM call)  understand it, translate to English
      -> retrieve    (0 LLM calls) vector search + evidence-strength verdict
      -> REFUSE HERE if the evidence is weak. No LLM call is spent.
      -> generate    (1 LLM call)  draft an answer from the chunks only
      -> verify      (2 LLM calls) split into claims, judge each one
      -> repair      (1 LLM call)  only if some claims failed
      -> answer + citations + a full audit trail

So a normal answered question costs 4 LLM calls, a repaired one 5, and a refused
one either 0 (weak evidence) or 4 (drafted, then blocked). Those numbers are
measured per request and reported in the eval, rather than estimated.

WHY REFUSE BEFORE GENERATING
----------------------------
If the best retrieved chunk is below the similarity threshold, no amount of
prompting will make the answer grounded — there is nothing to ground it in. The
LLM would simply write something plausible. Refusing here is both safer and
cheaper, and it is why an out-of-scope question costs us nothing.
"""

from __future__ import annotations

import re
import time

from app.config import Settings, get_settings
from app.llm_client import CallCounter, LLMClient, LLMError, log_event
from app.prompts import (
    ANSWER_GENERATOR_SYSTEM,
    ANSWER_GENERATOR_USER,
    ANSWER_REPAIR_SYSTEM,
    ANSWER_REPAIR_USER,
    refusal_for,
)
from app.schemas import (
    AnswerStatus,
    AskResponse,
    Citation,
    GuardrailDecision,
    GuardrailReport,
    Language,
    RetrievalResult,
    RetrievedChunk,
    Verdict,
)
from core.guardrail import Guardrail
from core.query_normalizer import normalize_query
from core.retriever import Retriever, format_evidence

# Matches the [1] / [2] markers the generator is told to emit.
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


# ==========================================================================
# Citations
# ==========================================================================


def extract_cited_indices(answer: str) -> list[int]:
    """Return the [n] markers used in the answer, in first-appearance order."""
    seen: list[int] = []
    for match in _CITATION_RE.finditer(answer or ""):
        index = int(match.group(1))
        if index not in seen:
            seen.append(index)
    return seen


def build_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """
    Build the citation list shown under an answer.

    We list the passages the answer actually cited. If it cited none (or cited
    numbers outside the evidence range, which the guardrail will have caught
    separately), we fall back to the retrieved passages so the user can always
    see where the information was meant to come from.
    """
    if not chunks:
        return []

    cited = [index for index in extract_cited_indices(answer) if 1 <= index <= len(chunks)]
    indices = cited or list(range(1, len(chunks) + 1))

    citations: list[Citation] = []
    for index in indices:
        chunk = chunks[index - 1]
        citations.append(
            Citation(
                index=index,
                chunk_id=chunk.chunk_id,
                scheme_id=chunk.metadata.scheme_id,
                scheme_name=chunk.metadata.scheme_name,
                section=chunk.metadata.section,
                source_url=chunk.metadata.source_url,
                last_verified=chunk.metadata.last_verified,
            )
        )
    return citations


# ==========================================================================
# Generation steps
# ==========================================================================


def generate_answer(
    llm: LLMClient,
    question: str,
    language: Language,
    chunks: list[RetrievedChunk],
    counter: CallCounter | None = None,
) -> str:
    """Draft an answer grounded in `chunks`, in the user's language."""
    result = llm.complete(
        prompt_name="generate_answer",
        system=ANSWER_GENERATOR_SYSTEM,
        user=ANSWER_GENERATOR_USER.format(
            evidence=format_evidence(chunks),
            question=question,
            language=language.value,
        ),
        max_tokens=800,
        counter=counter,
    )
    return result.text.strip()


def repair_answer(
    llm: LLMClient,
    question: str,
    language: Language,
    chunks: list[RetrievedChunk],
    supported_claims: list[str],
    counter: CallCounter | None = None,
) -> str:
    """
    Rewrite the answer using only the claims that survived verification.

    Note what this is NOT: it is not "ask the model to try again". The model is
    handed a fixed list of verified facts and told to state those and nothing
    else, so the repaired answer cannot reintroduce the claim that just failed.
    """
    claims_block = "\n".join(
        f"{index}. {claim}" for index, claim in enumerate(supported_claims, start=1)
    )
    result = llm.complete(
        prompt_name="repair_answer",
        system=ANSWER_REPAIR_SYSTEM.format(language=language.value),
        user=ANSWER_REPAIR_USER.format(
            evidence=format_evidence(chunks),
            claims=claims_block,
            question=question,
            language=language.value,
        ),
        max_tokens=700,
        counter=counter,
    )
    return result.text.strip()


# ==========================================================================
# The pipeline
# ==========================================================================


class AnswerPipeline:
    """
    Orchestrates normalize -> retrieve -> generate -> verify -> repair.

    Built once at FastAPI startup. Holds the retriever (and therefore the loaded
    embedding model and open Chroma collection) so no request pays for setup.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or get_settings()
        self.guardrail = Guardrail(llm, self.settings)

    # ---- the public entry point ------------------------------------------

    def ask(self, question: str, language_override: Language | None = None) -> AskResponse:
        """Answer one question, end to end."""
        started = time.perf_counter()
        counter = CallCounter()

        # --- 1. understand the question -----------------------------------
        normalized = normalize_query(self.llm, question, counter=counter)
        language = language_override or normalized.language

        # --- 2. retrieve ---------------------------------------------------
        retrieval = self.retriever.search(normalized.normalized_query)

        # --- 3. refuse early if the evidence is too weak -------------------
        if retrieval.low_evidence:
            return self._refusal_response(
                question=question,
                normalized=normalized.normalized_query,
                language=language,
                retrieval=retrieval,
                counter=counter,
                started=started,
                blocked=False,
                reason=(
                    f"Best retrieval score {retrieval.best_score:.3f} is below the "
                    f"{retrieval.threshold:.2f} threshold; refused without "
                    "generating, so no answer was produced to verify."
                ),
            )

        # --- 4. draft an answer --------------------------------------------
        try:
            draft = generate_answer(
                self.llm, question, language, retrieval.chunks, counter=counter
            )
        except LLMError as exc:
            # Generation itself failed. Refuse rather than return a 500: the user
            # gets an honest message and the failure is already in the call log.
            return self._refusal_response(
                question=question,
                normalized=normalized.normalized_query,
                language=language,
                retrieval=retrieval,
                counter=counter,
                started=started,
                blocked=True,
                reason=f"Answer generation failed: {exc}",
            )

        # --- 5. verify -----------------------------------------------------
        report = self.guardrail.verify(draft, retrieval.chunks, counter=counter)

        # --- 6. act on the verdict -----------------------------------------
        answer = draft
        status = AnswerStatus.VERIFIED

        if report.decision == GuardrailDecision.SKIPPED:
            # Guardrail switched off: the eval baseline. Label it honestly.
            status = AnswerStatus.UNVERIFIED

        elif report.decision == GuardrailDecision.BLOCK:
            answer = refusal_for(language.value, blocked=True)
            status = AnswerStatus.REFUSED

        elif report.decision == GuardrailDecision.REPAIR:
            supported = [
                check.claim
                for check in report.claims
                if check.verdict == Verdict.SUPPORTED
            ]
            try:
                answer = repair_answer(
                    self.llm, question, language, retrieval.chunks, supported,
                    counter=counter,
                )
                status = AnswerStatus.PARTIALLY_VERIFIED
            except LLMError as exc:
                # Could not rebuild a clean answer, and the draft is known to
                # contain unsupported claims. Refuse rather than ship it.
                report.notes.append(f"Repair failed ({exc}); refusing instead.")
                answer = refusal_for(language.value, blocked=True)
                status = AnswerStatus.REFUSED

        citations = (
            [] if status == AnswerStatus.REFUSED
            else build_citations(answer, retrieval.chunks)
        )

        log_event("guardrail_decision", report.decision.value, settings=self.settings)
        log_event("answer_status", status.value, settings=self.settings)

        return AskResponse(
            question=question,
            normalized_query=normalized.normalized_query,
            language=language,
            answer=answer,
            status=status,
            citations=citations,
            guardrail=report,
            latency_ms=int((time.perf_counter() - started) * 1000),
            llm_calls=counter.count,
        )

    # ---- refusal helper --------------------------------------------------

    def _refusal_response(
        self,
        question: str,
        normalized: str,
        language: Language,
        retrieval: RetrievalResult,
        counter: CallCounter,
        started: float,
        blocked: bool,
        reason: str,
    ) -> AskResponse:
        """Build a refusal response with the reason recorded in the report."""
        report = GuardrailReport(
            decision=GuardrailDecision.SKIPPED,
            claims=[],
            block_threshold=self.settings.guardrail_block_threshold,
            notes=[reason],
            llm_calls=0,
        )

        log_event("guardrail_decision", GuardrailDecision.SKIPPED.value,
                  detail=reason, settings=self.settings)
        log_event("answer_status", AnswerStatus.REFUSED.value, settings=self.settings)

        return AskResponse(
            question=question,
            normalized_query=normalized,
            language=language,
            answer=refusal_for(language.value, blocked=blocked),
            status=AnswerStatus.REFUSED,
            citations=[],
            guardrail=report,
            latency_ms=int((time.perf_counter() - started) * 1000),
            llm_calls=counter.count,
        )
