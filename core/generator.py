"""
Answer generation, and the pipeline that wires the whole /ask flow together.

THE PIPELINE, IN ORDER
----------------------
    question
      -> normalize   (1 LLM call)  understand it, translate to English
      -> retrieve    (0 LLM calls) vector search + evidence-strength verdict
      -> REFUSE HERE if the evidence is weak. No further LLM call is spent.
      -> generate    (1 LLM call)  draft an answer from the chunks only
      -> verify      (2 LLM calls) split into claims, judge each one
      -> repair      (1 LLM call)  only if some claims failed
      -> re-verify   (2 LLM calls) only after a repair: the repaired text is
                                   itself checked before anyone sees it
      -> answer + citations + a full audit trail and per-stage trace

So a normal answered question costs 4 LLM calls, a repaired one 7, and a refused
one either 1 (weak evidence: the normalizer ran, nothing else) or 4 (drafted,
then blocked). Normalization ALWAYS runs, so even an off-topic question costs one
call. Those numbers are measured per request and reported in the eval, rather
than estimated.

WHY REFUSE BEFORE GENERATING
----------------------------
If the best retrieved chunk is below the similarity threshold, no amount of
prompting will make the answer grounded — there is nothing to ground it in. The
LLM would simply write something plausible. Refusing here is both safer and
cheaper: an out-of-scope question stops after the single normalization call.

WHY EVERY REFUSAL CARRIES A REASON
----------------------------------
"We have no official information" and "the model service was down" must never
look the same to the user. Every refusal therefore has a `RefusalReason`. Any
LLM failure (rate limit, quota, outage, timeout) fails CLOSED — no unchecked text
is ever shown — but is reported as SERVICE_UNAVAILABLE, not as missing
information.
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
    refusal_message,
)
from app.schemas import (
    AnswerStatus,
    AskResponse,
    Citation,
    ClaimCheck,
    EvidencePassage,
    GuardrailDecision,
    GuardrailReport,
    Language,
    NormalizedQuery,
    RefusalReason,
    RetrievalResult,
    RetrievalSummary,
    RetrievedChunk,
    StageTrace,
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


def build_citations(
    answer: str,
    chunks: list[RetrievedChunk],
    verified_chunk_ids: list[str] | None = None,
) -> list[Citation]:
    """
    Build the citation list shown under an answer.

    The guardrail's verified chunk ids are the AUTHORITY: they are what Python
    validated as really containing the claims. The answer's own [n] markers are
    only the model's claim about where a fact came from, and a model can cite the
    wrong passage (audit finding M2). So:

    1. When verified ids exist, keep only the [n] markers whose chunk is among
       them. If none of the markers survive, cite the verified chunks themselves.
       Never fall back to "all retrieved passages" for an answer that has
       verified ids — that would imply evidence we did not verify.
    2. When there are no verified ids at all (guardrail switched off, the
       evaluation baseline), fall back to the answer's markers, and only if there
       are none, to every retrieved passage.
    """
    if not chunks:
        return []

    position_of = {chunk.chunk_id: index for index, chunk in enumerate(chunks, start=1)}

    verified_positions: list[int] = []
    for chunk_id in verified_chunk_ids or []:
        index = position_of.get(chunk_id)
        if index is not None and index not in verified_positions:
            verified_positions.append(index)

    cited = [index for index in extract_cited_indices(answer) if 1 <= index <= len(chunks)]

    if verified_positions:
        kept = [index for index in cited if index in verified_positions]
        # Keep the model's own markers that were verified, then ADD every other
        # verified chunk so a fact verified against a differently-numbered passage
        # still shows its source. Never "all retrieved".
        indices = kept + [p for p in verified_positions if p not in kept]
    else:
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
    query: str,
    language: Language,
    chunks: list[RetrievedChunk],
    supported: list[ClaimCheck],
    counter: CallCounter | None = None,
) -> str:
    """
    Rewrite the answer using only the claims that survived verification.

    Note what this is NOT: it is not "ask the model to try again". The model is
    handed a fixed list of verified facts, each tagged with the [n] of the
    passage the judge verified it against, and told to state those and nothing
    else.

    Two safety details:
      - `query` is the NORMALIZED ENGLISH query, never the user's raw text, so an
        instruction injected into the question gets no second route to the output
        (audit finding H1).
      - The docstring's old promise that a repaired answer "cannot reintroduce"
        the failed claim was not enforced by code. It is now: the pipeline runs
        the full guardrail on the repaired text before showing it.
    """
    position_of = {chunk.chunk_id: index for index, chunk in enumerate(chunks, start=1)}

    lines: list[str] = []
    for number, check in enumerate(supported, start=1):
        position = position_of.get(check.evidence_chunk_id or "")
        marker = f" [{position}]" if position is not None else ""
        lines.append(f"{number}. {check.claim}{marker}")

    result = llm.complete(
        prompt_name="repair_answer",
        system=ANSWER_REPAIR_SYSTEM.format(language=language.value),
        user=ANSWER_REPAIR_USER.format(
            evidence=format_evidence(chunks),
            claims="\n".join(lines),
            query=query,
            language=language.value,
        ),
        max_tokens=700,
        counter=counter,
    )
    return result.text.strip()


# ==========================================================================
# Evidence summary and trace helpers
# ==========================================================================


def build_retrieval_summary(retrieval: RetrievalResult, top_k: int) -> RetrievalSummary:
    """
    Describe what retrieval found, exactly as the generator and judge saw it.

    The `index` of each passage is the [n] the prompts used, so a citation in the
    answer can be traced to its passage; `above_threshold` shows which passages
    would individually clear the evidence bar.
    """
    return RetrievalSummary(
        query=retrieval.query,
        top_k=top_k,
        threshold=retrieval.threshold,
        best_score=retrieval.best_score,
        low_evidence=retrieval.low_evidence,
        passages=[
            EvidencePassage(
                index=index,
                chunk_id=chunk.chunk_id,
                scheme_id=chunk.metadata.scheme_id,
                scheme_name=chunk.metadata.scheme_name,
                section=chunk.metadata.section,
                source_url=chunk.metadata.source_url,
                score=chunk.score,
                above_threshold=chunk.score >= retrieval.threshold,
                text=chunk.text,
            )
            for index, chunk in enumerate(retrieval.chunks, start=1)
        ],
    )


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


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
        # The guardrail gets the retriever's embedder so it can record the
        # independent claim-evidence similarity signal (never used to decide).
        self.guardrail = Guardrail(llm, self.settings, embed_fn=retriever.embed)

    # ---- the public entry point ------------------------------------------

    def ask(
        self,
        question: str,
        language_override: Language | None = None,
        normalized: NormalizedQuery | None = None,
    ) -> AskResponse:
        """
        Answer one question, end to end.

        `normalized` lets a caller supply an already-computed normalization (the
        evaluation harness caches it) so the guardrail-OFF and guardrail-ON runs
        see byte-identical retrieval and the normalizer is paid for once.
        """
        started = time.perf_counter()
        counter = CallCounter()
        # One wall-clock budget for the whole request, so the server always
        # answers (honestly) before the UI's own HTTP timeout fires.
        counter.set_deadline(self.settings.request_deadline_seconds)
        trace: list[StageTrace] = []

        def mark(stage: str, t0: float, names_before: int, outcome: str = "") -> None:
            """Append a trace entry; llm_calls = calls that succeeded in this stage."""
            trace.append(
                StageTrace(
                    stage=stage,
                    duration_ms=_ms(t0),
                    llm_calls=len(counter.prompt_names) - names_before,
                    outcome=outcome,
                )
            )

        # --- 1. understand the question -----------------------------------
        t0, n0 = time.perf_counter(), len(counter.prompt_names)
        if normalized is None:
            normalized = normalize_query(self.llm, question, counter=counter)
            outcome = f"language={normalized.language.value}"
            if normalized.fallback_used:
                outcome += "; model unavailable, used the raw question"
        else:
            outcome = (
                f"language={normalized.language.value}; reused a precomputed "
                "normalization (not re-run)"
            )
        mark("normalize", t0, n0, outcome)
        language = language_override or normalized.language

        # --- 2. retrieve ---------------------------------------------------
        t0, n0 = time.perf_counter(), len(counter.prompt_names)
        retrieval = self.retriever.search(normalized.normalized_query)
        summary = build_retrieval_summary(retrieval, self.settings.retrieval_top_k)
        comparison = "<" if retrieval.low_evidence else ">="
        mark(
            "retrieve", t0, n0,
            f"best {retrieval.best_score:.3f} {comparison} {retrieval.threshold:.2f}",
        )

        def refuse(
            reason: RefusalReason,
            note: str,
            report: GuardrailReport | None = None,
            repair_report: GuardrailReport | None = None,
        ) -> AskResponse:
            return self._refusal_response(
                question=question, normalized=normalized, language=language,
                summary=summary, counter=counter, started=started, trace=trace,
                reason=reason, note=note, report=report, repair_report=repair_report,
            )

        # --- 3. refuse early if the evidence is too weak -------------------
        if retrieval.low_evidence:
            if normalized.fallback_used:
                # The query rewrite failed, so retrieval ran on the raw question
                # (poor for Hindi/Hinglish: "PM Kisan ka paisa kitna milta hai?"
                # scores 0.436 raw but ~0.57 once translated). Low scores here do
                # NOT show that we lack the information, so we must not say so.
                return refuse(
                    RefusalReason.SERVICE_UNAVAILABLE,
                    "The language model could not rewrite the question, so the "
                    f"search used the raw text (best score {retrieval.best_score:.3f}, "
                    f"threshold {retrieval.threshold:.2f}). That weak score does not "
                    "prove the information is missing; please retry.",
                )
            return refuse(
                RefusalReason.LOW_EVIDENCE,
                f"Best retrieval score {retrieval.best_score:.3f} is below the "
                f"{retrieval.threshold:.2f} threshold; refused without "
                "generating, so no answer was produced to verify.",
            )

        # --- 4. draft an answer --------------------------------------------
        t0, n0 = time.perf_counter(), len(counter.prompt_names)
        try:
            draft = generate_answer(
                self.llm, question, language, retrieval.chunks, counter=counter
            )
        except LLMError as exc:
            # Fail closed, honestly: nothing was drafted, so nothing is shown.
            mark("generate", t0, n0, f"failed ({exc.kind})")
            return refuse(
                RefusalReason.SERVICE_UNAVAILABLE,
                f"Answer generation failed: {exc.public_message}",
            )
        mark("generate", t0, n0, f"drafted {len(draft)} characters")

        # --- 5. verify -----------------------------------------------------
        n0 = len(counter.prompt_names)
        report = self.guardrail.verify(draft, retrieval.chunks, counter=counter)
        self._trace_verification(trace, report, counter, n0)

        # --- 6. act on the verdict -----------------------------------------
        answer = draft
        status = AnswerStatus.VERIFIED
        repair_report: GuardrailReport | None = None

        if report.decision == GuardrailDecision.SKIPPED:
            # Guardrail switched off: the eval baseline. Label it honestly.
            status = AnswerStatus.UNVERIFIED

        elif report.verification_error:
            # Verification could not run (rate limit / quota / outage). The draft
            # is unchecked, so it is withheld — but this is NOT "not supported".
            return refuse(
                RefusalReason.SERVICE_UNAVAILABLE,
                "The answer could not be verified because the checking service "
                "failed; no unchecked text is shown.",
                report=report,
            )

        elif report.decision == GuardrailDecision.PASS and not report.claims:
            # PASS with zero claims means the draft asserted no checkable facts —
            # in practice the model wrote its own refusal ("the passages only
            # cover Mudra Yojana..."). Vacuously true is not verified, so
            # labelling this "verified" and attaching citations would badge a
            # refusal as a confident answer.
            #
            # We substitute our own refusal rather than passing the model's
            # through, because the model's version leaks retrieval internals
            # (naming whichever unrelated schemes happened to come back) and is
            # not written in the user's language.
            return refuse(
                RefusalReason.NO_CLAIMS,
                "The draft contained no verifiable factual claims, so it is "
                "treated as a refusal rather than a verified answer.",
                report=report,
            )

        elif report.decision == GuardrailDecision.BLOCK:
            return refuse(
                RefusalReason.NOT_SUPPORTED,
                "Too many claims in the draft were not supported by the evidence.",
                report=report,
            )

        elif report.decision == GuardrailDecision.REPAIR:
            supported = [
                check for check in report.claims if check.verdict == Verdict.SUPPORTED
            ]

            t0, n0 = time.perf_counter(), len(counter.prompt_names)
            try:
                repaired = repair_answer(
                    self.llm, normalized.normalized_query, language,
                    retrieval.chunks, supported, counter=counter,
                )
            except LLMError as exc:
                # Could not rebuild a clean answer, and the draft is known to
                # contain unsupported claims. Refuse rather than ship it.
                mark("repair", t0, n0, f"failed ({exc.kind})")
                return refuse(
                    RefusalReason.SERVICE_UNAVAILABLE,
                    f"Repair failed: {exc.public_message}",
                    report=report,
                )
            mark("repair", t0, n0, f"rewrote the answer from {len(supported)} verified claim(s)")

            # The repaired text is NEW generated text, so it gets the same full
            # check as the draft. It ships only if that check is a clean PASS with
            # at least one claim; otherwise we refuse (audit finding H1).
            n0 = len(counter.prompt_names)
            repair_report = self.guardrail.verify(repaired, retrieval.chunks, counter=counter)
            self._trace_verification(trace, repair_report, counter, n0, prefix="re-check: ")

            if repair_report.verification_error:
                return refuse(
                    RefusalReason.SERVICE_UNAVAILABLE,
                    "The repaired answer could not be re-checked because the "
                    "checking service failed; it is withheld.",
                    report=report, repair_report=repair_report,
                )
            if repair_report.decision != GuardrailDecision.PASS or not repair_report.claims:
                return refuse(
                    RefusalReason.NOT_SUPPORTED,
                    "The repaired answer did not pass its own verification "
                    f"(decision {repair_report.decision.value}), so it is withheld.",
                    report=report, repair_report=repair_report,
                )

            answer = repaired
            status = AnswerStatus.PARTIALLY_VERIFIED

        # Chunks the judge verified a supported claim against. These Python-
        # validated ids are the authority for which sources to show. For a
        # repaired answer they come from the re-check of the text actually shown.
        source_report = repair_report if repair_report is not None else report
        verified_chunk_ids = [
            check.evidence_chunk_id
            for check in source_report.claims
            if check.verdict == Verdict.SUPPORTED
            and check.citation_valid
            and check.evidence_chunk_id
        ]

        citations = build_citations(answer, retrieval.chunks, verified_chunk_ids)

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
            latency_ms=_ms(started),
            llm_calls=counter.count,
            refusal_reason=None,
            normalization_fallback=normalized.fallback_used,
            retrieval=summary,
            trace=trace,
            repair_report=repair_report,
        )

    # ---- trace helper ----------------------------------------------------

    @staticmethod
    def _trace_verification(
        trace: list[StageTrace],
        report: GuardrailReport,
        counter: CallCounter,
        names_before: int,
        prefix: str = "",
    ) -> None:
        """
        Turn a GuardrailReport's timings into StageTrace entries.

        The guardrail records durations by stage name without knowing about the
        trace model; this is where they become trace rows. Only stages that
        actually ran appear.
        """
        new_calls = counter.prompt_names[names_before:]
        failed = bool(report.verification_error)
        judged = "judge_claims" in report.timings_ms

        if "split_claims" in report.timings_ms:
            if failed and not judged:
                outcome = "failed"
            else:
                outcome = f"{len(report.claims)} claim(s) extracted"
            trace.append(StageTrace(
                stage="split_claims", duration_ms=report.timings_ms["split_claims"],
                llm_calls=new_calls.count("split_claims"), outcome=prefix + outcome,
            ))

        if judged:
            if failed:
                outcome = "failed"
            else:
                outcome = (
                    f"{report.supported_count}/{len(report.claims)} supported "
                    f"-> {report.decision.value}"
                )
            trace.append(StageTrace(
                stage="judge_claims", duration_ms=report.timings_ms["judge_claims"],
                llm_calls=new_calls.count("judge_claims"), outcome=prefix + outcome,
            ))

        if "similarity" in report.timings_ms:
            recorded = any(check.best_similarity is not None for check in report.claims)
            outcome = (
                "recorded for calibration (not used in the decision)"
                if recorded else "unavailable"
            )
            trace.append(StageTrace(
                stage="similarity", duration_ms=report.timings_ms["similarity"],
                llm_calls=0, outcome=prefix + outcome,
            ))

    # ---- refusal helper --------------------------------------------------

    def _refusal_response(
        self,
        question: str,
        normalized: NormalizedQuery,
        language: Language,
        summary: RetrievalSummary,
        counter: CallCounter,
        started: float,
        trace: list[StageTrace],
        reason: RefusalReason,
        note: str,
        report: GuardrailReport | None = None,
        repair_report: GuardrailReport | None = None,
    ) -> AskResponse:
        """
        Build a refusal with its reason, and log it.

        When the guardrail never ran (weak evidence, generation failure) a SKIPPED
        report carries the explanation. When it did run, its own report is kept so
        the user can still see which claims failed.
        """
        if report is None:
            report = GuardrailReport(
                decision=GuardrailDecision.SKIPPED,
                claims=[],
                block_threshold=self.settings.guardrail_block_threshold,
                notes=[note],
                llm_calls=0,
            )
        else:
            report = report.model_copy(update={"notes": [*report.notes, note]})

        outage = reason == RefusalReason.SERVICE_UNAVAILABLE
        # An outage is logged on its own (answer_status "service_unavailable") so
        # it can never be mistaken for an honest refusal or a guardrail BLOCK in
        # /metrics (audit finding H2).
        # What /metrics counts here must match what the user got. A refusal whose
        # report says PASS (draft asserted nothing: NO_CLAIMS) or REPAIR (the
        # repaired text failed its re-check) was withheld, so it is counted as a
        # BLOCK, not as a PASS / REPAIR that shipped an answer.
        if outage:
            logged_decision = GuardrailDecision.SKIPPED.value
        elif report.decision in (GuardrailDecision.PASS, GuardrailDecision.REPAIR):
            logged_decision = GuardrailDecision.BLOCK.value
        else:
            logged_decision = report.decision.value
        log_event("guardrail_decision", logged_decision, detail=note, settings=self.settings)
        log_event(
            "answer_status",
            "service_unavailable" if outage else AnswerStatus.REFUSED.value,
            detail=reason.value, settings=self.settings,
        )

        return AskResponse(
            question=question,
            normalized_query=normalized.normalized_query,
            language=language,
            answer=refusal_message(language.value, reason),
            status=AnswerStatus.REFUSED,
            citations=[],
            guardrail=report,
            latency_ms=_ms(started),
            llm_calls=counter.count,
            refusal_reason=reason,
            normalization_fallback=normalized.fallback_used,
            retrieval=summary,
            trace=trace,
            repair_report=repair_report,
        )
