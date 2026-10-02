"""
The verification layer: check a drafted answer claim-by-claim against the
retrieved evidence, then decide whether to ship it, repair it, or refuse.

THE PROBLEM THIS SOLVES
-----------------------
Retrieval-augmented generation reduces hallucination; it does not remove it.
The model still blends real evidence with its own prior knowledge, and the result
reads exactly as confidently as a correct answer. Our index has 15 schemes, so
the most likely failure is not nonsense — it is a fluent, plausible answer about
a scheme we never indexed, assembled from whatever chunks happened to come back.
Measured: "What is the interest rate on Sukanya Samriddhi Yojana?" retrieves
PM-Mudra chunks at 0.71 similarity, well above our refusal threshold.

So we verify after generating, in four steps:

  1. LLM splits the draft into atomic claims.
  2. LLM judge checks each claim against ONLY the evidence passages, and must
     name the chunk_id that supports it.
  3. PYTHON checks that chunk_id actually exists. A judge that cites a
     non-existent chunk is hallucinating about its own evidence, so the claim is
     demoted regardless of the verdict it was given.
  4. PYTHON applies the decision policy. No LLM is involved in the decision.

WHY STEPS 3 AND 4 ARE IN PYTHON
-------------------------------
Because an LLM judging its own work is not a control — it is another generation.
The parts that must not be wrong (does this citation exist? is the unsupported
ratio over the limit?) are code: deterministic, unit-testable, and incapable of
being talked out of their answer.

KNOWN LIMIT, STATED HONESTLY
----------------------------
The judge is itself an LLM, so it can be wrong in both directions. We bias it
towards strictness ("probably correct" counts as NOT_SUPPORTED) because for
government benefit information, a wrongly-withheld answer costs a user one web
search, while a wrongly-confident answer can cost them a day at an office or an
application they were never eligible for.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.llm_client import CallCounter, LLMClient, LLMError
from app.prompts import (
    CLAIM_JUDGE_SYSTEM,
    CLAIM_JUDGE_USER,
    CLAIM_SPLITTER_SYSTEM,
    CLAIM_SPLITTER_USER,
)
from app.schemas import (
    ClaimCheck,
    GuardrailDecision,
    GuardrailReport,
    RetrievedChunk,
    Verdict,
)
from core.retriever import format_evidence

# A judge asked about too many claims at once gets sloppy and starts dropping
# entries. Our answers are short, so this is a safety valve rather than a limit
# we expect to reach.
MAX_CLAIMS = 20


# ==========================================================================
# Step 4 first: the decision policy (pure Python, no LLM, fully unit-tested)
# ==========================================================================


def decide(
    checks: list[ClaimCheck],
    block_threshold: float,
) -> tuple[GuardrailDecision, list[str]]:
    """
    Turn a list of per-claim verdicts into one decision.

    Returns (decision, notes). Pure function: same input, same output, no I/O.
    This is the function to read first if you want to know what the guardrail
    actually guarantees — and the one tests/test_guardrail.py pins down.

    Policy:
      - no claims to check            -> PASS   (nothing factual was asserted)
      - every claim supported         -> PASS
      - no claim supported            -> BLOCK  (nothing left to repair from)
      - unsupported ratio > threshold -> BLOCK
      - otherwise                     -> REPAIR (rebuild from supported claims)

    A claim whose citation failed Python validation has already been demoted to
    NOT_SUPPORTED by `validate_citations` before it reaches here.
    """
    notes: list[str] = []

    if not checks:
        # An answer with no factual claims is usually a refusal or a "please see
        # the official site" sentence. There is nothing to verify, so nothing to
        # block. Note it so the case is visible in the report.
        notes.append("No factual claims were extracted; nothing to verify.")
        return GuardrailDecision.PASS, notes

    total = len(checks)
    supported = sum(1 for check in checks if check.verdict == Verdict.SUPPORTED)
    unsupported = total - supported
    ratio = unsupported / total

    notes.append(
        f"{supported}/{total} claims supported by the evidence "
        f"(unsupported ratio {ratio:.2f}, block threshold {block_threshold:.2f})."
    )

    if unsupported == 0:
        return GuardrailDecision.PASS, notes

    if supported == 0:
        # Repair would have nothing true to say, so refusing is the only honest
        # option. Checked before the threshold so a lenient threshold cannot
        # produce an empty "repaired" answer.
        notes.append("No claim was supported; refusing instead of answering.")
        return GuardrailDecision.BLOCK, notes

    if ratio > block_threshold:
        notes.append(
            f"Unsupported ratio {ratio:.2f} exceeds the {block_threshold:.2f} "
            "limit; refusing instead of answering."
        )
        return GuardrailDecision.BLOCK, notes

    notes.append(
        f"Removing {unsupported} unsupported claim(s) and rebuilding the answer "
        "from the verified ones."
    )
    return GuardrailDecision.REPAIR, notes


# ==========================================================================
# Step 3: citation validation (also pure Python)
# ==========================================================================


def validate_citations(
    checks: list[ClaimCheck],
    valid_chunk_ids: set[str],
) -> list[ClaimCheck]:
    """
    Verify every cited chunk_id really exists, and demote the ones that do not.

    This catches a specific, dangerous failure: the judge marks a claim
    SUPPORTED and cites `pm_kisan#benefits#9` — an id that was never in the
    evidence. The verdict looks authoritative and is worthless. Code catches it
    because code can simply check set membership; a second LLM could not.
    """
    validated: list[ClaimCheck] = []

    for check in checks:
        if check.verdict != Verdict.SUPPORTED:
            # NOT_SUPPORTED claims are expected to carry no citation.
            validated.append(check.model_copy(update={"citation_valid": True}))
            continue

        chunk_id = (check.evidence_chunk_id or "").strip()

        if not chunk_id:
            validated.append(
                check.model_copy(
                    update={
                        "verdict": Verdict.NOT_SUPPORTED,
                        "citation_valid": False,
                        "note": "Marked supported but cited no evidence chunk.",
                    }
                )
            )
            continue

        if chunk_id not in valid_chunk_ids:
            validated.append(
                check.model_copy(
                    update={
                        "verdict": Verdict.NOT_SUPPORTED,
                        "citation_valid": False,
                        "note": (
                            f"Cited chunk '{chunk_id}' is not among the retrieved "
                            "passages (fabricated citation)."
                        ),
                    }
                )
            )
            continue

        validated.append(check.model_copy(update={"citation_valid": True}))

    return validated


# ==========================================================================
# Steps 1 and 2: the LLM calls
# ==========================================================================


def split_claims(
    llm: LLMClient,
    answer: str,
    counter: CallCounter | None = None,
) -> list[str]:
    """Break an answer into atomic, self-contained, English claims."""
    answer = (answer or "").strip()
    if not answer:
        return []

    parsed, _ = llm.complete_json(
        prompt_name="split_claims",
        system=CLAIM_SPLITTER_SYSTEM,
        user=CLAIM_SPLITTER_USER.format(answer=answer),
        max_tokens=900,
        counter=counter,
    )

    if not isinstance(parsed, dict):
        raise LLMError("claim splitter did not return a JSON object")

    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        raise LLMError("claim splitter returned no 'claims' list")

    claims: list[str] = []
    for item in raw_claims:
        text = str(item).strip() if not isinstance(item, dict) else str(
            item.get("claim", "")
        ).strip()
        if text:
            claims.append(text)

    return claims[:MAX_CLAIMS]


def judge_claims(
    llm: LLMClient,
    claims: list[str],
    chunks: list[RetrievedChunk],
    counter: CallCounter | None = None,
) -> list[ClaimCheck]:
    """
    Check every claim against the evidence in ONE call, then validate citations.

    One call rather than one per claim: it keeps "LLM calls per question" at a
    predictable 4, and it lets the judge see the whole evidence block once.
    """
    if not claims:
        return []

    evidence = format_evidence(chunks)
    numbered_claims = "\n".join(
        f"{index}. {claim}" for index, claim in enumerate(claims, start=1)
    )

    parsed, _ = llm.complete_json(
        prompt_name="judge_claims",
        system=CLAIM_JUDGE_SYSTEM,
        user=CLAIM_JUDGE_USER.format(evidence=evidence, claims=numbered_claims),
        max_tokens=1400,
        counter=counter,
    )

    if not isinstance(parsed, dict):
        raise LLMError("judge did not return a JSON object")

    raw_verdicts = parsed.get("verdicts")
    if not isinstance(raw_verdicts, list):
        raise LLMError("judge returned no 'verdicts' list")

    # Align verdicts to claims. Position is the primary key; exact claim text is
    # the fallback when the judge reorders or drops entries.
    by_text: dict[str, dict] = {}
    for entry in raw_verdicts:
        if isinstance(entry, dict):
            key = str(entry.get("claim", "")).strip().lower()
            if key and key not in by_text:
                by_text[key] = entry

    checks: list[ClaimCheck] = []
    for index, claim in enumerate(claims):
        entry: dict | None = None

        if index < len(raw_verdicts) and isinstance(raw_verdicts[index], dict):
            candidate = raw_verdicts[index]
            candidate_claim = str(candidate.get("claim", "")).strip().lower()
            # Trust the positional match only if the text agrees, or the judge
            # omitted the claim text entirely.
            if not candidate_claim or candidate_claim == claim.strip().lower():
                entry = candidate

        if entry is None:
            entry = by_text.get(claim.strip().lower())

        if entry is None:
            # FAIL CLOSED. A claim the judge did not rule on is unverified, and
            # unverified must never be treated as verified.
            checks.append(
                ClaimCheck(
                    claim=claim,
                    verdict=Verdict.NOT_SUPPORTED,
                    evidence_chunk_id=None,
                    citation_valid=False,
                    note="The judge returned no verdict for this claim.",
                )
            )
            continue

        raw_verdict = str(entry.get("verdict", "")).strip().upper()
        verdict = (
            Verdict.SUPPORTED if raw_verdict == "SUPPORTED" else Verdict.NOT_SUPPORTED
        )

        raw_chunk_id = entry.get("evidence_chunk_id")
        chunk_id = (
            str(raw_chunk_id).strip()
            if raw_chunk_id not in (None, "", "null", "None")
            else None
        )

        checks.append(
            ClaimCheck(claim=claim, verdict=verdict, evidence_chunk_id=chunk_id)
        )

    # Step 3: the Python check on the judge's own citations.
    return validate_citations(checks, {chunk.chunk_id for chunk in chunks})


# ==========================================================================
# The guardrail
# ==========================================================================


class Guardrail:
    """Runs the full verification pass and produces a GuardrailReport."""

    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    def verify(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        counter: CallCounter | None = None,
    ) -> GuardrailReport:
        """
        Verify `answer` against `chunks`.

        Returns a report; never raises. The caller reads `report.decision` to
        decide what the user sees.
        """
        threshold = self.settings.guardrail_block_threshold

        # The evaluation harness switches this off to measure what the guardrail
        # is actually buying us. SKIPPED is recorded so the two runs are
        # distinguishable in the metrics.
        if not self.settings.guardrail_enabled:
            return GuardrailReport(
                decision=GuardrailDecision.SKIPPED,
                claims=[],
                block_threshold=threshold,
                notes=["Guardrail disabled (GUARDRAIL_ENABLED=false)."],
                llm_calls=0,
            )

        calls_before = counter.count if counter else 0

        try:
            claims = split_claims(self.llm, answer, counter=counter)
        except LLMError as exc:
            # Cannot verify => do not ship. See the module docstring on why we
            # fail closed rather than falling back to the unverified draft.
            return GuardrailReport(
                decision=GuardrailDecision.BLOCK,
                claims=[],
                block_threshold=threshold,
                notes=[f"Could not split the answer into claims: {exc}",
                       "Refusing because the answer could not be verified."],
                llm_calls=(counter.count - calls_before) if counter else 0,
            )

        try:
            checks = judge_claims(self.llm, claims, chunks, counter=counter)
        except LLMError as exc:
            return GuardrailReport(
                decision=GuardrailDecision.BLOCK,
                claims=[],
                block_threshold=threshold,
                notes=[f"Could not verify the claims: {exc}",
                       "Refusing because the answer could not be verified."],
                llm_calls=(counter.count - calls_before) if counter else 0,
            )

        decision, notes = decide(checks, threshold)

        supported = sum(1 for check in checks if check.verdict == Verdict.SUPPORTED)
        unsupported = len(checks) - supported
        fabricated = sum(1 for check in checks if not check.citation_valid)
        if fabricated:
            notes.append(
                f"{fabricated} claim(s) cited an evidence chunk that does not "
                "exist and were rejected in code."
            )

        return GuardrailReport(
            decision=decision,
            claims=checks,
            supported_count=supported,
            unsupported_count=unsupported,
            unsupported_ratio=round(unsupported / len(checks), 4) if checks else 0.0,
            block_threshold=threshold,
            notes=notes,
            llm_calls=(counter.count - calls_before) if counter else 0,
        )
