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

So we verify after generating, in these steps:

  1. LLM splits the draft into atomic claims.
  2. LLM judge checks each claim against ONLY the evidence passages, and must
     name the chunk_id that supports it.
  3. PYTHON checks that chunk_id actually exists. A judge that cites a
     non-existent chunk is hallucinating about its own evidence, so the claim is
     demoted regardless of the verdict it was given.
  4. PYTHON checks every number and link in the answer really appears in the
     retrieved text (a check no LLM judgement can talk its way around).
  5. PYTHON applies the decision policy. No LLM is involved in the decision.

A SECOND, INDEPENDENT SIGNAL (recorded, not yet used)
-----------------------------------------------------
`attach_similarity` adds the "semantic similarity" signal from the research
plan: the cosine similarity between each claim and its cited evidence, using the
same embedding model as retrieval. It is recorded per claim so it can be
CALIBRATED against the judge's verdicts, and it is deliberately NOT used in the
decision policy until that calibration has been done (a second-half experiment).
A natural-language-inference (NLI) signal is planned but NOT implemented.

WHY THE CHECKS ARE IN PYTHON
----------------------------
Because an LLM judging its own work is not a control — it is another generation.
The parts that must not be wrong (does this citation exist? is the unsupported
ratio over the limit? is this number in the evidence?) are code: deterministic,
unit-testable, and incapable of being talked out of their answer.

KNOWN LIMIT, STATED HONESTLY
----------------------------
The judge is itself an LLM, so it can be wrong in both directions. We bias it
towards strictness ("probably correct" counts as NOT_SUPPORTED) because for
government benefit information, a wrongly-withheld answer costs a user one web
search, while a wrongly-confident answer can cost them a day at an office or an
application they were never eligible for.
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

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

logger = logging.getLogger("schemesure.guardrail")

# A judge asked about too many claims at once gets sloppy and starts dropping
# entries. Our answers are short, so this is a safety valve rather than a limit
# we expect to reach. When the splitter returns MORE than this, the extra claims
# are never judged, so the answer is never allowed to PASS (see Guardrail.verify).
MAX_CLAIMS = 20

# Sanity cap on what we even keep from the splitter (it only matters for the
# "more than MAX_CLAIMS" check; nothing beyond MAX_CLAIMS is ever judged).
_MAX_RAW_CLAIMS = 100


# ==========================================================================
# The decision policy (pure Python, no LLM, fully unit-tested)
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
# Citation validation (also pure Python)
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
# Deterministic grounding of numbers and links (pure Python)
# ==========================================================================
# The claim splitter is told to leave out "advice to visit a website" and
# disclaimers, so a made-up URL or phone number in such a sentence is never
# judged — and a long answer can have claims beyond MAX_CLAIMS that are never
# judged either (audit finding M3). This check closes that gap without an LLM:
# every number >= 10 and every URL / domain in the final answer must literally
# appear in the retrieved passages (or in their source URLs).

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# [1], [12], [1,2], [1; 2] — the citation markers; their digits are not facts.
_MARKER_RE = re.compile(r"\[\s*\d{1,2}(?:\s*[,;]\s*\d{1,2})*\s*\]")
# "1. First item" / "- 2) item": list numbering is layout, not a fact.
_LIST_PREFIX_RE = re.compile(r"(?m)^\s*(?:[-*•]\s+)?\d{1,3}[.)]\s+")
_URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"')\]]+"
    r"|\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*"
    # The TLD is matched case-SENSITIVELY: "year.In addition" (a missing space
    # after a full stop) is a typo, not a web address; real domains are lowercase.
    r"(?-i:\.(?:gov\.in|nic\.in|co\.in|in|com|org|net|edu|info|io)\b)"
    r"(?:/[^\s<>\"')\]]*)?",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

# A number WITH its context, so the check can tell "3 instalments" (harmless) from
# "Rs 5 lakh" or "8.2%" (the facts that matter). Groups: currency prefix, the
# number, an optional lakh/crore unit, an optional percent sign.
_AMOUNT_RE = re.compile(
    r"(?P<cur>(?:\b(?:rs|inr)\b\.?|₹)\s*)?"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?P<unit>\s*(?:lakh|lac|crore|करोड़|करोड|लाख))?"
    r"(?P<pct>\s*%)?",
    re.IGNORECASE,
)
_UNIT_MULTIPLIER = {
    "lakh": Decimal(100000), "lac": Decimal(100000), "लाख": Decimal(100000),
    "crore": Decimal(10000000), "करोड़": Decimal(10000000), "करोड": Decimal(10000000),
}


def _amount_value(match: "re.Match[str]") -> Decimal | None:
    """The full numeric value of one _AMOUNT_RE match ('2.00 Lakh' -> 200000)."""
    value = _canonical_number(match.group("num"))
    if value is None:
        return None
    unit = (match.group("unit") or "").strip().lower()
    return value * _UNIT_MULTIPLIER[unit] if unit else value


def _canonical_number(token: str) -> Decimal | None:
    """'6,000' -> Decimal('6000'); '10.50' -> Decimal('10.5'). None if unparseable."""
    cleaned = token.replace(",", "").rstrip(".")
    try:
        return Decimal(cleaned).normalize()
    except InvalidOperation:
        return None


def _normalize_url(url: str) -> str:
    """Lowercase, drop scheme / www / trailing punctuation, so forms compare equal."""
    text = url.lower().strip().rstrip(".,;:!?)]}>'\"/")
    text = re.sub(r"^https?://", "", text)
    return re.sub(r"^www\.", "", text)


def numeric_and_link_grounding(answer: str, chunks: list[RetrievedChunk]) -> list[str]:
    """
    Return the numbers and links in `answer` that appear in NONE of the passages.

    - Only bare whole numbers below 10 are ignored (counts like "three
      instalments"). Decimals ("8.2%"), percentages, rupee amounts and
      lakh / crore amounts are always checked, whatever their size.
    - "6,000", "Rs. 6000/-", "₹6,000" all normalise to 6000, and Devanagari
      digits are converted first, so formatting differences never raise a flag.
    - [n] citation markers and "1." list numbering are ignored.
    - A URL or domain must appear (scheme / "www." ignored) in a passage text or
      a passage's source_url.

    An empty list means every checked token is grounded.
    """
    text = (answer or "").translate(_DEVANAGARI_DIGITS)
    text = _MARKER_RE.sub(" ", text)
    text = _LIST_PREFIX_RE.sub("", text)

    haystack = " ".join(
        f"{chunk.text} {chunk.metadata.source_url}" for chunk in chunks
    ).translate(_DEVANAGARI_DIGITS).lower()
    url_haystack = re.sub(r"\bwww\.", "", re.sub(r"https?://", "", haystack))
    # Every number the passages contain, plus the full value of any "<x> lakh /
    # crore", so "Rs 2,00,000" in an answer matches "Rs 2.00 Lakh" in the text.
    known_numbers: set[Decimal] = set()
    for token in _NUMBER_RE.findall(haystack):
        value = _canonical_number(token)
        if value is not None:
            known_numbers.add(value)
    for match in _AMOUNT_RE.finditer(haystack):
        value = _amount_value(match)
        if value is not None:
            known_numbers.add(value)

    ungrounded: list[str] = []

    # ---- links ---------------------------------------------------------------
    for match in _URL_RE.finditer(text):
        raw = match.group(0).rstrip(".,;:!?)]}>'\"")
        if _normalize_url(raw) not in url_haystack and raw not in ungrounded:
            ungrounded.append(raw)

    # ---- numbers (URLs removed first so their digits are not double-counted) --
    for match in _AMOUNT_RE.finditer(_URL_RE.sub(" ", text)):
        value = _amount_value(match)
        if value is None:
            continue
        # Skip ONLY a bare whole number below 10 ("three instalments"). A decimal,
        # a percentage, a lakh/crore amount or a rupee amount is always checked.
        has_context = bool(match.group("cur") or match.group("unit") or match.group("pct"))
        is_whole = value == value.to_integral_value()
        if value < 10 and is_whole and not has_context:
            continue
        if value in known_numbers:
            continue
        shown = (match.group("num") + (match.group("unit") or "") + (match.group("pct") or "")).rstrip(",.")
        if shown not in ungrounded:
            ungrounded.append(shown)

    return ungrounded


# ==========================================================================
# Independent second signal: claim-evidence semantic similarity
# ==========================================================================


def attach_similarity(
    checks: list[ClaimCheck],
    chunks: list[RetrievedChunk],
    embed_fn: Callable[[list[str]], object],
) -> list[ClaimCheck]:
    """
    Record, per claim, how semantically close it is to the evidence.

    This is the "semantic similarity" signal from the research plan. It is an
    INDEPENDENT diagnostic: it is stored for calibration against the judge's
    verdicts and is deliberately NOT used by `decide` (no verdict or decision is
    ever changed here) until it has been calibrated — a second-half experiment.
    The NLI (entailment) signal is planned but not implemented.

    For each claim it sets
      evidence_similarity      cosine(claim, the chunk named by evidence_chunk_id),
                               or None if that chunk is not among `chunks`;
      best_similarity          the maximum cosine over all retrieved chunks;
      best_similarity_chunk_id which chunk that maximum belongs to.
    Values are rounded to 4 decimals. ONE batched `embed_fn` call embeds the
    claims and the chunk texts together; `embed_fn(texts)` must return one vector
    per text (any scale; they are re-normalised here, so the dot product is a
    true cosine).
    """
    if not checks or not chunks:
        return [check.model_copy() for check in checks]

    import numpy as np  # already installed with sentence-transformers / torch

    claim_texts = [check.claim for check in checks]
    chunk_texts = [chunk.text for chunk in chunks]
    vectors = np.asarray(embed_fn(claim_texts + chunk_texts), dtype=float)
    expected = len(claim_texts) + len(chunk_texts)
    if vectors.ndim != 2 or vectors.shape[0] != expected:
        raise ValueError(f"embed_fn returned shape {vectors.shape}, expected {expected} rows")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    claim_vectors = vectors[: len(claim_texts)]
    chunk_vectors = vectors[len(claim_texts):]
    similarity = claim_vectors @ chunk_vectors.T          # (claims, chunks) cosines

    position_of = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}

    updated: list[ClaimCheck] = []
    for row, check in enumerate(checks):
        best = int(np.argmax(similarity[row]))
        cited = position_of.get((check.evidence_chunk_id or "").strip())
        updated.append(
            check.model_copy(
                update={
                    "evidence_similarity": (
                        round(float(similarity[row][cited]), 4) if cited is not None else None
                    ),
                    "best_similarity": round(float(similarity[row][best]), 4),
                    "best_similarity_chunk_id": chunks[best].chunk_id,
                }
            )
        )
    return updated


# ==========================================================================
# Steps 1 and 2: the LLM calls
# ==========================================================================

# A chunk id looks like  pm_kisan#benefits#0 . A claim that carries one (or the
# words SUPPORTED / NOT_SUPPORTED) is trying to talk to the judge, not state a
# fact; both are stripped before judging (audit finding L13).
_CHUNK_ID_RE = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)*#[a-z0-9_\-]+#\d+\b")
_VERDICT_WORD_RE = re.compile(r"\b(?:NOT_SUPPORTED|SUPPORTED)\b")


def sanitize_claim(text: str) -> str:
    """Remove chunk-id patterns and verdict keywords that a claim could smuggle in."""
    cleaned = _CHUNK_ID_RE.sub("", text)
    cleaned = _VERDICT_WORD_RE.sub("", cleaned)
    cleaned = re.sub(r"\(\s*[,;:\s]*\)", "", cleaned)       # "( , )" left behind
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" ,;:")


def split_claims(
    llm: LLMClient,
    answer: str,
    counter: CallCounter | None = None,
) -> list[str]:
    """
    Break an answer into atomic, self-contained, English claims.

    Returns ALL claims the splitter produced (capped only by a sanity limit), so
    the caller can tell when there were more than MAX_CLAIMS and refuse to PASS.
    """
    answer = (answer or "").strip()
    if not answer:
        return []

    parsed, _ = llm.complete_json(
        prompt_name="split_claims",
        system=CLAIM_SPLITTER_SYSTEM,
        user=CLAIM_SPLITTER_USER.format(answer=answer),
        max_tokens=700,
        counter=counter,
    )

    if not isinstance(parsed, dict):
        raise LLMError("claim splitter did not return a JSON object", kind="bad_output")

    raw_claims = parsed.get("claims")
    if not isinstance(raw_claims, list):
        raise LLMError("claim splitter returned no 'claims' list", kind="bad_output")

    claims: list[str] = []
    for item in raw_claims:
        text = str(item).strip() if not isinstance(item, dict) else str(
            item.get("claim", "")
        ).strip()
        text = sanitize_claim(text)
        if text:
            claims.append(text)

    return claims[:_MAX_RAW_CLAIMS]


def _text_key(text: str) -> str:
    """NFKC + casefold + letters/digits only, so cosmetic differences compare equal."""
    folded = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def _resolve_chunk_id(raw: str, chunks: list[RetrievedChunk]) -> str:
    """
    Map "[2]" or "2" (a judge citing the evidence NUMBER instead of the id) to
    the id of the chunk at that position. Real ids pass through untouched, and
    anything else is left for validate_citations to reject.
    """
    if any(chunk.chunk_id == raw for chunk in chunks):
        return raw
    match = re.fullmatch(r"\[?\s*(\d{1,2})\s*\]?", raw)
    if match:
        position = int(match.group(1))
        if 1 <= position <= len(chunks):
            return chunks[position - 1].chunk_id
    return raw


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
        max_tokens=900,
        counter=counter,
    )

    if not isinstance(parsed, dict):
        raise LLMError("judge did not return a JSON object", kind="bad_output")

    raw_verdicts = parsed.get("verdicts")
    if not isinstance(raw_verdicts, list):
        raise LLMError("judge returned no 'verdicts' list", kind="bad_output")

    # ---- align verdicts to claims ---------------------------------------------
    # POSITION is the primary key whenever the judge returned exactly one verdict
    # per claim: it keeps working when the judge trims a full stop, swaps a
    # currency symbol or prefixes "1. " (all of which broke exact-text matching).
    # Only when the counts differ do we fall back to matching normalised text.
    claim_keys = [_text_key(claim) for claim in claims]
    by_text: dict[str, dict] = {}
    for entry in raw_verdicts:
        if isinstance(entry, dict):
            key = _text_key(entry.get("claim", ""))
            if key and key not in by_text:
                by_text[key] = entry

    same_length = len(raw_verdicts) == len(claims)

    checks: list[ClaimCheck] = []
    for index, claim in enumerate(claims):
        entry: dict | None = None

        if same_length and isinstance(raw_verdicts[index], dict):
            candidate = raw_verdicts[index]
            echoed = _text_key(candidate.get("claim", ""))
            # Position wins, with one safety: if the judge echoed the text of a
            # DIFFERENT claim, it reordered its answers, so trust the text.
            reordered = bool(echoed) and echoed != claim_keys[index] and echoed in claim_keys
            if not reordered:
                entry = candidate

        if entry is None:
            entry = by_text.get(claim_keys[index])

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
            _resolve_chunk_id(str(raw_chunk_id).strip(), chunks)
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


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


class Guardrail:
    """Runs the full verification pass and produces a GuardrailReport."""

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings | None = None,
        embed_fn: Callable[[list[str]], object] | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        # When given, the claim-evidence similarity signal is recorded after
        # judging (see attach_similarity). It never changes a verdict.
        self.embed_fn = embed_fn

    def verify(
        self,
        answer: str,
        chunks: list[RetrievedChunk],
        counter: CallCounter | None = None,
    ) -> GuardrailReport:
        """
        Verify `answer` against `chunks`.

        Returns a report; never raises. The caller reads `report.decision` to
        decide what the user sees, and `report.verification_error` to tell "the
        answer was judged bad" apart from "the answer could not be judged".
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
        timings: dict[str, int] = {}

        def calls_made() -> int:
            return (counter.count - calls_before) if counter else 0

        def fail_closed(step: str, exc: LLMError) -> GuardrailReport:
            # Cannot verify => do not ship. See the module docstring on why we
            # fail closed rather than falling back to the unverified draft. Only
            # the PUBLIC message goes into the report; raw provider text stays
            # in the server log (audit finding L1).
            logger.warning("guardrail step %s failed (%s): %s", step, exc.kind, exc.detail or exc)
            return GuardrailReport(
                decision=GuardrailDecision.BLOCK,
                claims=[],
                block_threshold=threshold,
                notes=[
                    f"Could not {step}: {exc.public_message}",
                    "Refusing because the answer could not be verified.",
                ],
                llm_calls=calls_made(),
                verification_error=exc.public_message,
                timings_ms=timings,
            )

        # ---- 1. split into claims ------------------------------------------
        started = time.perf_counter()
        try:
            raw_claims = split_claims(self.llm, answer, counter=counter)
        except LLMError as exc:
            timings["split_claims"] = _elapsed_ms(started)
            return fail_closed("split the answer into claims", exc)
        timings["split_claims"] = _elapsed_ms(started)

        # Judge at most MAX_CLAIMS; remember if there were more, because claims
        # that were never judged must not ride along under a PASS.
        overflow = len(raw_claims) - MAX_CLAIMS
        claims = raw_claims[:MAX_CLAIMS]

        # ---- 2 + 3. judge, then validate the judge's citations ---------------
        started = time.perf_counter()
        try:
            checks = judge_claims(self.llm, claims, chunks, counter=counter)
        except LLMError as exc:
            timings["judge_claims"] = _elapsed_ms(started)
            return fail_closed("verify the claims", exc)
        timings["judge_claims"] = _elapsed_ms(started)

        # ---- independent similarity signal (recorded only) -------------------
        notes_after: list[str] = []
        if self.embed_fn is not None and checks:
            started = time.perf_counter()
            try:
                checks = attach_similarity(checks, chunks, self.embed_fn)
            except Exception as exc:  # noqa: BLE001 - a diagnostic must never break verification
                notes_after.append(
                    f"Claim-evidence similarity could not be computed "
                    f"({type(exc).__name__}); verdicts are unaffected."
                )
            timings["similarity"] = _elapsed_ms(started)

        # ---- 5. decision policy ----------------------------------------------
        decision, notes = decide(checks, threshold)

        supported = sum(1 for check in checks if check.verdict == Verdict.SUPPORTED)
        unsupported = len(checks) - supported
        fabricated = sum(1 for check in checks if not check.citation_valid)
        if fabricated:
            notes.append(
                f"{fabricated} claim(s) cited an evidence chunk that does not "
                "exist and were rejected in code."
            )

        # ---- 4. deterministic number / link grounding --------------------------
        ungrounded = numeric_and_link_grounding(answer, chunks)
        if ungrounded:
            notes.append(
                "Numbers or links not found in the official text: "
                + ", ".join(ungrounded) + "."
            )
            if decision == GuardrailDecision.PASS and checks:
                decision = GuardrailDecision.REPAIR
                notes.append(
                    "Even though every checked claim was supported, the answer "
                    "mentions content that appears nowhere in the retrieved "
                    "passages, so it is rebuilt from the verified claims only."
                )

        # Claims beyond MAX_CLAIMS were never judged: never PASS with them.
        if overflow > 0:
            notes.append(
                f"The answer contained {overflow} more claim(s) than the "
                f"{MAX_CLAIMS} that can be checked; the extra ones were not verified."
            )
            if decision == GuardrailDecision.PASS:
                decision = GuardrailDecision.REPAIR
                notes.append("Rebuilding the answer from the verified claims only.")

        notes.extend(notes_after)

        return GuardrailReport(
            decision=decision,
            claims=checks,
            supported_count=supported,
            unsupported_count=unsupported,
            unsupported_ratio=round(unsupported / len(checks), 4) if checks else 0.0,
            block_threshold=threshold,
            notes=notes,
            llm_calls=calls_made(),
            timings_ms=timings,
            ungrounded_tokens=ungrounded,
        )
