"""
Tests for the guardrail's decision policy and citation validation.

These are the most important tests in the project. The decision policy is what
stands between a hallucinated answer and the user, and it is pure Python — so it
can be tested exhaustively with fabricated verdicts and no LLM, no network and
no flakiness. Every test here runs in microseconds and gives the same result
every time, which is exactly the property you want in a safety check.
"""

from __future__ import annotations

import pytest

from app.schemas import ClaimCheck, GuardrailDecision, Verdict
from core.guardrail import decide, validate_citations


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def supported(claim: str = "a fact", chunk_id: str = "pm_kisan#benefits#0") -> ClaimCheck:
    return ClaimCheck(claim=claim, verdict=Verdict.SUPPORTED, evidence_chunk_id=chunk_id)


def unsupported(claim: str = "a wrong fact") -> ClaimCheck:
    return ClaimCheck(claim=claim, verdict=Verdict.NOT_SUPPORTED, evidence_chunk_id=None)


# ==========================================================================
# decide() — the decision policy
# ==========================================================================


class TestDecisionPolicy:
    """The 50% default threshold is used throughout unless stated otherwise."""

    def test_all_supported_passes(self):
        checks = [supported("fact 1"), supported("fact 2"), supported("fact 3")]
        decision, notes = decide(checks, block_threshold=0.5)
        assert decision == GuardrailDecision.PASS
        assert any("3/3 claims supported" in note for note in notes)

    def test_single_supported_claim_passes(self):
        decision, _ = decide([supported()], block_threshold=0.5)
        assert decision == GuardrailDecision.PASS

    def test_no_claims_passes_with_a_note(self):
        """An answer asserting no facts (a refusal, a 'see the website') has
        nothing to verify, so there is nothing to block."""
        decision, notes = decide([], block_threshold=0.5)
        assert decision == GuardrailDecision.PASS
        assert any("nothing to verify" in note.lower() for note in notes)

    def test_minority_unsupported_repairs(self):
        """1 of 4 unsupported = 0.25, under the limit -> rebuild from the good 3."""
        checks = [supported(), supported(), supported(), unsupported()]
        decision, notes = decide(checks, block_threshold=0.5)
        assert decision == GuardrailDecision.REPAIR
        assert any("Removing 1 unsupported" in note for note in notes)

    def test_exactly_at_threshold_repairs_not_blocks(self):
        """Policy is 'ratio > threshold blocks', so exactly 0.5 repairs.

        Safe because REPAIR rebuilds from supported claims only — the failing
        claim cannot survive into the answer either way."""
        checks = [supported(), unsupported()]
        decision, _ = decide(checks, block_threshold=0.5)
        assert decision == GuardrailDecision.REPAIR

    def test_majority_unsupported_blocks(self):
        """3 of 4 unsupported = 0.75 > 0.5 -> refuse."""
        checks = [supported(), unsupported(), unsupported(), unsupported()]
        decision, notes = decide(checks, block_threshold=0.5)
        assert decision == GuardrailDecision.BLOCK
        assert any("exceeds" in note for note in notes)

    def test_all_unsupported_blocks(self):
        """The classic hallucination: fluent answer, nothing backing any of it."""
        checks = [unsupported(), unsupported(), unsupported()]
        decision, notes = decide(checks, block_threshold=0.5)
        assert decision == GuardrailDecision.BLOCK
        assert any("No claim was supported" in note for note in notes)

    def test_no_supported_claims_blocks_even_with_lenient_threshold(self):
        """Guard against a misconfigured threshold producing an empty answer:
        with nothing supported there is nothing for REPAIR to say."""
        decision, _ = decide([unsupported(), unsupported()], block_threshold=0.99)
        assert decision == GuardrailDecision.BLOCK

    def test_strict_threshold_blocks_a_single_bad_claim(self):
        """threshold=0.0 means 'any unsupported claim is unacceptable'."""
        checks = [supported(), supported(), supported(), unsupported()]
        decision, _ = decide(checks, block_threshold=0.0)
        assert decision == GuardrailDecision.BLOCK

    def test_strict_threshold_still_passes_a_clean_answer(self):
        decision, _ = decide([supported(), supported()], block_threshold=0.0)
        assert decision == GuardrailDecision.PASS

    @pytest.mark.parametrize(
        "n_supported,n_unsupported,threshold,expected",
        [
            (10, 0, 0.5, GuardrailDecision.PASS),
            (9, 1, 0.5, GuardrailDecision.REPAIR),    # 0.10
            (6, 4, 0.5, GuardrailDecision.REPAIR),    # 0.40
            (5, 5, 0.5, GuardrailDecision.REPAIR),    # 0.50, not >
            (4, 6, 0.5, GuardrailDecision.BLOCK),     # 0.60
            (0, 10, 0.5, GuardrailDecision.BLOCK),    # 1.00
            (7, 3, 0.25, GuardrailDecision.BLOCK),    # 0.30 > 0.25
            (8, 2, 0.25, GuardrailDecision.REPAIR),   # 0.20
        ],
    )
    def test_threshold_table(self, n_supported, n_unsupported, threshold, expected):
        """Pin the exact boundary behaviour so it cannot drift unnoticed."""
        checks = [supported(f"ok {i}") for i in range(n_supported)]
        checks += [unsupported(f"bad {i}") for i in range(n_unsupported)]
        decision, _ = decide(checks, block_threshold=threshold)
        assert decision == expected

    def test_decide_is_pure(self):
        """Same input, same output, and the input is not mutated."""
        checks = [supported(), unsupported()]
        snapshot = [check.model_copy(deep=True) for check in checks]
        first, _ = decide(checks, 0.5)
        second, _ = decide(checks, 0.5)
        assert first == second
        assert checks == snapshot


# ==========================================================================
# validate_citations() — catching the judge's own hallucinations
# ==========================================================================


class TestCitationValidation:
    """
    The judge is an LLM, so it can cite a chunk that does not exist. A SUPPORTED
    verdict backed by a fabricated id is worse than useless: it looks rigorous.
    Code catches it with a set-membership test.
    """

    VALID_IDS = {"pm_kisan#benefits#0", "pm_kisan#eligibility#0", "apy#overview#1"}

    def test_real_citation_survives(self):
        checks = [supported("Rs 6000 per year", "pm_kisan#benefits#0")]
        result = validate_citations(checks, self.VALID_IDS)
        assert result[0].verdict == Verdict.SUPPORTED
        assert result[0].citation_valid is True

    def test_fabricated_citation_is_demoted(self):
        """The whole point: a made-up chunk id flips SUPPORTED to NOT_SUPPORTED."""
        checks = [supported("Rs 9999 per year", "pm_kisan#benefits#99")]
        result = validate_citations(checks, self.VALID_IDS)
        assert result[0].verdict == Verdict.NOT_SUPPORTED
        assert result[0].citation_valid is False
        assert "fabricated" in result[0].note.lower()

    def test_supported_with_no_citation_is_demoted(self):
        checks = [ClaimCheck(claim="x", verdict=Verdict.SUPPORTED, evidence_chunk_id=None)]
        result = validate_citations(checks, self.VALID_IDS)
        assert result[0].verdict == Verdict.NOT_SUPPORTED
        assert result[0].citation_valid is False

    def test_supported_with_blank_citation_is_demoted(self):
        checks = [ClaimCheck(claim="x", verdict=Verdict.SUPPORTED, evidence_chunk_id="   ")]
        result = validate_citations(checks, self.VALID_IDS)
        assert result[0].verdict == Verdict.NOT_SUPPORTED

    def test_unsupported_claims_are_left_alone(self):
        """NOT_SUPPORTED claims are expected to carry no citation; that is not an
        error and must not be reported as a fabricated one."""
        result = validate_citations([unsupported("nope")], self.VALID_IDS)
        assert result[0].verdict == Verdict.NOT_SUPPORTED
        assert result[0].citation_valid is True

    def test_whitespace_in_citation_is_tolerated(self):
        checks = [supported("x", "  pm_kisan#benefits#0  ")]
        result = validate_citations(checks, self.VALID_IDS)
        assert result[0].verdict == Verdict.SUPPORTED

    def test_empty_valid_id_set_demotes_everything(self):
        """No evidence retrieved means no citation can possibly be valid."""
        result = validate_citations([supported("x", "anything")], set())
        assert result[0].verdict == Verdict.NOT_SUPPORTED

    def test_mixed_batch(self):
        checks = [
            supported("good", "apy#overview#1"),
            supported("bad", "apy#overview#404"),
            unsupported("also bad"),
        ]
        result = validate_citations(checks, self.VALID_IDS)
        assert [check.verdict for check in result] == [
            Verdict.SUPPORTED, Verdict.NOT_SUPPORTED, Verdict.NOT_SUPPORTED
        ]
        assert [check.citation_valid for check in result] == [True, False, True]


# ==========================================================================
# The two layers together
# ==========================================================================


class TestLayersCombined:
    def test_fabricated_citations_can_turn_a_pass_into_a_block(self):
        """
        End-to-end on the pure logic: the judge marked all three claims
        SUPPORTED, but two cite chunks that do not exist. Taking the judge at its
        word would ship the answer; validating in code refuses it.
        """
        valid_ids = {"pm_kisan#benefits#0"}
        judge_output = [
            supported("real fact", "pm_kisan#benefits#0"),
            supported("invented fact", "pm_kisan#benefits#7"),
            supported("another invention", "made_up#section#0"),
        ]

        # Without the code check, this would be 3/3 supported -> PASS.
        naive_decision, _ = decide(judge_output, 0.5)
        assert naive_decision == GuardrailDecision.PASS

        # With it: 1/3 supported, ratio 0.67 > 0.5 -> BLOCK.
        validated = validate_citations(judge_output, valid_ids)
        decision, _ = decide(validated, 0.5)
        assert decision == GuardrailDecision.BLOCK
