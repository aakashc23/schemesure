# Evaluation results

> **PRELIMINARY, n=20 of 50 golden-set questions (a subset chosen for language/category coverage, not proportional to the full set).** Do not quote these figures as the benchmark; with this few questions a single answer moves a rate by several points.

The same 20 golden-set questions run with the verification layer **off** (plain RAG, the baseline) and **on**. Each question is normalized once and both conditions retrieve with that identical query; the only difference is whether answers are checked before being returned.

## Setup

- **Date:** 2026-11-16
- **Questions:** n=20 (11 answerable, 9 unanswerable) of 50 in the golden set
- **LLM:** `qwen/qwen3.8-27b` via https://api.groq.com/openai/v1
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (cosine)
- **Retrieval:** top-5, refusal threshold 0.45
- **Block threshold:** 0.5 (unsupported-claim fraction above which an answer is refused)
- **Corpus:** 312 chunks from 15 central schemes
- **Claim audit:** on

## Headline comparison

| Metric | Guardrail OFF | Guardrail ON | Change |
|---|---|---|---|
| **Unsupported claim rate** (delivered answers) | 5.0% (2/40) | 0.0% (0/34) | better (-5.0 pp) |
| Answers containing any unsupported claim | 15.4% (2/13) | 0.0% (0/11) | better (-15.4 pp) |
| Correct refusal rate (unanswerable) | 77.8% (7/9) | 100.0% (9/9) | better (+22.2 pp) |
| Over-blocking rate (answerable refused) | 0.0% (0/11) | 0.0% (0/11) | no change |
| Answered rate (of answerable) | 100.0% (11/11) | 100.0% (11/11) | no change |
| Key-fact coverage (answered only) | 81.8% (9/11) | 81.8% (9/11) | no change |
| **End-to-end fact accuracy** (all answerable; refusal = 0) | 81.8% (9/11) | 81.8% (9/11) | no change |
| Questions fully correct (all key facts) | 81.8% (9/11) | 81.8% (9/11) | no change |
| Avg LLM calls / question | 1.85 | 3.8 | +1.95 |
| Avg tokens / question | 1216 | 2765 | |
| Median latency | 3.1s | 9.0s | |
| p95 latency | 6.8s | 44.9s | |
| Infrastructure failures (excluded) | 0 | 0 | |
| Questions scored (paired: the same ones in both) | 20 | 20 | |

*Retrieval is identical in both conditions by construction (shared cached normalization): expected scheme in top-5 for 100.0% (11/11) of answerable questions.*

### How to read this

- **Unsupported claim rate** is the headline: of all factual claims in answers the user actually saw, how many the judge found unsupported by the retrieved official text. For ON, a PASS answer has zero unsupported claims *by definition of PASS* and its verdicts come from the same judge, so ON's figure is not independent evidence; use `--audit-all` for an independent re-audit.

- **Correct refusal vs over-blocking** trade against each other: refusing more catches more unanswerable questions and blocks more good answers. Refusal here uses one detector for both conditions (pipeline status, canned refusal text, or an answer with no checkable claims); answers the detector could not classify are left out of the refusal rates in both.

- **End-to-end fact accuracy** counts a refusal as zero facts, so it does not flatter a system that simply refuses more. Key-fact coverage (answered questions only) is shown beside it for that reason.

- **Latency** includes time the harness spent waiting out free-tier rate limits, so read the median, not the mean.

## Guardrail OFF (baseline)

| Metric | Value |
|---|---|
| Questions run | 20 |
| Infrastructure failures (excluded from all rates) | 0 |
| Scored | 20 (11 answerable, 9 not) |
| Retrieval: expected scheme in top-5 | 100.0% (11/11) |
| Retrieval: top-1 | 72.7% (8/11) |
| Answered (of answerable) | 100.0% (11/11) |
| Over-blocking rate | 0.0% (0/11) |
| Correct refusal rate | 77.8% (7/9) |
| Key-fact coverage (answered only) | 81.8% (9/11) |
| End-to-end fact accuracy (all answerable) | 81.8% (9/11) |
| Unsupported claim rate (delivered) | 5.0% (2/40) |
| Answers with any unsupported claim | 15.4% (2/13) |
| Answers that could not be audited | 0 (audit failures: 0) |
| Fabricated citations caught in code (draft level) | 0 |
| Avg / median / p95 latency | 3.4s / 3.1s / 6.8s |
| Avg LLM calls per question (production) | 1.85 (normalization counts as 1) |
| Avg tokens per question (production) | 1216 |
| Extra calls used for measurement only | 30 |

**Guardrail decisions:** SKIPPED=20

**Refusals by how they were detected:** no_claims=4, pipeline=3

**Where delivered-answer verdicts came from:** audit=17

**Refusal rate by type of unanswerable question:**

| Type | Questions | Refused | Rate |
|---|---|---|---|
| `fake_scheme` | 4 | 3 | 75.0% |
| `not_indexed` | 1 | 1 | 100.0% |
| `off_topic` | 3 | 3 | 100.0% |
| `trick_unstated_fact` | 1 | 0 | 0.0% |

**Answerable questions by language** (small n: read with care):

| Language | n | Refused | Retrieval hit |
|---|---|---|---|
| en | 7 | 0 | 7/7 |
| hi | 2 | 0 | 2/2 |
| hinglish | 2 | 0 | 2/2 |

**Claim-evidence similarity (EXPLORATORY, not used in any decision):**

- not enough data: SUPPORTED n=38, NOT_SUPPORTED n=0 claims carry a similarity value (needs both classes; the backend must record `evidence_similarity`).

## Guardrail ON

| Metric | Value |
|---|---|
| Questions run | 20 |
| Infrastructure failures (excluded from all rates) | 0 |
| Scored | 20 (11 answerable, 9 not) |
| Retrieval: expected scheme in top-5 | 100.0% (11/11) |
| Retrieval: top-1 | 72.7% (8/11) |
| Answered (of answerable) | 100.0% (11/11) |
| Over-blocking rate | 0.0% (0/11) |
| Correct refusal rate | 100.0% (9/9) |
| Key-fact coverage (answered only) | 81.8% (9/11) |
| End-to-end fact accuracy (all answerable) | 81.8% (9/11) |
| Unsupported claim rate (delivered) | 0.0% (0/34) |
| Answers with any unsupported claim | 0.0% (0/11) |
| Answers that could not be audited | 0 (audit failures: 0) |
| Fabricated citations caught in code (draft level) | 1 |
| Avg / median / p95 latency | 16.5s / 9.0s / 44.9s |
| Avg LLM calls per question (production) | 3.8 (normalization counts as 1) |
| Avg tokens per question (production) | 2765 |
| Extra calls used for measurement only | 0 |

**Guardrail decisions:** PASS=14, REPAIR=3, SKIPPED=3

**Refusals by how they were detected:** pipeline=9

**Where delivered-answer verdicts came from:** guardrail_pass=10, repair_report=1

**Refusal rate by type of unanswerable question:**

| Type | Questions | Refused | Rate |
|---|---|---|---|
| `fake_scheme` | 4 | 4 | 100.0% |
| `not_indexed` | 1 | 1 | 100.0% |
| `off_topic` | 3 | 3 | 100.0% |
| `trick_unstated_fact` | 1 | 1 | 100.0% |

**Answerable questions by language** (small n: read with care):

| Language | n | Refused | Retrieval hit |
|---|---|---|---|
| en | 7 | 0 | 7/7 |
| hi | 2 | 0 | 2/2 |
| hinglish | 2 | 0 | 2/2 |

**Claim-evidence similarity (EXPLORATORY, not used in any decision):**

- not enough data: SUPPORTED n=34, NOT_SUPPORTED n=0 claims carry a similarity value (needs both classes; the backend must record `evidence_similarity`).

## What the categories mean

- **`off_topic`**: not about schemes at all ("capital of France"). The retrieval threshold refuses these before any generation, so they cost only the **1 normalization call** (normalization always runs first) and no answer or verification calls.
- **`fake_scheme`**: an invented but plausible scheme name. These score *above* the retrieval threshold because they look like scheme questions, so only claim-level verification can catch them.
- **`not_indexed`**: a **real** scheme we do not cover. Retrieval returns confident, real, *wrong-scheme* evidence, which makes it the hardest case.
- **`trick_unstated_fact`**: a real scheme, but the question asks for something the official source never states.

## Honest limitations of this evaluation

- **n=20 is small.** Differences of a few points are noise; the numbers show direction and rough size, not precision.
- **The judge is the same model family as the generator**, and the audit uses the same judge. A shared blind spot would be invisible here. Stronger: a different judge model plus a human-labelled subset.
- **Reused verdicts are not independent.** ON's unsupported-claim rate on PASS answers is zero by construction of PASS.
- **Key facts are matched by substring.** A correct answer that words a number differently ("2 lakh" vs "2.00 Lakh") or answers in Hindi against an English key fact is scored as a miss, which biases against Hindi/Hinglish.
- **Some gold labels are ambiguous** (for example the SVANidhi interest rate is stated two ways in the source). Per-question misses are worth reading in `raw_results.json` before drawing conclusions.
- **Written by the same people who built the system**, so the set may under-represent failure modes we did not think of.

---

Regenerate with `python eval/run_eval.py`. Per-question records (answers, verdicts, top-10 retrieval) are in `raw_results.json`.
