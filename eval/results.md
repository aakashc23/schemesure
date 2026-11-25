# Evaluation results

The same 50 golden-set questions run with the verification layer **off** (plain RAG, the baseline) and **on**. Each question is normalized once and both conditions retrieve with that identical query; the only difference is whether answers are checked before being returned.

## Setup

- **Date:** 2026-11-25
- **Questions:** n=50 (35 answerable, 15 unanswerable) of 50 in the golden set
- **LLM:** `qwen/qwen3.8-27b` via https://api.groq.com/openai/v1
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (cosine)
- **Retrieval:** top-5, refusal threshold 0.45
- **Block threshold:** 0.5 (unsupported-claim fraction above which an answer is refused)
- **Corpus:** 312 chunks from 15 central schemes
- **Claim audit:** on

## Headline comparison

| Metric | Guardrail OFF | Guardrail ON | Change |
|---|---|---|---|
| **Unsupported claim rate** (delivered answers) | 3.3% (3/92) | 0.0% (0/81) | better (-3.3 pp) |
| Answers containing any unsupported claim | 8.1% (3/37) | 0.0% (0/34) | better (-8.1 pp) |
| Correct refusal rate (unanswerable) | 78.6% (11/14) | 100.0% (14/14) | better (+21.4 pp) |
| Over-blocking rate (answerable refused) | 2.9% (1/35) | 2.9% (1/35) | no change |
| Answered rate (of answerable) | 97.1% (34/35) | 97.1% (34/35) | no change |
| Key-fact coverage (answered only) | 78.9% (30/38) | 78.9% (30/38) | no change |
| **End-to-end fact accuracy** (all answerable; refusal = 0) | 76.9% (30/39) | 76.9% (30/39) | no change |
| Questions fully correct (all key facts) | 74.3% (26/35) | 74.3% (26/35) | no change |
| Avg LLM calls / question | 1.88 | 3.74 | +1.86 |
| Avg tokens / question | 1240 | 2690 | |
| Median latency | 1.9s | 20.4s | |
| p95 latency | 9.6s | 39.2s | |
| Infrastructure failures (excluded) | 0 | 0 | |
| Questions scored (paired: the same ones in both) | 50 | 50 | |

*Refusal rates leave out answers whose refusal status could not be determined (no claim audit ran for them, so a decline in the model's own words was undetectable): OFF 1, ON 1.*

*Retrieval is identical in both conditions by construction (shared cached normalization): expected scheme in top-5 for 97.1% (34/35) of answerable questions.*

### How to read this

- **Unsupported claim rate** is the headline: of all factual claims in answers the user actually saw, how many the judge found unsupported by the retrieved official text. For ON, a PASS answer has zero unsupported claims *by definition of PASS* and its verdicts come from the same judge, so ON's figure is not independent evidence; use `--audit-all` for an independent re-audit.

- **Correct refusal vs over-blocking** trade against each other: refusing more catches more unanswerable questions and blocks more good answers. Refusal here uses one detector for both conditions (pipeline status, canned refusal text, or an answer with no checkable claims); answers the detector could not classify are left out of the refusal rates in both.

- **End-to-end fact accuracy** counts a refusal as zero facts, so it does not flatter a system that simply refuses more. Key-fact coverage (answered questions only) is shown beside it for that reason.

- **Latency** includes time the harness spent waiting out free-tier rate limits, so read the median, not the mean.

## Guardrail OFF (baseline)

| Metric | Value |
|---|---|
| Questions run | 50 |
| Infrastructure failures (excluded from all rates) | 0 |
| Scored | 50 (35 answerable, 15 not) |
| Retrieval: expected scheme in top-5 | 97.1% (34/35) |
| Retrieval: top-1 | 80.0% (28/35) |
| Answered (of answerable) | 97.1% (34/35) |
| Over-blocking rate | 2.9% (1/35) |
| Correct refusal rate | 78.6% (11/14) |
| Key-fact coverage (answered only) | 78.9% (30/38) |
| End-to-end fact accuracy (all answerable) | 76.9% (30/39) |
| Unsupported claim rate (delivered) | 3.3% (3/92) |
| Answers with any unsupported claim | 8.1% (3/37) |
| Answers that could not be audited | 1 (audit failures: 1) |
| Fabricated citations caught in code (draft level) | 0 |
| Avg / median / p95 latency | 3.2s / 1.9s / 9.6s |
| Avg LLM calls per question (production) | 1.88 (normalization counts as 1) |
| Avg tokens per question (production) | 1240 |
| Extra calls used for measurement only | 80 |

**Guardrail decisions:** SKIPPED=50

**Refusals by how they were detected:** no_claims=6, pipeline=6

**Where delivered-answer verdicts came from:** audit=43

**Refusal rate by type of unanswerable question:**

| Type | Questions | Refused | Rate |
|---|---|---|---|
| `fake_scheme` | 4 | 3 | 75.0% |
| `not_indexed` | 4 | 3 | 75.0% |
| `off_topic` | 5 | 5 | 100.0% |
| `trick_unstated_fact` | 1 | 0 | 0.0% |

**Answerable questions by language** (small n: read with care):

| Language | n | Refused | Retrieval hit |
|---|---|---|---|
| en | 26 | 1 | 26/26 |
| hi | 4 | 0 | 4/4 |
| hinglish | 5 | 0 | 4/5 |

**Claim-evidence similarity (EXPLORATORY, not used in any decision):**

- not enough data: SUPPORTED n=89, NOT_SUPPORTED n=0 claims carry a similarity value (needs both classes; the backend must record `evidence_similarity`).

## Guardrail ON

| Metric | Value |
|---|---|
| Questions run | 50 |
| Infrastructure failures (excluded from all rates) | 0 |
| Scored | 50 (35 answerable, 15 not) |
| Retrieval: expected scheme in top-5 | 97.1% (34/35) |
| Retrieval: top-1 | 80.0% (28/35) |
| Answered (of answerable) | 97.1% (34/35) |
| Over-blocking rate | 2.9% (1/35) |
| Correct refusal rate | 100.0% (14/14) |
| Key-fact coverage (answered only) | 78.9% (30/38) |
| End-to-end fact accuracy (all answerable) | 76.9% (30/39) |
| Unsupported claim rate (delivered) | 0.0% (0/81) |
| Answers with any unsupported claim | 0.0% (0/34) |
| Answers that could not be audited | 0 (audit failures: 0) |
| Fabricated citations caught in code (draft level) | 2 |
| Avg / median / p95 latency | 17.3s / 20.4s / 39.2s |
| Avg LLM calls per question (production) | 3.74 (normalization counts as 1) |
| Avg tokens per question (production) | 2690 |
| Extra calls used for measurement only | 0 |

**Guardrail decisions:** PASS=40, REPAIR=4, SKIPPED=6

**Refusals by how they were detected:** pipeline=16

**Where delivered-answer verdicts came from:** guardrail_pass=33, repair_report=1

**Refusal rate by type of unanswerable question:**

| Type | Questions | Refused | Rate |
|---|---|---|---|
| `fake_scheme` | 4 | 4 | 100.0% |
| `not_indexed` | 4 | 4 | 100.0% |
| `off_topic` | 5 | 5 | 100.0% |
| `trick_unstated_fact` | 1 | 1 | 100.0% |

**Answerable questions by language** (small n: read with care):

| Language | n | Refused | Retrieval hit |
|---|---|---|---|
| en | 26 | 1 | 26/26 |
| hi | 4 | 0 | 4/4 |
| hinglish | 5 | 0 | 4/5 |

**Claim-evidence similarity (EXPLORATORY, not used in any decision):**

- not enough data: SUPPORTED n=81, NOT_SUPPORTED n=0 claims carry a similarity value (needs both classes; the backend must record `evidence_similarity`).

## What the categories mean

- **`off_topic`**: not about schemes at all ("capital of France"). The retrieval threshold refuses these before any generation, so they cost only the **1 normalization call** (normalization always runs first) and no answer or verification calls.
- **`fake_scheme`**: an invented but plausible scheme name. These score *above* the retrieval threshold because they look like scheme questions, so only claim-level verification can catch them.
- **`not_indexed`**: a **real** scheme we do not cover. Retrieval returns confident, real, *wrong-scheme* evidence, which makes it the hardest case.
- **`trick_unstated_fact`**: a real scheme, but the question asks for something the official source never states.

## Honest limitations of this evaluation

- **n=50 is small.** Differences of a few points are noise; the numbers show direction and rough size, not precision.
- **The judge is the same model family as the generator**, and the audit uses the same judge. A shared blind spot would be invisible here. Stronger: a different judge model plus a human-labelled subset.
- **Reused verdicts are not independent.** ON's unsupported-claim rate on PASS answers is zero by construction of PASS.
- **Key facts are matched by substring.** A correct answer that words a number differently ("2 lakh" vs "2.00 Lakh") or answers in Hindi against an English key fact is scored as a miss, which biases against Hindi/Hinglish.
- **Some gold labels are ambiguous** (for example the SVANidhi interest rate is stated two ways in the source). Per-question misses are worth reading in `raw_results.json` before drawing conclusions.
- **Written by the same people who built the system**, so the set may under-represent failure modes we did not think of.

---

Regenerate with `python eval/run_eval.py`. Per-question records (answers, verdicts, top-10 retrieval) are in `raw_results.json`.
