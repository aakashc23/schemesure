# Evaluation results

## Status (2026-11-16)

| Benchmark | Questions | State | Results |
|---|---|---|---|
| Retrieval, raw vs normalised queries | 50 (35 answerable) | **complete** | [`retrieval_results.md`](retrieval_results.md) |
| Refusal-threshold sweep | 50 | **complete** | [`threshold_results.md`](threshold_results.md) |
| Guardrail OFF vs ON, preliminary subset | 20 (11 answerable, 9 unanswerable) | **complete, PRELIMINARY** | [`runs/subset-20q-c5a697aa/results.md`](runs/subset-20q-c5a697aa/results.md) |
| Guardrail OFF vs ON, full set | 50 | **not yet run** | this file will be replaced by the run |

The full OFF/ON run needs roughly 300,000 tokens. Groq's free tier allows about
200,000 a day and also limits output tokens to 1,000 a minute, so the run is done one
condition per day, ideally on a key separate from the live demo. A case-by-case
reading of the preliminary subset is in
[`docs/TEST_REPORT.md` §3](../docs/TEST_REPORT.md#3-end-to-end-evaluation-preliminary).
Read it before quoting any OFF/ON number.

## How to run

Retrieval only (0 LLM calls for `raw`, 1 cached call per question for `normalized`):

```bash
python -m eval.run_retrieval --variants raw,normalized
```

Full OFF/ON benchmark, one condition per day:

```bash
python -m eval.run_eval --condition off --delay 8
```

```bash
python -m eval.run_eval --condition on --delay 8
```

The second command merges with the first automatically, but only if the model,
embeddings, threshold and golden set are unchanged; otherwise it refuses to compare.
If a run is interrupted, add `--resume`. Questions that failed because the LLM
service was unavailable are retried, and until then they are excluded from **both**
conditions so the comparison stays paired.

A cheaper preliminary run on a subset that covers every language and failure mode:

```bash
python -m eval.run_eval --stratified 20 --delay 8
```

Publish the numbers into the README (it refuses to fill resume bullets from a
subset):

```bash
python scripts/update_readme_eval.py
```

## What is measured

The same golden-set questions, run twice — guardrail **off** (plain RAG, the
baseline) and **on** — with identical retrieval (one cached normalisation per
question, shared by both), model and prompts.

| Metric | What it means | Better |
|---|---|---|
| **Unsupported claim rate** | Of all factual claims in answers the user actually saw, how many the judge found unsupported by the retrieved official text | lower |
| Answers containing any unsupported claim | Share of answers with at least one ungrounded claim | lower |
| Correct refusal rate | Unanswerable questions that were refused | higher |
| Over-blocking rate | Answerable questions refused anyway — the cost of strictness | lower |
| Key-fact coverage / end-to-end accuracy | Expected facts present in the answer; end-to-end counts a refusal as 0 | higher |
| Retrieval hit rate | Expected scheme present in the top-5 passages | higher |
| LLM calls, tokens, latency | Measured per request, not estimated | lower |

Every proportion is reported with its n; empty denominators print as "n/a", never 0%.

### Choices that keep the measurement honest

- **The delivered answer is audited, not the draft.** When the guardrail returns
  PASS, the answer ships unchanged, so its verdicts are reused (re-auditing would be
  a byte-identical call). REPAIR answers carry the verdicts of their own re-check.
  `--audit-all` forces an independent re-audit of everything.
- **One refusal detector for both conditions**: pipeline status, canned refusal
  text, or an answer with no checkable claims.
- **Infrastructure failures are excluded, in pairs**, not counted as refusals.
- **Unanswerable questions are split by failure mode**, because the defence against
  each differs:

| Type | Example | What catches it |
|---|---|---|
| `off_topic` | "capital of France" | The retrieval threshold, before generation (1 LLM call: normalisation) |
| `fake_scheme` | "PM Free Laptop Yojana 2026" | Claim-level verification; all 5 score 0.45–0.62, above the threshold (in the 20-question subset, 3 of 4 were already refused with the guardrail OFF) |
| `not_indexed` | "Sukanya Samriddhi interest rate" | Not the threshold: up to 0.71, confident wrong-scheme evidence. In measured runs the draft declined (refused as `no_claims`); claim verification is the backstop |
| `trick_unstated_fact` | A real scheme, asked for something the source never states | Claim-level verification |

## The golden set

50 questions, built by `scripts/build_golden_set.py`: 35 answerable across all 15
schemes and 15 unanswerable across the four failure modes. By language: 36 English,
7 Hindi (Devanagari), 7 Hinglish (romanised). Every expected fact is checked to
appear in its scheme's source file when the set is built.

## Limitations to state alongside any number

- **Small n.** 50 questions overall; 4–5 answerable Hindi or Hinglish questions.
- **Same judge everywhere.** The judge is the same model as the generator, and the
  audit uses that judge too. In the preliminary subset the splitter and judge were over-strict at
  least once. A human-labelled subset and a different judge model are planned.
- **Key facts are matched by substring.** Both misses in the preliminary subset were
  correct answers worded differently.
- **Scheme-level relevance.** A hit means the right scheme was retrieved, not every
  passage needed to answer fully.
- **Author-written questions** may under-represent failure modes nobody thought of.
