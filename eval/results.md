# Evaluation results

> **Status: not yet run.** The harness, the 50-question golden set and the
> metrics are all built and tested; what is missing is a completed run against
> the live model. This file is written automatically by `python eval/run_eval.py`
> and will replace everything below with the measured numbers.
>
> No figures are quoted anywhere in this repository until that run completes.
> Inventing them would undercut the one thing the project is about.

## Why it has not run yet

Groq's free tier caps the account at **200,000 tokens per day** on a rolling
24-hour window. Measured cost of the full benchmark:

| Strategy | Tokens | Fits in one day? |
|---|---|---|
| Both conditions, with claim-level audit | ~368,000 | no |
| Both conditions, audit reused on PASS | ~302,000 | no |
| One condition, with audit | ~145,000–157,000 | **yes** |

So the benchmark needs either two days (one condition each) or a reduced
question set. The harness supports both: `--condition off` / `--condition on`
runs one half, and the results are merged into a single comparison table
automatically on the second run.

## How to run it

One command per condition. Each takes roughly 25–30 minutes.

```bash
python eval/run_eval.py --condition off --delay 2
```

Then, once the daily token budget has refreshed:

```bash
python eval/run_eval.py --condition on --delay 2
```

Finally, push the numbers into the README and the resume bullets:

```bash
python scripts/update_readme_eval.py
```

To check the harness works without spending much budget, run a smoke test first:

```bash
python eval/run_eval.py --limit 6 --delay 2
```

It writes `eval/_checkpoint.json` after every question, so an interrupted run
loses at most one question.

## What will be measured

The same 50 golden-set questions, run twice — guardrail **off** (plain RAG, the
baseline) and **on** — with identical retrieval, model and prompts. The only
difference is whether answers are verified before being returned.

| Metric | What it means | Better |
|---|---|---|
| **Unsupported claim rate** | Of all factual claims in answers the user actually saw, how many were not backed by the retrieved official text. The headline number. | lower |
| Answers containing any unsupported claim | Share of answers with at least one ungrounded claim | lower |
| Correct refusal rate | Unanswerable questions that were refused | higher |
| Over-blocking rate | Answerable questions that were refused anyway — the cost of strictness | lower |
| Key fact coverage | Expected facts that appeared in the answer | higher |
| Retrieval hit rate | Expected scheme present in the top-5 passages | higher |
| LLM calls / question | Measured per request, not estimated | lower |
| Latency (avg / median / p95) | Wall-clock per question | lower |

### Two choices that keep the measurement honest

**The delivered answer is audited, not the draft.** Measuring the draft would
flatter the guardrail — the draft is what it is about to fix. What matters is
what reaches the user, so the final answer is re-checked claim-by-claim. This
also catches the repair step writing a *new* unsupported claim while removing an
old one.

One exception, and it is not a shortcut: when the guardrail returns **PASS** the
answer ships unchanged, so the verdicts it already produced *are* verdicts on the
delivered answer, judged with the same prompt against the same evidence.
Re-auditing would issue a byte-identical duplicate call. REPAIR answers always
get a fresh audit, because that text is new.

**Unanswerable questions are split by failure mode**, because the defence against
each is different and a single averaged "refusal rate" would hide the only
interesting part:

| Type | Example | What catches it |
|---|---|---|
| `off_topic` | "capital of France" | The retrieval threshold, before any LLM call — so these cost **0** LLM calls |
| `fake_scheme` | "PM Free Laptop Yojana 2026" | **Only the guardrail.** Measured at 0.61 similarity — above the 0.45 threshold |
| `not_indexed` | "Sukanya Samriddhi interest rate" | **Only the guardrail.** Measured at 0.71 — real, confident, wrong-scheme evidence |
| `trick_unstated_fact` | A real scheme, asked for something the source never states | The guardrail, claim by claim |

The baseline is expected to already refuse the `off_topic` questions, because the
retrieval threshold catches those before any generation happens. The guardrail's
contribution should show up almost entirely in the other three categories — which
is exactly why they are reported separately.

## The golden set

50 questions, built by `scripts/build_golden_set.py`:

- **35 answerable**, covering all 15 schemes — 36 English, 7 Hindi (Devanagari),
  7 Hinglish (romanised).
- **15 unanswerable**, split across the four failure modes above.

Every expected fact is verified to appear in its scheme's `.md` file before the
set is written; if one does not, the build fails rather than producing a
benchmark that measures the author's memory.

## Limitations to state alongside any number this produces

- **50 questions is small.** Differences of a few percent are noise. The numbers
  show direction and rough magnitude, not precision.
- **The judge is the same model family as the generator**, so a shared blind spot
  would be invisible to this evaluation. A stronger setup uses a different model
  as judge plus a human-labelled subset to validate it.
- **The audit uses an LLM too**, so the unsupported-claim rate is itself an
  estimate, deliberately biased toward strictness by the judge prompt.
- **Key fact coverage uses substring matching**, so a correct answer that phrases
  a number differently is scored as a miss.
- **Written by the same person who built the system**, so the golden set may
  under-represent failure modes that were never considered.

---

*Regenerate with `python eval/run_eval.py`. Per-question records, including every
answer and verdict, are written to `eval/raw_results.json`.*
