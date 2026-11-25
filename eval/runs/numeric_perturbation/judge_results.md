# Number-perturbation test, LLM judge (E9b)

- **Date:** 2026-11-25  ·  **model:** `qwen/qwen3.8-27b`  ·  **seed:** 0
- **Judged:** 55 items from 28 sentences, one claim per call, each with its own top-5 evidence
- A claim is **flagged** when the production judge (`core.guardrail.judge_claims`, including the Python check that the cited passage was retrieved) returns NOT_SUPPORTED.
- Items are a seeded random sample of sentences; the quota decides how many.

## Flag rates: judge vs the Python number check, on the same items

For correct items a flag is a **false alarm**; for wrong-number items it is a **detection**.

| Item type | n | LLM judge flags | Python number check flags | either flags |
|---|---|---|---|---|
| `faithful_original` | 28 | 3.6% [0.6-17.7] | 0.0% [0.0-12.1] | 3.6% [0.6-17.7] |
| `faithful_format` | 0 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| `perturbed_out` | 0 | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| `perturbed_in` | 27 | 100.0% [87.5-100.0] | 0.0% [0.0-12.5] | 100.0% [87.5-100.0] |

## Wrong numbers the judge passed (0)

These reach the user in production: the number exists elsewhere in the evidence, so the Python check cannot see them either.


## Correct sentences the judge rejected (1)

- `faithful_original`: *3,000/- Second Instalment Childbirth is registered.*

Regenerate: `python eval/run_numeric_judge.py --report-only`.
