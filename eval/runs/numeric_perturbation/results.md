# Number-perturbation test (E9)

- **Date:** 2026-11-25  ·  **seed:** 0  ·  **evidence:** top-5 retrieved passages for the claim, source passage always included
- **Items:** 553 from 106 numeric sentences in 312 indexed passages
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **LLM judge:** not run (no API key in this run). The items are saved to `items.jsonl` so the judge and an NLI model can be scored on the same set.

## Detection rates

Flag rate per item type. For faithful items a flag is a **false alarm** (lower is better); for perturbed items it is a **detection** (higher is better). Similarity flags when cosine(claim, source) < 0.410, the threshold that keeps false alarms on faithful items at about 5%.

| Item type | n | numeric rule flags | similarity flags |
|---|---|---|---|
| `faithful_original` | 106 | 0.0% [0.0-3.5] | 4.7% [2.0-10.6] |
| `faithful_format` | 237 | 0.0% [0.0-1.6] | 4.6% [2.6-8.1] |
| `perturbed_out` | 106 | 100.0% [96.5-100.0] | 5.7% [2.6-11.8] |
| `perturbed_in` | 104 | 0.0% [0.0-3.6] | 1.0% [0.2-5.2] |

## Can similarity tell a wrong number from a right one?

AUROC of cosine(claim, source): faithful (positive) vs perturbed. 0.5 = no signal.

| Faithful vs | AUROC |
|---|---|
| `perturbed_out` | 0.5278 |
| `perturbed_in` | 0.5251 |

## Examples

- `faithful_format` (60 → same value, reformatted) — rule flags: none; similarity 0.994  
  *Atal Pension Yojana (Benefits): (iii) Return of the pension wealth to the nominee of the subscriber: After the demise of both the subscriber and the spouse, the nominee of the subscriber shall be enti*
- `faithful_format` (60 → same value, reformatted) — rule flags: none; similarity 0.994  
  *Atal Pension Yojana (Benefits): (iii) Return of the pension wealth to the nominee of the subscriber: After the demise of both the subscriber and the spouse, the nominee of the subscriber shall be enti*
- `perturbed_out` (60 → 1060) — rule flags: ['1060']; similarity 0.984  
  *Atal Pension Yojana (Benefits): (iii) Return of the pension wealth to the nominee of the subscriber: After the demise of both the subscriber and the spouse, the nominee of the subscriber shall be enti*
- `perturbed_out` (80 → 87) — rule flags: ['87']; similarity 0.991  
  *Atal Pension Yojana (Benefits): Contributions to the Atal Pension Yojana (APY) are eligible for tax benefits similar to the National Pension System (NPS) under section 87CCD(1).*
- `perturbed_in` (60 → 3000) — rule flags: none; similarity 0.972  
  *Atal Pension Yojana (Benefits): (iii) Return of the pension wealth to the nominee of the subscriber: After the demise of both the subscriber and the spouse, the nominee of the subscriber shall be enti*
- `perturbed_in` (80 → 40) — rule flags: none; similarity 0.989  
  *Atal Pension Yojana (Benefits): Contributions to the Atal Pension Yojana (APY) are eligible for tax benefits similar to the National Pension System (NPS) under section 40CCD(1).*

## False alarms of the numeric rule (0)


## Reading this
- `perturbed_in` is constructed to defeat an exact-match check, so a low rule detection rate there is expected; it measures how often the blind spot *can* occur, not how often a model *does* make that error.
- Claims are source sentences with one number changed, not LLM-written claims; real claims are paraphrased, which matters for similarity.
- Regenerate: `python eval/run_numeric_perturbation.py`.
