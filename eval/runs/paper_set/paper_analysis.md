# Paper analysis: retrieval similarity is not answerability

Question file: `paper_set.jsonl`, n=279 (105 answerable, 174 unanswerable), every question in English, Hindi and Hinglish. Retrieval variant: **raw** (the question as typed, 0 LLM calls). Production refuses on the LLM-normalised query, which has not been run on this set yet.

## 1. Separability per failure mode

AUROC of the best retrieval score: answerable (positive) vs each kind of unanswerable question. 1.0 = a perfect threshold exists; 0.5 = no signal.

| Unanswerable type | n | median best score | AUROC vs answerable |
|---|---|---|---|
| *(answerable)* | 105 | 0.626 | — |
| `off_topic` | 45 | 0.260 | 0.989 |
| `fake_scheme` | 45 | 0.578 | 0.643 |
| `not_indexed` | 45 | 0.519 | 0.734 |
| `trick_unstated_fact` | 39 | 0.580 | 0.616 |
| **all unanswerable** | 174 | 0.511 | **0.750** |

## 2. What a threshold can catch at a fixed over-block budget

For each budget, the highest threshold that wrongly refuses at most that share of answerable questions, and the share of each failure mode it refuses. The configured production threshold (0.45) is shown for reference.

| Threshold | Over-block (answerable) | `off_topic` refused | `fake_scheme` refused | `not_indexed` refused | `trick_unstated_fact` refused |
|---|---|---|---|---|---|
| 0.44 (budget ≤5%) | 5/105 (5%) | 42/45 (93%) | 4/45 (9%) | 8/45 (18%) | 4/39 (10%) |
| 0.47 (budget ≤10%) | 9/105 (9%) | 43/45 (96%) | 5/45 (11%) | 9/45 (20%) | 5/39 (13%) |
| 0.45 (configured) | 7/105 (7%) | 42/45 (93%) | 4/45 (9%) | 8/45 (18%) | 4/39 (10%) |

## 3. Near misses: real schemes we do not index

Which indexed scheme the retriever returns first for a real scheme that is not in the corpus. A score above the threshold means retrieval alone would pass this question on to the generator with the wrong scheme's evidence.

| Real scheme asked about | top-1 indexed scheme (en / hi / hinglish) | best score (en / hi / hinglish) |
|---|---|---|
| Sukanya Samriddhi Yojana | `pm_mudra` / `pm_mudra` / `pm_mudra` | 0.71 / 0.65 / 0.66 |
| Ayushman Bharat PM-JAY | `pm_vishwakarma` / `atal_pension_yojana` / `pm_svanidhi` | 0.56 / 0.54 / 0.59 |
| Ayushman Bharat PM-JAY | `pm_kisan` / `pmjjby` / `pm_mudra` | 0.54 / 0.56 / 0.51 |
| Atal Innovation Mission | `nmmss` / `pm_mudra` / `pm_mudra` | 0.43 / 0.43 / 0.51 |
| PMAY-Gramin | `pmay_urban` / `pmay_urban` / `pm_matru_vandana` | 0.57 / 0.61 / 0.43 |
| PM Kisan Maandhan Yojana | `atal_pension_yojana` / `pm_fasal_bima` / `atal_pension_yojana` | 0.72 / 0.59 / 0.72 |
| Pradhan Mantri Jan Dhan Yojana | `pmsby` / `kisan_credit_card` / `pm_kisan` | 0.52 / 0.48 / 0.52 |
| MGNREGA | `pmkvy_stt` / `pmsby` / `pmay_urban` | 0.47 / 0.47 / 0.37 |
| PM Garib Kalyan Anna Yojana | `pm_fasal_bima` / `pm_kisan` / `pm_matru_vandana` | 0.49 / 0.51 / 0.52 |
| National Pension System | `atal_pension_yojana` / `atal_pension_yojana` / `atal_pension_yojana` | 0.65 / 0.65 / 0.57 |
| PM Surya Ghar Muft Bijli Yojana | `pmay_urban` / `pm_ujjwala` / `pmay_urban` | 0.48 / 0.44 / 0.54 |
| PM-KUSUM | `pm_fasal_bima` / `pm_fasal_bima` / `pm_ujjwala` | 0.52 / 0.49 / 0.49 |
| Swachh Bharat Mission (Gramin) | `pm_mudra` / `pmay_urban` / `pm_matru_vandana` | 0.43 / 0.42 / 0.48 |
| Senior Citizens Savings Scheme | `atal_pension_yojana` / `atal_pension_yojana` / `atal_pension_yojana` | 0.62 / 0.60 / 0.68 |
| PM Internship Scheme | `pm_vishwakarma` / `nmmss` / `pmkvy_stt` | 0.46 / 0.43 / 0.55 |

**37/45** not-indexed questions score at or above the configured threshold (0.45), i.e. retrieval alone does not stop them.

Fake-scheme questions, most frequent top-1 scheme: `atal_pension_yojana` ×10, `pm_matru_vandana` ×9, `pm_mudra` ×6, `pmsby` ×4, `pm_kisan` ×4.

## 4. Language effect (paired by translation)

Recall@1 = the expected scheme owns the top chunk. Each answerable concept is compared with itself across languages; McNemar uses only the discordant pairs.

| Comparison | both hit | only en | only other | neither | exact McNemar p | median best-score drop (en − other) |
|---|---|---|---|---|---|---|
| en vs hi | 21 | 8 | 0 | 6 | 0.008 | 0.138 |
| en vs hinglish | 24 | 5 | 2 | 4 | 0.453 | 0.104 |

**One threshold, three languages.** Refusals at the configured threshold (0.45), by language:

| Language | Over-block (answerable) | Unanswerable refused |
|---|---|---|
| en | 0/35 (0%) 0.0% [0.0-9.9] | 17/58 (29%) |
| hi | 3/35 (9%) 8.6% [3.0-22.4] | 23/58 (40%) |
| hinglish | 4/35 (11%) 11.4% [4.5-26.0] | 18/58 (31%) |

## Caveats

- Raw queries only. The LLM normaliser translates Hindi/Hinglish to English before retrieval in production; its effect on this set is not yet measured.
- Author-written questions and translations; not yet reviewed by a second native speaker.
- Relevance is scheme-level (no gold passage ids).
- This analysis is about the retrieval gate only. Whether claim-level verification catches what the threshold misses is measured end-to-end by `eval/run_eval.py`, which needs the LLM.

Regenerate: `python eval/paper_analysis.py`.
