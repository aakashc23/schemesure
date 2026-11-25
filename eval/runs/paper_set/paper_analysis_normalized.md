# Paper analysis: retrieval similarity is not answerability

Question file: `paper_set.jsonl`, n=279 (105 answerable, 174 unanswerable), every question in English, Hindi and Hinglish. Retrieval variant: **normalized** (the question translated to English by one LLM call before retrieval, exactly as production does).

## 1. Separability per failure mode

AUROC of the best retrieval score: answerable (positive) vs each kind of unanswerable question. 1.0 = a perfect threshold exists; 0.5 = no signal.

| Unanswerable type | n | median best score | AUROC vs answerable |
|---|---|---|---|
| *(answerable)* | 105 | 0.726 | — |
| `off_topic` | 45 | 0.232 | 1.000 |
| `fake_scheme` | 45 | 0.615 | 0.766 |
| `not_indexed` | 45 | 0.516 | 0.855 |
| `trick_unstated_fact` | 39 | 0.648 | 0.666 |
| **all unanswerable** | 174 | 0.530 | **0.827** |

## 2. What a threshold can catch at a fixed over-block budget

For each budget, the highest threshold that wrongly refuses at most that share of answerable questions, and the share of each failure mode it refuses. The configured production threshold (0.45) is shown for reference.

| Threshold | Over-block (answerable) | `off_topic` refused | `fake_scheme` refused | `not_indexed` refused | `trick_unstated_fact` refused |
|---|---|---|---|---|---|
| 0.52 (budget ≤5%) | 4/105 (4%) | 45/45 (100%) | 9/45 (20%) | 25/45 (56%) | 7/39 (18%) |
| 0.57 (budget ≤10%) | 10/105 (10%) | 45/45 (100%) | 14/45 (31%) | 28/45 (62%) | 12/39 (31%) |
| 0.45 (configured) | 0/105 (0%) | 45/45 (100%) | 3/45 (7%) | 7/45 (16%) | 0/39 (0%) |

## 3. Near misses: real schemes we do not index

Which indexed scheme the retriever returns first for a real scheme that is not in the corpus. A score above the threshold means retrieval alone would pass this question on to the generator with the wrong scheme's evidence.

| Real scheme asked about | top-1 indexed scheme (en / hi / hinglish) | best score (en / hi / hinglish) |
|---|---|---|
| Sukanya Samriddhi Yojana | `pm_mudra` / `pm_mudra` / `pm_mudra` | 0.71 / 0.72 / 0.71 |
| Ayushman Bharat PM-JAY | `pm_vishwakarma` / `pm_ujjwala` / `pm_ujjwala` | 0.57 / 0.61 / 0.65 |
| Ayushman Bharat PM-JAY | `pm_kisan` / `pm_mudra` / `pm_mudra` | 0.54 / 0.51 / 0.51 |
| Atal Innovation Mission | `nmmss` / `nmmss` / `nmmss` | 0.43 / 0.44 / 0.43 |
| PMAY-Gramin | `pmay_urban` / `pmay_urban` / `pmay_urban` | 0.73 / 0.62 / 0.64 |
| PM Kisan Maandhan Yojana | `atal_pension_yojana` / `atal_pension_yojana` / `atal_pension_yojana` | 0.74 / 0.75 / 0.74 |
| Pradhan Mantri Jan Dhan Yojana | `pmsby` / `pmsby` / `pmsby` | 0.52 / 0.52 / 0.52 |
| MGNREGA | `pmkvy_stt` / `pmkvy_stt` / `pmkvy_stt` | 0.47 / 0.47 / 0.47 |
| PM Garib Kalyan Anna Yojana | `pm_kisan` / `pm_kisan` / `pm_mudra` | 0.49 / 0.50 / 0.51 |
| National Pension System | `atal_pension_yojana` / `atal_pension_yojana` / `atal_pension_yojana` | 0.65 / 0.66 / 0.53 |
| PM Surya Ghar Muft Bijli Yojana | `pmay_urban` / `pmay_urban` / `pmay_urban` | 0.46 / 0.49 / 0.49 |
| PM-KUSUM | `pm_fasal_bima` / `pm_fasal_bima` / `pm_ujjwala` | 0.52 / 0.53 / 0.50 |
| Swachh Bharat Mission (Gramin) | `pm_mudra` / `pm_mudra` / `pm_mudra` | 0.43 / 0.42 / 0.45 |
| Senior Citizens Savings Scheme | `atal_pension_yojana` / `atal_pension_yojana` / `atal_pension_yojana` | 0.62 / 0.60 / 0.62 |
| PM Internship Scheme | `atal_pension_yojana` / `pm_vishwakarma` / `atal_pension_yojana` | 0.44 / 0.46 / 0.44 |

**38/45** not-indexed questions score at or above the configured threshold (0.45), i.e. retrieval alone does not stop them.

Fake-scheme questions, most frequent top-1 scheme: `atal_pension_yojana` ×11, `pm_mudra` ×9, `pm_matru_vandana` ×8, `pm_kisan` ×5, `pm_svanidhi` ×4.

## 4. Language effect (paired by translation)

Recall@1 = the expected scheme owns the top chunk. Each answerable concept is compared with itself across languages; McNemar uses only the discordant pairs.

| Comparison | both hit | only en | only other | neither | exact McNemar p | median best-score drop (en − other) |
|---|---|---|---|---|---|---|
| en vs hi | 28 | 0 | 1 | 6 | 1.000 | 0.010 |
| en vs hinglish | 28 | 0 | 2 | 5 | 0.500 | 0.004 |

**One threshold, three languages.** Refusals at the configured threshold (0.45), by language:

| Language | Over-block (answerable) | Unanswerable refused |
|---|---|---|
| en | 0/35, 0.0% [0.0-9.9] | 18/58 (31%) |
| hi | 0/35, 0.0% [0.0-9.9] | 20/58 (34%) |
| hinglish | 0/35, 0.0% [0.0-9.9] | 17/58 (29%) |

## Caveats

- Normalised queries: one LLM call per question translated it to English first, as in production.
- Author-written questions and translations; not yet reviewed by a second native speaker.
- Relevance is scheme-level (no gold passage ids).
- This analysis is about the retrieval gate only. Whether claim-level verification catches what the threshold misses is measured end-to-end by `eval/run_eval.py`, which needs the LLM.

Regenerate: `python eval/paper_analysis.py`.
