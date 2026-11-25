# Paper plan

**Working title:** *Relevance Is Not Answerability: Claim-Level Verification for
Multilingual RAG on Indian Welfare Schemes*

This is the plan for the paper chosen in November 2026: the "relevance is not
answerability" study (RQ2 + RQ3 in [`RESEARCH.md`](RESEARCH.md)), with the verifier
comparison (RQ4) as its second half. Every number below comes from a file in
`eval/runs/`. A number with no file behind it does not go in the paper.

---

## 1. The argument in one paragraph

A RAG system that refuses when retrieval similarity is low will catch questions
that have nothing to do with its corpus, and almost nothing else. Questions about
schemes that do not exist, real schemes that are not indexed, and facts the source
never states all look like real scheme questions to an embedding model, so they
retrieve confident evidence about the *wrong* scheme. On welfare information, that
is the dangerous failure. We measure it per failure mode, across English, Hindi and
Hinglish questions that are translations of each other, and show which defence
catches which failure: the retrieval gate, a claim-level LLM judge, or deterministic
Python checks (citation existence, number grounding).

## 2. Contributions

1. **A translation-paired, failure-mode-labelled benchmark** for multilingual RAG on
   Indian welfare schemes: 279 questions (93 concepts × English / Hindi / Hinglish).
   It has 105 answerable questions and 174 unanswerable ones in four failure modes. Every answerable key fact
   and every "the source does not say this" label is machine-checked against the
   official source text (`scripts/build_paper_set.py`).
2. **A per-failure-mode analysis of the retrieval gate** showing that similarity
   separates off-topic questions (AUROC 0.989) but not fake schemes (0.643),
   unstated facts (0.616) or non-indexed real schemes (0.734).
3. **A paired language analysis**: the same question retrieves significantly worse
   in Hindi than in English (raw queries), and one fixed threshold over-blocks
   Hindi and Hinglish users more than English ones.
4. **A number-perturbation test** separating what a deterministic number check and
   an embedding-similarity check can and cannot catch.
5. *(pending the LLM runs)* The end-to-end effect of claim-level verification on
   unsupported claims delivered vs over-blocking, per failure mode and language.
6. *(pending human labels)* A comparison of LLM judge vs NLI vs similarity vs hybrid
   verifiers on Hindi/Hinglish-derived claims.

## 3. Results so far (0 LLM calls, reproducible)

### 3.1 Retrieval gate vs failure mode — `eval/runs/paper_set/paper_analysis.md`

| Unanswerable type | n | median best score | AUROC vs answerable (median 0.626) | refused at 0.45 |
|---|---|---|---|---|
| off_topic | 45 | 0.260 | **0.989** | 42/45 (93%) |
| fake_scheme | 45 | 0.578 | 0.643 | 4/45 (9%) |
| not_indexed | 45 | 0.519 | 0.734 | 8/45 (18%) |
| trick_unstated_fact | 39 | 0.580 | 0.616 | 4/39 (10%) |

At 0.45, 7/105 (7%) answerable questions are wrongly refused. **37 of 45**
not-indexed questions pass the gate, typically to a lookalike scheme: PM Kisan
Maandhan → Atal Pension Yojana (0.72), Sukanya Samriddhi → PM Mudra (0.71),
Senior Citizens Savings Scheme → Atal Pension Yojana (0.60–0.68), PMAY-Gramin →
PMAY-Urban (0.57–0.61).

### 3.2 Language, paired by translation — same file

| | both hit | only en | only other | neither | exact McNemar p | median score drop |
|---|---|---|---|---|---|---|
| en vs hi (Recall@1) | 21 | 8 | 0 | 6 | **0.008** | 0.138 |
| en vs hinglish | 24 | 5 | 2 | 4 | 0.453 | 0.104 |

Over-block at 0.45: en 0/35, hi 3/35, hinglish 4/35. **Raw queries only**: the
production normaliser translates to English first, and its effect on this set is
the next measurement (needs the LLM; see §4).

### 3.3 Number perturbation — `eval/runs/numeric_perturbation/results.md`

553 items from 106 numeric sentences in the indexed passages.

| Item type | n | numeric rule flags | similarity flags (cos < 0.41) |
|---|---|---|---|
| faithful, original | 106 | 0% | 4.7% |
| faithful, reformatted (Indian grouping, Devanagari digits, lakh) | 237 | **0%** | 4.6% |
| wrong number, absent from evidence | 106 | **100%** | 5.7% |
| wrong number, present elsewhere in evidence | 104 | 0% (blind spot) | 1.0% |

Embedding similarity is at chance on number swaps (AUROC 0.53). The number rule is
exact on what it can see and blind by construction to swaps within the evidence.
That case is left to the LLM judge, and measuring how often the judge catches it is
experiment E9b below.

## 4. What is still needed, in order

| # | Experiment | Needs | Output |
|---|---|---|---|
| 1 | Normalised-query retrieval on the paper set (does translation-to-English close the Hindi gap? does it equalise over-blocking?) | LLM key; 279 calls, cached | `python eval/run_retrieval.py --questions eval/paper_set.jsonl --variants raw,normalized` |
| 2 | End-to-end guardrail OFF vs ON on the paper set: unsupported-claim rate, correct refusal and over-blocking **per failure mode and per language** | LLM key; ~1.7M tokens, so several days on the Groq free tier or a paid key | `eval/run_eval.py` (needs the same `--questions` flag added) |
| 3 | E9b: run the LLM judge on `numeric_perturbation/items.jsonl` (especially `perturbed_in`) | LLM key; ~550 judge calls | detection rate of the judge vs the rule |
| 4 | Second-family judge on the same claims (validity: same model drafts and judges) | a second provider's key | judge agreement |
| 5 | Human labels: ~300 (claim, passage) pairs, 2 annotators, Cohen's κ; LLM judge vs NLI vs similarity vs hybrid | 2 people, ~6 hours each; a multilingual NLI model | RQ4 table |
| 6 | Native-speaker review of the Hindi and Hinglish questions | 1 Hindi speaker, ~2 hours | fixes to `build_paper_set.py` |

Items 1–3 are the minimum for a workshop paper. Items 4–6 turn it into a stronger
one; item 6 is required before submitting anything.

## 5. Outline

1. Introduction: the failure mode (confident wrong-scheme evidence), why welfare
   information makes it costly, contributions.
2. Related work: hallucination and faithfulness; atomic claims and attribution
   (FActScore, ALCE, AIS); RAG abstention (AbstentionBench, DeclineCraft, selective
   QA); multilingual and code-mixed RAG; LLM-as-judge reliability. Re-check every
   citation before submitting; several 2026 preprints were found only via search.
3. Task and benchmark: corpus (15 schemes, official myScheme API), the four failure
   modes, translation pairing, machine-checked labels, limitations.
4. System: retrieval gate, grounded drafting, claim splitting, judge, Python checks,
   decision policy, repair with re-verification, fail-closed.
5. Experiments: §3.1–3.3 above, then items 1–5 from §4.
6. Discussion: per-mode defences, language fairness of a fixed threshold, cost
   (LLM calls per question), what the number rule can and cannot promise.
7. Limitations: author-written questions, 15 schemes, judge = generator family,
   snapshot data, scheme-level relevance.

## 6. Where to submit (check each call for papers yourself)

Realistic first targets: ACL Student Research Workshop; a workshop on NLP for social
good or on Indian languages (e.g. WILDRE at LREC); FIRE or ICON (Indian NLP/IR
venues); ACM COMPASS. Deadlines were not checked when this plan was written.

## 7. Draft abstract (update the numbers only from `eval/runs/`)

> Retrieval-augmented assistants for government welfare schemes often refuse when
> retrieval similarity is low. We show that this gate catches only questions
> unrelated to the corpus. On a new benchmark of 279 questions about Indian welfare
> schemes, each asked in English, Hindi and Hinglish, retrieval similarity separates
> off-topic questions from answerable ones almost perfectly (AUROC 0.99). It barely
> separates questions about non-existent schemes (0.64), facts the official source
> does not state (0.62) and real schemes outside the corpus (0.73). Most of the
> latter retrieve a lookalike scheme with high confidence. The same questions
> retrieve significantly worse in Hindi than in English, so one fixed threshold
> refuses more Hindi and Hinglish users. A deterministic number check catches every
> number absent from the evidence without false alarms across Indian number
> formats, while embedding similarity is at chance on the same errors. *[End-to-end
> claim-verification results to be added.]*
