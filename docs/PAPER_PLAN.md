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

## 3. Results so far (reproducible)

Two retrieval variants: **raw** (the question as typed, 0 LLM calls) and
**translated** (one LLM call rewrites it in English first, as production does).
Production refuses on the translated score, so that column is the one that matters
for the system; the raw column shows what the translation step buys.

### 3.1 Retrieval gate vs failure mode — `eval/runs/paper_set/paper_analysis*.md`

| Unanswerable type | n | AUROC raw | **AUROC translated** | refused at 0.45 (translated) |
|---|---|---|---|---|
| off_topic | 45 | 0.989 | **1.000** | 45/45 (100%) |
| fake_scheme | 45 | 0.643 | **0.766** | 3/45 (7%) |
| not_indexed | 45 | 0.734 | **0.855** | 7/45 (16%) |
| trick_unstated_fact | 39 | 0.616 | **0.666** | 0/39 (0%) |
| all unanswerable | 174 | 0.750 | **0.827** | 55/174 (32%) |

Translated, 0.45 wrongly refuses **0/105** answerable questions, but **38 of 45**
not-indexed questions pass the gate, each to a lookalike scheme: PM Kisan Maandhan →
Atal Pension Yojana (0.74–0.75), PMAY-Gramin → PMAY-Urban (0.62–0.73), Sukanya
Samriddhi → PM Mudra (0.71–0.72), National Pension System → Atal Pension (0.53–0.66).
Even at the threshold that allows 10% over-blocking (0.57), a threshold refuses only
31% of fake-scheme and 31% of unstated-fact questions.

### 3.2 Language, paired by translation — same files

Recall@1 (right scheme ranked first), 35 answerable concepts per language:

| | English | Hindi | Hinglish | en vs hi (only-en / only-hi, McNemar p) |
|---|---|---|---|---|
| raw | 82.9% | **60.0%** | 74.3% | 8 / 0, **p = 0.008** |
| translated | 80.0% | 82.9% | 85.7% | 0 / 1, p = 1.0 |

Over-block at 0.45: raw en 0/35, hi 3/35, hinglish 4/35; **translated 0/35 in all
three**. So the translation step is what makes one fixed threshold fair across
languages: it removes the Hindi retrieval gap and the extra Hindi/Hinglish
over-blocking. It does **not** help with the failure modes: those questions are
about schemes, so they translate into fluent scheme questions too.

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
That case is left to the LLM judge.

**E9b — the LLM judge on the same items** (`eval/runs/numeric_perturbation/judge_results.md`;
seeded random sample of 28 sentences, sized to the day's quota, one claim per call):

| Item type | n | LLM judge flags | number rule flags |
|---|---|---|---|
| correct sentence | 28 | 1 (3.6%) — a garbled table fragment | 0 |
| wrong number present elsewhere in evidence | 27 | **27 (100%)** | 0 |

The two checks are complementary: the rule covers every number absent from the
evidence at zero false alarms; the judge covered every in-evidence swap in this
sample. Caveats: each test claim is the source sentence with one number changed,
which is easier than a paraphrased claim; a few swaps are implausible ("2500000%")
and the generator garbled two units; n is small. Remaining items can be judged
later with `--resume`-style re-runs of `eval/run_numeric_judge.py`.

## 4. What is still needed, in order

| # | Experiment | Needs | Output |
|---|---|---|---|
| 1 | ~~Normalised-query retrieval on the paper set~~ **Done** (§3.1–3.2) | — | `eval/runs/paper_set/` |
| 2 | End-to-end guardrail OFF vs ON on the paper set: unsupported-claim rate, correct refusal and over-blocking **per failure mode and per language** | LLM key; ~1.7M tokens, so several days on the Groq free tier or a paid key | `eval/run_eval.py` (needs the same `--questions` flag added) |
| 3 | E9b: LLM judge on the number test — **sample done** (27/27 in-evidence swaps caught); extend to paraphrased claims and the remaining items | LLM key; ~500 judge calls | `eval/run_numeric_judge.py` |
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
