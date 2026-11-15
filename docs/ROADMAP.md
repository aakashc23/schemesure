# Roadmap: second half (December 2026 – May 2027)

The first half produced a deployed, integrated prototype (see
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md)). The second half is about **evidence**:
measuring what the system does at a scale that supports a conclusion, adding the
verification method the research plan proposed (NLI), and widening coverage. Each item
below builds on an existing module; none requires redesigning the architecture.

Priorities: **P0** blocks the final evaluation, **P1** is a core research deliverable,
**P2** strengthens the work, **P3** is polish and dissemination.

Months refer to the research plan's Gantt chart (plan phases in brackets).

---

## P0 — Evaluation at a meaningful scale (Dec 2026 – Feb 2027) [Evaluation dataset; Experiments]

| # | Work item | Builds on | Done when |
|---|---|---|---|
| 1 | **Run the full 50-question benchmark, guardrail OFF vs ON**, one condition per day, on a separate Groq key so the live demo's quota is untouched | `eval/run_eval.py` (resume, per-condition checkpoints, infra-failure exclusion already implemented) | `eval/results.md` holds both conditions with matching metadata, n and intervals |
| 2 | **Add gold evidence chunk ids** to every answerable question (propose candidates automatically by matching key facts against the expected scheme's chunks, then confirm by hand) | deterministic chunk ids in `core/ingest.py` | passage-level Recall@k is reported alongside scheme-level |
| 3 | **Grow the golden set to 300+** questions, ≥30% Hindi/Hinglish, with translation-paired questions so language effects are not confounded with difficulty; collect real phrasings from Hindi/Hinglish speakers (junior member's planned role) | `scripts/build_golden_set.py` (fact-in-source check) | set passes the builder's checks; composition table published |
| 4 | **Calibrate the evidence threshold on a dev split** and report on a held-out split, per failure mode | `eval/run_retrieval.py` threshold sweep | threshold choice no longer uses test questions |
| 5 | **Human-label a claim subset** (~200 claim–passage pairs, two annotators, agreement reported) | claims and verdicts already saved per question by the eval | labels file + Cohen's κ |

## P1 — The verification research (Jan – Mar 2027) [Verification module; Experiments]

| # | Work item | Builds on | Done when |
|---|---|---|---|
| 6 | **NLI verifier**: a multilingual NLI cross-encoder scoring each (claim, cited passage) pair, behind a config flag; CPU-only, loaded once | `Guardrail` already takes an `embed_fn`; add an analogous `nli_fn`. The decision policy is a pure function, so only its inputs change | entailment probability recorded per claim, like `evidence_similarity` today |
| 7 | **Verifier comparison (RQ4)**: LLM judge vs NLI vs similarity vs a calibrated hybrid against the human labels | items 5 and 6; similarity is already recorded | table of agreement / precision / recall; decision on which signals gate answers |
| 8 | **Cross-model judge**: a different model family judges the same claims | `LLM_MODEL` is configurable; add a judge-model override | disagreement analysis written up |
| 8b | **Claim splitter keeps conditions**: "Aadhaar (not mandatory in Assam/Meghalaya)" must not become "Aadhaar is required"; found in both the preliminary eval (`a15`) and the demo run | `CLAIM_SPLITTER_SYSTEM` in `app/prompts.py`; the Ujjwala question becomes a regression case | the repaired Ujjwala answer keeps Aadhaar with its exception |
| 9 | **Plain-LLM baseline (RQ5)**: answer without retrieval, audit against the same evidence | eval harness conditions | three-way comparison table |

## P2 — Coverage and robustness (Feb – Apr 2027) [Corpus; Experiments and tuning]

| # | Work item | Builds on | Done when |
|---|---|---|---|
| 10 | **Expand the corpus to 25–30 schemes**; add PDF ingestion (with OCR only where needed) for official guideline documents not on myScheme | `scripts/fetch_schemes.py`, `validate_data.py`, section-aware chunker | `validate_data.py` passes on the larger corpus; eligibility rules encoded with provenance |
| 11 | **Scheduled re-fetch with diff**: flag changed amounts or conditions for human review instead of relying on `last_verified` | raw responses in `data/raw/` | a report of changed fields per run |
| 12 | **Scaling retrieval**: metadata pre-filter by scheme/ministry when a scheme is named; hybrid lexical + dense retrieval for scheme names and numbers | `core/retriever.py` | Recall@k does not drop as the corpus grows |
| 13 | **Ablations**: embedding model (MiniLM / e5 / BGE-M3), top-k, block threshold, repair vs regenerate | eval flags | ablation tables |
| 14 | **Robustness suite**: prompt-injection questions, perturbed numbers, very long answers | `numeric_and_link_grounding`, judge isolation | pass/fail table in the test report |
| 15 | **Conditional eligibility rules** (e.g. Stand-Up India's "if male, must be SC/ST"; PMAY-U income bands) | `core/eligibility.py` already separates modelled and unmodelled conditions | those conditions move from "check yourself" to checked rules, with tests |
| 16 | **Verdict cache** keyed by (claim, chunk id) to cut repeat verification cost | `core/answer_cache.py` pattern | cache hit rate on the benchmark reported |

## P3 — Dissemination (Mar – May 2027) [Web UI and deployment; Report, paper, final demo]

| # | Work item | Done when |
|---|---|---|
| 17 | Small user study (E10) on comprehension and trust of the verification view | protocol, consent, results |
| 18 | Demo hardening: separate demo key, pre-demo warm-up, documented fallback (Docker image run locally) | `docs/DEMO_GUIDE.md` checklist rehearsed |
| 19 | Final report and a paper draft built around RQ1–RQ5 | drafts reviewed by the guide |
| 20 | Release the evaluation set and code with documentation | tagged release |

---

## What is deliberately *not* planned

- **Fine-tuning a model.** The research question is about verification of a pretrained
  model's output, and fine-tuning would need labelled data at a scale the project does
  not have.
- **A React frontend.** The Streamlit UI already shows everything the research needs;
  effort is better spent on evaluation.
- **State-level schemes.** Thousands of schemes with heterogeneous sources; out of
  scope for a year-long project. The architecture (metadata per chunk, rules per
  scheme) would extend to them.

## Risks

| Risk | Mitigation |
|---|---|
| Free-tier LLM limits (about 200k tokens/day; per minute: 30 requests, 8k tokens, 1k output tokens) limit evaluation speed and are shared with the live demo | one condition per day; normalisation cache; separate eval key; resume after interruption |
| Judge and generator share blind spots | cross-model judge (8) and human labels (5) |
| Live demo host sleeps when idle and cold-starts slowly | warm-up step in the demo checklist; Docker image as local fallback |
| Official pages change | re-fetch with diff (11); `last_verified` shown on every citation |
