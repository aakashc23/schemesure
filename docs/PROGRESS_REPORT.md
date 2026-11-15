# Progress report: first half (June – November 2026)

**Project:** SchemeSure — a hallucination-aware, multilingual RAG assistant for Indian
government schemes (B.Tech final-year project, Group A7).
**Reporting point:** half-yearly evaluation, about five months into a year-long project.
**Basis:** the research plan (`SchemeSure_Research_Plan_A7_final.pdf`), the repository at
the `midterm-improvements` merge, and the live deployment.

This report separates what is **done and measured**, what is **built but not yet
validated**, and what is **not started**. Numbers come from files in the repository;
each is linked.

---

## 1. Summary

| Area | State |
|---|---|
| End-to-end prototype (ask, verify, eligibility, UI, API) | **Working and deployed** |
| Verification method (claim splitting, LLM judge, Python checks, repair + re-check) | **Implemented**; effectiveness measured only on a preliminary subset |
| Verification method proposed in the plan (NLI / similarity scoring) | **Not implemented**; similarity is recorded per claim, NLI is planned |
| Corpus | 15 of the planned 25–30 schemes |
| Evaluation dataset | 50 of the planned 300–500 questions; 28% Hindi/Hinglish vs ≥30% target |
| Retrieval evaluation | **Run in full** (50 questions, raw vs normalised queries) |
| Guardrail OFF vs ON evaluation | **Preliminary subset only** (see §4); the full run is blocked by the free-tier LLM quota |
| Plain-LLM baseline, human labels, user study | **Not started** |

The engineering half of the plan is ahead of schedule. The research half — evidence
that the verification works, at a scale that supports a conclusion, and the NLI
comparison the plan proposed — is behind, and is the focus of the second half
([`ROADMAP.md`](ROADMAP.md)).

## 2. Completed

| Plan item | What exists | Evidence |
|---|---|---|
| Official-source corpus | 15 central schemes fetched from the myScheme JSON API; raw responses kept; gaps written as "Not specified in official source" | `data/raw/`, `data/schemes/`, `scripts/fetch_schemes.py` |
| Data validation | 10 automated checks incl. rule provenance; runs in CI | `scripts/validate_data.py` |
| Section-aware chunking and index | 312 passages ≤110 tokens, deterministic ids, cosine Chroma | `core/ingest.py` |
| Multilingual query handling | LLM normalisation to English with visible fallback | `core/query_normalizer.py` |
| Evidence gate | refusal before generation below 0.45 | `core/retriever.py`, `eval/threshold_results.md` |
| Grounded generation with citations | evidence-only prompt, `[n]` citations, reply in the user's language | `core/generator.py`, `app/prompts.py` |
| Claim-level verification | atomic claims, LLM judge on evidence only, Python citation validation, Python number/link grounding, pure-function decision policy, repair with full re-verification, fail-closed | `core/guardrail.py`, `core/generator.py` |
| Rule-based eligibility | 15 rules files with the official sentence per rule; free-text and form input; three-valued result | `core/eligibility.py`, `data/schemes/*.rules.json` |
| API | 9 endpoints, input limits, error mapping, answer cache | `app/main.py` |
| Web UI | Ask (with claims, evidence and pipeline views), Eligibility, Schemes, How it works, System | `ui/streamlit_app.py` |
| Monitoring | every LLM call logged; `/metrics` and the System tab read it | `app/llm_client.py` |
| Testing and CI | 574 offline tests; CI validates data, builds the index, runs tests, scans for secrets, builds and smoke-tests the Docker image | `tests/`, `.github/workflows/ci-deploy.yml` |
| Deployment | live on Streamlit Community Cloud; Docker image as the portable path | `docs/DEPLOYMENT.md` |
| Retrieval benchmark | Recall@1/3/5, MRR@10 with Wilson intervals, raw vs normalised, threshold sweep | `eval/retrieval_results.md`, `eval/threshold_results.md` |

## 3. Partially complete

| Plan item | Done | Missing |
|---|---|---|
| Verification by similarity + NLI | claim–evidence cosine similarity computed and stored per claim | NLI model; validation of either signal against human labels; using them in the decision |
| Evaluation dataset | 50 questions, 15 unanswerable split by failure mode, facts checked against sources at build time | 300+ questions, gold passage ids, translation-paired Hindi/Hinglish, real user phrasings |
| Guardrail effectiveness (RQ3) | harness with paired conditions and infra-failure exclusion; preliminary subset run | full 50-question run in both conditions; Wilson intervals in the OFF/ON report; then the larger set |
| Threshold calibration (RQ2) | sweep per failure mode; AUROC for answerable vs off-topic and vs all unanswerable | dev/held-out split; the current threshold was chosen on overlapping queries |
| Corpus scale | 15 schemes | 25–30 schemes; PDF guideline ingestion |
| Eligibility coverage | 7 rule fields per scheme | conditional rules (e.g. Stand-Up India "if male, SC/ST"), PMAY-U income bands |

## 4. Measured results so far

### 4.1 Retrieval (complete run, n = 50, 35 answerable)

From [`eval/retrieval_results.md`](../eval/retrieval_results.md):

| Query form | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| Raw (as typed) | 74.3% [57.9–85.8] | 94.3% [81.4–98.4] | 0.806 |
| Normalised (production) | 80.0% [64.1–90.0] | 97.1% [85.5–99.5] | 0.851 |

Normalisation helped Hindi (Recall@1 2/4 → 4/4) and Hinglish (3/5 → 4/5). With 4–5
questions per language, this is a direction, not a result.

### 4.2 Evidence threshold (complete run)

From [`eval/threshold_results.md`](../eval/threshold_results.md), normalised queries at
the configured 0.45:

| Group | Refused at 0.45 |
|---|---|
| Answerable (over-blocking) | 0/35 |
| Off-topic | 5/5 |
| Real scheme, not indexed | 1/4 |
| Non-existent scheme | 0/5 |
| Fact not stated | 0/1 |

The threshold separates off-topic questions perfectly (AUROC 1.00) and catches little else (1 of 4 unindexed-scheme questions). Questions about fake or unindexed schemes mostly score like real ones (0.43–0.71), so catching them is almost entirely the job of claim-level verification. This is the quantitative case for the guardrail.

### 4.3 Guardrail OFF vs ON (preliminary)

**PRELIMINARY, 20 of 50 questions** (11 answerable, 9 unanswerable, all paired):

| Metric | OFF (baseline RAG) | ON |
|---|---|---|
| Unsupported claims in delivered answers | 2/40 (5.0%) | 0/34 (0.0%) |
| Unanswerable questions correctly refused | 7/9 | 9/9 |
| Answerable questions wrongly refused | 0/11 | 0/11 |
| LLM calls per question | 1.85 | 3.80 |

Read with the case analysis in
[`TEST_REPORT.md` §3](TEST_REPORT.md#3-end-to-end-evaluation-preliminary). Both
"unsupported" baseline claims turned out not to be invented scheme facts. In one, the
claim splitter dropped an exception ("Aadhaar, not mandatory in Assam/Meghalaya" became
"Aadhaar is required"), the judge rejected it, and the repair removed a true document
from the answer. So the
subset shows no over-blocking, and that nothing the guardrail's own judge marked unsupported was shipped. ON answers were not re-audited independently, so ON's 0/34 is zero by construction of PASS, not independent proof of safety. The guardrail also turns half-refusals into clean ones. It does **not** yet show a measured
reduction in real hallucinations. That needs the full run, more questions and
human-validated verdicts.

## 5. Report–code inconsistencies found and resolved

An audit compared the research plan, the README and the code. Resolved items:

| Inconsistency | Resolution |
|---|---|
| Plan describes NLI + similarity verification; code uses an LLM judge | Documented as a deliberate deviation with reasons ([`RESEARCH.md` §4](RESEARCH.md#4-deviations-from-the-research-plan-and-why)); similarity now recorded; NLI scheduled (P1) |
| Plan: regenerate with a stricter prompt; code: repair | Documented; repair is now re-verified before showing |
| README presented eval results as pending without a harness that could finish under the quota | Harness rewritten to resume, pair conditions and exclude infra failures; retrieval benchmark run in full |
| "Partially verified" answers were never re-checked | Fixed (re-verification) |
| Provider outages shown as "no official information" | Fixed (distinct refusal reasons) |
| Citations could list all retrieved passages | Fixed (verified evidence only) |
| Income "2.5 lakh" parsed as 25 | Fixed (`parse_income`) |
| Model size stated as ~120 MB | Corrected to ~470 MB |
| Plan: deploy on HF Spaces / Render | Documented: HF Docker Spaces need a paid plan, Render's 512 MB is below the ~1.2 GB footprint; Streamlit Cloud used, Docker image kept and tested in CI |

Still open (planned, not hidden): NLI verifier, plain-LLM baseline, 300+ questions,
human labels, 25–30 schemes, RAGAS-style reference-free metrics.

## 6. What changed in this milestone

The functional, AI and engineering changes made for the mid-term are listed with
reasons in [`DECISIONS.md` §8](DECISIONS.md#8-mid-term-revision-november-2026). In
brief:

- **Verification:** re-check after repair; deterministic number/link grounding;
  citations restricted to verified evidence; refusal reasons; similarity signal recorded.
- **Reliability:** error-type-aware retry policy, per-request deadline, answer cache,
  sanitised provider errors, thread-safe startup on Streamlit Cloud.
- **Eligibility:** form path without the LLM, occupation vocabulary and synonyms,
  lakh/crore income parsing, broad occupations → NEED_MORE_INFO.
- **UI:** rebuilt around inspectability — per-answer claims table, evidence table with
  scores against the threshold, stage-by-stage trace, scheme detail view, live system
  metrics, a "How it works" diagram, with step-by-step text that shows the live configuration from /health.
- **Evaluation:** retrieval benchmark, threshold sweep, resumable paired OFF/ON
  harness, honest labelling of partial runs.
- **Docs:** this report, [`ARCHITECTURE.md`](ARCHITECTURE.md),
  [`RESEARCH.md`](RESEARCH.md), [`DEMO_GUIDE.md`](DEMO_GUIDE.md),
  [`TEST_REPORT.md`](TEST_REPORT.md), [`ROADMAP.md`](ROADMAP.md).

## 7. Honest assessment

**Strong:** the system design. The parts that must not be wrong are deterministic,
unit-tested Python; the LLM is confined to language tasks; every failure path refuses
honestly; and every answer can be inspected claim by claim in the UI. The retrieval
benchmark is complete and reproducible.

**Weak:** the evidence. There is not yet a full, paired measurement of what the
guardrail buys, the judge has not been validated against human labels, the judge and
generator are the same model, and the dataset is small and author-written. Until those
exist, "hallucination-aware" is a design property, not a measured one.

**Risk:** the free-tier LLM limits (per minute: 30 requests, 8k tokens, 1k output tokens; about 200k tokens a day) limit both the
evaluation and the live demo, which share a key. A separate evaluation key is the
cheapest fix.
