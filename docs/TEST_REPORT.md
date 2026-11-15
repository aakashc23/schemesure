# Test report

Results recorded on **2026-11-16** for the `midterm-improvements` code. Each section
gives the exact command, so every number can be reproduced. Nothing here is estimated.

| Layer | What it shows | Result |
|---|---|---|
| [1. Automated tests](#1-automated-tests) | the code does what it claims, offline | **574 passed, 0 failed** |
| [2. Retrieval benchmark](#2-retrieval-benchmark-full-run) | the right scheme comes back | full run, n = 50 |
| [3. End-to-end evaluation](#3-end-to-end-evaluation-preliminary) | guardrail OFF vs ON | **preliminary**, subset |
| [4. Manual and live checks](#4-manual-and-live-checks) | the deployed app works | see table |
| [5. What is not tested](#5-what-is-not-tested) | the gaps | listed |

---

## 1. Automated tests

```bash
python -m pytest -q
```

No API key and no network: every LLM call is replaced by a fake, while retrieval
uses the real embedding model and a real Chroma index. Run time on the development
laptop (CPU only) was 57–211 s across two runs on the same day, most of it the Streamlit UI tests.

| File | Tests | What it pins down |
|---|---|---|
| `test_eligibility_inputs.py` | 90 | income parsing ("2.5 lakh", "1,50,000", values under ₹1,000), age parsing, occupation synonyms and vocabulary, broad occupations → NEED_MORE_INFO |
| `test_ui.py` | 60 | every tab renders (Streamlit AppTest, fake API); status pills; claims/evidence/pipeline views; eligibility text, form and re-check flows; error messages; old and new API response shapes |
| `test_api_features.py` | 55 | refusal reasons, evidence summary, stage trace, answer cache, `/eligibility/check`, `/eligibility/options`, `/schemes/{id}`, `/health` and `/metrics` fields |
| `test_guardrail_grounding.py` | 54 | number/link grounding on real chunk text in English, Hindi and Hinglish; re-check after repair; citation restriction; judge alignment; similarity recording never changing a verdict |
| `test_eligibility.py` | 51 | the rule engine: age/income boundaries (inclusive), null rules, missing fields, gender, occupation, a decisive failure outranking missing data, Indian digit grouping |
| `test_llm_client_policy.py` | 35 | error classification; no retry on quota/auth/too-large; waits only for per-minute limits; per-request deadline; provider text never reaches a response |
| `test_review_fixes.py` | 33 | one regression test group per finding from the adversarial code review |
| `test_api.py` | 32 | endpoint contracts through the real retriever: verified answer with citations, out-of-scope refusal, repair, block, fabricated citations, fail-closed, length limit (400), 503 without a key |
| `test_guardrail.py` | 28 | `decide()` at every boundary, fabricated citation ids, fail-closed |
| `test_eval_run.py` | 26 | OFF/ON pairing, infra-failure exclusion from both conditions, refusal detection, report and README publish guards |
| `test_query_normalizer.py` | 24 | JSON parsing, language labels, fallback on failure |
| `test_ingest.py` | 21 | heading-aware chunking, token budget with the real tokenizer, deterministic ids |
| `test_llm_client.py` | 21 | wait-hint parsing, JSON extraction, call logging |
| `test_answer_cache.py` | 13 | TTL expiry, LRU eviction, key normalisation, copies not shared objects |
| `test_prompts_and_model_lock.py` | 12 | prompt placeholders, the repair prompt not containing the raw question, thread-safe model loading |
| `test_eval_common.py` | 9 | stratified selection, Wilson intervals, AUROC, fact matching with digit boundaries |
| `test_eval_retrieval.py` | 7 | Recall@k, MRR, threshold sweep |
| `test_retriever_embed.py` | 3 | similarity reuses Chroma's stored passage vectors, identical to re-embedding; only claims are encoded |
| **Total** | **574** | |

**Result:** 574 passed, 0 failed (full run, exit code 0).

CI (`.github/workflows/ci-deploy.yml`) runs the same suite on every push to main and every pull request into main: it validates the data, builds the index, runs the tests and scans tracked files for secrets; a separate parallel job builds and smoke-tests the Docker image.

### How the tests were reviewed

The new code was reviewed by two independent reviewers with different briefs
(correctness, and UI + evaluation). They raised 16 findings, which were reproduced
before fixing. Among them:

- a broad occupation produced a false NOT_ELIGIBLE;
- unpaired infra-failure exclusion made OFF/ON non-comparable;
- the Hugging Face deploy omitted the theme file;
- "8.2%" was not checked by the number grounding.

Each finding has a regression test in `test_review_fixes.py`.

---

## 2. Retrieval benchmark (full run)

```bash
python -m eval.run_retrieval --variants raw,normalized
```

All 50 golden-set questions; 35 answerable. Relevance is scheme-level. 95% Wilson
intervals in brackets. Full tables: [`eval/retrieval_results.md`](../eval/retrieval_results.md).

| Variant | Recall@1 | Recall@3 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| Raw question (0 LLM calls) | 74.3% [57.9–85.8] | 91.4% [77.6–97.0] | 94.3% [81.4–98.4] | 0.806 |
| Normalised to English (production) | 80.0% [64.1–90.0] | 91.4% [77.6–97.0] | 97.1% [85.5–99.5] | 0.851 |

Per language (normalised): English Recall@1 76.9% (n=26), Hindi 100% (n=4),
Hinglish 80.0% (n=5). Normaliser fallbacks during the run: 0.

**Threshold check** ([`eval/threshold_results.md`](../eval/threshold_results.md)),
normalised queries, threshold 0.45:

| Group | Best-score range | Refused |
|---|---|---|
| Answerable | 0.528 – 0.882 | 0 / 35 |
| Off-topic | 0.109 – 0.375 | 5 / 5 |
| Real scheme, not indexed | 0.433 – 0.710 | 1 / 4 |
| Non-existent scheme | 0.451 – 0.619 | 0 / 5 |
| Fact not stated in source | 0.568 | 0 / 1 |

AUROC (best score): 1.000 answerable vs off-topic; 0.916 answerable vs all
unanswerable. The 0.45 threshold was originally chosen on queries that overlap this
set, so this is a consistency check, not a held-out validation.

---

## 3. End-to-end evaluation (preliminary)

```bash
python -m eval.run_eval --stratified 20 --condition both --delay 8
python -m eval.run_eval --stratified 20 --condition both --delay 8 --resume
```

**PRELIMINARY.** These results cover 20 of the 50 questions: 11 answerable and 9 unanswerable. The subset was chosen to cover every language and every failure mode, not in proportion to the full set. The first pass lost one ON question (`a16`) to the free tier's output-token-per-minute limit. That failure was excluded from both conditions. It was rerun with `--resume` after the retry-policy fix (DECISIONS §8), so all 20 are scored as pairs. Full report: [`eval/runs/subset-20q-c5a697aa/results.md`](../eval/runs/subset-20q-c5a697aa/results.md). Per-question answers and retrieval: `raw_results.json` in the same folder.

| Metric | Guardrail OFF (baseline RAG) | Guardrail ON |
|---|---|---|
| Unsupported claim rate (delivered answers) | 5.0% (2/40) | 0.0% (0/34) |
| Answers containing any unsupported claim | 15.4% (2/13) | 0.0% (0/11) |
| Correct refusal (unanswerable) | 7/9 | 9/9 |
| Over-blocking (answerable refused) | 0/11 | 0/11 |
| End-to-end key-fact accuracy (refusal = 0) | 9/11 | 9/11 |
| Avg LLM calls / question | 1.85 | 3.80 |
| Avg tokens / question | 1,216 | 2,765 |
| Median / p95 latency | 3.1 s / 6.8 s | 9.0 s / 44.9 s |

The p95 latency includes time spent waiting out rate limits.

### What the per-question records show

The headline numbers need reading against the individual cases. We checked each case below by hand against the source files.

1. **The two "unsupported" claims in the OFF condition are not invented scheme facts.**
   - **`a15`:** "documents for PM Ujjwala … and Aadhaar". The official list contains "Aadhaar (not mandatory for Assam and Meghalaya)", and that passage was retrieved. The claim splitter produced the unconditional claim "Aadhaar is required", dropping the exception. The judge rejected the claim as stated. That is defensible, but strict: a human reader would probably accept "Aadhaar" in a list of required documents. The ON condition then *repaired* the answer, dropping Aadhaar from the list of required documents and keeping only the sentence that it is not mandatory in Assam and Meghalaya. The answer was shown as "Partially verified". The same thing happened in the separately recorded demo run (DEMO_GUIDE Part 2), so the behaviour is reproducible, not a one-off.
   - **`u15`:** a statement about which passages were retrieved ("the provided passages only contain details about the Kisan Credit Card and Atal Pension Yojana"), not a claim about a scheme.

   So the 5.0% → 0.0% change reflects the judge's strictness and the claim splitter's loss of a condition, not hallucinations the guardrail prevented. The same judge produced both the guardrail verdicts and the audit. **This is why a human-labelled claim set (Roadmap P0-5) and a different judge model (P1-8) come first in the second half.**

2. **Most fake-scheme questions were refused by the model itself.** In both conditions, the model's own draft for 4 of the 9 unanswerable questions asserted nothing checkable. Those count as refusals in both conditions. The guardrail added value on 2 questions (`u10`, `u15`). There the baseline gave a half-refusal: "I do not have official information on X" followed by facts about an unrelated scheme. The guardrail replaced each with a clean refusal.

3. **Both key-fact misses are scoring artifacts.** They are identical in both conditions:
   - `a05` answered "paid Income Tax in **the** last assessment year".
   - `a12` answered "18 years, and the maximum age is 40 years".

   Both are correct, but the key facts are matched as literal substrings. True accuracy on this subset is 11/11 in both conditions. The golden set was **not** edited after seeing results; the planned fix is normalised or entailment-based fact matching on the expanded set.

4. **Scheme-level retrieval hides incomplete evidence.** For `a15` the right scheme was retrieved, but only one of its two "Documents Required" passages was. Both conditions therefore omitted Photo ID, ration card and address proof. Recall@k counts this as a hit. Gold passage ids (Roadmap P0-2) would expose it.

**Conclusion at this stage:** two things work as designed, and one effect is not shown yet.

- **Works:** no delivered answer contained a claim the guardrail's own judge marked unsupported (ON verdicts were reused from the guardrail, not independently re-audited, so this is zero by construction for PASS answers); no over-blocking on this subset; half-refusals were turned into clean refusals.
- **Works:** the infrastructure for the full comparison is in place.
- **Not shown yet:** that the guardrail prevents *real* hallucinations at a measurable rate. The baseline produced too few of them on 20 questions, and the one judge we have is both strict and unvalidated.

---

## 4. Manual and live checks

| Check | How | Result |
|---|---|---|
| API without an LLM key | `uvicorn app.main:app`, key empty | `/health` 200 (15 schemes, 312 chunks, `llm_ready=false`); `/ask` and `/eligibility` 503 with a clear message; `/eligibility/check` 200 with no LLM; `/schemes/does_not_exist` 404 |
| Form eligibility | `POST /eligibility/check` age 40, income 1,50,000, Bihar, "kisan", male, OBC | "kisan" normalised to farmer; Atal Pension ELIGIBLE at the inclusive age-40 boundary |
| Data validation | `python scripts/validate_data.py` | passed, 15 schemes; 14 content warnings (e.g. one reference URL not on a .gov.in host) |
| Index build | `python -m core.ingest` | 312 chunks, none over the 110-token budget |
| Live app before this milestone | built-in browser, `schemesureaakash.streamlit.app` | Hinglish "PM Kisan ka paisa kitna milta hai?" → Verified, Hinglish answer, cited `pm_kisan#benefits#0` |
| GitHub Actions CI on `main` | run 43760842109 for commit `4d2f1eb` | success: "Validate data and run tests" and "Build and smoke-test the container" both passed; the manual HF deploy job was skipped as designed |
| Live app after deployment | built-in browser, after the merge to `main` (commit `4d2f1eb`) | new UI served; Ask and form eligibility work, see [§4.1](#41-live-deployment-check) |

### 4.1 Live deployment check

Checked on 2026-11-16 in the built-in browser at
<https://schemesureaakash.streamlit.app>, a few minutes after `main` was pushed and
Streamlit Cloud redeployed.

| Check | Result |
|---|---|
| Sidebar status | API reachable · 15 schemes · 312 passages · `qwen/qwen3.8-27b` · verification on · key configured · answer cache on |
| Hindi example, "अटल पेंशन योजना में शामिल होने की आयु सीमा क्या है?" | **Verified**, Hindi answer (minimum 18, maximum 40), 4 LLM calls, 1.3 s, 2/2 claims supported, source `atal_pension_yojana#eligibility#0` |
| Trap example, "What is the interest rate on Sukanya Samriddhi Yojana?" | **Not enough official info**, refusal reason `no_claims`, 3 LLM calls, 0.6 s |
| Eligibility → Fill in a form, occupation *Street vendor*, all else empty | 2 likely eligible, 8 need more info, 5 not eligible, **0 LLM calls** |

**Bug found by this check and fixed:** for the `no_claims` refusal, the Verification
tab read "Decision: PASS — 0 of 0 claim(s) supported", which looks like a pass on a
withheld answer. The UI now labels what happened to the answer ("nothing to verify —
answer withheld", and similar labels for the other refusal reasons), with a regression
test in `tests/test_ui.py`.

---

## 5. What is not tested

- **The LLM judge's accuracy.** Tests prove the code acts correctly on given verdicts,
  not that the verdicts are right. That needs human labels (Roadmap P0-5).
- **The full 50-question OFF/ON comparison.** Only a subset has run (§3).
- **Load.** One process, one worker; no concurrency test beyond the cache lock.
- **Real browsers other than the built-in one.** UI tests are headless AppTest.
- **Prompt injection.** Planned as a robustness suite (Roadmap P2-14).
