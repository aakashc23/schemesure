# Research foundation

This document states the research problem SchemeSure addresses, the method as it is
**actually implemented**, how it is evaluated, where it deliberately differs from the
original research plan, and the experiments planned for the second half of the
project. It is written to be checked against the code: every implemented claim
names the file that implements it.

Status labels used throughout:

- **Implemented** — in the code, covered by tests, running in the live demo.
- **In progress** — partially implemented, or implemented but not yet validated.
- **Planned** — designed, not built. Scheduled in [`ROADMAP.md`](ROADMAP.md).

---

## 1. Problem

Indian welfare-scheme information is spread across many portals, written in formal
English, and consequential: a wrong benefit amount or eligibility condition can cost
a citizen an entitlement or a wasted trip to a government office.

Retrieval-augmented generation (RAG) grounds a language model in retrieved documents
[6], but retrieval does not guarantee faithfulness: models still blend retrieved
evidence with parametric knowledge, and noisy retrieval produces fluent answers about
the wrong scheme [1], [2]. Standard RAG pipelines have no step that checks the answer
before it reaches the user.

### 1.1 Formal statement

Given a user question *q* in language *L* ∈ {English, Hindi (Devanagari), Hinglish
(romanised Hindi)} and a corpus *C* of official scheme documents, return either

- an answer *a* in language *L* with citations, such that **every atomic factual claim
  in *a* is supported by a passage retrieved from *C***, or
- an explicit abstention ("I do not have official information on this").

The system is judged on two kinds of error, which trade off against each other:

| Error | Meaning | Cost to the user |
|---|---|---|
| **Unsupported claim delivered** | a claim reaches the user that the official text does not state | acts on wrong information |
| **Over-blocking** | an answerable question is refused | one extra web search |

The design deliberately prefers over-blocking to unsupported claims.

### 1.2 Failure modes the system must handle

| ID | Failure mode | Example | Which defence catches it |
|---|---|---|---|
| F1 | Off-topic question | "capital of France" | retrieval-score threshold |
| F2 | Real scheme, not in the corpus | "Sukanya Samriddhi interest rate" | claim-level verification (retrieval score is high: measured 0.71) |
| F3 | Non-existent scheme | "PM Free Laptop Yojana 2026" | claim-level verification |
| F4 | Real scheme, fact not stated in the source | asking for a figure the page never gives | claim-level verification |
| F5 | Wrong number or condition in the draft | "Rs 8,000" instead of "Rs 6,000" | claim judge + deterministic numeric grounding check |
| F6 | Fabricated evidence citation by the judge | judge cites a chunk that was never retrieved | Python citation validation |
| F7 | Language mismatch | Hinglish query retrieves poorly | query normalisation to English before retrieval |

---

## 2. Research questions

| RQ | Question | Status |
|---|---|---|
| RQ1 | Does normalising Hindi and Hinglish queries to English close the retrieval gap with English queries (Recall@k, MRR)? | **In progress** — first full run on the 50-question set (`eval/retrieval_results.md`): Recall@1 for Hindi 2/4 → 4/4 and Hinglish 3/5 → 4/5 with normalisation, overall MRR@10 0.806 → 0.851. Per-language n is 4–5, so this is a direction, not a finding. **Translation-paired set built** (`eval/paper_set.jsonl`, 35 concepts × 3 languages): on raw queries Hindi Recall@1 is significantly below English for the same questions (21 both / 8 only-en / 0 only-hi, exact McNemar p = 0.008; `eval/runs/paper_set/paper_analysis.md`). Normalised run pending (needs the LLM) |
| RQ2 | How well does a retrieval-score threshold separate answerable from unanswerable questions, per failure mode (F1–F4)? | **In progress** — sweep on normalised queries (`eval/threshold_results.md`): at 0.45, 0/35 answerable refused, 5/5 off-topic refused, but only 1/4 not-indexed and 0/5 fake-scheme questions refused (AUROC 1.00 vs off-topic, 0.92 vs all unanswerable). Confirms the threshold only handles F1. **Replicated at n=279** (`eval/paper_set.jsonl`, raw queries): AUROC 0.989 vs off-topic but 0.643 fake-scheme, 0.734 not-indexed, 0.616 unstated-fact; 37/45 not-indexed questions pass 0.45. Held-out calibration planned |
| RQ3 | How much does claim-level verification reduce unsupported claims reaching the user compared with baseline RAG, and what does it cost in over-blocking, latency and LLM calls? | **In progress** — harness implemented (`eval/run_eval.py`, guardrail OFF vs ON); preliminary subset only, full run planned |
| RQ4 | Which claim verifier agrees best with human judgement: LLM-as-judge, an NLI cross-encoder, embedding similarity, or a hybrid? | **Planned** — similarity signal is already recorded per claim; NLI verifier and human labels planned |
| RQ5 | How do plain LLM, baseline RAG and SchemeSure compare on faithfulness, answer accuracy and abstention quality? | **Planned** — plain-LLM condition designed, not built |

---

## 3. Method as implemented

```
question ─► normalise (LLM) ─► retrieve (embeddings + Chroma) ─► evidence gate (Python)
         ─► draft answer (LLM, evidence-only, [n] citations)
         ─► split into atomic claims (LLM) ─► judge each claim vs evidence (LLM)
         ─► validate cited chunk ids (Python) ─► numeric/link grounding (Python)
         ─► claim–evidence similarity (embeddings, recorded only)
         ─► decision policy (Python): PASS / REPAIR / BLOCK
         ─► REPAIR: rewrite from verified claims only (LLM) ─► re-verify (same checks)
```

| Step | What it does | Where | Status |
|---|---|---|---|
| Corpus construction | 15 central schemes from the official myScheme JSON API; raw responses kept for audit; gaps written as "Not specified in official source" | `scripts/fetch_schemes.py`, `data/raw/`, `data/schemes/` | Implemented |
| Corpus validation | 10 automated checks (front-matter, sections, URLs, entities, rule provenance) | `scripts/validate_data.py` | Implemented |
| Section-aware chunking | split by headings, then list items; token-budgeted to the embedding model's own tokenizer (≤110 tokens); deterministic ids `scheme#section#i` | `core/ingest.py` | Implemented |
| Embedding + index | `paraphrase-multilingual-MiniLM-L12-v2` [12], 384-dim, ChromaDB with cosine space | `core/ingest.py`, `core/retriever.py` | Implemented |
| Query normalisation | one LLM call → English query + language label; visible fallback when it fails | `core/query_normalizer.py` | Implemented |
| Evidence gate | refuse before drafting when best cosine < 0.45 (selective answering, cf. [14]) | `core/retriever.py`, `core/generator.py` | Implemented; calibration in progress |
| Grounded drafting | answer only from numbered passages, cite `[n]`, reply in the user's language | `app/prompts.py`, `core/generator.py` | Implemented |
| Atomic claim splitting | one sentence can carry several facts; split before judging (cf. FActScore [3]) | `core/guardrail.py` | Implemented |
| Claim judging | LLM-as-judge [10] sees only the evidence (never the question), must name the supporting chunk id; unknown → NOT_SUPPORTED | `core/guardrail.py` | Implemented |
| Citation validation | the cited chunk id must exist in the retrieved set, checked in Python (attribution must be to an identified source, cf. AIS [15], ALCE [13]) | `core/guardrail.py` | Implemented |
| Numeric/link grounding | every number ≥ 10, every percentage, decimal and lakh/crore amount, and every URL in the answer must occur in the evidence (amounts compared after normalising "₹2.00 Lakh" = "2,00,000"); a rule-based check independent of the LLM | `core/guardrail.py::numeric_and_link_grounding` | Implemented (new this milestone) |
| Claim–evidence similarity | cosine similarity between each claim and its cited passage, recorded for calibration; **not** used for routing | `core/guardrail.py` | In progress |
| Decision policy | pure function: all supported → PASS; none supported or > 50% unsupported → BLOCK; otherwise REPAIR | `core/guardrail.py::decide` | Implemented |
| Repair + re-verification | rewrite from verified claims only, then verify the rewrite again before showing it (cf. draft–verify–revise in CoVe [9]) | `core/generator.py` | Implemented (re-verification new this milestone) |
| Fail-closed | if any verification step cannot run, refuse and say the service is unavailable — never show unchecked text | `core/generator.py` | Implemented |
| Eligibility | LLM extracts a profile; plain Python evaluates per-scheme rules with a reason per rule and the official sentence each rule came from; missing data → NEED_MORE_INFO | `core/eligibility.py` | Implemented |
| Monitoring | every LLM call logged to SQLite (prompt, model, tokens, latency, attempts) | `app/llm_client.py` | Implemented |

### 3.1 Why the "LLM for language, Python for rules" split matters

The parts of the system that must not be wrong — whether a cited passage exists,
whether a number appears in the source, whether the unsupported ratio exceeds the
limit, whether an age is above a limit — are deterministic Python and unit-tested.
The LLM is used only where language understanding is unavoidable: translating,
drafting, splitting and judging text. An LLM checking an LLM is another generation,
not a control; the controls are code.

---

## 4. Deviations from the research plan (and why)

| Plan said | Implemented | Reason | Plan for the second half |
|---|---|---|---|
| Verification by semantic similarity + NLI entailment (DeBERTa-v3 MNLI/XNLI) with a combined confidence score | LLM-as-judge per claim + Python citation validation + Python numeric/link grounding; similarity recorded but not used for routing | An LLM judge handles Hindi/Hinglish-derived claims and paraphrase without adding a second model to an app that already uses ~1.2 GB RSS on a free host. NLI models need calibration on our data before they can gate answers. | Add an NLI verifier (multilingual DeBERTa-style cross-encoder [22], [23]) and compare LLM-judge vs NLI vs similarity vs hybrid against human labels (RQ4). |
| Deliver / regenerate with stricter prompt / abstain | PASS / REPAIR (rewrite from verified claims only, then re-verify) / BLOCK | Regenerating with a stricter prompt can reproduce the same error; rewriting from a fixed list of verified claims cannot introduce a claim that already failed, and the rewrite is verified again. | Keep; ablate against "regenerate with stricter prompt". |
| 25–30 schemes, PDFs with PyMuPDF/OCR | 15 schemes from the official myScheme JSON API | The portal is a JavaScript app; its public JSON API gives the editors' structured text without OCR errors, with a traceable raw response per scheme. | Expand to 25–30 schemes; add PDF/OCR ingestion for guideline documents not on myScheme. |
| 300–500 QA pairs with gold evidence passages, ≥30% Hindi/Hinglish | 50 questions (28% Hindi/Hinglish), scheme-level relevance, no gold passage ids | Built to make the harness runnable first. | Grow to 300+ with gold chunk ids, ≥30% Hindi/Hinglish, translation-paired questions. |
| Plain LLM vs baseline RAG vs SchemeSure | Baseline RAG (guardrail OFF) vs SchemeSure (guardrail ON) | Plain-LLM condition designed but not yet built. | Add the plain-LLM condition (RQ5). |
| LangChain or LlamaIndex | Plain Python | Every step is a few dozen lines; a framework would hide exactly the steps the project is about. | Keep. |
| multilingual-e5 or BGE-M3 embeddings | multilingual MiniLM | Measured: e5 compressed on- and off-topic scores into a narrow band (0.70 vs 0.81), leaving no room for a threshold; BGE-M3 (~2.2 GB) does not fit the free host. | Embedding ablation on the retrieval benchmark (RQ1). |
| Hugging Face Spaces / Render | Streamlit Community Cloud (Docker image still built and smoke-tested in CI) | HF Docker Spaces now require a paid plan; Render's 512 MB limit is below the measured ~1.2 GB footprint. | Keep; document cold-start behaviour. |

---

## 5. Evaluation protocol

### 5.1 Datasets

- **Golden set** (`eval/golden_set.jsonl`): 50 questions — 35 answerable across all 15
  schemes, 15 unanswerable split by failure mode (off_topic 5, fake_scheme 5,
  not_indexed 4, trick_unstated_fact 1). 36 English, 7 Hindi, 7 Hinglish. Every
  expected fact is checked to appear in the scheme's source file when the set is built
  (`scripts/build_golden_set.py`).
- **Known weaknesses** (from the evaluation review): Hindi/Hinglish questions are not
  translations of English ones, so per-language comparisons mix language with
  difficulty; per-language n is small (4–5 answerable); key facts are substring-matched
  in English; relevance is scheme-level, not passage-level; the 0.45 threshold was
  originally chosen on queries that overlap this set.

### 5.2 Metrics

| Metric | Definition | Denominator |
|---|---|---|
| Recall@k | expected scheme appears among the top-k retrieved chunks | answerable questions |
| MRR@10 | 1 / rank of the first chunk from the expected scheme | answerable questions |
| Over-block rate (threshold) | answerable questions with best score < t | answerable questions |
| Abstention recall (threshold) | unanswerable questions with best score < t, per failure mode | unanswerable questions of that type |
| Unsupported claim rate | claims in the **delivered** answer judged not supported by the retrieved evidence | all audited claims |
| Answers with any unsupported claim | delivered answers containing ≥1 unsupported claim | audited answers |
| Correct refusal rate | unanswerable questions refused | unanswerable questions |
| Over-blocking rate (end-to-end) | answerable questions refused | answerable questions |
| Key-fact coverage / end-to-end accuracy | expected facts present in the answer; end-to-end counts a refusal as 0 | answered / all answerable |
| Cost and latency | LLM calls, tokens, wall-clock per question | all questions |

Infrastructure failures (rate limits, quota, outages) are recorded separately and
excluded from these rates rather than counted as refusals. Proportions are reported with n. The retrieval benchmark (`eval/run_retrieval.py`) also gives a Wilson 95% interval; the end-to-end OFF-vs-ON report (`eval/run_eval.py`) does not yet, and adding intervals there is planned (E3). A metric with an empty denominator is reported as "n/a", never as 0%.

### 5.3 Controls

- Temperature 0 for every LLM call.
- The same normalised query (cached once per question) is used by every condition, so
  conditions see identical evidence.
- The delivered answer — not the draft — is audited, so the guardrail is not credited
  for errors it introduced while repairing.

### 5.4 Threats to validity

- **Judge = generator family.** The same model drafts and judges; a shared blind spot
  is invisible. Mitigation planned: a different judge model and a human-labelled subset
  (RQ4), following the judge-validation practice in ARES [5] and the judge-bias
  findings of [10].
- **Small n.** At 50 questions, differences of a few points are noise; per-language
  cells are 4–5 questions.
- **Author-written questions.** The set may under-represent failure modes the authors
  did not think of. The junior team member's planned role (collecting real Hindi and
  Hinglish questions) addresses this.
- **Snapshot data.** The corpus reflects official pages on the `last_verified` date.

---

## 6. Related work (verified references)

Each reference below was checked against the ACL Anthology, arXiv, proceedings pages
or Crossref. References [1]–[5] in the original research plan are retained as
[1], [2], [7], [8] and [26] here.

**Hallucination and faithfulness.** Huang et al. [1] and Ji et al. [2] survey
hallucination, separating intrinsic (contradicts the source) from extrinsic
(unverifiable) errors — SchemeSure's NOT_SUPPORTED verdict deliberately covers both.
TRUE [16] shows NLI- and QA-based consistency metrics are complementary; SummaC [17]
motivates checking claims against individual passages rather than whole documents;
AlignScore [18] is a compact non-LLM consistency scorer and a candidate ablation.

**Atomic claims and attribution.** FActScore [3] scores the fraction of atomic facts
supported by a knowledge source, the direct precedent for SchemeSure's unsupported-
claim rate. AIS [15] formalises "attributable to an identified source"; ALCE [13]
defines citation-support metrics that map onto the citation validation step.

**Retrieval-augmented generation.** Lewis et al. [6] introduced RAG; Fan et al. [7]
survey retrieval-augmented LLMs. Self-RAG [19] and CRAG [20] add critique or
retrieval-confidence evaluation inside the generation loop; SchemeSure instead uses an
external, auditable verification gate with no fine-tuning. Chain-of-Verification [9]
is the closest precedent for draft → verify → revise. Liu et al. [21] motivate a
small top-k with the strongest passages first.

**Evaluating RAG.** RAGAS [4] offers reference-free faithfulness metrics; ARES [5]
validates LLM judges against a small human-labelled set; Salemi and Zamani [8] show
that retrieval quality should be judged by downstream usefulness, which is why
SchemeSure measures retrieval and answer faithfulness separately. Zheng et al. [10]
document LLM-judge biases relevant to the limitations above. SelfCheckGPT [11] is a
contrasting zero-resource approach (self-consistency rather than evidence).

**Abstention.** Kamath et al. [14] study selective question answering under domain
shift and show raw confidence is overconfident — the motivation for calibrating the
retrieval threshold per failure mode instead of trusting one score.

**Multilingual retrieval and inference.** The embedding model follows Reimers and
Gurevych's multilingual distillation [12]. Multilingual E5 [24] and BGE-M3 [25] are
candidate embeddings for the ablation; MuRIL [27] targets Indian languages including
transliterated text. XNLI [23] and IndicXNLI [28] provide Hindi NLI data for testing
an NLI verifier; DeBERTa-v3 [22] is the usual backbone for NLI cross-encoders.
Pan et al. [26] argue for combining LLMs with structured knowledge — relevant to the
structured eligibility rules.

### References

[1] L. Huang et al., "A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions," *ACM Trans. Inf. Syst.*, 2025, doi: 10.1145/3703155.
[2] Z. Ji et al., "Survey of Hallucination in Natural Language Generation," *ACM Comput. Surv.*, vol. 55, no. 12, Art. 248, 2023, doi: 10.1145/3571730.
[3] S. Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," in *Proc. EMNLP*, 2023, pp. 12076–12100, doi: 10.18653/v1/2023.emnlp-main.741.
[4] S. Es, J. James, L. Espinosa Anke, and S. Schockaert, "RAGAs: Automated Evaluation of Retrieval Augmented Generation," in *Proc. EACL: System Demonstrations*, 2024, pp. 150–158, doi: 10.18653/v1/2024.eacl-demo.16.
[5] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems," in *Proc. NAACL*, 2024, pp. 338–354, doi: 10.18653/v1/2024.naacl-long.20.
[6] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Proc. NeurIPS*, vol. 33, 2020, arXiv:2005.11401.
[7] W. Fan et al., "A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models," in *Proc. 30th ACM SIGKDD*, 2024, pp. 6491–6501, doi: 10.1145/3637528.3671470.
[8] A. Salemi and H. Zamani, "Evaluating Retrieval Quality in Retrieval-Augmented Generation," in *Proc. 47th ACM SIGIR*, 2024, doi: 10.1145/3626772.3657957.
[9] S. Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in Large Language Models," in *Findings of ACL*, 2024, pp. 3563–3578, doi: 10.18653/v1/2024.findings-acl.212.
[10] L. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," in *Proc. NeurIPS Datasets and Benchmarks Track*, 2023, arXiv:2306.05685.
[11] P. Manakul, A. Liusie, and M. Gales, "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," in *Proc. EMNLP*, 2023, pp. 9004–9017, doi: 10.18653/v1/2023.emnlp-main.557.
[12] N. Reimers and I. Gurevych, "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation," in *Proc. EMNLP*, 2020, pp. 4512–4525, doi: 10.18653/v1/2020.emnlp-main.365.
[13] T. Gao, H. Yen, J. Yu, and D. Chen, "Enabling Large Language Models to Generate Text with Citations," in *Proc. EMNLP*, 2023, pp. 6465–6488, doi: 10.18653/v1/2023.emnlp-main.398.
[14] A. Kamath, R. Jia, and P. Liang, "Selective Question Answering under Domain Shift," in *Proc. ACL*, 2020, pp. 5684–5696, doi: 10.18653/v1/2020.acl-main.503.
[15] H. Rashkin et al., "Measuring Attribution in Natural Language Generation Models," *Comput. Linguist.*, vol. 49, no. 4, pp. 777–840, 2023, doi: 10.1162/coli_a_00486.
[16] O. Honovich et al., "TRUE: Re-evaluating Factual Consistency Evaluation," in *Proc. NAACL*, 2022, pp. 3905–3920, doi: 10.18653/v1/2022.naacl-main.287.
[17] P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization," *Trans. Assoc. Comput. Linguist.*, vol. 10, pp. 163–177, 2022, doi: 10.1162/tacl_a_00453.
[18] Y. Zha, Y. Yang, R. Li, and Z. Hu, "AlignScore: Evaluating Factual Consistency with a Unified Alignment Function," in *Proc. ACL*, 2023, pp. 11328–11348, doi: 10.18653/v1/2023.acl-long.634.
[19] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection," in *Proc. ICLR*, 2024, arXiv:2310.11511.
[20] S.-Q. Yan, J.-C. Gu, Y. Zhu, and Z.-H. Ling, "Corrective Retrieval Augmented Generation," arXiv:2401.15884, 2024.
[21] N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," *Trans. Assoc. Comput. Linguist.*, vol. 12, pp. 157–173, 2024, doi: 10.1162/tacl_a_00638.
[22] P. He, J. Gao, and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," in *Proc. ICLR*, 2023, arXiv:2111.09543.
[23] A. Conneau et al., "XNLI: Evaluating Cross-lingual Sentence Representations," in *Proc. EMNLP*, 2018, pp. 2475–2485, doi: 10.18653/v1/D18-1269.
[24] L. Wang et al., "Multilingual E5 Text Embeddings: A Technical Report," arXiv:2402.05672, 2024.
[25] J. Chen et al., "BGE M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation," in *Findings of ACL*, 2024, pp. 2318–2335, arXiv:2402.03216.
[26] S. Pan et al., "Unifying Large Language Models and Knowledge Graphs: A Roadmap," *IEEE Trans. Knowl. Data Eng.*, vol. 36, no. 7, pp. 3580–3599, 2024, doi: 10.1109/TKDE.2024.3352100.
[27] S. Khanuja et al., "MuRIL: Multilingual Representations for Indian Languages," arXiv:2103.10730, 2021.
[28] D. Aggarwal, V. Gupta, and A. Kunchukuttan, "IndicXNLI: Evaluating Multilingual Inference for Indian Languages," in *Proc. EMNLP*, 2022, pp. 10994–11006, doi: 10.18653/v1/2022.emnlp-main.755.

> Before submitting a paper, re-check volume/issue/page details for [1] and [8] on the
> publisher pages; the reference checker could confirm their DOIs but not every detail.

---

## 7. Planned experiments (second half)

| ID | Experiment | Design | Answers |
|---|---|---|---|
| E1 | Per-language retrieval | raw vs normalised queries, Recall@1/3/5 + MRR per language, on the expanded set with translation-paired questions | RQ1 |
| E2 | Threshold calibration | choose the evidence threshold on a dev split, report abstention precision/recall per failure mode on a held-out split | RQ2 |
| E3 | Guardrail OFF vs ON | full golden set, then the 300+ set; unsupported-claim rate, over-blocking, correct refusal, cost, latency, with Wilson intervals | RQ3 |
| E4 | Plain LLM baseline | same questions, no retrieval; audited against the same evidence | RQ5 |
| E5 | Verifier comparison | ~200 human-labelled (claim, passage) pairs; LLM judge vs NLI cross-encoder vs embedding similarity vs hybrid; agreement (Cohen's κ), precision/recall of NOT_SUPPORTED | RQ4 |
| E6 | Cross-model judge | a different model family judges the same claims; disagreement analysis | RQ4, validity |
| E7 | Embedding ablation | MiniLM vs multilingual-e5 vs BGE-M3 (offline, retrieval-only) | RQ1 |
| E8 | Policy ablation | block threshold 0.3/0.5/0.7; top-k 3/5/8; repair vs regenerate | RQ3 |
| E9 | Robustness | prompt-injection questions ("ignore the evidence…"), number-perturbation tests. **Number perturbation done without the LLM** (`eval/runs/numeric_perturbation/results.md`): numeric rule 100% on wrong numbers absent from evidence, 0% false alarms over Indian number formats, 0% on in-evidence swaps (by design); similarity AUROC 0.53. LLM-judge run on the same items pending | RQ3 |
| E10 | User study | small task-based study with Hindi/Hinglish speakers on trust and comprehension of the verification view | usability |

The paper built on RQ2–RQ4 is planned in [`PAPER_PLAN.md`](PAPER_PLAN.md).

## 8. Contribution (stated honestly)

No model is trained. The contribution is an **auditable verification gateway** for a
high-stakes, multilingual domain — claim-level judging plus deterministic Python
checks (citation existence, numeric/link grounding, decision policy) with honest
abstention — and its **evaluation on Indian scheme data across English, Hindi and
Hinglish**, including failure modes (non-indexed and fake schemes) that a retrieval
threshold alone cannot catch.
