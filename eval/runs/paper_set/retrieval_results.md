# Retrieval benchmark

Does the retriever put the right scheme in the top results? Measured with no answer generation and no verification, so only the embedding model and the index are being tested.

## Setup

- **Date:** 2026-11-25
- **Questions:** n=279 (`paper_set.jsonl` has 279)
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, cosine similarity
- **Depth stored:** top 10 chunks per question
- **Relevance:** scheme-level. A hit means the expected scheme owns at least one of the top-k chunks (the golden set has no gold chunk ids).
- **Intervals:** 95% Wilson score intervals in brackets.

## Variant `raw` (question as typed, 0 LLM calls)

| Language | n answerable | Recall@1 | Recall@3 | Recall@5 | MRR@10 | MRR@10 (distinct schemes) | Language-ID accuracy |
|---|---|---|---|---|---|---|---|
| **all** | 105 | 72.4% [63.2-80.0] | 87.6% [80.0-92.6] | 92.4% [85.7-96.1] | 0.797 | 0.809 | 95.0% [91.8-97.0] |
| en | 35 | 82.9% [67.3-91.9] | 97.1% [85.5-99.5] | 100.0% [90.1-100.0] | 0.887 | 0.895 | 100.0% [96.0-100.0] |
| hi | 35 | 60.0% [43.6-74.4] | 82.9% [67.3-91.9] | 88.6% [74.0-95.5] | 0.710 | 0.732 | 100.0% [96.0-100.0] |
| hinglish | 35 | 74.3% [57.9-85.8] | 82.9% [67.3-91.9] | 88.6% [74.0-95.5] | 0.796 | 0.801 | 84.9% [76.3-90.8] |

*Language-ID for `raw` is the offline script/word heuristic (`detect_language_heuristic`), not an LLM.*

> The `normalized` variant has **not been run** (it needs the LLM, one call per question, cached afterwards). Until it runs, the effect of the normalizer is unmeasured and no comparison is claimed.

## How to read this

- Every question is asked in English, Hindi and Hinglish (same `pair_id`), so per-language differences hold the content fixed. The translations are author-written and not yet reviewed by a second native speaker.
- Per-language n is small (see the n column); the Wilson intervals are wide on purpose. A single question moves a small-group percentage a lot.
- Recall@k only says the right *scheme* was retrieved, not that the retrieved chunk contains the answer. End-to-end accuracy is measured by `eval/run_eval.py`.

---

Regenerate with `python eval/run_retrieval.py`. Full ranked lists: `retrieval_raw.json`.
