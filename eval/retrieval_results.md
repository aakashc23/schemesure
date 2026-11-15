# Retrieval benchmark

Does the retriever put the right scheme in the top results? Measured with no answer generation and no verification, so only the embedding model and the index are being tested.

## Setup

- **Date:** 2026-11-16
- **Questions:** n=50 (golden set has 50)
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, cosine similarity
- **Depth stored:** top 10 chunks per question
- **Relevance:** scheme-level. A hit means the expected scheme owns at least one of the top-k chunks (the golden set has no gold chunk ids).
- **Intervals:** 95% Wilson score intervals in brackets.

## Variant `raw` (question as typed, 0 LLM calls)

| Language | n answerable | Recall@1 | Recall@3 | Recall@5 | MRR@10 | MRR@10 (distinct schemes) | Language-ID accuracy |
|---|---|---|---|---|---|---|---|
| **all** | 35 | 74.3% [57.9-85.8] | 91.4% [77.6-97.0] | 94.3% [81.4-98.4] | 0.806 | 0.819 | 100.0% [92.9-100.0] |
| en | 26 | 80.8% [62.1-91.5] | 96.2% [81.1-99.3] | 100.0% [87.1-100.0] | 0.867 | 0.878 | 100.0% [90.4-100.0] |
| hi | 4 | 50.0% [15.0-85.0] | 100.0% [51.0-100.0] | 100.0% [51.0-100.0] | 0.667 | 0.708 | 100.0% [64.6-100.0] |
| hinglish | 5 | 60.0% [23.1-88.2] | 60.0% [23.1-88.2] | 60.0% [23.1-88.2] | 0.600 | 0.600 | 100.0% [64.6-100.0] |

*Language-ID for `raw` is the offline script/word heuristic (`detect_language_heuristic`), not an LLM.*

## Variant `normalized` (English rewrite from the LLM, as in production)

| Language | n answerable | Recall@1 | Recall@3 | Recall@5 | MRR@10 | MRR@10 (distinct schemes) | Language-ID accuracy |
|---|---|---|---|---|---|---|---|
| **all** | 35 | 80.0% [64.1-90.0] | 91.4% [77.6-97.0] | 97.1% [85.5-99.5] | 0.851 | 0.862 | 100.0% [92.9-100.0] |
| en | 26 | 76.9% [57.9-89.0] | 92.3% [75.9-97.9] | 100.0% [87.1-100.0] | 0.838 | 0.853 | 100.0% [90.4-100.0] |
| hi | 4 | 100.0% [51.0-100.0] | 100.0% [51.0-100.0] | 100.0% [51.0-100.0] | 1.000 | 1.000 | 100.0% [64.6-100.0] |
| hinglish | 5 | 80.0% [37.6-96.4] | 80.0% [37.6-96.4] | 80.0% [37.6-96.4] | 0.800 | 0.800 | 100.0% [64.6-100.0] |

*Normalizer fallbacks (LLM failed, raw question used): 0.*

## How to read this

- Hindi and Hinglish questions in the golden set are **not translations** of the English ones, so a per-language difference mixes language with question difficulty. Treat it as a hint, not a controlled comparison.
- Per-language n is small (see the n column); the Wilson intervals are wide on purpose. A single question moves a small-group percentage a lot.
- Recall@k only says the right *scheme* was retrieved, not that the retrieved chunk contains the answer. End-to-end accuracy is measured by `eval/run_eval.py`.

---

Regenerate with `python eval/run_retrieval.py`. Full ranked lists: `retrieval_raw.json`.
