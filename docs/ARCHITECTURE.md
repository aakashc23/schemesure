# SchemeSure architecture

This document describes the system **as implemented** on the `midterm-improvements`
branch (November 2026). Planned components are marked *planned* and drawn with dashed
lines; nothing here describes behaviour that the code does not have.

Contents:

1. [System context](#1-system-context)
2. [Module map](#2-module-map)
3. [Request flow: `POST /ask`](#3-request-flow-post-ask)
4. [The guardrail decision](#4-the-guardrail-decision)
5. [Eligibility flow](#5-eligibility-flow)
6. [Data pipeline](#6-data-pipeline)
7. [Deployment and CI](#7-deployment-and-ci)
8. [Extension points](#8-extension-points)
9. [Design invariants](#9-design-invariants)

---

## 1. System context

```mermaid
flowchart LR
    U[Citizen / evaluator<br/>browser] -->|HTTPS| UI[Streamlit UI<br/>ui/streamlit_app.py]
    UI -->|HTTP JSON<br/>127.0.0.1:8000| API[FastAPI backend<br/>app/main.py]
    API --> IDX[(ChromaDB index<br/>312 passages, 15 schemes)]
    API --> EMB[Embedding model<br/>multilingual MiniLM, local CPU]
    API -->|OpenAI-compatible API| LLM[Groq<br/>qwen/qwen3.8-27b]
    API --> LOG[(SQLite<br/>llm_calls.db)]
    SRC[myscheme.gov.in<br/>official pages] -.->|fetched once,<br/>scripts/fetch_schemes.py| DATA[data/schemes/*.md<br/>*.rules.json]
    DATA -->|core/ingest.py| IDX
```

The UI holds **no business logic**. It renders what the API returns. Every decision
about what the user sees is made in Python on the backend.

## 2. Module map

```mermaid
flowchart TB
    subgraph app [app/ — web layer and shared plumbing]
        main[main.py<br/>endpoints, startup, error handlers]
        config[config.py<br/>Settings from .env]
        schemas[schemas.py<br/>Pydantic request/response models]
        llm[llm_client.py<br/>LLM calls, retry policy, call log, metrics]
        prompts[prompts.py<br/>every prompt + canned refusals]
    end
    subgraph core [core/ — the actual logic]
        ingest[ingest.py<br/>chunk, embed, index]
        norm[query_normalizer.py<br/>LLM: language + English query]
        ret[retriever.py<br/>top-k search, low_evidence flag]
        gen[generator.py<br/>AnswerPipeline: draft, act on verdict, repair]
        guard[guardrail.py<br/>claims, judge, citation check,<br/>number/link grounding, decide]
        elig[eligibility.py<br/>profile extraction + rule checks]
        cache[answer_cache.py<br/>TTL + LRU answer cache]
    end
    subgraph evalh [eval/ — measurement]
        rr[run_retrieval.py]
        re[run_eval.py]
        common[common.py / retrieval_metrics.py / norm_cache.py]
    end
    main --> gen & elig & cache & ret
    gen --> norm & ret & guard & llm
    guard --> llm
    elig --> llm
    norm --> llm
    llm --> prompts
    ret --> ingest
    rr --> ret & norm
    re --> gen & guard
```

| File | One-line job | LLM? |
|---|---|---|
| `app/main.py` | Endpoints; loads the model and index once at startup; maps errors to 400/404/502/503 | no |
| `app/llm_client.py` | The only place that talks to the LLM: retries, error classification, per-request deadline, SQLite log | calls it |
| `app/prompts.py` | All prompt text and the canned refusal messages (en / hi / hinglish) | — |
| `core/query_normalizer.py` | Detect language, rewrite the question as an English search query | 1 call |
| `core/retriever.py` | Cosine search in Chroma; `low_evidence` when best score < 0.45 | no |
| `core/generator.py` | `AnswerPipeline.ask`: orchestrates the whole request and decides what is shown | 1 call (+1 on repair) |
| `core/guardrail.py` | Split into claims, judge each claim, then Python checks and the decision | 2 calls (+2 on repair) |
| `core/eligibility.py` | Extract a profile (LLM) or accept a form (no LLM); apply rules in Python | 0–1 call |
| `core/answer_cache.py` | Serve a repeated question without new LLM calls (API layer only) | no |
| `ui/streamlit_app.py` | Five tabs: Ask, Eligibility, Schemes, How it works, System | no |
| `streamlit_app.py` | Streamlit Community Cloud entry point: starts the API in a thread, then runs the UI | no |

## 3. Request flow: `POST /ask`

```mermaid
sequenceDiagram
    autonumber
    participant UI as Streamlit UI
    participant API as FastAPI /ask
    participant C as AnswerCache
    participant P as AnswerPipeline
    participant N as Normalizer (LLM)
    participant R as Retriever (Chroma)
    participant G as Generator (LLM)
    participant V as Guardrail

    UI->>API: {question, language?}
    API->>API: length check (≤ 500 chars, else 400)
    API->>C: get(question, language, guardrail flag)
    alt cache hit
        C-->>API: stored response
        API-->>UI: response, cached=true, 0 LLM calls
    else cache miss
        API->>P: ask(question)
        P->>N: normalise
        N-->>P: English query + language (falls back to raw text on failure)
        P->>R: search(top_k=5)
        R-->>P: passages + scores, low_evidence?
        alt best score < 0.45
            P-->>API: refusal (LOW_EVIDENCE, or SERVICE_UNAVAILABLE if the normaliser fell back), no generation
        else enough evidence
            P->>G: draft answer from passages only, [n] citations
            G-->>P: draft
            P->>V: verify(draft, passages)
            V-->>P: GuardrailReport (PASS / REPAIR / BLOCK)
            opt REPAIR
                P->>G: rebuild from supported claims only
                P->>V: verify(repaired) — full second check
            end
            P-->>API: AskResponse (answer or refusal + reason, citations,<br/>claims, evidence table, stage trace)
        end
        API->>C: put(response) unless service_unavailable / fallback
        API-->>UI: AskResponse
    end
```

LLM calls per question: **1** for a low-evidence refusal, **4** for a normal answer
(normalise, generate, split, judge), **7** when a repair happens (repair, split, judge
again). A cache hit costs **0**.

Every response carries a `trace` (stage, duration, LLM calls, outcome), so the UI's
**Pipeline** view shows what actually ran rather than a generic diagram.

## 4. The guardrail decision

```mermaid
flowchart TD
    A[Draft answer] --> B[LLM: split into atomic claims]
    B -->|LLM error| X1[[verification_error<br/>→ refuse: SERVICE_UNAVAILABLE]]
    B --> C[LLM judge: each claim vs the retrieved passages only<br/>SUPPORTED / NOT_SUPPORTED + evidence chunk id]
    C -->|LLM error| X1
    C --> D[Python: does the cited chunk id exist<br/>in the retrieved set? If not → NOT_SUPPORTED]
    D --> E[Python: similarity claim↔evidence<br/>recorded only, never used to decide]
    E --> F{decide — pure function}
    F -->|no claims| P0[PASS with 0 claims<br/>→ refuse: NO_CLAIMS]
    F -->|all supported| P[PASS]
    F -->|none supported| BL[BLOCK → refuse: NOT_SUPPORTED]
    F -->|unsupported ratio > 0.5| BL
    F -->|some unsupported| RP[REPAIR]
    P --> G{Python: every number ≥ 10, %, decimal,<br/>lakh/crore amount and link<br/>appears in the passages?<br/>≤ 20 claims?}
    G -->|yes| OK[Show answer — Verified]
    G -->|no| RP
    RP --> H[LLM: rebuild from supported claims only]
    H --> I[Full guardrail again on the rebuilt text]
    I -->|PASS with ≥ 1 claim| PV[Show answer — Partially verified]
    I -->|anything else| BL
```

What each layer is responsible for:

| Check | Done by | Can it be fooled? |
|---|---|---|
| Is there enough evidence to try? | Python threshold on retrieval score | Yes, by on-topic but unindexed questions (see eval results) |
| Is each claim supported? | LLM judge, sees only the passages | Yes; same model family as the generator. Human labels are planned (Roadmap P0-5) |
| Does the cited evidence exist? | Python | No — fabricated ids are rejected in code |
| Do numbers and links appear in the evidence? | Python regex + normalisation | Narrowly: numbers below 10 are not checked |
| Is the final decision consistent? | Python `decide()` (unit-tested) | No — it is a pure function |

Refusal reasons shown to the user are distinct: `low_evidence` ("no official
information"), `not_supported` (the draft failed verification), `no_claims` (the draft
asserted nothing checkable) and `service_unavailable` (the LLM failed; we do **not**
claim the information is missing).

## 5. Eligibility flow

```mermaid
flowchart LR
    T[Free text<br/>'I am a 35 year old kisan<br/>from Bihar earning 1.5 lakh'] -->|POST /eligibility| X[LLM: extract UserProfile<br/>missing → null]
    F[Form fields] -->|POST /eligibility/check<br/>no LLM| NP
    X --> NP[Python: normalise<br/>kisan→farmer, '1.5 lakh'→150000,<br/>unknown occupation→null]
    NP --> R{For each of 15 schemes:<br/>check every rule}
    R --> E1[ELIGIBLE — all modelled rules pass]
    R --> E2[NOT_ELIGIBLE — a rule fails<br/>with the official sentence]
    R --> E3[NEED_MORE_INFO — a needed field is missing<br/>or the occupation is too broad]
```

The LLM never decides eligibility. It only fills a form. Conditions that are not
machine-checkable (e.g. "not an income-tax payer") are listed per scheme as
`unmodelled_conditions` so the user knows what to check themselves.

## 6. Data pipeline

```mermaid
flowchart LR
    A[myscheme.gov.in API<br/>scripts/fetch_schemes.py] --> B[data/raw/*.json<br/>raw responses kept]
    B --> C[data/schemes/*.md<br/>front-matter + 6 sections]
    C --> D[scripts/build_rules.py<br/>hand-written translation table<br/>→ data/schemes/*.rules.json with rule_sources]
    C & D --> E[scripts/validate_data.py<br/>schema + provenance checks]
    E --> F[core/ingest.py<br/>split by heading, ≤110 tokens,<br/>deterministic ids scheme#section#n]
    F --> G[(Chroma, cosine)]
    C --> H[scripts/make_verify_checklist.py<br/>docs/VERIFY_CHECKLIST.md]
```

Deterministic chunk ids mean re-running ingestion never duplicates passages, and the
evaluation can refer to a passage by id.

## 7. Deployment and CI

```mermaid
flowchart TB
    dev[Developer push to main] --> gh[GitHub]
    gh --> ci[GitHub Actions ci-deploy.yml]
    ci --> t1[validate_data → build index → pytest<br/>LLM mocked, no key] --> t2[secret scan of tracked files]
    ci --> t3[parallel job: docker build + smoke test<br/>Streamlit health, /health, /schemes]
    t3 -.->|manual dispatch only| hf[Hugging Face Docker Space]
    gh -->|auto redeploy on push| scc[Streamlit Community Cloud<br/>LIVE DEMO]
    subgraph scc_proc [One Python process on Streamlit Cloud]
        s1[streamlit_app.py] -->|thread, cached once| s2[uvicorn app.main:app<br/>127.0.0.1:8000]
        s1 -->|runpy| s3[ui/streamlit_app.py]
        s3 -->|HTTP| s2
    end
    scc --> scc_proc
```

* **Live demo:** Streamlit Community Cloud runs the root `streamlit_app.py`, which
  starts FastAPI in a background thread and then runs the UI. Secrets are set in the
  Streamlit Cloud dashboard. The app sleeps when idle; the first visit after sleep
  rebuilds the index (cold start).
* **Container:** `Dockerfile` + `start.sh` build the index and download the model at
  build time and run the API and UI as two processes. CI builds and smoke-tests this image on every push to main and on every pull request to main. Deployment to a Hugging Face Space is manual
  (`scripts/deploy_hf.py`).

## 8. Extension points

These are the seams the second-half work plugs into (see `ROADMAP.md`):

| Planned work | Where it attaches | What changes |
|---|---|---|
| NLI verifier (*planned*) | `Guardrail(..., embed_fn=...)` already accepts an optional signal function; an `nli_fn` would sit beside it | A new per-claim field, recorded first; routing only after it is validated |
| Cross-model judge (*planned*) | `judge_claims(llm, ...)` takes the client as an argument | A second `LLMClient` with a different model |
| Larger corpus (*planned*) | `data/schemes/` + `validate_data.py` + `ingest.py` | More files; no code change for Markdown sources |
| Metadata pre-filter (*planned*) | `Retriever.search` | A `where={"scheme_id": ...}` filter in the Chroma query |
| Conditional eligibility rules (*planned*) | `check_scheme` | New rule types; unmodelled conditions move into checked rules |

## 9. Design invariants

These hold everywhere in the code and are covered by tests:

1. **The LLM never decides eligibility or the guardrail outcome.** Both are plain Python
   (`check_scheme`, `decide`).
2. **Fail closed.** If verification cannot run, the draft is withheld and the user is told
   the service failed. An unverified answer is never labelled verified.
3. **Refusals are honest about their cause.** A provider outage is never reported as
   "no official information".
4. **With the guardrail on, the citations shown are the passages that verified a claim**, never "everything retrieved". (With the guardrail off, the evaluation baseline, the answer's own [n] markers are used, or every retrieved passage if it has none.)
5. **Every LLM call is logged** (prompt name, model, tokens, latency, attempts, ok) to
   SQLite, and `/metrics` reads from that log.
6. **Secrets live only in `.env` / platform secrets.** Provider error text is logged
   server-side only and never returned to the browser.
