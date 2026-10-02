# SchemeSure — Project Brief (verbatim build spec)

> This file stores the original build instructions so any future session (human or
> Claude) has the full context. Treat it as the source of truth for *intent*;
> `docs/DECISIONS.md` records where reality forced a different choice.

## Quick orientation for a new session

- **What it is:** a hallucination-aware RAG assistant for Indian government schemes.
- **Languages:** English, Hindi, Hinglish.
- **Non-negotiable:** LLM for language, Python for rules. Every answer is verified
  claim-by-claim against retrieved evidence before the user sees it.
- **Run tests:** `./venv/Scripts/python.exe -m pytest -q`
- **Run API:** `./venv/Scripts/python.exe -m uvicorn app.main:app --reload`
- **Run UI:** `./venv/Scripts/python.exe -m streamlit run ui/streamlit_app.py`

---

## ORIGINAL PROMPT (do not edit)

You are going to build, test, push to GitHub, and deploy a complete project for me with
minimal interaction. I am a fresher preparing for AI/GenAI job interviews. I have never
used Docker. Work autonomously through all phases below. Only stop and ask me when you
are truly blocked (missing credentials, a failed login, or a decision only I can make).
For every other choice, pick the simplest reasonable option, log it in docs/DECISIONS.md,
and keep going.

=====================================================================
PROJECT: SchemeSure
A hallucination-aware RAG assistant for Indian government scheme information.
Answers in English, Hindi and Hinglish. Every answer is checked against retrieved
evidence before reaching the user. Eligibility is decided by rule-based Python code.
=====================================================================

STACK (do not add anything else):
- Python 3.11, FastAPI, Pydantic, pydantic-settings
- ChromaDB (persistent), sentence-transformers multilingual embedding model (free, local)
- LLM via the `openai` Python client pointed at Groq's OpenAI-compatible endpoint.
  Configurable via .env: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
  Check Groq's current model list and pick a fast, cheap, capable model.
- Streamlit frontend that calls FastAPI over HTTP
- pytest, Docker (deployed to a Hugging Face Docker Space), GitHub Actions CI/CD

CORE DESIGN PRINCIPLES (follow strictly):
1. LLM for language, code for rules. Eligibility is plain Python, never an LLM decision.
2. Never trust LLM output blindly. Every answer is verified claim-by-claim against evidence.
3. Weak evidence -> honest refusal ("I do not have official information on this").
4. Every answer shows citations (scheme, section, official URL).
5. Log every LLM call (prompt name, model, tokens, latency) to SQLite for monitoring.
6. Simple, readable, well-commented code. I must be able to explain every file in an interview.

SECURITY RULES (non-negotiable):
- Secrets live only in .env. .env must be in .gitignore BEFORE the first commit.
- Before every git push, scan staged files for API keys/tokens. Abort if any found.
- Never print my secrets in your output.

FOLDER STRUCTURE:
schemesure/
  app/         main.py, config.py, schemas.py, llm_client.py, prompts.py
  core/        ingest.py, query_normalizer.py, retriever.py, generator.py, guardrail.py, eligibility.py
  data/schemes/  one .md per scheme + one .rules.json per scheme
  eval/        golden_set.jsonl, run_eval.py, results.md
  ui/          streamlit_app.py
  tests/
  scripts/     validate_data.py, deploy_hf.py
  docs/        DECISIONS.md, INTERVIEW_NOTES.md, LEARN.md, DEPLOYMENT.md, VERIFY_CHECKLIST.md
  Dockerfile, start.sh, requirements.txt, .env.example, .gitignore, README.md, CLAUDE.md
  .github/workflows/ci-deploy.yml

---------------------------------------------------------------------
PHASE 0: Setup
- Save this whole prompt into CLAUDE.md so you remember it.
- Check that python, git and gh are installed and that `gh auth status` works.
  If not, tell me exactly what to install/run, then wait.
- Create the venv, folder structure, requirements.txt (pinned versions), .gitignore
  (include .env, venv, chroma data, __pycache__, *.db).
- Create .env with empty values for LLM_API_KEY and HF_TOKEN plus sensible defaults for
  everything else. Then STOP and ask me: "Fill LLM_API_KEY and HF_TOKEN in .env and reply done."
- After I reply, verify the LLM key with one tiny test call.
- git init, first commit.

PHASE 1: Scheme data (from official sources only)
- Collect data for 15 popular central government schemes (e.g. PM-KISAN, Ayushman Bharat PM-JAY,
  PMAY, Sukanya Samriddhi Yojana, PM Ujjwala, Atal Pension Yojana, PM Mudra, PMJJBY, PMSBY,
  Stand-Up India, PM SVANidhi, National Scholarship schemes, etc.).
- Use ONLY official sources (*.gov.in, *.nic.in, myscheme.gov.in). If a page cannot be fetched,
  try another official page; if none works, skip that scheme and pick another.
- One markdown file per scheme with front-matter (scheme_id, name, ministry, source_url,
  last_verified=today) and sections: Overview, Benefits, Eligibility, Documents Required, How to Apply.
- One .rules.json per scheme: min_age, max_age, max_annual_income, allowed_states,
  allowed_occupations, gender, category (null = no restriction).
- STRICT: every fact must come from a page you actually fetched. Never fill gaps from memory.
  If something is not stated officially, write "Not specified in official source".
- Write scripts/validate_data.py and run it.
- Create docs/VERIFY_CHECKLIST.md: per scheme, the source URL and the 3 most important facts,
  so I can spot-check them later. Commit.

PHASE 2: Ingestion + retrieval
- core/ingest.py: section-aware chunking (split by headings, not fixed size), metadata
  (scheme_id, scheme_name, section, source_url, last_verified), deterministic IDs so
  re-running does not duplicate. Embed + store in Chroma.
- core/query_normalizer.py: one LLM call -> JSON {normalized_query (English), language}.
- core/retriever.py: top-k search with scores; low_evidence=True if best score < threshold.
- Test with ~10 queries (English, Hindi, Hinglish, out-of-scope), look at real scores,
  set the threshold, and record the reasoning in DECISIONS.md. Commit.

PHASE 3: Generation + guardrail (the heart of the project)
- core/generator.py: low_evidence -> refusal with no LLM call. Otherwise answer ONLY from chunks
  with [n] citations, in the user's language. Prompts live in app/prompts.py.
- core/guardrail.py:
  1. LLM splits the answer into atomic claims (JSON).
  2. LLM judge checks each claim against ONLY the retrieved chunks ->
     {claim, verdict: SUPPORTED|NOT_SUPPORTED, evidence_chunk_id}.
  3. Python verifies evidence_chunk_id really exists (reject fake citations in code).
  4. Python decision policy: all supported -> PASS; some unsupported -> REPAIR (regenerate from
     supported claims only); >50% unsupported (configurable) -> BLOCK with safe refusal.
  5. Return a GuardrailReport. Add a GUARDRAIL_ENABLED flag for evals.
- Unit tests for the decision policy with fake verdicts (no LLM). Commit.

PHASE 4: Eligibility engine
- LLM extracts a user profile from free text into a Pydantic model (missing fields = None).
- Pure Python check per scheme -> ELIGIBLE / NOT_ELIGIBLE / NEED_MORE_INFO with a reason per rule.
- Thorough unit tests for edge cases (boundary values, null rules, missing fields). Commit.

PHASE 5: FastAPI
- POST /ask, POST /eligibility, GET /schemes, GET /health, GET /metrics
  (LLM call count, avg latency, guardrail PASS/REPAIR/BLOCK counts).
- Pydantic request/response models, input length limit, error handling, CORS.
- Load the embedding model and Chroma once at startup.
- Integration tests with the LLM mocked. Run the full test suite. Commit.

PHASE 6: Streamlit UI
- Tab "Ask": chat, clickable citations, badge (Verified / Partially verified / Not enough
  official info), expander "How was this verified?" showing each claim + verdict.
- Tab "Check eligibility": text box -> table of schemes, status, reasons.
- Tab "Schemes": covered schemes with source links and last_verified dates.
- Disclaimer footer. No business logic in the UI. API URL comes from an env variable.
- Run the backend + UI locally, test the full flow yourself, fix bugs. Commit.

PHASE 7: Evaluation
- Generate eval/golden_set.jsonl with 50 questions from the scheme data: ~35 answerable
  (mix of English/Hindi/Hinglish, with expected_scheme and key_facts) and ~15 unanswerable
  (fake schemes, out-of-scope, trick questions). Facts must match the data files.
- eval/run_eval.py runs each question with guardrail OFF vs ON and computes: retrieval hit rate,
  unsupported claim rate, correct refusal rate, over-blocking rate, avg latency, LLM calls/question.
  Respect Groq rate limits (add delays/retries).
- Run it and write the comparison table to eval/results.md. Commit.

PHASE 8: GitHub
- Create a PUBLIC GitHub repo named "schemesure" with `gh repo create`, with a good description
  and topics (rag, llm, genai, fastapi, hallucination-detection, vector-database, python).
- Run the secret scan, then push. Keep the commit history clean and meaningful.

PHASE 9: Deployment (Hugging Face Docker Space, single container)
- Dockerfile: install deps, copy code, build the Chroma index AT BUILD TIME from data/schemes
  (dataset is tiny, so no persistent disk is needed). Pre-download the embedding model at build time.
- start.sh: start FastAPI with uvicorn on port 8000 in the background, then Streamlit on port
  7860 (the port HF exposes), with Streamlit calling http://localhost:8000.
- Add the Hugging Face Space front-matter to the README if the Space needs it.
- scripts/deploy_hf.py using the huggingface_hub Python library and HF_TOKEN from .env:
  create the Space (sdk=docker) if it does not exist, upload the repo files (excluding .env,
  venv, local DBs), set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL as Space SECRETS via the API
  (never in files), then poll the build status until it is running or fails.
- If the build fails, read the logs, fix, redeploy. Repeat until it works.
- Verify the live app loads and a test question returns a verified answer.
- Check the current Hugging Face free tier limits. If they make this impossible, tell me and
  propose the simplest alternative.

PHASE 10: CI/CD
- .github/workflows/ci-deploy.yml: on every push to main, run pytest (LLM mocked), and if tests
  pass, run deploy_hf.py to redeploy.
- Set HF_TOKEN and the LLM secrets as GitHub repo secrets using `gh secret set` (read values from
  .env without printing them). Push and confirm the workflow passes.

PHASE 11: Documentation (for my interviews)
- README.md: live demo link, problem, Mermaid architecture diagram, how it works step by step,
  eval results table, tech stack, local setup, limitations, future work.
- docs/DECISIONS.md as a table: Decision | Why | Alternatives | Trade-off.
- docs/LEARN.md: a beginner-friendly walkthrough of the whole codebase in the order a request
  flows through it, explaining every file in simple words, plus a plain explanation of Docker,
  CI/CD and how the deployment works. Use analogies and simple tricks to remember.
- docs/INTERVIEW_NOTES.md: 2-minute pitch; 20 likely follow-up questions with short honest
  answers (hallucination detection, over-blocking, stale data, scaling to 1000 schemes, cost,
  why not fine-tune, LLM-as-judge limits, why Chroma, why Docker, CI/CD, improvements);
  "Problems I faced and how I fixed them", filled from the real problems you hit while building.
- 3 ATS-friendly resume bullets using the REAL numbers from eval/results.md.
- docs/DEPLOYMENT.md: how the deployment works and how to redeploy.
- Final commit and push (which also triggers the CI/CD redeploy).

FINAL REPORT (send me at the end):
- Live app URL and GitHub repo URL
- Eval results table
- Resume bullets
- Anything I must do manually (e.g. spot-check VERIFY_CHECKLIST.md)
- A recommended order for me to read the files to understand the project
