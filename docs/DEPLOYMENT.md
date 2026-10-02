# Deployment

How SchemeSure gets from this repository onto the internet, and how to redeploy it.

---

## Where it runs, and why not where it was meant to

The project was built for a **Hugging Face Docker Space**. The `Dockerfile`,
`start.sh` and `scripts/deploy_hf.py` were all written for that and all still
work. Partway through, actually deploying it returned:

```
402 Payment Required
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces
on free cpu-basic requires a PRO subscription.
```

Probing every SDK confirmed the situation:

| Space SDK | Free tier |
|---|---|
| `static` | ✅ available — but cannot run Python |
| `gradio` | ❌ requires PRO |
| `docker` | ❌ requires PRO |
| `streamlit` | ❌ no longer exists (the API accepts only `gradio`, `docker`, `static`) |

So **no free Hugging Face Space can host this app**. The live demo runs on
**Streamlit Community Cloud** instead: free, no payment method, and it redeploys
itself on every push to `main`.

Other free tiers were ruled out by a measurement: the app uses **~1.2 GB RSS**,
almost all of it torch. That is above the 512 MB ceiling on Render's and Fly.io's
free tiers.

---

## The two deployment shapes

The important thing is that moving hosts did **not** flatten the architecture.

### Shape A — the container (production path)

```
┌─────────────────── one Docker container ───────────────────┐
│                                                            │
│   Streamlit  :7860  ──HTTP──▶  FastAPI  :8000              │
│   (published)                  (127.0.0.1 only)            │
│                                     │                      │
│                                     ▼                      │
│                        ChromaDB + embedding model          │
│                        (both baked in at build time)       │
└────────────────────────────────────────────────────────────┘
```

Two processes supervised by `start.sh`. This is what `docker run` gives you, and
what a Space with PRO or a Cloud Run service would run.

### Shape B — one process (Streamlit Community Cloud)

Streamlit Cloud runs exactly one script in one process. The cheap move would be to
have the UI import `core/` directly — but then the UI contains business logic, and
the evaluation harness would be measuring a different code path from the one users
hit.

Instead, [`streamlit_app.py`](../streamlit_app.py) starts the **real FastAPI app
in a background thread**:

```
┌────────────── one Python process ──────────────┐
│                                                │
│  Streamlit (main thread)                       │
│       │ HTTP to 127.0.0.1:8000                 │
│       ▼                                        │
│  FastAPI (daemon thread, same process)         │
│       ▼                                        │
│  ChromaDB + embedding model                    │
│  (index built at startup if missing)           │
└────────────────────────────────────────────────┘
```

The HTTP boundary is unchanged. The UI still only knows how to make HTTP calls,
all logic still lives behind the API, and `eval/run_eval.py` still measures the
same pipeline. Only the process topology differs.

Why a **thread** and not a subprocess: Streamlit re-runs its script on every user
interaction. A subprocess would be orphaned or re-spawned each time; a
module-level daemon thread guarded by a global starts once per process and dies
with it.

---

## First-time setup (Streamlit Community Cloud)

Done once. Takes about five minutes.

1. Go to **<https://share.streamlit.io>** and sign in with GitHub.
2. Click **New app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `aakashc23/schemesure`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`  ← the root file, **not** `ui/streamlit_app.py`
4. Open **Advanced settings**:
   - **Python version:** `3.12`
   - **Secrets:** paste the block below, with your real key.
5. Click **Deploy**.

```toml
LLM_API_KEY = "gsk_your_groq_key_here"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "qwen/qwen3.8-27b"
GUARDRAIL_ENABLED = "true"
RETRIEVAL_MIN_SCORE = "0.45"
```

Streamlit Cloud exposes secrets as environment variables, which is exactly where
`app/config.py` reads from — so no code change is needed.

### What the first boot does

Expect **2–4 minutes**, mostly installing dependencies:

1. `pip install -r requirements.txt` — including CPU-only torch (~200 MB).
2. The app starts; the thread launches FastAPI.
3. The embedding model downloads (~120 MB) and loads (~15 s).
4. **No Chroma index exists**, so `app/main.py` builds it from `data/schemes/`
   (~25 s, 312 chunks). The UI shows a progress bar while this happens.
5. `/health` answers 200 and the UI renders.

Subsequent visits are instant while the app stays warm. It sleeps after 12 hours
idle and takes ~30 s to wake.

> **Do not commit `chroma_db/`.** It is gitignored, and the app rebuilds it in
> ~25 s. Committing a binary vector index would bloat the repo and go stale
> silently whenever the data changed.

---

## Redeploying

### Normal case: just push

```bash
git push origin main
```

Streamlit Cloud watches the repo and rebuilds automatically. Nothing else to do.

**Be honest about what this means:** Streamlit Cloud rebuilds on push regardless
of whether CI passed. The gate that actually holds is that tests run on **every
pull request**, so a change should not reach `main` red. If you want a hard gate,
work on branches and merge via PR.

### Changing a secret

Secrets live in the Streamlit Cloud dashboard, not in the repo:
**Manage app → Settings → Secrets**. Saving restarts the app.

### Changing the scheme data

```bash
python scripts/fetch_schemes.py
```

```bash
python scripts/build_rules.py && python scripts/validate_data.py
```

```bash
python scripts/make_verify_checklist.py && python -m core.ingest
```

Then commit and push. The deployed app rebuilds its own index on restart, so no
index files need committing.

### Forcing a rebuild without a code change

**Manage app → Reboot app** in the dashboard.

---

## Deploying the container instead

The Docker path is unchanged and still the production shape.

### Locally

```bash
docker build -t schemesure .
```

```bash
docker run -p 7860:7860 --env-file .env schemesure
```

Then open <http://localhost:7860>. The image builds its own index, so the first
build takes a few minutes and every start after that is immediate.

### Hugging Face Space (needs PRO)

```bash
python scripts/deploy_hf.py
```

This creates the Space if needed, sets `LLM_API_KEY` / `LLM_BASE_URL` /
`LLM_MODEL` as **Space secrets** through the API (never in a file), uploads the
files, and polls the build. Useful flags:

```bash
python scripts/deploy_hf.py --status    # report the current stage
python scripts/deploy_hf.py --no-wait   # push and return immediately
python scripts/deploy_hf.py --verify    # check the live app responds
```

There is also a manual GitHub Actions job:
**Actions → CI/CD → Run workflow**. It is manual-only on purpose — on a free
account it returns 402, and a job that fails on every push trains everyone to
ignore a red tick.

Two details in `deploy_hf.py` worth knowing:

- **Uploads use an explicit allow-list**, not an ignore-list. A deny-list fails
  silently the day someone adds a file that should not be published, and on a
  public Space that is irreversible. With an allow-list, a new file is simply not
  uploaded until someone adds it — failing in the safe direction. It also refuses
  outright if `.env` or any database file matches.
- **The Space README is generated at upload time.** A Space needs YAML
  front-matter at the top of its README; keeping that in the repo's README makes
  GitHub render a stray table. So it is prepended during upload.

### Google Cloud Run

Runs the same image with no changes. Free tier covers roughly 100 hours/month of a
1 GiB container, but a credit card is required for verification and cold starts
take 30–60 s while the 1.2 GB of model and torch loads.

```bash
gcloud run deploy schemesure --source . --port 7860 \
  --memory 2Gi --allow-unauthenticated \
  --set-env-vars LLM_BASE_URL=https://api.groq.com/openai/v1,LLM_MODEL=qwen/qwen3.8-27b \
  --set-secrets LLM_API_KEY=groq-key:latest
```

---

## Configuration reference

Everything is read by `app/config.py` from environment variables (or `.env`
locally). Nothing else in the codebase reads `os.environ`.

| Variable | Default | Notes |
|---|---|---|
| `LLM_API_KEY` | — | **Required.** Free key from <https://console.groq.com/keys> |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen/qwen3.8-27b` | Avoid reasoning models — see below |
| `RETRIEVAL_MIN_SCORE` | `0.45` | Below this, refuse. Calibrated on 22 queries |
| `RETRIEVAL_TOP_K` | `5` | Passages retrieved per question |
| `GUARDRAIL_ENABLED` | `true` | `false` only for the evaluation baseline |
| `GUARDRAIL_BLOCK_THRESHOLD` | `0.5` | Unsupported fraction above which we refuse |
| `MAX_QUESTION_LENGTH` | `500` | Characters |
| `CHROMA_PATH` | `./chroma_db` | Rebuilt automatically if empty |
| `LLM_LOG_DB` | `./llm_calls.db` | Powers `/metrics` |
| `API_URL` | `http://localhost:8000` | Where the UI looks for the API |

---

## Troubleshooting

**The UI says the backend is not reachable.**
Locally, the API is a separate process — start it with
`python -m uvicorn app.main:app --port 8000`. On Streamlit Cloud, check the logs
for a thread crash; the usual cause is running out of memory.

**Every question is refused.**
Check `GET /health`. If `chunks_indexed` is `0`, the index is missing and the
automatic build failed — run `python -m core.ingest` and read the error. If
`llm_key_configured` is `false`, the secret is not reaching the process.

**`/ask` returns 503.**
`LLM_API_KEY` is empty or was rejected at startup. The server deliberately still
boots in this state so `/health` can tell you so.

**`/ask` returns 502.**
Groq itself failed. 502 means "upstream broke", 500 would mean "our bug" — the
distinction is there so logs are readable.

**Answers come back empty, or JSON parsing fails.**
You are probably using a **reasoning** model. Groq's `openai/gpt-oss-*` models
spend completion tokens on hidden reasoning before emitting content, so a small
`max_tokens` budget is consumed entirely by reasoning and the content arrives
empty. `app/llm_client.py` detects this and says so explicitly. Either raise
`LLM_MAX_TOKENS` substantially or use a non-reasoning model like
`qwen/qwen3.8-27b`.

**Lots of rate-limit errors.**
Groq's free tier allows **8,000 tokens per minute**, and one verified answer costs
roughly 1,500–2,000. The client parses Groq's own "try again in 1.68s" hint and
waits exactly that long. For bulk runs, pace them:
`python eval/run_eval.py --delay 25`.

**The container exits immediately with "no such file or directory" for `start.sh`.**
`start.sh` has Windows CRLF line endings, so Linux is looking for an interpreter
called `bash\r`. `.gitattributes` forces LF on `*.sh` to prevent this; if you
created the file outside git, convert it.

**Docker build fails downloading torch.**
Confirm the CPU index is being used. PyPI's default Linux `torch` wheel is the
~2.5 GB CUDA build and can exhaust the build disk or time limit.
