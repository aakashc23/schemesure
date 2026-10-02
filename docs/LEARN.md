# Understanding SchemeSure, file by file

A walkthrough for someone who has never seen this codebase. We follow **one
question** from the moment it is typed to the moment an answer appears, and stop
at each file on the way. No prior knowledge of RAG, Docker or CI/CD assumed.

If you only remember one sentence, make it this one:

> **The language model handles language. Python handles truth.**

Everything else in this project is a consequence of that.

---

## Part 0 — The idea in one picture

Imagine a very fluent but over-confident intern.

Ask them "how much does PM-KISAN pay?" and they will *always* give you a
confident answer — even when they do not know, because sounding uncertain feels
like failing. That is a large language model.

Now imagine you give that intern a folder of official documents and three rules:

1. **Only use what is in the folder.**
2. **Show me which page each fact came from.**
3. **A second person checks every sentence against the folder before it reaches me.**

The folder is our **vector database**. Rule 2 is **citations**. Rule 3 is the
**guardrail**. That is the whole project.

**Memory trick — "RAG":** **R**etrieve (find the right pages), **A**ugment (paste
them into the prompt), **G**enerate (write the answer). SchemeSure adds a fourth
step most systems skip: **V**erify. *R-A-G-V.*

---

## Part 1 — Before any question arrives: building the folder

This happens once, before the app serves anyone. Three scripts, in order.

### `scripts/fetch_schemes.py` — getting real data

**Job:** download official information about 15 government schemes.

The obvious approach — fetch `myscheme.gov.in/schemes/pm-kisan` and read the
HTML — returns almost nothing. The page is a **JavaScript app**: the server sends
an empty shell, and the browser fills it in afterwards by calling an API.

So we call that API directly:

```
https://www.myscheme.gov.in/api/apisetu/schemes?slug=pm-kisan&lang=en
```

Same official source, far more reliable, and it hands us the editors' own
markdown. The response is saved **untouched** in `data/raw/`, so any fact can be
traced back to a response we really received.

> **Analogy:** the web page is a restaurant's plated dish. The API is the
> kitchen. We order from the kitchen.

**The rule that matters:** if the official page does not say something, the file
says `"Not specified in official source"`. We never fill a gap from the model's
memory. A visible gap is information; an invented fact is a bug nobody can spot
later.

**Output:** `data/schemes/pm_kisan.md` and 14 more, each with a header
(`scheme_id`, `name`, `ministry`, `source_url`, `last_verified`) and six sections:
Overview, Benefits, Eligibility, Exclusions, Documents Required, How to Apply.

### `scripts/build_rules.py` — turning sentences into numbers

**Job:** convert prose like *"The age of the applicant must be between 18 to 50
Years"* into something code can check:

```json
{ "min_age": 18, "max_age": 50, "max_annual_income": null, ... }
```

Two things make this honest:

- **`null` means "the official source says nothing"** — never "probably no
  limit". If a scheme states no income cap, the engine simply does not test
  income.
- **Every number cites the sentence it came from**, in `rule_sources`. A number
  with no sentence behind it is an unverifiable claim.

Some official conditions genuinely do not fit seven fields. Stand-Up India says
*"if the applicant is male, he must be from SC/ST"* — that is a **conditional**
rule. Squeezing it into a `gender` or `category` field would produce confidently
wrong filters. So those go into `unmodelled_conditions` word-for-word and are
always shown to the user as "still to verify".

> **Why this matters in an interview:** knowing the limits of your own data model,
> and surfacing them instead of hiding them, is the difference between a demo and
> a system.

### `scripts/validate_data.py` — the smoke alarm

Ten checks: are all six sections present, is the source URL really a `.gov.in`
address, is `last_verified` a real date, does every `.md` have a matching
`.rules.json`, does every active rule cite its source, is `min_age <= max_age`,
did any HTML entity survive?

**Why bother:** if the data is wrong, every answer about that scheme is wrong, and
it will look like "the AI is bad" rather than "row 3 is malformed". This turns a
silent data bug into a loud, early error.

---

## Part 2 — `core/ingest.py`: cutting documents into findable pieces

A computer cannot "search for meaning" in a 5,000-word document. It needs small
pieces, each turned into a list of numbers.

### Embeddings, in plain words

An **embedding** turns text into ~384 numbers that represent its *meaning*.
Similar meanings land near each other, even in different languages.

> **Analogy:** a map. Delhi and Noida sit close together; Delhi and Tokyo do not.
> Embeddings are a map of meaning, and "nearby" means "about the same thing".

Closeness is measured by **cosine similarity**: `1.0` = identical direction,
`0.0` = unrelated, negative = opposite. All our scores are this number.

### Chunking — and why it is section-aware

The naive approach is "split every 500 characters". That cuts sentences in half
and can put PM-KISAN's benefits and Ujjwala's eligibility in the same piece —
making a citation meaningless.

Our documents already have real boundaries (`## Benefits`, `## Eligibility`), so
we split on those first. **Every chunk therefore belongs to exactly one scheme and
one section**, which is what makes "PM-KISAN, Benefits" a truthful citation.

### The token budget — a real bug avoided

Our embedding model reads at most **128 tokens** (~90 words). Anything longer is
**silently thrown away**.

That is the nastiest kind of bug: you believe you indexed a document, the model
only ever saw the first third, and retrieval is quietly worse forever with no
error anywhere.

So we measure every chunk with the model's own tokenizer and cap it at **110**
tokens, splitting at sentence boundaries when needed. `core/ingest.py` refuses to
finish if any chunk is over budget, and the test suite re-checks it.

> **Memory trick:** *count with the model's own ruler, not your own.*

**Chunk ids are readable:** `pm_kisan#benefits#0`. Two payoffs — re-running
ingestion overwrites instead of duplicating, and because the guardrail judge must
quote these ids, a made-up one is obvious at a glance.

**Result:** 312 chunks, stored in **ChromaDB** (a vector database — a folder that
stores embeddings and answers "what is nearest to this?" quickly).

One line in that file matters more than it looks:

```python
metadata={"hnsw:space": "cosine"}
```

Chroma's default is *squared L2 distance*, a different scale entirely. Without
this line, every similarity threshold in the project would be comparing against a
meaningless number.

---

## Part 3 — A question arrives

Now the live path. Five files, in order.

### `app/main.py` — the front door

A **FastAPI** app: Python that answers HTTP requests. Five endpoints:

| Endpoint | Meaning |
|---|---|
| `POST /ask` | answer a question |
| `POST /eligibility` | check a profile against all 15 schemes |
| `GET /schemes` | what we cover |
| `GET /health` | am I alive, is my index loaded |
| `GET /metrics` | how many LLM calls, how slow, how often blocked |

Two deliberate choices:

- **The heavy objects load once at startup**, not per request. Loading a 120 MB
  model on every question would add seconds to every answer.
- **A missing API key does not stop the server.** It boots, `/health` reports
  `llm_key_configured: false`, `/schemes` still works, and `/ask` returns a clear
  503. A container that refuses to start because a secret is missing is much
  harder to debug than one that starts and tells you what is wrong.

This file contains **no business logic** — it validates input and calls `core/`.
That is why the evaluation harness can test the same logic with no web server.

### `app/config.py` — one place for every setting

Every tunable (model name, threshold, top-k) read from `.env` with a type.
Nothing anywhere else reads `os.environ`. A typo'd setting fails loudly at
startup instead of silently behaving differently.

### `app/schemas.py` — the shapes of things

**Pydantic** models describing every piece of data. Pydantic validates
automatically: if something should be an `int` between 0 and 120 and a string
arrives, you get a clear error, not a crash three functions later.

The interesting ones are the enums, because they encode the design:

- `GuardrailDecision`: `PASS` / `REPAIR` / `BLOCK` / `SKIPPED`
- `AnswerStatus`: `verified` / `partially_verified` / `refused` / `unverified`
- `EligibilityStatus`: `ELIGIBLE` / `NOT_ELIGIBLE` / **`NEED_MORE_INFO`**

That third eligibility value is the whole philosophy in one enum: *we are allowed
to say "I need more information."*

### `app/llm_client.py` — the only door to the model

Every LLM call in the project goes through here, which buys three things:

1. **Logging.** Every call recorded in SQLite: which prompt, how many tokens, how
   long, did it fail. Without this you cannot answer "how many LLM calls does one
   question cost?" — the first question anyone asks.
2. **Retries that listen.** Groq's free tier allows 8,000 tokens per minute, and
   when you exceed it the error says *"Please try again in 1.68s"*. We parse that
   and wait exactly that long. Blind exponential backoff either sleeps far too
   long or retries too early and burns an attempt.
3. **JSON tolerance.** Models wrap JSON in ``` fences or add "Sure! Here you go:"
   even when told not to. `extract_json` strips that rather than failing.

We talk to **Groq** (fast, free tier) using the **OpenAI client library**, because
Groq speaks the same format. Switching providers means editing `.env`, not code.

### `app/prompts.py` — all five prompts in one file

Prompts *are* the behaviour of an LLM app. Scattering them through the code makes
them impossible to review. Here you can read all five in two minutes.

Also here: the **refusal messages**, hand-written in English, Hindi and Hinglish.
A refusal must never itself be hallucinated — and we often refuse precisely
because the model cannot be trusted on that input. Asking it to write the refusal
would be circular.

---

## Part 4 — The five steps of answering

### Step 1 · `core/query_normalizer.py` — understand the question

One LLM call returns `{normalized_query, language}`: the question rewritten in
English, plus which language the user actually used.

**This step looks optional. It is not.** Here are the real measured scores for
the *same* PM-KISAN question, against the chunk that actually answers it:

| Asked in | Score | Found the right scheme? |
|---|---|---|
| English | **0.574** | yes |
| Hindi (Devanagari) | 0.610 | **no** |
| Hinglish (`PM Kisan ka paisa kitna milta hai?`) | **0.436** | **no** |

Our refusal threshold is 0.45. So without this step, a Hinglish speaker gets
*"I have no information"* about a scheme we have fully indexed.

Why? Our documents are English, and the embedding model learned Devanagari Hindi
and English — **not Hindi spelled in Latin letters**. Hinglish is the hardest case
and the most common way Indians actually type.

If this call fails we fall back to the raw question plus a simple script check,
and set `fallback_used=True` — degraded, but visibly so.

### Step 2 · `core/retriever.py` — find the evidence

Embed the English query, ask Chroma for the 5 nearest chunks.

The important part: the retriever does not just return passages, it returns a
**judgement** — `low_evidence=True` when even the best match is too far away.

**A vector search always returns something.** Ask for a pizza recipe and you get
five scheme chunks back, just with poor scores. Hand those to a model and it will
write something plausible. That is how RAG systems hallucinate *with a perfectly
good index*.

The threshold was **calibrated, not guessed** — 22 real queries:

| Query type | Score range |
|---|---|
| Real, answerable (English) | 0.574 – 0.893 |
| Nonsense ("capital of France", "pizza") | 0.100 – 0.350 |
| **Gap between them** | **0.350 – 0.574** |

`0.45` sits near the middle of that gap.

**And here is the limitation you should be able to state out loud.** The threshold
catches *off-topic*. It does **not** catch *wrong-scheme*:

> "What is the interest rate on Sukanya Samriddhi Yojana?" — a real scheme we do
> **not** index — retrieves PM-Mudra chunks at **0.71**. Far above the threshold.

Because it looks exactly like a scheme question. No threshold can separate those.
That is the guardrail's job. **Two failure modes, two different defences.**

### Step 3 · `core/generator.py` — write a draft

If evidence is weak → **refuse immediately, with zero LLM calls**. There is
nothing to ground an answer in, so generating one can only produce fiction.
Out-of-scope questions are therefore free.

Otherwise, one call: here are the passages, answer using only these, cite them as
`[1]`, reply in the user's language.

### Step 4 · `core/guardrail.py` — the heart of the project

The draft is *not* shown to anyone yet. Four steps:

**4a. Split into atomic claims** (1 LLM call).
"Rs 6,000 per year, paid in three instalments" becomes **two** claims. Why split?
Check it as one sentence and it passes if *either* half is right — which is
exactly how a wrong number survives verification.

**4b. Judge each claim** (1 LLM call).
Each claim is checked against **only the passages**. Two deliberate rules:

- **The judge never sees the question.** A judge who knows what the answer was
  *supposed* to say is biased toward approving it.
- **"Probably true in the real world" counts as NOT_SUPPORTED.** If the passages
  do not state it, it is not grounded — even if it happens to be correct.

**4c. Python checks the judge's homework.**
The judge must name the `chunk_id` supporting each claim. Code then checks that
id actually exists.

This catches something genuinely dangerous: the judge marking a claim SUPPORTED
while citing `pm_kisan#benefits#99`, a passage that was never retrieved. It looks
rigorous and is worthless. **Checking set membership is something code gets right
every single time — an LLM cannot be relied on to.**

There is a test (`tests/test_guardrail.py`) that shows this turning a naive
`PASS` into a correct `BLOCK`.

**4d. Python decides.**

```python
def decide(checks, block_threshold):
    if not checks:                  return PASS      # nothing factual was claimed
    if unsupported == 0:            return PASS      # all good → ship it
    if supported == 0:              return BLOCK     # nothing true left to say
    if ratio > block_threshold:     return BLOCK     # mostly wrong → refuse
    return REPAIR                                    # mostly right → rebuild
```

A **pure function**: same input, same output, no network, no model. This is the
one piece that must never be wrong, so it is code — and it is tested
exhaustively, including every boundary.

> **Why not let the LLM decide?** An LLM judging its own work is not a control,
> it is another generation. The parts that must not be wrong are Python.

### Step 5 · Repair, or refuse

- **PASS** → ✅ Verified. Ship it.
- **REPAIR** → the model gets the list of *verified* claims and is told to state
  those and nothing else. Not "try again" — the failed claim **cannot** come
  back, because it is not in the list. → ⚠️ Partially verified.
- **BLOCK** → 🛑 the hand-written refusal.

**One more trap, found by actually running it.** For a scheme we do not index, the
model sometimes writes its own prose refusal. The splitter then finds **zero**
claims, `decide()` returns `PASS` ("nothing to verify"), and the answer was
getting badged **✅ Verified** with citations to whatever unrelated scheme came
back. Vacuously true is not verified. Now zero claims on a drafted answer is
treated as a **refusal**.

**Cost summary — measured, not estimated:**

| Case | LLM calls |
|---|---|
| Off-topic (refused early) | **1** (just normalize) |
| Normal verified answer | **4** |
| Answer needing repair | **5** |

---

## Part 5 — `core/eligibility.py`: where no LLM is allowed near a decision

Different flow, same philosophy.

You write *"I am a 35 year old farmer from Bihar earning about 2 lakh a year."*

**One LLM call** turns that into fields:
`{age: 35, occupation: "farmer", state: "Bihar", annual_income: 200000}`.
That is all it does — *extraction*. It converts "2 lakh" to `200000`.

**Everything after that is `if` statements.** Three reasons:

1. It is arithmetic and set membership. Code is exactly right; an LLM is
   approximately right.
2. It must be explainable. *"Not eligible: your age 35 is above the maximum of
   25"* is a reason you can check and argue with. *"The model said no"* is not.
3. It decides whether someone applies for money they are entitled to.

**The missing-data rule.** If the user never mentioned income and a scheme has an
income cap, the answer is **NEED_MORE_INFO** — not a guess in either direction.

There is a test for a subtle version of this: a profile saying "farmer" with no
gender stated must return NEED_MORE_INFO for a women-only scheme, **not**
NOT_ELIGIBLE. "Farmer" says nothing about gender, and quietly assuming male would
hide a scheme a woman is entitled to.

And a Python trap worth knowing: `if profile.age:` is **wrong**, because `0` is
falsy — a newborn would read as "age not provided". The code uses
`if profile.age is None:`. There is a test for exactly that.

---

## Part 6 — `ui/streamlit_app.py`: the screen

**Streamlit** turns Python into a web page with no HTML or JavaScript.

Three tabs: **Ask**, **Check eligibility**, **Schemes**. Plus the part that makes
the project convincing — the expander **"How was this verified?"**, which shows
every claim, its verdict, and the passage behind it.

> A badge alone is just another confident assertion. Showing the audit trail is
> what makes it believable.

**This file contains no business logic.** It calls the API over HTTP and renders
the result. That is why the API, the UI and the evaluation all exercise the same
code — if logic lived here, the eval would measure something users never touch.

---

## Part 7 — Testing: 166 tests, no internet

Run them with `python -m pytest -q`. They need **no API key** and make **no
network calls** — so they are fast (~17s) and never flaky.

**What is faked and what is real:** only the LLM is faked. The embedding model,
the Chroma index, retrieval scores, the decision policy and the eligibility rules
are all real. Mocking retrieval too would leave the tests asserting that mocks
return mocks.

The fake judge even reads chunk ids out of the prompt it is handed, so a
"supported" verdict cites a genuinely retrieved chunk and faces the same Python
citation check as the real thing.

| File | What it protects |
|---|---|
| `test_guardrail.py` | the decision policy, every boundary; fabricated citations |
| `test_eligibility.py` | boundary ages/incomes, null rules, missing fields |
| `test_ingest.py` | chunking, deterministic ids, the token budget (real tokenizer) |
| `test_query_normalizer.py` | language detection, graceful degradation |
| `test_llm_client.py` | JSON extraction, rate-limit retry parsing |
| `test_api.py` | all 5 endpoints end-to-end with real retrieval |

**Memory trick:** *fake the expensive and unpredictable thing; keep everything
else real.*

---

## Part 8 — Docker, explained from scratch

### The problem it solves

"It works on my machine." Your laptop has Python 3.12, some library versions, and
a model cached in a folder. A server has none of that. Setup instructions drift,
and the app breaks in a way nobody can reproduce.

### What Docker actually is

> **Analogy:** a shipping container. Before containers, loading a ship meant
> handling barrels, sacks and crates differently. Standardise the box and any
> crane can lift any of them. Docker standardises the box around software.

- **Image** = the recipe's result. A frozen, complete snapshot: OS, Python,
  libraries, your code, the model. Read-only.
- **Container** = a running copy of an image.
- **Dockerfile** = the recipe.

**Image is to container as class is to object** — or as a *cake recipe* is to
*a cake*.

### Reading our `Dockerfile`

```dockerfile
FROM python:3.12-slim
```
Start from a minimal Linux with Python 3.12. (`slim` = no extras we do not need.)

```dockerfile
RUN useradd -m -u 1000 user
USER user
```
Do not run as root. Also what the host requires.

```dockerfile
COPY requirements.txt ./
RUN pip install ...
```
**Dependencies are copied before the code, and this ordering is the point.**
Docker caches each step and only redoes steps after the first change. Code
changes far more often than dependencies, so this way editing a Python file
reuses the cached install instead of re-downloading 300 MB.

> **Memory trick:** *things that rarely change go first.*

```dockerfile
RUN pip install torch==2.14.1 --index-url .../whl/cpu
```
On Linux, the default `torch` is the **GPU build — about 2.5 GB**. There is no GPU
here, so it would be 2.5 GB of wasted download. The CPU-only build is a fraction
of that.

```dockerfile
RUN python -c "...SentenceTransformer(...)"
RUN python -m core.ingest
```
Download the model **and build the search index while building the image**. Two
wins: the container starts instantly ready, and if the data is broken the **image
build fails** — a broken corpus never reaches users.

```dockerfile
CMD ["bash", "start.sh"]
```

### `start.sh` — two programs, one port

The host exposes exactly **one** port. We need two programs. So:

1. Start FastAPI on `127.0.0.1:8000` — internal only, invisible from outside.
2. **Wait for `/health` to really return 200.** Launching the UI first would show
   the very first visitor a connection error.
3. Start Streamlit on the public port, pointing at `http://127.0.0.1:8000`.

> **Analogy:** a restaurant. The kitchen (FastAPI) has no street door; the counter
> (Streamlit) does. Orders go kitchen-ward through an internal hatch.

**The one-character bug worth knowing.** Windows ends lines with `\r\n`, Linux
with `\n`. A `start.sh` saved Windows-style makes Linux look for an interpreter
called `bash\r`, and you get `no such file or directory` — pointing at a file
that plainly exists. `.gitattributes` forces `*.sh` to LF so this can never
happen.

### Useful commands

```bash
docker build -t schemesure .
```

```bash
docker run -p 7860:7860 --env-file .env schemesure
```

`-p 7860:7860` means "connect my port 7860 to the container's 7860".

---

## Part 9 — CI/CD, explained from scratch

**CI** = Continuous Integration: every push, run the tests automatically.
**CD** = Continuous Deployment: if they pass, ship it.

> **Analogy:** a factory quality gate. Every item coming off the line is checked
> by the same machine, the same way, every time — rather than whenever someone
> remembers to look.

### What `.github/workflows/ci-deploy.yml` does

On every push and pull request, GitHub rents a fresh Linux machine and runs:

1. `python scripts/validate_data.py` — is the corpus sane? **First**, because if
   the data is broken, every later failure is a confusing symptom of the same
   cause.
2. `python -m core.ingest` — build the index (the tests use real retrieval).
3. `python -m pytest -v` — all 166 tests.
4. `python scripts/scan_secrets.py --tracked` — did a credential sneak in?

The whole thing takes about two minutes, and you see a ✓ or ✗ on the commit.

**Why it is worth it:** it catches the thing you did not think to re-check. You
fix a typo in one file, and a test in a file you have not opened for a week tells
you that you broke something.

### Where the app is actually hosted

This project was built for a **Hugging Face Docker Space**. Partway through,
deploying returned **402 Payment Required**:

> *"Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on
> free cpu-basic requires a PRO subscription."*

Only `static` Spaces are still free, and those cannot run Python. So the live demo
moved to **Streamlit Community Cloud** — free, no card, and it redeploys itself on
every push to `main`.

**But the architecture did not collapse to fit it.** Streamlit Cloud gives you one
process. Instead of making the UI import `core/` directly, `streamlit_app.py`
starts the **real FastAPI app in a background thread** and the UI talks to it over
HTTP on localhost. The boundary is unchanged; only the process layout differs. The
Dockerfile is still the production path and still runs anywhere.

Two honest notes:

- Streamlit Cloud rebuilds on push **whether or not** the tests passed. The gate
  that really holds is that tests run on every pull request.
- The HF deploy job still exists but is **manual-only**, because a job that fails
  on every push teaches everyone to ignore a red tick.

### Secrets

Your API key lives in `.env`, which is in `.gitignore` and was never committed.
For CI, the same values are stored as **GitHub Secrets** — encrypted, injected as
environment variables, and masked in logs.

`scripts/scan_secrets.py` runs before every push as a second line of defence. It
scans exactly what git is about to send, checks `.env` is still untracked, and
**redacts anything it finds** so a secret is never printed even in a failure.

> **Why so careful:** a key pushed to a public repo is leaked *permanently*.
> Deleting the commit does not help — it was already cloned, cached and indexed.
> The only fix is to never push it.

---

## Part 10 — Reading order, and the five sentences to remember

If you are opening this project cold, read in this order:

1. `README.md` — what and why
2. **This file** — how
3. `data/schemes/pm_kisan.md` — what the system actually knows
4. `app/prompts.py` — what the model is and is not allowed to do
5. `core/guardrail.py` — the heart; read `decide()` first
6. `tests/test_guardrail.py` — the guarantees, as executable statements
7. `core/eligibility.py` — rules without an LLM
8. `core/generator.py` — the pipeline tying it together
9. `eval/results.md` — did any of it work
10. `docs/DECISIONS.md` — every trade-off, with its measurement

### The five sentences

1. **LLM for language, Python for truth.** Every threshold, comparison and
   verdict is code.
2. **Retrieval always returns something**, so the retriever must also judge
   whether that something is good enough.
3. **A similarity threshold catches off-topic, not wrong-scheme.** Measured: a
   question about an unindexed scheme scored 0.71, well above our 0.45 cut-off.
   Only claim-level verification catches that.
4. **Verify claim by claim, and check the checker in code.** A fabricated
   `chunk_id` is caught by set membership, which code never gets wrong.
5. **When in doubt, refuse.** A withheld answer costs the user one web search; a
   confidently wrong one can cost them a wasted day at a government office.
