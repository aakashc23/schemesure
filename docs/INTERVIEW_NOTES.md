# Interview notes

Everything you need to talk about SchemeSure confidently and honestly. The
numbers here are real measurements from this repository — if you quote them, you
can back them up.

---

## The 2-minute pitch

> **SchemeSure is a question-answering assistant for Indian government welfare
> schemes that is built around the assumption that the language model will
> sometimes be wrong.**
>
> The problem: someone wants to know if they qualify for a scheme. A general
> chatbot gives a fluent, confident answer that is occasionally wrong — a
> slightly wrong amount, an age limit borrowed from a different scheme. For
> welfare information that is not cosmetic: it sends someone to a government
> office with the wrong documents, or convinces them not to apply for money they
> were entitled to.
>
> Plain RAG does not fix this, and I can show you why with a number from my own
> system. If you ask about *Sukanya Samriddhi Yojana* — a real scheme I deliberately
> do **not** index — retrieval returns PM-Mudra passages at **0.71** cosine
> similarity, far above my 0.45 refusal threshold. A similarity threshold cannot
> tell "a scheme question I can answer" from "a scheme question I can't", because
> they look identical. So a naive pipeline answers that question confidently, from
> the wrong scheme's text.
>
> My answer is two ideas. First, **the LLM handles language and Python handles
> truth** — every threshold, comparison and eligibility verdict is code, not a
> model output. Second, **every answer is verified before anyone sees it**: it is
> split into atomic claims, each claim is checked against only the retrieved
> passages, and then — critically — *Python* verifies that the chunk the judge
> cited actually exists, and *Python* decides whether to publish, repair or
> refuse. An LLM judging its own work is not a control, it's another generation.
>
> I measured it both ways on a 50-question golden set, guardrail off versus on.
> *(quote your headline numbers from `eval/results.md`)*
>
> It's FastAPI, ChromaDB and a local multilingual embedding model, with a
> Streamlit frontend, 169 tests that run with no network calls, Dockerised, and
> deployed with CI. It answers in English, Hindi and Hinglish.

**If you have 30 seconds instead:** "A RAG assistant for Indian government
schemes where every answer is fact-checked claim-by-claim against the retrieved
official text before the user sees it, and eligibility is decided by Python rules
rather than the model. When the evidence isn't there, it refuses instead of
guessing."

---

## 20 likely follow-up questions

### On hallucination detection

**1. How do you actually detect a hallucination?**
Three layers. (a) Before generating: if the best retrieved passage is below a
calibrated similarity threshold, I refuse without spending an LLM call — there's
nothing to ground an answer in. (b) After generating: the draft is split into
atomic claims and a judge checks each one against *only* the passages. (c) In
code: Python verifies the judge's cited `chunk_id` really exists, and a pure
function decides pass/repair/block. The detection is LLM-assisted; the *decision*
is deterministic.

**2. Why split into claims instead of judging the whole answer?**
Because one sentence often carries two facts. "Rs 6,000 per year, paid in three
instalments" is two claims. Judged as one sentence, it passes if *either* half is
right — which is exactly how a wrong number survives verification. Splitting is
what makes the verdict mean something.

**3. Isn't using an LLM to check an LLM circular?**
Partly, and I don't claim otherwise. Two things break the circle. The judge sees
only the evidence, never the question, so it can't be swayed by what the answer
was *supposed* to say. And the two checks that must not be wrong are code: does
the cited chunk exist, and is the unsupported ratio over the limit. Those are set
membership and arithmetic — code gets them right every time. The honest residual
risk is that judge and generator share a blind spot, because they're the same
model family. The fix is a different model as judge plus a human-labelled subset;
I'd do that next.

**4. What's the fabricated-citation check, and why does it matter?**
The judge must name the chunk supporting each claim. Sometimes it marks a claim
SUPPORTED and cites `pm_kisan#benefits#99` — an id that was never retrieved. That
verdict looks rigorous and is worthless. Python checks set membership and demotes
the claim. There's a test showing this flipping a naive PASS into a correct BLOCK:
the judge said 3/3 supported, two citations were fabricated, so the real result
was 1/3 and the answer was refused.

**5. Why do you treat a true-but-unstated fact as unsupported?**
Because grounding is the property I'm enforcing, not correctness. If a model adds
"beneficiaries need an Aadhaar-linked account" and the retrieved passage doesn't
say it, I can't verify it — and I can't tell that case apart from the one where
it's wrong. Allowing "probably true" reopens exactly the hole the guardrail
exists to close. It does cost me some over-blocking, which I measure rather than
hide.

### On over-blocking and the trade-off

**6. Doesn't this make it refuse too much?**
It raises refusals, yes, and I report that as "over-blocking rate" next to the
benefit rather than burying it. The asymmetry is deliberate: a wrongly withheld
answer costs the user one web search; a wrongly confident one can cost them a
wasted day at a government office. For welfare information that trade is clearly
worth it. For a movie-recommendation bot it wouldn't be.

**7. How would you tune that balance?**
`GUARDRAIL_BLOCK_THRESHOLD` is one env variable — the unsupported fraction above
which I refuse rather than repair. At 0.5, half-unsupported answers get repaired;
at 0.0, any unsupported claim blocks. The REPAIR path is what makes this safe to
tune: instead of all-or-nothing, it rebuilds the answer from verified claims only,
so the failing claim can't survive either way.

**8. What happens if the verification step itself fails?**
It fails closed — refuse. If the splitter or judge call errors, I have an
unverified draft and no way to check it. Shipping it would mean the guardrail
silently stops protecting the user exactly when the model is misbehaving. The cost
is availability: a Groq outage produces refusals instead of unchecked answers.
That's the right trade here, and every such refusal is logged with its reason.

### On retrieval

**9. How did you pick the similarity threshold?**
Measured, not guessed. I ran 22 queries across answerable, out-of-scope, and
trick categories. Answerable English questions scored **0.574–0.893**;
out-of-scope ones **0.100–0.350**. That leaves a gap of 0.350–0.574, and I put
the threshold at **0.45**, near the midpoint — about 0.10 above the worst
off-topic case and 0.12 below the weakest real question.

**10. Why that embedding model?**
I picked it on measured *separation*, not benchmark reputation.
`multilingual-e5-small` has a longer context window, but it compressed everything
into a narrow band: off-topic "capital of France" scored **0.70** against an
on-topic **0.81**, and a question about a scheme I don't index also scored 0.80. A
threshold needs a gap to sit in, and e5 left none.
`paraphrase-multilingual-MiniLM-L12-v2` gave me 0.52 on-topic versus −0.08
off-topic. Its cost is a 128-token limit, which forces small chunks — and that
turned out to help, because smaller chunks make claim-level citations more precise.

**11. Why not just use the multilingual model directly on Hindi queries?**
I tried; it doesn't work reliably. The same PM-KISAN question scored **0.574** in
English (correct scheme), **0.610** in Devanagari Hindi (*wrong* scheme), and
**0.436** in Hinglish (*wrong* scheme, and below my 0.45 threshold). So without a
normalisation step, a Hinglish speaker would be told I have no information about a
scheme I've fully indexed. Romanised Hindi is the hard case — the model saw
Devanagari Hindi and English in training, not Hindi spelled in Latin letters. One
LLM call fixes it and tells me the language so I can answer in kind.

**12. How do you chunk, and why not fixed-size?**
Section-aware: split on the document's own `##` headings first, then pack into
chunks under a token budget, splitting at sentence boundaries. Fixed windows would
merge two schemes' eligibility rules into one chunk and make citations
meaningless. The budget is measured with the model's *own tokenizer* at 110 of its
128 tokens — because anything over the limit is silently truncated, so you'd
believe you indexed a document the model only partly saw. That's the kind of bug
that shows up as "retrieval is mysteriously bad" with no error anywhere.

### On eligibility

**13. Why not let the LLM decide eligibility?**
Three reasons. It's arithmetic and set membership — code is exactly right every
time, an LLM is approximately right most of the time. It has to be auditable:
"not eligible, your age 35 is above the maximum of 25" is a reason someone can
check and dispute; "the model said no" isn't. And it decides whether someone
applies for money they're entitled to. The model's only job is turning "I'm a 35
year old farmer earning 2 lakh" into fields.

**14. What if the user doesn't tell you something you need?**
That's `NEED_MORE_INFO`, a first-class third result. If a scheme has an income cap
and they never mentioned income, the honest answer is a question, not a guess.
There's a test for a subtle version: a profile saying "farmer" with no gender
stated must return NEED_MORE_INFO for a women-only scheme, not NOT_ELIGIBLE —
"farmer" says nothing about gender, and quietly assuming male would hide a scheme
a woman is entitled to.

**15. What can't your rule engine express?**
Conditional rules. Stand-Up India says "if the applicant is male, he must be
SC/ST" — a flat seven-field schema can't represent that, and forcing it into a
`gender` or `category` filter would produce confidently wrong answers. So those
conditions go into `unmodelled_conditions` verbatim and are always shown to the
user as "still to verify". ELIGIBLE in my system means "passes every rule I can
mechanically check", never "you will get this benefit" — and the UI says so.

### On data

**16. Where does the data come from, and how do I know it's real?**
All 15 schemes come from the official myScheme portal, run by Digital India
Corporation under MeitY. The portal is a JavaScript app, so fetching the page
returns an empty shell — I found the public JSON API its own frontend calls and
read that. Every raw response is committed to `data/raw/`, so any fact is
traceable to a response I actually received. That's a checkable property, not a
promise. Where the official page says nothing, the file says "Not specified in
official source" rather than filling the gap, and `validate_data.py` runs 10
checks including that every encoded rule cites the official sentence it came from.

**17. How do you handle data going stale?**
Imperfectly, and I'd call that the biggest real-world gap. Every document carries
`last_verified` and the UI shows it on every citation, so a user can judge the age
of what they're reading. But nothing re-checks the source automatically. The fix
is a scheduled re-fetch that diffs the official payload and flags changed figures
for review — the fetch is already one idempotent script, so it's mostly scheduling
plus a diff.

### On scale and cost

**18. What breaks if you go from 15 schemes to 1,000?**
Not the vector search — Chroma handles that volume fine. Two things break.
Retrieval precision: with 1,000 schemes, top-5 from one flat collection will mix
similar schemes, so I'd pre-filter by scheme or ministry using metadata before the
vector search, probably with a cheap classifier step. And the guardrail cost, which
is per-answer, becomes the dominant bill — so I'd cache claim verdicts keyed on
`(claim, chunk_id)`, since common questions repeat the same claims. The data
pipeline scales as-is; it's already a script.

**19. What does a question cost, and how do you know?**
Measured per request, not estimated: **1** LLM call for an out-of-scope question
(refused before generating), **4** for a verified answer, **5** when a repair
runs. Every call is logged to SQLite with prompt name, tokens and latency, and
`/metrics` reads that back — so the numbers survive a restart and come from real
measurements rather than in-memory counters.

**20. Why not fine-tune a model instead?**
Fine-tuning teaches style and format, not facts — and it would make the facts
*harder* to keep honest, because they'd be baked into weights with no citation and
no way to update when a scheme changes. My problem is grounding and freshness, and
retrieval solves both: I change a markdown file and the answer changes. I also
couldn't cite a source from a fine-tuned model, and citations are the feature.
Fine-tuning would make sense for the *judge*, trained on labelled
supported/unsupported pairs — that's a narrow, stable task where it would help.

**Bonus — why Chroma, why Docker, why CI?**
*Chroma:* zero setup, runs in-process, persists to a folder, and embeds in the
container. At this scale a hosted vector DB would be operational overhead for no
gain; the retriever interface is small enough to swap later.
*Docker:* the app needs Python 3.12, pinned libraries, a 120 MB model and a
pre-built index. Docker makes that one artifact that runs identically everywhere,
and it lets me build the index at image-build time so a broken corpus fails the
build instead of reaching users.
*CI:* 169 tests on every push catches the thing I didn't think to re-check. It
also validates the data and scans for leaked credentials before anything ships.

---

## Problems I faced, and how I fixed them

These all actually happened while building this. They're the most useful part of
this document in an interview — specific, debuggable, and each one has a lesson.

**1. The official website returned no data at all.**
`myscheme.gov.in/schemes/pm-kisan` fetched fine but contained no scheme
information — just a footer and a support email. It's a JavaScript single-page
app: the server sends an empty shell. I opened the page in a real browser and
watched the network tab, which showed it calling
`/api/apisetu/schemes?slug=pm-kisan&lang=en`. I read that API instead — same
official domain, far more reliable, and it returns the editors' own markdown.
*Lesson: when scraping returns nothing, look at what the page itself is calling.*

**2. Everything crashed on the first Hindi character.**
`UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d`. Windows defaults to
cp1252, which can't represent Devanagari or the rupee sign. I now pass
`encoding="utf-8"` explicitly on every file read and write, and set
`PYTHONIOENCODING=utf-8` in the Dockerfile and CI.
*Lesson: never rely on the platform default encoding for multilingual data.*

**3. HTML entities survived into the data — twice.**
Chunks contained `farmers&#39; families`. I added `html.unescape`, and some
*still* had entities, because a few fields are escaped twice
(`&amp;#39;` → `&#39;` → `'`). I now unescape in a loop until the string stops
changing, and `validate_data.py` fails the build if any entity survives.
*Lesson: assert the absence of the bug, don't just fix it once.*

**4. I nearly indexed a third of my documents invisibly.**
My embedding model truncates at 128 tokens. My first chunks were
character-budgeted, so longer ones were silently cut — I'd have believed the
documents were indexed while the model never saw most of them, with no error
anywhere and retrieval just quietly worse forever. I switched to measuring with
the model's own tokenizer, capped chunks at 110 tokens, and made ingestion exit
non-zero if any chunk is over budget. A test re-checks it with the real tokenizer.
*Lesson: silent truncation is worse than a crash. Assert the invariant.*

**5. My similarity threshold was comparing against a meaningless number.**
Chroma's default distance is *squared L2*, not cosine. My threshold was tuned for
cosine similarity, so the comparison was nonsense. Fixed by creating the
collection with `hnsw:space=cosine` explicitly and normalising embeddings, so
`score = 1 − distance` is genuinely cosine similarity.
*Lesson: check your vector store's default metric. Don't assume cosine.*

**6. The "better" embedding model was much worse for my actual need.**
`multilingual-e5-small` has 512-token context versus 128, so I nearly switched.
Then I measured: off-topic scored **0.70** against on-topic **0.81**, and a
non-indexed scheme also scored 0.80. There was no gap for a threshold to live in.
I kept the smaller-context model, which separates cleanly.
*Lesson: benchmark against the property you actually depend on, not the spec sheet.*

**7. Hinglish users would have been told their scheme didn't exist.**
Testing retrieval in all three languages, raw Hinglish scored **0.436** against
the correct passage — *below* my 0.45 refusal threshold — and retrieved the wrong
scheme. This wasn't a bug so much as a discovery that the query-normalisation step
was load-bearing rather than a nice-to-have, and it gave me the numbers to prove
it in `DECISIONS.md`.
*Lesson: test with the input your users will actually type, not the clean case.*

**8. Successful API calls returned empty answers.**
After wiring up the real key, calls returned HTTP 200, reported 10 completion
tokens, and gave me `''`. JSON mode failed with an empty `failed_generation`. The
cause: Groq's `openai/gpt-oss-*` are *reasoning* models that spend completion
tokens on hidden reasoning before emitting content — with `max_tokens=10`, 53
tokens of reasoning consumed the entire budget. Setting `reasoning_effort=low`
made it worse differently: the normalizer stopped translating and echoed the
Hinglish query back. I also found the Llama models 404 on this account. I switched
to `qwen/qwen3.8-27b`, which uses zero reasoning tokens, was fastest, scored 5/5
on a judge probe, and was the only candidate to label Hinglish correctly. I added
an explicit guard that names this cause instead of letting it surface as a JSON
parse error.
*Lesson: a 200 response is not a successful call. Validate the content.*

**9. A rate limit was silently turning good answers into refusals.**
A PM-KISAN question that had worked suddenly came back BLOCK with only 3 LLM
calls. The call log showed the judge call exhausting its retries on a 429: Groq's
free tier allows 8,000 tokens/minute and the error says *"Please try again in
1.68s"* — but my fixed exponential backoff was sleeping 2s, 4s, 8s and burning
attempts at the wrong times. Now the client parses that hint (and any
`Retry-After` header) and waits exactly as long as asked.
*Lesson: when a service tells you how long to wait, listen instead of guessing.
Also: the fail-closed design did the right thing — it refused rather than
shipping unverified — but the root cause was infrastructure, not the model.*

**10. A refusal was being badged as a verified answer.**
Running real queries, I asked about Sukanya Samriddhi — a scheme I don't index.
The model wrote its own prose refusal, so the claim splitter extracted *zero*
claims, `decide()` returned PASS ("nothing to verify"), and the UI showed
**✅ Verified** with citations to PM-Mudra. Technically consistent, completely
misleading. Vacuously true is not verified: zero claims on a drafted answer is now
a refusal, with my own wording rather than the model's (which leaked retrieval
internals by naming unrelated schemes).
*Lesson: "no evidence of a problem" is not "evidence of no problem". Check your
edge cases against real inputs, not just unit tests.*

**11. My own test was wrong, and the code was right.**
A test asserted a 35-year-old farmer should be NOT_ELIGIBLE for a women-only
scheme. It failed with NEED_MORE_INFO — because the profile never stated a gender,
and the engine correctly refused to infer one from "farmer". I'd written the
expectation carelessly. I fixed the test and added a second one pinning the
behaviour explicitly, since it's exactly the kind of inference that would hide a
scheme from someone entitled to it.
*Lesson: when a test fails, work out which side is wrong before changing either.*

**12. The deployment target stopped being free mid-project.**
`create_repo` returned **402 Payment Required**: Hugging Face now requires PRO for
Docker Spaces. I probed every SDK — only `static` is free, and the Streamlit SDK
no longer exists at all. Since a static Space can't run Python, no free HF Space
could host this. I also measured the app at ~1.2 GB RSS, which ruled out the
512 MB free tiers. I moved to Streamlit Community Cloud, which is free and needs
no card — but it gives you one process, and I didn't want to collapse the
architecture to fit. So the root entrypoint starts the real FastAPI app in a
background thread and the UI talks to it over HTTP on localhost. The boundary is
unchanged, the Dockerfile is still the production path, and the eval still
measures the same code. I also had to make the app build its index at startup,
since that host has no build step.
*Lesson: infrastructure assumptions expire. Keep the architecture portable so a
host change is a deployment change, not a rewrite.*

---

## Resume bullets

Use the real numbers from `eval/results.md`. Placeholders marked `<...>`.

**1.**
> Built and deployed a hallucination-aware RAG assistant for Indian government
> welfare schemes (FastAPI, ChromaDB, Streamlit, Docker) serving 15 central
> schemes across 312 indexed passages in English, Hindi and Hinglish; a
> claim-level verification layer cut the unsupported-claim rate in delivered
> answers from `<OFF>%` to `<ON>%` on a 50-question benchmark.

**2.**
> Designed a four-stage LLM guardrail — atomic claim extraction, evidence-only
> LLM judging, Python-side citation validation, and a deterministic decision
> policy — raising correct refusal of unanswerable questions to `<ON>%` while
> holding over-blocking at `<X>%`, at a measured cost of 4 LLM calls per answer.

**3.**
> Engineered a rule-based eligibility engine (pure Python, 169 automated tests,
> zero network calls in CI) that returns an auditable per-rule reason and an
> explicit NEED_MORE_INFO state instead of guessing, with every rule traceable to
> the official source sentence it was derived from; calibrated the retrieval
> refusal threshold on 22 measured queries and shipped via GitHub Actions CI/CD.

### Interchangeable phrasings

- *"reduced ungrounded claims by X percentage points"*
- *"designed for a deliberate precision/recall trade-off, measured in both
  directions"*
- *"end-to-end ownership: data acquisition from official APIs, retrieval tuning,
  evaluation harness, containerisation, CI/CD, deployment"*

---

## Questions to ask them

Signals seniority, and genuinely useful:

1. "How do you currently evaluate LLM output quality — human review, automated
   judges, or production signals?"
2. "When a model-based feature gets something wrong in production, how do you
   find out? What's the feedback loop?"
3. "Where do you draw the line between what a model decides and what code
   decides?"
4. "How do you handle knowledge going stale in retrieval systems?"

---

## Things to be honest about if asked

Never oversell these — being straight about them reads as senior.

- **The judge is the same model family as the generator**, so a shared blind spot
  would be invisible to my evaluation. The fix is a different judge model plus a
  human-labelled subset.
- **50 eval questions is small**; differences of a few percent are noise, and I
  wrote the golden set myself, so it may under-represent failure modes I didn't
  think of. Every expected fact is at least verified to exist in the source data.
- **Only 15 schemes.** India has hundreds of central schemes and thousands of
  state ones. Two very well-known ones (Ayushman Bharat PM-JAY, Sukanya Samriddhi)
  are missing because I couldn't fetch them from an official source — and rather
  than write them from memory, I left them out and turned them into test cases.
- **Key-fact matching in the eval is substring-based**, so a correct answer that
  phrases a number differently scores as a miss.
- **I have not load-tested it.** Single container, single worker.
- **The live demo is not the Docker container** — it's the same code in one
  process, because the intended host stopped being free.
