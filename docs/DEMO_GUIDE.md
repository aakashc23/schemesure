# Demo guide

A script for presenting SchemeSure in 10–12 minutes, with the exact inputs to type,
what the app shows, and what is happening underneath. The outputs below were
recorded on **2026-11-16** against the real model (`qwen/qwen3.8-27b` on Groq); the
model runs at temperature 0, but wording can still vary slightly between runs.
Statuses and refusals should not.

**Live app:** <https://schemesureaakash.streamlit.app>

---

## Before the demo

| When | Do this | Why |
|---|---|---|
| The day before | Make sure the demo's Groq key has not been used for evaluation runs that day | Evaluation and demo share the free daily quota (~200k tokens). A spent quota makes every question answer "language model unavailable". |
| 10 minutes before | Open the live app and wait for it to load | Streamlit Cloud sleeps idle apps; a cold start loads the model and builds the index (up to ~2 minutes) |
| 10 minutes before | Sidebar: API reachable, 15 schemes, 312 passages, LLM key set | Confirms the backend and index are healthy |
| 10 minutes before | Click the first example once | Warms the model and fills the answer cache, so the live run of that question is instant |
| Fallback | If the hosted app is down: `docker build -t schemesure .` then `docker run -p 7860:7860 --env-file .env schemesure` locally (build before the demo; see docs/DEPLOYMENT.md), or run the API and UI locally (README, *Local setup*) | Same code, no dependency on the host |
| Fallback | If the LLM quota is gone: use the **Eligibility → Fill in a form** mode and the **Schemes** tab, which make no LLM call | Shows the rule engine and data provenance anyway |

Do not ask more than about two questions a minute: the free tier also limits output
tokens per minute, and the app will (honestly) report the model as busy.

---

## The story in one sentence

> "A normal RAG chatbot answers confidently even when its evidence is about the wrong
> scheme. SchemeSure checks every claim in its answer against the official text before
> showing it, and the parts that must not be wrong are plain Python, not the LLM."

---

## Part 1 — A verified answer (English) · 2 min

**Click:** *PM-KISAN amount (English)*
**Input:** `How much money does PM-KISAN give per year?`

If you asked this during the warm-up, it now comes from the answer cache (0 LLM
calls, "served from cache" in the meta line). That is fine to show, but say so.

Recorded output:

> **Verified** · English · 4 LLM calls
> PM-KISAN provides a financial benefit of Rs. 6000 per annum per family [1].
>
> Source: Pradhan Mantri Kisan Samman Nidhi — `pm_kisan#benefits#0`

| Under the hood | Value |
|---|---|
| Best retrieval score | 0.574 (threshold 0.45) — the PM-KISAN benefits passage ranks first |
| Claims | 1 claim, SUPPORTED by `pm_kisan#benefits#0` (similarity 0.77) |
| Number check | "6000" found in the passage |
| Decision | PASS |

**Then open the three tabs under the answer:**

- **Verification** — each claim, SUPPORTED, and the passage id it was checked
  against. Point out the line saying the similarity column is recorded only.
- **Evidence** — five retrieved passages with scores; the threshold is 0.45; the
  PM-KISAN benefits passage is marked *cited*.
- **Pipeline** — normalise → retrieve → generate → split claims → judge → similarity (recorded only, 0 calls), each with its duration and LLM calls (4 in total).

**Say:** "The model wrote the answer and the model judged the claims, but whether the
cited passage exists, whether every number appears in the source, and the final
publish/repair/refuse decision are Python."

## Part 2 — Hindi and Hinglish · 2 min

**Click:** *Atal Pension age limit (Hindi)*
**Input:** `अटल पेंशन योजना में शामिल होने की आयु सीमा क्या है?`

Recorded output:

> **Verified** · Hindi · 4 LLM calls
> अटल पेंशन योजना में शामिल होने की न्यूनतम आयु 18 वर्ष और अधिकतम आयु 40 वर्ष है [1]।

| Under the hood | Value |
|---|---|
| Normalised query | "What is the age limit for joining the Atal Pension Yojana?" |
| Best retrieval score | 0.876 — `atal_pension_yojana#eligibility#0` |
| Claims | 2 (minimum 18, maximum 40), both SUPPORTED by that passage |
| Decision | PASS |

The claims are checked in English against the English passage; the answer is shown
in Hindi.

**Click:** *Ujjwala documents (Hinglish)*
**Input:** `Ujjwala yojana ke liye kya documents chahiye?`

Recorded output — a **repair**, which is worth showing:

> **Partially verified** · Hinglish · 7 LLM calls
> Pradhan Mantri Ujjwala Yojana ke liye bank account number aur IFSC code ki
> zaroorat hai [4]. Passport size photograph bhi submit karna hota hai [4]. Assam aur
> Meghalaya mein Aadhaar card mandatory nahi hai [4].

| Under the hood | Value |
|---|---|
| Draft claims | 6; 5 SUPPORTED, 1 NOT_SUPPORTED: "An Aadhaar card is required" |
| Decision | REPAIR (1/6 unsupported, below the 0.5 block threshold) |
| Re-check of the repaired text | 5/5 SUPPORTED → PASS → shown as *Partially verified* |
| Pipeline | normalise, retrieve, generate, split, judge, similarity, repair, split, judge, similarity |

**Be ready to explain this one honestly.** The official list says *"Aadhaar (not
mandatory for Assam and Meghalaya)"*. The claim splitter turned that into an
unconditional "Aadhaar is required", the judge rejected it as stated, and the repair
dropped Aadhaar from the list. The result is safe but strict, and less complete than
the source. The answer also misses Photo ID, ration card and address proof, which
sit in a second "Documents Required" passage that was not retrieved. Both effects
are documented in [`TEST_REPORT.md` §3](TEST_REPORT.md#3-end-to-end-evaluation-preliminary)
and motivate the second-half work (human labels, passage-level retrieval metrics).

**Say:** "The question is translated to English for search — that is measured: on our
50-question benchmark, Hindi Recall@1 went from 2/4 to 4/4 with it — but the answer
comes back in the user's language."

## Part 3 — The trap: a real scheme we do not have · 2 min

**Click:** *Trick: Sukanya Samriddhi interest rate*
**Input:** `What is the interest rate on Sukanya Samriddhi Yojana?`

Recorded output:

> **Not enough official info** · 3 LLM calls
> I do not have official information on this. SchemeSure only answers from the
> official scheme documents it has indexed, and nothing in them covers your question.
> Please check myscheme.gov.in or the scheme's own official website.

| Under the hood | Value |
|---|---|
| Best retrieval score | **0.707** — a PM Mudra passage, far above the 0.45 threshold |
| Draft | asserted no checkable fact about Sukanya Samriddhi |
| Decision | refusal reason `no_claims`: a draft with nothing verifiable is not shown as "verified" |

**Open Evidence:** the best score is well above the 0.45 threshold, because savings
and interest passages from other schemes look similar.

**Say:** "This is the case a retrieval threshold cannot catch. On the full benchmark,
fake and unindexed schemes score 0.43–0.71, overlapping the 0.53–0.88 range of real questions, and the
threshold stops only 1 of 9 of them. The claim check is what stops it."

## Part 4 — Off-topic · 30 s

**Click:** *Off-topic: capital of France*

Recorded output: **Not enough official info**, 1 LLM call. Best retrieval score
0.235, below 0.45, so the pipeline stops after retrieval. The Pipeline tab shows
only *normalise* and *retrieve*.

**Say:** "Below the threshold, so nothing is generated: one LLM call, for the
translation step."

## Part 5 — Eligibility, decided by code · 2 min

### Free text (1 LLM call)

**Eligibility → Describe yourself**
**Input:** `I am a 35 year old kisan from Bihar, OBC, earning about 1.5 lakh a year.`

Recorded output (1 LLM call, then Python):

| Extracted field | Value |
|---|---|
| age | 35 |
| annual_income | 150000 ("1.5 lakh") |
| state | Bihar |
| occupation | farmer ("kisan", mapped by a synonym table in Python) |
| category | OBC |
| gender | *not provided* — not guessed from "kisan" |

Result: **10 ELIGIBLE**, **2 NEED_MORE_INFO**, **3 NOT_ELIGIBLE**.

- NEED_MORE_INFO: PM Matru Vandana and PM Ujjwala — "This scheme is only for
  applicants of gender: female. Your gender was not provided." The system does not
  infer gender, so it does not hide a scheme from a woman farmer.
- NOT_ELIGIBLE: NMMSS (students), PM SVANidhi (street vendors), PM Vishwakarma
  (artisans), each with the occupation rule as the reason.

**Show:** the "What we understood" form. The LLM only filled these fields; change
the age to 45 and click **Re-check** — no LLM call, and Atal Pension Yojana flips to
NOT_ELIGIBLE ("Your age 45 is above the maximum age of 40").

### Form mode (0 LLM calls)

**Eligibility → Fill in a form:** age 28, occupation *street vendor*, gender *female*,
everything else empty.

Recorded result (Python only, deterministic):

| Status | Schemes |
|---|---|
| ELIGIBLE (9) | Atal Pension, PM SVANidhi, PMJJBY, PMKVY, PM Matru Vandana, PM Mudra, PMSBY, PM Ujjwala, Stand-Up India |
| NEED_MORE_INFO (1) | PMAY-Urban — "Annual income must not exceed Rs 18,00,000, but your income was not provided." |
| NOT_ELIGIBLE (5) | KCC, PM Fasal Bima, PM-KISAN (for farmers), PM Vishwakarma (artisans), NMMSS (students) |

**Then** change occupation to *self employed*, age 45: every occupation-restricted scheme (KCC, PM Fasal Bima, PM-KISAN, PM SVANidhi, PM Vishwakarma, NMMSS) becomes **NEED_MORE_INFO** ("'self_employed' is a broad term, so please tell us your exact work") rather than a wrong "not eligible", and Atal Pension becomes NOT_ELIGIBLE (age above 40). Result: 7 ELIGIBLE / 7 NEED_MORE_INFO / 1 NOT_ELIGIBLE.

**Say:** "ELIGIBLE means 'passes every rule we can check mechanically'. Each scheme
also lists the conditions it cannot check — for example PMAY-Urban's income bands —
under *still to check yourself*, with the official sentence for every rule."

## Part 6 — Transparency tabs · 1 min

- **Schemes** → pick *PM-KISAN*: sections, official links, the encoded rules next to
  the official sentences they came from, and the number of indexed passages.
- **System**: real LLM call counts and latencies per prompt, guardrail decisions and
  cache hits since the app started.
- **How it works**: the pipeline diagrams, plus a step-by-step description filled in with the live models, top-k, score threshold and block threshold from /health.

## Part 7 — Close with the honest part · 1 min

**Say:**

- "Retrieval is benchmarked in full: Recall@5 is 97% on 35 answerable questions."
- "The guardrail OFF-versus-ON comparison has run only on a preliminary subset,
  because the free tier allows about 200k tokens a day; the full run and a
  human-labelled check of the judge are the first items of the second half."
- "The judge is the same model as the generator, which is a known threat to
  validity; a cross-model judge and an NLI verifier are planned."

---

## Likely questions and short answers

| Question | Answer |
|---|---|
| Why not just trust RAG? | Retrieval always returns something. For the Sukanya question it returns other schemes' interest passages at a high score; a plain RAG answer would be built from them. |
| What if the judge is wrong? | Then a wrong claim can pass. That is why citation existence and numbers are checked in Python too, and why human labels are the next step. |
| Why an LLM judge and not NLI, as in the plan? | Hindi/Hinglish-derived claims and paraphrases, and memory limits on the free host. NLI is planned as a comparison, not dropped. |
| Why does it refuse so much? | It is designed to prefer refusing to being wrong. The cost of each is in the README. Over-blocking is measured. |
| How much does a question cost? | 4 LLM calls (about 2,600–4,200 tokens, ~3,000 on average in the preliminary run) normally; 1 for off-topic; 7 if repaired; 0 if cached. |
| Can it handle all Indian schemes? | No: 15 central schemes now, 25–30 planned. The architecture (metadata per passage, rules per scheme) extends; the data collection is the work. |
