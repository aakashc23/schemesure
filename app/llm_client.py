"""
The single door to the LLM, plus the SQLite log of every call through it.

WHY A WRAPPER INSTEAD OF CALLING openai DIRECTLY
------------------------------------------------
1. Observability. Every call is recorded (prompt name, model, tokens, latency,
   success/failure). Without this you cannot answer "how many LLM calls does one
   question cost?" — which is the first question anyone asks about a RAG system.
2. Retries. Groq's free tier returns 429 under load. One place to back off.
3. JSON discipline. Three of our four prompts must return parseable JSON. The
   repair-and-parse logic belongs in one place, not copy-pasted four times.

Groq speaks the OpenAI wire format, so the official `openai` client works with
nothing changed but `base_url`.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI

from app.config import Settings, get_settings

logger = logging.getLogger("schemesure.llm")

# A server-suggested wait longer than this is never honoured. Per-minute rate
# limits ask for at most a minute (Groq's free tier also caps OUTPUT tokens at
# 1,000/minute, and a judge call hitting that is told to wait ~25 s); a hint of
# minutes ("try again in 3m12s") means a daily quota, which no amount of waiting
# inside one HTTP request can clear. The per-request deadline still applies, so
# a wait that would overrun it is refused regardless of this cap.
MAX_SERVER_WAIT_SECONDS = 60.0

# The ONLY text about a provider failure that may reach a response, a report note
# or the UI. Raw provider messages carry organisation ids, tiers and usage
# figures, so they are logged server-side and never returned (audit finding L1).
PUBLIC_MESSAGES: dict[str, str] = {
    "rate_limit": "The language model is busy right now (rate limit reached).",
    "quota": "The language model's usage quota has been used up for now.",
    "auth": "The language model credentials were rejected by the provider.",
    "too_large": "The request was too large for the language model.",
    "timeout": "The language model did not respond in time.",
    "bad_output": "The language model returned a response that could not be used.",
    "upstream": "The language model service is temporarily unavailable.",
}


class LLMError(RuntimeError):
    """
    Raised when a call fails after all retries, or returns unusable output.

    `kind` classifies the failure so callers (and the UI) can react honestly:
      rate_limit | quota | auth | too_large | timeout | bad_output | upstream
    `str(exc)` and `public_message` are safe to show to users. `detail` holds
    the raw provider text for server-side logs only; never put it in a response.
    """

    def __init__(self, message: str, kind: str = "upstream", detail: str = "") -> None:
        super().__init__(message)
        self.kind = kind if kind in PUBLIC_MESSAGES else "upstream"
        self.detail = detail

    @property
    def public_message(self) -> str:
        return PUBLIC_MESSAGES[self.kind]


# ==========================================================================
# SQLite call log
# ==========================================================================
# One tiny file, two tables. SQLite is the right call here: zero setup, it ships
# inside the container, and "how many tokens did today cost?" becomes one SELECT.
# A write lock keeps concurrent FastAPI worker threads from tripping over
# each other.

_DB_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    prompt_name       TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    attempts          INTEGER NOT NULL DEFAULT 1,
    ok                INTEGER NOT NULL DEFAULT 1,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_prompt ON llm_calls(prompt_name);

-- Application-level events: guardrail decisions and answer outcomes. Keeping
-- them beside the call log means /metrics is a read of one file.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    kind        TEXT NOT NULL,   -- 'guardrail_decision' | 'answer_status'
    value       TEXT NOT NULL,   -- 'PASS' | 'BLOCK' | 'verified' | ...
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, value);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(settings: Settings | None = None) -> None:
    """Create the log tables if they do not exist. Safe to call repeatedly."""
    settings = settings or get_settings()
    with _DB_LOCK, _connect(settings.llm_log_path) as connection:
        connection.executescript(_SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_event(kind: str, value: str, detail: str = "", settings: Settings | None = None) -> None:
    """
    Record an application event (guardrail decision, answer status).

    Monitoring must never break answering, so failures here are swallowed.
    """
    settings = settings or get_settings()
    try:
        with _DB_LOCK, _connect(settings.llm_log_path) as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT INTO events (created_at, kind, value, detail) VALUES (?, ?, ?, ?)",
                (_now(), kind, value, detail[:500]),
            )
    except Exception:  # noqa: BLE001 - monitoring must never break answering
        pass


def _log_call(
    settings: Settings,
    prompt_name: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: int,
    attempts: int,
    ok: bool,
    error: str = "",
) -> None:
    try:
        with _DB_LOCK, _connect(settings.llm_log_path) as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                """
                INSERT INTO llm_calls (
                    created_at, prompt_name, model, prompt_tokens, completion_tokens,
                    total_tokens, latency_ms, attempts, ok, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(), prompt_name, model, prompt_tokens, completion_tokens,
                    prompt_tokens + completion_tokens, latency_ms, attempts,
                    1 if ok else 0, (error or None),
                ),
            )
    except Exception:  # noqa: BLE001
        # Logging must NEVER break a request or, worse, make the caller think
        # the LLM call failed and send it again (audit finding L2). Any failure
        # here (sqlite error, OSError from mkdir, ...) is swallowed.
        logger.warning("could not write the LLM call log", exc_info=True)


# ==========================================================================
# Result type
# ==========================================================================


@dataclass
class LLMResult:
    """One completed LLM call."""

    text: str
    prompt_name: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CallCounter:
    """
    Counts LLM calls made while serving one request.

    Exists because "LLM calls per question" is a headline metric in the
    evaluation, and the only honest way to get it is to count at the call site.
    """

    count: int = 0
    prompt_names: list[str] = field(default_factory=list)
    # Failed attempts (429s, timeouts, ...) are real cost too: they use up quota.
    failed_attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Optional monotonic-clock deadline for the whole request. complete() never
    # starts an attempt, or sleeps, past it.
    deadline: float | None = None

    def record(self, result: LLMResult) -> None:
        self.count += 1
        self.prompt_names.append(result.prompt_name)
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens

    def record_failure(self) -> None:
        self.failed_attempts += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def set_deadline(self, seconds: float) -> None:
        """Start the clock: this request has `seconds` left from now."""
        self.deadline = time.monotonic() + seconds

    def remaining(self) -> float | None:
        """Seconds left before the deadline, or None when there is no deadline."""
        if self.deadline is None:
            return None
        return self.deadline - time.monotonic()


# ==========================================================================
# JSON extraction
# ==========================================================================

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


def extract_json(text: str) -> Any:
    """
    Parse JSON out of an LLM response, tolerating the usual decorations.

    Models wrap JSON in ``` fences or add a sentence of preamble even when told
    not to. Rather than failing the request, we strip the wrapper and retry,
    and only then give up. This is pure string handling — no second LLM call.
    """
    if not text or not text.strip():
        raise LLMError("empty response from LLM", kind="bad_output")

    candidate = text.strip()

    # 1. Straight parse.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 2. Strip a ```json ... ``` fence.
    fenced = _FENCE_RE.match(candidate)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            candidate = fenced.group(1).strip()

    # 3. Take the outermost {...} or [...] span.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue

    # The model's text goes to `detail` (server logs), not into the message.
    raise LLMError(
        "could not parse JSON from the model response",
        kind="bad_output",
        detail=text[:200],
    )


# ==========================================================================
# The client
# ==========================================================================

# Markers in a provider message that mean "this can never succeed on retry".
_QUOTA_MARKERS = ("per day", "(tpd)", "(rpd)", "tokens per day", "requests per day")
_TOO_LARGE_MARKERS = (
    "request too large", "reduce your message size", "context length",
    "context_length", "maximum context",
)


def classify_error(exc: Exception) -> tuple[str, bool]:
    """
    Decide (kind, retryable) for a failed provider call.

    Classified by exception TYPE and HTTP STATUS CODE, not by hunting for
    substrings like "500" in the message (audit finding H3: "500" also matches
    "Limit 7500"). Message text is consulted only to tell a daily quota or an
    oversized request apart from an ordinary per-minute 429, because those two
    can never succeed on retry.
    """
    text = str(exc).lower()

    # Timeouts first: APITimeoutError is also an APIConnectionError, and the
    # builtin TimeoutError is also an OSError.
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return "timeout", True
    if isinstance(exc, (APIConnectionError, ConnectionError)):
        return "upstream", True

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if any(marker in text for marker in _TOO_LARGE_MARKERS) or status == 413:
            return "too_large", False
        if status == 429:
            if any(marker in text for marker in _QUOTA_MARKERS):
                return "quota", False
            return "rate_limit", True
        if status in (401, 403):
            return "auth", False
        if status == 408:
            return "timeout", True
        if status == 409 or status >= 500:
            return "upstream", True
        return "upstream", False  # any other 4xx: our request is wrong, do not retry

    if isinstance(exc, OSError):
        return "upstream", True
    # Unknown exception type: do not blindly retry it.
    return "upstream", False


# "Please try again in 1.68s" / "try again in 2m30.5s" / Retry-After: 3
_RETRY_HINT_RE = re.compile(
    r"try again in\s+(?:(\d+)m)?([\d.]+)s", re.I
)
_RETRY_AFTER_HEADER_RE = re.compile(r"retry-after['\"]?\s*[:=]\s*['\"]?([\d.]+)", re.I)


def _retry_after_seconds(exc: Exception) -> float | None:
    """
    Extract how long the server asked us to wait, if it said.

    Groq returns the wait in the 429 message body rather than only in a header,
    so we parse both. Returns None when there is no usable hint.
    """
    # A Retry-After header, when the SDK surfaces the response.
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw:
                return float(raw)
        except (TypeError, ValueError):
            pass

    text = str(exc)
    match = _RETRY_HINT_RE.search(text)
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds

    match = _RETRY_AFTER_HEADER_RE.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None

    return None


class LLMClient:
    """Thin, logged, retrying wrapper over the OpenAI-compatible chat API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.require_llm_key()
        self._client = OpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout_seconds,
            # We implement our own backoff so that every attempt is logged.
            max_retries=0,
        )
        init_db(self.settings)

    # ---- core call ------------------------------------------------------

    def complete(
        self,
        prompt_name: str,
        system: str,
        user: str,
        json_mode: bool = False,
        max_tokens: int | None = None,
        counter: CallCounter | None = None,
    ) -> LLMResult:
        """
        Run one chat completion.

        `prompt_name` is the label that shows up in the call log and /metrics,
        so pass something meaningful ("judge_claims", not "call3").
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        if json_mode:
            # Server-side JSON guarantee where the model supports it. We still
            # parse defensively — see extract_json.
            kwargs["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        max_attempts = max(1, self.settings.llm_max_retries)

        response = None
        error: LLMError | None = None
        raw_error = ""            # provider text: server-side log only
        attempt = 0

        while attempt < max_attempts:
            attempt += 1

            # Respect the per-request deadline: never begin an attempt after it,
            # and never let one run longer than the time that is left.
            remaining = counter.remaining() if counter is not None else None
            if remaining is not None and remaining <= 0:
                error = LLMError(
                    f"LLM call '{prompt_name}' not attempted: request deadline reached",
                    kind="timeout",
                )
                break
            call_timeout = self.settings.llm_timeout_seconds
            if remaining is not None:
                call_timeout = max(1.0, min(call_timeout, remaining))

            try:
                # httpx applies each limit PER PHASE, so a bare number could let a
                # slow connect plus a slow first byte take ~2x the budget. Split
                # the budget instead: connect + read never exceed call_timeout.
                connect_limit = min(5.0, call_timeout / 2)
                response = self._client.chat.completions.create(
                    **kwargs,
                    timeout=httpx.Timeout(
                        connect=connect_limit,
                        read=max(0.5, call_timeout - connect_limit),
                        write=connect_limit,
                        pool=connect_limit,
                    ),
                )
                break
            except Exception as exc:  # noqa: BLE001 - classify, then retry or raise
                kind, retryable = classify_error(exc)
                raw_error = f"{type(exc).__name__}: {exc}"
                error = LLMError(
                    f"LLM call '{prompt_name}' failed ({kind})", kind=kind,
                    detail=raw_error,
                )
                if counter is not None:
                    counter.record_failure()
                logger.warning("LLM call %s attempt %d failed (%s): %s",
                               prompt_name, attempt, kind, raw_error[:300])

                if not retryable or attempt >= max_attempts:
                    break

                # Prefer the server's own advice over guessing. Groq's per-minute
                # 429 says "Please try again in 1.68s", so sleeping exactly that
                # long clears it. A hint longer than MAX_SERVER_WAIT_SECONDS means
                # a quota that will not clear inside one request: give up now
                # instead of sleeping for minutes (audit finding H3).
                hint = _retry_after_seconds(exc)
                if hint is not None and hint > MAX_SERVER_WAIT_SECONDS:
                    break
                delay = hint if hint is not None else float(2**attempt)  # 2s, 4s, 8s
                delay = min(delay + 0.5, MAX_SERVER_WAIT_SECONDS)

                remaining = counter.remaining() if counter is not None else None
                if remaining is not None and delay >= remaining:
                    # Sleeping would run past the deadline.
                    error = LLMError(
                        f"LLM call '{prompt_name}' gave up: retry would pass the "
                        "request deadline",
                        kind="timeout", detail=raw_error,
                    )
                    break
                time.sleep(delay)
                continue

        latency_ms = int((time.perf_counter() - started) * 1000)

        # ---- the call returned: read it --------------------------------------
        if response is not None:
            # A provider can answer 200 with an empty `choices` list. Treat that
            # like empty content (kind="bad_output", logged) instead of letting an
            # IndexError escape as an unexplained 500.
            choice = response.choices[0] if getattr(response, "choices", None) else None
            message = getattr(choice, "message", None)
            text = (getattr(message, "content", None) or "").strip()
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

            # Empty content despite a successful HTTP call. This really happens,
            # and the cause is worth naming: a *reasoning* model (Groq's
            # openai/gpt-oss-*) spends completion tokens on hidden reasoning
            # before emitting any content, so a small max_tokens budget is
            # consumed entirely by reasoning and the content comes back as "".
            # Without this guard the caller sees an empty answer or an
            # unexplained JSON parse failure instead of the real cause.
            if text:
                # Logging happens AFTER the call succeeded and cannot raise, so a
                # logging problem can never trigger a duplicate LLM request.
                _log_call(
                    self.settings, prompt_name, self.settings.llm_model,
                    prompt_tokens, completion_tokens, latency_ms, attempt, ok=True,
                )
                result = LLMResult(
                    text=text,
                    prompt_name=prompt_name,
                    model=self.settings.llm_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    attempts=attempt,
                )
                if counter is not None:
                    counter.record(result)
                return result

            raw_error = (
                f"model '{self.settings.llm_model}' returned empty content "
                f"(finish_reason={getattr(choice, 'finish_reason', '?')}, "
                f"completion_tokens={completion_tokens}). If this is a reasoning "
                "model, raise LLM_MAX_TOKENS or use a non-reasoning model."
            )
            error = LLMError(
                f"LLM call '{prompt_name}' returned empty content",
                kind="bad_output", detail=raw_error,
            )
            if counter is not None:
                counter.record_failure()
            logger.warning("LLM call %s: %s", prompt_name, raw_error)
            _log_call(
                self.settings, prompt_name, self.settings.llm_model,
                prompt_tokens, completion_tokens, latency_ms, attempt,
                ok=False, error=raw_error,
            )
            raise error

        # ---- every attempt failed -------------------------------------------
        assert error is not None  # the loop only exits without a response on error
        _log_call(
            self.settings, prompt_name, self.settings.llm_model,
            0, 0, latency_ms, attempt, ok=False,
            error=(raw_error or error.detail or str(error)),
        )
        raise error

    # ---- JSON convenience ----------------------------------------------

    def complete_json(
        self,
        prompt_name: str,
        system: str,
        user: str,
        max_tokens: int | None = None,
        counter: CallCounter | None = None,
    ) -> tuple[Any, LLMResult]:
        """Run a call that must return JSON, and return (parsed, raw_result)."""
        result = self.complete(
            prompt_name, system, user, json_mode=True,
            max_tokens=max_tokens, counter=counter,
        )
        return extract_json(result.text), result


# ==========================================================================
# Metrics read-back
# ==========================================================================


def read_metrics(settings: Settings | None = None) -> dict[str, Any]:
    """
    Aggregate the call log for GET /metrics.

    Reads the same SQLite file the client writes to, so the numbers are real
    measurements rather than in-memory counters that reset on restart.
    """
    settings = settings or get_settings()
    empty: dict[str, Any] = {
        "llm_calls_total": 0, "llm_calls_failed": 0,
        "avg_latency_ms": 0.0, "p95_latency_ms": 0.0,
        "total_prompt_tokens": 0, "total_completion_tokens": 0,
        "calls_by_prompt": {},
        "guardrail_counts": {"PASS": 0, "REPAIR": 0, "BLOCK": 0, "SKIPPED": 0},
        "questions_answered": 0, "questions_refused": 0,
        "by_prompt": {}, "answer_status_counts": {}, "cache_hits": 0,
        "log_started_at": None,
    }

    try:
        with _DB_LOCK, _connect(settings.llm_log_path) as connection:
            connection.executescript(_SCHEMA)

            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failed,
                       AVG(CASE WHEN ok = 1 THEN latency_ms END) AS avg_ms,
                       SUM(prompt_tokens) AS ptok,
                       SUM(completion_tokens) AS ctok
                FROM llm_calls
                """
            ).fetchone()

            metrics = dict(empty)
            metrics["llm_calls_total"] = int(row["total"] or 0)
            metrics["llm_calls_failed"] = int(row["failed"] or 0)
            metrics["avg_latency_ms"] = round(float(row["avg_ms"] or 0.0), 1)
            metrics["total_prompt_tokens"] = int(row["ptok"] or 0)
            metrics["total_completion_tokens"] = int(row["ctok"] or 0)

            # p95 by index into the sorted successful latencies. Exact rather
            # than interpolated — the row counts here are small.
            latencies = [
                int(r["latency_ms"])
                for r in connection.execute(
                    "SELECT latency_ms FROM llm_calls WHERE ok = 1 ORDER BY latency_ms"
                )
            ]
            if latencies:
                index = min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))
                metrics["p95_latency_ms"] = float(latencies[index])

            metrics["calls_by_prompt"] = {
                r["prompt_name"]: int(r["n"])
                for r in connection.execute(
                    "SELECT prompt_name, COUNT(*) AS n FROM llm_calls GROUP BY prompt_name"
                )
            }

            counts = dict(empty["guardrail_counts"])
            for r in connection.execute(
                "SELECT value, COUNT(*) AS n FROM events "
                "WHERE kind = 'guardrail_decision' GROUP BY value"
            ):
                if r["value"] in counts:
                    counts[r["value"]] = int(r["n"])
            metrics["guardrail_counts"] = counts

            metrics["by_prompt"] = {
                r["prompt_name"]: {
                    "calls": int(r["calls"]),
                    "failed": int(r["failed"] or 0),
                    "avg_latency_ms": round(float(r["avg_ms"] or 0.0), 1),
                    "prompt_tokens": int(r["ptok"] or 0),
                    "completion_tokens": int(r["ctok"] or 0),
                }
                for r in connection.execute(
                    """
                    SELECT prompt_name,
                           COUNT(*) AS calls,
                           SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS failed,
                           AVG(CASE WHEN ok = 1 THEN latency_ms END) AS avg_ms,
                           SUM(prompt_tokens) AS ptok,
                           SUM(completion_tokens) AS ctok
                    FROM llm_calls GROUP BY prompt_name
                    """
                )
            }

            status_counts: dict[str, int] = {}
            for r in connection.execute(
                "SELECT value, COUNT(*) AS n FROM events "
                "WHERE kind = 'answer_status' GROUP BY value"
            ):
                status_counts[r["value"]] = int(r["n"])
                if r["value"] == "refused":
                    metrics["questions_refused"] = int(r["n"])
                elif r["value"] == "service_unavailable":
                    # An outage is neither an answer nor an honest refusal; it is
                    # counted on its own (answer_status_counts) so it cannot
                    # inflate either number.
                    pass
                else:
                    metrics["questions_answered"] += int(r["n"])
            metrics["answer_status_counts"] = status_counts

            hits = connection.execute(
                "SELECT COUNT(*) AS n FROM events WHERE kind = 'cache' AND value = 'hit'"
            ).fetchone()
            metrics["cache_hits"] = int(hits["n"] or 0)

            first = connection.execute(
                """
                SELECT MIN(ts) AS first_ts FROM (
                    SELECT MIN(created_at) AS ts FROM llm_calls
                    UNION ALL
                    SELECT MIN(created_at) AS ts FROM events
                )
                """
            ).fetchone()
            metrics["log_started_at"] = first["first_ts"] if first and first["first_ts"] else None

            return metrics
    except sqlite3.Error:
        return empty
