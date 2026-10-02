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
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import Settings, get_settings


class LLMError(RuntimeError):
    """Raised when a call fails after all retries, or returns unusable output."""


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
    except sqlite3.Error:
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
    except sqlite3.Error:
        pass  # never let logging break a request


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

    def record(self, result: LLMResult) -> None:
        self.count += 1
        self.prompt_names.append(result.prompt_name)


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
        raise LLMError("empty response from LLM")

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

    raise LLMError(f"could not parse JSON from response: {text[:200]!r}")


# ==========================================================================
# The client
# ==========================================================================

# Error substrings worth retrying. Anything else (bad key, unknown model) is a
# permanent failure and retrying just wastes the user's time.
_RETRYABLE = (
    "rate limit", "rate_limit", "429", "timeout", "timed out", "connection",
    "temporarily", "503", "502", "500", "overloaded", "try again",
)


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

        last_error = ""
        started = time.perf_counter()

        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                latency_ms = int((time.perf_counter() - started) * 1000)

                text = (response.choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

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

            except Exception as exc:  # noqa: BLE001 - classify, then re-raise
                last_error = f"{type(exc).__name__}: {exc}"
                retryable = any(hint in str(exc).lower() for hint in _RETRYABLE)
                if retryable and attempt < self.settings.llm_max_retries:
                    # Exponential backoff: 2s, 4s, 8s. Groq's free tier is
                    # per-minute, so waiting really does clear a 429.
                    time.sleep(2**attempt)
                    continue
                break

        latency_ms = int((time.perf_counter() - started) * 1000)
        _log_call(
            self.settings, prompt_name, self.settings.llm_model,
            0, 0, latency_ms, self.settings.llm_max_retries, ok=False, error=last_error,
        )
        raise LLMError(f"LLM call '{prompt_name}' failed: {last_error}")

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

            for r in connection.execute(
                "SELECT value, COUNT(*) AS n FROM events "
                "WHERE kind = 'answer_status' GROUP BY value"
            ):
                if r["value"] == "refused":
                    metrics["questions_refused"] = int(r["n"])
                else:
                    metrics["questions_answered"] += int(r["n"])

            return metrics
    except sqlite3.Error:
        return empty
