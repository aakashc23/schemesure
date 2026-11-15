"""
Disk cache for the query-normalizer LLM call, shared by every eval script.

WHY
Normalizing a question (translate to English + detect language) costs one LLM
call. The evaluation needs the result in three places: the retrieval benchmark,
the guardrail-OFF run and the guardrail-ON run. Without a cache the same
question is paid for three times -- on Groq's free tier (a daily token cap) that
is the difference between one sitting and three days.

It also makes the experiment FAIRER: OFF and ON now retrieve with the identical
normalized query, so any difference between them comes from the guardrail alone,
not from the normalizer answering slightly differently on each call.

CACHE KEY
sha1(question) + llm_model + sha1(normalizer system prompt). Change the model or
edit the normalizer prompt and old entries stop matching (they stay in the file,
harmlessly). A fallback result (the LLM failed, so we used the raw question) is
NEVER cached: caching a degraded answer would make the degradation permanent.

The cache file is append-only JSON Lines under eval/cache/ (generated data; it is
safe to delete, the next run simply pays for the normalizer calls again).
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

from app.llm_client import CallCounter
from app.prompts import QUERY_NORMALIZER_SYSTEM
from app.schemas import Language, NormalizedQuery
from core.query_normalizer import normalize_query

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "normalized.jsonl"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cache_key(question: str, model: str, system_prompt: str = QUERY_NORMALIZER_SYSTEM) -> str:
    """Key that changes when the question, the model or the prompt changes."""
    return f"{_sha1(question.strip())}|{model}|{_sha1(system_prompt)}"


@dataclass
class NormalizationResult:
    """What the cache hands back to the eval scripts."""

    normalized: NormalizedQuery
    from_cache: bool
    latency_ms: int = 0       # how long the (original) LLM call took
    llm_calls: int = 0        # calls made NOW (0 on a cache hit)
    prompt_tokens: int = 0    # tokens of the original call (kept for cost reporting)
    completion_tokens: int = 0


class NormalizationCache:
    """Look up / fill normalized queries. `llm=None` makes it read-only."""

    def __init__(self, llm, model: str, path: Path | None = None) -> None:
        self.llm = llm
        self.model = model
        self.path = Path(path or DEFAULT_CACHE_PATH)
        self._entries: dict[str, dict] = {}
        self._load()

    # ---- persistence -----------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        with io.open(self.path, encoding="utf-8") as fh:
            for line in fh.read().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written last line must not kill the run
                if "key" in entry:
                    self._entries[entry["key"]] = entry

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with io.open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- the one public method ------------------------------------------

    def get(self, question: str) -> NormalizationResult:
        """Return the cached normalization, or call the LLM once and store it."""
        key = cache_key(question, self.model)
        hit = self._entries.get(key)
        if hit:
            return NormalizationResult(
                normalized=NormalizedQuery(
                    normalized_query=hit["normalized_query"],
                    language=Language(hit["language"]),
                    fallback_used=False,
                ),
                from_cache=True,
                latency_ms=int(hit.get("latency_ms", 0)),
                prompt_tokens=int(hit.get("prompt_tokens", 0)),
                completion_tokens=int(hit.get("completion_tokens", 0)),
            )

        counter = CallCounter()
        started = time.perf_counter()
        normalized = normalize_query(self.llm, question, counter=counter)
        latency_ms = int((time.perf_counter() - started) * 1000)
        prompt_tokens = int(getattr(counter, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(counter, "completion_tokens", 0) or 0)

        if not normalized.fallback_used:
            entry = {
                "key": key,
                "question_sha1": _sha1(question.strip()),
                "model": self.model,
                "prompt_sha1": _sha1(QUERY_NORMALIZER_SYSTEM),
                "normalized_query": normalized.normalized_query,
                "language": normalized.language.value,
                "fallback_used": False,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            self._entries[key] = entry
            self._append(entry)

        return NormalizationResult(
            normalized=normalized,
            from_cache=False,
            latency_ms=latency_ms,
            llm_calls=counter.count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
