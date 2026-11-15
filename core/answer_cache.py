"""
A small in-memory answer cache for POST /ask.

WHY IT EXISTS
-------------
The deployed demo runs on a free LLM tier with a fixed daily token budget, and one
answered question costs several thousand tokens. A handful of curious visitors
clicking the same example questions could use the whole day's quota and leave the
demo showing "service unavailable" to everyone. Serving an identical repeated
question from memory costs zero tokens and zero seconds.

WHAT IT IS (AND IS NOT)
-----------------------
- TTL + LRU: an entry expires after `ttl_seconds`, and when the cache is full the
  least-recently-used entry is dropped. An OrderedDict plus a Lock is all this
  needs; there is no third-party dependency.
- It lives in the API layer only (app/main.py). `AnswerPipeline` never touches it,
  so evaluation runs are never served from a cache and always measure real work.
- Outage answers are never stored: a "please retry" message must not be repeated
  to the next visitor after the service has recovered.
- Setting `max_entries` (or `ttl_seconds`) to 0 disables it completely.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from app.schemas import AskResponse, RefusalReason


def make_key(question: str, language: str | None, guardrail_enabled: bool) -> tuple[str, str, bool]:
    """
    The cache key for one /ask request.

    The question is lowercased with whitespace collapsed, so "How much does
    PM-KISAN pay?" and "  how much does  pm-kisan pay? " share an entry. The
    language override is part of the key (the answer is written in that language)
    and so is the guardrail flag (a guardrail-off answer is not a verified one).
    """
    normalized = re.sub(r"\s+", " ", (question or "").strip()).lower()
    return (normalized, language or "", bool(guardrail_enabled))


class AnswerCache:
    """Thread-safe TTL + LRU cache of finished AskResponse objects."""

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock            # injectable so tests can move time
        self._lock = threading.Lock()
        # key -> (expires_at, response); order = least- to most-recently used.
        self._entries: OrderedDict[Any, tuple[float, AskResponse]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.ttl_seconds > 0

    def get(self, key: Any) -> AskResponse | None:
        """Return a copy of the cached response, or None on a miss / expiry."""
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, response = entry
            if self._clock() >= expires_at:
                del self._entries[key]       # expired: drop it, count as a miss
                return None
            self._entries.move_to_end(key)   # mark as most recently used
            # A copy, so a caller changing the response cannot corrupt the cache.
            return response.model_copy(deep=True)

    def put(self, key: Any, response: AskResponse) -> bool:
        """
        Store a response. Returns True if it was stored.

        Not stored: when the cache is disabled, and when the answer is a
        service-unavailable refusal (a temporary failure must not be replayed).
        """
        if not self.enabled:
            return False
        if response.refusal_reason == RefusalReason.SERVICE_UNAVAILABLE:
            return False
        with self._lock:
            self._entries[key] = (
                self._clock() + self.ttl_seconds,
                response.model_copy(deep=True),
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)   # evict the least recently used
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
