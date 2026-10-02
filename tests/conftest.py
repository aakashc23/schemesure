"""
Shared pytest setup.

Two things are arranged here before anything imports the settings:

1. The LLM call log is redirected to a temporary SQLite file, so running the
   tests never pollutes the real `llm_calls.db` that `/metrics` reports from.
2. A dummy `LLM_API_KEY` is set. No test ever makes a real network call — the
   client is always replaced by a fake — but the key must be non-empty for the
   app to build its pipeline at startup.

Environment variables take precedence over `.env` in pydantic-settings, so
setting them here is enough; the settings cache is cleared so the override is
picked up.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Must happen before `app.config` is imported anywhere.
_TMP_DIR = Path(tempfile.mkdtemp(prefix="schemesure_tests_"))
os.environ["LLM_LOG_DB"] = str(_TMP_DIR / "test_llm_calls.db")
os.environ.setdefault("LLM_API_KEY", "test-key-not-used-no-network-calls-are-made")

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def _confirm_test_isolation():
    """Fail loudly if a test would write to the real call log."""
    settings = get_settings()
    assert "schemesure_tests_" in str(settings.llm_log_path), (
        f"tests would write to the real log at {settings.llm_log_path}"
    )
    yield
