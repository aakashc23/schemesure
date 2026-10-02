"""
All configuration in one place, loaded from .env via pydantic-settings.

WHY: nothing in this project reads os.environ directly. Every tunable lives here
with a type and a default, so a typo in a setting name fails loudly at startup
instead of silently behaving differently in production.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = the directory containing this app/ package's parent.
ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed view of .env. Field names match the env var names (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate extra keys in .env rather than crashing
    )

    # ---- LLM -------------------------------------------------------------
    # Groq exposes an OpenAI-compatible endpoint, so the standard `openai`
    # client works unchanged — only base_url and the model name differ.
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    # Temperature 0: this is an extraction/judging system, not a creative one.
    # We want the same answer for the same evidence every time.
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # ---- Hugging Face deployment ----------------------------------------
    hf_token: str = ""
    hf_space_id: str = ""
    hf_space_sdk: str = "docker"

    # ---- Embeddings ------------------------------------------------------
    # Multilingual on purpose: the same model must place an English question and
    # its Hindi translation near the same English scheme text.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"

    # ---- Vector store ----------------------------------------------------
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "schemesure_schemes"

    # ---- Retrieval -------------------------------------------------------
    retrieval_top_k: int = 5
    # Cosine-similarity floor. Below this we refuse rather than answer.
    # Calibrated against real scores in Phase 2 — see docs/DECISIONS.md.
    retrieval_min_score: float = 0.45

    # ---- Guardrail -------------------------------------------------------
    guardrail_enabled: bool = True
    # Fraction of unsupported claims at or above which the whole answer is
    # blocked instead of repaired.
    guardrail_block_threshold: float = 0.5

    # ---- Monitoring ------------------------------------------------------
    llm_log_db: str = "./llm_calls.db"

    # ---- API / UI --------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"
    max_question_length: int = 500
    cors_allow_origins: str = "*"

    # ---- Derived helpers -------------------------------------------------

    @property
    def data_dir(self) -> Path:
        """Where the scheme .md and .rules.json files live."""
        return ROOT / "data" / "schemes"

    @property
    def chroma_dir(self) -> Path:
        """Absolute path to the Chroma directory (relative paths resolve to root)."""
        path = Path(self.chroma_path)
        return path if path.is_absolute() else ROOT / path

    @property
    def llm_log_path(self) -> Path:
        """Absolute path to the SQLite file that records every LLM call."""
        path = Path(self.llm_log_db)
        return path if path.is_absolute() else ROOT / path

    @property
    def cors_origins_list(self) -> list[str]:
        """CORS origins as a list ("*" or a comma-separated allow-list)."""
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    def require_llm_key(self) -> None:
        """Fail with a clear message when the LLM key is missing."""
        if not self.llm_api_key.strip():
            raise RuntimeError(
                "LLM_API_KEY is empty. Add it to .env (local) or set it as a Space "
                "secret (deployed). Get a free key at https://console.groq.com/keys"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached accessor so .env is parsed once per process.

    FastAPI, the ingestion script, the evaluation harness and the tests all call
    this, and they must all see the same values.
    """
    return Settings()
