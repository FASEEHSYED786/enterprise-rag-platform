"""Runtime configuration, read from environment variables (12-factor)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    db_path: str = field(default_factory=lambda: _env("RAG_DB_PATH", "ragplatform.db"))
    llm_provider: str = field(default_factory=lambda: _env("RAG_LLM_PROVIDER", "extractive"))  # extractive|anthropic|openai
    llm_model: str = field(default_factory=lambda: _env("RAG_LLM_MODEL", "claude-sonnet-5-5"))
    llm_base_url: str = field(default_factory=lambda: _env("RAG_LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("RAG_LLM_API_KEY", ""))
    embed_dim: int = field(default_factory=lambda: int(_env("RAG_EMBED_DIM", "384")))
    chunk_tokens: int = field(default_factory=lambda: int(_env("RAG_CHUNK_TOKENS", "220")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("RAG_CHUNK_OVERLAP", "40")))
    rate_limit_per_min: int = field(default_factory=lambda: int(_env("RAG_RATE_LIMIT_PER_MIN", "60")))
    admin_key: str = field(default_factory=lambda: _env("RAG_ADMIN_KEY", ""))
