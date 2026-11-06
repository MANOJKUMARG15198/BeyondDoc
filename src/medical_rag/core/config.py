"""Typed settings loaded from environment variables (prefix MEDRAG_) and an optional .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDRAG_", env_file=".env", extra="ignore")

    env: Literal["local", "ci", "demo"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Query and answer text is medical content; it is not logged unless explicitly enabled.
    log_query_text: bool = False

    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and each new connection waits ~2 s.
    opensearch_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:9200")
    opensearch_timeout_s: float = Field(default=10.0, gt=0, le=120)

    data_dir: Path = Path("data")

    # Claude API (ADR-016 demo default). The key is read from ANTHROPIC_API_KEY and never logged or returned.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5-5"  # answer generation (user choice 2026-11-04)
    # Ragas judge, separate from the generator. User choice 2026-11-05: Kimi K3 served by Fireworks AI
    # (OpenAI-compatible endpoint).
    judge_provider: Literal["fireworks", "anthropic"] = "fireworks"
    # Alternative names (KIMI_MODEL, KIMIK3_API_KEY, KIMI_BASE_URL) are accepted as well.
    judge_model: str = Field(
        default="accounts/fireworks/models/kimi-k3",
        validation_alias=AliasChoices("MEDRAG_JUDGE_MODEL", "KIMI_MODEL"),
    )
    judge_reasoning_effort: Literal["low", "medium", "high"] | None = "low"
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    fireworks_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("FIREWORKS_API_KEY", "KIMIK3_API_KEY")
    )
    fireworks_base_url: str = Field(
        default="https://api.fireworks.ai/inference/v1",
        validation_alias=AliasChoices("MEDRAG_FIREWORKS_BASE_URL", "KIMI_BASE_URL"),
    )
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Answer prompt (2026-11-06): v4.1 = v3 + coverage (full / related / none: related-information answers instead
    # of refusals) + dose safeguard; v4 = without the safeguard; v3 = polite, cited, lists only for 3+ points;
    # v2 = first polite numbered version; v1 = original.
    answer_prompt: Literal["v1", "v2", "v3", "v4", "v4.1"] = "v4.1"
    llm_timeout_s: float = Field(default=120.0, gt=0, le=600)

    # Chat sessions: Redis short-term memory and rate limits, ClickHouse history (local dev services).
    redis_url: str = "redis://127.0.0.1:6379/0"
    clickhouse_host: str = "127.0.0.1"
    clickhouse_port: int = 8123
    clickhouse_user: str = "medrag"
    clickhouse_password: SecretStr = SecretStr("")
    clickhouse_database: str = "medrag"
    chat_retention_days: int = Field(default=90, ge=1, le=3650)
    chat_memory_turns: int = Field(default=6, ge=1, le=20)  # previous messages used for follow-ups
    chat_memory_ttl_s: int = Field(default=86400, ge=60)
    chat_rate_limit_per_min: int = Field(default=20, ge=1, le=1000)  # questions per user per minute

    # Search index served by the API and retrieval defaults (demo defaults, provisional).
    index_alias: str = "medrag-chunks"
    retrieval_k: int = Field(default=8, gt=0, le=50)
    retrieval_candidates: int = Field(default=50, gt=0, le=200)
    # Retrieval v1.2 plan preset for hybrid search: "none" keeps the plain hybrid baseline; "v12_t3" adds the
    # medicine/source/section-aware plan (experimental, selected on the v2.0 dev split). Alias table: newest built.
    retrieval_plan: Literal["none", "v12_t3"] = "none"
    alias_table_path: str | None = None

    # Tracing (ADR-008): "langfuse" sends one trace per request to LANGFUSE_HOST (Langfuse Cloud or self-hosted).
    # Metadata only unless trace_content is true (synthetic demo inputs; CLAUDE.md rule 9).
    tracing: Literal["none", "langfuse"] = "none"
    langfuse_public_key: SecretStr | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: SecretStr | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")
    trace_content: bool = False
    trace_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
