"""Embedding settings (ADR-011). The model is the user's 2026-11-04 selection; query_instruction is an open setting."""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

EMBEDDER_VERSION = "1"

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = "BAAI/bge-base-en-v1.5"
    revision: str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    weights_sha256: str = "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7"
    dimension: int = Field(default=768, gt=0)
    max_tokens: int = Field(default=512, gt=0)
    normalize: bool = True
    batch_size: int = Field(default=1, gt=0)  # measured fastest on the demo CPU (2.4 chunks/s)
    threads: int = Field(default=4, gt=0)
    # Open: with vs without is compared on retrieval metrics. Queries only; stored chunk vectors do not depend on it.
    query_instruction: str | None

    @property
    def embedder_id(self) -> str:
        """Deterministic id of everything that changes stored document vectors (not threads or batch size)."""
        identity = self.model_dump(include={"model_name", "revision", "weights_sha256", "max_tokens", "normalize"})
        identity["embedder_version"] = EMBEDDER_VERSION
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
