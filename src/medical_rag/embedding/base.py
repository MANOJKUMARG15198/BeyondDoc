"""Model-neutral embedding interface. Any model implementing Embedder can be swapped in by configuration."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from medical_rag.embedding.config import EmbeddingConfig


class EmbeddingError(RuntimeError):
    """Raised when a model cannot be loaded or verified, or produces unexpected output."""


def chunk_input(chunk: Mapping[str, Any]) -> str:
    """The exact text the 512-token ceiling was measured on: prefix, newline, body."""
    return f"{chunk['prefix']}\n{chunk['text']}"


@dataclass(frozen=True)
class EmbedResult:
    vectors: np.ndarray  # (n, dimension) float32
    truncated: tuple[int, ...]  # indices of inputs longer than max_tokens (their ends were cut)


class Embedder(Protocol):
    config: EmbeddingConfig

    def embed_documents(self, texts: Sequence[str]) -> EmbedResult: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def query_text(config: EmbeddingConfig, text: str) -> str:
    return f"{config.query_instruction}{text}" if config.query_instruction else text


def check_vectors(vectors: np.ndarray, n: int, config: EmbeddingConfig) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (n, config.dimension):
        raise EmbeddingError(f"expected shape {(n, config.dimension)}, got {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise EmbeddingError("non-finite values in embeddings")
    if config.normalize and n and not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise EmbeddingError("vectors are not L2-normalised")
    return vectors
