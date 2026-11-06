"""sentence-transformers implementation of Embedder, loaded offline from the pinned snapshot in the shared HF cache."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from medical_rag.embedding.base import EmbeddingError, EmbedResult, check_vectors, query_text
from medical_rag.embedding.config import EmbeddingConfig


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise EmbeddingError(f"checksum mismatch for {path.name}: {digest.hexdigest()} != {expected}")


# Only these files are downloaded (safetensors weights; the .bin and ONNX copies are not needed).
MODEL_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "1_Pooling/config.json",
    "model.safetensors",
)


def snapshot_path(config: EmbeddingConfig) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                config.model_name, revision=config.revision, allow_patterns=list(MODEL_FILES), local_files_only=True
            )
        )
    except Exception as exc:  # noqa: BLE001 - any cache miss becomes one clear error
        raise EmbeddingError(f"{config.model_name}@{config.revision} is not in the local HF cache") from exc


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingConfig, path: Path | None = None) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.config = config
        path = path or snapshot_path(config)
        verify_sha256(path / "model.safetensors", config.weights_sha256)
        torch.set_num_threads(config.threads)
        self._model = SentenceTransformer(str(path), device="cpu", local_files_only=True)
        self._model.max_seq_length = config.max_tokens
        if self._model.get_sentence_embedding_dimension() != config.dimension:
            raise EmbeddingError("model dimension does not match the configuration")

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return check_vectors(vectors, len(texts), self.config)

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        encoded = self._model.tokenizer(list(texts), add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def embed_documents(self, texts: Sequence[str]) -> EmbedResult:
        lengths = self.token_lengths(texts)
        truncated = tuple(i for i, n in enumerate(lengths) if n > self.config.max_tokens)
        return EmbedResult(self._encode(texts), truncated)

    def with_query_instruction(self, instruction: str | None) -> SentenceTransformerEmbedder:
        """Same loaded model, different query instruction (document vectors are unaffected)."""
        clone = copy.copy(self)
        clone.config = self.config.model_copy(update={"query_instruction": instruction})
        return clone

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([query_text(self.config, text)])[0]
