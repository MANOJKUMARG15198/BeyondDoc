"""Embedding interface, config and resumable store, with a fake model (no downloads, no torch)."""

import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

from medical_rag.embedding.base import EmbeddingError, EmbedResult, check_vectors, chunk_input, query_text
from medical_rag.embedding.config import BGE_QUERY_INSTRUCTION, EmbeddingConfig
from medical_rag.embedding.st_embedder import verify_sha256
from medical_rag.embedding.store import embed_to_dir, load_vectors

DIM = 4


def cfg(**changes):
    return EmbeddingConfig(**{"query_instruction": None, "dimension": DIM, **changes})


class FakeEmbedder:
    def __init__(self, config, fail_after=None):
        self.config = config
        self.calls = 0
        self.fail_after = fail_after

    def embed_documents(self, texts):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("interrupted")
        self.calls += 1
        vectors = np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return EmbedResult(vectors, tuple(i for i, t in enumerate(texts) if "LONG" in t))

    def embed_query(self, text):
        return self.embed_documents([query_text(self.config, text)]).vectors[0]


def test_query_instruction_is_required_open_setting():
    with pytest.raises(ValidationError):
        EmbeddingConfig()


def test_chunk_input_is_prefix_newline_body():
    chunk = {"prefix": "Lipitor — Dosage", "text": "Take once daily."}
    assert chunk_input(chunk) == "Lipitor — Dosage\nTake once daily."


def test_query_instruction_only_when_set():
    assert query_text(cfg(), "dose?") == "dose?"
    assert query_text(cfg(query_instruction=BGE_QUERY_INSTRUCTION), "dose?") == BGE_QUERY_INSTRUCTION + "dose?"


def test_embedder_id_ignores_query_instruction_threads_and_batch():
    base = cfg().embedder_id
    assert cfg(query_instruction="x: ", threads=8, batch_size=16).embedder_id == base
    assert cfg(revision="other").embedder_id != base
    assert cfg(max_tokens=256).embedder_id != base


def test_check_vectors_rejects_bad_output():
    good = np.eye(2, DIM, dtype=np.float32)
    assert check_vectors(good, 2, cfg()).dtype == np.float32
    with pytest.raises(EmbeddingError):
        check_vectors(good, 3, cfg())
    with pytest.raises(EmbeddingError):
        check_vectors(good * 2, 2, cfg())
    with pytest.raises(EmbeddingError):
        check_vectors(np.full((2, DIM), np.nan), 2, cfg(normalize=False))


def test_checksum_mismatch_fails(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"weights")
    verify_sha256(f, hashlib.sha256(b"weights").hexdigest())
    with pytest.raises(EmbeddingError):
        verify_sha256(f, "0" * 64)


def test_store_writes_vectors_ids_manifest_and_truncation(tmp_path):
    ids = [f"c{i}" for i in range(5)]
    texts = ["a", "bb", "LONG ccc", "d", "LONG e"]
    manifest = embed_to_dir(FakeEmbedder(cfg()), ids, texts, tmp_path, block=2, manifest_extra={"experiment": "E3"})
    got_ids, vectors, saved = load_vectors(tmp_path)
    assert got_ids == ids and vectors.shape == (5, DIM)
    assert saved == manifest and manifest["truncated_chunk_ids"] == ["c2", "c4"]
    assert manifest["experiment"] == "E3" and "query_instruction" not in manifest


def test_store_resumes_after_interruption(tmp_path):
    ids, texts = [f"c{i}" for i in range(6)], ["x" * (i + 1) for i in range(6)]
    with pytest.raises(RuntimeError):
        embed_to_dir(FakeEmbedder(cfg(), fail_after=2), ids, texts, tmp_path, block=2)
    second = FakeEmbedder(cfg())
    manifest = embed_to_dir(second, ids, texts, tmp_path, block=2)
    assert second.calls == 1 and manifest["embedded_this_run"] == 2
    fresh = tmp_path / "fresh"
    embed_to_dir(FakeEmbedder(cfg()), ids, texts, fresh, block=2)
    assert np.array_equal(np.load(tmp_path / "vectors.npy"), np.load(fresh / "vectors.npy"))


def test_store_refuses_a_different_run(tmp_path):
    embed_to_dir(FakeEmbedder(cfg()), ["a"], ["x"], tmp_path)
    with pytest.raises(EmbeddingError):
        embed_to_dir(FakeEmbedder(cfg(revision="other")), ["a"], ["x"], tmp_path)
    with pytest.raises(EmbeddingError):
        embed_to_dir(FakeEmbedder(cfg()), ["b"], ["x"], tmp_path)


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(EmbeddingError):
        embed_to_dir(FakeEmbedder(cfg()), ["a", "a"], ["x", "y"], tmp_path)
    assert not (tmp_path / "parts").exists()
