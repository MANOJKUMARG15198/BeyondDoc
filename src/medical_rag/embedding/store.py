"""Resumable block-wise embedding to disk: parts/part_NNNN.npz, then vectors.npy, chunk_ids.jsonl, manifest.json."""

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from medical_rag.embedding.base import Embedder, EmbeddingError

Progress = Callable[[int, int, float], None]


def embed_to_dir(
    embedder: Embedder,
    chunk_ids: Sequence[str],
    texts: Sequence[str],
    out_dir: Path,
    *,
    block: int = 256,
    manifest_extra: dict[str, Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    if len(chunk_ids) != len(texts) or len(set(chunk_ids)) != len(chunk_ids):
        raise EmbeddingError("chunk ids must be unique and match the texts")
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    plan = {"embedder_id": embedder.config.embedder_id, "count": len(texts), "block": block}
    plan_file = parts / "plan.json"
    ids_digest = json.dumps(list(chunk_ids))
    if plan_file.exists():
        saved = json.loads(plan_file.read_text(encoding="utf-8"))
        if saved["plan"] != plan or saved["ids"] != ids_digest:
            raise EmbeddingError(f"{parts} belongs to a different run; remove it to start over")
    else:
        plan_file.write_text(json.dumps({"plan": plan, "ids": ids_digest}), encoding="utf-8")

    start = time.perf_counter()
    embedded_now = 0
    for k, begin in enumerate(range(0, len(texts), block)):
        part = parts / f"part_{k:04d}.npz"
        if part.exists():
            continue
        result = embedder.embed_documents(texts[begin : begin + block])
        tmp = parts / f"part_{k:04d}.tmp.npz"
        np.savez(tmp, vectors=result.vectors, truncated=np.array(result.truncated, dtype=np.int64) + begin)
        tmp.replace(part)
        embedded_now += len(result.vectors)
        if progress:
            progress(min(begin + block, len(texts)), len(texts), time.perf_counter() - start)
    elapsed = time.perf_counter() - start

    loaded = [np.load(parts / f"part_{k:04d}.npz") for k in range(-(-len(texts) // block))]
    vectors = np.concatenate([p["vectors"] for p in loaded]) if loaded else np.zeros((0, embedder.config.dimension))
    truncated = sorted(int(i) for p in loaded for i in p["truncated"])
    np.save(out_dir / "vectors.npy", vectors.astype(np.float32))
    (out_dir / "chunk_ids.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunk_ids), encoding="utf-8")
    manifest = {
        **(manifest_extra or {}),
        **embedder.config.model_dump(exclude={"query_instruction"}),
        "embedder_id": embedder.config.embedder_id,
        "count": len(texts),
        "truncated_chunk_ids": [chunk_ids[i] for i in truncated],
        "embedded_this_run": embedded_now,
        "seconds_this_run": round(elapsed, 1),
        "chunks_per_second_this_run": round(embedded_now / elapsed, 2) if embedded_now and elapsed else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_vectors(out_dir: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    ids = [json.loads(line) for line in (out_dir / "chunk_ids.jsonl").read_text(encoding="utf-8").splitlines()]
    vectors = np.load(out_dir / "vectors.npy")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(ids) != len(vectors):
        raise EmbeddingError("vector and id counts differ")
    return ids, vectors, manifest
