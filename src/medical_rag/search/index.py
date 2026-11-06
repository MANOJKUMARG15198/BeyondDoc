"""Chunk index: BM25 text + HNSW dense vectors in one OpenSearch index (demo defaults ADR-012/013, provisional).

Index names carry the chunk config_id and embedder_id so a different chunking or model never mixes into an index.
The alias `medrag-chunks` points at the index the API serves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from opensearchpy import OpenSearch, helpers

from medical_rag.embedding.base import chunk_input

ALIAS = "medrag-chunks"


@dataclass(frozen=True)
class IndexSettings:
    dimension: int = 768
    space_type: str = "cosinesimil"  # vectors are L2-normalised
    engine: str = "lucene"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128
    ef_search: int = 100
    analyzer: str = "english"  # BM25 analyzer for search_text, provisional


def index_name(chunk_config_id: str, embedder_id: str) -> str:
    return f"{ALIAS}-{chunk_config_id}-{embedder_id}"


def mapping(s: IndexSettings) -> dict[str, Any]:
    keyword = {"type": "keyword"}
    return {
        "settings": {
            "index": {"knn": True, "knn.algo_param.ef_search": s.ef_search, "number_of_shards": 1},
            "number_of_replicas": 0,
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": keyword,
                "document_id": keyword,
                "document_version": keyword,
                "source": keyword,
                "source_reference": keyword,
                "title": {"type": "text", "fields": {"raw": keyword}},
                "section_path": {"type": "text"},
                "section_role": keyword,
                "evidence_type": keyword,
                "medicine_names": {"type": "text", "fields": {"raw": keyword}},
                "kind": keyword,
                "ordinal": {"type": "integer"},
                "token_count": {"type": "integer"},
                "overlap_tokens": {"type": "integer"},
                "notices": keyword,
                "split_levels": keyword,
                "prefix": {"type": "text", "index": False},
                "text": {"type": "text", "index": False},
                # The same text the embedder saw: prefix, newline, body. BM25 searches this field.
                "search_text": {"type": "text", "analyzer": s.analyzer},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": s.dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": s.space_type,
                        "engine": s.engine,
                        "parameters": {"m": s.hnsw_m, "ef_construction": s.hnsw_ef_construction},
                    },
                },
            },
        },
    }


DOC_FIELDS = (
    "chunk_id",
    "document_id",
    "document_version",
    "source",
    "source_reference",
    "title",
    "section_path",
    "section_role",
    "evidence_type",
    "medicine_names",
    "kind",
    "ordinal",
    "token_count",
    "overlap_tokens",
    "notices",
    "split_levels",
    "prefix",
    "text",
)


def to_document(chunk: Mapping[str, Any], vector: np.ndarray) -> dict[str, Any]:
    doc = {field: chunk.get(field) for field in DOC_FIELDS}
    path = chunk.get("section_path")
    doc["section_path"] = " > ".join(path) if isinstance(path, list | tuple) else path
    doc["search_text"] = chunk_input(chunk)
    doc["embedding"] = [float(x) for x in vector]
    return doc


def bulk_actions(index: str, chunks: Sequence[Mapping[str, Any]], vectors: np.ndarray) -> Iterable[dict[str, Any]]:
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors differ in length")
    for chunk, vector in zip(chunks, vectors, strict=True):
        yield {"_index": index, "_id": chunk["chunk_id"], "_source": to_document(chunk, vector)}


def build_index(
    client: OpenSearch,
    index: str,
    chunks: Sequence[Mapping[str, Any]],
    vectors: np.ndarray,
    settings: IndexSettings | None = None,
    *,
    point_alias: bool = True,
) -> dict[str, Any]:
    """Create the index (fails if it exists), bulk-load, refresh, verify the count, optionally move the alias."""
    client.indices.create(index=index, body=mapping(settings or IndexSettings()))
    ok, errors = helpers.bulk(client, bulk_actions(index, chunks, vectors), chunk_size=200, raise_on_error=False)
    client.indices.refresh(index=index)
    count = client.count(index=index)["count"]
    if errors or count != len(chunks):
        raise RuntimeError(f"indexed {count} of {len(chunks)} chunks; {len(errors)} bulk errors")
    if point_alias:
        actions: list[dict[str, Any]] = [{"add": {"index": index, "alias": ALIAS}}]
        if client.indices.exists_alias(name=ALIAS):
            actions = [{"remove": {"index": old, "alias": ALIAS}} for old in client.indices.get_alias(name=ALIAS)]
            actions.append({"add": {"index": index, "alias": ALIAS}})
        client.indices.update_aliases(body={"actions": actions})
    return {"index": index, "count": count, "alias": ALIAS if point_alias else None}
