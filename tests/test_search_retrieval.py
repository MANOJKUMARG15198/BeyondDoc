"""Unit tests for the index mapping, document building and RRF fusion (no OpenSearch needed)."""

import numpy as np

from medical_rag.search.index import IndexSettings, bulk_actions, index_name, mapping, to_document
from medical_rag.search.retrieval import Retriever, bm25_body, dense_body, rrf

CHUNK = {
    "chunk_id": "c1",
    "document_id": "dailymed:x",
    "document_version": "3",
    "source": "dailymed",
    "source_reference": "https://dailymed.test/x",
    "title": "Lipitor",
    "section_path": ["Dosage", "Adults"],
    "section_role": "dosage",
    "evidence_type": "general",
    "medicine_names": ["Lipitor", "atorvastatin"],
    "kind": "text",
    "ordinal": 0,
    "token_count": 12,
    "prefix": "Lipitor — Dosage > Adults",
    "text": "Take 10 mg once daily.",
    "content_spans": [],
}


def test_mapping_is_hnsw_lucene_cosine_and_strict():
    m = mapping(IndexSettings())
    vector = m["mappings"]["properties"]["embedding"]
    assert vector["dimension"] == 768 and vector["method"]["name"] == "hnsw"
    assert (vector["method"]["engine"], vector["method"]["space_type"]) == ("lucene", "cosinesimil")
    assert m["mappings"]["dynamic"] == "strict" and m["settings"]["index"]["knn"] is True


def test_index_name_carries_config_and_embedder_ids():
    assert index_name("cfg", "emb") == "medrag-chunks-cfg-emb"


def test_document_has_provenance_search_text_and_vector():
    doc = to_document(CHUNK, np.array([0.6, 0.8], dtype=np.float32))
    assert doc["search_text"] == "Lipitor — Dosage > Adults\nTake 10 mg once daily."
    assert doc["section_path"] == "Dosage > Adults" and doc["source_reference"] == "https://dailymed.test/x"
    assert doc["embedding"] == [np.float32(0.6), np.float32(0.8)] and "content_spans" not in doc
    assert set(doc) <= set(mapping(IndexSettings())["mappings"]["properties"])


def test_bulk_actions_use_chunk_id_as_document_id():
    (action,) = bulk_actions("idx", [CHUNK], np.zeros((1, 2)))
    assert (action["_index"], action["_id"]) == ("idx", "c1")


def test_rrf_scores_and_deterministic_ties():
    fused = rrf({"bm25": ["a", "b", "c"], "dense": ["b", "d"]}, k=60)
    assert fused[0][0] == "b" and fused[0][2] == {"bm25": 2, "dense": 1}
    assert fused[0][1] == 1 / 62 + 1 / 61
    assert [c for c, _, _ in fused[1:3]] == ["a", "d"]  # equal scores 1/61, ordered by id


def test_query_bodies_with_filters():
    body = bm25_body("dose", 5, {"source": "dailymed"})
    assert body["query"]["bool"]["filter"] == [{"terms": {"source": ["dailymed"]}}]
    knn = dense_body(np.ones(2), 5, {"source": ["pmc_oa"]})["query"]["knn"]["embedding"]
    assert knn["k"] == 5 and knn["filter"]["bool"]["filter"] == [{"terms": {"source": ["pmc_oa"]}}]


class FakeClient:
    def __init__(self, results):
        self.results, self.bodies = results, []

    def search(self, index, body):
        self.bodies.append(body)
        name = "dense" if "knn" in body["query"] else "bm25"
        return {"hits": {"hits": [{"_id": c, "_score": 1.0, "_source": {"chunk_id": c}} for c in self.results[name]]}}


class FakeEmbedder:
    def embed_query(self, text):
        return np.ones(2)


def test_hybrid_search_fuses_and_reports_component_timings():
    client = FakeClient({"bm25": ["a", "b"], "dense": ["b", "c"]})
    result = Retriever(client, FakeEmbedder()).search("q", k=2, candidates=10)
    assert [h.chunk_id for h in result.hits] == ["b", "a"]
    assert {"bm25_ms", "embed_query_ms", "dense_ms", "fusion_ms", "total_ms"} <= set(result.timings_ms)
    assert all(b["size"] == 10 for b in client.bodies)


def test_single_mode_returns_component_ranking():
    result = Retriever(FakeClient({"bm25": ["x", "y"]}), FakeEmbedder()).search("q", mode="bm25", k=2)
    assert [(h.chunk_id, h.ranks) for h in result.hits] == [("x", {"bm25": 1}), ("y", {"bm25": 2})]
