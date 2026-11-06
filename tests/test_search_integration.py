"""Index build and hybrid search on the local Docker OpenSearch, synthetic chunks. Run: pytest -m integration"""

import uuid

import numpy as np
import pytest

from medical_rag.core.config import Settings
from medical_rag.search.client import cluster_health, make_client
from medical_rag.search.index import IndexSettings, build_index
from medical_rag.search.retrieval import Retriever

pytestmark = pytest.mark.integration
DIM = 4


@pytest.fixture(scope="module")
def client():
    c = make_client(Settings(_env_file=None))
    if cluster_health(c)["status"] not in ("green", "yellow"):
        pytest.skip("OpenSearch not running (docker compose up -d)")
    return c


def chunk(i, text, source="dailymed"):
    return {
        "chunk_id": f"c{i}",
        "document_id": f"{source}:d{i}",
        "document_version": "1",
        "source": source,
        "source_reference": f"https://example.test/{i}",
        "title": f"Synthetic {i}",
        "section_path": ["Section"],
        "section_role": "other",
        "evidence_type": "general",
        "medicine_names": ["examplamide"],
        "kind": "text",
        "ordinal": 0,
        "token_count": 5,
        "prefix": f"Synthetic {i} — Section",
        "text": text,
    }


class AxisEmbedder:
    """Query 'axis N' embeds to unit vector N, so dense ranking is known in advance."""

    def embed_query(self, text):
        v = np.zeros(DIM)
        v[int(text.split()[-1])] = 1.0
        return v


@pytest.fixture(scope="module")
def index(client):
    name = f"medrag-it-{uuid.uuid4().hex[:8]}"
    chunks = [
        chunk(0, "hepatotoxicity monitoring liver enzymes"),
        chunk(1, "renal dosing adjustment"),
        chunk(2, "liver injury case report", "nih_livertox"),
        chunk(3, "pregnancy lactation"),
    ]
    vectors = np.eye(DIM, dtype=np.float32)
    report = build_index(client, name, chunks, vectors, IndexSettings(dimension=DIM), point_alias=False)
    assert report["count"] == 4
    yield name
    client.indices.delete(index=name, ignore_unavailable=True)


def test_bm25_dense_and_hybrid(client, index):
    r = Retriever(client, AxisEmbedder(), index=index)
    assert r.search("liver", mode="bm25", k=2).hits[0].chunk_id in {"c0", "c2"}
    assert r.search("axis 3", mode="dense", k=1).hits[0].chunk_id == "c3"
    hybrid = r.search("liver 2", mode="hybrid", k=3)  # BM25 finds c0/c2 ("liver"); dense ranks c2 first
    assert hybrid.hits[0].chunk_id == "c2" and set(hybrid.hits[0].ranks) == {"bm25", "dense"}
    assert hybrid.hits[0].source["source_reference"] == "https://example.test/2"


def test_filters_apply_to_both_components(client, index):
    r = Retriever(client, AxisEmbedder(), index=index)
    result = r.search("liver 0", mode="hybrid", k=5, filters={"source": "nih_livertox"})
    assert [h.chunk_id for h in result.hits] == ["c2"]


def test_existing_index_is_not_overwritten(client, index):
    with pytest.raises(Exception):  # noqa: B017 - opensearch-py raises RequestError
        build_index(client, index, [], np.zeros((0, DIM)), IndexSettings(dimension=DIM), point_alias=False)
