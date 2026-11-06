"""/search and /ask API contracts with a fake retriever and a fake LLM (no OpenSearch, model or API key)."""

import pytest
from fastapi.testclient import TestClient

from medical_rag.api import rag
from medical_rag.api.main import app
from medical_rag.search.retrieval import Hit, SearchResult

SOURCE = {
    "chunk_id": "c1",
    "document_id": "dailymed:a",
    "title": "Lipitor",
    "section_path": "Dosage",
    "source": "dailymed",
    "source_reference": "https://dailymed.test/a",
    "prefix": "Lipitor — Dosage",
    "text": "The starting dose is 10 or 20 mg once daily.",
}


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, **kwargs):
        self.calls.append(kwargs)
        return SearchResult(self.hits, {"total_ms": 1.0})


class FakeLLM:
    model = "fake-llm"

    def complete_json(self, system, user, schema):
        return {"abstain": False, "answer": "10 or 20 mg once daily [S1].", "cited_sources": ["S1"]}, {}


@pytest.fixture
def client(monkeypatch):
    retriever = FakeRetriever([Hit("c1", 0.03, SOURCE, {"bm25": 1, "dense": 2})])
    app.dependency_overrides[rag.get_retriever] = lambda: retriever
    monkeypatch.setattr(rag, "_llm", lambda: FakeLLM())
    yield TestClient(app), retriever
    app.dependency_overrides.clear()


def test_search_returns_passages_with_provenance_and_filters(client):
    http, retriever = client
    r = http.post("/search", json={"query": "lipitor dose", "sources": ["dailymed"], "k": 3})
    assert r.status_code == 200
    p = r.json()["passages"][0]
    assert (p["chunk_id"], p["source_reference"], p["component_ranks"]) == (
        "c1",
        "https://dailymed.test/a",
        {"bm25": 1, "dense": 2},
    )
    assert retriever.calls[0]["filters"] == {"source": ["dailymed"]} and "Not medical advice" in r.json()["disclaimer"]


def test_ask_returns_cited_answer_and_timings(client):
    r = client[0].post("/ask", json={"question": "What is the Lipitor starting dose?"})
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert body["citations"][0]["chunk_id"] == "c1" and "total_ms" in body["timings_ms"]


def test_ask_abstains_when_nothing_is_retrieved(client):
    client[1].hits = []
    body = client[0].post("/ask", json={"question": "Unrelated question?"}).json()
    assert body["abstained"] and body["abstain_reason"] == "no_passages_retrieved"


def test_invalid_requests_rejected(client):
    assert client[0].post("/search", json={"query": "x"}).status_code == 422
    assert client[0].post("/search", json={"query": "dose", "sources": ["web"]}).status_code == 422
