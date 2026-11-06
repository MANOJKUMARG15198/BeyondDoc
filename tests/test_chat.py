"""Chat sessions: service turn events, follow-up rewrite, memory rehydration, API ownership and SSE (fakes only)."""

import json

import pytest
from fastapi.testclient import TestClient

from medical_rag.api import chat as chat_api
from medical_rag.api.main import app
from medical_rag.chat.memory import InMemoryMemory
from medical_rag.chat.service import ChatService, title_from
from medical_rag.chat.store import InMemoryStore
from medical_rag.search.retrieval import Hit, SearchResult

SOURCE = {
    "chunk_id": "c1",
    "document_id": "dailymed:a",
    "title": "gabapentin",
    "section_path": "2 DOSAGE > 2.3 Renal",
    "section_role": "body",
    "source": "dailymed",
    "source_reference": "https://dailymed.test/a",
    "medicine_names": ["gabapentin"],
    "kind": "text",
    "token_count": 120,
    "prefix": "gabapentin — 2 DOSAGE > 2.3 Renal",
    "text": "Dose depends on creatinine clearance.",
}


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, **kw):
        self.queries.append((query, kw))
        return SearchResult([Hit("c1", 0.03, SOURCE, {"bm25": 1, "dense": 1})], {"total_ms": 5.0})


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.calls = []

    def complete_json(self, system, user, schema):
        self.calls.append(user)
        if "standalone_query" in schema["properties"]:
            return {"standalone_query": "gabapentin dose in hemodialysis"}, {}
        return {"abstain": False, "answer": "It depends on creatinine clearance [S1].", "cited_sources": ["S1"]}, {}


def make_service():
    return ChatService(InMemoryStore(), InMemoryMemory(max_turns=6), FakeRetriever(), FakeLLM(), 6)


def test_turn_streams_stages_sources_answer_and_persists():
    svc = make_service()
    s = svc.store.create_session("u1", "New consultation")
    events = list(svc.turn(s, "How is gabapentin dosed in renal impairment?"))
    names = [e["event"] for e in events]
    assert names[0] == "user_message" and names[-1] == "answer" and "sources" in names
    stages = [(e["data"]["stage"], e["data"]["status"]) for e in events if e["event"] == "stage"]
    assert stages == [(st, x) for st in ("understand", "search", "compose") for x in ("active", "done")]
    answer = events[-1]["data"]
    assert answer["message"]["cited"] == ["S1"] and answer["message"]["sources"][0]["chunk_id"] == "c1"
    assert answer["session"]["title"] == "How is gabapentin dosed in renal impairment?"
    assert [m.role for m in svc.store.list_messages(s.session_id)] == ["user", "assistant"]
    assert svc.retriever.queries[0][0] == "How is gabapentin dosed in renal impairment?"  # no rewrite on turn 1


def test_follow_up_is_rewritten_with_history_and_history_reaches_answer():
    svc = make_service()
    s = svc.store.create_session("u1", "New consultation")
    list(svc.turn(s, "How is gabapentin dosed in renal impairment?"))
    list(svc.turn(s, "And on hemodialysis?"))
    assert svc.retriever.queries[1][0] == "gabapentin dose in hemodialysis"
    assert "earlier_conversation" in svc.llm.calls[-1] and "creatinine clearance" in svc.llm.calls[-1]


def test_memory_rehydrates_from_store_when_cache_is_empty():
    svc = make_service()
    s = svc.store.create_session("u1", "x")
    list(svc.turn(s, "How is gabapentin dosed in renal impairment?"))
    svc.memory.turns.clear()  # e.g. Redis TTL expired or another worker
    assert [t["role"] for t in svc.history(s.session_id)] == ["user", "assistant"]


def test_rate_limit_and_title():
    mem = InMemoryMemory(rate_per_min=2)
    assert [mem.allow("u"), mem.allow("u"), mem.allow("u")] == [True, True, False]
    assert title_from("word " * 30).endswith("…") and len(title_from("word " * 30)) <= 60


@pytest.fixture
def http():
    svc = make_service()
    app.dependency_overrides[chat_api.get_chat_service] = lambda: svc
    yield TestClient(app), svc
    app.dependency_overrides.clear()


def parse_sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_api_sets_cookie_scopes_sessions_and_streams(http):
    client, svc = http
    r = client.post("/api/sessions", json={})
    assert r.status_code == 201 and "medrag_uid" in r.cookies
    sid = r.json()["session_id"]
    stream = client.post(f"/api/sessions/{sid}/messages", json={"content": "gabapentin renal dosing?", "k": 5})
    assert stream.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(stream.text)
    assert events[-1][0] == "done" and any(e == "answer" for e, _ in events)
    detail = client.get(f"/api/sessions/{sid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert client.get("/api/sessions").json()["sessions"][0]["title"] == "gabapentin renal dosing?"

    other = TestClient(app)  # different browser: no cookie, new user
    assert other.get(f"/api/sessions/{sid}").status_code == 404
    assert other.get("/api/sessions").json()["sessions"] == []


def test_api_rename_delete_and_validation(http):
    client, _ = http
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    assert client.patch(f"/api/sessions/{sid}", json={"title": "Renamed"}).json()["title"] == "Renamed"
    assert client.post(f"/api/sessions/{sid}/messages", json={"content": "hi"}).status_code == 422
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_forged_cookie_is_replaced(http):
    client, _ = http
    client.cookies.set("medrag_uid", "not-a-valid-id")
    r = client.get("/api/me")
    assert r.status_code == 200 and r.cookies.get("medrag_uid") not in (None, "not-a-valid-id")
