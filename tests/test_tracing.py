"""Langfuse tracing (ADR-008) with the real SDK and an in-memory span exporter: no network, no keys."""

import json
import uuid

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from medical_rag.chat.memory import InMemoryMemory
from medical_rag.chat.service import ChatService
from medical_rag.chat.store import InMemoryStore
from medical_rag.core.config import Settings
from medical_rag.observability.tracing import TEXT_LIMIT, LangfuseTracer, Tracer, hash_user, mask_secrets
from medical_rag.search.retrieval import Hit, SearchResult

QUESTION = "How is gabapentin dosed in renal impairment?"
SOURCE = {"chunk_id": "c1", "document_id": "dailymed:a", "title": "gabapentin", "section_path": "2.3 Renal",
          "source": "dailymed", "source_reference": "https://dailymed.test/a", "prefix": "gabapentin",
          "text": "Dose depends on creatinine clearance."}  # fmt: skip


class FakeRetriever:
    def search(self, query, **kw):
        return SearchResult([Hit("c1", 0.03, SOURCE, {"bm25": 1})], {"total_ms": 5.0, "bm25_ms": 1.0})


class FakeLLM:
    model = "fake-model"

    def complete_json(self, system, user, schema):
        if "standalone_query" in schema["properties"]:
            return {"standalone_query": "gabapentin dose in hemodialysis"}, {"input_tokens": 40, "output_tokens": 9}
        answer = "It depends on creatinine clearance [S1]."
        return {"abstain": False, "answer": answer, "cited_sources": ["S1"]}, {"input_tokens": 900, "output_tokens": 30}


@pytest.fixture
def traced():
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key=f"pk-lf-test-{uuid.uuid4().hex}", secret_key="sk-lf-test", host="http://127.0.0.1:9",
                      span_exporter=exporter, tracer_provider=TracerProvider(), mask=mask_secrets)  # fmt: skip

    def make(content=False):
        tracer = LangfuseTracer(client, content=content, environment="test")
        svc = ChatService(InMemoryStore(), InMemoryMemory(max_turns=6), FakeRetriever(), FakeLLM(), 6,
                          tracer=tracer, trace_tags=["env:test", "prompt:v4.1"])  # fmt: skip
        return svc, tracer

    def spans():
        client.flush()
        return exporter.get_finished_spans()

    yield make, spans, exporter


def all_values(span) -> str:
    return json.dumps({k: str(v) for k, v in dict(span.attributes).items()})


def test_default_tracer_is_a_noop_and_missing_keys_disable_langfuse():
    assert type(LangfuseTracer.from_settings(Settings(_env_file=None, tracing="langfuse"))) is Tracer
    t = Tracer()
    handle = t.start("x", user_id="u")
    with handle.span("s") as s:
        s.update(output="anything")
    handle.end(output="done")
    assert t.text("question") is None


def test_chat_turn_is_one_trace_with_stages_and_no_text_by_default(traced):
    make, spans, _ = traced
    svc, _ = make(content=False)
    s = svc.store.create_session("cookie-uid", "New consultation")
    list(svc.turn(s, QUESTION))
    list(svc.turn(s, "And on hemodialysis?"))  # follow-up: the rewrite becomes a generation
    got = spans()
    names = [x.name for x in got]
    assert names.count("chat_turn") == 2 and names.count("compose") == 2 and names.count("retrieve") == 2
    second = [x for x in got if x.context.trace_id == got[-1].context.trace_id]
    assert {x.name for x in second} == {"chat_turn", "understand", "retrieve", "compose"}
    root = next(x for x in second if x.name == "chat_turn")
    assert all(x.parent and x.parent.span_id == root.context.span_id for x in second if x is not root)
    for x in got:
        attrs = dict(x.attributes)
        assert attrs["user.id"] == hash_user("cookie-uid") != "cookie-uid" and attrs["session.id"] == s.session_id
        text = all_values(x)
        assert QUESTION not in text and "creatinine clearance [S1]" not in text and "hemodialysis?" not in text
    compose = next(x for x in second if x.name == "compose")
    attrs = dict(compose.attributes)
    assert attrs["langfuse.observation.type"] == "generation"
    assert json.loads(attrs["langfuse.observation.usage_details"]) == {"input": 900, "output": 30}
    assert "c1" in all_values(next(x for x in second if x.name == "retrieve"))  # chunk ids are metadata
    understand = next(x for x in second if x.name == "understand")
    assert json.loads(dict(understand.attributes)["langfuse.observation.usage_details"]) == {"input": 40, "output": 9}


def test_content_switch_adds_truncated_masked_text(traced):
    make, spans, _ = traced
    svc, tracer = make(content=True)
    s = svc.store.create_session("u", "New consultation")
    list(svc.turn(s, QUESTION + " my key is sk-ant-abcdefghijklmnopqrstu"))
    text = " ".join(all_values(x) for x in spans())
    assert "How is gabapentin dosed" in text and "creatinine clearance [S1]" in text
    assert "sk-ant-abcdefghijklmnop" not in text and "[REDACTED]" in text
    assert len(tracer.text("x" * (TEXT_LIMIT + 50))) == TEXT_LIMIT


def test_tracing_errors_never_break_a_turn_and_errors_inside_spans_propagate(traced):
    make, spans, _ = traced
    svc, tracer = make()

    class Broken:
        def start_as_current_observation(self, **_):
            raise RuntimeError("langfuse down")

        def flush(self):
            raise RuntimeError("langfuse down")

    tracer.client = Broken()
    s = svc.store.create_session("u", "New consultation")
    events = list(svc.turn(s, QUESTION))
    assert events[-1]["event"] == "answer"
    tracer.flush()  # swallowed
    svc2, _ = make()
    handle = svc2.tracer.start("ask")
    with pytest.raises(ValueError), handle.span("boom"):
        raise ValueError("real error")
    handle.end()
    assert {"boom", "ask"} <= {x.name for x in spans()}


def test_mask_secrets_walks_nested_values():
    data = {"a": ["Authorization: Bearer abcdefghijklmnop", {"k": "api_key=supersecret1"}], "n": 3}
    out = mask_secrets(data=data)
    assert "abcdefghijklmnop" not in json.dumps(out) and "supersecret1" not in json.dumps(out) and out["n"] == 3


def test_search_and_ask_endpoints_each_send_one_trace(traced, monkeypatch):
    from fastapi.testclient import TestClient

    from medical_rag.api import rag
    from medical_rag.api.main import app

    make, spans, _ = traced
    _, tracer = make()

    class Retr:
        def search(self, query, **kw):
            return FakeRetriever().search(query)

    app.dependency_overrides[rag.get_retriever] = lambda: Retr()
    monkeypatch.setattr(rag, "_llm", lambda: FakeLLM())
    monkeypatch.setattr(rag, "get_tracer", lambda: tracer)
    try:
        http = TestClient(app)
        assert http.post("/search", json={"query": QUESTION, "k": 3}).status_code == 200
        assert http.post("/ask", json={"question": QUESTION, "k": 3}).status_code == 200
    finally:
        app.dependency_overrides.clear()
    got = spans()
    assert sorted(x.name for x in got if x.parent is None) == ["ask", "search"]
    assert {x.name for x in got} == {"search", "ask", "retrieve", "compose"}
    assert all(QUESTION not in all_values(x) for x in got)


def test_from_settings_exports_only_langfuse_spans(monkeypatch):
    import langfuse
    from langfuse.span_filter import is_langfuse_span

    seen = {}

    class Fake:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(langfuse, "Langfuse", Fake)
    settings = Settings(
        _env_file=None, tracing="langfuse", LANGFUSE_PUBLIC_KEY="pk-lf-x", LANGFUSE_SECRET_KEY="sk-lf-y"
    )
    tracer = LangfuseTracer.from_settings(settings)
    assert isinstance(tracer, LangfuseTracer) and seen["should_export_span"] is is_langfuse_span
    assert seen["mask"] is mask_secrets and tracer.content is False
