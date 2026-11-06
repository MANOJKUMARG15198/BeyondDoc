"""Retrieval metrics, gold-chunk matching and Ragas adapters (fake LLM, no API calls)."""

import asyncio

import pytest
from pydantic import BaseModel

from medical_rag.evaluation.ragas_adapters import ClaudeJudge, strict_schema
from medical_rag.evaluation.retrieval_metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k, score_run
from medical_rag.evaluation.testset import TestItem, gold_chunks, load_testset, write_testset


def test_metrics_on_known_ranking():
    ranked, gold = ["x", "a", "y", "b"], {"a", "b"}
    assert hit_at_k(ranked, gold, 1) == 0.0 and hit_at_k(ranked, gold, 2) == 1.0
    assert recall_at_k(ranked, gold, 2) == 0.5 and recall_at_k(ranked, gold, 4) == 1.0
    assert mrr_at_k(ranked, gold, 5) == 0.5
    assert ndcg_at_k(ranked, gold, 4) == pytest.approx((1 / 1.5849625 + 1 / 2.3219281) / (1 + 1 / 1.5849625))


def test_score_run_skips_queries_without_gold():
    out = score_run([(["a"], {"a"}), (["b"], set())], ks=(1,))
    assert out == {"queries": 1.0, "hit@1": 1.0, "recall@1": 1.0, "mrr@1": 1.0, "ndcg@1": 1.0}


def test_gold_chunks_include_overlap_copies_in_same_document_only():
    chunks = [
        {"chunk_id": "c1", "document_id": "d1", "text": "Take 10 mg\nonce daily."},
        {"chunk_id": "c2", "document_id": "d1", "text": "...take 10 MG once daily. Next."},
        {"chunk_id": "c3", "document_id": "d2", "text": "take 10 mg once daily"},
    ]
    assert gold_chunks("take 10 mg once daily", "d1", chunks) == ["c1", "c2"]
    assert gold_chunks("", "d1", chunks) == []


def test_testset_roundtrip(tmp_path):
    item = TestItem(id="q1", question="Dose?", kind="answerable", gold_quote="10 mg", gold_chunk_ids=["c1"])
    write_testset(tmp_path / "t.jsonl", [item])
    assert load_testset(tmp_path / "t.jsonl") == [item]


class Verdicts(BaseModel):
    class V(BaseModel):
        title: str
        verdict: int = 0

    items: list[V]


def test_strict_schema_requires_all_fields_and_keeps_property_names():
    s = strict_schema(Verdicts.model_json_schema())
    v = s["$defs"]["V"]
    assert v["additionalProperties"] is False and v["required"] == ["title", "verdict"]
    assert "title" in v["properties"] and "default" not in v["properties"]["verdict"]


def test_judge_validates_into_response_model():
    class FakeLLM:
        def complete_json(self, system, user, schema):
            return {"items": [{"title": "t", "verdict": 1}]}, {}

    judge = ClaudeJudge(FakeLLM())
    assert judge.generate("p", Verdicts).items[0].verdict == 1
    assert asyncio.run(judge.agenerate("p", Verdicts)).items[0].title == "t" and judge.calls == 2


def test_judge_concurrency_limit():
    import threading
    import time

    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class SlowLLM:
        def complete_json(self, system, user, schema):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return {"items": []}, {}

    judge = ClaudeJudge(SlowLLM(), max_concurrency=2)

    async def many():
        await asyncio.gather(*(judge.agenerate("p", Verdicts) for _ in range(8)))

    asyncio.run(many())
    assert state["peak"] <= 2 and judge.calls == 8


def test_judge_records_outputs_with_tags_only_when_enabled():
    from medical_rag.evaluation.ragas_adapters import JUDGE_TAG

    class FakeLLM:
        def complete_json(self, system, user, schema):
            return {"items": [{"title": "t", "verdict": 1}]}, {}

    async def tagged(judge, tag):
        JUDGE_TAG.set(tag)
        return await judge.agenerate("secret prompt", Verdicts)

    async def both(judge):
        await asyncio.gather(tagged(judge, "q1:run0:faithfulness"), tagged(judge, "q2:run0:context_recall"))

    off = ClaudeJudge(FakeLLM())
    asyncio.run(both(off))
    assert off.trace == []

    on = ClaudeJudge(FakeLLM(), max_concurrency=2, record=True)
    asyncio.run(both(on))
    assert sorted(t["tag"] for t in on.trace) == ["q1:run0:faithfulness", "q2:run0:context_recall"]
    assert all(t["schema"] == "Verdicts" and t["output"]["items"][0]["verdict"] == 1 for t in on.trace)
    assert "secret prompt" not in str(on.trace)  # only a hash and the length of the prompt are kept


def test_gold_with_equivalents():
    item = TestItem(id="q1", question="?", kind="answerable", gold_chunk_ids=["a"], alt_gold_chunk_ids=["b"])
    assert item.gold_with_equivalents() == {"a", "b"}
    assert TestItem(id="q2", question="?", kind="unanswerable").gold_with_equivalents() == set()
