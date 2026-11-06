"""v2 support in the Ragas runner: item adapter, Evidence Recall@K, format detection, per-group breakdown."""

import importlib.util
import json
from pathlib import Path

from medical_rag.evaluation.testset_v2 import TestItemV2, evidence_recall, is_v2_file, to_eval_item

ROOT = Path(__file__).resolve().parents[1]


def v2_item(**over):
    base = {
        "id": "v2-q001", "category": "multi_document", "question": "Compare A and B?", "reference_answer": "Both.",
        "answerability": "answerable", "difficulty": "hard", "retrieval_type": "cross_document", "split": "dev",
        "evidence": [
            {"document_id": "dailymed:a", "source_chunk_id": "a1", "quote": "x", "chunk_ids": ["a1", "a2"]},
            {"document_id": "pmc_oa:b", "source_chunk_id": "b1", "quote": "y", "chunk_ids": ["b1"]},
        ],
    }  # fmt: skip
    return TestItemV2(**(base | over))


def test_adapter_maps_v2_items_for_the_runner():
    e = to_eval_item(v2_item())
    assert e.kind == "answerable" and e.source == "dailymed" and e.split == "dev" and e.difficulty == "hard"
    assert e.gold_chunk_ids == ["a1", "a2", "b1"] and e.gold_with_equivalents() == {"a1", "a2", "b1"}
    assert e.evidence_sets == [{"a1", "a2"}, {"b1"}]
    u = to_eval_item(v2_item(answerability="insufficient_evidence", category="unanswerable", evidence=[],
                             retrieval_type="none"))  # fmt: skip
    assert u.kind == "unanswerable" and u.answerability == "insufficient_evidence" and u.source is None


def test_evidence_recall_counts_evidence_items_not_chunks():
    sets = [{"a1", "a2"}, {"b1"}]
    assert evidence_recall(["a2", "z"], sets, k=8) == 0.5  # one of two evidence items found
    assert evidence_recall(["a1", "a2", "z"], sets, k=8) == 0.5  # two copies of the same item count once
    assert evidence_recall(["z", "a1", "b1"], sets, k=2) == 0.5 and evidence_recall(["z", "a1", "b1"], sets) == 1.0
    assert evidence_recall(["a1"], []) is None


def test_format_detection(tmp_path):
    v2 = tmp_path / "v2.jsonl"
    v2.write_text(v2_item().model_dump_json() + "\n", encoding="utf-8")
    v1 = tmp_path / "v1.jsonl"
    v1.write_text(json.dumps({"id": "q1", "question": "?", "kind": "answerable"}) + "\n", encoding="utf-8")
    assert is_v2_file(v2) and not is_v2_file(v1)


def test_breakdown_groups_rows():
    spec = importlib.util.spec_from_file_location("eval_ragas", ROOT / "scripts/phase3/eval_ragas.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"kind": "answerable", "abstained": False, "category": "pmc", "evidence_recall_top_k": 1.0,
         "ragas": {"faithfulness": 1.0}},
        {"kind": "answerable", "abstained": True, "category": "pmc", "evidence_recall_top_k": 0.0},
        {"kind": "unanswerable", "abstained": True, "category": "unanswerable"},
    ]  # fmt: skip
    out = module.breakdown(rows, "category", 8)
    assert out["pmc"]["questions"] == 2 and out["pmc"]["false_abstentions"] == 1
    assert out["pmc"]["evidence_recall@8"] == 0.5 and out["pmc"]["ragas"]["faithfulness"] == 1.0
    assert out["unanswerable"]["not_answerable_abstained"] == "1/1"
