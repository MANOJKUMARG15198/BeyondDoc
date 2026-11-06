"""Evaluation dataset v2.0 schema and mechanical checks (no corpus files: in-memory chunks)."""

import pytest
from pydantic import ValidationError

from medical_rag.evaluation.testset_v2 import TestItemV2, check_items, load_v2, write_v2

CHUNKS = {
    "c1": {"chunk_id": "c1", "document_id": "d1", "text": "Store at 25°C (77°F).\nKeep dry.",
           "section_path": ["16 HOW SUPPLIED"], "source_reference": "https://x/d1", "document_version": 3},
    "c2": {"chunk_id": "c2", "document_id": "d1", "text": "...store at 25°C (77°F). Overlap copy.",
           "section_path": ["16 HOW SUPPLIED"], "source_reference": "https://x/d1", "document_version": 3},
    "c3": {"chunk_id": "c3", "document_id": "d2", "text": "Store at 25°C (77°F).",
           "section_path": ["HOW SUPPLIED"], "source_reference": "https://x/d2", "document_version": 1},
}  # fmt: skip


def item(**over):
    base = {
        "id": "v2-q001", "category": "ingredients", "question": "How to store it?",
        "reference_answer": "At 25°C (77°F).", "answerability": "answerable", "difficulty": "easy",
        "retrieval_type": "exact",
        "evidence": [{"document_id": "d1", "source_chunk_id": "c1", "quote": "store at 25°C  (77°F)"}],
    }  # fmt: skip
    return TestItemV2(**(base | over))


def test_quote_resolves_to_every_chunk_of_the_same_document_only():
    it = item()
    assert check_items([it], CHUNKS) == {"v2-q001": []}
    ev = it.evidence[0]
    assert ev.chunk_ids == ["c1", "c2"] and ev.section_path == "16 HOW SUPPLIED"
    assert ev.source_url == "https://x/d1" and ev.document_version == "3"
    assert it.gold_chunk_ids() == {"c1", "c2"} and it.evidence_sets() == [{"c1", "c2"}]


@pytest.mark.parametrize(
    ("over", "expected"),
    [
        ({"evidence": [{"document_id": "d1", "source_chunk_id": "c1", "quote": "store at 30°C"}]}, "quote not found"),
        ({"reference_answer": "At 25°C (77°F), up to 40°C."}, "numbers not in evidence ['40']"),
        ({"evidence": []}, "without evidence"),
        (
            {"evidence": [{"document_id": "d2", "source_chunk_id": "c1", "quote": "store at 25°C"}]},
            "does not match chunk",
        ),
    ],
)
def test_blocking_problems(over, expected):
    assert expected in " ".join(check_items([item(**over)], CHUNKS)["v2-q001"])


def test_v1_gold_chunks_cannot_be_reused_and_ids_are_unique():
    out = check_items([item(), item()], CHUNKS, excluded_chunks={"c2"})
    assert "reuses an excluded" in " ".join(out["v2-q001"]) and "duplicate id" in " ".join(out["v2-q001"])


def test_schema_rejects_bad_ids_and_unknown_fields():
    with pytest.raises(ValidationError):
        item(id="q001")
    with pytest.raises(ValidationError):
        item(gold_chunk_ids=["c1"])


def test_unanswerable_items_need_no_evidence(tmp_path):
    u = item(id="v2-q150", category="unanswerable", answerability="unanswerable", retrieval_type="none",
             reference_answer="Abstain: not in the corpus.", evidence=[])  # fmt: skip
    assert check_items([u], CHUNKS) == {"v2-q150": []}
    write_v2(tmp_path / "v2.jsonl", [u])
    assert load_v2(tmp_path / "v2.jsonl") == [u]
