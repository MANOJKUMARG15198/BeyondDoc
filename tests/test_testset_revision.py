"""Versioned test-set revisions: only approved, verifiable changes are applied; the base items are not mutated."""

import json

import pytest

from medical_rag.evaluation.testset import TestItem
from medical_rag.evaluation.testset_revision import RevisionError, apply_changes, parse_changes

CHUNKS = [
    {"chunk_id": "g1", "document_id": "d1", "text": "Take the missed dose as soon as possible."},
    {"chunk_id": "e1", "document_id": "d1", "text": "If you miss a dose, take it when you remember."},
    {"chunk_id": "e2", "document_id": "d1", "text": "Copy: if you miss a dose, take it when you remember."},
    {"chunk_id": "p1", "document_id": "d2", "text": "For more information call 1-800-000-0000."},
]


def base():
    return [
        TestItem(id="q1", question="Missed dose?", kind="answerable", reference_answer="Take it.",
                 gold_quote="Take the missed dose", document_id="d1", gold_chunk_ids=["g1"]),
        TestItem(id="q2", question="Phone?", kind="unanswerable"),
    ]  # fmt: skip


def change(id, action, value="", status="approved"):
    return {"id": id, "action": action, "value": value, "reason": "why", "status": status}


def test_alt_gold_resolves_every_chunk_with_the_quote_and_keeps_strict_gold():
    value = json.dumps({"quote": "if you miss a dose, take it", "document_id": "d1"})
    items = base()
    revised, log = apply_changes(items, parse_changes([change("q1", "add_alt_gold", value)]), CHUNKS)
    q1 = revised[0]
    assert q1.alt_gold_chunk_ids == ["e1", "e2"] and q1.gold_chunk_ids == ["g1"]
    assert q1.gold_with_equivalents() == {"g1", "e1", "e2"} and q1.review_status == "edited"
    assert "[add_alt_gold] why" in q1.notes and log[0]["applied"]
    assert items[0].alt_gold_chunk_ids == [] and items[0].review_status == "draft"  # base not mutated


def test_relabel_answerable_sets_gold_from_the_quote():
    value = json.dumps({"quote": "call 1-800-000-0000", "document_id": "d2", "reference_answer": "Call it."})
    revised, _ = apply_changes(base(), parse_changes([change("q2", "relabel_answerable", value)]), CHUNKS)
    assert revised[1].kind == "answerable" and revised[1].gold_chunk_ids == ["p1"]


def test_only_approved_rows_are_applied():
    rows = [change("q1", "reject", status="proposed"), change("q1", "set_reference", "New.", status="")]
    revised, log = apply_changes(base(), parse_changes(rows), CHUNKS)
    assert revised[0] == base()[0] and not any(e["applied"] for e in log)


def test_reject_keeps_the_item_marked_rejected():
    revised, _ = apply_changes(base(), parse_changes([change("q2", "reject")]), CHUNKS)
    assert [i.id for i in revised] == ["q1", "q2"] and revised[1].review_status == "rejected"


@pytest.mark.parametrize(
    "row",
    [
        change("q1", "add_alt_gold", json.dumps({"quote": "not in the corpus", "document_id": "d1"})),
        change("q1", "add_alt_gold", "not json"),
        change("q9", "set_reference", "x"),
        change("q2", "relabel_answerable", json.dumps({"quote": "call 1-800-000-0000", "document_id": "d2"})),
    ],
)
def test_unverifiable_changes_fail(row):
    with pytest.raises(RevisionError):
        apply_changes(base(), parse_changes([row]), CHUNKS)


def test_unknown_action_fails():
    with pytest.raises(RevisionError):
        parse_changes([change("q1", "delete")])
