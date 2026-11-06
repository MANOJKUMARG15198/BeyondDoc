"""Versioned test-set revisions: apply human-approved changes from a change log to a base test set.

The base file is never modified. Only change-log rows with status "approved" are applied; every applied change is
verified against the chunk store (quotes must exist verbatim, after whitespace/case normalisation, in the named
document) and recorded in the item's notes.

Change-log columns: id, action, value, reason, proposed_by, status, reviewer
Actions and value formats:
  add_alt_gold        JSON {"quote": ..., "document_id": ...}: every chunk of that document containing the quote is
                      added to alt_gold_chunk_ids (equivalent evidence)
  relabel_answerable  JSON {"quote": ..., "document_id": ..., "reference_answer": ...}: kind -> answerable, gold set
                      from the quote
  relabel_unanswerable  (no value): kind -> unanswerable, gold and reference cleared
  set_reference       new reference answer text
  set_question        new question text
  reject              (no value): review_status -> rejected (kept in the file, skipped by the evaluators)
  approve             (no value): review_status -> approved, no other change
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medical_rag.evaluation.testset import TestItem, gold_chunks

ACTIONS = {
    "add_alt_gold",
    "relabel_answerable",
    "relabel_unanswerable",
    "set_reference",
    "set_question",
    "reject",
    "approve",
}


class RevisionError(ValueError):
    pass


@dataclass(frozen=True)
class Change:
    id: str
    action: str
    value: str
    reason: str
    status: str
    reviewer: str = ""
    proposed_by: str = ""


def parse_changes(rows: Iterable[Mapping[str, str]]) -> list[Change]:
    out = []
    for n, row in enumerate(rows, start=2):  # line 1 is the CSV header
        action = (row.get("action") or "").strip()
        if action not in ACTIONS:
            raise RevisionError(f"line {n}: unknown action {action!r}")
        out.append(
            Change(
                id=row["id"].strip(),
                action=action,
                value=(row.get("value") or "").strip(),
                reason=(row.get("reason") or "").strip(),
                status=(row.get("status") or "").strip().lower(),
                reviewer=(row.get("reviewer") or "").strip(),
                proposed_by=(row.get("proposed_by") or "").strip(),
            )
        )
    return out


def _quote_chunks(value: str, chunks: Sequence[Mapping[str, Any]], change: Change) -> tuple[dict, list[str]]:
    try:
        spec = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RevisionError(f"{change.id} {change.action}: value is not JSON") from exc
    ids = gold_chunks(spec.get("quote", ""), spec.get("document_id"), chunks)
    if not ids:
        raise RevisionError(f"{change.id} {change.action}: quote not found in {spec.get('document_id')}")
    return spec, ids


def apply_changes(
    items: Sequence[TestItem], changes: Sequence[Change], chunks: Sequence[Mapping[str, Any]]
) -> tuple[list[TestItem], list[dict[str, Any]]]:
    """Return (revised items in the original order, log of every change with applied/skipped and resolved ids)."""
    by_id = {i.id: i.model_copy(deep=True) for i in items}
    log: list[dict[str, Any]] = []
    for change in changes:
        entry = {**change.__dict__, "applied": False, "resolved_chunk_ids": []}
        log.append(entry)
        if change.status != "approved":
            continue
        if change.id not in by_id:
            raise RevisionError(f"{change.id}: not in the base test set")
        item = by_id[change.id]
        if change.action == "add_alt_gold":
            _, ids = _quote_chunks(change.value, chunks, change)
            item.alt_gold_chunk_ids = sorted(set(item.alt_gold_chunk_ids) | set(ids) - set(item.gold_chunk_ids))
            entry["resolved_chunk_ids"] = ids
        elif change.action == "relabel_answerable":
            spec, ids = _quote_chunks(change.value, chunks, change)
            if not spec.get("reference_answer"):
                raise RevisionError(f"{change.id} relabel_answerable: reference_answer is required")
            item.kind, item.gold_quote, item.document_id = "answerable", spec["quote"], spec["document_id"]
            item.reference_answer, item.gold_chunk_ids = spec["reference_answer"], ids
            entry["resolved_chunk_ids"] = ids
        elif change.action == "relabel_unanswerable":
            item.kind, item.reference_answer, item.gold_quote = "unanswerable", "", ""
            item.gold_chunk_ids, item.alt_gold_chunk_ids, item.document_id = [], [], None
        elif change.action == "set_reference":
            item.reference_answer = change.value
        elif change.action == "set_question":
            item.question = change.value
        if change.action == "reject":
            item.review_status = "rejected"
        elif change.action == "approve":
            item.review_status = "approved" if item.review_status != "edited" else "edited"
        else:
            item.review_status = "edited"
        note = f"[{change.action}] {change.reason}".strip()
        item.notes = f"{item.notes}; {note}" if item.notes else note
        entry["applied"] = True
    return [by_id[i.id] for i in items], log
