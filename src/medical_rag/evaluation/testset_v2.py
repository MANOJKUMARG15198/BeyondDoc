"""Evaluation dataset v2.0 schema and deterministic checks (independent of the v1 test set).

Every question carries its evidence as verbatim quotes; gold chunk ids are resolved from the quotes (every chunk of
the same document containing the quote, which covers overlap copies). Checks here are mechanical (quote present,
numbers in the reference present in the evidence, v1 gold not reused); medical correctness is a human review field.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from medical_rag.evaluation.testset import squash

Category = Literal[
    "ingredients", "interactions", "pharmacology", "multi_document", "pmc", "livertox", "fda", "unanswerable"
]
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_chunk_id: str  # the passage the question was written from
    quote: str  # verbatim (after whitespace/case normalisation) in source_chunk_id
    chunk_ids: list[str] = Field(default_factory=list)  # every chunk of the document containing the quote
    section_path: str = ""
    source_url: str = ""
    document_version: str = ""


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")

    medical_review: Literal["pending", "approved", "edited", "rejected"] = "pending"
    reviewer: str = ""
    reviewed_at: str = ""
    notes: str = ""
    absence_verified: bool | None = None  # unanswerable items: text search plus retrieval check confirmed


class TestItemV2(BaseModel):
    __test__: ClassVar[bool] = False
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^v2-q\d{3}$")
    category: Category
    question: str
    reference_answer: str
    answerability: Literal["answerable", "unanswerable", "insufficient_evidence"]
    difficulty: Literal["easy", "medium", "hard"]
    retrieval_type: Literal["exact", "semantic", "multi_hop", "cross_document", "none"]
    question_style: Literal["neutral", "personal"] = "neutral"
    evidence: list[Evidence] = Field(default_factory=list)  # empty only when not answerable
    evidence_rule: Literal["all", "any"] = "all"  # all: every evidence item is needed (multi-hop, cross-document)
    split: Literal["dev", "holdout", "unassigned"] = "unassigned"
    provenance: dict[str, Any] = Field(default_factory=dict)  # drafter, pool, seed, pool rank
    review: Review = Field(default_factory=Review)

    def gold_chunk_ids(self) -> set[str]:
        return {c for e in self.evidence for c in e.chunk_ids}

    def evidence_sets(self) -> list[set[str]]:
        """One set of acceptable chunks per evidence item (Evidence Recall@K counts the sets hit)."""
        return [set(e.chunk_ids) for e in self.evidence]


@dataclass(frozen=True)
class EvalItem:
    """What the Ragas runner needs from a test item, for v1 and v2 alike (v1 TestItem already has these names)."""

    id: str
    question: str
    kind: Literal["answerable", "unanswerable"]  # runner's view: anything not answerable must be abstained on
    reference_answer: str
    source: str | None
    review_status: str
    gold_chunk_ids: list[str]
    evidence_sets: list[set[str]] = field(default_factory=list)
    category: str | None = None
    split: str | None = None
    difficulty: str | None = None
    answerability: str | None = None

    def gold_with_equivalents(self) -> set[str]:
        return set(self.gold_chunk_ids)


def to_eval_item(item: TestItemV2) -> EvalItem:
    first = item.evidence[0].document_id if item.evidence else ""
    return EvalItem(
        id=item.id,
        question=item.question,
        kind="answerable" if item.answerability == "answerable" else "unanswerable",
        reference_answer=item.reference_answer,
        source=first.split(":", 1)[0] if first else None,
        review_status=item.review.medical_review,
        gold_chunk_ids=sorted(item.gold_chunk_ids()),
        evidence_sets=item.evidence_sets(),
        category=item.category,
        split=item.split,
        difficulty=item.difficulty,
        answerability=item.answerability,
    )


def evidence_recall(ranked: Sequence[str], evidence_sets: Sequence[set[str]], k: int | None = None) -> float | None:
    """Evidence Recall@K: share of evidence items with at least one acceptable chunk in the top k (None: no items)."""
    if not evidence_sets:
        return None
    top = set(ranked if k is None else ranked[:k])
    return round(sum(bool(s & top) for s in evidence_sets) / len(evidence_sets), 4)


def is_v2_file(path: Path) -> bool:
    with path.open(encoding="utf-8") as f:
        first = next((line for line in f if line.strip()), "")
    return bool(first) and "category" in json.loads(first)


def load_v2(path: Path) -> list[TestItemV2]:
    return [TestItemV2(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_v2(path: Path, items: Sequence[TestItemV2]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")


def resolve_evidence(item: TestItemV2, chunks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Fill chunk_ids, section, URL and version from the chunk store; return problems (empty when all quotes match)."""
    problems: list[str] = []
    for ev in item.evidence:
        src = chunks.get(ev.source_chunk_id)
        if src is None:
            problems.append(f"{item.id}: unknown chunk {ev.source_chunk_id}")
            continue
        if src["document_id"] != ev.document_id:
            problems.append(f"{item.id}: document_id {ev.document_id} does not match chunk {ev.source_chunk_id}")
            continue
        needle = squash(ev.quote)
        if not needle or needle not in squash(src["text"]):
            problems.append(f"{item.id}: quote not found in {ev.source_chunk_id}")
            continue
        ev.chunk_ids = sorted(
            c["chunk_id"] for c in chunks.values() if c["document_id"] == ev.document_id and needle in squash(c["text"])
        )
        ev.section_path = " > ".join(src.get("section_path") or [])
        ev.source_url = src.get("source_reference", "")
        ev.document_version = str(src.get("document_version", ""))
    return problems


def numbers_missing(item: TestItemV2, chunks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Numbers in the reference answer that appear in none of the evidence chunks."""
    text = " ".join(squash(chunks[e.source_chunk_id]["text"]) for e in item.evidence if e.source_chunk_id in chunks)
    return sorted({n for n in _NUMBER.findall(item.reference_answer) if n not in text})


def check_items(
    items: Sequence[TestItemV2], chunks: Mapping[str, Mapping[str, Any]], excluded_chunks: Iterable[str] = ()
) -> dict[str, list[str]]:
    """Blocking problems per item id: quotes, numbers, reuse of excluded (v1 gold) chunks, schema consistency."""
    excluded = set(excluded_chunks)
    seen: set[str] = set()
    out: dict[str, list[str]] = {}
    for item in items:
        problems = resolve_evidence(item, chunks)
        if item.id in seen:
            problems.append(f"{item.id}: duplicate id")
        seen.add(item.id)
        if item.answerability == "answerable" and not item.evidence:
            problems.append(f"{item.id}: answerable item without evidence")
        if item.answerability != "answerable" and item.evidence and item.category == "unanswerable":
            problems.append(f"{item.id}: unanswerable item should not carry gold evidence")
        if item.gold_chunk_ids() & excluded:
            problems.append(f"{item.id}: reuses an excluded (v1 gold) chunk")
        if item.answerability == "answerable" and (missing := numbers_missing(item, chunks)):
            problems.append(f"{item.id}: numbers not in evidence {missing}")
        out[item.id] = problems
    return out
