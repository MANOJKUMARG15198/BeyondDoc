"""Retrieval/RAG test set: questions with an exact quoted gold span; gold chunks are every chunk containing it."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

_SPACE = re.compile(r"\s+")


def squash(text: str) -> str:
    return _SPACE.sub(" ", text).strip().lower()


class TestItem(BaseModel):
    __test__: ClassVar[bool] = False  # not a pytest test class
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    kind: Literal["answerable", "unanswerable"]
    reference_answer: str = ""  # short answer from the gold span (Ragas reference); empty when unanswerable
    gold_quote: str = ""  # exact text from the source chunk
    source_chunk_id: str | None = None
    document_id: str | None = None
    source: str | None = None
    section_role: str | None = None
    review_status: Literal["draft", "approved", "rejected", "edited"] = "draft"
    notes: str = ""
    gold_chunk_ids: list[str] = Field(default_factory=list)
    # Human-approved chunks that state the same evidence in other words or another section (v1.1+). Reported
    # separately from the strict gold so the two can be compared; empty for every v1 item.
    alt_gold_chunk_ids: list[str] = Field(default_factory=list)

    def gold_with_equivalents(self) -> set[str]:
        return set(self.gold_chunk_ids) | set(self.alt_gold_chunk_ids)


def gold_chunks(quote: str, document_id: str | None, chunks: Iterable[Mapping[str, Any]]) -> list[str]:
    """Chunks of the same document whose body contains the quote (whitespace and case normalised)."""
    needle = squash(quote)
    if not needle:
        return []
    return [
        c["chunk_id"]
        for c in chunks
        if (document_id is None or c["document_id"] == document_id) and needle in squash(c["text"])
    ]


def load_testset(path: Path) -> list[TestItem]:
    return [TestItem(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_testset(path: Path, items: Sequence[TestItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
