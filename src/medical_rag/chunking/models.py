"""Chunk, exclusion and document records: the JSONL contract of the chunking stage (M5, M8).

Offsets are code points into one block's normalized `Block.text`, identified by its `source_pointer`, which
leads back to the XML element (or PDF page) in the raw file named by the document record. A chunk lists its
original content spans separately from its overlap spans; the prefix and the generated table/figure labels are
not source text. A table chunk does not map to one contiguous text span: it names its table and body-row range.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceSpan(_Record):
    block_pointer: str
    start: int  # code points into that block's normalized Block.text
    end: int


class TableRef(_Record):
    block_pointer: str
    header_rows: int
    rows: tuple[int, int]  # 1-based inclusive body rows; (0, 0) for a header-only or notes-only part
    row_count: int
    row_part: tuple[int, int] | None = None  # (k, n) when one oversized row is split into n labelled parts
    notes_only: bool = False


class Chunk(_Record):
    chunk_id: str
    ordinal: int  # position within the document
    document_id: str
    document_version: str
    source: str
    source_reference: str
    title: str
    section_id: str
    section_path: list[str]
    section_role: str
    evidence_type: str
    case_id: str | None
    medicine_names: list[str]
    kind: Literal["text", "table", "table_notes", "figure"]
    prefix: str  # generated heading; embedding input is prefix + separator + text
    text: str
    token_count: int  # complete chunk text (prefix + separator + text), BERT uncased, no special tokens
    content_spans: list[SourceSpan]  # original content this chunk is responsible for
    overlap_spans: list[SourceSpan] = []  # repeated from the previous chunk; not counted as coverage
    overlap_tokens: int = 0
    table: TableRef | None = None
    split_levels: list[str]
    notices: list[str]
    config_id: str
    normalizer_version: str
    chunker_version: str


class Exclusion(_Record):
    document_id: str
    section_id: str
    block_pointer: str | None  # None: the whole section
    reason: str
    char_count: int


class DocumentRecord(_Record):
    document_id: str
    document_version: str
    source: str
    source_format: str
    title: str
    medicine_names: list[str]
    source_reference: str
    raw_file: str
    raw_member: str | None
    raw_sha256: str
    alternate_files: list[str]
    related_documents: list[dict[str, str | bool]]
    metadata: dict[str, str | list[str]]
    normalizer_version: str
    normalization_changes: dict[str, int]
    highlights: list[str] = []  # filled only when highlights="metadata_only"
    abstracts: list[str] = []  # filled only when abstracts="metadata_only"


def chunk_id(
    *,
    document_id: str,
    document_version: str,
    normalizer_version: str,
    section_id: str,
    kind: str,
    content_spans: list[SourceSpan],
    table: TableRef | None,
    config_id: str,
) -> str:
    """32 hex characters (128 bits) of SHA-256 over the chunk's identity, as canonical JSON.

    Any change to the document version, normalizer, section, content span or table rows, or to any setting in
    config_id (overlap, tokenizer identity, ceiling, chunker version...), gives a new ID.
    """
    identity = {
        "document_id": document_id,
        "document_version": document_version,
        "normalizer_version": normalizer_version,
        "section_id": section_id,
        "kind": kind,
        "content_spans": [s.model_dump() for s in content_spans],
        "table": table.model_dump() if table else None,
        "config_id": config_id,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]
