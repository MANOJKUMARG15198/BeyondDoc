"""Post-chunking checks V1-V11 and coverage statistics (rule B, M5, section 7 of the Phase 2 review).

Coverage counts only original content spans; overlap spans, generated prefixes, repeated table headers and
generated table/figure labels are not coverage, and approved exclusions are subtracted from what must be covered.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from medical_rag.chunking.chunker import ChunkResult
from medical_rag.chunking.config import ChunkingConfig
from medical_rag.chunking.tables import MARKER, serialize
from medical_rag.chunking.tokenizer import Counts
from medical_rag.ingestion.models import BlockKind, ParsedDocument, SectionRole


@dataclass(slots=True)
class Validation:
    failures: list[str] = field(default_factory=list)
    text_chars: int = 0  # non-whitespace characters of kept text/figure blocks
    covered_chars: int = 0
    table_rows: int = 0
    covered_rows: int = 0

    @property
    def text_coverage(self) -> float:
        return self.covered_chars / self.text_chars if self.text_chars else 1.0

    @property
    def row_coverage(self) -> float:
        return self.covered_rows / self.table_rows if self.table_rows else 1.0


def validate(doc: ParsedDocument, result: ChunkResult, cfg: ChunkingConfig, counter: Counts) -> Validation:
    v = Validation()
    fail = v.failures.append
    sections = {s.section_id: s for _, s in doc.iter_sections()}
    owners = {}
    pointers: Counter[str] = Counter()
    for s in sections.values():
        for b in s.blocks:
            pointers[b.source_pointer] += 1
            owners[b.source_pointer] = (s.section_id, b)
    for pointer, n in pointers.items():
        if n > 1:
            fail(f"duplicate block pointer {pointer!r} ({n}x)")
    excluded_sections = {e.section_id for e in result.exclusions if e.block_pointer is None}
    excluded_blocks = {e.block_pointer for e in result.exclusions if e.block_pointer}

    for cid, n in Counter(c.chunk_id for c in result.chunks).items():
        if n > 1:
            fail(f"V9 duplicate chunk id {cid}")
    covered: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    rows: defaultdict[str, set[int]] = defaultdict(set)
    notes_text: defaultdict[str, str] = defaultdict(str)
    chunked_sections = set()
    for c in result.chunks:
        where = f"chunk {c.ordinal} ({c.section_id})"
        tokens = counter.count(c.prefix + cfg.prefix_separator + c.text)
        if tokens != c.token_count or tokens > cfg.budget:
            fail(f"V1 {where}: {tokens} tokens (recorded {c.token_count}, budget {cfg.budget})")
        if not c.text.strip():
            fail(f"V2 {where}: empty")
        section = sections.get(c.section_id)
        if section is None or c.case_id != section.case_id:
            fail(f"V7 {where}: section or case_id mismatch")
        if c.section_id in excluded_sections:
            fail(f"V11 {where}: chunk from an excluded section")
        if c.overlap_tokens > cfg.overlap_tokens or (c.overlap_spans and c.kind != "text"):
            fail(f"V8 {where}: overlap {c.overlap_tokens} > {cfg.overlap_tokens} or outside a text run")
        for span in [*c.content_spans, *c.overlap_spans]:
            owner = owners.get(span.block_pointer)
            if owner is None or owner[0] != c.section_id:
                fail(f"V7 {where}: span {span.block_pointer!r} outside its section")
            elif span.block_pointer in excluded_blocks:
                fail(f"V11 {where}: span from excluded block {span.block_pointer!r}")
            elif owner[1].text[span.start : span.end] not in c.text:
                fail(f"V3 {where}: span text of {span.block_pointer!r} not found verbatim in the chunk")
        for span in c.content_spans:
            covered[span.block_pointer].append((span.start, span.end))
        if c.table:
            rows[c.table.block_pointer].update(range(c.table.rows[0], c.table.rows[1] + 1) if c.table.rows[0] else ())
            if c.table.notes_only:
                notes_text[c.table.block_pointer] += c.text
        chunked_sections.add(c.section_id)

    for c in result.chunks:
        if c.table is None or c.table.notes_only:
            continue
        table = owners[c.table.block_pointer][1].table
        if c.table.row_part is None and cfg.table_format == "pipe":
            head = serialize(table, (0, 0), "pipe", [], ()).splitlines()[1:]  # header rows exactly as rendered
            if not all(line in c.text.splitlines() for line in head) or (table.caption and table.caption not in c.text):
                fail(f"V5 chunk {c.ordinal}: table caption or header rows missing")
        for number in MARKER.findall(c.text):
            note = next((f for f in table.footnotes if f.startswith(f"(footnote {number})")), None)
            if note and note not in c.text and note not in notes_text[c.table.block_pointer]:
                fail(f"V6 chunk {c.ordinal}: footnote {number} text missing")

    for sid, section in sections.items():
        if sid in excluded_sections:
            continue
        if section.role is SectionRole.BOXED_WARNING and any(b.text.strip() for b in section.blocks):
            if sid not in chunked_sections:
                fail(f"V10 boxed warning {sid} has no chunk")
        for b in section.blocks:
            if b.source_pointer in excluded_blocks:
                continue
            if b.kind is BlockKind.TABLE and b.table is not None:
                expected = set(range(1, len(b.table.body_rows) + 1))
                v.table_rows += len(expected)
                v.covered_rows += len(expected & rows[b.source_pointer])
                if expected - rows[b.source_pointer]:
                    fail(f"V4 table {b.source_pointer!r}: rows not covered {sorted(expected - rows[b.source_pointer])}")
                continue
            marks = [False] * len(b.text)
            for start, end in covered[b.source_pointer]:
                marks[start:end] = [True] * (end - start)
            need = [k for k, ch in enumerate(b.text) if not ch.isspace()]
            missing = sum(not marks[k] for k in need)
            v.text_chars += len(need)
            v.covered_chars += len(need) - missing
            if missing:
                fail(f"V3 block {b.source_pointer!r}: {missing} characters not covered")
    return v
