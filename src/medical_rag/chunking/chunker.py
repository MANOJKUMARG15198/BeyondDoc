"""Structure-aware recursive chunking of one ParsedDocument (ADR-010).

Each section's own blocks form the unit: chunks never cross a section, so never a LiverTox case report or a boxed
warning. Inside a section, consecutive text blocks form a run that is split (Stage A) and packed (Stage B);
tables and figures become their own chunks. Overlap is added after packing and only inside a run. Every chunk's
complete text (prefix + separator + text) is counted before it is emitted; content that cannot fit raises
ChunkingError instead of being truncated. The chunker reads no files and calls no model or search service.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from medical_rag.chunking import splitter
from medical_rag.chunking.config import CHUNKER_VERSION, ChunkingConfig
from medical_rag.chunking.filters import figure_decision, section_decision
from medical_rag.chunking.models import Chunk, DocumentRecord, Exclusion, SourceSpan, TableRef, chunk_id
from medical_rag.chunking.tables import split_table
from medical_rag.chunking.tokenizer import ChunkingError, Counts
from medical_rag.ingestion.models import Block, BlockKind, HeaderSource, ParsedDocument, Section, SectionRole


@dataclass(slots=True)
class ChunkResult:
    document: DocumentRecord
    chunks: list[Chunk]
    exclusions: list[Exclusion]


def heading(section: Section) -> str:
    return f"{section.number} {section.title}" if section.number else section.title


def chunk_document(doc: ParsedDocument, cfg: ChunkingConfig, counter: Counts) -> ChunkResult:
    builder = _Builder(doc, cfg, counter)
    kept: dict[SectionRole, list[str]] = {SectionRole.HIGHLIGHTS: [], SectionRole.ABSTRACT: []}
    metadata_only = {SectionRole.HIGHLIGHTS: cfg.highlights, SectionRole.ABSTRACT: cfg.abstracts}
    for ancestors, section in doc.iter_sections():
        decision = section_decision(section, cfg)
        if decision.keep:
            builder.section(ancestors, section)
            continue
        if section.blocks:
            builder.exclude(section, None, decision.reason or "", sum(len(b.text) for b in section.blocks))
        if metadata_only.get(section.role) == "metadata_only":
            kept[section.role] += [b.text for b in section.blocks]
    record = _document_record(doc, kept[SectionRole.HIGHLIGHTS], kept[SectionRole.ABSTRACT])
    return ChunkResult(record, builder.chunks, builder.exclusions)


@dataclass(frozen=True, slots=True)
class _Unit:
    section: Section
    path: list[str]
    prefix: str  # heading + separator, exactly as counted


class _Builder:
    def __init__(self, doc: ParsedDocument, cfg: ChunkingConfig, counter: Counts) -> None:
        self.doc, self.cfg, self.counter = doc, cfg, counter
        self.chunks: list[Chunk] = []
        self.exclusions: list[Exclusion] = []
        self.doc_notices = (
            ["corrected_article"] if any("correction" in r.relation for r in doc.related_documents) else []
        )

    def exclude(self, section: Section, pointer: str | None, reason: str, chars: int) -> None:
        self.exclusions.append(
            Exclusion(
                document_id=self.doc.document_id,
                section_id=section.section_id,
                block_pointer=pointer,
                reason=reason,
                char_count=chars,
            )
        )

    def section(self, ancestors: tuple[Section, ...], section: Section) -> None:
        path = [heading(s) for s in (*ancestors, section)]
        prefix = (
            self.cfg.prefix_template.format(title=self.doc.title, path=" > ".join(path)) + self.cfg.prefix_separator
        )
        if section.blocks and self.counter.count(prefix) >= self.cfg.budget:
            raise ChunkingError(f"heading prefix alone exceeds the budget in section {section.section_id}")
        unit = _Unit(section, path, prefix)
        run: list[Block] = []
        blocks, i = section.blocks, 0
        while i < len(blocks):
            block = blocks[i]
            if block.kind is BlockKind.FIGURE:
                nxt = blocks[i + 1] if i + 1 < len(blocks) else None
                caption = (
                    nxt
                    if nxt
                    and nxt.kind is BlockKind.CAPTION
                    and nxt.source_pointer.startswith(block.source_pointer + "/")
                    else None
                )
                self._run(unit, run)
                run = []
                self._figure(unit, block, caption)
                i += 2 if caption else 1
                continue
            if block.kind is BlockKind.TABLE and block.table is not None:
                self._run(unit, run)
                run = []
                self._table(unit, block)
            elif block.text.strip():
                run.append(block)
            i += 1
        self._run(unit, run)

    def _run(self, unit: _Unit, run: list[Block]) -> None:
        if not run:
            return
        text = "\n".join(b.text for b in run)
        bounds, pos = [], 0
        for b in run:
            bounds.append((pos, pos + len(b.text)))
            pos += len(b.text) + 1
        n = self.cfg.overlap_tokens
        base_budget = self.cfg.budget - (n if self.cfg.overlap_fit == "reserve" else 0)
        limit = base_budget - self.counter.count(unit.prefix)
        spans = splitter.pack(
            text, splitter.pieces(text, bounds, self.counter, limit), self.counter, unit.prefix, base_budget
        )
        for k, span in enumerate(spans):
            start = span.start
            if k and n:
                previous = spans[k - 1]
                room = self.cfg.budget - self.counter.count(unit.prefix + text[span.start : span.end])
                start = splitter.overlap_start(text, previous, self.counter, min(n, room), self.cfg.overlap_method)
                if start >= previous.end or self.counter.count(unit.prefix + text[start : span.end]) > self.cfg.budget:
                    start = span.start  # nothing fits: no overlap rather than an oversized chunk
            overlap = _block_spans(run, bounds, start, spans[k - 1].end) if start < span.start else []
            self._emit(
                unit,
                "text",
                text[start : span.end],
                _block_spans(run, bounds, span.start, span.end),
                overlap,
                overlap_tokens=self.counter.count(text[start : span.start]) if overlap else 0,
                levels=list(span.levels),
            )

    def _table(self, unit: _Unit, block: Block) -> None:
        table = block.table
        assert table is not None
        parts = split_table(
            table, unit.prefix, self.counter, self.cfg.budget, self.cfg.table_format, self.cfg.table_notes_max_share
        )
        if table.layout:
            notices = ["table_layout_prose"]
        elif table.header_source is HeaderSource.INFERRED:
            notices = ["table_header_inferred"]
        else:
            notices = ["table_header_missing"] if table.header_source is HeaderSource.NONE else []
        for part in parts:
            ref = TableRef(
                block_pointer=block.source_pointer,
                header_rows=len(table.header_rows),
                rows=part.rows,
                row_count=len(table.body_rows),
                row_part=part.row_part,
                notes_only=part.notes_only,
            )
            kind = "table_notes" if part.notes_only else "table"
            self._emit(unit, kind, part.text, [], [], table=ref, levels=list(part.levels), notices=notices)

    def _figure(self, unit: _Unit, figure: Block, caption: Block | None) -> None:
        cap = caption.text if caption else ""
        decision = figure_decision(figure.text, cap, self.cfg)
        if not decision.keep:
            for b in (figure, caption):
                if b is not None:
                    self.exclude(unit.section, b.source_pointer, decision.reason or "", len(b.text))
            return
        lines = [f"Figure: {cap}" if cap else "Figure", f"Description: {figure.text}" if figure.text else ""]
        body = "\n".join(line for line in lines if line)
        sources = [b for b in (figure, caption) if b is not None and b.text]
        if self.counter.count(unit.prefix + body) <= self.cfg.budget:
            spans = [SourceSpan(block_pointer=b.source_pointer, start=0, end=len(b.text)) for b in sources]
            self._emit(unit, "figure", body, spans, [], levels=["block"])
        else:
            self._run(unit, [b for b in (caption, figure) if b is not None and b.text.strip()])

    def _emit(
        self,
        unit: _Unit,
        kind: str,
        body: str,
        content: list[SourceSpan],
        overlap: list[SourceSpan],
        *,
        levels: list[str],
        overlap_tokens: int = 0,
        table: TableRef | None = None,
        notices: list[str] | None = None,
    ) -> None:
        tokens = self.counter.count(unit.prefix + body)
        if tokens > self.cfg.budget:
            raise ChunkingError(
                f"{kind} chunk of {tokens} tokens exceeds {self.cfg.budget} in {unit.section.section_id}"
            )
        flags = (["boxed_warning"] if unit.section.role is SectionRole.BOXED_WARNING else []) + self.doc_notices
        flags += notices or []
        if any(level in splitter.TOKEN_CUTS for level in levels):
            flags.append("token_split")
        # BERT splits on whitespace and punctuation before its per-word length limit, so only an unbroken
        # run of letters/digits longer than the limit becomes [UNK]; long chemical names with hyphens do not.
        if re.search(rf"[^\W_]{{{self.cfg.long_word_chars + 1},}}", body):
            flags.append("long_word_unk")
        doc, section = self.doc, unit.section
        cid = chunk_id(
            document_id=doc.document_id,
            document_version=doc.document_version,
            normalizer_version=doc.normalizer_version,
            section_id=section.section_id,
            kind=kind,
            content_spans=content,
            table=table,
            config_id=self.cfg.config_id,
        )
        self.chunks.append(
            Chunk(
                chunk_id=cid,
                ordinal=len(self.chunks),
                document_id=doc.document_id,
                document_version=doc.document_version,
                source=doc.source,
                source_reference=doc.source_reference,
                title=doc.title,
                section_id=section.section_id,
                section_path=unit.path,
                section_role=section.role.value,
                evidence_type=section.evidence_type.value,
                case_id=section.case_id,
                medicine_names=list(doc.medicine_names),
                kind=kind,
                prefix=unit.prefix[: len(unit.prefix) - len(self.cfg.prefix_separator)],
                text=body,
                token_count=tokens,
                content_spans=content,
                overlap_spans=overlap,
                overlap_tokens=overlap_tokens,
                table=table,
                split_levels=levels,
                notices=flags,
                config_id=self.cfg.config_id,
                normalizer_version=doc.normalizer_version,
                chunker_version=CHUNKER_VERSION,
            )
        )


def _block_spans(run: list[Block], bounds: list[tuple[int, int]], start: int, end: int) -> list[SourceSpan]:
    """Map a range of the run text back to per-block offsets into each Block.text."""
    spans = []
    for block, (a, b) in zip(run, bounds, strict=True):
        lo, hi = max(start, a), min(end, b)
        if lo < hi:
            spans.append(SourceSpan(block_pointer=block.source_pointer, start=lo - a, end=hi - a))
    return spans


def _document_record(doc: ParsedDocument, highlights: list[str], abstracts: list[str]) -> DocumentRecord:
    return DocumentRecord(
        document_id=doc.document_id,
        document_version=doc.document_version,
        source=doc.source,
        source_format=doc.source_format,
        title=doc.title,
        medicine_names=list(doc.medicine_names),
        source_reference=doc.source_reference,
        raw_file=doc.raw_file,
        raw_member=doc.raw_member,
        raw_sha256=doc.raw_sha256,
        alternate_files=list(doc.alternate_files),
        related_documents=[asdict(r) for r in doc.related_documents],
        metadata=dict(doc.metadata),
        normalizer_version=doc.normalizer_version,
        normalization_changes=dict(doc.normalization_changes),
        highlights=highlights,
        abstracts=abstracts,
    )
