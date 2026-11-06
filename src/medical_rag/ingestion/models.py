"""Source-neutral structure produced by the source parsers (chunking itself is pending ADR-010).

Parsers turn SPL XML, JATS XML, LiverTox NXML and PDF letters into a ParsedDocument; chunking reads
only this representation and never the raw files. Parsers tag content with a SectionRole but never
exclude it: content rules are applied later by the chunking filters, from configuration.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum


class BlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CAPTION = "caption"
    REFERENCE = "reference"
    FIGURE = "figure"  # image: text is its alt text (no OCR); source_pointer names the media object


class SectionRole(StrEnum):
    BODY = "body"  # ordinary narrative content
    ABSTRACT = "abstract"  # PMC abstract
    HIGHLIGHTS = "highlights"  # SPL "Highlights of Prescribing Information" (P2-18)
    BOXED_WARNING = "boxed_warning"  # SPL 34066-1
    PACKAGE_LABEL = "package_label"  # SPL 51945-4 principal display panel
    PRODUCT_DATA = "product_data"  # SPL 48780-1 structured listing data
    CASE_REPORT = "case_report"  # one individual published case
    BIBLIOGRAPHY = "bibliography"  # reference lists, LiverTox annotated bibliography (P2-6)
    BACK_MATTER = "back_matter"  # funding, conflicts of interest, contributions, ethics (P2-19)
    LETTER_SIGNATURE = "letter_signature"  # FDA letter signature block


class EvidenceType(StrEnum):
    GENERAL = "general"  # label text, review/article text, monograph text
    INDIVIDUAL_CASE = "individual_case"  # one published case report
    REGULATORY_ACTION = "regulatory_action"  # FDA approval-letter text


class TitleSource(StrEnum):
    SOURCE = "source"  # title present in the raw document
    DERIVED = "derived"  # untitled section; title built by the parser (e.g. parent title + ordinal)


class HeaderSource(StrEnum):
    THEAD = "thead"  # rows inside <thead>
    TH_CELLS = "th_cells"  # leading rows made of <th> cells
    STYLED = "styled"  # leading rows detected from bold/header styling
    INFERRED = "inferred"  # no header markup; first row reads as column names over numeric data (T-1, flagged)
    NONE = "none"  # no header row could be identified


@dataclass(frozen=True, slots=True)
class Table:
    caption: str
    header_rows: tuple[tuple[str, ...], ...]
    body_rows: tuple[tuple[str, ...], ...]  # rectangular grid after span expansion
    header_source: HeaderSource
    spans_expanded: bool  # True when colspan/rowspan values were repeated into covered cells
    footnotes: tuple[str, ...] = ()
    layout: bool = False  # prose laid out as a table (one logical column, long cells): chunked as prose (T-1)


@dataclass(frozen=True, slots=True)
class ListItem:
    text: str  # normalized with keep_newlines=True (an item may hold several paragraphs)
    level: int = 0  # 0 = top level; each nested list adds 1 (S4)


@dataclass(frozen=True, slots=True)
class Block:
    kind: BlockKind
    text: str  # NFC-normalized text; table serialization for embedding is pending (P2-3)
    source_pointer: str  # path into the raw file, e.g. "section[3]/text/table[1]" or "page:2"
    table: Table | None = None
    items: tuple[ListItem, ...] = ()  # LIST blocks only; `text` is the items joined by newlines
    page: int | None = None


@dataclass(frozen=True, slots=True)
class Artifact:
    """Format noise a parser removed (not a content decision); recorded so coverage checks can account for it."""

    kind: str  # e.g. "pdf_page_header", "pdf_page_footer", "letterhead_ocr_noise"
    source_pointer: str
    text_sha256: str
    char_count: int


@dataclass(frozen=True, slots=True)
class RelatedDocument:
    relation: str  # source vocabulary, e.g. JATS related-article-type "correction-forward"
    identifier: str  # e.g. "PMC13006926"
    title: str = ""
    in_corpus: bool = False  # whether the related document was collected


@dataclass(slots=True)
class Section:
    section_id: str  # SPL section GUID, JATS/NXML sec@id, or a derived ordinal path when the source has none
    title: str
    title_source: TitleSource
    role: SectionRole = SectionRole.BODY
    raw_title: str | None = None  # title as printed, when it differs from `title`
    number: str | None = None  # normalized "7.1" when the source numbers it
    raw_number: str | None = None  # as printed, e.g. "5. 2"
    code: str | None = None  # LOINC section code for SPL
    evidence_type: EvidenceType = EvidenceType.GENERAL
    case_id: str | None = None
    blocks: list[Block] = field(default_factory=list)
    children: list[Section] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocument:
    document_id: str  # "<source>:<source_document_id>" (docs/architecture.md)
    source: str  # "dailymed" | "pmc_oa" | "nih_livertox" | "fda_drugsatfda"
    source_format: str  # "spl_xml" | "jats_xml" | "livertox_nxml" | "pdf"
    document_version: str
    title: str
    medicine_names: tuple[str, ...]
    source_reference: str  # canonical public URL
    raw_file: str
    raw_member: str | None  # member inside a ZIP/tar, if any
    raw_sha256: str  # SHA-256 of raw_file as recorded in the batch manifest
    alternate_files: tuple[str, ...] = ()  # other representations (e.g. PDF) that are not chunked
    related_documents: tuple[RelatedDocument, ...] = ()
    metadata: dict[str, str | list[str]] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    normalizer_version: str = ""  # text.NORMALIZER_VERSION used to build this document
    normalization_changes: dict[str, int] = field(default_factory=dict)  # code point -> count; U+FFFD fails validation

    def iter_sections(self) -> Iterator[tuple[tuple[Section, ...], Section]]:
        """Yield (ancestors, section) depth-first in document order."""
        stack: list[tuple[tuple[Section, ...], Section]] = [((), s) for s in reversed(self.sections)]
        while stack:
            ancestors, section = stack.pop()
            yield ancestors, section
            stack.extend(((*ancestors, section), c) for c in reversed(section.children))
