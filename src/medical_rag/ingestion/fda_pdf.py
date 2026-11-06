"""FDA approval-letter PDF parser (F-1): one letter -> ParsedDocument.

`parse_letter_pages` is pure (page texts in), so it is tested without PDF files; `parse_fda_pdf` only adds pypdf
text extraction. Running headers and footers (lines repeated in the top or bottom three lines of several pages,
"Page N", "Reference ID: …", FDA letterhead) and dashed rules are removed and recorded as Artifacts. Lines are
joined into paragraphs at blank lines, headings, bullets and footnote URLs; a line ending in "-" joins the next
without a space and keeps its hyphen (these letters only break at real compound hyphens). All-caps lines are
section headings. The signature block and the electronic-signature page are LETTER_SIGNATURE sections, with the
professional names kept. Every block records its page.
"""

from __future__ import annotations

import io
import re
import statistics
from collections import Counter
from dataclasses import dataclass

from pypdf import PdfReader

from medical_rag.ingestion.models import (
    Artifact,
    Block,
    BlockKind,
    EvidenceType,
    ParsedDocument,
    Section,
    SectionRole,
    TitleSource,
)
from medical_rag.ingestion.text import NORMALIZER_VERSION, normalize_text, text_sha256

ZONE = 3  # header/footer zone: first and last non-empty lines of a page
PAGE_LINE = re.compile(r"^Page \d+( of \d+)?$", re.IGNORECASE)
REFERENCE_ID = re.compile(r"^Reference ID: \d+$")
RULE = re.compile(r"^-{5,}$")
LETTERHEAD = re.compile(
    r"^(DEPARTMENT OF HEALTH (&|AND) HUMAN SERVICES.*|Public Health Service|Food and Drug|Administration|"
    r"(Rockville|Silver Spring),? MD\s+\d{5}|U\.S\. Food (&|and) Drug Administration|www\.fda\.gov)$",
    re.IGNORECASE,
)
SIGNATURE_PAGE = re.compile(r"This is a representation of an electronic record", re.IGNORECASE)
SINCERELY = re.compile(r"^Sincerely,?$", re.IGNORECASE)
ENCLOSURE = re.compile(r"^ENCLOSURES?(\(S\))?:?$", re.IGNORECASE)
BULLET = re.compile(r"^(•|o)\s")
FOOTNOTE_URL = re.compile(r"^\d+\s+https?://")
SALUTATION = re.compile(r"^Dear\s.+[:,]$")
APPLICATION_LINE = re.compile(r"^(NDA|BLA|ANDA)\s")
HEADING_CHARS = re.compile(r"[A-Z0-9 &/(),.:;'’½\-–]+")


@dataclass(frozen=True, slots=True)
class _Line:
    page: int
    number: int  # 1-based line number on its page
    text: str


def is_heading(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return (
        len(letters) >= 4
        and text == text.upper()
        and HEADING_CHARS.fullmatch(text) is not None  # OCR letterhead noise such as "\,~ ~ ADMINISTRATION" is not
        and len(text) <= 90
        and any(len(w) >= 3 and w.isalpha() for w in re.split(r"\W+", text))
        and not APPLICATION_LINE.match(text)
        and not ENCLOSURE.match(text)
    )


def parse_fda_pdf(
    data: bytes, *, source_document_id: str, source_reference: str, raw_file: str, raw_sha256: str
) -> ParsedDocument:
    pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]
    return parse_letter_pages(
        pages,
        source_document_id=source_document_id,
        source_reference=source_reference,
        raw_file=raw_file,
        raw_sha256=raw_sha256,
    )


def parse_letter_pages(
    pages: list[str], *, source_document_id: str, source_reference: str, raw_file: str, raw_sha256: str
) -> ParsedDocument:
    lines = [_Line(p, n, t.rstrip()) for p, text in enumerate(pages, 1) for n, t in enumerate(text.splitlines(), 1)]
    artifacts, kept = _remove_artifacts(lines)
    changes: Counter[str] = Counter()
    signature_pages = {ln.page for ln in kept if SIGNATURE_PAGE.search(ln.text)}
    body = [ln for ln in kept if ln.page not in signature_pages]

    sections: list[Section] = []
    current = Section("letter.1", "Letter", TitleSource.DERIVED, evidence_type=EvidenceType.REGULATORY_ACTION)
    sections.append(current)
    per_page: Counter[int] = Counter()

    def block(page: int, text: str) -> Block:
        per_page[page] += 1
        return Block(BlockKind.PARAGRAPH, text, f"page:{page}/paragraph[{per_page[page]}]", page=page)

    for page, text in _paragraphs(body):
        text = normalize_text(text, changes=changes)
        if not text:
            continue
        starts_signature = bool(SINCERELY.match(text))
        if is_heading(text) or starts_signature or ENCLOSURE.match(text):
            role = SectionRole.LETTER_SIGNATURE if starts_signature else SectionRole.BODY
            title = "Signature" if starts_signature else text
            current = Section(
                f"letter.{len(sections) + 1}",
                title,
                TitleSource.DERIVED if starts_signature else TitleSource.SOURCE,
                role=role,
                evidence_type=EvidenceType.REGULATORY_ACTION,
            )
            sections.append(current)
            if not starts_signature:
                continue
        current.blocks.append(block(page, text))
    if signature_pages:
        esig = Section(
            f"letter.{len(sections) + 1}",
            "Electronic signature",
            TitleSource.DERIVED,
            role=SectionRole.LETTER_SIGNATURE,
            evidence_type=EvidenceType.REGULATORY_ACTION,
        )
        for ln in kept:
            if ln.page in signature_pages and (text := normalize_text(ln.text, changes=changes)):
                esig.blocks.append(block(ln.page, text))
        sections.append(esig)
    sections = [s for s in sections if s.blocks or s.role is not SectionRole.BODY or s.title != "Letter"]

    application, _, submission = source_document_id.partition(":")
    headings = [s.title for s in sections if s.title_source is TitleSource.SOURCE and not ENCLOSURE.match(s.title)]
    return ParsedDocument(
        document_id=f"fda_drugsatfda:{source_document_id}",
        source="fda_drugsatfda",
        source_format="pdf",
        document_version=submission,
        title=f"FDA letter {application}" + (f" — {headings[0]}" if headings else ""),
        medicine_names=(),
        source_reference=source_reference,
        raw_file=raw_file,
        raw_member=None,
        raw_sha256=raw_sha256,
        metadata={"application": application, "submission": submission, "pages": str(len(pages))},
        sections=sections,
        artifacts=artifacts,
        normalizer_version=NORMALIZER_VERSION,
        normalization_changes=dict(changes),
    )


def _remove_artifacts(lines: list[_Line]) -> tuple[list[Artifact], list[_Line]]:
    by_page: dict[int, list[_Line]] = {}
    for ln in lines:
        if ln.text.strip():
            by_page.setdefault(ln.page, []).append(ln)
    zone = {id(ln) for page in by_page.values() for ln in (*page[:ZONE], *page[-ZONE:])}
    pages_with: Counter[str] = Counter()  # on how many pages each line occurs (anywhere on the page)
    for page in by_page.values():
        pages_with.update({ln.text.strip() for ln in page})
    first_page = by_page.get(1, [])[:6]

    def kind(ln: _Line) -> str | None:
        text = ln.text.strip()
        if RULE.match(text):
            return "pdf_rule_line"
        if id(ln) in zone and (pages_with[text] >= 2 or PAGE_LINE.match(text) or REFERENCE_ID.match(text)):
            return "pdf_running_header_footer"
        if LETTERHEAD.match(text) and (id(ln) in zone or ln in first_page):
            return "pdf_letterhead"
        return None

    artifacts, kept = [], []
    for ln in lines:
        if ln.text.strip() and (k := kind(ln)):
            artifacts.append(Artifact(k, f"page:{ln.page}/line[{ln.number}]", text_sha256(ln.text), len(ln.text)))
        else:
            kept.append(ln)
    return artifacts, kept


def _paragraphs(lines: list[_Line]) -> list[tuple[int, str]]:
    """(page, text) paragraphs. Headings, the salutation, 'Sincerely', enclosure lines and footnote URLs stand
    alone; from 'Sincerely' to the next heading or enclosure list every line stands alone (names, titles)."""
    lengths = [len(ln.text.strip()) for ln in lines if ln.text.strip()]
    typical = statistics.median(lengths) if lengths else 0
    out: list[tuple[int, str]] = []
    current: list[str] = []
    page = 0
    in_signature = False

    def flush() -> None:
        if current:
            out.append((page, _join(current)))
            current.clear()

    for ln in lines:
        text = ln.text.strip()
        if not text:
            flush()
            continue
        if SINCERELY.match(text):
            in_signature = True
        elif is_heading(text) or ENCLOSURE.match(text):
            in_signature = False
        standalone = (
            in_signature
            or is_heading(text)
            or ENCLOSURE.match(text)
            or FOOTNOTE_URL.match(text)
            or SALUTATION.match(text)
        )
        if standalone or BULLET.match(text):
            flush()
            page = ln.page
            current.append(text)
            if standalone:
                flush()
            continue
        if current and len(current[-1]) < 0.6 * typical and current[-1].endswith((".", ":", ";")):
            flush()  # a short line ending a sentence closes the paragraph (letters without blank lines)
        if not current:
            page = ln.page
        current.append(text)
    flush()
    return out


def _join(lines: list[str]) -> str:
    text = lines[0]
    for line in lines[1:]:
        text = text + line if text.endswith("-") else f"{text} {line}"
    return text
