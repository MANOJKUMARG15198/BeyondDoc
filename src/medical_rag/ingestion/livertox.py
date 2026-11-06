"""LiverTox NXML parser: one chapter (book-part) -> ParsedDocument (pure: bytes in, no file access).

Uses the JATS block builder. Each case report ("Case N. ...") is its own CASE_REPORT section with case_id = its
sec@id and evidence_type INDIVIDUAL_CASE; its child sections (Key Points, Laboratory Values, Comment) inherit
both, so a case's tables stay linked to the case (L-1). The annotated bibliography is a BIBLIOGRAPHY section
with one REFERENCE block per entry (P2-6, chunked); cited references and other reference links are metadata
only. The version is the chapter's "updated" date.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from medical_rag.ingestion.jats import Builder, iso_date
from medical_rag.ingestion.models import ParsedDocument, Section, SectionRole, TitleSource
from medical_rag.ingestion.text import NORMALIZER_VERSION
from medical_rag.ingestion.xml_text import text_of

SOURCE_URL = "https://www.ncbi.nlm.nih.gov/books/{}/"
CASE_TITLE = re.compile(r"^Case \d+\s*[.:]", re.IGNORECASE)
METADATA_ONLY = ("OTHER_REFERENCE_LINKS", "CITED_REFERENCE", "CITED_REFERENCES")


def parse_livertox(
    xml: bytes,
    *,
    nbk_id: str,
    raw_file: str,
    raw_member: str | None,
    raw_sha256: str,
    alternate_files: tuple[str, ...] = (),
) -> ParsedDocument:
    root = ET.fromstring(xml)
    part = root if root.tag == "book-part" else root.find("book-part")
    if part is None:
        raise ValueError(f"not a LiverTox book-part: root element {root.tag!r}")
    meta = part.find("book-part-meta")
    plain = Builder()
    title = text_of(meta.find("title-group/title"), plain.ctx) if meta is not None else ""
    updated = iso_date(meta.find(".//date[@date-type='updated']")) if meta is not None else ""

    sections: list[Section] = []
    for k, abstract in enumerate(meta.findall("abstract") if meta is not None else [], 1):
        sid = abstract.get("id") or f"abstract[{k}]"
        blocks = plain.blocks(abstract, f"section:{sid}", SectionRole.ABSTRACT)
        if blocks:
            sections.append(Section(sid, "Abstract", TitleSource.DERIVED, role=SectionRole.ABSTRACT, blocks=blocks))

    cited, links = [], []
    body = part.find("body")
    for i, sec in enumerate(body.findall("sec") if body is not None else [], 1):
        sid = sec.get("id") or ""
        if sid.endswith(METADATA_ONLY):
            texts = [text_of(r, plain.ctx) for r in sec.iter("ref")] or [
                text_of(li, plain.ctx) for li in sec.iter("list-item")
            ]
            texts = texts or [text_of(p, plain.ctx) for p in sec.iter("p")]
            (links if sid.endswith("OTHER_REFERENCE_LINKS") else cited).extend(t for t in texts if t)
            continue
        if sid.endswith("ANNOTATED_BIBLIOGRAPHY"):
            bib = Builder(references_as_blocks=True)
            bib.ctx, bib.unknown = plain.ctx, plain.unknown  # one set of counters for the whole document
            sections.append(bib.section(sec, f"body.{i}", SectionRole.BIBLIOGRAPHY))
            continue
        sections.append(
            plain.section(sec, f"body.{i}", SectionRole.BODY, case_rule=lambda t: bool(CASE_TITLE.match(t)))
        )

    metadata: dict[str, str | list[str]] = {
        "updated": updated,
        "cited_references": cited,
        "other_reference_links": links,
        "formulas": str(plain.ctx.formulas),
        "unknown_elements": [f"{k}:{v}" for k, v in sorted(plain.unknown.items())],
    }
    return ParsedDocument(
        document_id=f"nih_livertox:{nbk_id}",
        source="nih_livertox",
        source_format="livertox_nxml",
        document_version=updated,
        title=title or nbk_id,
        medicine_names=(title,) if title else (),
        source_reference=SOURCE_URL.format(nbk_id),
        raw_file=raw_file,
        raw_member=raw_member,
        raw_sha256=raw_sha256,
        alternate_files=alternate_files,
        metadata=metadata,
        sections=sections,
        normalizer_version=NORMALIZER_VERSION,
        normalization_changes=dict(plain.ctx.changes),
    )
