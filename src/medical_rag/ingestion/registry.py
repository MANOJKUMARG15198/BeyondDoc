"""Choose the source-specific parser for a raw document, then attach its batch-plan catalogue entry."""

from __future__ import annotations

from medical_rag.ingestion.catalog import apply_catalog
from medical_rag.ingestion.fda_pdf import parse_fda_pdf
from medical_rag.ingestion.jats import parse_jats
from medical_rag.ingestion.livertox import parse_livertox
from medical_rag.ingestion.models import ParsedDocument
from medical_rag.ingestion.sources import RawDocument
from medical_rag.ingestion.spl import parse_spl


def parse_raw(raw: RawDocument) -> ParsedDocument:
    return apply_catalog(_parse(raw), raw.catalog)


def _parse(raw: RawDocument) -> ParsedDocument:
    common = {"raw_file": raw.raw_file, "raw_sha256": raw.raw_sha256}
    if raw.source == "dailymed":
        return parse_spl(raw.data, raw_member=raw.raw_member, alternate_files=raw.alternate_files, **common)
    if raw.source == "pmc_oa":
        return parse_jats(raw.data, raw_member=raw.raw_member, alternate_files=raw.alternate_files, **common)
    if raw.source == "nih_livertox":
        return parse_livertox(
            raw.data,
            nbk_id=raw.source_document_id,
            raw_member=raw.raw_member,
            alternate_files=raw.alternate_files,
            **common,
        )
    if raw.source == "fda_drugsatfda":
        return parse_fda_pdf(
            raw.data, source_document_id=raw.source_document_id, source_reference=raw.source_url, **common
        )
    raise ValueError(f"no parser for source {raw.source!r}")
