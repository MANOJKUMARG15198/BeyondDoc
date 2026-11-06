"""Attach the batch plan's curated title and drug to a parsed document (B2).

The Phase 1 plan manifest records, for every collected document, a curated title and the drug it was selected
for. FDA letters and PMC articles do not name their medicine in a parsable place, so their chunk prefixes would
carry no drug name. This step fills `medicine_names` when the parser found none, gives FDA letters their
catalogue title ("Letter — Synthroid (NDA021402, ORIG 1)"), and always records both values and their origin in
the metadata. Parsed text is never changed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace

from medical_rag.ingestion.models import ParsedDocument

LETTER_BRAND = re.compile(r"—\s*(.+?)\s*\(")  # "Letter — Synthroid (NDA021402, ORIG 1)" -> "Synthroid"


def apply_catalog(doc: ParsedDocument, entry: Mapping[str, str]) -> ParsedDocument:
    title, drug = (entry.get("title") or "").strip(), (entry.get("drug") or "").strip()
    if not title and not drug:
        return doc
    metadata = {**doc.metadata, "catalog_title": title, "catalog_drug": drug}
    names = doc.medicine_names
    new_title = doc.title
    if doc.source == "fda_drugsatfda" and title:
        new_title = f"FDA {title}"
        brand = LETTER_BRAND.search(title)
        names = names or tuple(n for n in (brand[1] if brand else "", drug) if n)
    elif not names and drug:
        names = (drug,)
    if names != doc.medicine_names:
        metadata["medicine_names_source"] = "batch plan"
    return replace(doc, title=new_title, medicine_names=names, metadata=metadata)
