"""Parse check for all sources: structure counts, character coverage and suspected merged words per document.

Coverage: every letter/digit of the source content that the parser is responsible for must reappear in the
parse (titles, blocks, table cells, captions, footnotes). Source content per format: SPL section titles, text and
excerpts; JATS abstract, body, back matter (without the reference list) and floats; LiverTox body without the
metadata-only reference sections; FDA page text without the recorded artifact lines. "Suspect words" are parsed
words that never occur in the source when every text node (or PDF line) is split apart: legitimate joins
(formulas, 10^9, inline-styled words) or merged words; the most frequent are listed for review.

.venv/Scripts/python -X utf8 scripts/phase2/parse_check.py --set sample20
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

from chunk_sample import SETS
from evaluate_tokenizers import OUT
from pypdf import PdfReader

from medical_rag.ingestion.livertox import METADATA_ONLY
from medical_rag.ingestion.models import BlockKind, ParsedDocument, TitleSource
from medical_rag.ingestion.registry import parse_raw
from medical_rag.ingestion.sources import RawDocument, iter_documents
from medical_rag.ingestion.spl import NS
from medical_rag.ingestion.text import text_sha256

WORD = re.compile(r"\w+")
MARKER = re.compile(r"\(footnote [^\s()]+\)|\[e-mail address removed\]")
BLOCK_ORDER = (BlockKind.PARAGRAPH, BlockKind.LIST, BlockKind.TABLE, BlockKind.FIGURE, BlockKind.REFERENCE)
HEADER_ORDER = ("thead", "th_cells", "styled", "inferred", "none")


def source_parts(raw: RawDocument, doc: ParsedDocument, sep: str) -> str:
    if raw.source == "fda_drugsatfda":
        removed = {a.text_sha256 for a in doc.artifacts}
        pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(raw.data)).pages]
        return sep.join(
            ln for p in pages for ln in (x.rstrip() for x in p.splitlines()) if text_sha256(ln) not in removed
        )
    root = ET.fromstring(raw.data)
    elements: list[ET.Element] = []
    if raw.source == "dailymed":
        for sec in root.iter(NS + "section"):
            elements += [
                e
                for e in (sec.find(NS + "title"), sec.find(NS + "text"), *sec.findall(NS + "excerpt"))
                if e is not None
            ]
    elif raw.source == "pmc_oa":
        meta = root.find("front/article-meta")
        elements += [e for e in (meta.find("title-group/article-title"), *meta.findall("abstract")) if e is not None]
        elements += [e for e in (root.find("body"), root.find("floats-group")) if e is not None]
        back = root.find("back")
        elements += [c for c in (back if back is not None else []) if c.tag != "ref-list"]
    else:
        part = root.find("book-part")
        elements += [
            e
            for e in (part.find("book-part-meta/title-group/title"), *part.findall("book-part-meta/abstract"))
            if e is not None
        ]
        elements += [s for s in part.findall("body/sec") if not (s.get("id") or "").endswith(METADATA_ONLY)]
    return sep.join(sep.join(_itertext(e)) for e in elements)


def _itertext(el: ET.Element):
    """Text of el, without <object-id> (identifiers of tables and figures, not content)."""
    if el.tag == "object-id":
        yield el.tail or ""
        return
    yield el.text or ""
    for child in el:
        yield from _itertext(child)
    yield el.tail or ""


def parsed_text(doc: ParsedDocument) -> str:
    parts = [doc.title]
    for _, s in doc.iter_sections():
        if s.title_source is TitleSource.SOURCE:
            parts.append(s.raw_title or s.title)
            if s.raw_number and s.raw_title is None:  # a <label> number kept as the section number
                parts.append(s.raw_number)
        for b in s.blocks:
            if b.table is not None:
                t = b.table
                parts += [t.caption, *(c for r in (*t.header_rows, *t.body_rows) for c in r), *t.footnotes]
            elif b.kind is not BlockKind.FIGURE or doc.source != "dailymed":  # SPL alt text lives outside <text>
                parts.append(b.text)
    return MARKER.sub(" ", "\n".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, choices=sorted(SETS))
    name = parser.parse_args().set
    lines = [
        f"# Parse check — {name}",
        "",
        "| document | source | sections | case sections | blocks (para/list/table/figure/ref) | headers "
        "(thead/th/styled/inferred/none, layout) | artifacts | char coverage | suspect words |",
        "|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    suspects_by_source: dict[str, Counter[str]] = {}
    for raw in iter_documents(SETS[name]):
        doc = parse_raw(raw)
        secs = [s for _, s in doc.iter_sections()]
        blocks = [b for s in secs for b in s.blocks]
        kinds = Counter(b.kind for b in blocks)
        tables = [b.table for b in blocks if b.table is not None]
        heads = Counter(t.header_source.value for t in tables)
        src = Counter(ch for ch in source_parts(raw, doc, "") if ch.isalnum())
        got = Counter(ch for ch in parsed_text(doc) if ch.isalnum())
        coverage = sum(min(n, got[c]) for c, n in src.items()) / max(sum(src.values()), 1)
        vocabulary = {w.casefold() for w in WORD.findall(source_parts(raw, doc, " "))}
        suspects = Counter(w for w in WORD.findall(parsed_text(doc).casefold()) if w not in vocabulary)
        suspects_by_source.setdefault(doc.source, Counter()).update(suspects)
        lines.append(
            f"| {doc.document_id} | {doc.source} | {len(secs)} | {sum(s.case_id is not None for s in secs)} "
            f"| {'/'.join(str(kinds[k]) for k in BLOCK_ORDER)} "
            f"| {'/'.join(str(heads[h]) for h in HEADER_ORDER)}, {sum(t.layout for t in tables)} "
            f"| {len(doc.artifacts)} | {coverage:.2%} "
            f"| {sum(suspects.values())} |"
        )
    for source, suspects in sorted(suspects_by_source.items()):
        lines += ["", f"## Most frequent suspect words — {source}", ""]
        lines += [f"- `{w}` × {n}" for w, n in suspects.most_common(25)] or ["- none"]
    report = "\n".join(lines) + "\n"
    print(report)
    path = OUT / f"parse_check_{name}.md"
    path.write_text(report, encoding="utf-8")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
