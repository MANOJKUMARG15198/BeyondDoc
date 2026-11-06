"""Run the SPL parser on the 10 DailyMed labels of the benchmark sample and report what it produced.

Raw files are checksum-verified and read in memory only. Checks per label: structure counts, character
coverage (every letter/digit under a section's <title>, <text> and <excerpt> must reappear in the parse) and
"suspect words": parsed words that never occur in the source when every text node is split apart. These are
either legitimate joins (inline-split words, chemical formulas, 10^9) or merged words; examples are listed.

.venv/Scripts/python -X utf8 scripts/phase2/parse_spl_sample.py
"""

from __future__ import annotations

import csv
import io
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

from evaluate_tokenizers import MANIFESTS, OUT, SAMPLE, checked

from medical_rag.ingestion.models import BlockKind, ParsedDocument, TitleSource
from medical_rag.ingestion.spl import NS, parse_spl

WORD = re.compile(r"\w+")
MARKER = re.compile(r"\(footnote \d+\)")


def source_text(root: ET.Element, sep: str) -> str:
    parts = []
    for sec in root.iter(NS + "section"):
        for el in (sec.find(NS + "title"), sec.find(NS + "text"), *sec.findall(NS + "excerpt")):
            if el is not None:
                parts.append(sep.join(el.itertext()))
    return sep.join(parts)


def parsed_text(doc: ParsedDocument) -> str:
    parts = []
    for _, s in doc.iter_sections():
        if s.title_source is TitleSource.SOURCE:
            parts.append(s.raw_title or s.title)
        for b in s.blocks:
            if b.kind is BlockKind.TABLE:
                t = b.table
                parts += [t.caption, *(c for r in (*t.header_rows, *t.body_rows) for c in r), *t.footnotes]
            elif b.kind is not BlockKind.FIGURE:  # alt text lives outside <text>
                parts.append(b.text)
    return MARKER.sub(" ", "\n".join(parts))


def alnum(text: str) -> Counter[str]:
    return Counter(ch for ch in text if ch.isalnum())


def main() -> int:
    rows = [
        r
        for r in csv.DictReader((MANIFESTS / "batch_01_downloaded.csv").open(encoding="utf-8"))
        if r["source_name"] == "dailymed" and r["format"] == "zip" and r["source_document_id"] in SAMPLE["dailymed"]
    ]
    lines = ["# SPL parser on the 10 sample labels", ""]
    lines += [
        "| label | title | sections | derived titles | highlights | tables (thead/th/styled/none) | spans | "
        "lists (nested) | figures (with alt) | footnotes | char coverage | suspect words | changes | unknown |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    suspects_all: Counter[str] = Counter()
    start = time.perf_counter()
    for r in rows:
        with zipfile.ZipFile(io.BytesIO(checked(r["file"], r["sha256"]))) as zf:
            member = next(n for n in zf.namelist() if n.lower().endswith(".xml"))
            xml = zf.read(member)
        doc = parse_spl(xml, raw_file=r["file"], raw_member=member, raw_sha256=r["sha256"])
        root = ET.fromstring(xml)

        secs = [s for _, s in doc.iter_sections()]
        blocks = [b for s in secs for b in s.blocks]
        tables = [b.table for b in blocks if b.table is not None]
        heads = Counter(t.header_source.value for t in tables)
        lists = [b for b in blocks if b.kind is BlockKind.LIST]
        figures = [b for b in blocks if b.kind is BlockKind.FIGURE]
        footnotes = sum(len(t.footnotes) for t in tables) + sum(b.source_pointer.endswith("/footnote") for b in blocks)

        src, got = alnum(source_text(root, "")), alnum(parsed_text(doc))
        coverage = sum(min(n, got[c]) for c, n in src.items()) / max(sum(src.values()), 1)
        separated = {w.casefold() for w in WORD.findall(source_text(root, " "))}
        suspects = Counter(w for w in WORD.findall(parsed_text(doc).casefold()) if w not in separated)
        suspects_all.update(suspects)

        lines.append(
            f"| {r['source_document_id'][:8]} | {doc.title[:40]} | {len(secs)} "
            f"| {sum(s.title_source is TitleSource.DERIVED and s.role.value != 'highlights' for s in secs)} "
            f"| {sum(s.role.value == 'highlights' for s in secs)} "
            f"| {len(tables)} ({heads['thead']}/{heads['th_cells']}/{heads['styled']}/{heads['none']}) "
            f"| {sum(t.spans_expanded for t in tables)} "
            f"| {len(lists)} ({sum(any(i.level for i in b.items) for b in lists)}) "
            f"| {len(figures)} ({sum(bool(b.text) for b in figures)}) | {footnotes} | {coverage:.2%} "
            f"| {sum(suspects.values())} | {doc.normalization_changes or '-'} "
            f"| {', '.join(doc.metadata['unknown_elements']) or '-'} |"
        )
    elapsed = time.perf_counter() - start
    lines += ["", f"Parsed {len(rows)} labels in {elapsed:.2f} s (including checksum and unzip).", ""]
    lines += ["## Most frequent suspect words (all labels)", ""]
    lines += [f"- `{w}` × {n}" for w, n in suspects_all.most_common(40)]
    report = "\n".join(lines) + "\n"
    print(report)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spl_parse_sample.md").write_text(report, encoding="utf-8")
    print(f"written: {OUT / 'spl_parse_sample.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
