"""Export readable text of each parsed document (DailyMed, PMC, LiverTox) for writing test questions by hand.

Read-only on the raw files: they are checksum-verified and parsed in memory; output goes to data/eval/source_text/.
Text is the parsed, normalised text the chunks are built from (section headings, paragraphs, lists, tables).

.venv/Scripts/python -X utf8 scripts/phase3/export_source_text.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from medical_rag.ingestion.models import BlockKind
from medical_rag.ingestion.registry import parse_raw
from medical_rag.ingestion.sources import iter_documents

OUT = Path("data/eval/source_text")
SOURCES = ("dailymed", "pmc_oa", "nih_livertox")


def slug(text: str, n: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:n] or "untitled"


def table_text(block) -> str:
    t = block.table
    if t is None:
        return block.text
    lines = [f"Table: {t.caption}"] if t.caption else []
    for row in [*t.header_rows, *t.body_rows]:
        lines.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    lines += [f"Note: {f}" for f in t.footnotes]
    return "\n".join(lines)


def render(doc) -> str:
    out = [
        f"# {doc.title}",
        "",
        f"- document_id: {doc.document_id}",
        f"- version: {doc.document_version}",
        f"- medicines: {', '.join(doc.medicine_names)}",
        f"- source: {doc.source_reference}",
        "",
    ]
    for ancestors, section in doc.iter_sections():
        path = " > ".join(s.title for s in (*ancestors, section))
        out += [f"{'#' * min(len(ancestors) + 2, 6)} {section.title}", f"[section: {path} | role: {section.role}]", ""]
        for block in section.blocks:
            if block.kind == BlockKind.TABLE:
                out.append(table_text(block))
            elif block.kind == BlockKind.LIST and block.items:
                out += [f"{'  ' * item.level}- {item.text}" for item in block.items]
            else:
                out.append(block.text)
            out.append("")
    return "\n".join(out)


def main() -> int:
    index = []
    for raw in iter_documents(lambda source, sid: source in SOURCES):
        doc = parse_raw(raw)
        text = render(doc)
        folder = OUT / doc.source
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{slug(doc.title, 50)}__{slug(doc.document_id.split(':', 1)[1], 40)}.md"
        (folder / name).write_text(text, encoding="utf-8")
        index.append([doc.source, doc.title, ", ".join(doc.medicine_names[:3]), doc.document_id,
                      doc.source_reference, len(text.split()), f"{doc.source}/{name}"])  # fmt: skip
    index.sort()
    with (OUT / "index.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["source", "title", "medicines", "document_id", "url", "words", "file"])
        w.writerows(index)
    counts = {s: sum(1 for r in index if r[0] == s) for s in SOURCES}
    print(f"wrote {len(index)} documents {counts}, {sum(r[5] for r in index):,} words -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
