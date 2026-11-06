"""Table serialization and row-wise splitting (M4, rule E).

Tables keep their structure in the parse (ingestion.models.Table); this module only renders it as text. Every part
names its table and body-row range, repeats the caption and all header rows ("pipe") or carries the full column
label in every cell ("key_value"), so multi-row headers and unit rows travel with the data. Footnotes referenced
by "(footnote N)" in a part's rows (or in the caption or header) go into that part. General notes repeat while
they stay within a share of the budget, otherwise they become notes-only parts. A row too large for one part is
rendered as labelled cells ("Header / Unit: value") across parts; a header too large for any part raises
ChunkingError. Nothing is dropped: empty cells stay as empty pipe fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from medical_rag.chunking.splitter import split
from medical_rag.chunking.tokenizer import ChunkingError, Counts
from medical_rag.ingestion.models import Table

MARKER = re.compile(r"\(footnote ([^\s()]+)\)")  # SPL numbers footnotes; JATS uses labels such as a, b, *
_PLACEHOLDER = 99  # widest part number assumed while packing; the final text is re-counted


@dataclass(frozen=True, slots=True)
class TablePart:
    text: str
    rows: tuple[int, int]  # 1-based inclusive body rows; (0, 0) for header-only or notes-only parts
    row_part: tuple[int, int] | None = None
    notes_only: bool = False
    levels: tuple[str, ...] = ("rows",)


def cell(value: str) -> str:
    """Cell text for one line: '|' escaped, in-cell line breaks as '; '."""
    return value.replace("|", "\\|").replace("\n", "; ")


def column_labels(table: Table) -> tuple[str, ...]:
    """Per column, the distinct header-row texts joined with ' / ' (so "Cmax / (ng/mL)"), else "Column N"."""
    width = max((len(r) for r in (*table.header_rows, *table.body_rows)), default=0)
    labels = []
    for j in range(width):
        parts: list[str] = []
        for row in table.header_rows:
            value = cell(row[j]) if j < len(row) else ""
            if value and (not parts or parts[-1] != value):
                parts.append(value)
        labels.append(" / ".join(parts) or f"Column {j + 1}")
    return tuple(labels)


def title_line(table: Table) -> str:
    return f"Table: {cell(table.caption)}" if table.caption else "Table"


def serialize(
    table: Table,
    rows: tuple[int, int],
    fmt: Literal["pipe", "key_value", "prose"],
    notes: list[str],
    labels: tuple[str, ...],
) -> str:
    """Caption line with the row range, header rows (pipe), the body rows a..b, then notes."""
    first, last = rows
    title = title_line(table) + (f" (rows {first}–{last} of {len(table.body_rows)})" if first else "")
    lines = [title]
    if fmt == "pipe":
        lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in table.header_rows]
    for i in range(first, last + 1) if first else ():
        lines.append(_row_line(table, labels, i, fmt))
    return "\n".join([*lines, *notes])


def split_table(
    table: Table,
    prefix: str,
    counter: Counts,
    budget: int,
    fmt: Literal["pipe", "key_value"],
    notes_share: float,
) -> list[TablePart]:
    """Parts whose complete text (prefix + part) fits the budget, covering every body row and every note.

    A layout table (Table.layout: prose in table cells) is rendered as prose lines whatever `fmt` is.
    """
    fmt = "prose" if table.layout else fmt
    labels = column_labels(table)
    referenced = {m[1]: note for note in table.footnotes if (m := MARKER.match(note))}
    general_raw = [note for note in table.footnotes if not MARKER.match(note)]
    general = [f"Note: {note}" for note in general_raw]
    header_text = "\n".join([table.caption, *(" ".join(r) for r in table.header_rows)])
    header_notes = _notes_for(header_text, referenced)

    def fits(body: str) -> bool:
        return counter.count(prefix + body) <= budget

    inline_general = general if general and counter.count("\n".join(general)) <= notes_share * budget else []
    if not fits(serialize(table, (0, 0), fmt, [*header_notes, *inline_general], labels)):
        inline_general = []
        if not fits(serialize(table, (0, 0), fmt, header_notes, labels)):
            raise ChunkingError(f"table header exceeds the budget: {title_line(table)[:60]!r}")

    def render(first: int, last: int) -> str:
        rows_text = "\n".join(" ".join(r) for r in table.body_rows[first - 1 : last])
        notes = _unique([*header_notes, *_notes_for(rows_text, referenced), *inline_general])
        return serialize(table, (first, last), fmt, notes, labels)

    parts: list[TablePart] = []
    n, i = len(table.body_rows), 1
    while i <= n:
        if not fits(render(i, i)):
            row_notes = _unique([*header_notes, *_notes_for(" ".join(table.body_rows[i - 1]), referenced)])
            parts += _row_parts(table, labels, i, row_notes, prefix, counter, budget)
            i += 1
            continue
        last = i
        while last < n and fits(render(i, last + 1)):
            last += 1
        parts.append(TablePart(render(i, last), (i, last)))
        i = last + 1
    if n == 0:
        parts.append(TablePart(serialize(table, (0, 0), fmt, [*header_notes, *inline_general], labels), (0, 0)))
    if general and not inline_general:
        notes = [("Note", g) for g in general_raw]
        parts += _labelled_parts(f"{title_line(table)} (notes", notes, prefix, counter, budget, notes_only=True)
    return parts


def _row_line(table: Table, labels: tuple[str, ...], i: int, fmt: str) -> str:
    row = table.body_rows[i - 1]
    if fmt == "pipe":
        return "| " + " | ".join(cell(v) for v in row) + " |"
    if fmt == "prose":
        return "\n".join(dict.fromkeys(v for v in row if v.strip()))  # spanned cells repeat one value
    return f"Row {i}: " + "; ".join(f"{label}: {cell(v)}" for label, v in zip(labels, row, strict=False) if v)


def _notes_for(text: str, referenced: dict[str, str]) -> list[str]:
    return _unique([referenced[n] for n in MARKER.findall(text) if n in referenced])


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _row_parts(
    table: Table, labels: tuple[str, ...], i: int, notes: list[str], prefix: str, counter: Counts, budget: int
) -> list[TablePart]:
    """One oversized row as labelled cells (layout tables: unlabelled prose), split across parts."""
    row = table.body_rows[i - 1]
    if table.layout:
        cells = [("", value) for value in dict.fromkeys(v for v in row if v.strip())]
    else:
        cells = [(label, cell(v)) for label, v in zip(labels, row, strict=False) if v]
    cells += [("Note", note) for note in notes]
    lead = f"{title_line(table)} (row {i} of {len(table.body_rows)}"
    return [
        TablePart(p.text, (i, i), p.row_part, levels=p.levels)
        for p in _labelled_parts(lead, cells, prefix, counter, budget)
    ]


def _labelled_parts(
    lead: str,
    cells: list[tuple[str, str]],
    prefix: str,
    counter: Counts,
    budget: int,
    *,
    notes_only: bool = False,
) -> list[TablePart]:
    """Pack 'label: value' lines (bare values when the label is empty) under '<lead>, part k of n)';
    split over-long values word-safely."""
    probe = f"{lead}, part {_PLACEHOLDER} of {_PLACEHOLDER})"
    lines: list[str] = []
    levels = {"labelled_cells"} if any(label for label, _ in cells) else {"prose_rows"}
    for label, value in cells:
        head = f"{label}: " if label else ""
        line = head + value
        if counter.count(prefix + probe + "\n" + line) <= budget:
            lines.append(line)
            continue
        continued = f"{label} (continued): " if label else ""
        limit = budget - counter.count(prefix + probe + "\n" + continued)
        if limit < 1:
            raise ChunkingError(f"table column label exceeds the budget: {label[:60]!r}")
        pieces = split(value, 0, len(value), counter, limit)
        levels |= {p.level for p in pieces}
        lines += [(head if k == 0 else continued) + value[p.start : p.end] for k, p in enumerate(pieces)]
    groups: list[list[str]] = []
    for line in lines:
        if groups and counter.count(prefix + "\n".join([probe, *groups[-1], line])) <= budget:
            groups[-1].append(line)
        else:
            groups.append([line])
    groups = groups or [[]]
    parts = []
    for k, group in enumerate(groups, 1):
        text = "\n".join([f"{lead}, part {k} of {len(groups)})", *group])
        if counter.count(prefix + text) > budget:
            raise ChunkingError(f"labelled table part does not fit: {text[:60]!r}")
        row_part = None if notes_only else (k, len(groups))
        parts.append(TablePart(text, (0, 0), row_part, notes_only, tuple(sorted(levels))))
    return parts
