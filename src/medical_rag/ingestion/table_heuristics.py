"""Header inference and layout detection for tables that carry no header markup (T-1, approved 2026-11-04).

Used by every parser only after thead, th cells and bold styling have all failed. Both results are flagged
downstream (HeaderSource.INFERRED, Table.layout), never silent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_NUMERIC = re.compile(r"[\d\s.,%±+\-–−()<>≥≤×^*]+")


def is_numeric(cell: str) -> bool:
    return any(ch.isdigit() for ch in cell) and bool(_NUMERIC.fullmatch(cell.strip()))


def infer_header_rows(rows: Sequence[Sequence[str]]) -> int:
    """1 when the first row reads as column names (at least two distinct non-empty, non-numeric cells) and at
    least 30% of the non-empty cells below it are numeric; otherwise 0."""
    if len(rows) < 2:
        return 0
    labels = [c for c in rows[0] if c.strip()]
    if len(set(labels)) < 2 or any(is_numeric(c) for c in labels):
        return 0
    rest = [c for r in rows[1:] for c in r if c.strip()]
    return 1 if rest and sum(map(is_numeric, rest)) / len(rest) >= 0.3 else 0


def header_leaves_no_data(header_rows: int, rows: Sequence[Sequence[str]]) -> bool:
    """A 'header' that would take every row (e.g. a boxed warning laid out as one bold cell) is not a header."""
    return header_rows > 0 and header_rows >= len(rows)


def is_layout(rows: Sequence[Sequence[str]]) -> bool:
    """Prose laid out as a table: at least 80% of rows hold one distinct value (one column, or cells spanned
    across the row), and the distinct cells average at least 20 words. A single such row counts too."""
    if not rows:
        return False
    singles = sum(len({c for c in r if c.strip()}) <= 1 for r in rows)
    values = [c for r in rows for c in dict.fromkeys(x for x in r if x.strip())]
    if singles < 0.8 * len(rows) or not values:
        return False
    return sum(len(v.split()) for v in values) / len(values) >= 20
