import pytest

from medical_rag.chunking.tables import column_labels, serialize, split_table
from medical_rag.chunking.tokenizer import ChunkingError
from medical_rag.ingestion.models import HeaderSource, Table

PREFIX = "Drug — 12.3 Pharmacokinetics\n"
HEADER = (("Parameter", "Cmax"), ("", "(ng/mL)"))  # second header row carries the unit


def table(rows, footnotes=(), header=HEADER, caption="PK values"):
    return Table(caption, header, tuple(rows), HeaderSource.THEAD, False, tuple(footnotes))


def test_small_table_keeps_headers_units_notes_and_escapes_pipes(fake_counter):
    t = table([("Dose 1 (footnote 1)", "12.3"), ("Dose 2", "15|1")], ["(footnote 1) Fasting", "Source: study"])
    (part,) = split_table(t, PREFIX, fake_counter, 512, "pipe", 0.25)
    assert part.rows == (1, 2)
    assert part.text.splitlines() == [
        "Table: PK values (rows 1–2 of 2)",
        "| Parameter | Cmax |",
        "|  | (ng/mL) |",
        "| Dose 1 (footnote 1) | 12.3 |",
        "| Dose 2 | 15\\|1 |",
        "(footnote 1) Fasting",
        "Note: Source: study",
    ]


def test_large_table_splits_by_complete_rows_and_repeats_header(fake_counter):
    rows = [(f"Dose {i}" + (" (footnote 1)" if i == 30 else ""), f"{i}.0") for i in range(1, 41)]
    t = table(rows, ["(footnote 1) Fasting"])
    parts = split_table(t, PREFIX, fake_counter, 80, "pipe", 0.25)
    assert len(parts) > 1
    covered = [r for p in parts for r in range(p.rows[0], p.rows[1] + 1)]
    assert covered == list(range(1, 41))  # every row once, in order
    for p in parts:
        lines = p.text.splitlines()
        assert lines[0] == f"Table: PK values (rows {p.rows[0]}–{p.rows[1]} of 40)"
        assert lines[1:3] == ["| Parameter | Cmax |", "|  | (ng/mL) |"]  # header and unit row repeated
        assert fake_counter.count(PREFIX + p.text) <= 80
        assert ("(footnote 1) Fasting" in lines) == (p.rows[0] <= 30 <= p.rows[1])  # note travels with its row


def test_long_general_notes_become_notes_only_parts(fake_counter):
    t = table([("Dose 1", "12")], ["Source: " + "word " * 40])
    parts = split_table(t, PREFIX, fake_counter, 80, "pipe", 0.25)
    assert [p.notes_only for p in parts] == [False, True]
    assert "Note" not in parts[0].text
    assert parts[1].text.startswith("Table: PK values (notes, part 1 of 1)\nNote: Source: word")


def test_oversized_row_becomes_labelled_cells(fake_counter):
    t = table([("Dose 1", "12"), ("Dose 2", "x " * 120)])
    parts = split_table(t, PREFIX, fake_counter, 80, "pipe", 0.25)
    row2 = [p for p in parts if p.rows == (2, 2)]
    assert len(row2) > 1
    assert [p.row_part for p in row2] == [(k, len(row2)) for k in range(1, len(row2) + 1)]
    assert row2[0].text.splitlines()[0] == f"Table: PK values (row 2 of 2, part 1 of {len(row2)})"
    assert "Cmax / (ng/mL): x" in row2[0].text  # label carries header and unit
    assert all("Cmax / (ng/mL) (continued): x" in p.text for p in row2[1:] if "Cmax" in p.text)
    assert all(fake_counter.count(PREFIX + p.text) <= 80 for p in parts)
    assert "labelled_cells" in row2[0].levels


def test_header_too_large_raises(fake_counter):
    t = table([("a", "b")], header=(("word " * 60, "x"),))
    with pytest.raises(ChunkingError):
        split_table(t, PREFIX, fake_counter, 40, "pipe", 0.25)


def test_key_value_format(fake_counter):
    t = table([("Dose 1", "12.3"), ("Dose 2", "")])
    (part,) = split_table(t, PREFIX, fake_counter, 512, "key_value", 0.25)
    assert part.text.splitlines() == [
        "Table: PK values (rows 1–2 of 2)",
        "Row 1: Parameter: Dose 1; Cmax / (ng/mL): 12.3",
        "Row 2: Parameter: Dose 2",
    ]


def test_header_only_table(fake_counter):
    (part,) = split_table(table([]), PREFIX, fake_counter, 512, "pipe", 0.25)
    assert part.rows == (0, 0)
    assert part.text.splitlines() == ["Table: PK values", "| Parameter | Cmax |", "|  | (ng/mL) |"]


def test_column_labels_dedupe_spans_and_fallback():
    spanned = table([("1", "2")], header=(("PK", "PK"), ("Cmax", "AUC")))
    assert column_labels(spanned) == ("PK / Cmax", "PK / AUC")
    assert column_labels(table([("1", "2")], header=())) == ("Column 1", "Column 2")


def test_serialize_marks_row_range():
    text = serialize(table([("a", "b"), ("c", "d")]), (2, 2), "pipe", [], ("Parameter", "Cmax"))
    assert text.splitlines()[0] == "Table: PK values (rows 2–2 of 2)"
    assert text.splitlines()[-1] == "| c | d |"
