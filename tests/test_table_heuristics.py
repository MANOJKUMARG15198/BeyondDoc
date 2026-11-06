import pytest

from medical_rag.chunking.tables import split_table
from medical_rag.ingestion.models import HeaderSource, Table
from medical_rag.ingestion.table_heuristics import infer_header_rows, is_layout, is_numeric


@pytest.mark.parametrize(
    ("cell", "numeric"),
    [
        ("10%", True),
        ("-12.4 (-36.6, 82.7)", True),
        ("26,811 ± 7055", True),
        ("1/2025", False),
        ("LDL‑C", False),
        ("Placebo (n=195)", False),
        ("", False),
    ],
)
def test_is_numeric(cell, numeric):
    assert is_numeric(cell) is numeric


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([("Dose", "N", "LDL-C"), ("Placebo", "21", "4"), ("10 mg", "22", "-39")], 1),  # names over numbers
        ([("", "Placebo (N=12)", "10 mg (N=37)"), ("TG", "-12.4", "-41.0")], 1),  # empty stub cell allowed
        ([("Potential impact: x", "Potential impact: x"), ("Drug", "Effect"), ("Antacids", "Reduce")], 0),  # spanned
        ([("Indications (1)", "1/2025"), ("Dosage", "Removed 1/2025")], 0),  # label/value, not numeric data
        ([("adults", "1 tablet daily"), ("children", "ask a doctor")], 0),
        ([("Dose", "N")], 0),  # one row only
    ],
)
def test_infer_header_rows(rows, expected):
    assert infer_header_rows(rows) == expected


def test_layout_detection():
    prose = "What is the medicine? " + "word " * 40
    assert is_layout([(prose, prose, prose), ("Do not take it if " + "word " * 30,) * 3, (prose,) * 3])
    assert not is_layout([("Dose", "N"), ("Placebo", "21")])
    assert not is_layout([("short",), ("also short",)])  # one column but not prose
    assert is_layout([(prose,)])  # a single prose cell (e.g. a boxed warning laid out as a table)
    assert not is_layout([])


def test_layout_table_is_chunked_as_prose_with_row_references(fake_counter):
    prose = "What is the medicine? It lowers cholesterol.\nTake it daily."
    t = Table(
        "",
        (),
        ((prose, prose), ("Do not take it if allergic.", "Do not take it if allergic.")),
        HeaderSource.NONE,
        True,
        (),
        layout=True,
    )
    (part,) = split_table(t, "Drug — Patient Information\n", fake_counter, 512, "pipe", 0.25)
    assert part.text.splitlines() == ["Table (rows 1–2 of 2)", *prose.splitlines(), "Do not take it if allergic."]
    assert "|" not in part.text and part.rows == (1, 2)


def test_oversized_layout_row_is_split_without_column_labels(fake_counter):
    prose = "Sentence " + " ".join(["word"] * 50) + ". Another sentence " + " ".join(["word"] * 50) + "."
    t = Table("", (), ((prose,), ("Short row.",)), HeaderSource.NONE, False, (), layout=True)
    parts = split_table(t, "Drug — Guide\n", fake_counter, 60, "pipe", 0.25)
    row1 = [p for p in parts if p.rows == (1, 1)]
    assert len(row1) > 1 and all("Column 1" not in p.text for p in row1)
    assert all(fake_counter.count("Drug — Guide\n" + p.text) <= 60 for p in parts)
