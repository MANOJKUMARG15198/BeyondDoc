"""B2: titles and medicine names from the batch plan, and parser title rules."""

import pytest

from medical_rag.ingestion.catalog import apply_catalog
from medical_rag.ingestion.fda_pdf import is_heading
from medical_rag.ingestion.models import ParsedDocument
from medical_rag.ingestion.spl import parse_spl


def doc(source, title="Parsed title", names=()):
    return ParsedDocument(
        document_id=f"{source}:x",
        source=source,
        source_format="x",
        document_version="1",
        title=title,
        medicine_names=names,
        source_reference="https://x.test",
        raw_file="f",
        raw_member=None,
        raw_sha256="0",
        metadata={"keep": "me"},
    )


def test_fda_letter_gets_catalogue_title_and_names():
    out = apply_catalog(
        doc("fda_drugsatfda", "FDA letter NDA021402 — ENCLOSURE"),
        {"title": "Letter — Synthroid (NDA021402, ORIG 1)", "drug": "levothyroxine"},
    )
    assert out.title == "FDA Letter — Synthroid (NDA021402, ORIG 1)"
    assert out.medicine_names == ("Synthroid", "levothyroxine")
    assert out.metadata == {
        "keep": "me",
        "catalog_title": "Letter — Synthroid (NDA021402, ORIG 1)",
        "catalog_drug": "levothyroxine",
        "medicine_names_source": "batch plan",
    }


def test_pmc_keeps_its_title_and_gains_the_drug():
    out = apply_catalog(doc("pmc_oa", "Baclofen review"), {"title": "Baclofen review", "drug": "baclofen"})
    assert (out.title, out.medicine_names, out.metadata["medicine_names_source"]) == (
        "Baclofen review",
        ("baclofen",),
        "batch plan",
    )


def test_parsed_names_are_never_replaced():
    original = doc("dailymed", "Lipitor", ("Lipitor", "ATORVASTATIN CALCIUM"))
    out = apply_catalog(original, {"title": "Lipitor — ATORVASTATIN", "drug": "atorvastatin"})
    assert (out.title, out.medicine_names) == ("Lipitor", ("Lipitor", "ATORVASTATIN CALCIUM"))
    assert "medicine_names_source" not in out.metadata and out.metadata["catalog_drug"] == "atorvastatin"


def test_no_catalogue_entry_changes_nothing():
    original = doc("pmc_oa")
    assert apply_catalog(original, {}) is original


SPL = (
    '<document xmlns="urn:hl7-org:v3"><title>{title}</title><setId root="s"/><component><structuredBody>'
    '<component><section><id root="p"/><code code="48780-1"/><subject><manufacturedProduct><manufacturedProduct>'
    "<name>Brandname</name></manufacturedProduct></manufacturedProduct></subject></section></component>"
    "</structuredBody></component></document>"
)


@pytest.mark.parametrize(
    ("printed", "title"),
    [
        ("87786-081", "Brandname"),
        ("Drug Facts", "Brandname"),
        ("These highlights do not include all the information needed", "Brandname"),
        (
            "TRAZODONE HYDROCHLORIDE TABLETS These highlights do not include all the information",
            "TRAZODONE HYDROCHLORIDE TABLETS",
        ),
        ("Synthroid", "Synthroid"),
    ],
)
def test_spl_title_rules(printed, title):
    parsed = parse_spl(SPL.format(title=printed).encode("utf-8"), raw_file="f", raw_member=None, raw_sha256="0")
    assert parsed.title == title
    assert parsed.metadata["spl_title"] == printed  # the printed title is always kept


@pytest.mark.parametrize("text", ["\\,~ ~ ADMINISTRATION", "ENCLOSURES:", "~~ FOOD AND DRUG"])
def test_letterhead_noise_is_not_a_heading(text):
    assert not is_heading(text)
