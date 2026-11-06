"""LiverTox parser on a small synthetic chapter (no real chapter text); minified XML."""

import pytest

from medical_rag.ingestion.livertox import parse_livertox
from medical_rag.ingestion.models import BlockKind, EvidenceType, SectionRole

CHAPTER = """<?xml version="1.0" encoding="UTF-8"?>
<book-part-wrapper><book-meta><book-title-group><book-title>LiverTox</book-title></book-title-group></book-meta>\
<book-part><book-part-meta><title-group><title>Testamol</title></title-group>\
<pub-history><date date-type="updated"><day>30</day><month>7</month><year>2020</year></date></pub-history>\
</book-part-meta><body>\
<sec id="T.OVERVIEW"><title>OVERVIEW</title><sec id="T.Introduction"><title>Introduction</title>\
<p>Testamol is a drug.</p></sec></sec>\
<sec id="T.CASE_REPORTS"><title>CASE REPORTS</title>\
<sec id="T.Case_1_Enzymes"><title>Case 1. Enzymes rose during therapy.(1)</title><p>A patient developed jaundice.</p>\
<sec id="T.Key_Points"><title>Key Points</title><table-wrap><table><thead><tr><th>Medication:</th>\
<th>Testamol</th></tr></thead><tbody><tr><td>Pattern:</td><td>Cholestatic</td></tr></tbody></table></table-wrap></sec>\
<sec id="T.Comment"><title>Comment</title><p>Typical course.</p></sec></sec>\
<sec id="T.Case_2_Hepatitis"><title>Case 2. Hepatitis.(2)</title><p>Another patient.</p></sec></sec>\
<sec id="T.CITED_REFERENCES"><title>CITED REFERENCES</title><ref-list><ref><mixed-citation>Doe J. Case. 2001.\
</mixed-citation></ref></ref-list></sec>\
<sec id="T.ANNOTATED_BIBLIOGRAPHY"><title>ANNOTATED BIBLIOGRAPHY</title><p>References updated 2020.</p><ref-list>\
<ref><mixed-citation>Roe R. Review. 2010.</mixed-citation> (Review of liver injury.)</ref>\
<ref><mixed-citation>Poe P. Series. 2012.</mixed-citation></ref></ref-list></sec>\
<sec id="T.OTHER_REFERENCE_LINKS" sec-type="link-group"><title>OTHER REFERENCE LINKS</title><list>\
<list-item><p>PubMed search</p></list-item></list></sec></body></book-part></book-part-wrapper>"""


@pytest.fixture(scope="module")
def doc():
    return parse_livertox(
        CHAPTER.encode("utf-8"), nbk_id="NBK1", raw_file="lt.tar.gz", raw_member="lt/Testamol.nxml", raw_sha256="0" * 64
    )


@pytest.fixture(scope="module")
def sections(doc):
    return {s.section_id: s for _, s in doc.iter_sections()}


def test_identity_and_version(doc):
    assert (doc.document_id, doc.title, doc.document_version) == ("nih_livertox:NBK1", "Testamol", "2020-07-30")
    assert doc.source_reference == "https://www.ncbi.nlm.nih.gov/books/NBK1/"
    assert doc.medicine_names == ("Testamol",) and doc.raw_member == "lt/Testamol.nxml"


def test_each_case_report_is_separate_and_children_inherit(sections):
    case1, case2 = sections["T.Case_1_Enzymes"], sections["T.Case_2_Hepatitis"]
    assert (case1.role, case1.case_id, case1.evidence_type) == (
        SectionRole.CASE_REPORT,
        "T.Case_1_Enzymes",
        EvidenceType.INDIVIDUAL_CASE,
    )
    assert case2.case_id == "T.Case_2_Hepatitis"
    for child in ("T.Key_Points", "T.Comment"):
        assert (sections[child].role, sections[child].case_id) == (SectionRole.CASE_REPORT, "T.Case_1_Enzymes")
    assert sections["T.Key_Points"].blocks[0].table.body_rows == (("Pattern:", "Cholestatic"),)
    parent = sections["T.CASE_REPORTS"]
    assert (parent.role, parent.case_id) == (SectionRole.BODY, None)
    assert sections["T.Introduction"].case_id is None


def test_annotated_bibliography_is_chunkable_references(sections):
    bib = sections["T.ANNOTATED_BIBLIOGRAPHY"]
    assert bib.role is SectionRole.BIBLIOGRAPHY
    assert [(b.kind, b.text) for b in bib.blocks] == [
        (BlockKind.PARAGRAPH, "References updated 2020."),
        (BlockKind.REFERENCE, "Roe R. Review. 2010. (Review of liver injury.)"),
        (BlockKind.REFERENCE, "Poe P. Series. 2012."),
    ]


def test_structured_citation_parts_do_not_merge():
    xml = CHAPTER.replace(
        "<ref><mixed-citation>Poe P. Series. 2012.</mixed-citation></ref>",
        "<ref><element-citation><person-group><name><surname>Hamer</surname><given-names>HM</given-names></name>"
        "<name><surname>Morris</surname><given-names>HH</given-names></name></person-group><article-title>Gabapentin"
        "</article-title><source>Neurology</source><year>2000</year></element-citation></ref>",
    )
    doc = parse_livertox(xml.encode("utf-8"), nbk_id="NBK1", raw_file="f", raw_member=None, raw_sha256="0")
    bib = next(s for s in doc.sections if s.role is SectionRole.BIBLIOGRAPHY)
    assert bib.blocks[-1].text == "Hamer HM Morris HH Gabapentin Neurology 2000"


def test_cited_references_and_links_are_metadata_only(doc, sections):
    assert "T.CITED_REFERENCES" not in sections and "T.OTHER_REFERENCE_LINKS" not in sections
    assert doc.metadata["cited_references"] == ["Doe J. Case. 2001."]
    assert doc.metadata["other_reference_links"] == ["PubMed search"]


def test_rejects_non_livertox():
    with pytest.raises(ValueError):
        parse_livertox(b"<article/>", nbk_id="X", raw_file="f", raw_member=None, raw_sha256="0")
