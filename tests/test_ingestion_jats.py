"""PMC JATS parser and JATS inline rules on a small synthetic article (no real article text); minified XML."""

import xml.etree.ElementTree as ET

import pytest

from medical_rag.ingestion.jats import iso_date, parse_jats
from medical_rag.ingestion.models import BlockKind, HeaderSource, SectionRole

ARTICLE = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="review-article"><front>\
<journal-meta><journal-title-group><journal-title>Test Journal</journal-title></journal-title-group></journal-meta>\
<article-meta><article-id pub-id-type="pmcid">PMC100</article-id><article-id pub-id-type="pmcid-ver">PMC100.2\
</article-id><article-id pub-id-type="doi">10.0/test</article-id>\
<title-group><article-title>Testamol review</article-title></title-group>\
<contrib-group><contrib><name><surname>Doe</surname><given-names>Jane</given-names></name>\
<email>jane@example.test</email></contrib></contrib-group>\
<pub-date pub-type="epub"><day>3</day><month>7</month><year>2026</year></pub-date>\
<permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/"><p>CC BY</p></license></permissions>\
<related-article related-article-type="correction-forward" xlink:href="PMC200">Correction: Testamol review\
</related-article>\
<abstract><sec><title>Background</title><p>Abstract text.</p></sec></abstract>\
<kwd-group><kwd>testamol</kwd></kwd-group></article-meta></front>\
<body><boxed-text><caption><title>Main Points</title></caption><p>Key point.</p></boxed-text>\
<sec id="s1"><title>1. Introduction</title><p>Levels exceed 10<sup>9</sup>/L in B<sub>12</sub>deficiency.</p>\
<p>Pain fell.<sup><xref ref-type="bibr" rid="r1">7</xref></sup></p><p>Bracketed [<xref ref-type="bibr" rid="r1">1\
</xref>,<xref ref-type="bibr" rid="r2">2</xref>] and bare <xref ref-type="bibr" rid="r2">3</xref> citations.</p>\
<p>See <xref ref-type="table" rid="T1">Table 1</xref>.<list list-type="bullet"><list-item><p>First</p>\
<list><list-item><p>Nested</p></list-item></list></list-item></list>After the list.</p></sec>\
<sec><title>Methods</title><sec><title>2.1. Design</title><p>Line one<break/>line two.</p></sec></sec>\
<sec><title>Ethics</title><p>Approved by a board.</p></sec></body>\
<back><ack><p>Thanks.</p></ack><fn-group><fn><p>Conflicts: none.</p></fn></fn-group>\
<ref-list><ref id="r1"><mixed-citation>Doe J. A study. 2020.</mixed-citation></ref>\
<ref id="r2"><mixed-citation>Roe R. Another. 2021.</mixed-citation></ref></ref-list></back>\
<floats-group><table-wrap id="T1"><label>Table 1</label><caption><title>Doses</title></caption><table>\
<thead><tr><th>Dose</th><th colspan="2">Cmax</th></tr></thead><tbody><tr><td>10 mg<xref ref-type="table-fn" \
rid="tf1"><sup>a</sup></xref></td><td>5</td><td>6</td></tr></tbody></table><table-wrap-foot><fn id="tf1">\
<label>a</label><p>Fasting.</p></fn></table-wrap-foot></table-wrap>\
<fig id="F1"><label>Figure 1</label><caption><p>Flow diagram.</p></caption><graphic xlink:href="f1.jpg"/></fig>\
</floats-group></article>"""


@pytest.fixture(scope="module")
def doc():
    return parse_jats(ARTICLE.encode("utf-8"), raw_file="a.xml", raw_member=None, raw_sha256="0" * 64)


@pytest.fixture(scope="module")
def sections(doc):
    return {s.section_id: s for _, s in doc.iter_sections()}


def texts(section):
    return [b.text for b in section.blocks]


def test_identity_metadata_and_correction(doc):
    assert (doc.document_id, doc.document_version, doc.title) == ("pmc_oa:PMC100.2", "2", "Testamol review")
    assert doc.source_reference == "https://pmc.ncbi.nlm.nih.gov/articles/PMC100/"
    assert doc.metadata["authors"] == ["Jane Doe"]
    assert doc.metadata["pub_date_epub"] == "2026-07-03"
    assert doc.metadata["references"] == ["Doe J. A study. 2020.", "Roe R. Another. 2021."]
    assert doc.metadata["license"] == "https://creativecommons.org/licenses/by/4.0/"
    assert [(r.relation, r.identifier, r.in_corpus) for r in doc.related_documents] == [
        ("correction-forward", "PMC200", False)
    ]


def test_no_email_anywhere(doc):
    assert "jane@example.test" not in repr(doc)


def test_abstract_and_back_matter_roles(doc, sections):
    abstract = doc.sections[0]
    assert abstract.role is SectionRole.ABSTRACT and abstract.children[0].role is SectionRole.ABSTRACT
    assert texts(abstract.children[0]) == ["Abstract text."]
    ethics = next(s for s in sections.values() if s.title == "Ethics")
    assert ethics.role is SectionRole.BACK_MATTER
    back = [s for s in doc.sections if s.role is SectionRole.BACK_MATTER]
    assert [s.title for s in back] == ["Ethics", "Acknowledgments", "Notes"]
    assert texts(back[2]) == ["Conflicts: none."]


def test_inline_rules_s1_s2_s3(sections):
    intro = sections["s1"]
    assert (intro.number, intro.title) == ("1", "Introduction")
    assert texts(intro)[:3] == [
        "Levels exceed 10^9/L in B12 deficiency.",
        "Pain fell.[7]",
        "Bracketed [1,2] and bare [3] citations.",
    ]
    design = next(s for s in sections.values() if s.title == "Design")
    assert (design.number, texts(design)) == ("2.1", ["Line one line two."])


def test_paragraph_split_around_nested_list_and_float_placed_after_citing_block(sections):
    kinds = [(b.kind, b.text.splitlines()[0]) for b in sections["s1"].blocks[3:]]
    assert kinds == [
        (BlockKind.PARAGRAPH, "See Table 1."),
        (BlockKind.LIST, "First"),
        (BlockKind.PARAGRAPH, "After the list."),
        (BlockKind.TABLE, "Table 1 Doses"),
    ]
    lst = sections["s1"].blocks[4]
    assert [(i.text, i.level) for i in lst.items] == [("First", 0), ("Nested", 1)]


def test_table_spans_header_and_footnote_markers(sections):
    table = next(b.table for b in sections["s1"].blocks if b.table)
    assert table.header_source is HeaderSource.THEAD and table.spans_expanded
    assert table.header_rows == (("Dose", "Cmax", "Cmax"),)
    assert table.body_rows == (("10 mg (footnote a)", "5", "6"),)
    assert table.footnotes == ("(footnote a) Fasting.",)


def test_table_wrap_with_two_tables_keeps_both():
    xml = ARTICLE.replace(
        "<tbody><tr><td>10 mg",
        "<tbody><tr><td>X</td><td>1</td><td>2</td></tr></tbody></table><table><thead><tr><th>Cost</th><th>Total</th>"
        "<th>Mean</th></tr></thead><tbody><tr><td>10 mg",
    )
    doc = parse_jats(xml.encode("utf-8"), raw_file="a.xml", raw_member=None, raw_sha256="0")
    tables = [b for _, s in doc.iter_sections() for b in s.blocks if b.table]
    assert [t.table.caption for t in tables] == ["Table 1 Doses (table 1 of 2)", "Table 1 Doses (table 2 of 2)"]
    assert tables[1].table.header_rows == (("Cost", "Total", "Mean"),)
    assert [t.source_pointer.rsplit("/", 1)[-1] for t in tables] == ["table[1]", "table[2]"]


def test_uncited_figure_goes_to_a_derived_section(doc):
    last = doc.sections[-1]
    assert (last.section_id, last.title) == ("floats-unreferenced", "Tables and figures")
    figure, caption = last.blocks
    assert (figure.kind, figure.source_pointer) == (BlockKind.FIGURE, "floats-group/fig[@id='F1']@f1.jpg")
    assert (caption.text, caption.source_pointer) == (
        "Figure 1 Flow diagram.",
        "floats-group/fig[@id='F1']@f1.jpg/caption",
    )


def test_loose_body_content_and_derived_ids(doc, sections):
    box = sections["path:body.0"]
    assert [b.text for b in box.blocks] == ["Main Points", "Key point."]
    assert any(sid.startswith("path:body.") for sid in sections if sid != "path:body.0")


def test_block_pointers_are_unique(doc):
    pointers = [b.source_pointer for _, s in doc.iter_sections() for b in s.blocks]
    assert len(pointers) == len(set(pointers))


@pytest.mark.parametrize(
    ("xml", "iso"),
    [
        ("<d><year>2026</year><month>7</month><day>3</day></d>", "2026-07-03"),
        ("<d><year>2013</year></d>", "2013"),
        ("<d iso-8601-date='2026-08-28 09:25'><year>2026</year></d>", "2026-08-28"),
    ],
)
def test_iso_date(xml, iso):
    assert iso_date(ET.fromstring(xml)) == iso


def test_rejects_non_jats():
    with pytest.raises(ValueError):
        parse_jats(b"<document/>", raw_file="f", raw_member=None, raw_sha256="0")
