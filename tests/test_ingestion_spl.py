"""SPL parser tests on a small synthetic label (no real label text); minified XML so no source whitespace helps."""

import pytest

from medical_rag.ingestion.models import BlockKind, HeaderSource, ListItem, SectionRole, TitleSource
from medical_rag.ingestion.spl import parse_spl

NDC = "2.16.840.1.113883.6.69"
LABEL = f"""<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3"><id root="doc-guid"/>\
<code code="34391-3" displayName="HUMAN PRESCRIPTION DRUG LABEL"/>\
<title>These highlights do not include all the information needed to use TESTDRUG safely.</title>\
<effectiveTime value="20260101"/><setId root="set-guid"/><versionNumber value="7"/><component><structuredBody>\
<component><section><id root="s-box"/><code code="34066-1"/><title>WARNING: RISK</title>\
<excerpt><highlight><text><paragraph>Short boxed summary.</paragraph></text></highlight></excerpt>\
<text><paragraph>Serious risk.</paragraph></text></section></component>\
<component><section><id root="s5"/><code code="43685-7"/><title>5 WARNINGS AND PRECAUTIONS</title>\
<component><section><id root="s52"/><title>5. 2 Renal Effects</title><text>\
<paragraph>Reduce dose in renal<br/>failure.</paragraph>\
<paragraph>Monitor ×10<sup>9</sup>/L and C<sub>max</sub>.</paragraph>\
</text></section></component>\
<component><section><id root="s5u"/><text><paragraph>Untitled body.</paragraph></text></section></component>\
</section></component>\
<component><section><id root="s-list"/><title>Uses</title><text>\
<list listType="unordered"><caption>Indications</caption>\
<item>Pain<list><item>Headache</item></list></item><item>Fever</item></list></text></section></component>\
<component><section><id root="s-tab"/><title>Interactions</title><text><table><caption>Table 1</caption>\
<thead><tr><th>Drug</th><th>Effect</th></tr></thead><tbody>\
<tr><td rowspan="2">Warfarin<footnote ID="f1">INR monitored</footnote></td><td>Bleeding<br/>risk</td></tr>\
<tr><td></td></tr><tr><td colspan="2">All others</td></tr></tbody><tfoot><tr><td>Source: study</td></tr></tfoot>\
</table><renderMultiMedia referencedObject="MM1"/></text><component><observationMedia ID="MM1">\
<text>Structural formula</text><value><reference value="img1.jpg"/></value></observationMedia></component>\
</section></component>\
<component><section><id root="s-styled"/><title>Doses</title><text><table><tbody>\
<tr><td><content styleCode="bold">Dose</content></td><td><content styleCode="bold">Use</content></td></tr>\
<tr><td>1 mg</td><td>daily</td></tr></tbody></table><unknownTag>Odd text</unknownTag><paragraph>log⁡x</paragraph>\
</text></section></component>\
<component><section><id root="s-pkg"/><code code="51945-4"/><title>PRINCIPAL DISPLAY PANEL</title>\
<text><paragraph>NDC 0000-0000-00</paragraph></text></section></component>\
<component><section><id root="s-prod"/><code code="48780-1"/><subject><manufacturedProduct><manufacturedProduct>\
<code code="0000-0000" codeSystem="{NDC}"/><name>Testdrug</name><formCode displayName="TABLET"/>\
<asEntityWithGeneric><genericMedicine><name>TESTAMOL</name></genericMedicine></asEntityWithGeneric>\
<asContent><containerPackagedProduct><code code="0000-0000-00" codeSystem="{NDC}"/></containerPackagedProduct>\
</asContent></manufacturedProduct></manufacturedProduct></subject></section></component>\
</structuredBody></component></document>"""


@pytest.fixture(scope="module")
def doc():
    return parse_spl(LABEL.encode("utf-8"), raw_file="label.zip", raw_member="label.xml", raw_sha256="0" * 64)


@pytest.fixture(scope="module")
def sections(doc):
    return {s.section_id: s for _, s in doc.iter_sections()}


def _blocks(text_xml):
    xml = (
        '<document xmlns="urn:hl7-org:v3"><setId root="s"/><component><structuredBody><component><section>'
        f"<id root='x'/><title>T</title><text>{text_xml}</text></section></component></structuredBody></component>"
        "</document>"
    )
    return parse_spl(xml.encode("utf-8"), raw_file="f", raw_member=None, raw_sha256="0").sections[0].blocks


def test_document_identity_and_products(doc):
    assert doc.document_id == "dailymed:set-guid"
    assert doc.document_version == "7"
    assert doc.source_reference == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=set-guid"
    assert doc.title == "Testdrug"  # the Highlights disclaimer is not used as the title
    assert doc.metadata["spl_title"].startswith("These highlights")
    assert doc.medicine_names == ("Testdrug", "TESTAMOL")
    assert doc.metadata["ndc_product_codes"] == ["0000-0000"]
    assert doc.metadata["ndc_package_codes"] == ["0000-0000-00"]
    assert doc.metadata["dosage_forms"] == ["TABLET"]
    assert doc.normalizer_version == "1"


def test_roles_are_tagged_not_excluded(sections):
    assert sections["s-box"].role is SectionRole.BOXED_WARNING
    assert sections["s-pkg"].role is SectionRole.PACKAGE_LABEL
    assert sections["s-pkg"].blocks[0].text == "NDC 0000-0000-00"
    assert sections["s-prod"].role is SectionRole.PRODUCT_DATA
    assert sections["s5"].role is SectionRole.BODY


def test_highlights_become_a_child_section(sections):
    hl = sections["s-box:highlights"]
    assert hl.role is SectionRole.HIGHLIGHTS
    assert hl.title_source is TitleSource.DERIVED
    assert [b.text for b in hl.blocks] == ["Short boxed summary."]
    assert [b.text for b in sections["s-box"].blocks] == ["Serious risk."]


def test_section_numbers_and_derived_titles(sections):
    s52 = sections["s52"]
    assert (s52.number, s52.raw_number, s52.title, s52.raw_title) == (
        "5.2",
        "5. 2",
        "Renal Effects",
        "5. 2 Renal Effects",
    )
    assert sections["s5"].number == "5"
    untitled = sections["s5u"]
    assert (untitled.title, untitled.title_source) == ("WARNINGS AND PRECAUTIONS (part 1)", TitleSource.DERIVED)


def test_line_breaks_superscripts_and_block_separators(sections):
    texts = [b.text for b in sections["s52"].blocks]
    assert texts == ["Reduce dose in renal failure.", "Monitor ×10^9/L and Cmax."]


def test_nested_list_levels_and_caption(sections):
    caption, lst = sections["s-list"].blocks
    assert (caption.kind, caption.text) == (BlockKind.CAPTION, "Indications")
    assert lst.items == (ListItem("Pain", 0), ListItem("Headache", 1), ListItem("Fever", 0))
    assert lst.text == "Pain\nHeadache\nFever"


def test_table_spans_footnotes_and_empty_cells(sections):
    table_block, figure = sections["s-tab"].blocks
    t = table_block.table
    assert t.header_source is HeaderSource.THEAD
    assert t.caption == "Table 1"
    assert t.header_rows == (("Drug", "Effect"),)
    assert t.body_rows == (
        ("Warfarin (footnote 1)", "Bleeding\nrisk"),
        ("Warfarin (footnote 1)", ""),
        ("All others", "All others"),
    )
    assert t.spans_expanded
    assert t.footnotes == ("Source: study", "(footnote 1) INR monitored")
    assert (figure.kind, figure.text, figure.source_pointer) == (
        BlockKind.FIGURE,
        "Structural formula",
        "section:s-tab/text/renderMultiMedia[1]@MM1",
    )


def test_styled_header_unknown_elements_and_changes(doc, sections):
    table_block, odd, log = sections["s-styled"].blocks
    assert table_block.table.header_source is HeaderSource.STYLED
    assert table_block.table.header_rows == (("Dose", "Use"),)
    assert odd.text == "Odd text"  # unknown element: text kept, element counted
    assert doc.metadata["unknown_elements"] == ["unknownTag:1"]
    assert log.text == "logx"
    assert doc.normalization_changes == {"U+2061": 1}


def test_th_header_and_no_header():
    (th,) = _blocks("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>")
    assert (th.table.header_source, th.table.header_rows) == (HeaderSource.TH_CELLS, (("A", "B"),))
    (plain,) = _blocks("<table><tr><td>1</td><td>2</td></tr></table>")
    assert (plain.table.header_source, plain.table.header_rows) == (HeaderSource.NONE, ())


def test_one_bold_cell_boxed_warning_is_prose_not_a_header():
    warning = "WARNING: SERIOUS RISKS. " + "Patients must be monitored closely for respiratory depression. " * 60
    (block,) = _blocks(f'<table><tbody><tr><td><content styleCode="bold">{warning}</content></td></tr></tbody></table>')
    t = block.table
    assert (t.header_source, t.header_rows, t.layout) == (HeaderSource.NONE, (), True)
    assert t.body_rows == ((warning.strip(),),)


def test_superscript_caret_only_after_a_digit():
    (p,) = _blocks("<paragraph>25 mg/m<sup>2</sup> Testdrug<sup>®</sup> 10<sup>-3</sup></paragraph>")
    assert p.text == "25 mg/m2 Testdrug® 10^-3"


@pytest.mark.parametrize(
    ("xml", "text"),
    [
        ("vitamin B\n    <sub>12</sub>levels", "vitamin B12 levels"),
        ("B<sub>12</sub>Deficiency", "B12 Deficiency"),
        ("C\n  <sub>max</sub>was higher", "Cmax was higher"),
        ("<sup>b</sup>Peak plasma level", "b Peak plasma level"),
        ("C\n  <sub>18</sub>H\n  <sub>21</sub>NO\n  <sub>4</sub>•HCl", "C18H21NO4•HCl"),
        ("Na<sub>2</sub>CO<sub>3</sub>", "Na2CO3"),
        ("dose <sub>x</sub>", "dose x"),  # a plain space before <sub> is kept
    ],
)
def test_sub_and_sup_layout_repairs(xml, text):
    (p,) = _blocks(f"<paragraph>{xml}</paragraph>")
    assert p.text == text


def test_paragraph_footnote_follows_its_block():
    para, note = _blocks('<paragraph>Take with food<footnote ID="n">If tolerated</footnote>.</paragraph>')
    assert para.text == "Take with food (footnote 1)."
    assert (note.text, note.source_pointer) == ("(footnote 1) If tolerated", "section:x/text/paragraph[1]/footnote[1]")


def test_block_pointers_are_unique():
    blocks = _blocks(
        '<paragraph>A<footnote ID="a">one</footnote> B<footnote ID="b">two</footnote>'
        '<renderMultiMedia referencedObject="M1"><caption>First</caption></renderMultiMedia>'
        '<renderMultiMedia referencedObject="M2"><caption>Second</caption></renderMultiMedia></paragraph>'
    )
    pointers = [b.source_pointer for b in blocks]
    assert len(pointers) == len(set(pointers)) == 7  # paragraph, 2 footnotes, 2 x (figure + caption)
    assert pointers[1:3] == ["section:x/text/paragraph[1]/footnote[1]", "section:x/text/paragraph[1]/footnote[2]"]
    assert pointers[3:5] == [
        "section:x/text/paragraph[1]/renderMultiMedia[1]@M1",
        "section:x/text/paragraph[1]/renderMultiMedia[1]@M1/caption",
    ]


def test_rejects_non_spl_xml():
    with pytest.raises(ValueError):
        parse_spl(b"<article/>", raw_file="f", raw_member=None, raw_sha256="0")
