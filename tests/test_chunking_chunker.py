"""Chunker + validation on a synthetic ParsedDocument (no real label text), with the fake counter."""

import copy
from dataclasses import replace

import pytest

from medical_rag.chunking.chunker import chunk_document
from medical_rag.chunking.config import ChunkingConfig
from medical_rag.chunking.filters import figure_decision
from medical_rag.chunking.validation import validate
from medical_rag.ingestion.models import (
    Block,
    BlockKind,
    EvidenceType,
    HeaderSource,
    ParsedDocument,
    RelatedDocument,
    Section,
    SectionRole,
    Table,
    TitleSource,
)

OPEN = dict(
    overlap_tokens=0,
    overlap_method="sentence",
    overlap_fit="reserve",
    highlights="exclude",
    abstracts="include",
    table_format="pipe",
)


def sentence(i, n):
    return " ".join([f"S{i}", *["word"] * (n - 2)]) + "."


def para(pointer, text):
    return Block(BlockKind.PARAGRAPH, text, pointer)


def sec(sid, title, blocks, role=SectionRole.BODY, number=None, children=(), **kw):
    return Section(
        sid, title, TitleSource.SOURCE, role=role, number=number, blocks=list(blocks), children=list(children), **kw
    )


def make_doc():
    pk = Table(
        "PK values",
        (("Parameter", "Cmax"), ("", "(ng/mL)")),
        tuple((f"Dose {i}" + (" (footnote 1)" if i == 7 else ""), f"{i}.0") for i in range(1, 61))
        + (("Dose X", "x " * 600),),  # one oversized row
        HeaderSource.THEAD,
        False,
        ("(footnote 1) Fasting state", "Source: pooled studies"),
    )
    long_text = " ".join(sentence(i, 100) for i in range(1, 8))  # 700 tokens
    sections = [
        sec(
            "box",
            "WARNING: BLEEDING",
            [para("box/p1", "Serious risk of bleeding. Monitor closely.")],
            role=SectionRole.BOXED_WARNING,
            children=[
                sec(
                    "box:highlights",
                    "WARNING (highlights)",
                    [para("box/hl", "Boxed summary.")],
                    role=SectionRole.HIGHLIGHTS,
                )
            ],
        ),
        sec(
            "s1",
            "INDICATIONS",
            [para("s1/p1", long_text)],
            number="1",
            children=[
                sec("s1.1", "Adults", [para("s1.1/p1", " ".join(sentence(i, 85) for i in range(4)))], number="1.1")
            ],
        ),
        sec("otc", "Purpose", [para("otc/p1", "Antihistamine")]),
        sec(
            "pk",
            "Pharmacokinetics",
            [
                para("pk/p1", "See the table."),
                Block(BlockKind.TABLE, "reading text", "pk/table1", table=pk),
                Block(BlockKind.FIGURE, "Structural formula of the drug", "pk/mm1@M1"),
                Block(BlockKind.CAPTION, "Figure 1: Chemical structure", "pk/mm1@M1/caption"),
                Block(BlockKind.FIGURE, "NDC 0000-1111-22 carton label", "pk/mm2@M2"),
                Block(BlockKind.FIGURE, "img", "pk/mm3@M3"),
                para("pk/p2", "Absorption " + "-" * 600 + " is fast."),  # unbroken 600-token 'word'
            ],
            number="12.3",
        ),
        sec(
            "case1",
            "Case 1",
            [para("case1/p1", sentence(1, 60))],
            role=SectionRole.CASE_REPORT,
            case_id="case-1",
            evidence_type=EvidenceType.INDIVIDUAL_CASE,
        ),
        sec(
            "case2",
            "Case 2",
            [para("case2/p1", sentence(2, 60))],
            role=SectionRole.CASE_REPORT,
            case_id="case-2",
            evidence_type=EvidenceType.INDIVIDUAL_CASE,
        ),
        sec("pkg", "PRINCIPAL DISPLAY PANEL", [para("pkg/p1", "NDC 0000-1111-22")], role=SectionRole.PACKAGE_LABEL),
    ]
    return ParsedDocument(
        document_id="dailymed:test",
        source="dailymed",
        source_format="spl_xml",
        document_version="3",
        title="Testdrug",
        medicine_names=("Testdrug",),
        source_reference="https://example.test/label",
        raw_file="data/raw/test.zip",
        raw_member="test.xml",
        raw_sha256="0" * 64,
        related_documents=(RelatedDocument("correction-forward", "PMC000"),),
        sections=sections,
        normalizer_version="1",
    )


def run(counter, doc=None, **changes):
    cfg = ChunkingConfig(**{**OPEN, **changes})
    doc = doc or make_doc()
    result = chunk_document(doc, cfg, counter)
    return result, validate(doc, result, cfg, counter), cfg


def by_section(result, sid):
    return [c for c in result.chunks if c.section_id == sid]


def test_valid_and_complete(fake_counter):
    result, check, cfg = run(fake_counter)
    assert check.failures == []
    assert check.text_coverage == 1.0 and check.row_coverage == 1.0  # nothing silently omitted
    assert all(c.token_count <= cfg.budget for c in result.chunks)


def test_340_tokens_intact_and_700_tokens_packed(fake_counter):
    result, _, _ = run(fake_counter)
    assert len(by_section(result, "s1.1")) == 1
    long = by_section(result, "s1")
    assert len(long) == 2 and long[0].text.endswith(sentence(5, 100))


@pytest.mark.parametrize("overlap", [0, 32, 64, 96])
@pytest.mark.parametrize("fit", ["reserve", "fill"])
@pytest.mark.parametrize("method", ["sentence", "word"])
def test_every_overlap_candidate_is_valid(fake_counter, overlap, fit, method):
    result, check, cfg = run(fake_counter, overlap_tokens=overlap, overlap_fit=fit, overlap_method=method)
    assert check.failures == []
    assert all(c.overlap_tokens <= overlap for c in result.chunks)


def test_overlap_stays_inside_its_section(fake_counter):
    result, _, _ = run(fake_counter, overlap_tokens=96, overlap_method="word")
    starts = {}
    for c in result.chunks:
        assert all(s.block_pointer.startswith(c.section_id + "/") for s in c.overlap_spans)
        starts.setdefault(c.section_id, c)
    assert all(c.overlap_tokens == 0 for c in starts.values())  # a section's first chunk has no overlap
    assert any(c.overlap_tokens for c in result.chunks)


def test_case_reports_stay_separate(fake_counter):
    result, _, _ = run(fake_counter)
    for sid, case in (("case1", "case-1"), ("case2", "case-2")):
        (chunk,) = by_section(result, sid)
        assert (chunk.case_id, chunk.evidence_type) == (case, "individual_case")
    assert not any("S1 " in c.text and "S2 " in c.text for c in result.chunks if c.case_id)


def test_boxed_warning_is_its_own_chunk_and_highlights_follow_the_setting(fake_counter):
    result, _, _ = run(fake_counter)
    (box,) = by_section(result, "box")
    assert box.text == "Serious risk of bleeding. Monitor closely." and box.notices[0] == "boxed_warning"
    assert by_section(result, "box:highlights") == []
    assert any(e.reason == "highlights:exclude" for e in result.exclusions)
    included, _, _ = run(fake_counter, highlights="include")
    assert [c.text for c in by_section(included, "box:highlights")] == ["Boxed summary."]
    meta, _, _ = run(fake_counter, highlights="metadata_only")
    assert meta.document.highlights == ["Boxed summary."] and by_section(meta, "box:highlights") == []


@pytest.mark.parametrize("mode", ["include", "exclude", "metadata_only"])
def test_abstracts_follow_their_setting(fake_counter, mode):
    doc = make_doc()
    doc.sections.insert(0, sec("abs", "Abstract", [para("abs/p1", "Short abstract.")], role=SectionRole.ABSTRACT))
    result, check, _ = run(fake_counter, doc=doc, abstracts=mode)
    assert check.failures == []
    assert [c.text for c in by_section(result, "abs")] == (["Short abstract."] if mode == "include" else [])
    assert result.document.abstracts == (["Short abstract."] if mode == "metadata_only" else [])


def test_short_otc_section_keeps_its_heading(fake_counter):
    result, _, _ = run(fake_counter)
    (otc,) = by_section(result, "otc")
    assert (otc.prefix, otc.text) == ("Testdrug — Purpose", "Antihistamine")
    (adults,) = by_section(result, "s1.1")
    assert adults.prefix == "Testdrug — 1 INDICATIONS > 1.1 Adults"


def test_large_table_keeps_header_units_footnotes_and_row_ranges(fake_counter):
    result, _, _ = run(fake_counter, max_tokens=120)
    tables = [c for c in result.chunks if c.kind == "table" and c.table.row_part is None]
    assert len(tables) > 1
    for c in tables:
        assert "| Parameter | Cmax |\n|  | (ng/mL) |" in c.text
        assert c.table.block_pointer == "pk/table1" and c.table.header_rows == 2 and c.table.row_count == 61
    with_note = [c for c in tables if c.table.rows[0] <= 7 <= c.table.rows[1]]
    assert "(footnote 1) Fasting state" in with_note[0].text


def test_oversized_row_is_split_into_labelled_parts(fake_counter):
    result, check, _ = run(fake_counter, max_tokens=120)
    parts = [c for c in result.chunks if c.table and c.table.row_part]
    assert parts and all(c.table.rows == (61, 61) for c in parts)
    assert "Cmax / (ng/mL): x" in parts[0].text
    assert check.failures == []


def test_token_fallback_keeps_text_and_is_reported(fake_counter):
    result, check, _ = run(fake_counter)
    flagged = [c for c in by_section(result, "pk") if "token_split" in c.notices]
    assert flagged
    assert check.failures == [] and check.text_coverage == 1.0


def test_figures_keep_source_text_only_and_exclusions_are_recorded(fake_counter):
    result, _, cfg = run(fake_counter)
    (figure,) = [c for c in result.chunks if c.kind == "figure"]
    assert figure.text == "Figure: Figure 1: Chemical structure\nDescription: Structural formula of the drug"
    assert [s.block_pointer for s in figure.content_spans] == ["pk/mm1@M1", "pk/mm1@M1/caption"]
    reasons = {e.block_pointer: e.reason for e in result.exclusions if e.block_pointer}
    assert reasons == {"pk/mm2@M2": "package_label_image", "pk/mm3@M3": "no_meaningful_text"}
    assert figure_decision("Mean plasma concentration over time", "", cfg).keep


def test_package_label_excluded_and_notices(fake_counter):
    result, _, _ = run(fake_counter)
    assert by_section(result, "pkg") == []
    assert any(e.section_id == "pkg" and e.reason == "excluded_role:package_label" for e in result.exclusions)
    assert all("corrected_article" in c.notices for c in result.chunks)


def test_source_references_are_traceable(fake_counter):
    doc = make_doc()
    result, _, _ = run(fake_counter, doc=doc)
    blocks = {b.source_pointer: b for _, s in doc.iter_sections() for b in s.blocks}
    for c in result.chunks:
        assert c.source_reference == "https://example.test/label"
        for span in c.content_spans:
            assert blocks[span.block_pointer].text[span.start : span.end] in c.text
    assert (result.document.raw_file, result.document.raw_member) == ("data/raw/test.zip", "test.xml")


def test_chunk_ids_are_deterministic_and_config_sensitive(fake_counter):
    first, _, _ = run(fake_counter)
    again, _, _ = run(fake_counter)
    other, _, _ = run(fake_counter, overlap_tokens=32)
    ids = [c.chunk_id for c in first.chunks]
    assert ids == [c.chunk_id for c in again.chunks]
    assert len(ids) == len(set(ids)) and all(len(i) == 32 for i in ids)
    assert not set(ids) & {c.chunk_id for c in other.chunks}


def test_long_word_flag_follows_berts_rule(fake_counter):
    chemical = "2,2-dimethyl-" + "-".join(["hexahydro"] * 14)  # >100 chars but broken by punctuation
    unbroken = "a" * 101
    doc = make_doc()
    doc.sections.append(sec("chem", "Chemistry", [para("chem/p1", f"Name: {chemical}."), para("chem/p2", "x")]))
    doc.sections.append(sec("seq", "Sequence", [para("seq/p1", f"Sequence {unbroken} end.")]))
    result, _, _ = run(fake_counter, doc=doc)
    assert all("long_word_unk" not in c.notices for c in by_section(result, "chem"))
    assert all("long_word_unk" in c.notices for c in by_section(result, "seq"))


def test_input_document_is_not_modified(fake_counter):
    doc = make_doc()
    before = copy.deepcopy(doc)
    run(fake_counter, doc=doc)
    assert doc == before


def test_validation_catches_broken_output(fake_counter):
    doc = make_doc()
    result, _, cfg = run(fake_counter, doc=doc)
    big = result.chunks[0].model_copy(update={"token_count": 9999})
    result.chunks[0] = big
    del result.chunks[-1]
    failures = validate(doc, result, cfg, fake_counter).failures
    assert any(f.startswith("V1") for f in failures)
    assert any(f.startswith(("V3", "V4", "V10")) for f in failures)


def test_section_with_only_a_title_produces_nothing(fake_counter):
    doc = replace(make_doc(), sections=[sec("empty", "Empty", [])])
    result, check, _ = run(fake_counter, doc=doc)
    assert result.chunks == [] and check.failures == []
