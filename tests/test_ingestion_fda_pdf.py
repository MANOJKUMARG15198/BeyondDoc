"""FDA letter parser on synthetic page texts (no real letter text, no PDF library needed)."""

import pytest

from medical_rag.ingestion.fda_pdf import is_heading, parse_letter_pages
from medical_rag.ingestion.models import EvidenceType, SectionRole

PAGES = [
    "\n".join(
        [
            "DEPARTMENT OF HEALTH & HUMAN SERVICES Public Health Service",
            "Food and Drug",
            "Administration",
            "Rockville MD  20857",
            "NDA 99-999",
            "Example Pharma Inc.",
            "Attention: Pat Example, Director",
            "",
            "SUPPLEMENT APPROVAL",
            "Dear Pat Example:",
            "We have completed our review of this supplemental application. It is approved for use as",
            "recommended in the enclosed labeling, including the 1000-",
            "count bottles.",
            "",
            "APPROVAL & LABELING",
            "• Content of labeling must be identical to the",
            "enclosed labeling.",
            "1 http://www.example.test/labeling",
            "Reference ID: 1234567",
        ]
    ),
    "\n".join(
        [
            "NDA 99-999",
            "Page 2",
            "REPORTING REQUIREMENTS",
            "You must comply with all reporting requirements for an approved application.",
            "Sincerely,",
            "{See appended electronic signature page}",
            "Pat Reviewer, M.D.",
            "Director, Division of Examples",
            "ENCLOSURE(S):",
            "• Content of Labeling",
            "Reference ID: 1234567",
        ]
    ),
    "\n".join(
        [
            "--------------------------------------------------------------",
            "This is a representation of an electronic record that was signed",
            "electronically.",
            "--------------------------------------------------------------",
            "/s/",
            "PAT REVIEWER",
            "01/02/2024 10:00:00 AM",
        ]
    ),
]


@pytest.fixture(scope="module")
def doc():
    return parse_letter_pages(
        PAGES,
        source_document_id="NDA099999:12345",
        source_reference="https://example.test/letter.pdf",
        raw_file="letter.pdf",
        raw_sha256="0" * 64,
    )


def by_title(doc, title):
    return next(s for s in doc.sections if s.title == title)


def texts(section):
    return [b.text for b in section.blocks]


def test_identity(doc):
    assert (doc.document_id, doc.document_version, doc.source_format) == (
        "fda_drugsatfda:NDA099999:12345",
        "12345",
        "pdf",
    )
    assert doc.title == "FDA letter NDA099999 — SUPPLEMENT APPROVAL"
    assert doc.metadata["application"] == "NDA099999"
    assert all(s.evidence_type is EvidenceType.REGULATORY_ACTION for s in doc.sections)


def test_running_headers_letterhead_and_rules_become_artifacts(doc):
    kinds = {(a.kind, a.source_pointer) for a in doc.artifacts}
    assert ("pdf_letterhead", "page:1/line[1]") in kinds
    assert ("pdf_running_header_footer", "page:2/line[2]") in kinds  # "Page 2"
    assert ("pdf_running_header_footer", "page:1/line[19]") in kinds  # Reference ID footer
    assert ("pdf_rule_line", "page:3/line[1]") in kinds
    all_text = " ".join(b.text for s in doc.sections for b in s.blocks)
    assert "Reference ID" not in all_text and "Page 2" not in all_text and "Rockville" not in all_text


def test_paragraphs_keep_compound_hyphens_and_pages(doc):
    approval = by_title(doc, "SUPPLEMENT APPROVAL")
    assert texts(approval) == [
        "Dear Pat Example:",
        "We have completed our review of this supplemental application. It is approved for use as recommended in "
        "the enclosed labeling, including the 1000-count bottles.",
    ]
    assert [b.page for b in approval.blocks] == [1, 1]


def test_headings_bullets_and_footnotes(doc):
    assert [s.title for s in doc.sections] == [
        "Letter",
        "SUPPLEMENT APPROVAL",
        "APPROVAL & LABELING",
        "REPORTING REQUIREMENTS",
        "Signature",
        "ENCLOSURE(S):",
        "Electronic signature",
    ]
    assert texts(by_title(doc, "APPROVAL & LABELING")) == [
        "• Content of labeling must be identical to the enclosed labeling.",
        "1 http://www.example.test/labeling",
    ]


def test_signature_names_are_kept(doc):
    signature = by_title(doc, "Signature")
    assert signature.role is SectionRole.LETTER_SIGNATURE
    assert texts(signature) == [
        "Sincerely,",
        "{See appended electronic signature page}",
        "Pat Reviewer, M.D.",
        "Director, Division of Examples",
    ]
    esig = by_title(doc, "Electronic signature")
    assert esig.role is SectionRole.LETTER_SIGNATURE and "PAT REVIEWER" in texts(esig)
    assert "Attention: Pat Example, Director" in " ".join(texts(by_title(doc, "Letter")))  # contact name kept


def test_pointers_unique_and_traceable(doc):
    pointers = [b.source_pointer for s in doc.sections for b in s.blocks]
    assert len(pointers) == len(set(pointers)) and all(p.startswith("page:") for p in pointers)


@pytest.mark.parametrize(
    ("text", "heading"),
    [
        ("APPROVAL & LABELING", True),
        ("WAIVER OF ½ PAGE LENGTH REQUIREMENT FOR HIGHLIGHTS", True),
        ("NDA 020702/S-081", False),
        ("ENCLOSURE(S):", False),
        ("Dear Pat:", False),
        ("/s/", False),
    ],
)
def test_is_heading(text, heading):
    assert is_heading(text) is heading
