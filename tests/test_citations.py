"""Per-sentence citation coverage (no LLM)."""

from medical_rag.evaluation.citations import citation_coverage, split_sentences


def test_citation_after_full_stop_stays_with_its_sentence():
    assert split_sentences("Take 10 mg daily. [S1] Do not double. [S2][S3]") == [
        "Take 10 mg daily [S1].",
        "Do not double [S2][S3].",
    ]


def test_abbreviations_and_decimals_do_not_split():
    s = split_sentences("Use a strong inhibitor (e.g. ketoconazole) at 0.5 mg [S1]. Next one [S2].")
    assert s == ["Use a strong inhibitor (e.g. ketoconazole) at 0.5 mg [S1].", "Next one [S2]."]


def test_sentences_are_classified():
    answer = (
        "The dose is 300 mg three times daily [S2]. "
        "The mean decrease was approximately 0.3 g percent for hemoglobin. "
        "The passages do not say whether the gelatin is animal-derived. "
        "No. "
        "This is general label information, not personal medical advice; consult a healthcare professional."
    )
    c = citation_coverage(answer)
    assert len(c.cited) == 1 and len(c.uncited_factual) == 1
    assert len(c.absence) == 1 and len(c.short) == 1 and len(c.disclaimers) == 1
    assert c.coverage == 0.5


def test_coverage_is_none_without_claims():
    assert citation_coverage("Consult a healthcare professional.").coverage is None
    assert citation_coverage("Fact one [S1]. Fact two [S1].").as_dict()["coverage"] == 1.0
