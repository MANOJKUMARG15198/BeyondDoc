import pytest

from medical_rag.chunking.sentences import sentence_ends


def pieces(text):
    bounds = [0, *sentence_ends(text), len(text)]
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:], strict=False) if text[a:b].strip()]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # basic punctuation
        ("Take one tablet daily. Do not crush.", ["Take one tablet daily.", "Do not crush."]),
        ("Stop now! Call a doctor? Yes.", ["Stop now!", "Call a doctor?", "Yes."]),
        ("Line one.\nLine two.", ["Line one.", "Line two."]),
        # abbreviations and initials (M6)
        ("Consult Dr. Smith before use.", ["Consult Dr. Smith before use."]),
        ("Give 1 g i.v. over 30 minutes.", ["Give 1 g i.v. over 30 minutes."]),
        ("Use NSAIDs, e.g. Ibuprofen, with care.", ["Use NSAIDs, e.g. Ibuprofen, with care."]),
        ("Use St. John's wort with caution.", ["Use St. John's wort with caution."]),
        ("Smith et al. Reported similar findings.", ["Smith et al. Reported similar findings."]),
        ("Written by J. Smith. Read it.", ["Written by J. Smith.", "Read it."]),
        # units are not abbreviations; decimals never split
        ("Take 5 mg. Then reassess.", ["Take 5 mg.", "Then reassess."]),
        ("Take 0.5 mg twice daily. Increase to 1.5 mg.", ["Take 0.5 mg twice daily.", "Increase to 1.5 mg."]),
        ("Avoid rash, fever, etc. Stop use if severe.", ["Avoid rash, fever, etc.", "Stop use if severe."]),
        # numbered clinical sections, cross-references and labels
        ("1. Introduction\n5.2. Renal Effects\na. First item", ["1. Introduction\n5.2. Renal Effects\na. First item"]),
        ("Risk increases [see Warnings (5.2)]. Monitor INR.", ["Risk increases [see Warnings (5.2)].", "Monitor INR."]),
        ("See Table 3. Mean values are shown.", ["See Table 3. Mean values are shown."]),
        ("Take 2. Wait.", ["Take 2.", "Wait."]),
        # sentences beginning with lowercase scientific terms or Greek letters
        (
            "Levels rose. mRNA expression fell. p-glycoprotein is inhibited. eGFR declined.",
            ["Levels rose.", "mRNA expression fell.", "p-glycoprotein is inhibited.", "eGFR declined."],
        ),
        ("Levels rose. the dose was cut.", ["Levels rose. the dose was cut."]),
        ("Use with care. β-blockers may mask symptoms.", ["Use with care.", "β-blockers may mask symptoms."]),
        # Unicode punctuation and brackets
        ("He said “stop.” Then he left… Afterwards, rest.", ["He said “stop.”", "Then he left…", "Afterwards, rest."]),
        ("Wait... Then resume.", ["Wait...", "Then resume."]),
        ("Avoid alcohol. (See section 17.)", ["Avoid alcohol.", "(See section 17.)"]),
        ("Avoid alcohol. (see section 17.)", ["Avoid alcohol. (see section 17.)"]),
        # drug names containing punctuation
        (
            "Take Tylenol® P.M. Extra at bedtime. Do not exceed 2 doses.",
            ["Take Tylenol® P.M. Extra at bedtime.", "Do not exceed 2 doses."],
        ),
        ("Made by Mylan Inc. Morgantown, WV.", ["Made by Mylan Inc. Morgantown, WV."]),
        ("Co-trimoxazole may interact. Ca++ levels rise.", ["Co-trimoxazole may interact.", "Ca++ levels rise."]),
    ],
)
def test_sentence_boundaries(text, expected):
    assert pieces(text) == expected


@pytest.mark.parametrize(
    "text",
    ["Take vitamin K. Monitor INR.", "Give 1 g i.v. Infuse slowly.", "Dose 10 mg b.i.d. Patients improved."],
)
def test_documented_under_splits_keep_text_together(text):
    assert pieces(text) == [text]  # accepted: a missed boundary only makes the piece coarser


def test_range_is_respected():
    text = "One. Two. Three. Four."
    assert sentence_ends(text, 5, 16) == [9]  # only the end of "Two." lies inside "Two. Three."


def test_pieces_rejoin_to_the_original():
    text = "Take 5 mg. Then reassess [see (5.2)]. mRNA rose… Stop! Dr. Smith agreed."
    bounds = [0, *sentence_ends(text), len(text)]
    assert "".join(text[a:b] for a, b in zip(bounds, bounds[1:], strict=False)) == text
