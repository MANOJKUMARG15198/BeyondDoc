from collections import Counter

import pytest

from medical_rag.ingestion.text import normalize_text, text_sha256


@pytest.mark.parametrize("text", ["25 µg", "mg/m²", "½ tablet", "≥ 10 °C", "± 2.5"])
def test_clinical_symbols_are_unchanged(text):
    assert normalize_text(text) == text


def test_nfc_composes_accents():
    assert normalize_text("Sjögren") == "Sjögren"


def test_invisible_characters_are_removed_and_counted():
    changes = Counter()
    assert normalize_text("log⁡x hepato­toxicity ﻿dose", changes=changes) == "logx hepatotoxicity dose"
    assert changes == {"U+2061": 1, "U+00AD": 1, "U+FEFF": 1}


@pytest.mark.parametrize("separator", ["​", "\x00", "\x1f", "\x85"])
def test_zero_width_space_and_controls_never_merge_words(separator):
    changes = Counter()
    assert normalize_text(f"renal{separator}failure", changes=changes) == "renal failure"
    assert changes == {f"U+{ord(separator):04X}": 1}


def test_zwj_and_zwnj_are_kept():
    assert normalize_text("a‍b‌c") == "a‍b‌c"


def test_replacement_character_is_kept_and_counted():
    changes = Counter()
    assert normalize_text("dose � mg", changes=changes) == "dose � mg"
    assert changes == {"U+FFFD": 1}


def test_whitespace_collapses_without_being_counted():
    changes = Counter()
    assert normalize_text("  10 mg\t daily\n\n", changes=changes) == "10 mg daily"
    assert not changes


def test_keep_newlines_keeps_one_break_per_line():
    assert normalize_text(" Reduce dose \r\n\r\n  in renal failure \r", keep_newlines=True) == (
        "Reduce dose\nin renal failure"
    )


def test_changes_accumulate_across_calls():
    changes = Counter()
    normalize_text("a⁡b", changes=changes)
    normalize_text("c⁡d", changes=changes)
    assert changes == {"U+2061": 2}


def test_text_sha256_is_utf8_digest():
    assert text_sha256("µg") == "c2ab9680237665dc375320db7b2c3e7cb5cfccc6a79e30d7c692bcb404cc4010"
