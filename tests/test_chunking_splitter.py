import random

import pytest

from medical_rag.chunking.config import TokenizerSpec
from medical_rag.chunking.splitter import Span, overlap_start, pack, pieces, split
from medical_rag.chunking.tokenizer import ChunkingError, TokenCounter

PREFIX = "T — S\n"  # 3 fake tokens


def sentence(i, n):
    """A sentence of exactly n fake tokens: 'S<i>', n-2 words, full stop (not a single letter: that is an initial)."""
    return " ".join([f"S{i}", *["word"] * (n - 2)]) + "."


def covered(text, parts):
    marks = [False] * len(text)
    for p in parts:
        for k in range(p.start, p.end):
            marks[k] = True
    return all(marks[k] or text[k].isspace() for k in range(len(text)))


def test_340_token_section_stays_intact(fake_counter):
    text = " ".join(sentence(i, 85) for i in range(4))  # 340 tokens
    spans = pack(text, pieces(text, [(0, len(text))], fake_counter, 509), fake_counter, PREFIX, 512)
    assert [(s.start, s.end) for s in spans] == [(0, len(text))]
    assert spans[0].levels == ("block",)


def test_700_token_section_is_packed_not_fragmented(fake_counter):
    text = " ".join(sentence(i, 100) for i in range(1, 8))  # seven 100-token sentences
    parts = pieces(text, [(0, len(text))], fake_counter, 509)
    assert len(parts) == 7 and {p.level for p in parts} == {"sentence"}  # Stage A
    spans = pack(text, parts, fake_counter, PREFIX, 512)  # Stage B
    assert len(spans) == 2
    first = text[spans[0].start : spans[0].end]
    assert first.startswith("S1 ") and first.endswith(sentence(5, 100))
    assert fake_counter.count(PREFIX + text[spans[0].start : spans[0].end]) == 503  # s1-s5 + prefix
    assert fake_counter.count(text[spans[1].start : spans[1].end]) == 200  # s6-s7
    assert covered(text, spans)


def test_every_span_fits_the_complete_budget(fake_counter):
    rng = random.Random(7)
    text = " ".join(sentence(i, rng.randint(5, 140)) for i in range(60))
    spans = pack(text, pieces(text, [(0, len(text))], fake_counter, 509), fake_counter, PREFIX, 512)
    assert all(fake_counter.count(PREFIX + text[s.start : s.end]) <= 512 for s in spans)
    assert [s.start for s in spans] == sorted(s.start for s in spans)  # source order
    assert covered(text, spans)


def test_levels_are_tried_in_order(fake_counter):
    clause = "Alpha w w w; beta w w w, gamma w w w"  # one sentence, 3 clauses/phrases
    parts = split(clause, 0, len(clause), fake_counter, 6)
    assert {p.level for p in parts} == {"clause", "phrase"}
    assert covered(clause, parts)


def test_block_boundaries_come_first(fake_counter):
    text = "Alpha w w.\nBeta w w."
    blocks = [(0, 10), (11, len(text))]
    parts = pieces(text, blocks, fake_counter, 509)
    assert [(p.start, p.end, p.level) for p in parts] == [(0, 10, "block"), (11, len(text), "block")]


def test_word_level_then_reported_token_cut(fake_counter):
    text = "Alpha " + "-" * 30 + " omega"  # a 30-token unbroken 'word'
    parts = split(text, 0, len(text), fake_counter, 10)
    assert covered(text, parts)
    assert all(fake_counter.count(text[p.start : p.end]) <= 10 for p in parts)
    assert "word_start" in {p.level for p in parts}


def test_piece_too_large_for_prefix_raises(fake_counter):
    text = sentence(1, 20)
    with pytest.raises(ChunkingError):
        pack(text, pieces(text, [(0, len(text))], fake_counter, 509), fake_counter, PREFIX, 10)


@pytest.mark.parametrize("n", [0, 32, 64, 96])
def test_sentence_overlap_takes_whole_sentences_within_n(fake_counter, n):
    text = " ".join(sentence(i, 30) for i in range(10))  # 300 tokens
    previous = Span(0, len(text), ("sentence",))
    start = overlap_start(text, previous, fake_counter, n, "sentence")
    tail = text[start : previous.end]
    assert fake_counter.count(tail) <= n
    assert fake_counter.count(tail) == 30 * (n // 30)  # whole sentences only
    assert start == previous.end or text[start] == "S"


def test_word_overlap_falls_back_to_whole_words(fake_counter):
    text = sentence(1, 100)
    previous = Span(0, len(text), ("block",))
    assert overlap_start(text, previous, fake_counter, 32, "sentence") == previous.end  # no sentence fits
    start = overlap_start(text, previous, fake_counter, 32, "word")
    assert 0 < fake_counter.count(text[start:]) <= 32
    assert text[start - 1] == " "  # starts at a word boundary


SPEC = TokenizerSpec()


@pytest.mark.skipif(not SPEC.file.exists(), reason="local BERT tokenizer file not available")
def test_real_tokenizer_long_chemical_name_is_cut_at_punctuation():
    counter = TokenCounter(SPEC)
    name = "4-[4-(4-chlorophenyl)-4-hydroxypiperidin-1-yl]-1-(4-fluorophenyl)butan-1-one"
    text = f"Haloperidol is {name} and is given orally."
    parts = split(text, 0, len(text), counter, 8)
    assert covered(text, parts)
    assert all(counter.count(text[p.start : p.end]) <= 8 for p in parts)
    assert "punctuation" in {p.level for p in parts}
    assert "subword" not in {p.level for p in parts}
